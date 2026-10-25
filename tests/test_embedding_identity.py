"""Vector reuse and queries must stay within one declared vector space."""

from dataclasses import replace
import hashlib
from pathlib import Path
from uuid import uuid4

import pytest

from core.embedding_identity import EmbeddingIdentity, identity_for
from core.embedder import embed_branch_chunks, make_fake_embedder
from core.retrieval import semantic_query
from db.connection import apply_migrations, MIGRATIONS_DIR


async def test_embedding_identity_is_required_for_anonymous_query():
    async def unknown(texts):
        raise AssertionError("Should reject before invoking embedder")

    with pytest.raises(ValueError, match="embedding_identity"):
        await semantic_query(None, 1, "q", unknown)


async def test_models_dimensions_and_representation_are_isolated(clean_repo):
    pool, repo, branch = clean_repo
    text = f"unique embedding input {uuid4()}"
    digest = hashlib.sha256(text.encode()).hexdigest()
    async with pool.acquire() as conn:
        fv = await conn.fetchval(
            """
            INSERT INTO file_versions(repo_id,content_hash,language,raw_content)
            VALUES ($1,$2,'python',$3) RETURNING id
        """,
            repo,
            digest,
            text,
        )
        await conn.execute(
            "INSERT INTO branch_files(branch_id,path,file_version_id) VALUES ($1,'file.py',$2)",
            branch,
            fv,
        )
        await conn.execute(
            """
            INSERT INTO chunks(branch_id,file_version_id,granularity,content,content_hash)
            VALUES ($1,$2,'function',$3,$4)
        """,
            branch,
            fv,
            text,
            digest,
        )
    identity = EmbeddingIdentity("model", "v1", 3)
    identities = [
        identity,
        replace(identity, model_name="other"),
        replace(identity, model_version="v2"),
        replace(identity, dimensions=4),
        replace(identity, representation_version="chunk-v2"),
    ]
    for space in identities:
        fake, _ = make_fake_embedder(space.dimensions)
        calls = []

        async def embed(texts):
            calls.extend(texts)
            return await fake(texts)

        assert (
            await embed_branch_chunks(
                pool, branch, embed, space.model_name, embedding_identity=space
            )
        ).embedded == 1
        assert (
            await embed_branch_chunks(
                pool, branch, embed, space.model_name, embedding_identity=space
            )
        ).embedded == 0
        assert calls == [text]
        for granularities in (None, ("function",)):
            results = await semantic_query(
                pool,
                branch,
                text,
                embed,
                embedding_identity=space,
                granularities=granularities,
                languages=["python"],
            )
            assert len(results) == 1 and results[0].content == text
    async with pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM chunk_embeddings WHERE content_hash=$1", digest
            )
            == 5
        )


async def test_mismatched_declared_identity_is_rejected():
    embed, _ = make_fake_embedder(3)
    with pytest.raises(ValueError, match="differs"):
        identity_for(embed, EmbeddingIdentity("other", "v1", 3))
    with pytest.raises(ValueError, match="dimensions"):
        EmbeddingIdentity("m", "v1", 0)


async def test_upgrade_retains_legacy_vectors_without_assigning_current_identity(
    pg_pool, tmp_path: Path
):
    schema = f"upgrade_{uuid4().hex}"
    async with pg_pool.acquire() as conn:
        tx = conn.transaction()
        await tx.start()
        try:
            await conn.execute(f'CREATE SCHEMA "{schema}"')
            await conn.execute(f'SET LOCAL search_path TO "{schema}", public')
            for source in sorted(MIGRATIONS_DIR.glob("*.up.sql")):
                if source.name < "0007":
                    (tmp_path / source.name).write_text(source.read_text())
            await apply_migrations(conn, tmp_path)
            vector = "[" + ",".join(["1"] + ["0"] * 4095) + "]"
            await conn.execute(
                """
                INSERT INTO chunk_embeddings(content_hash, embedding, model_name)
                VALUES ('legacy', $1::vector, 'model')
            """,
                vector,
            )
            source = MIGRATIONS_DIR / "0007_embedding_identity.up.sql"
            (tmp_path / source.name).write_text(source.read_text())
            assert await apply_migrations(conn, tmp_path) == [source.name]
            row = await conn.fetchrow("SELECT * FROM chunk_embeddings WHERE content_hash='legacy'")
            assert row["dimensions"] == 4096
            assert row["model_version"] == row["representation_version"] == "legacy-unversioned"
            await conn.execute("""
                INSERT INTO chunk_embeddings
                    (content_hash,embedding,model_name,model_version,dimensions,representation_version)
                VALUES ('legacy', '[1,0,0]'::vector, 'model', 'v2', 3, 'chunk-v1')
            """)
            assert await conn.fetchval("SELECT count(*) FROM chunk_embeddings") == 2
        finally:
            await tx.rollback()
