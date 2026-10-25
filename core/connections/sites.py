"""Snapshot-scoped call sites and searchable targets using existing language rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import PurePosixPath
import re
from uuid import UUID, uuid5

import asyncpg

from core.config_loader import load_language_config
from core.grammar_meta import LANGUAGES, language_for_path, parser_for_path
from .snapshots import fingerprint, load_snapshot

SITE_REPRESENTATION_VERSION = "site-v1"
EXTRACTOR_VERSION = "sites-v1"
_TERMS = re.compile(r"[\w$]+(?:[./:\-][\w$]+)*", re.UNICODE)
_STOP = frozenset(
    "def class function func fn return let const var import from public private "
    "async await void string int bool true false null none self this new if else "
    "for while try catch throw package export default use pub mut the and or".split()
)


def exact_terms(text: str) -> tuple[str, ...]:
    terms = {}
    for term in _TERMS.findall(text):
        normalized = term.lower()
        if len(term) > 2 and normalized not in _STOP:
            terms[normalized] = None
    return tuple(terms)


def lexical_text(text: str) -> str:
    split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).replace("_", " ")
    return text + "\n" + split


@dataclass(frozen=True)
class SiteOptions:
    context_chars: int = 1200
    passage_chars: int = 1800
    passage_overlap: int = 200

    def __post_init__(self):
        if self.context_chars < 100 or self.passage_chars < 100:
            raise ValueError("Site context and passage budgets must be at least 100 characters")
        if not 0 <= self.passage_overlap < self.passage_chars:
            raise ValueError("Passage overlap must be nonnegative and smaller than its budget")


@dataclass(frozen=True)
class ExtractedSite:
    kind: str
    start_byte: int
    end_byte: int
    retrieval_text: str
    terms: tuple[str, ...]
    metadata: dict


@dataclass(frozen=True)
class SiteIndex:
    snapshot_id: str
    index_version: str
    stats: dict


def index_version(options: SiteOptions) -> str:
    configs = {lang: load_language_config(lang).raw for lang in LANGUAGES}
    grammars = {
        lang: version(spec.module_name.replace("_", "-")) for lang, spec in LANGUAGES.items()
    }
    return fingerprint(
        {
            "extractor": EXTRACTOR_VERSION,
            "representation": SITE_REPRESENTATION_VERSION,
            "options": asdict(options),
            "configs": configs,
            "grammars": grammars,
            "tree_sitter": version("tree-sitter"),
        }
    )


def extract_file(
    path: str, raw: bytes, options: SiteOptions = SiteOptions()
) -> list[ExtractedSite]:
    """Extract all configured call shapes, including locally resolved calls.

    Unsupported text is still indexed as passages. No protocol classification or
    semantic-resolution confidence filter decides which calls enter the inventory.
    """
    if b"\0" in raw:
        return []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return []
    sites = []
    language = language_for_path(path)
    parser = parser_for_path(path)
    module = str(PurePosixPath(path).with_suffix("")).replace("/", " ")
    if parser is not None:
        cfg = load_language_config(language)
        calls = {rule.node_type: rule for rule in cfg.calls}
        definitions = {rule.node_type: rule for rule in cfg.definitions}
        tree = parser.parse(raw)
        stack = [(tree.root_node, (), "module", "")]
        while stack:
            node, scope, scope_kind, enclosing_header = stack.pop()
            rule = definitions.get(node.type)
            is_definition = rule is not None and (
                not rule.require_enclosing_scope_kind
                or scope_kind in rule.require_enclosing_scope_kind
            )
            name_node = (
                node.child_by_field_name(rule.name_field) if rule and rule.name_field else None
            )
            name = name_node.text.decode("utf-8") if name_node else node.type
            body = node.child_by_field_name("body")
            header_end = (
                body.start_byte
                if body
                else min(node.end_byte, node.start_byte + options.context_chars)
            )
            header = raw[node.start_byte : header_end].decode("utf-8", errors="replace")[
                : options.context_chars
            ]
            is_call = node.type in calls
            if is_call or is_definition:
                expression = raw[node.start_byte : node.end_byte].decode("utf-8")
                # The exact expression leads, followed by bounded enclosing context.
                # Source coordinates always cover the full node, never the clipped input.
                if is_call:
                    call_rule = calls[node.type]
                    function = node.child_by_field_name(call_rule.function_field)
                    callee = function.text.decode("utf-8") if function else ""
                    preceding = raw[
                        max(0, node.start_byte - options.context_chars) : node.start_byte
                    ]
                    nearby = "\n".join(
                        preceding.decode("utf-8", errors="replace").splitlines()[-3:]
                    )
                    parts = [
                        expression[: options.context_chars],
                        enclosing_header,
                        nearby,
                        " ".join(scope),
                        module,
                    ]
                    terms = exact_terms(callee + " " + expression[: options.context_chars])
                else:
                    parts = [header, expression[: options.context_chars], " ".join(scope), module]
                    terms = exact_terms(name + " " + header)
                # Preserve the first occurrence of each nonempty context fragment.
                unique_parts = {}
                for part in parts:
                    stripped = part.strip()
                    if stripped:
                        unique_parts[stripped] = None
                retrieval = "\n".join(unique_parts)
                sites.append(
                    ExtractedSite(
                        "call" if is_call else "definition",
                        node.start_byte,
                        node.end_byte,
                        retrieval,
                        terms,
                        {
                            "language": language,
                            "node_type": node.type,
                            "scope": list(scope),
                            "name": callee if is_call else name,
                            "start_line": node.start_point[0] + 1,
                            "end_line": node.end_point[0] + 1,
                            "parse_errors": node.has_error,
                            "input_clipped": len(expression) > options.context_chars,
                        },
                    )
                )
            if is_definition and rule.scope_boundary:
                scope = (*scope, name)
                scope_kind, enclosing_header = rule.kind, header
            for child in reversed(node.named_children):
                stack.append((child, scope, scope_kind, enclosing_header))
    # Generic overlapping passages retain configuration, build declarations,
    # unsupported languages, and context outside AST definitions.
    start, byte_start = 0, 0
    step = options.passage_chars - options.passage_overlap
    while start < len(text):
        passage = text[start : start + options.passage_chars]
        if passage.strip():
            sites.append(
                ExtractedSite(
                    "declaration" if language else "configuration",
                    byte_start,
                    byte_start + len(passage.encode()),
                    passage.strip() + "\n" + module,
                    exact_terms(passage),
                    {"language": language, "passage": True},
                )
            )
        if start + options.passage_chars >= len(text):
            break
        byte_start += len(text[start : start + step].encode())
        start += step
    return sites


async def index_snapshot_sites(
    pool: asyncpg.Pool,
    snapshot_id: str | UUID,
    options: SiteOptions = SiteOptions(),
) -> SiteIndex:
    sid = UUID(str(snapshot_id))
    async with pool.acquire() as conn, conn.transaction():
        await load_snapshot(conn, sid)  # Reject single-repo or unsealed input before indexing.
        iv = index_version(options)
        # Serialize extraction for this snapshot, including callers with other options.
        await conn.execute("SELECT id FROM connection_snapshots WHERE id=$1 FOR UPDATE", sid)
        existing = await conn.fetchrow(
            """
            SELECT stats, completed_at FROM connection_site_indexes
            WHERE snapshot_id=$1 AND index_version=$2
        """,
            sid,
            iv,
        )
        if existing and existing["completed_at"]:
            return SiteIndex(str(sid), iv, json.loads(existing["stats"]))
        if existing:
            raise ValueError("An incomplete site index already exists")
        await conn.execute(
            """
            INSERT INTO connection_site_indexes(snapshot_id,index_version,representation_version)
            VALUES ($1,$2,$3)
        """,
            sid,
            iv,
            SITE_REPRESENTATION_VERSION,
        )
        files = await conn.fetch(
            """
            SELECT f.*, b.content FROM connection_snapshot_files f
            JOIN connection_source_blobs b USING (content_hash)
            WHERE f.snapshot_id=$1 ORDER BY f.repo_id,f.path
        """,
            sid,
        )
        stats = {"files": 0, "calls": 0, "targets": 0, "skipped": []}
        for file in files:
            if file["mode"] not in {"100644", "100755"}:
                stats["skipped"].append(
                    {
                        "repo_id": file["repo_id"],
                        "path": file["path"],
                        "reason": "symlink_or_gitlink",
                    }
                )
                continue
            raw = bytes(file["content"])
            extracted = extract_file(file["path"], raw, options)
            if not extracted:
                stats["skipped"].append(
                    {
                        "repo_id": file["repo_id"],
                        "path": file["path"],
                        "reason": "binary_non_utf8_or_empty",
                    }
                )
                continue
            stats["files"] += 1
            records = []
            for site in extracted:
                key = fingerprint(
                    [iv, file["repo_id"], file["path"], site.kind, site.start_byte, site.end_byte]
                )
                records.append(
                    (
                        uuid5(sid, key),
                        sid,
                        file["repo_id"],
                        file["path"],
                        site.start_byte,
                        site.end_byte,
                        file["content_hash"],
                        site.kind,
                        site.retrieval_text,
                        SITE_REPRESENTATION_VERSION,
                        iv,
                        hashlib.sha256(site.retrieval_text.encode()).hexdigest(),
                        list(site.terms),
                        lexical_text(site.retrieval_text),
                        json.dumps(site.metadata),
                    )
                )
                stats["calls"] += site.kind == "call"
                stats["targets"] += 1
            await conn.executemany(
                """
                INSERT INTO interaction_sites
                    (id,snapshot_id,repo_id,path,start_byte,end_byte,content_hash,kind,
                     retrieval_text,representation_version,index_version,retrieval_hash,
                     terms,fts_doc,metadata)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,
                        to_tsvector('simple',$14),$15::jsonb)
            """,
                records,
            )
        await conn.execute(
            """
            UPDATE connection_site_indexes SET stats=$3::jsonb, completed_at=now()
            WHERE snapshot_id=$1 AND index_version=$2
        """,
            sid,
            iv,
            json.dumps(stats),
        )
    return SiteIndex(str(sid), iv, stats)
