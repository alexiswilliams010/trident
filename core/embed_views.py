"""Per-chunk embedding views.

A "view" is a piece of text we feed to the embedder. Every chunk gets one
embedding row per view kind, all aggregated at retrieval time via a lateral
that picks the best-scoring view per chunk. The chunk's `content` column —
what's *displayed* back to the caller — is unchanged: views only affect
what gets *embedded*.

Today's views:

  source    The chunk's `content` verbatim. Preserves the exact behavior we
            had before multi-view (so a re-embed produces equivalent vectors
            for the source view).

  enriched  Source plus a `# referenced types` block that inlines the source
            of every definition the chunk references (resolved via the
            existing `references.target_def_id`). Closes the gap where a
            function chunk like `function handle(req: UserDTO)` embeds
            without `UserDTO`'s field surface — that surface comes from a
            different file, so cosine never sees it on the source view.

The `enriched` view is generated deterministically at embed time. No LLM
calls; it's just SQL + greedy-packed source slices.
"""

from __future__ import annotations

import asyncpg

from .chunk_assembler import _greedy_pack_pieces, count_tokens

VIEW_KINDS: tuple[str, ...] = ("source", "enriched")

# Budget for the `# referenced types` block appended to the source. Mirrors
# the module-chunk budget — generous enough to inline a handful of class
# bodies, small enough that the enriched view stays embeddable.
ENRICHED_BUDGET_TOKENS = 1024

# Per-kind caps on how much of the referenced def's source we inline.
# Containers (classes, structs, interfaces) carry their field surface, so we
# include up to ~200 tokens of body. Functions only need their signature.
# Leaf kinds (variables, enums, constants) get a short head.
_CONTAINER_KINDS: frozenset[str] = frozenset({
    "class", "interface", "struct", "record", "contract", "library", "type",
    "type_alias", "enum",
})
_FUNCTION_KINDS: frozenset[str] = frozenset({
    "function", "method", "constructor", "modifier",
})

_CONTAINER_CAP_TOKENS = 200
_LEAF_CAP_LINES = 3


async def build_views(
    conn: asyncpg.Connection,
    chunk_id: int,
    chunk_content: str,
    anchor_def_id: int | None,
    granularity: str,
) -> dict[str, str]:
    """Return {view_kind: text} for every view we want to embed for this chunk.

    `granularity` is currently unused — the algorithm derives everything from
    the anchor's CST byte range — but we keep it on the signature so future
    granularity-specific views are a one-line change.
    """
    del granularity  # reserved for future use
    views: dict[str, str] = {"source": chunk_content}
    if anchor_def_id is None:
        # Synthetic chunks (none today) without an anchor get the source view
        # only; there's nothing to walk references against.
        views["enriched"] = chunk_content
        return views
    enriched = await _enriched_view(
        conn, anchor_def_id, chunk_content, ENRICHED_BUDGET_TOKENS,
    )
    views["enriched"] = enriched
    return views


async def _enriched_view(
    conn: asyncpg.Connection,
    anchor_def_id: int,
    base_content: str,
    budget_tokens: int,
) -> str:
    """Append a `# referenced types` block listing the source of every
    definition this chunk's anchor references.

    Algorithm:
      1. Look up the anchor's CST byte range.
      2. Pull every reference whose CST node lies *inside* that range and
         resolves to a definition outside the anchor.
      3. For each target, fetch (qualified_name, kind, source_slice).
      4. Format per kind (container body / function signature / leaf head).
      5. Greedy-pack under `budget_tokens`. If nothing fits, return the
         base content unchanged.
    """
    anchor_row = await conn.fetchrow(
        """
        SELECT n.file_id, n.start_byte, n.end_byte
        FROM definitions d JOIN nodes n ON n.id = d.node_id
        WHERE d.id = $1
        """,
        anchor_def_id,
    )
    if anchor_row is None:
        return base_content

    target_rows = await conn.fetch(
        """
        SELECT DISTINCT r.target_def_id
        FROM "references" r JOIN nodes n ON n.id = r.node_id
        WHERE n.file_id = $1
          AND n.start_byte >= $2 AND n.end_byte <= $3
          AND r.target_def_id IS NOT NULL
          AND r.target_def_id != $4
        """,
        anchor_row["file_id"], anchor_row["start_byte"], anchor_row["end_byte"],
        anchor_def_id,
    )
    target_ids = [r["target_def_id"] for r in target_rows]
    if not target_ids:
        return base_content

    def_rows = await conn.fetch(
        """
        SELECT d.id, d.qualified_name, d.kind,
               substring(f.raw_content from n.start_byte + 1
                         for n.end_byte - n.start_byte) AS src
        FROM definitions d
        JOIN nodes n ON n.id = d.node_id
        JOIN files f ON f.id = n.file_id
        WHERE d.id = ANY($1::bigint[])
        ORDER BY d.qualified_name
        """,
        target_ids,
    )

    pieces: list[str] = []
    for r in def_rows:
        qn = r["qualified_name"] or ""
        kind = r["kind"] or ""
        src = (r["src"] or "").strip()
        if not src:
            continue
        formatted = _format_target(qn, kind, src)
        if formatted:
            pieces.append(formatted)
    if not pieces:
        return base_content

    # `_greedy_pack_pieces` takes a list of fallback lists; one fallback per
    # piece is fine.
    base_tokens = count_tokens(base_content)
    header = "\n\n# referenced types\n"
    header_tokens = count_tokens(header)
    accepted = _greedy_pack_pieces(
        base_tokens=base_tokens + header_tokens,
        candidates=[[p] for p in pieces],
        budget=base_tokens + header_tokens + budget_tokens,
    )
    if not accepted:
        return base_content
    return base_content + header + "\n\n".join(accepted)


def _format_target(qualified_name: str, kind: str, src: str) -> str:
    """Render one referenced def into a compact block.

    Container kinds (class/interface/struct/etc.): up to ~200 tokens of
    source — this is the field-surface case (`UserDTO` → its fields).
    Function kinds: signature only (first line).
    Leaf kinds (variable/constant/enum field): first ~3 lines.
    """
    label_kind = kind or "def"
    label = f"## {label_kind} {qualified_name}".rstrip()
    if kind in _FUNCTION_KINDS:
        first = src.splitlines()[0] if src else ""
        return f"{label}\n{first}" if first else label
    if kind in _CONTAINER_KINDS:
        body = _truncate_to_token_cap(src, _CONTAINER_CAP_TOKENS)
        return f"{label}\n{body}" if body else label
    # Leaf / unknown kind: small head.
    head_lines = src.splitlines()[:_LEAF_CAP_LINES]
    head = "\n".join(head_lines).strip()
    return f"{label}\n{head}" if head else label


def _truncate_to_token_cap(text: str, cap_tokens: int) -> str:
    """Best-effort token cap by repeatedly trimming lines. We don't need
    exact truncation — the caller's greedy-pack budget catches anything that
    slips through."""
    if count_tokens(text) <= cap_tokens:
        return text
    lines = text.splitlines()
    while lines and count_tokens("\n".join(lines)) > cap_tokens:
        lines.pop()
    return "\n".join(lines)
