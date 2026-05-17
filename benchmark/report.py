"""Render a `BenchmarkResult` to stdout (human summary) and to a JSON file."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.runner import AgentResult, BenchmarkResult


def _money(v: float | None) -> str:
    return f"${v:.4f}" if v is not None else "n/a"


def _agent_summary(r: AgentResult, label: str, index_cost_usd: float | None = None) -> str:
    lines = [f"── {label} ─────────────────────────────"]
    if r.is_error:
        lines.append(f"  ERROR: {r.error}")
    in_tok = r.usage.get("input_tokens")
    out_tok = r.usage.get("output_tokens")
    cache_r = r.usage.get("cache_read_input_tokens")
    cache_c = r.usage.get("cache_creation_input_tokens")
    lines.append(
        f"  tokens: in={in_tok} out={out_tok} cache_read={cache_r} cache_create={cache_c}"
    )
    lines.append(f"  turns: {r.num_turns}    duration_ms: {r.duration_ms}    wall_ms: {r.wall_ms}")
    if index_cost_usd is not None and r.total_cost_usd is not None:
        total = r.total_cost_usd + index_cost_usd
        lines.append(
            f"  cost: agent={_money(r.total_cost_usd)} + index={_money(index_cost_usd)} "
            f"= {_money(total)}"
        )
    else:
        lines.append(f"  cost: agent={_money(r.total_cost_usd)}")
    if r.model_usage:
        lines.append("  per-model:")
        for model_name, stats in r.model_usage.items():
            cost = stats.get("costUSD") if isinstance(stats, dict) else None
            in_t = stats.get("inputTokens") if isinstance(stats, dict) else None
            out_t = stats.get("outputTokens") if isinstance(stats, dict) else None
            lines.append(f"    {model_name}: cost={_money(cost)} in={in_t} out={out_t}")
    return "\n".join(lines)


def print_summary(result: BenchmarkResult) -> None:
    idx = result.index
    print()
    print("═" * 64)
    print(f"  question: {result.question}")
    print(f"  repo:     {result.repo_name}  ({result.repo_path})")
    print(f"  db:       {idx.db_name}  branch={result.branch}")
    print(f"  model:    {result.model}    wall_ms: {result.wall_ms}")
    print("═" * 64)
    print()
    print("── index ─────────────────────────────")
    print(
        f"  files indexed={idx.indexed_files} skipped={idx.skipped_files} "
        f"deleted={idx.deleted_files}"
    )
    print(
        f"  chunks seen={idx.chunks_seen} embedded={idx.chunks_embedded} "
        f"skipped={idx.chunks_skipped}"
    )
    print(
        f"  embed model={idx.embed_model} tokens={idx.embed_tokens} "
        f"api_calls={idx.embed_api_calls} cost={_money(idx.cost_usd)}"
    )
    print()
    print(_agent_summary(result.trident_agent, "trident agent", idx.cost_usd))
    print()
    print(_agent_summary(result.baseline_agent, "baseline agent"))
    print()


def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(result: BenchmarkResult, output_dir: Path, stamp: str | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or _timestamp()
    out_path = output_dir / f"{stamp}.json"
    payload = _to_jsonable(result)
    if (
        result.index.cost_usd is not None
        and result.trident_agent.total_cost_usd is not None
    ):
        payload["trident_agent"]["total_with_index_usd"] = (
            result.trident_agent.total_cost_usd + result.index.cost_usd
        )
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def _answer_block(heading: str, r: AgentResult) -> str:
    body = r.answer.strip() or "_(empty)_"
    if r.is_error:
        body = f"**ERROR**: {r.error}\n\n{body}"
    meta = (
        f"cost {_money(r.total_cost_usd)} · "
        f"{r.num_turns} turns · "
        f"{r.wall_ms} ms wall"
    )
    # H2 header makes the section jump to the eye and shows up in any md
    # outline view; <details open> keeps the answer visible by default but
    # still lets the reader collapse one side to focus on the other.
    return (
        f"---\n\n"
        f"## {heading}\n\n"
        f"> {meta}\n\n"
        f"<details open>\n"
        f"<summary>show / hide answer</summary>\n\n"
        f"{body}\n\n"
        f"</details>"
    )


def write_markdown(result: BenchmarkResult, output_dir: Path, stamp: str | None = None) -> Path:
    """Write a single markdown file with both answers in collapsible <details>
    blocks so the user can read each one without it dominating the terminal."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or _timestamp()
    out_path = output_dir / f"{stamp}.md"
    idx = result.index

    trident_total = None
    if idx.cost_usd is not None and result.trident_agent.total_cost_usd is not None:
        trident_total = result.trident_agent.total_cost_usd + idx.cost_usd

    lines: list[str] = []
    lines.append(f"# Benchmark — {stamp}")
    lines.append("")
    lines.append(f"- **Question**: {result.question}")
    lines.append(f"- **Repo**: `{result.repo_name}` at `{result.repo_path}` (branch `{result.branch}`)")
    lines.append(f"- **DB**: `{idx.db_name}`")
    lines.append(f"- **Model**: `{result.model}`")
    lines.append(f"- **Wall time**: {result.wall_ms} ms")
    lines.append("")
    lines.append("## Cost summary")
    lines.append("")
    lines.append("| Side | Agent cost | Index cost | Total |")
    lines.append("|------|-----------:|-----------:|------:|")
    lines.append(
        f"| trident  | {_money(result.trident_agent.total_cost_usd)} "
        f"| {_money(idx.cost_usd)} | {_money(trident_total)} |"
    )
    lines.append(
        f"| baseline | {_money(result.baseline_agent.total_cost_usd)} "
        f"| — | {_money(result.baseline_agent.total_cost_usd)} |"
    )
    lines.append("")
    lines.append("## Indexing")
    lines.append("")
    lines.append(
        f"- files: indexed={idx.indexed_files}, skipped={idx.skipped_files}, "
        f"deleted={idx.deleted_files}"
    )
    lines.append(
        f"- chunks: seen={idx.chunks_seen}, embedded={idx.chunks_embedded}, "
        f"skipped={idx.chunks_skipped}"
    )
    lines.append(
        f"- embed: model=`{idx.embed_model}`, tokens={idx.embed_tokens}, "
        f"api_calls={idx.embed_api_calls}"
    )
    lines.append("")
    lines.append("# Answers")
    lines.append("")
    lines.append(_answer_block("Trident agent answer", result.trident_agent))
    lines.append("")
    lines.append(_answer_block("Baseline agent answer", result.baseline_agent))
    lines.append("")

    out_path.write_text("\n".join(lines))
    return out_path
