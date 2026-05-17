"""Run a Claude Code agent twice over the same question: once with the
trident skill enabled, once with stock Claude Code tools only.

Both runs use `permission_mode="bypassPermissions"` so the agent can run
`make` (trident side) or `Read`/`Grep`/`Bash` (baseline side) without
stopping for confirmation. The trident-enabled run uses the trident repo as
cwd so `.claude/skills/trident/SKILL.md` is discoverable via
`setting_sources=["user","project"]`; the baseline run uses the target repo
as cwd with `setting_sources=["user"]` so no project skills load.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Import lazily inside the function so `import benchmark` doesn't hard-require
# the SDK for users who only need indexing helpers.


TRIDENT_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AgentResult:
    label: str
    answer: str
    total_cost_usd: float | None
    usage: dict[str, Any] = field(default_factory=dict)
    model_usage: dict[str, Any] = field(default_factory=dict)
    num_turns: int | None = None
    duration_ms: int | None = None
    wall_ms: int = 0
    is_error: bool = False
    error: str | None = None


def _preview(value: Any, n: int = 160) -> str:
    s = repr(value) if not isinstance(value, str) else value
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


async def _run_query(label: str, prompt: str, options, *, verbose: bool = False) -> AgentResult:
    """Drive the SDK's async iterator once and collect a single AgentResult.

    When `verbose=True`, every tool invocation and tool result is logged with
    a `[label]` prefix so concurrent agent streams stay distinguishable.
    """
    from claude_agent_sdk import (
        AssistantMessage, ResultMessage, SystemMessage, TextBlock, ThinkingBlock,
        ToolResultBlock, ToolUseBlock, UserMessage, query,
    )

    answer_parts: list[str] = []
    result = AgentResult(label=label, answer="", total_cost_usd=None)
    started = time.monotonic()
    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        answer_parts.append(block.text)
                        if verbose and block.text.strip():
                            print(f"[{label}] text: {_preview(block.text)}", flush=True)
                    elif isinstance(block, ToolUseBlock) and verbose:
                        # Inputs vary by tool; show name + a one-line preview of the args.
                        print(
                            f"[{label}] tool→ {block.name}({_preview(block.input)})",
                            flush=True,
                        )
                    elif isinstance(block, ThinkingBlock) and verbose:
                        print(f"[{label}] think: {_preview(block.thinking, 100)}", flush=True)
            elif isinstance(message, UserMessage) and verbose:
                # Tool results come back as UserMessages with ToolResultBlocks.
                content = message.content
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, ToolResultBlock):
                            tag = " ERROR" if block.is_error else ""
                            print(
                                f"[{label}] tool←{tag} {_preview(block.content)}",
                                flush=True,
                            )
            elif isinstance(message, SystemMessage) and verbose:
                # SDK surfaces task progress / mirror errors / etc. Useful when
                # the agent stalls — you'll at least see the subtype.
                subtype = getattr(message, "subtype", None)
                if subtype and subtype not in {"init"}:
                    print(f"[{label}] system: {subtype}", flush=True)
            elif isinstance(message, ResultMessage):
                result.total_cost_usd = getattr(message, "total_cost_usd", None)
                usage = getattr(message, "usage", None) or {}
                if hasattr(usage, "model_dump"):
                    usage = usage.model_dump()
                result.usage = dict(usage) if usage else {}
                model_usage = getattr(message, "model_usage", None) or {}
                if hasattr(model_usage, "model_dump"):
                    model_usage = model_usage.model_dump()
                result.model_usage = dict(model_usage) if model_usage else {}
                result.num_turns = getattr(message, "num_turns", None)
                result.duration_ms = getattr(message, "duration_ms", None)
                result.is_error = bool(getattr(message, "is_error", False))
    except Exception as exc:
        result.is_error = True
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        result.wall_ms = int((time.monotonic() - started) * 1000)
        result.answer = "".join(answer_parts).strip()
    return result


def _build_options(
    *,
    cwd: Path,
    model: str,
    skills: list[str],
    extra_dirs: list[Path] | None = None,
):
    from claude_agent_sdk import ClaudeAgentOptions

    return ClaudeAgentOptions(
        cwd=str(cwd),
        model=model,
        # Explicit skill allowlist. `["trident"]` makes the trident skill the
        # only one available; `[]` suppresses every skill (stock baseline).
        # The SDK injects Skill(<name>) into allowed_tools and defaults
        # setting_sources to ["user","project"] so the skill is discovered.
        skills=skills,
        # Additional directories the agent can Read/Edit beyond `cwd`. The
        # trident agent uses this to peek at target-repo files when it needs
        # to verify something post-query without leaving its make-target cwd.
        add_dirs=[str(p) for p in (extra_dirs or [])],
        permission_mode="bypassPermissions",
        system_prompt={"type": "preset", "preset": "claude_code"},
        # Pin every model lane to the benchmark model so subagents (Task
        # tool's general-purpose/Explore/etc., context compaction, session
        # summarization, title generation) don't quietly downgrade to Haiku
        # and skew the comparison. The alias env vars cover agent definitions
        # that say `model: haiku`/`sonnet`/`opus`; the others cover the
        # explicit Claude Code lanes.
        env={
            "ANTHROPIC_MODEL": model,
            "ANTHROPIC_SMALL_FAST_MODEL": model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        },
    )


def trident_prompt(*, question: str, repo_name: str, db_name: str, branch: str) -> str:
    return (
        f"You are answering a question about the repository indexed in trident as "
        f"`{repo_name}` (DB=`{db_name}`, branch=`{branch}`).\n\n"
        f"Use the trident skill for code exploration. The make targets need "
        f"DB={db_name} and REPO_NAME={repo_name} / REPO_LIST={repo_name}.\n\n"
        f"Question: {question}"
    )


def baseline_prompt(*, question: str) -> str:
    return (
        "Answer the following question about the repository in your current "
        f"working directory.\n\nQuestion: {question}"
    )


async def run_trident_agent(
    *,
    question: str,
    repo_name: str,
    db_name: str,
    branch: str,
    repo_path: Path,
    model: str,
    verbose: bool = False,
) -> AgentResult:
    options = _build_options(
        cwd=TRIDENT_REPO_ROOT,
        model=model,
        skills=["trident"],
        extra_dirs=[repo_path.resolve()],
    )
    prompt = trident_prompt(
        question=question, repo_name=repo_name, db_name=db_name, branch=branch,
    )
    return await _run_query("trident", prompt, options, verbose=verbose)


async def run_baseline_agent(
    *,
    question: str,
    repo_path: Path,
    model: str,
    verbose: bool = False,
) -> AgentResult:
    options = _build_options(
        cwd=repo_path.resolve(),
        model=model,
        skills=[],  # explicitly suppress every skill
    )
    prompt = baseline_prompt(question=question)
    return await _run_query("baseline", prompt, options, verbose=verbose)
