"""Convert embedding token counts to USD using a single env-configured rate.

The agent-side cost comes from the Claude Agent SDK's `ResultMessage.total_cost_usd`
directly, so this module only handles the embedding side.
"""

from __future__ import annotations

import os


def embedding_cost_usd(total_input_tokens: int) -> float | None:
    """Returns USD cost or None if no rate is configured.

    Reads `EMBEDDING_COST_PER_M_TOKENS` (USD per 1,000,000 input tokens).
    Zero tokens → 0.0 regardless of rate (full cache-hit indexing).
    """
    if total_input_tokens == 0:
        return 0.0
    raw = os.environ.get("EMBEDDING_COST_PER_M_TOKENS")
    if not raw:
        return None
    try:
        rate = float(raw)
    except ValueError:
        return None
    return (total_input_tokens / 1_000_000.0) * rate
