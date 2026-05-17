"""Drop-in replacement for `OpenAICompatibleEmbedder` that records the token
usage returned by the gateway on every embeddings response.

The base embedder at `core/embedder.py:64` discards `resp.usage`; the
benchmark needs it to compute indexing cost. Subclassing (rather than
modifying the base) keeps the production path unchanged.
"""

from __future__ import annotations

from core.embedder import EmbedderConfig, OpenAICompatibleEmbedder


class TrackingEmbedder(OpenAICompatibleEmbedder):
    """Same behaviour as the parent, plus a running token / call tally."""

    def __init__(self, config: EmbedderConfig):
        super().__init__(config)
        self.total_input_tokens: int = 0
        self.api_calls: int = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        kwargs: dict = {"model": self.config.model, "input": texts}
        # Mirror the conditional dimension logic from the parent (embedder.py:82).
        if "openai.com" in self.config.base_url or self.config.model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self.config.dim
        resp = await self._client.embeddings.create(**kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            # OpenAI shape: usage.total_tokens / usage.prompt_tokens. Some
            # gateways only populate one; prefer total_tokens, fall back.
            tokens = getattr(usage, "total_tokens", None) or getattr(usage, "prompt_tokens", None) or 0
            self.total_input_tokens += int(tokens)
        self.api_calls += 1
        return [list(item.embedding) for item in resp.data]
