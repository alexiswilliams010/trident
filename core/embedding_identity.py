"""Explicit vector-space identity used by embedding writes and retrieval."""

from dataclasses import dataclass

CHUNK_REPRESENTATION_VERSION = "chunk-v1"


@dataclass(frozen=True)
class EmbeddingIdentity:
    model_name: str
    model_version: str
    dimensions: int
    representation_version: str = CHUNK_REPRESENTATION_VERSION

    def __post_init__(self):
        if not all(
            s.strip() for s in (self.model_name, self.model_version, self.representation_version)
        ):
            raise ValueError("Embedding model and representation versions must be explicit")
        if not 1 <= self.dimensions <= 16000:
            raise ValueError("Embedding dimensions must be between 1 and 16000")


def identity_for(embed_fn, explicit: EmbeddingIdentity | None = None) -> EmbeddingIdentity:
    owner = getattr(embed_fn, "__self__", embed_fn)
    declared = getattr(owner, "embedding_identity", None)
    if explicit is not None:
        if declared is not None and declared != explicit:
            raise ValueError("Requested embedding identity differs from the embedder")
        return explicit
    if declared is None:
        raise ValueError("Pass embedding_identity for an embedder without declared identity")
    return declared
