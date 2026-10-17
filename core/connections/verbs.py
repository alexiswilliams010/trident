"""Versioned interaction vocabulary. Parser links and similarity are not generated edges."""

from dataclasses import dataclass
from types import MappingProxyType

from .contracts import EndpointKind as K

VOCABULARY_VERSION = 1


@dataclass(frozen=True)
class Verb:
    definition: str
    endpoint_pairs: frozenset[tuple[K, K]]


_SITE = frozenset({(K.SITE, K.SITE)})
_RESOURCE = frozenset({(K.SITE, K.RESOURCE)})
VERBS = MappingProxyType({
    "INVOKES": Verb("Requests execution of an operation.", _SITE),
    "DISPATCHES_TO": Verb("Routes an operation to another execution target.", _SITE),
    "SPAWNS": Verb("Starts a process or executable.", _SITE | _RESOURCE),
    "PRODUCES": Verb("Publishes or creates an event or message for consumption.", _RESOURCE),
    "CONSUMES": Verb("Subscribes to or processes events or messages from a resource.", _RESOURCE),
    "READS": Verb("Retrieves data from a resource.", _RESOURCE),
    "WRITES": Verb("Modifies data in a resource.", _RESOURCE),
    "PROVISIONS": Verb("Creates or manages an infrastructure resource.", _RESOURCE),
    "CONFIGURES": Verb("Supplies settings controlling a component or resource.", _SITE | _RESOURCE),
})


def normalize_verb(value: str) -> str:
    canonical = value.strip().upper()
    if canonical == "EMITS":
        canonical = "PRODUCES"
    if canonical not in VERBS:
        raise ValueError(f"Unsupported interaction verb: {value}")
    return canonical
