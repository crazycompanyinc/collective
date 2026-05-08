from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Agent:
    name: str
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class Memory:
    id: int
    agent: str
    content: str
    created_at: str


@dataclass(frozen=True)
class Fact:
    id: int
    subject: str
    predicate: str
    object: str
    content: str
    source_agent: str
    confidence: float
    status: str
    created_at: str


@dataclass(frozen=True)
class SearchResult:
    kind: str
    id: int
    content: str
    score: float
    metadata: dict[str, Any]
