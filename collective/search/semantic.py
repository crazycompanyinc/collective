from __future__ import annotations

import math
import re
from collections import Counter

from collective.core.db import CollectiveDB
from collective.core.models import SearchResult


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
SYNONYMS = {
    "auth": {"authentication", "jwt", "token", "tokens", "session", "sessions"},
    "authentication": {"auth", "jwt", "token", "tokens", "session", "sessions"},
    "frontend": {"ui", "client", "browser"},
    "architecture": {"design", "system", "services"},
}


def tokens(text: str) -> list[str]:
    raw = [t.lower() for t in TOKEN_RE.findall(text)]
    expanded: list[str] = []
    for token in raw:
        expanded.append(token)
        expanded.extend(SYNONYMS.get(token, set()))
    return expanded


class RelevanceRanker:
    def score(self, query: str, content: str) -> float:
        q = Counter(tokens(query))
        c = Counter(tokens(content))
        if not q or not c:
            return 0.0
        dot = sum(q[t] * c[t] for t in q)
        q_norm = math.sqrt(sum(v * v for v in q.values()))
        c_norm = math.sqrt(sum(v * v for v in c.values()))
        lexical_boost = 0.15 if query.lower() in content.lower() else 0.0
        return round((dot / (q_norm * c_norm)) + lexical_boost, 6)


class SemanticSearch:
    def __init__(self, db: CollectiveDB, ranker: RelevanceRanker | None = None) -> None:
        self.db = db
        self.ranker = ranker or RelevanceRanker()

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        results: list[SearchResult] = []
        for memory in self.db.list_memories():
            score = self.ranker.score(query, memory.content)
            if score > 0:
                results.append(SearchResult("memory", memory.id, memory.content, score, {"agent": memory.agent}))
        for fact in self.db.list_facts():
            haystack = f"{fact.subject} {fact.predicate} {fact.object} {fact.content}"
            score = self.ranker.score(query, haystack)
            if score > 0:
                results.append(
                    SearchResult(
                        "fact",
                        fact.id,
                        fact.content,
                        score * max(fact.confidence, 0.1),
                        {"source_agent": fact.source_agent, "status": fact.status, "subject": fact.subject},
                    )
                )
        results.sort(key=lambda result: (-result.score, result.kind, result.id))
        return results[:limit]
