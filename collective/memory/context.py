from __future__ import annotations

from dataclasses import dataclass

from collective.core.db import CollectiveDB
from collective.core.models import Fact, Memory
from collective.search.semantic import SemanticSearch


@dataclass(frozen=True)
class SharedContext:
    agent: str
    question: str
    memories: list[Memory]
    facts: list[Fact]
    inherited_fact_ids: set[int]

    def summary(self) -> str:
        parts = [f"Context for {self.agent}: {self.question}"]
        parts.extend(f"- memory:{m.agent}: {m.content}" for m in self.memories)
        parts.extend(f"- fact:{f.source_agent}: {f.content} [{f.status}]" for f in self.facts)
        if self.inherited_fact_ids:
            parts.append(f"Inherited facts: {', '.join(map(str, sorted(self.inherited_fact_ids)))}")
        return "\n".join(parts)


class ContextManager:
    def __init__(self, db: CollectiveDB, search: SemanticSearch | None = None) -> None:
        self.db = db
        self.search = search or SemanticSearch(db)

    def share(self, agent: str, content: str) -> Memory:
        return self.db.add_memory(agent, content)

    def get_context(self, agent: str, question: str, limit: int = 8) -> SharedContext:
        self.db.create_agent(agent)
        results = self.search.search(question, limit=limit)
        memory_ids = {result.id for result in results if result.kind == "memory"}
        fact_ids = {result.id for result in results if result.kind == "fact"}
        memories = [m for m in self.db.list_memories() if m.id in memory_ids]
        facts = [f for f in self.db.list_facts() if f.id in fact_ids]
        return SharedContext(agent, question, memories, facts, self.db.inherited_fact_ids(agent))
