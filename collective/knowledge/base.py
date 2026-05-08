from __future__ import annotations

import re

from collective.core.db import CollectiveDB
from collective.core.models import Fact
from collective.search.semantic import SemanticSearch


class KnowledgeBase:
    def __init__(self, db: CollectiveDB) -> None:
        self.db = db
        self.search_engine = SemanticSearch(db)

    def remember(self, agent: str, content: str) -> Fact:
        subject, predicate, object_ = self.extract_claim(content)
        return self.add_fact(subject, predicate, object_, content, agent)

    def add_fact(
        self,
        subject: str,
        predicate: str,
        object_: str,
        content: str,
        agent: str,
        confidence: float = 0.7,
        status: str = "proposed",
    ) -> Fact:
        return self.db.add_fact(subject, predicate, object_, content, agent, confidence, status)

    def query(self, question: str, limit: int = 5) -> list[Fact]:
        fact_ids = [result.id for result in self.search_engine.search(question, limit=limit) if result.kind == "fact"]
        facts_by_id = {fact.id: fact for fact in self.db.list_facts()}
        return [facts_by_id[fact_id] for fact_id in fact_ids if fact_id in facts_by_id]

    def all_facts(self) -> list[Fact]:
        return self.db.list_facts()

    @staticmethod
    def extract_claim(content: str) -> tuple[str, str, str]:
        lowered = content.lower()
        if "jwt" in lowered or "token" in lowered or "session" in lowered or "auth" in lowered:
            method = "JWT tokens" if "jwt" in lowered else "sessions" if "session" in lowered else "tokens"
            return ("auth", "uses", method)
        match = re.search(r"we use ([a-zA-Z0-9 _-]+)", content, flags=re.I)
        if match:
            return ("system", "uses", match.group(1).strip())
        words = content.strip().split()
        subject = words[0].lower() if words else "knowledge"
        return (subject, "states", content.strip())
