from __future__ import annotations

from collective.core.db import CollectiveDB
from collective.core.models import Fact


class QualityScorer:
    def __init__(self, db: CollectiveDB) -> None:
        self.db = db

    def score(self, fact: Fact) -> float:
        validations = self.db.validations_for_fact(fact.id)
        validation_delta = sum(row["rating"] for row in validations) * 0.1
        source_bonus = 0.05 if fact.source_agent else 0.0
        status_bonus = 0.1 if fact.status == "accepted" else -0.1 if fact.status == "rejected" else 0.0
        return round(max(0.0, min(1.0, fact.confidence + validation_delta + source_bonus + status_bonus)), 4)


class KnowledgeValidator:
    def __init__(self, db: CollectiveDB, scorer: QualityScorer | None = None) -> None:
        self.db = db
        self.scorer = scorer or QualityScorer(db)

    def rate(self, fact_id: int, agent: str, rating: int, note: str = "") -> Fact:
        self.db.validate_fact(fact_id, agent, rating, note)
        fact = next(f for f in self.db.list_facts() if f.id == fact_id)
        quality = self.scorer.score(fact)
        status = "accepted" if quality >= 0.8 else "rejected" if quality <= 0.3 else fact.status
        return self.db.update_fact_status(fact_id, status, quality)
