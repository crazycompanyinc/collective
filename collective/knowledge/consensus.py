from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from collective.core.db import CollectiveDB
from collective.core.models import Fact


@dataclass(frozen=True)
class ConsensusResult:
    topic: str
    winner: Fact | None
    accepted: list[Fact]
    rejected: list[Fact]
    explanation: str


class ConsensusEngine:
    def __init__(self, db: CollectiveDB) -> None:
        self.db = db

    def build(self, topic: str) -> ConsensusResult:
        facts = self.db.facts_for_topic(topic)
        if not facts:
            return ConsensusResult(topic, None, [], [], f"No facts found for topic '{topic}'.")
        groups: dict[tuple[str, str, str], list[Fact]] = defaultdict(list)
        for fact in facts:
            groups[(fact.subject.lower(), fact.predicate.lower(), fact.object.lower())].append(fact)
        scored: list[tuple[float, tuple[str, str, str], list[Fact]]] = []
        for key, grouped in groups.items():
            validation_score = sum(sum(row["rating"] for row in self.db.validations_for_fact(fact.id)) for fact in grouped)
            source_score = len({fact.source_agent for fact in grouped}) * 0.25
            confidence_score = sum(fact.confidence for fact in grouped)
            scored.append((confidence_score + validation_score + source_score, key, grouped))
        scored.sort(key=lambda item: item[0], reverse=True)
        winning_group = scored[0][2]
        winning_ids = {fact.id for fact in winning_group}
        accepted: list[Fact] = []
        rejected: list[Fact] = []
        for fact in facts:
            if fact.id in winning_ids:
                accepted.append(self.db.update_fact_status(fact.id, "accepted", min(1.0, fact.confidence + 0.2)))
            else:
                rejected.append(self.db.update_fact_status(fact.id, "rejected", max(0.1, fact.confidence - 0.2)))
        winner = accepted[0] if accepted else None
        explanation = f"Consensus for '{topic}' accepted '{winner.object if winner else 'none'}' from {len(accepted)} supporting fact(s)."
        if rejected:
            explanation += f" Rejected {len(rejected)} conflicting fact(s)."
        return ConsensusResult(topic, winner, accepted, rejected, explanation)
