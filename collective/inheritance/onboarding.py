from __future__ import annotations

from dataclasses import dataclass

from collective.core.db import CollectiveDB
from collective.core.models import Fact


@dataclass(frozen=True)
class OnboardingReport:
    agent: str
    inherited: list[Fact]


class KnowledgeInheritance:
    def __init__(self, db: CollectiveDB) -> None:
        self.db = db

    def inherit(self, new_agent: str, include_rejected: bool = False) -> list[Fact]:
        self.db.create_agent(new_agent)
        facts = [fact for fact in self.db.list_facts() if include_rejected or fact.status != "rejected"]
        for fact in facts:
            self.db.add_inheritance_event(new_agent, fact.id, fact.source_agent)
        return facts


class AgentOnboarding:
    def __init__(self, db: CollectiveDB, inheritance: KnowledgeInheritance | None = None) -> None:
        self.db = db
        self.inheritance = inheritance or KnowledgeInheritance(db)

    def onboard(self, agent: str) -> OnboardingReport:
        return OnboardingReport(agent=agent, inherited=self.inheritance.inherit(agent))
