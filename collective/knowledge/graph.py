from __future__ import annotations

import json
from typing import Any

import networkx as nx

from collective.core.db import CollectiveDB


class KnowledgeGraph:
    def __init__(self, db: CollectiveDB) -> None:
        self.db = db

    def build(self) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        for agent in self.db.list_agents():
            graph.add_node(agent.name, type="agent")
        for fact in self.db.list_facts():
            graph.add_node(fact.subject, type="subject")
            graph.add_node(fact.object, type="object")
            graph.add_edge(
                fact.subject,
                fact.object,
                predicate=fact.predicate,
                fact_id=fact.id,
                source_agent=fact.source_agent,
                status=fact.status,
                confidence=fact.confidence,
            )
            graph.add_edge(fact.source_agent, fact.subject, predicate="contributed", fact_id=fact.id)
        return graph

    def to_dict(self) -> dict[str, Any]:
        graph = self.build()
        return {
            "nodes": [{"id": node, **data} for node, data in graph.nodes(data=True)],
            "edges": [{"source": u, "target": v, **data} for u, v, data in graph.edges(data=True)],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def ascii(self) -> str:
        lines = ["Knowledge Graph"]
        for edge in self.to_dict()["edges"]:
            lines.append(f"{edge['source']} -[{edge['predicate']}]-> {edge['target']}")
        return "\n".join(lines)
