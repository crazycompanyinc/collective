from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from collective.core.db import CollectiveDB
from collective.inheritance.onboarding import AgentOnboarding
from collective.knowledge.base import KnowledgeBase
from collective.knowledge.consensus import ConsensusEngine
from collective.knowledge.graph import KnowledgeGraph
from collective.memory.context import ContextManager
from collective.search.semantic import SemanticSearch


class ShareRequest(BaseModel):
    agent: str
    memory: str


class QueryRequest(BaseModel):
    agent: str
    question: str


def create_app(db_path: str | None = None) -> FastAPI:
    db = CollectiveDB(db_path or os.environ.get("COLLECTIVE_DB", ".collective/collective.db"))
    app = FastAPI(title="Collective", version="0.1.0")

    @app.post("/share")
    def share(request: ShareRequest) -> dict:
        memory = ContextManager(db).share(request.agent, request.memory)
        fact = KnowledgeBase(db).remember(request.agent, request.memory)
        return {"memory": memory.__dict__, "fact": fact.__dict__}

    @app.post("/query")
    def query(request: QueryRequest) -> dict:
        context = ContextManager(db).get_context(request.agent, request.question)
        return {
            "agent": context.agent,
            "question": context.question,
            "memories": [memory.__dict__ for memory in context.memories],
            "facts": [fact.__dict__ for fact in context.facts],
        }

    @app.get("/search")
    def search(q: str) -> dict:
        return {"results": [result.__dict__ for result in SemanticSearch(db).search(q)]}

    @app.post("/consensus/{topic}")
    def consensus(topic: str) -> dict:
        result = ConsensusEngine(db).build(topic)
        return {
            "topic": result.topic,
            "winner": result.winner.__dict__ if result.winner else None,
            "accepted": [fact.__dict__ for fact in result.accepted],
            "rejected": [fact.__dict__ for fact in result.rejected],
            "explanation": result.explanation,
        }

    @app.get("/knowledge-graph")
    def graph() -> dict:
        return KnowledgeGraph(db).to_dict()

    @app.post("/onboard/{agent}")
    def onboard(agent: str) -> dict:
        report = AgentOnboarding(db).onboard(agent)
        return {"agent": report.agent, "inherited": [fact.__dict__ for fact in report.inherited]}

    return app


app = create_app()
