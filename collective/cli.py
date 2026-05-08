from __future__ import annotations

import os

import click

from collective.core.db import CollectiveDB
from collective.curation.validation import KnowledgeValidator
from collective.inheritance.onboarding import AgentOnboarding
from collective.knowledge.base import KnowledgeBase
from collective.knowledge.consensus import ConsensusEngine
from collective.knowledge.graph import KnowledgeGraph
from collective.memory.context import ContextManager
from collective.search.semantic import SemanticSearch


def db_path() -> str:
    return os.environ.get("COLLECTIVE_DB", ".collective/collective.db")


def open_db() -> CollectiveDB:
    return CollectiveDB(db_path())


@click.group()
def cli() -> None:
    """Shared memory and knowledge layer for agent societies."""


@cli.command()
def init() -> None:
    """Initialize the Collective database."""
    db = open_db()
    click.echo(f"Initialized Collective at {db.path}")


@cli.command()
@click.argument("agent")
@click.option("--memory", "memory_content", required=True, help="Memory content to share.")
def share(agent: str, memory_content: str) -> None:
    """Share memory and extract a knowledge fact."""
    db = open_db()
    memory = ContextManager(db).share(agent, memory_content)
    fact = KnowledgeBase(db).remember(agent, memory_content)
    click.echo(f"Shared memory #{memory.id} from {agent}")
    click.echo(f"Fact #{fact.id}: {fact.subject} {fact.predicate} {fact.object}")


@cli.command()
@click.argument("agent")
@click.option("--question", required=True, help="Question to answer from shared context.")
def query(agent: str, question: str) -> None:
    """Query relevant shared context for an agent."""
    db = open_db()
    context = ContextManager(db).get_context(agent, question)
    click.echo(context.summary())
    if context.facts:
        best = context.facts[0]
        click.echo(f"Answer: {best.content}")
    else:
        click.echo("Answer: no shared knowledge found")


@cli.command("knowledge-graph")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON graph.")
def knowledge_graph(as_json: bool) -> None:
    """Show the knowledge graph."""
    graph = KnowledgeGraph(open_db())
    click.echo(graph.to_json() if as_json else graph.ascii())


@cli.command()
@click.option("--topic", required=True, help="Topic to build consensus on.")
def consensus(topic: str) -> None:
    """Build consensus and resolve conflicts for a topic."""
    result = ConsensusEngine(open_db()).build(topic)
    click.echo(result.explanation)
    if result.winner:
        click.echo(f"Winner: {result.winner.content}")


@cli.command()
@click.argument("query_text")
def search(query_text: str) -> None:
    """Search across all shared memories and facts."""
    results = SemanticSearch(open_db()).search(query_text)
    if not results:
        click.echo("No results")
        return
    for result in results:
        click.echo(f"{result.kind} #{result.id} score={result.score:.3f}: {result.content}")


@cli.command()
@click.argument("new_agent")
def onboard(new_agent: str) -> None:
    """Create an agent and inherit existing knowledge."""
    report = AgentOnboarding(open_db()).onboard(new_agent)
    click.echo(f"Onboarded {new_agent}")
    click.echo(f"Inherited {len(report.inherited)} fact(s)")
    for fact in report.inherited:
        click.echo(f"- {fact.content}")


@cli.command()
@click.argument("fact_id", type=int)
@click.argument("agent")
@click.argument("rating", type=int)
@click.option("--note", default="", help="Validation note.")
def validate(fact_id: int, agent: str, rating: int, note: str) -> None:
    """Rate a fact: -1 reject, 0 neutral, 1 support."""
    fact = KnowledgeValidator(open_db()).rate(fact_id, agent, rating, note)
    click.echo(f"Fact #{fact.id} quality={fact.confidence:.2f} status={fact.status}")


@cli.command()
def demo() -> None:
    """Run the complete Collective demo."""
    db = open_db()
    context = ContextManager(db)
    kb = KnowledgeBase(db)
    inheritance = AgentOnboarding(db)
    consensus_engine = ConsensusEngine(db)
    graph = KnowledgeGraph(db)

    click.echo("Collective demo")
    for agent in ["Felix-CTO", "Agent-Alpha", "Felix-Jim"]:
        db.create_agent(agent)
        click.echo(f"Agent ready: {agent}")

    memory = context.share("Felix-CTO", "We use JWT tokens")
    fact = kb.remember("Felix-CTO", memory.content)
    click.echo(f"Felix-CTO shared: {memory.content}")

    alpha_report = inheritance.onboard("Agent-Alpha")
    click.echo(f"Agent-Alpha inherited {len(alpha_report.inherited)} fact(s)")

    jim_context = context.get_context("Felix-Jim", "What auth do we use?")
    answer = jim_context.facts[0].content if jim_context.facts else "unknown"
    click.echo(f"Felix-Jim asks: What auth do we use?")
    click.echo(f"Collective answers: {answer}")

    conflict = kb.remember("Agent-Beta", "We use sessions")
    click.echo(f"Conflict introduced by Agent-Beta: {conflict.content}")
    db.validate_fact(fact.id, "Agent-Alpha", 1, "Inherited and verified")
    db.validate_fact(fact.id, "Felix-Jim", 1, "Matches observed implementation")
    db.validate_fact(conflict.id, "Felix-CTO", -1, "Superseded by JWT")
    result = consensus_engine.build("auth")
    click.echo(result.explanation)

    click.echo(graph.ascii())

    new_report = inheritance.onboard("New-Agent")
    click.echo(f"New-Agent inherited ALL existing accepted/proposed knowledge: {len(new_report.inherited)} fact(s)")


if __name__ == "__main__":
    cli()
