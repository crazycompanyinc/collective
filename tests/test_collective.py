from __future__ import annotations

import json

from click.testing import CliRunner

from collective.cli import cli
from collective.core.db import CollectiveDB
from collective.curation.validation import KnowledgeValidator, QualityScorer
from collective.inheritance.onboarding import AgentOnboarding, KnowledgeInheritance
from collective.knowledge.base import KnowledgeBase
from collective.knowledge.consensus import ConsensusEngine
from collective.knowledge.graph import KnowledgeGraph
from collective.memory.context import ContextManager
from collective.search.semantic import RelevanceRanker, SemanticSearch
from collective.server.app import create_app


def make_db(tmp_path):
    return CollectiveDB(tmp_path / "collective.db")


def seed_auth(db):
    memory = ContextManager(db).share("Felix-CTO", "We use JWT tokens")
    fact = KnowledgeBase(db).remember("Felix-CTO", memory.content)
    return memory, fact


def test_db_initializes_schema(tmp_path):
    db = make_db(tmp_path)
    assert db.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'agents'")


def test_create_agent_is_idempotent(tmp_path):
    db = make_db(tmp_path)
    db.create_agent("A")
    db.create_agent("A")
    assert [a.name for a in db.list_agents()] == ["A"]


def test_create_agent_rejects_blank_name(tmp_path):
    db = make_db(tmp_path)
    try:
        db.create_agent(" ")
    except ValueError as exc:
        assert "agent name" in str(exc)
    else:
        raise AssertionError("blank agent accepted")


def test_share_creates_memory_and_agent(tmp_path):
    db = make_db(tmp_path)
    memory = ContextManager(db).share("A", "A learned auth")
    assert memory.agent == "A"
    assert db.list_agents()[0].name == "A"


def test_empty_memory_is_rejected(tmp_path):
    db = make_db(tmp_path)
    try:
        ContextManager(db).share("A", "")
    except ValueError as exc:
        assert "memory content" in str(exc)
    else:
        raise AssertionError("empty memory accepted")


def test_remember_extracts_jwt_auth_fact(tmp_path):
    db = make_db(tmp_path)
    fact = KnowledgeBase(db).remember("Felix-CTO", "We use JWT tokens")
    assert (fact.subject, fact.predicate, fact.object) == ("auth", "uses", "JWT tokens")


def test_remember_extracts_session_conflict(tmp_path):
    db = make_db(tmp_path)
    fact = KnowledgeBase(db).remember("Agent-Beta", "We use sessions")
    assert fact.object == "sessions"


def test_remember_extracts_generic_we_use_fact(tmp_path):
    db = make_db(tmp_path)
    fact = KnowledgeBase(db).remember("A", "We use Postgres")
    assert (fact.subject, fact.predicate, fact.object) == ("system", "uses", "Postgres")


def test_add_fact_deduplicates_per_source(tmp_path):
    db = make_db(tmp_path)
    kb = KnowledgeBase(db)
    one = kb.add_fact("auth", "uses", "JWT", "We use JWT", "A")
    two = kb.add_fact("auth", "uses", "JWT", "We use JWT", "A")
    assert one.id == two.id
    assert len(db.list_facts()) == 1


def test_multiple_agents_can_support_same_fact(tmp_path):
    db = make_db(tmp_path)
    kb = KnowledgeBase(db)
    kb.add_fact("auth", "uses", "JWT", "We use JWT", "A")
    kb.add_fact("auth", "uses", "JWT", "We use JWT", "B")
    assert len(db.list_facts()) == 2


def test_relevance_ranker_scores_related_auth_terms():
    ranker = RelevanceRanker()
    assert ranker.score("What auth do we use?", "We use JWT tokens") > 0


def test_search_returns_memory_and_fact(tmp_path):
    db = make_db(tmp_path)
    seed_auth(db)
    kinds = {result.kind for result in SemanticSearch(db).search("auth")}
    assert {"memory", "fact"} <= kinds


def test_search_orders_best_match_first(tmp_path):
    db = make_db(tmp_path)
    seed_auth(db)
    ContextManager(db).share("Jim", "Frontend uses React")
    results = SemanticSearch(db).search("auth jwt")
    assert "JWT" in results[0].content


def test_knowledge_query_returns_facts(tmp_path):
    db = make_db(tmp_path)
    seed_auth(db)
    assert KnowledgeBase(db).query("authentication method")[0].object == "JWT tokens"


def test_context_includes_shared_fact_from_other_agent(tmp_path):
    db = make_db(tmp_path)
    seed_auth(db)
    context = ContextManager(db).get_context("Felix-Jim", "What auth do we use?")
    assert context.facts[0].source_agent == "Felix-CTO"


def test_context_summary_mentions_inherited_facts(tmp_path):
    db = make_db(tmp_path)
    _, fact = seed_auth(db)
    AgentOnboarding(db).onboard("Agent-Alpha")
    summary = ContextManager(db).get_context("Agent-Alpha", "auth").summary()
    assert str(fact.id) in summary


def test_inheritance_records_existing_facts(tmp_path):
    db = make_db(tmp_path)
    _, fact = seed_auth(db)
    inherited = KnowledgeInheritance(db).inherit("Agent-Alpha")
    assert inherited[0].id == fact.id
    assert db.inherited_fact_ids("Agent-Alpha") == {fact.id}


def test_inheritance_skips_rejected_by_default(tmp_path):
    db = make_db(tmp_path)
    _, fact = seed_auth(db)
    db.update_fact_status(fact.id, "rejected")
    assert KnowledgeInheritance(db).inherit("A") == []


def test_inheritance_can_include_rejected(tmp_path):
    db = make_db(tmp_path)
    _, fact = seed_auth(db)
    db.update_fact_status(fact.id, "rejected")
    assert KnowledgeInheritance(db).inherit("A", include_rejected=True)[0].id == fact.id


def test_onboarding_returns_report(tmp_path):
    db = make_db(tmp_path)
    seed_auth(db)
    report = AgentOnboarding(db).onboard("New")
    assert report.agent == "New"
    assert len(report.inherited) == 1


def test_validator_accepts_high_quality_fact(tmp_path):
    db = make_db(tmp_path)
    _, fact = seed_auth(db)
    updated = KnowledgeValidator(db).rate(fact.id, "B", 1)
    assert updated.status == "accepted"


def test_validator_rejects_low_quality_fact(tmp_path):
    db = make_db(tmp_path)
    _, fact = seed_auth(db)
    db.update_fact_status(fact.id, "proposed", 0.25)
    updated = KnowledgeValidator(db).rate(fact.id, "B", -1)
    assert updated.status == "rejected"


def test_validator_rejects_invalid_rating(tmp_path):
    db = make_db(tmp_path)
    _, fact = seed_auth(db)
    try:
        KnowledgeValidator(db).rate(fact.id, "B", 2)
    except ValueError as exc:
        assert "rating" in str(exc)
    else:
        raise AssertionError("invalid rating accepted")


def test_quality_scorer_bounds_scores(tmp_path):
    db = make_db(tmp_path)
    _, fact = seed_auth(db)
    db.validate_fact(fact.id, "A", 1)
    assert 0 <= QualityScorer(db).score(fact) <= 1


def test_consensus_accepts_supported_jwt_over_sessions(tmp_path):
    db = make_db(tmp_path)
    _, jwt = seed_auth(db)
    sessions = KnowledgeBase(db).remember("Agent-Beta", "We use sessions")
    db.validate_fact(jwt.id, "A", 1)
    db.validate_fact(sessions.id, "B", -1)
    result = ConsensusEngine(db).build("auth")
    assert result.winner.object == "JWT tokens"
    assert result.rejected[0].object == "sessions"


def test_consensus_handles_missing_topic(tmp_path):
    result = ConsensusEngine(make_db(tmp_path)).build("nothing")
    assert result.winner is None
    assert "No facts" in result.explanation


def test_consensus_groups_same_fact_from_multiple_agents(tmp_path):
    db = make_db(tmp_path)
    kb = KnowledgeBase(db)
    kb.add_fact("auth", "uses", "JWT", "A says JWT", "A")
    kb.add_fact("auth", "uses", "JWT", "B says JWT", "B")
    result = ConsensusEngine(db).build("auth")
    assert len(result.accepted) == 2


def test_knowledge_graph_contains_agents_and_fact_edges(tmp_path):
    db = make_db(tmp_path)
    seed_auth(db)
    graph = KnowledgeGraph(db).to_dict()
    node_ids = {node["id"] for node in graph["nodes"]}
    assert {"Felix-CTO", "auth", "JWT tokens"} <= node_ids
    assert any(edge["predicate"] == "uses" for edge in graph["edges"])


def test_knowledge_graph_json_is_valid(tmp_path):
    db = make_db(tmp_path)
    seed_auth(db)
    parsed = json.loads(KnowledgeGraph(db).to_json())
    assert "nodes" in parsed and "edges" in parsed


def test_knowledge_graph_ascii_has_relationship(tmp_path):
    db = make_db(tmp_path)
    seed_auth(db)
    assert "auth -[uses]-> JWT tokens" in KnowledgeGraph(db).ascii()


def test_fastapi_app_registers_share_and_query_routes(tmp_path):
    app = create_app(str(tmp_path / "api.db"))
    paths = {route.path for route in app.routes}
    assert {"/share", "/query"} <= paths


def test_fastapi_app_registers_graph_endpoint(tmp_path):
    app = create_app(str(tmp_path / "api.db"))
    paths = {route.path for route in app.routes}
    assert "/knowledge-graph" in paths


def test_cli_init(tmp_path):
    result = CliRunner().invoke(cli, ["init"], env={"COLLECTIVE_DB": str(tmp_path / "cli.db")})
    assert result.exit_code == 0
    assert "Initialized" in result.output


def test_cli_share_and_query(tmp_path):
    env = {"COLLECTIVE_DB": str(tmp_path / "cli.db")}
    runner = CliRunner()
    assert runner.invoke(cli, ["share", "Felix-CTO", "--memory", "We use JWT tokens"], env=env).exit_code == 0
    result = runner.invoke(cli, ["query", "Felix-Jim", "--question", "What auth do we use?"], env=env)
    assert result.exit_code == 0
    assert "JWT tokens" in result.output


def test_cli_search(tmp_path):
    env = {"COLLECTIVE_DB": str(tmp_path / "cli.db")}
    runner = CliRunner()
    runner.invoke(cli, ["share", "Felix-CTO", "--memory", "We use JWT tokens"], env=env)
    result = runner.invoke(cli, ["search", "auth"], env=env)
    assert "JWT tokens" in result.output


def test_cli_onboard(tmp_path):
    env = {"COLLECTIVE_DB": str(tmp_path / "cli.db")}
    runner = CliRunner()
    runner.invoke(cli, ["share", "Felix-CTO", "--memory", "We use JWT tokens"], env=env)
    result = runner.invoke(cli, ["onboard", "Agent-Alpha"], env=env)
    assert "Inherited 1 fact" in result.output


def test_cli_consensus_resolves_conflict(tmp_path):
    env = {"COLLECTIVE_DB": str(tmp_path / "cli.db")}
    runner = CliRunner()
    runner.invoke(cli, ["share", "Felix-CTO", "--memory", "We use JWT tokens"], env=env)
    runner.invoke(cli, ["share", "Agent-Beta", "--memory", "We use sessions"], env=env)
    result = runner.invoke(cli, ["consensus", "--topic", "auth"], env=env)
    assert "Consensus" in result.output


def test_cli_knowledge_graph_json(tmp_path):
    env = {"COLLECTIVE_DB": str(tmp_path / "cli.db")}
    runner = CliRunner()
    runner.invoke(cli, ["share", "Felix-CTO", "--memory", "We use JWT tokens"], env=env)
    result = runner.invoke(cli, ["knowledge-graph", "--json"], env=env)
    assert json.loads(result.output)["nodes"]


def test_cli_demo_runs_complete_flow(tmp_path):
    result = CliRunner().invoke(cli, ["demo"], env={"COLLECTIVE_DB": str(tmp_path / "demo.db")})
    assert result.exit_code == 0
    assert "Felix-Jim asks" in result.output
    assert "Conflict introduced" in result.output
    assert "Knowledge Graph" in result.output
    assert "New-Agent inherited ALL" in result.output
