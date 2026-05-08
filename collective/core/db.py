from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

from collective.core.models import Agent, Fact, Memory, utc_now


class CollectiveDB:
    """Small SQLite repository used by the CLI, API, and tests."""

    def __init__(self, path: str | Path = ".collective/collective.db") -> None:
        self.path = Path(path)
        if self.path != Path(":memory:"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.init()

    def init(self) -> None:
        self.conn.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS agents (
                name TEXT PRIMARY KEY,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(agent) REFERENCES agents(name)
            );
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                content TEXT NOT NULL,
                source_agent TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'proposed',
                created_at TEXT NOT NULL,
                UNIQUE(subject, predicate, object, source_agent),
                FOREIGN KEY(source_agent) REFERENCES agents(name)
            );
            CREATE TABLE IF NOT EXISTS validations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id INTEGER NOT NULL,
                agent TEXT NOT NULL,
                rating INTEGER NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                UNIQUE(fact_id, agent),
                FOREIGN KEY(fact_id) REFERENCES facts(id),
                FOREIGN KEY(agent) REFERENCES agents(name)
            );
            CREATE TABLE IF NOT EXISTS inheritance_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent TEXT NOT NULL,
                inherited_fact_id INTEGER NOT NULL,
                source_agent TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(agent, inherited_fact_id),
                FOREIGN KEY(agent) REFERENCES agents(name),
                FOREIGN KEY(inherited_fact_id) REFERENCES facts(id),
                FOREIGN KEY(source_agent) REFERENCES agents(name)
            );
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def create_agent(self, name: str) -> Agent:
        name = name.strip()
        if not name:
            raise ValueError("agent name cannot be empty")
        now = utc_now()
        self.conn.execute(
            "INSERT OR IGNORE INTO agents(name, created_at) VALUES (?, ?)",
            (name, now),
        )
        self.conn.commit()
        row = self.conn.execute("SELECT * FROM agents WHERE name = ?", (name,)).fetchone()
        return Agent(name=row["name"], created_at=row["created_at"])

    def list_agents(self) -> list[Agent]:
        rows = self.conn.execute("SELECT * FROM agents ORDER BY created_at, name").fetchall()
        return [Agent(name=row["name"], created_at=row["created_at"]) for row in rows]

    def add_memory(self, agent: str, content: str) -> Memory:
        self.create_agent(agent)
        content = content.strip()
        if not content:
            raise ValueError("memory content cannot be empty")
        now = utc_now()
        cur = self.conn.execute(
            "INSERT INTO memories(agent, content, created_at) VALUES (?, ?, ?)",
            (agent, content, now),
        )
        self.conn.commit()
        return Memory(id=cur.lastrowid, agent=agent, content=content, created_at=now)

    def list_memories(self, agent: str | None = None, limit: int | None = None) -> list[Memory]:
        sql = "SELECT * FROM memories"
        params: list[Any] = []
        if agent:
            sql += " WHERE agent = ?"
            params.append(agent)
        sql += " ORDER BY created_at DESC, id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [Memory(id=row["id"], agent=row["agent"], content=row["content"], created_at=row["created_at"]) for row in rows]

    def add_fact(
        self,
        subject: str,
        predicate: str,
        object_: str,
        content: str,
        source_agent: str,
        confidence: float = 0.7,
        status: str = "proposed",
    ) -> Fact:
        self.create_agent(source_agent)
        now = utc_now()
        self.conn.execute(
            """
            INSERT INTO facts(subject, predicate, object, content, source_agent, confidence, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(subject, predicate, object, source_agent)
            DO UPDATE SET confidence = excluded.confidence, status = excluded.status, content = excluded.content
            """,
            (subject, predicate, object_, content, source_agent, confidence, status, now),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT * FROM facts WHERE subject = ? AND predicate = ? AND object = ? AND source_agent = ?",
            (subject, predicate, object_, source_agent),
        ).fetchone()
        return self._fact_from_row(row)

    def update_fact_status(self, fact_id: int, status: str, confidence: float | None = None) -> Fact:
        if confidence is None:
            self.conn.execute("UPDATE facts SET status = ? WHERE id = ?", (status, fact_id))
        else:
            self.conn.execute("UPDATE facts SET status = ?, confidence = ? WHERE id = ?", (status, confidence, fact_id))
        self.conn.commit()
        row = self.conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        if row is None:
            raise KeyError(f"fact {fact_id} not found")
        return self._fact_from_row(row)

    def list_facts(self, status: str | None = None) -> list[Fact]:
        sql = "SELECT * FROM facts"
        params: list[Any] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY confidence DESC, created_at DESC, id DESC"
        rows = self.conn.execute(sql, params).fetchall()
        return [self._fact_from_row(row) for row in rows]

    def facts_for_topic(self, topic: str) -> list[Fact]:
        needle = f"%{topic.lower()}%"
        rows = self.conn.execute(
            """
            SELECT * FROM facts
            WHERE lower(subject) LIKE ? OR lower(predicate) LIKE ? OR lower(object) LIKE ? OR lower(content) LIKE ?
            ORDER BY confidence DESC, id DESC
            """,
            (needle, needle, needle, needle),
        ).fetchall()
        return [self._fact_from_row(row) for row in rows]

    def validate_fact(self, fact_id: int, agent: str, rating: int, note: str = "") -> None:
        if rating < -1 or rating > 1:
            raise ValueError("rating must be -1, 0, or 1")
        self.create_agent(agent)
        now = utc_now()
        self.conn.execute(
            """
            INSERT INTO validations(fact_id, agent, rating, note, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(fact_id, agent)
            DO UPDATE SET rating = excluded.rating, note = excluded.note, created_at = excluded.created_at
            """,
            (fact_id, agent, rating, note, now),
        )
        self.conn.commit()

    def validations_for_fact(self, fact_id: int) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM validations WHERE fact_id = ?", (fact_id,)).fetchall()

    def add_inheritance_event(self, agent: str, fact_id: int, source_agent: str) -> None:
        self.create_agent(agent)
        self.create_agent(source_agent)
        self.conn.execute(
            """
            INSERT OR IGNORE INTO inheritance_events(agent, inherited_fact_id, source_agent, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (agent, fact_id, source_agent, utc_now()),
        )
        self.conn.commit()

    def inherited_fact_ids(self, agent: str) -> set[int]:
        rows = self.conn.execute("SELECT inherited_fact_id FROM inheritance_events WHERE agent = ?", (agent,)).fetchall()
        return {row["inherited_fact_id"] for row in rows}

    def execute(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, tuple(params)).fetchall()

    @staticmethod
    def _fact_from_row(row: sqlite3.Row) -> Fact:
        return Fact(
            id=row["id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            content=row["content"],
            source_agent=row["source_agent"],
            confidence=row["confidence"],
            status=row["status"],
            created_at=row["created_at"],
        )
