# Collective

Collective is a shared memory and knowledge layer for agent societies. It gives agents a common context, a queryable knowledge base, memory inheritance, consensus building, conflict resolution, knowledge curation, semantic search, and a graph of shared knowledge.

## Quick Start

```bash
python3 -m collective.cli init
python3 -m collective.cli share Felix-CTO --memory "We use JWT tokens"
python3 -m collective.cli onboard Agent-Alpha
python3 -m collective.cli query Felix-Jim --question "What auth do we use?"
python3 -m collective.cli demo
```

By default, the CLI stores data in `.collective/collective.db`. Set `COLLECTIVE_DB=/path/to/db.sqlite` to use another database.

## Components

- `collective/core`: SQLite database and domain models
- `collective/memory`: shared context and context manager
- `collective/knowledge`: knowledge base, graph, and consensus engine
- `collective/inheritance`: knowledge inheritance and onboarding
- `collective/search`: semantic search and relevance ranking
- `collective/curation`: validation and quality scoring
- `collective/server`: FastAPI application
- `collective/cli.py`: Click CLI

## Test

```bash
python3 -m pytest
```
