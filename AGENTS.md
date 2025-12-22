# Repository Guidelines

## Project Structure & Module Organization
- `app/`: Core application code (LangGraph nodes, state, MCP clients, agents).
- `utils/`: Shared helpers (seed loader, IO utilities, JSON helpers).
- `scripts/`: One-off automation such as seed bootstrap (`bootstrap_seed_assets.py`).
- `seeds/`: Seed case definitions and generated manifests (`seed_cases.json`, `seed_assets.json`).
- `artifacts/`: Runtime outputs (databases, logs, SARIF, semantic packs). Treated as generated data.
- `run_graph.py`: Main entry point to execute the LangGraph pipeline.
- `langgraph.json`: Graph configuration.

## Build, Test, and Development Commands
Python 3.12+ is required.
- Install deps: `pip install -r requirements.txt` or `uv sync` (if using `uv.lock`).
- Bootstrap seed assets (Step A):  
  `PYTHONPATH=. python scripts/bootstrap_seed_assets.py --workspace-dir ./artifacts/run`
- Run the graph (Steps B/C):  
  `PYTHONPATH=. python run_graph.py`
- A/B regression helper runs are in `artifacts/gpac_ab_test/` and can be rerun via local scripts.

## Coding Style & Naming Conventions
- Indentation: 4 spaces, Python standard style.
- File names: snake_case for scripts, PascalCase for classes, lower_snake for functions.
- Keep state serializable; do not store clients/loggers inside state objects.
- Prefer small, single‑responsibility node functions in `app/graph/nodes.py`.

## Testing Guidelines
No dedicated test suite is present yet. When adding tests, place them under `tests/` and use `pytest` conventions (`test_*.py`). Include minimal integration tests for seed bootstrap and graph execution when feasible.

## Commit & Pull Request Guidelines
No strict commit convention is enforced in this repo. Use concise, imperative commit messages (e.g., “Fix baseline search paths”). PRs should include:
- A short description of behavior changes.
- Links to relevant issues or seeds.
- Evidence artifacts (logs/SARIF paths) when changing analysis logic.

## Configuration & Security
- `.env` controls CodeQL paths, model settings, and API keys. Do not commit secrets.
- `BASELINE_QUERY_PATH` should point to CodeQL packs in `codeql/qlpacks` for stable resolution.
- `CODEQL_SEARCH_PATH` should include required pack roots.
