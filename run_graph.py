# run_graph.py
from __future__ import annotations

from dotenv import load_dotenv

from app.graph.graph import build_graph

if __name__ == "__main__":
    load_dotenv(override=True)
    graph = build_graph()
    # Minimal initial state: point to seeds file and target info later.
    init = {
        "target_cwe_id": "CWE-120",
        "seed_cases": [],  # will load from seeds/seed_cases.json by default
        "max_iterations": 3,
        "target_repo_path": "",  # set when you wire target scanning
    }
    out = graph.invoke(init)
    print(out)
