from __future__ import annotations

"""Bootstrap seed assets (repo checkout + CodeQL DB + vulnerability location).

This script is the automated "Step A" for DeepVuln 2.0.

It reads a seed case file (default: seeds/seed_cases.json), ensures each seed has:
  - a local repo checkout at <workspace_dir>/repos/... pinned to vuln_commit_sha
  - a CodeQL database at <workspace_dir>/codeql_dbs/... built from that checkout
  - a best-effort vuln_line within vulnerable_file for contextual prompting

It then writes the enriched seed list to an output JSON file.
"""

import argparse
import logging
from typing import Any, Dict, List

from dotenv import load_dotenv

from app.mcp.codeql_mcp import CodeQLMCPClient
from utils.seed_loader import bootstrap_seed_case, load_seed_cases, save_seed_cases

logger = logging.getLogger("deepvuln.bootstrap")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> int:
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description="Bootstrap DeepVuln seed assets")
    parser.add_argument("--seed-json", default="seeds/seed_cases.json")
    parser.add_argument("--out-json", default="seeds/seed_assets.json")
    parser.add_argument("--workspace-dir", default="./artifacts/run")
    parser.add_argument("--overwrite-db", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _configure_logging(args.verbose)

    codeql = CodeQLMCPClient()

    seeds: List[Dict[str, Any]] = load_seed_cases(args.seed_json)
    out: List[Dict[str, Any]] = []

    for seed in seeds:
        try:
            enriched = bootstrap_seed_case(
                workspace_dir=args.workspace_dir,
                seed=seed,
                codeql=codeql,
                overwrite_db=args.overwrite_db,
            )
        except Exception as exc:
            logger.exception("Failed to bootstrap seed %s", seed.get("cve_id"), exc_info=exc)
            seed_err = dict(seed)
            seed_err["bootstrap_error"] = str(exc)
            out.append(seed_err)
            continue
        out.append(enriched)

    save_seed_cases(args.out_json, out)
    logger.info("Wrote enriched seeds to %s", args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
