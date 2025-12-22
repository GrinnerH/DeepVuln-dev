# app/semantics/builder.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BuildInputs:
    pack_dir: Path
    baseline_query_path: Path
    semantic_facts: Dict[str, Any]


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def _or_list(values: List[str], var: str = "n") -> str:
    # Generates: n = "a" or n = "b" or ...
    if not values:
        return f'{var} = "__deepvuln_no_match__"'
    parts = [f'{var} = "{v}"' for v in values]
    return " or ".join(parts)


def build_semantic_pack(
    pack_dir: Path,
    baseline_query_path: Path,
    semantic_facts: Dict[str, Any],
    templates_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """
    Deterministically render a semantic pack:
      - qlpack.yml
      - ProjectSemantics.qll
      - queries/_regression_with_semantics.ql (baseline query with import ProjectSemantics)
      - SemanticFacts.json (audit)
    Returns paths for downstream usage.
    """
    if templates_dir is None:
        templates_dir = Path(__file__).parent / "templates"

    pack_dir.mkdir(parents=True, exist_ok=True)
    queries_dir = (pack_dir / "queries")
    queries_dir.mkdir(parents=True, exist_ok=True)

    # 1) qlpack.yml
    qlpack_tpl = _read_text(templates_dir / "qlpack.yml.tpl")
    _write_text(pack_dir / "qlpack.yml", qlpack_tpl)

    # 2) ProjectSemantics.qll
    qll_tpl = _read_text(templates_dir / "ProjectSemantics.qll.tpl")

    wrapper_models = semantic_facts.get("wrapper_models") or []
    wrapper_names = [wm.get("name") for wm in wrapper_models if wm.get("name")]
    # MVP: use the first wrapper model as representative for dest/size indices
    # You can extend later to generate one class per wrapper.
    dest_arg = 0
    size_arg = -1  # -1 means unknown/unbounded
    if wrapper_models:
        dest_arg = int(wrapper_models[0].get("dest_arg", 0))
        sa = wrapper_models[0].get("size_arg", None)
        size_arg = -1 if sa is None else int(sa)

    wrapper_name_clause = _or_list(wrapper_names, var="n")

    if size_arg >= 0:
        size_expr_body = f"result = this.(FunctionCall).getArgument({size_arg})"
    else:
        # Unknown size => return no result from getSizeExpr (conservative)
        size_expr_body = "none()"

    cap_macros = semantic_facts.get("capacity_macros") or []
    cap_fields = semantic_facts.get("capacity_fields") or []
    cap_macro_clause = _or_list(cap_macros, var="name")
    cap_field_clause = _or_list(cap_fields, var="name")

    qll = (
        qll_tpl
        .replace("// __WRAPPER_NAME_OR_LIST__", wrapper_name_clause)
        .replace("__DEST_ARG__", str(dest_arg))
        .replace("__SIZE_EXPR_BODY__", size_expr_body)
        .replace("// __CAP_MACRO_OR_LIST__", cap_macro_clause)
        .replace("// __CAP_FIELD_OR_LIST__", cap_field_clause)
        .replace("// __GUARD_PATTERN_BODY__", "e = e and 1 = 0 and")
    )

    _write_text(pack_dir / "ProjectSemantics.qll", qll)

    # 3) Regression wrapper query
    base = _read_text(baseline_query_path)
    reg_tpl = _read_text(templates_dir / "_regression_with_semantics.ql.tpl")
    reg_query = reg_tpl.replace("__BASELINE_QUERY_BODY__", base)
    reg_path = queries_dir / "_regression_with_semantics.ql"
    _write_text(reg_path, reg_query)

    # 4) Audit facts
    _write_text(pack_dir / "SemanticFacts.json", json.dumps(semantic_facts, ensure_ascii=False, indent=2))

    return {
        "pack_dir": str(pack_dir),
        "qll_path": str(pack_dir / "ProjectSemantics.qll"),
        "regression_query_path": str(reg_path),
        "qlpack_path": str(pack_dir / "qlpack.yml"),
        "facts_path": str(pack_dir / "SemanticFacts.json"),
    }
