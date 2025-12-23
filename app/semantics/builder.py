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


def _wrapper_class_name(idx: int) -> str:
    return f"WrapperModel_{idx}"


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
    wrapper_defs: List[str] = []
    wrapper_access_terms: List[str] = []
    overlay_access_terms: List[str] = []

    for idx, wm in enumerate(wrapper_models):
        name = wm.get("name")
        if not name:
            continue
        dest_arg = int(wm.get("dest_arg", 0))
        size_arg = wm.get("size_arg", None)
        size_expr_body = (
            f"result = this.(FunctionCall).getArgument({int(size_arg)})"
            if size_arg is not None
            else "none()"
        )
        cls_name = _wrapper_class_name(idx)
        wrapper_defs.append(
            "\n".join(
                [
                    f"  class {cls_name} extends BufferAccess {{",
                    f"    {cls_name}() {{",
                    "      exists(FunctionCall c |",
                    f"        c.getTarget().getName() = \"{name}\" and",
                    "        this = c",
                    "      )",
                    "    }",
                    "",
                    "    override string getName() {",
                    "      result = this.(FunctionCall).getTarget().getName()",
                    "    }",
                    "",
                    "    override Expr getBuffer(string bufferDesc, int accessType) {",
                    f"      result = this.(FunctionCall).getArgument({dest_arg}) and",
                    "      bufferDesc = \"destination buffer\" and",
                    "      accessType = 2",
                    "    }",
                    "",
                    "    override Expr getSizeExpr() {",
                    f"      {size_expr_body}",
                    "    }",
                    "  }",
                ]
            )
        )
        wrapper_access_terms.append(f"ba instanceof {cls_name}")

        overlay_access_terms.append(f"ba instanceof ProjectSemantics::{cls_name}")

    cap_macros = semantic_facts.get("capacity_macros") or []
    cap_fields = semantic_facts.get("capacity_fields") or []
    cap_macro_clause = _or_list(cap_macros, var="name")
    cap_field_clause = _or_list(cap_fields, var="name")

    wrapper_defs_body = "\n\n".join(wrapper_defs) if wrapper_defs else "  // no wrapper models"
    wrapper_access_body = (
        " or ".join(wrapper_access_terms) if wrapper_access_terms else "ba = ba and 1 = 0"
    )
    overlay_access_body = (
        " or ".join(overlay_access_terms) if overlay_access_terms else "ba = ba and 1 = 0"
    )

    qll = (
        qll_tpl
        .replace("// __WRAPPER_CLASS_DEFS__", wrapper_defs_body)
        .replace("// __WRAPPER_ACCESS_PRED__", wrapper_access_body)
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

    # 4) Overlay regression query
    overlay_tpl = _read_text(templates_dir / "_regression_semantic_overlay.ql.tpl")
    overlay_query = overlay_tpl.replace("// __OVERLAY_ACCESS_PRED__", overlay_access_body)
    overlay_path = queries_dir / "_regression_semantic_overlay.ql"
    _write_text(overlay_path, overlay_query)

    # 5) Audit facts
    _write_text(pack_dir / "SemanticFacts.json", json.dumps(semantic_facts, ensure_ascii=False, indent=2))

    return {
        "pack_dir": str(pack_dir),
        "qll_path": str(pack_dir / "ProjectSemantics.qll"),
        "regression_query_path": str(reg_path),
        "overlay_query_path": str(overlay_path),
        "qlpack_path": str(pack_dir / "qlpack.yml"),
        "facts_path": str(pack_dir / "SemanticFacts.json"),
    }
