from __future__ import annotations
import json
from typing import Any, Dict, List, Optional


def json_loads_best_effort(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON extraction:
    - If text is pure JSON, parse directly.
    - Else try to parse the first '{' ... last '}' substring.
    """
    text = (text or "").strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    l = text.find("{")
    r = text.rfind("}")
    if l >= 0 and r > l:
        try:
            obj = json.loads(text[l : r + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    return None


def ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]
