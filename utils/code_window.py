from __future__ import annotations
from typing import Any, Dict


def extract_code_window(source_text: str, line_number: int, window: int = 25) -> Dict[str, Any]:
    """
    Return a window around `line_number` (1-indexed) with line numbers.
    Marks the focal line with '>>'.
    """
    lines = source_text.splitlines()
    n = len(lines)
    if n == 0:
        return {"start_line": 1, "end_line": 1, "snippet": ""}

    ln = max(1, min(int(line_number), n))
    start = max(1, ln - window)
    end = min(n, ln + window)

    numbered = []
    for i in range(start, end + 1):
        prefix = ">>" if i == ln else "  "
        numbered.append(f"{prefix}{i:5d}: {lines[i-1]}")
    return {"start_line": start, "end_line": end, "snippet": "\n".join(numbered)}
