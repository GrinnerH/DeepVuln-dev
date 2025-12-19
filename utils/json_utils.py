import json
import logging
import re
from typing import Any

import json_repair

logger = logging.getLogger(__name__)


def _extract_json_from_content(content: str) -> str:
    content = content.strip()
    brace_count = 0
    bracket_count = 0
    seen_opening_brace = False
    seen_opening_bracket = False
    in_string = False
    escape_next = False
    last_valid_end = -1

    for i, char in enumerate(content):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            brace_count += 1
            seen_opening_brace = True
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and seen_opening_brace:
                last_valid_end = i
        elif char == "[":
            bracket_count += 1
            seen_opening_bracket = True
        elif char == "]":
            bracket_count -= 1
            if bracket_count == 0 and seen_opening_bracket:
                last_valid_end = i
    if last_valid_end > 0:
        return content[: last_valid_end + 1]
    return content


def repair_json_output(content: str) -> str:
    content = content.strip()
    if not content:
        return content
    content = _extract_json_from_content(content)
    try:
        repaired = json_repair.loads(content)
        if not isinstance(repaired, (dict, list)):
            return content
        return json.dumps(repaired, ensure_ascii=False)
    except Exception as e:
        logger.debug("JSON repair failed: %s", e)
        return content
