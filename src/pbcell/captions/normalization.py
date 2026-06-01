"""Utilities for morphology vocabulary normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:
    yaml = None


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def load_normalization(path: Path) -> Dict[str, Dict[str, str]]:
    raw = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(raw)
    else:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:  
            raise ValueError(
                "PyYAML is not installed and the spec is not JSON-compatible. "
                "Install PyYAML or use JSON syntax."
            ) from exc
    if not isinstance(data, dict):
        raise ValueError("Normalization spec root must be a mapping/dict.")
    value_map = data.get("value_map", {})
    cell_type_map = data.get("cell_type_map", {})
    if not isinstance(value_map, dict):
        raise ValueError("Normalization value_map must be a mapping/dict.")
    if not isinstance(cell_type_map, dict):
        raise ValueError("Normalization cell_type_map must be a mapping/dict.")
    return {"value_map": value_map, "cell_type_map": cell_type_map}


def normalize_value(value: Any, value_map: Dict[str, str]) -> str:
    text = _collapse_whitespace(str(value))
    if text in value_map:
        text = _collapse_whitespace(str(value_map[text]))
    return text.lower()


def normalize_cell_type(value: Any, cell_type_map: Dict[str, str]) -> str:
    text = _collapse_whitespace(str(value))
    if text in cell_type_map:
        return cell_type_map[text]
    return text.title()
