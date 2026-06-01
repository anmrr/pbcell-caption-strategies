"""Load morphology KB for captioning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:  
    yaml = None


def load_kb(path: Path) -> Dict[str, Any]:
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
        raise ValueError("KB root must be a mapping/dict.")
    features = data.get("feature_schema")
    prototypes = {
    k: v.get("prototypes", {})
    for k, v in data.get("cell_types", {}).items()
    }
    if not isinstance(features, dict) or not isinstance(data.get("cell_types"), dict):
        raise ValueError("KB must include features and prototypes mappings.")
    return data
