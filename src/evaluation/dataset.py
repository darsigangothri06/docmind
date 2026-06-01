from __future__ import annotations

import json
from pathlib import Path


class EvalDataset:
    """Load evaluation test cases from JSON."""

    def load(self, path: str) -> list[dict]:
        data = json.loads(Path(path).read_text())
        return data
