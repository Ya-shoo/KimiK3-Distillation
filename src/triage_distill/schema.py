"""Frozen label space + output schemas for Triage-Distill.

The label space is the single source of truth for every model in the panel, the
teacher, the student, and the ablation. It is derived once from the Bitext
dataset by `data/download.py` and frozen to `artifacts/label_space.json`; nothing
should hardcode the class list.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL_SPACE_PATH = REPO_ROOT / "artifacts" / "label_space.json"

PRIORITY_LEVELS = ("low", "medium", "high", "urgent")


@lru_cache(maxsize=None)
def load_label_space(path: str | Path | None = None) -> tuple[str, ...]:
    """Return the frozen, sorted tuple of category labels (default: Bitext).

    Pass a dataset's `label_space.json` path (e.g. from `datasets.cfg(...)`) to load a
    different benchmark's label space; no arg keeps the original Bitext behavior.
    """
    p = Path(path) if path else LABEL_SPACE_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Label space not found at {p}. Run the dataset's build step first "
            "(e.g. `triage_distill.data.download` for Bitext, `triage_distill.data.clinc` for CLINC)."
        )
    return tuple(json.loads(p.read_text())["labels"])


# --- Phase 1 output: category only (gold-labeled) ---------------------------
class CategoryPrediction(BaseModel):
    """What a Phase-1 model emits per ticket (rationale optional, for distillation)."""

    rationale: str | None = Field(default=None, description="Optional reasoning trace (distillation signal).")
    category: str

    @field_validator("category")
    @classmethod
    def _known_category(cls, v: str) -> str:
        labels = load_label_space()
        if v not in labels:
            raise ValueError(f"category '{v}' not in frozen label space ({len(labels)} classes)")
        return v


# --- Phase 2 output: full triage (priority/escalate added on owned data) -----
class Triage(BaseModel):
    category: str
    priority: str
    escalate: bool
    rationale: str | None = None

    @field_validator("category")
    @classmethod
    def _known_category(cls, v: str) -> str:
        labels = load_label_space()
        if v not in labels:
            raise ValueError(f"category '{v}' not in frozen label space")
        return v

    @field_validator("priority")
    @classmethod
    def _known_priority(cls, v: str) -> str:
        if v not in PRIORITY_LEVELS:
            raise ValueError(f"priority '{v}' not in {PRIORITY_LEVELS}")
        return v


def category_json_schema() -> dict:
    """JSON schema (category constrained to the label space) for constrained decoding."""
    labels = list(load_label_space())
    return {
        "type": "object",
        "properties": {
            "rationale": {"type": "string"},
            "category": {"type": "string", "enum": labels},
        },
        "required": ["category"],
        "additionalProperties": False,
    }
