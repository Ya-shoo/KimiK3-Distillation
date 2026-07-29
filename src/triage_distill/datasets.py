"""Per-dataset path registry so multiple benchmarks share one pipeline.

Each stage (subsample -> label -> targets -> prepare) resolves its paths through
`cfg(dataset)`, so Bitext and CLINC coexist without touching each other's files.
`bitext` keeps the original M0/M1 paths verbatim; `clinc` mirrors them under a
`clinc/` namespace. Add a new benchmark by adding one row here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_A = REPO_ROOT / "artifacts"
_D = REPO_ROOT / "data"
_P = REPO_ROOT / "prompts"


@dataclass(frozen=True)
class DatasetPaths:
    name: str
    splits_dir: Path          # {train,val,test}.parquet
    label_space: Path         # frozen label_space.json
    label_dir: Path           # subsample.parquet, labeled.jsonl, targets/
    subsample_manifest: Path
    teacher_prompt: Path       # the teacher's design-surface .md
    train_dir: Path           # *.messages.jsonl + val_eval.jsonl


DATASETS = {
    "bitext": DatasetPaths(
        name="bitext",
        splits_dir=_D / "splits",
        label_space=_A / "label_space.json",
        label_dir=_D / "label",
        subsample_manifest=_A / "subsample_manifest.json",
        teacher_prompt=_P / "teacher.md",
        train_dir=_D / "train",
    ),
    "clinc": DatasetPaths(
        name="clinc",
        splits_dir=_D / "clinc" / "splits",
        label_space=_A / "clinc" / "label_space.json",
        label_dir=_D / "clinc" / "label",
        subsample_manifest=_A / "clinc" / "subsample_manifest.json",
        teacher_prompt=_P / "teacher_clinc.md",
        train_dir=_D / "clinc" / "train",
    ),
}


def cfg(name: str) -> DatasetPaths:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset '{name}'; choose from {list(DATASETS)}")
    return DATASETS[name]
