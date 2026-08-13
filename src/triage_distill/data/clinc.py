"""Build the CLINC150 benchmark: native splits + frozen 151-label space + teacher-prompt scaffold.

CLINC150 (Larson et al., 2019) is a real intent-classification benchmark - 150
assistant intents across 10 domains, plus an **out-of-scope (OOS)** class for queries
that fit none. We keep its *standard* train/val/test splits (comparable to the
literature) and fold OOS in as a 151st label - which doubles as our `escalate` signal
("OOS ≈ escalate to a human", SPEC Section 4). Unlike Bitext this is a curated set, so we do
NOT dedup/re-split; the value is using the canonical partition.

    uv run python -m triage_distill.data.clinc

Writes data/clinc/splits/{train,val,test}.parquet, artifacts/clinc/label_space.json +
split_manifest.json, and scaffolds prompts/teacher_clinc.md (only if absent - it's the
teacher's design surface).
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd

from triage_distill.datasets import cfg

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
RAW = REPO_ROOT / "data" / "clinc" / "raw" / "data_full.json"
OOS_LABEL = "oos"
C = cfg("clinc")


def _hash(df: pd.DataFrame) -> str:
    payload = "\n".join(f"{t}\t{lbl}" for t, lbl in zip(df["text"], df["label"]))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _download() -> dict:
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading CLINC150 -> {RAW.relative_to(REPO_ROOT)}")
        urllib.request.urlretrieve(SOURCE_URL, RAW)  # noqa: S310 (trusted, pinned host)
    return json.loads(RAW.read_text())


def _split_df(data: dict, in_key: str, oos_key: str) -> pd.DataFrame:
    rows = [(t, lbl) for t, lbl in data[in_key]] + [(t, OOS_LABEL) for t, _ in data[oos_key]]
    return pd.DataFrame(rows, columns=["text", "label"])


def _teacher_prompt_scaffold(labels: list[str]) -> str:
    label_block = ", ".join(labels)
    return f'''# Teacher prompt - CLINC150 (design surface)

**Scaffold - YOURS to refine.** I pre-filled the 151-label list (150 intents + `oos`) and
the reason-before-commit skeleton, reusing the same JSON fields as `teacher.md` so the
whole pipeline (smoke / labeling / targets / prepare) works unchanged. You own the
reasoning design.

## DECISIONS FOR YOU
- [ ] Group the 150 intents by their 10 domains (banking, credit_cards, travel, kitchen,
      home, auto, work, small_talk, meta, utility) with short glosses on the confusable ones -
      a flat list is a weak prompt for 150 classes.
- [ ] Add 1-2 few-shot exemplars (an in-scope one and an `oos` one).
- [ ] Tune the OOS instruction - when should K3 prefer `oos` over a low-confidence intent?

---

## SYSTEM PROMPT
```text
You are an intent classifier for a virtual assistant. Read the user query, reason
briefly, then commit to EXACTLY ONE intent label. Think FIRST, then commit - your JSON
must place the reasoning fields BEFORE "category".

If the query does not clearly fit any listed intent, label it "oos" (out of scope)
rather than forcing a poor match.

INTENT LABELS (choose exactly one for `category`):
{label_block}

Return ONLY a JSON object with these fields, in this order:
{{
  "evidence_to_intent": "<one sentence: evidence in the query and the intent it implies>",
  "why_not_alternatives": "<one sentence: the most-confusable other intent(s) and why they're wrong>",
  "category": "<exactly one label from the list above, or 'oos'>"
}}

Example:
Query: "how do i get to the nearest airport"
{{"evidence_to_intent": "They ask how to get to the airport - a request for travel directions.", "why_not_alternatives": "Not distance (they want the route, not the mileage) and not oos (it maps to a known travel intent).", "category": "directions"}}

# TODO(you): add an `oos` exemplar + domain grouping, then delete this line.
```
'''


def main() -> None:
    data = _download()
    C.splits_dir.mkdir(parents=True, exist_ok=True)
    C.label_space.parent.mkdir(parents=True, exist_ok=True)

    parts = {
        "train": _split_df(data, "train", "oos_train"),
        "val": _split_df(data, "val", "oos_val"),
        "test": _split_df(data, "test", "oos_test"),
    }
    manifest = {"seed": "native-clinc-splits", "source": SOURCE_URL, "oos_label": OOS_LABEL, "splits": {}}
    for name, df in parts.items():
        df = df.reset_index(drop=True)
        df.to_parquet(C.splits_dir / f"{name}.parquet", index=False)
        manifest["splits"][name] = {"rows": len(df), "hash": _hash(df), "n_classes": int(df["label"].nunique())}

    labels = sorted(set(pd.concat(parts.values())["label"]))
    C.label_space.write_text(json.dumps(
        {"labels": labels, "n_classes": len(labels), "oos_label": OOS_LABEL, "source": SOURCE_URL}, indent=2))
    (C.subsample_manifest.parent / "split_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Scaffold the teacher prompt only if the user hasn't started one.
    if not C.teacher_prompt.exists():
        C.teacher_prompt.write_text(_teacher_prompt_scaffold(labels))
        print(f"scaffolded {C.teacher_prompt.relative_to(REPO_ROOT)} (yours to refine)")

    print(json.dumps(manifest, indent=2))
    print(f"labels: {len(labels)} (150 intents + '{OOS_LABEL}')")
    print(f"-> {C.splits_dir.relative_to(REPO_ROOT)}/  +  {C.label_space.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
