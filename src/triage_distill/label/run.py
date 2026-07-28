"""Batch-label the frozen subsample with the K3 teacher.

Reads `data/label/subsample.parquet`, calls K3 once per ticket, and appends each
result to `data/label/labeled.jsonl` (one JSON object per line). Key properties:

- Resumable / checkpointed: every finished row is written immediately and keyed by
  `id`; re-running skips ids already present, so a crash costs nothing already paid
  for. Delete the file (or pass --restart) to relabel from scratch.
- Gold-gated preview: records K3's predicted `category`, the Bitext `gold`, and a
  `gold_match` flag. The gating (keep rationale only when they agree) happens when
  building training targets — here we keep everything so mismatches are inspectable.
- Native trace archived: K3's hidden reasoning is stored (design decision — we pay
  for it regardless). See `label/targets.py` for what becomes the student's target.
- Cost accounting: running token + $ totals, printed as it goes.

    uv run python -m triage_distill.label.run --limit 20      # dry run (~$0.15)
    uv run python -m triage_distill.label.run --workers 6     # full batch (resumable)
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from triage_distill.models.teacher import Teacher

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSAMPLE = REPO_ROOT / "data" / "label" / "subsample.parquet"
LABELED = REPO_ROOT / "data" / "label" / "labeled.jsonl"

BATCH_MAX_TOKENS = 1024  # ~5x observed completion; small OpenRouter hold, no truncation


def _done_ids(path: Path) -> set[int]:
    """Ids already labeled without error — safe to skip on resume."""
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "error" not in rec:
            done.add(rec["id"])
    return done


def label_one(teacher: Teacher, row: pd.Series) -> dict:
    """Label a single ticket; on failure return an {error} record instead of raising."""
    try:
        out = teacher.label(row.text)
        pred = out["category"]
        return {
            "id": int(row.id),
            "text": row.text,
            "gold": row.label,
            "pred": pred,
            "gold_match": pred == row.label,
            "result": out["result"],          # tidy JSON: evidence_to_intent / why_not / category
            "reasoning": out["reasoning"],     # archived native trace
            "usage": out["usage"],
            "cost_usd": out["cost_usd"],
        }
    except Exception as e:  # noqa: BLE001 — never let one bad row kill the batch
        return {"id": int(row.id), "text": row.text, "gold": row.label, "error": repr(e)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=6, help="concurrent K3 calls")
    ap.add_argument("--limit", type=int, default=None, help="only label the first N pending rows (dry run)")
    ap.add_argument("--max-tokens", type=int, default=BATCH_MAX_TOKENS)
    ap.add_argument("--restart", action="store_true", help="ignore + overwrite existing labeled.jsonl")
    args = ap.parse_args()

    if not SUBSAMPLE.exists():
        raise FileNotFoundError(f"{SUBSAMPLE} missing. Run `python -m triage_distill.data.subsample` first.")

    df = pd.read_parquet(SUBSAMPLE)
    if args.restart and LABELED.exists():
        LABELED.unlink()
    done = _done_ids(LABELED)
    pending = df[~df["id"].isin(done)]
    if args.limit:
        pending = pending.head(args.limit)

    print(f"subsample: {len(df)} rows | already done: {len(done)} | to label now: {len(pending)}")
    if pending.empty:
        print("nothing to do.")
        return

    teacher = Teacher(max_tokens=args.max_tokens)
    LABELED.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    fh = LABELED.open("a")
    tally = {"ok": 0, "match": 0, "err": 0, "cost": 0.0, "in_tok": 0, "out_tok": 0}
    t0 = time.monotonic()

    def record(rec: dict) -> None:
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if "error" in rec:
                tally["err"] += 1
            else:
                tally["ok"] += 1
                tally["match"] += int(rec["gold_match"])
                tally["cost"] += rec["cost_usd"]
                tally["in_tok"] += rec["usage"]["prompt_tokens"]
                tally["out_tok"] += rec["usage"]["completion_tokens"]
            n = tally["ok"] + tally["err"]
            if n % 25 == 0 or n == len(pending):
                rate = tally["match"] / tally["ok"] if tally["ok"] else 0
                print(f"  {n}/{len(pending)} | gold-match {tally['match']}/{tally['ok']} "
                      f"({rate:.1%}) | err {tally['err']} | ${tally['cost']:.3f}")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(label_one, teacher, row) for _, row in pending.iterrows()]
            for fut in as_completed(futures):
                record(fut.result())
    finally:
        fh.close()

    dt = time.monotonic() - t0
    print(f"\ndone in {dt:.0f}s | labeled {tally['ok']} (+{tally['err']} errors)")
    print(f"gold-match rate: {tally['match']}/{tally['ok']} "
          f"({tally['match'] / tally['ok']:.1%})" if tally["ok"] else "no successes")
    print(f"tokens: in {tally['in_tok']:,} / out {tally['out_tok']:,} | cost ${tally['cost']:.3f}")
    print(f"-> {LABELED.relative_to(REPO_ROOT)} (build targets: python -m triage_distill.label.targets)")


if __name__ == "__main__":
    main()
