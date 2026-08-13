"""Frontier-panel eval - label-only inference over a benchmark's TEST split.

This is the M3 counterpart to the 4090's student eval: it measures how the
finalized frontier/efficient panel (`configs/models.yaml`) classifies each
benchmark's held-out test tickets, so the paper can put the distilled student
on the same axes (macro-F1 + cost). Plumbing only - the numbers are the models'.

Scope: the OpenRouter-metered panel entries (provider `openrouter`). The two
Anthropic entries (`claude-code-cli`, free via the Claude Code sub) run through a
separate path and are not called here.

Design (mirrors `label/run.py` so behavior is familiar):
- **Label-only**: every model gets the SAME prompt (the teacher's label glosses,
  reused verbatim so it's a fair, generous baseline) but a label-only output
  contract: `{"category": "<one label>"}`. Same label space, same test rows for all.
- **Graded by identical code**: preds are `{id, pred}` aligned to `test_eval.jsonl`
  gold by row-id; scored with `eval.score` (invalid / hallucinated label = WRONG).
- **Resumable / checkpointed**: each row is appended immediately, keyed by `id`;
  re-running skips ids already resolved. Transport errors are NOT marked done
  (retried on resume); a model that returns a valid response but an unusable label
  after retries is recorded as `pred=null` (a real, measured wrong answer).
- **Cost accounting**: records per-call token usage + list-price cost (prices.yaml,
  the chart's axis) and OpenRouter's reported actual spend when available.

    # smoke: 8 val rows on every OpenRouter model, reasoning off (cheap)
    uv run python -m triage_distill.eval.panel --dataset bitext --split val --limit 8
    # full test run, one model
    uv run python -m triage_distill.eval.panel --dataset bitext --model deepseek-v3.2
    # full test run, all OpenRouter models, then aggregate + score
    uv run python -m triage_distill.eval.panel --dataset bitext --model all --score
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from triage_distill.datasets import cfg
from triage_distill.eval.score import score
from triage_distill.models.teacher import _base_url, _client, extract_system_prompt
from triage_distill.schema import load_label_space

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_YAML = REPO_ROOT / "configs" / "models.yaml"
PRICES_YAML = REPO_ROOT / "configs" / "prices.yaml"

load_dotenv(REPO_ROOT / ".env")

# Per-dataset task framing for the label-only prompt. The label list + confusable
# guidance are lifted verbatim from the teacher prompt (single source of truth for
# label semantics - zero transcription risk), so only the framing differs here.
TASK_LINE = {
    "bitext": (
        "You are an expert customer-support ticket triager. Read the ticket and "
        "classify it into EXACTLY ONE intent label from the list below."
    ),
    "clinc": (
        "You are an intent classifier for a virtual assistant. Read the user query "
        "and classify it into EXACTLY ONE intent label from the list below. If the "
        'query does not clearly fit any listed intent, label it "oos" rather than '
        "forcing a poor match."
    ),
}
OUTPUT_LINE = (
    'Respond with ONLY a JSON object of the form {"category": "<one label from the '
    'list above>"}. Output no other fields, no reasoning, and no extra text.'
)


def build_label_only_prompt(dataset: str) -> str:
    """The shared label-only system prompt: teacher's label block + a label-only contract."""
    teacher_sys = extract_system_prompt(cfg(dataset).teacher_prompt)
    i = teacher_sys.find("INTENT LABELS")
    j = teacher_sys.find("Return ONLY")
    if i == -1 or j == -1:
        raise ValueError(f"could not locate the label block in the {dataset} teacher prompt")
    labels_block = teacher_sys[i:j].strip()  # label list (+ confusable families for bitext)
    task = TASK_LINE.get(dataset)
    if task is None:
        raise KeyError(f"no TASK_LINE defined for dataset '{dataset}'")
    return f"{task}\n\n{labels_block}\n\n{OUTPUT_LINE}"


def parse_category(raw: str, labels: set[str]) -> str | None:
    """Extract a valid `category` from a model's raw text; None if unparseable/out-of-space.

    Handles bare JSON, markdown-fenced JSON (```json … ``` - Anthropic models do this even
    under response_format), and prose with an embedded object. A returned-but-invalid label
    yields None (scored WRONG), never a crash.
    """
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    cat = None
    try:
        cat = json.loads(s).get("category")
    except (json.JSONDecodeError, AttributeError):
        m = re.search(r"\{.*\}", s, re.S)  # fall back to the first {...} object
        if m:
            try:
                cat = json.loads(m.group(0)).get("category")
            except (json.JSONDecodeError, AttributeError):
                cat = None
    return cat if cat in labels else None


def load_panel(openrouter_only: bool = True) -> dict[str, dict]:
    """Panel entries from models.yaml (default: only the OpenRouter-metered ones)."""
    panel = yaml.safe_load(MODELS_YAML.read_text())["panel"]
    if openrouter_only:
        panel = {k: v for k, v in panel.items() if v.get("provider") == "openrouter"}
    return panel


def _prices() -> dict[str, dict]:
    return yaml.safe_load(PRICES_YAML.read_text())["models"]


class PanelClient:
    """One OpenRouter model, prompted label-only. `.predict(text)` -> validated label + usage."""

    # OpenRouter's unified `reasoning` control (https://openrouter.ai/docs/use-cases/reasoning-tokens).
    _REASONING = {
        "off": {"enabled": False},      # cheapest + strict label-only (skip thinking where supported)
        "low": {"effort": "low"},       # minimal thinking budget
        "default": None,                # provider default (may reason a lot - costly)
    }

    def __init__(self, model: str, price_key: str, *, dataset: str, reasoning: str = "off",
                 max_tokens: int = 512, temperature: float = 0.0):
        self.base = _base_url()
        if "openrouter" not in self.base:
            raise RuntimeError(
                f"panel eval expects an OpenRouter base_url; TEACHER_BASE_URL={self.base!r}")
        self.client: OpenAI = _client(self.base)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = build_label_only_prompt(dataset)
        self.labels = set(load_label_space(cfg(dataset).label_space))
        p = _prices().get(price_key, {})
        self._in_price = (p.get("input") or 0) / 1e6
        self._out_price = (p.get("output") or 0) / 1e6
        # NB: no `provider.require_parameters` here (unlike the teacher client). That filter
        # rejects the OpenAI endpoints outright ("no endpoints handle the requested
        # parameters" 404). We don't need it: we validate + retry the JSON ourselves and a
        # non-parseable reply counts as WRONG, so a provider ignoring response_format only
        # hurts itself, never mis-scores. `usage.include` asks OpenRouter for actual $ spend.
        self._extra_body: dict = {"usage": {"include": True}}
        r = self._REASONING.get(reasoning, self._REASONING["off"])
        if r is not None:
            self._extra_body["reasoning"] = r

    def predict(self, text: str, max_retries: int = 2) -> dict:
        """Return {'pred','invalid','usage','cost_usd','cost_or','raw'}.

        `pred` is a valid label or None (None => model never produced a usable label;
        counts as wrong downstream). Raises on transport/API errors so the caller can
        record a retry-on-resume error record instead of a measured wrong answer.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f'Ticket: "{text}"'},
        ]
        last_raw = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": None}
        cost_or = None
        for _ in range(max_retries + 1):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                extra_body=self._extra_body,
            )
            msg = resp.choices[0].message
            raw = msg.content or ""
            last_raw = raw
            u = resp.usage
            details = getattr(u, "completion_tokens_details", None)
            usage = {
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "reasoning_tokens": getattr(details, "reasoning_tokens", None),
            }
            cost_or = getattr(u, "cost", None)
            if cost_or is None and getattr(u, "model_extra", None):
                cost_or = u.model_extra.get("cost")
            cost = u.prompt_tokens * self._in_price + u.completion_tokens * self._out_price
            cat = self._parse(raw)
            if cat is not None:
                return {"pred": cat, "invalid": False, "usage": usage,
                        "cost_usd": cost, "cost_or": cost_or, "raw": raw}
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "That was not a valid label. Reply with ONLY "
                 '{"category": "<one exact label from the list>"}.'},
            ]
        # Valid HTTP responses but no usable label after retries -> a measured wrong answer.
        cost = usage["prompt_tokens"] * self._in_price + usage["completion_tokens"] * self._out_price
        return {"pred": None, "invalid": True, "usage": usage,
                "cost_usd": cost, "cost_or": cost_or, "raw": last_raw}

    def _parse(self, raw: str) -> str | None:
        return parse_category(raw, self.labels)


def _resolve_claude_bin() -> str:
    """Path to the REAL Claude Code binary, avoiding terminal-multiplexer shims.

    A `claude` on PATH can be a cmux (or similar) shim that execs a session wrapper whose
    hooks fail under concurrent headless invocations (`exit 1`, empty stderr). We want the
    actual install. Order: $CLAUDE_BIN → ~/.local/bin/claude → first non-shim PATH entry.
    """
    override = os.environ.get("CLAUDE_BIN")
    if override and Path(override).exists():
        return override
    local = Path.home() / ".local" / "bin" / "claude"
    if local.exists():
        return str(local)
    for p in os.environ.get("PATH", "").split(os.pathsep):
        cand = Path(p) / "claude"
        if cand.exists() and "cmux-cli-shims" not in str(cand):
            return str(cand)
    return "claude"


def _clean_cli_env() -> dict:
    """Env with the multiplexer's node injection + shim vars stripped, so the real binary
    runs clean under concurrency (NODE_OPTIONS pre-loads a cmux .cjs; CMUX_* drive the shim)."""
    return {k: v for k, v in os.environ.items()
            if k != "NODE_OPTIONS" and not k.startswith("CMUX_")}


class ClaudeCliClient:
    """A Claude model via the Claude Code CLI (`claude -p`) - free via the user's sub, no API key.

    Same `.predict(text)` shape as `PanelClient`. Runs each ticket in a fresh headless
    session with the default agent prompt REPLACED by our label-only prompt
    (`--system-prompt` + `--exclude-dynamic-system-prompt-sections`), no MCP, no tools -
    so the Anthropic entries see the SAME task framing as the OpenRouter panel. The CLI's
    reported `total_cost_usd` is the Anthropic API-list-equivalent of that call, so it
    doubles as the cost-axis figure (prices.yaml has no Fable list price).
    """

    def __init__(self, model: str, price_key: str, *, dataset: str, timeout_s: int = 300,
                 **_ignored):
        self.model = model
        self.price_key = price_key
        self.timeout_s = timeout_s
        self.system_prompt = build_label_only_prompt(dataset)
        self.labels = set(load_label_space(cfg(dataset).label_space))
        self.bin = _resolve_claude_bin()
        self.env = _clean_cli_env()

    def _argv(self, text: str) -> list[str]:
        return [
            self.bin, "-p", f'Ticket: "{text}"',
            "--model", self.model,
            "--system-prompt", self.system_prompt,
            "--exclude-dynamic-system-prompt-sections",
            "--output-format", "json",
            "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
            "--tools", "",
        ]

    def predict(self, text: str, max_retries: int = 1) -> dict:
        """Return {'pred','invalid','usage','cost_usd','cost_or','raw'}; raises on CLI/transport error."""
        last_raw = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": None}
        cost = 0.0
        for attempt in range(max_retries + 1):
            argv = self._argv(text)
            if attempt:  # firmer nudge on retry (stateless CLI - can't continue the turn)
                argv[2] += '\n\nReturn ONLY the JSON object, e.g. {"category": "track_order"}.'
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout_s,
                                   env=self.env)
            if proc.returncode != 0:
                raise RuntimeError(f"claude -p exit {proc.returncode}: {proc.stderr.strip()[:200]}")
            d = json.loads(proc.stdout)
            if d.get("is_error"):
                raise RuntimeError(f"claude -p api_error: {d.get('api_error_status')} {str(d.get('result'))[:160]}")
            u = d.get("usage", {})
            last_raw = d.get("result", "") or ""
            usage = {
                "prompt_tokens": (u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                                  + u.get("cache_read_input_tokens", 0)),
                "completion_tokens": u.get("output_tokens", 0),
                "reasoning_tokens": None,
            }
            cost = d.get("total_cost_usd", 0.0) or 0.0
            cat = self._parse(last_raw)
            if cat is not None:
                return {"pred": cat, "invalid": False, "usage": usage,
                        "cost_usd": cost, "cost_or": cost, "raw": last_raw}
        return {"pred": None, "invalid": True, "usage": usage,
                "cost_usd": cost, "cost_or": cost, "raw": last_raw}

    def _parse(self, result_text: str) -> str | None:
        return parse_category(result_text, self.labels)


def make_client(entry: dict, *, dataset: str, reasoning: str, max_tokens: int):
    """Dispatch to the right backend for a panel entry's provider."""
    provider = entry.get("provider")
    if provider == "openrouter":
        return PanelClient(entry["model"], entry["price_key"], dataset=dataset,
                           reasoning=reasoning, max_tokens=max_tokens)
    if provider == "claude-code-cli":
        return ClaudeCliClient(entry["model"], entry["price_key"], dataset=dataset)
    raise ValueError(f"unknown provider '{provider}' for {entry}")


# --------------------------------------------------------------------------- run

def _resolved_ids(path: Path) -> set[int]:
    """Ids already resolved (a pred was produced - valid or measured-invalid); safe to skip.

    Transport-error records ({"error": ...}) are NOT resolved, so they re-run on resume.
    """
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


def _predict_one(client: PanelClient, row: dict) -> dict:
    try:
        out = client.predict(row["text"])
        return {"id": int(row["id"]), "pred": out["pred"], "invalid": out["invalid"],
                "usage": out["usage"], "cost_usd": out["cost_usd"], "cost_or": out["cost_or"]}
    except Exception as e:  # noqa: BLE001 - transport/API error: record so resume retries it
        return {"id": int(row["id"]), "error": repr(e)}


def run_model(model_key: str, entry: dict, gold_rows: list[dict], out_dir: Path, *,
              dataset: str, reasoning: str, max_tokens: int, workers: int,
              limit: int | None, restart: bool) -> dict:
    """Run one panel model over the gold rows; write <key>.preds.jsonl; return a summary."""
    preds_path = out_dir / f"{model_key}.preds.jsonl"
    if restart and preds_path.exists():
        preds_path.unlink()
    done = _resolved_ids(preds_path)
    pending = [r for r in gold_rows if int(r["id"]) not in done]
    if limit is not None:
        pending = pending[:limit]

    print(f"\n[{model_key}] {entry['model']} | tier={entry['tier']} reasoning={reasoning} "
          f"| gold={len(gold_rows)} done={len(done)} to-run={len(pending)}")
    if not pending:
        print(f"[{model_key}] nothing to run.")
        return _summarize(model_key, entry, gold_rows, preds_path, dataset)

    client = make_client(entry, dataset=dataset, reasoning=reasoning, max_tokens=max_tokens)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    fh = preds_path.open("a")
    tally = {"ok": 0, "invalid": 0, "err": 0, "cost": 0.0, "cost_or": 0.0,
             "in_tok": 0, "out_tok": 0, "reason_tok": 0}
    t0 = time.monotonic()

    def record(rec: dict) -> None:
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if "error" in rec:
                tally["err"] += 1
            else:
                tally["ok"] += 1
                tally["invalid"] += int(rec["invalid"])
                tally["cost"] += rec["cost_usd"]
                tally["cost_or"] += rec.get("cost_or") or 0.0
                tally["in_tok"] += rec["usage"]["prompt_tokens"]
                tally["out_tok"] += rec["usage"]["completion_tokens"]
                tally["reason_tok"] += rec["usage"].get("reasoning_tokens") or 0
            n = tally["ok"] + tally["err"]
            if n % 50 == 0 or n == len(pending):
                print(f"  [{model_key}] {n}/{len(pending)} | invalid {tally['invalid']} | "
                      f"err {tally['err']} | list-$ {tally['cost']:.3f} "
                      f"| OR-$ {tally['cost_or']:.3f}")

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_predict_one, client, r) for r in pending]
            for fut in as_completed(futures):
                record(fut.result())
    finally:
        fh.close()

    dt = time.monotonic() - t0
    print(f"[{model_key}] done in {dt:.0f}s | ok {tally['ok']} invalid {tally['invalid']} "
          f"err {tally['err']} | tokens in {tally['in_tok']:,}/out {tally['out_tok']:,} "
          f"(reason {tally['reason_tok']:,}) | list-$ {tally['cost']:.4f} OR-$ {tally['cost_or']:.4f}")
    return _summarize(model_key, entry, gold_rows, preds_path, dataset)


def _summarize(model_key: str, entry: dict, gold_rows: list[dict], preds_path: Path,
               dataset: str) -> dict:
    """Score this model's preds vs gold + roll up tokens/cost for the panel summary."""
    labels = list(load_label_space(cfg(dataset).label_space))
    gold = {int(r["id"]): r["gold"] for r in gold_rows}
    recs = [json.loads(l) for l in preds_path.read_text().splitlines() if l.strip()] \
        if preds_path.exists() else []
    preds = {r["id"]: r.get("pred") for r in recs if "error" not in r}
    errs = sum(1 for r in recs if "error" in r)
    # Score each model on exactly the test rows it PREDICTED (its ids ∩ gold), not the full
    # gold set. For the full-test models this is the whole test split (no change); for a
    # subsample entry (e.g. Fable on a fixed 2k sample) it scores only those rows - a
    # never-attempted row must not count as a wrong answer. Returned-but-invalid preds
    # (pred=null) stay in `preds`, so they still score as wrong.
    gold_sub = {i: gold[i] for i in preds if i in gold}
    rep = score(preds, gold_sub, labels)

    n = max(len(preds), 1)
    in_tok = sum(r["usage"]["prompt_tokens"] for r in recs if "error" not in r)
    out_tok = sum(r["usage"]["completion_tokens"] for r in recs if "error" not in r)
    reason_tok = sum((r["usage"].get("reasoning_tokens") or 0) for r in recs if "error" not in r)
    list_cost = sum(r["cost_usd"] for r in recs if "error" not in r)
    or_cost = sum((r.get("cost_or") or 0.0) for r in recs if "error" not in r)
    mean_in, mean_out = in_tok / n, out_tok / n
    # Cost per 1k tickets on the list-price axis (what the money_chart plots). Built from
    # the recorded per-call list cost so it's uniform across backends: OpenRouter rows carry
    # a tokens×prices.yaml cost; claude-cli rows carry the CLI's API-list-equivalent cost
    # (prices.yaml has no Fable list price, so a token×price formula couldn't cover it).
    cost_per_1k = list_cost / n * 1000

    return {
        "model_key": model_key, "model": entry["model"], "tier": entry["tier"],
        "label": entry.get("label", model_key), "price_key": entry["price_key"],
        "n_scored": rep["n"], "n_preds": len(preds), "transport_errors": errs,
        "macro_f1": rep["macro_f1"], "accuracy": rep["accuracy"],
        "invalid": rep["invalid"], "invalid_rate": rep["invalid_rate"],
        "worst_classes": rep["worst_classes"],
        "mean_input_tokens": round(mean_in, 1), "mean_output_tokens": round(mean_out, 1),
        "total_reasoning_tokens": reason_tok,
        "cost_per_1k_tickets_list": round(cost_per_1k, 5),
        "spend_list_usd": round(list_cost, 4), "spend_openrouter_usd": round(or_cost, 4),
        "per_class_f1": rep["per_class_f1"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="bitext", choices=["bitext", "clinc"])
    ap.add_argument("--model", default="all",
                    help="'all' (every OpenRouter entry), 'all-cli' (the claude-code-cli entries), "
                         "or a comma-separated list of panel keys (e.g. anthropic-haiku-4.5,anthropic-fable-5)")
    ap.add_argument("--exclude", default="",
                    help="comma-separated panel keys to skip (e.g. openai-gpt-5.6-sol)")
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="which gold split to run on (test = the sacred once-only M3 eval)")
    ap.add_argument("--gold", default=None,
                    help="override gold path (e.g. a fixed stratified subsample); default: {split}_eval.jsonl")
    ap.add_argument("--reasoning", default="off", choices=["off", "low", "default"],
                    help="OpenRouter reasoning control (off = cheapest, strict label-only)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only run the first N rows (smoke test)")
    ap.add_argument("--restart", action="store_true", help="ignore + overwrite existing preds")
    ap.add_argument("--score", action="store_true", help="write per-model + panel_summary.json after running")
    args = ap.parse_args()

    c = cfg(args.dataset)
    gold_path = Path(args.gold) if args.gold else c.train_dir / (f"{args.split}_eval.jsonl")
    if not gold_path.exists():
        raise FileNotFoundError(f"{gold_path} missing. Run `python -m triage_distill.train.prepare "
                                f"--dataset {args.dataset}` first.")
    gold_rows = [json.loads(l) for l in gold_path.read_text().splitlines() if l.strip()]

    full = load_panel(openrouter_only=False)
    if args.model == "all":
        panel = {k: v for k, v in full.items() if v.get("provider") == "openrouter"}
    elif args.model == "all-cli":
        panel = {k: v for k, v in full.items() if v.get("provider") == "claude-code-cli"}
    else:
        keys = [x.strip() for x in args.model.split(",") if x.strip()]
        bad = [k for k in keys if k not in full]
        if bad:
            raise KeyError(f"unknown panel key(s) {bad}; choose from {list(full)}, 'all', or 'all-cli'")
        panel = {k: full[k] for k in keys}
    for k in (x.strip() for x in args.exclude.split(",") if x.strip()):
        panel.pop(k, None)

    out_dir = REPO_ROOT / "artifacts" / args.dataset / "eval" / "panel"
    print(f"dataset={args.dataset} split={args.split} rows={len(gold_rows)} "
          f"models={list(panel)} reasoning={args.reasoning}")

    for key, entry in panel.items():
        run_model(key, entry, gold_rows, out_dir, dataset=args.dataset,
                  reasoning=args.reasoning, max_tokens=args.max_tokens,
                  workers=args.workers, limit=args.limit, restart=args.restart)

    # Leaderboard + combined summary are rebuilt from EVERY model with preds on disk (the
    # full panel, not just what ran this invocation) - so a partial run (e.g. only the
    # claude-cli entries) refreshes rather than clobbers panel_summary.json.
    summaries = [_summarize(k, e, gold_rows, out_dir / f"{k}.preds.jsonl", args.dataset)
                 for k, e in full.items() if (out_dir / f"{k}.preds.jsonl").exists()]
    summaries.sort(key=lambda s: s["macro_f1"], reverse=True)
    print("\n=== panel leaderboard (macro-F1) ===")
    for s in summaries:
        print(f"  {s['label']:<20} F1={s['macro_f1']:.4f} acc={s['accuracy']:.4f} "
              f"invalid={s['invalid_rate']:.1%} ${s['cost_per_1k_tickets_list']:.3f}/1k "
              f"(spend ${s['spend_openrouter_usd']:.3f})")
    if args.score:
        for s in summaries:
            (out_dir / f"{s['model_key']}.json").write_text(json.dumps(s, indent=2))
        summary = {
            "dataset": args.dataset, "split": args.split, "n": len(gold_rows),
            "reasoning_openrouter": args.reasoning, "prompt": "label-only (shared)",
            "models": summaries,
        }
        (out_dir / "panel_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\n-> {(out_dir / 'panel_summary.json').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
