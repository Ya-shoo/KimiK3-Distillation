"""Kimi K3 teacher client — any OpenAI-compatible host (Together / OpenRouter / Moonshot).

Plumbing only: auth, JSON decoding + validation, retry, token/cost accounting.
The teacher PROMPT (the load-bearing design) lives in `prompts/teacher.md` and is
passed in verbatim — this module is prompt-agnostic so the prompt can be iterated
freely without touching code. Only `category` is validated (against the frozen
27-label space); the rationale field names are whatever the prompt defines.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from triage_distill.schema import load_label_space

REPO_ROOT = Path(__file__).resolve().parents[3]
TEACHER_MD = REPO_ROOT / "prompts" / "teacher.md"
PRICES_YAML = REPO_ROOT / "configs" / "prices.yaml"
MODELS_YAML = REPO_ROOT / "configs" / "models.yaml"

load_dotenv(REPO_ROOT / ".env")


def extract_system_prompt(md_path: Path = TEACHER_MD) -> str:
    """Pull the fenced block under the '## SYSTEM PROMPT' heading of teacher.md."""
    text = md_path.read_text()
    m = re.search(r"##\s*SYSTEM PROMPT.*?```(?:[a-zA-Z]+)?\n(.*?)```", text, re.S)
    if not m:
        raise ValueError(f"No fenced SYSTEM PROMPT block found in {md_path}")
    return m.group(1).strip()


def _teacher_model() -> str:
    env = os.environ.get("TEACHER_MODEL")
    if env:
        return env
    cfg = yaml.safe_load(MODELS_YAML.read_text())
    return cfg["teacher"]["kimi-k3"]["model"]


def _base_url() -> str:
    return os.environ.get("TEACHER_BASE_URL", "https://openrouter.ai/api/v1")


def _max_tokens() -> int:
    # Cap the completion budget. K3 is a reasoning model (thinking billed as output),
    # but a one-label classification + 2-sentence rationale needs nowhere near the
    # model's 65_536 ceiling. This matters on OpenRouter, which RESERVES credit for
    # the full max_tokens up front — leaving it unset reserves ~$0.98/call and blocks
    # low-balance accounts outright. Tune down once the smoke test shows real usage.
    return int(os.environ.get("TEACHER_MAX_TOKENS", "4096"))


def _client(base: str) -> OpenAI:
    key = os.environ.get("TEACHER_API_KEY")
    if not key:
        raise RuntimeError("TEACHER_API_KEY not set. Copy .env.example -> .env and add your key.")
    return OpenAI(api_key=key, base_url=base)


def _price_per_token() -> tuple[float, float]:
    p = yaml.safe_load(PRICES_YAML.read_text())["models"].get("kimi-k3", {})
    return ((p.get("input") or 0) / 1e6, (p.get("output") or 0) / 1e6)


class Teacher:
    """One K3 call per ticket → validated JSON + usage/cost."""

    def __init__(self, model: str | None = None, temperature: float = 0.0, max_tokens: int | None = None,
                 prompt_path: Path | None = None, label_space_path: Path | None = None):
        self.base = _base_url()
        self.client = _client(self.base)
        self.model = model or _teacher_model()
        self.temperature = temperature
        self.max_tokens = max_tokens if max_tokens is not None else _max_tokens()
        # prompt_path + label_space_path let one client serve any benchmark (see datasets.cfg);
        # defaults keep the original Bitext behavior.
        self.system_prompt = extract_system_prompt(prompt_path or TEACHER_MD)
        self.labels = set(load_label_space(label_space_path))
        self._in_price, self._out_price = _price_per_token()
        # On OpenRouter, only route to providers that honor our params (response_format).
        self._extra_body = {"provider": {"require_parameters": True}} if "openrouter" in self.base else {}

    def label(self, ticket: str, max_retries: int = 2) -> dict:
        """Return {'result','category','usage','cost_usd','raw'}; retries on invalid JSON/label."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f'Ticket: "{ticket}"'},
        ]
        last_err: Exception | None = None
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
            # K3 is a reasoning model; OpenRouter surfaces its hidden thinking here.
            # We already pay for these tokens, so we archive the trace (design decision).
            reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
            u = resp.usage
            details = getattr(u, "completion_tokens_details", None)
            usage = {
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "reasoning_tokens": getattr(details, "reasoning_tokens", None),
            }
            cost = u.prompt_tokens * self._in_price + u.completion_tokens * self._out_price
            try:
                data = json.loads(raw)
                cat = data.get("category")
                if cat not in self.labels:
                    raise ValueError(f"category '{cat}' not in the 27 labels")
                return {"result": data, "category": cat, "reasoning": reasoning, "usage": usage, "cost_usd": cost, "raw": raw}
            except (json.JSONDecodeError, ValueError) as e:
                last_err = e
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"That was invalid ({e}). Return ONLY the JSON object with a valid `category` from the list."},
                ]
        raise RuntimeError(f"Teacher failed after {max_retries + 1} tries: {last_err}")
