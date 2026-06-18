from __future__ import annotations
import json
import os
import re
import time
from itertools import cycle
from typing import Iterator

from google import genai
from google.genai import types
from dotenv import load_dotenv

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

load_dotenv()

# ── Key rotation ─────────────────────────────────────────────────────────────
_raw_keys = os.getenv("GEMINI_API_KEYS", "")
_KEYS: list[str] = [k.strip() for k in _raw_keys.split(",") if k.strip()]
if not _KEYS:
    raise EnvironmentError("No GEMINI_API_KEYS found in .env")

_key_cycle: Iterator[str] = cycle(_KEYS)
_MODEL_NAME: str = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash-lite")

# Fallback model chain if primary is unavailable
_FALLBACK_MODELS: list[str] = [
    _MODEL_NAME,
    "models/gemini-2.0-flash-lite-001",
    "models/gemini-2.5-flash",
    "models/gemini-flash-lite-latest",
]

# Pre-build clients for each key (one per key for clean rotation)
_clients: list[genai.Client] = [genai.Client(api_key=k) for k in _KEYS]
_client_cycle: Iterator[genai.Client] = cycle(_clients)
_model_cycle: Iterator[str] = cycle(_FALLBACK_MODELS)


def _call_llm(system_prompt: str, user_message: str, retries: int = 10) -> tuple[str, int]:
    """Call Gemini API with key+model rotation and smart backoff."""
    last_err = None
    wait_secs = 10.0
    for attempt in range(retries):
        client = next(_client_cycle)
        # Rotate through fallback models every 2 failures
        model = _FALLBACK_MODELS[min(attempt // 2, len(_FALLBACK_MODELS) - 1)]
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt.strip(),
                    temperature=0.2,
                    max_output_tokens=512,
                ),
            )
            text = response.text.strip() if response.text else ""
            try:
                tokens = response.usage_metadata.total_token_count or 0
            except Exception:
                tokens = len(user_message.split()) + len(text.split())
            return text, tokens
        except Exception as e:
            last_err = e
            err_str = str(e)
            # Extract retry_delay from 429 response
            retry_match = re.search(r"seconds:\s*(\d+)", err_str)
            if retry_match:
                wait_secs = float(retry_match.group(1)) + 2
            elif "503" in err_str or "UNAVAILABLE" in err_str:
                wait_secs = min(10.0 * (1.5 ** attempt), 90)
            else:
                wait_secs = min(wait_secs * 1.5, 90)
            print(f"  [warn] attempt {attempt+1}/{retries} model={model} failed: {err_str[:100]} — wait {wait_secs:.0f}s", flush=True)
            time.sleep(wait_secs)
    raise RuntimeError(f"LLM call failed after {retries} retries: {last_err}")


# ── Actor ─────────────────────────────────────────────────────────────────────
def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> str:
    context_text = "\n".join(
        f"[{chunk.title}]: {chunk.text}" for chunk in example.context
    )
    reflection_block = ""
    if reflection_memory:
        reflection_block = "\n\nPrevious reflection notes:\n" + "\n".join(
            f"- {note}" for note in reflection_memory
        )

    user_msg = (
        f"Question: {example.question}\n\n"
        f"Context:\n{context_text}"
        f"{reflection_block}"
    )
    text, _ = _call_llm(ACTOR_SYSTEM, user_msg)

    # Parse "Answer: <answer>" line; fall back to last non-empty line
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.lower().startswith("answer:"):
            return line[len("answer:"):].strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text


# ── Evaluator ─────────────────────────────────────────────────────────────────
def evaluator(example: QAExample, answer: str) -> JudgeResult:
    norm_gold = normalize_answer(example.gold_answer)
    norm_pred = normalize_answer(answer)

    # Fast-path 1: exact match after normalization
    if norm_gold == norm_pred:
        return JudgeResult(score=1, reason="Exact match after normalization.",
                           missing_evidence=[], spurious_claims=[])

    # Fast-path 2: gold is contained in prediction (e.g. "classical" in "classical music")
    if norm_gold and norm_gold in norm_pred:
        return JudgeResult(score=1, reason="Gold answer found within predicted answer.",
                           missing_evidence=[], spurious_claims=[])

    # Fast-path 3: prediction is contained in gold (e.g. "thames" in "river thames")
    if norm_pred and norm_pred in norm_gold:
        return JudgeResult(score=1, reason="Predicted answer is a substring of gold answer.",
                           missing_evidence=[], spurious_claims=[])

    # Fast-path 4: high word overlap (Jaccard >= 0.6)
    gold_words = set(norm_gold.split())
    pred_words = set(norm_pred.split())
    if gold_words and pred_words:
        jaccard = len(gold_words & pred_words) / len(gold_words | pred_words)
        if jaccard >= 0.6:
            return JudgeResult(score=1, reason=f"High word overlap (Jaccard={jaccard:.2f}).",
                               missing_evidence=[], spurious_claims=[])

    # LLM evaluator for ambiguous cases
    user_msg = (
        f"Question: {example.question}\n"
        f"Gold answer: {example.gold_answer}\n"
        f"Predicted answer: {answer}"
    )
    text, _ = _call_llm(EVALUATOR_SYSTEM, user_msg)

    raw = _extract_json(text)
    try:
        data = json.loads(raw)
        return JudgeResult(
            score=int(data.get("score", 0)),
            reason=str(data.get("reason", text)),
            missing_evidence=data.get("missing_evidence", []),
            spurious_claims=data.get("spurious_claims", []),
        )
    except (json.JSONDecodeError, KeyError):
        score = 1 if norm_gold in normalize_answer(text) else 0
        return JudgeResult(score=score, reason=text[:300], missing_evidence=[], spurious_claims=[])


# ── Reflector ─────────────────────────────────────────────────────────────────
def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    user_msg = (
        f"Question: {example.question}\n"
        f"Attempt number: {attempt_id}\n"
        f"Evaluator feedback: {judge.reason}\n"
        f"Missing evidence: {judge.missing_evidence}\n"
        f"Spurious claims: {judge.spurious_claims}"
    )
    text, _ = _call_llm(REFLECTOR_SYSTEM, user_msg)

    raw = _extract_json(text)
    try:
        data = json.loads(raw)
        return ReflectionEntry(
            attempt_id=int(data.get("attempt_id", attempt_id)),
            failure_reason=str(data.get("failure_reason", judge.reason)),
            lesson=str(data.get("lesson", "Need to complete all reasoning hops.")),
            next_strategy=str(data.get("next_strategy", "Re-read context and trace each hop explicitly.")),
        )
    except (json.JSONDecodeError, KeyError):
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Need to complete all reasoning hops before giving the final answer.",
            next_strategy="Re-read each context passage and explicitly trace every reasoning hop.",
        )


# ── Helpers ───────────────────────────────────────────────────────────────────
def _extract_json(text: str) -> str:
    """Extract first JSON object from a string (handles markdown code blocks)."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text
