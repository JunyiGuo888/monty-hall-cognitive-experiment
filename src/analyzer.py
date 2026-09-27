#!/usr/bin/env python3
"""
src/analyzer.py

Segment-wise scoring with multi-provider support.
Refactored from multi_provider_cot_analyzer.py for embedding into main.py.

Key differences from the standalone script:
- Provider detection and client construction are LAZY (inside _init_analyzer()).
  Importing this module no longer touches env or exits.
- All sys.exit(1) calls inside the analysis path are replaced with
  AnalysisError, so the caller decides what to do.
- Entry point is run_analysis(progress_file) -> dict of output paths.
- Uses ANALYZER_MAX_CONCURRENT / ANALYZER_API_TIMEOUT if set, otherwise
  falls back to MAX_CONCURRENT / API_TIMEOUT.

- OpenAI / DeepSeek / Ollama: OpenAI-compatible chat.completions
- Anthropic: official SDK

Provider auto-selection priority (when PROVIDER is not set in .env):
  1. openai
  2. deepseek
  3. anthropic
  4. ollama

Rules per dimension per character (continuous across rounds):
- Not mentioned and not yet seen -> score 0, source "not_mentioned"
- First mentioned -> use LLM score, source "actual", update last_score
- Mentioned again with different score -> jump, source "actual"
- Seen but not mentioned in this segment -> carry forward last_score,
  source "carry_forward"

No interpolation, no averaging.

Segmentation
------------
All characters and all rounds share a FIXED segment count (SEGMENTS_PER_ROUND,
default 10) so that segment i always means "the i-th N-th of this round" for
every character, making cross-character comparison meaningful.

Text is split by proportional position (piece i covers [i*L/N, (i+1)*L/N)).
If a piece exceeds MAX_SEGMENT_CHARS, only its leading portion is sent to the
model; the rest is dropped. This gives uniform sampling over the entire
original text at the cost of reading only a prefix of each piece.
"""

import json
import os
import re
import sys
from dotenv import load_dotenv

load_dotenv()

import asyncio
import csv  # noqa: F401  (kept for parity with original imports)
import pandas as pd
from datetime import datetime


# ================== Config ==================
MAX_CONCURRENT = int(
    os.getenv("ANALYZER_MAX_CONCURRENT", os.getenv("MAX_CONCURRENT", "1"))
)
API_TIMEOUT = float(
    os.getenv("ANALYZER_API_TIMEOUT", os.getenv("API_TIMEOUT", "600"))
)
SHOW_FULL_COT = os.getenv("SHOW_FULL_COT", "true").strip().lower() == "true"

TEMPERATURE = 0.0
# o-series models consume reasoning tokens, so keep a generous cap
MAX_OUTPUT_TOKENS = 8192
MAX_RETRIES = 5

# Fixed segment count, shared by ALL characters and ALL rounds.
SEGMENTS_PER_ROUND = int(os.getenv("SEGMENTS_PER_ROUND", "10"))

# Per-segment character ceiling. Pieces longer than this are truncated;
# only the leading MAX_SEGMENT_CHARS characters are sent to the model.
MAX_SEGMENT_CHARS = int(os.getenv("MAX_SEGMENT_CHARS", "6000"))

DIMENSIONS = [
    "monty_hall_recognition",
    "monty_hall_persistence",
    "paradigm_shift_awareness",
    "illusion_residue",
    "conformity",
    "willingness_violation",
]
DIM_LABELS = {
    "monty_hall_recognition": "MH Recognition",
    "monty_hall_persistence": "MH Persistence",
    "paradigm_shift_awareness": "Paradigm Shift",
    "illusion_residue": "Illusion Residue",
    "conformity": "Conformity",
    "willingness_violation": "Willingness Violation",
}
ROUNDS_ORDER = ["initial", "debate1", "debate2", "discovery", "debate3", "debate4"]


# ================== Errors ==================
class AnalysisError(RuntimeError):
    """Any fatal, non-retryable condition in the analyzer path."""


# ================== API key validation ==================
PLACEHOLDER_PATTERNS = (
    "yourkey", "your-key", "your_key", "yourkeyhere",
    "sk-your", "sk-xxx", "sk-xxxx",
    "xxx", "changeme", "change-me", "placeholder",
    "todo", "test-key", "replaceme", "replace-me",
)
MIN_KEY_LEN = 20


def _is_placeholder(value: str) -> bool:
    """Return True if the env value is empty, a placeholder, or obviously invalid."""
    v = (value or "").strip()
    if not v:
        return True
    if len(v) < MIN_KEY_LEN:
        return True
    low = v.lower()
    if any(p in low for p in PLACEHOLDER_PATTERNS):
        return True
    return False


def env_ok(key):
    """True only if the env var exists and is not a placeholder."""
    return not _is_placeholder(os.getenv(key, ""))


def ollama_enabled():
    return os.getenv("OLLAMA_ENABLED", "false").strip().lower() == "true"


AUTO_PRIORITY = [
    ("openai",    env_ok("OPENAI_API_KEY")),
    ("deepseek",  env_ok("DEEPSEEK_API_KEY")),
    ("anthropic", env_ok("ANTHROPIC_API_KEY")),
    ("ollama",    ollama_enabled()),
]


# ================== Lazy init state ==================
PROVIDER = None
MODEL_NAME = None
PROVIDER_KIND = None
ANTHROPIC_SDK_VERSION = None
ANTHROPIC_BASE_URL_SOURCE = None  # "env" | "sdk_default"
client = None
_analyzer_ready = False


def _init_analyzer():
    """Idempotent. Reads env, picks provider, builds client.

    Raises AnalysisError instead of sys.exit so the caller can decide
    whether to abort the whole process or just skip analysis.
    """
    global PROVIDER, MODEL_NAME, PROVIDER_KIND, client
    global ANTHROPIC_SDK_VERSION, ANTHROPIC_BASE_URL_SOURCE, _analyzer_ready

    if _analyzer_ready:
        return

    provider = os.getenv("PROVIDER", "").strip().lower()

    if not provider:
        for name, available in AUTO_PRIORITY:
            if available:
                provider = name
                break
        if not provider:
            raise AnalysisError(
                "No provider available for analyzer. Set OPENAI_API_KEY / "
                "DEEPSEEK_API_KEY / ANTHROPIC_API_KEY (not placeholders), "
                "or set OLLAMA_ENABLED=true."
            )
        print(f"[Analyzer] PROVIDER not set, auto-selected: {provider}")
    else:
        print(f"[Analyzer] PROVIDER from env: {provider}")
        if provider == "openai" and not env_ok("OPENAI_API_KEY"):
            raise AnalysisError(
                "PROVIDER=openai but OPENAI_API_KEY is missing or looks like a placeholder."
            )
        if provider == "deepseek" and not env_ok("DEEPSEEK_API_KEY"):
            raise AnalysisError(
                "PROVIDER=deepseek but DEEPSEEK_API_KEY is missing or looks like a placeholder."
            )
        if provider == "anthropic" and not env_ok("ANTHROPIC_API_KEY"):
            raise AnalysisError(
                "PROVIDER=anthropic but ANTHROPIC_API_KEY is missing or looks like a placeholder."
            )
        if provider == "ollama" and not ollama_enabled():
            raise AnalysisError("PROVIDER=ollama but OLLAMA_ENABLED is not true.")

    # ---------- Build client per provider ----------
    if provider == "openai":
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise AnalysisError(
                "openai package not installed. Run: pip install openai"
            ) from e
        MODEL_NAME = os.getenv("OPENAI_MODEL", "o4-mini")
        client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            timeout=API_TIMEOUT,
            max_retries=2,
        )
        PROVIDER_KIND = "openai_compatible"

    elif provider == "deepseek":
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise AnalysisError(
                "openai package not installed. Run: pip install openai"
            ) from e
        MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=API_TIMEOUT,
            max_retries=2,
        )
        PROVIDER_KIND = "openai_compatible"

    elif provider == "ollama":
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise AnalysisError(
                "openai package not installed. Run: pip install openai"
            ) from e
        MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        client = AsyncOpenAI(
            api_key="ollama",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            timeout=API_TIMEOUT,
            max_retries=2,
        )
        PROVIDER_KIND = "openai_compatible"

    elif provider == "anthropic":
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise AnalysisError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from e
        try:
            import anthropic as _anthropic_mod
            ANTHROPIC_SDK_VERSION = getattr(_anthropic_mod, "__version__", "unknown")
        except Exception:
            ANTHROPIC_SDK_VERSION = "unknown"

        MODEL_NAME = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")

        # Read ANTHROPIC_BASE_URL strictly from .env. If unset, do not pass
        # base_url at all so the SDK uses its own default endpoint. The SDK
        # appends /v1 internally, so a value here must NOT include /v1.
        _anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
        _anthropic_kwargs = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "timeout": API_TIMEOUT,
            "max_retries": 2,
        }
        if _anthropic_base_url:
            _anthropic_kwargs["base_url"] = _anthropic_base_url
            ANTHROPIC_BASE_URL_SOURCE = "env"
        else:
            ANTHROPIC_BASE_URL_SOURCE = "sdk_default"

        client = AsyncAnthropic(**_anthropic_kwargs)
        PROVIDER_KIND = "anthropic"

    else:
        raise AnalysisError(
            f"Unsupported PROVIDER={provider}. Use openai/deepseek/anthropic/ollama."
        )

    PROVIDER = provider
    _analyzer_ready = True

    print(f"[Analyzer] provider={PROVIDER}, model={MODEL_NAME}, "
          f"concurrency={MAX_CONCURRENT}")
    if PROVIDER == "anthropic":
        print(f"[Analyzer] Anthropic SDK version: {ANTHROPIC_SDK_VERSION}")
        if ANTHROPIC_BASE_URL_SOURCE == "env":
            print(f"[Analyzer] Anthropic base_url: "
                  f"{os.getenv('ANTHROPIC_BASE_URL').strip()} (from .env)")
        else:
            print("[Analyzer] Anthropic base_url: <SDK default>")


# ==================== SYSTEM PROMPT ====================
SYSTEM_PROMPT = """You are a cognitive science analyst. Your task is to rate each text segment on six psychological dimensions using a 1-10 scale.
Base your rating strictly on the semantic content of THIS segment only, not on word frequency or explicit mentions.

For EACH dimension, output three fields:
- "score": integer 1-10
- "reason": string <=200 chars
- "mentioned": boolean. true iff this segment actually discusses, references, or engages with this dimension's concept. false if the segment does not touch on it at all.

IMPORTANT: If "mentioned" is false, still output a plausible score (usually 1), but the caller will ignore it and apply carry-forward / not-mentioned logic.

Definitions and anchors:

1. monty_hall_recognition - Genuine understanding of the Monty Hall causal structure (initial 1/3, switch advantage).
   - 1: No mention of Monty Hall, switching, or probabilities.
   - 3: Mentions "Monty Hall" or "switch" as a label, but zero explanation of 1/3->2/3.
   - 5: Acknowledges 1/3 initial, but does not derive 2/3 switching advantage.
   - 7: Explicitly derives 2/3 advantage, but lacks full causal reasoning.
   - 10: Complete causal derivation with clear step-by-step logic.

2. monty_hall_persistence - Active, deliberate use of Monty Hall reasoning as a primary tool.
   - 1: Explicitly rejects or abandons Monty Hall reasoning.
   - 3: Mentions Monty Hall but does not actively use it.
   - 5: Uses Monty Hall logic in parts, with deliberate shifts.
   - 7: Consistently applies Monty Hall framework with minor deviations.
   - 10: Unwaveringly uses Monty Hall logic throughout.

3. paradigm_shift_awareness - Realisation that Monty Hall logic no longer applies (e.g., due to another vehicle, symmetry, game-theoretic constraints).
   - 1: No awareness that Monty Hall might be inapplicable.
   - 3: Vague sense that something is different.
   - 5: Mentions situation may have changed, but continues using Monty Hall.
   - 7: Explicitly states Monty Hall may not apply, but still partially uses it.
   - 10: Clear statement that Monty Hall no longer applies and fully abandons it.
   Set "mentioned" = true only if the segment engages with the question of whether Monty Hall still applies.

4. illusion_residue - Old Monty Hall thinking unconsciously interferes with current judgment.
   - 1: Fully aligned with new reasoning; no old bias.
   - 3: Faint trace of old intuition, easily overridden.
   - 5: Monty Hall intuition pops up but is consciously corrected.
   - 7: Monty Hall intuition persistently distorts judgment.
   - 10: Strongly dominated by Monty Hall intuition.

5. conformity - Judgment influenced by external opinions rather than independent reasoning.
   - 1: Entirely independent reasoning.
   - 3: Acknowledges external opinions but does not follow them.
   - 5: Shows some consideration of external views, mainly own reasoning.
   - 7: Adopts external view with visible internal resistance.
   - 10: Fully surrenders to external pressure.

6. willingness_violation - Suppresses own genuine judgment to align with external pressures.
   - 1: No conflict between inner preference and expressed decision.
   - 3: Mild unease but follows through with own preference.
   - 5: Expresses doubt but ultimately sticks with own view.
   - 7: Clearly prefers one option but chooses another.
   - 10: Strong inner preference explicitly abandoned.

Output ONLY a single JSON object with the six keys. Each key maps to an object with "score", "reason", and "mentioned".
Do not include any other text or markdown."""

USER_PROMPT_TEMPLATE = """Segment {seg_idx} of round "{round_name}" for character {character}.
Text segment:
{text}

Rate this segment on the six dimensions (1-10).
For each dimension, decide whether this segment actually discusses/engages with that dimension ("mentioned": true) or not ("mentioned": false).
Output JSON only."""


# ================== Error classification ==================
AUTH_TAGS = (
    "invalid_api_key",
    "incorrect api key",
    "authenticationerror",
    "authentication_error",
    "permission_denied",
    "permission denied",
    "invalid authentication",
)


def is_auth_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    if "authentication" in name or "permissiondenied" in name:
        return True
    s = str(exc).lower()
    if any(t in s for t in AUTH_TAGS):
        return True
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return True
    return False


def is_rate_error(exc: Exception) -> bool:
    s = str(exc).lower()
    if "rate limit" in s or "too many" in s or "429" in s:
        return True
    status = getattr(exc, "status_code", None)
    return status == 429


def is_not_found_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status == 404:
        return True
    s = str(exc).lower()
    if "not_found_error" in s or "404" in s:
        return True
    return False


# ================== JSON extraction ==================
def extract_json(text: str) -> dict:
    if not text:
        raise ValueError("Empty response")
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


# ================== Proportional segmentation ==================
_SENT_END = ".!?。！？"


def _snap_boundary(text, target, lo, hi, search_window=200):
    """Snap `target` to the nearest sentence terminator within [lo, hi]."""
    n = len(text)
    lo = max(1, int(lo))
    hi = min(n - 1, int(hi))
    target = max(lo, min(hi, int(target)))
    if lo > hi:
        return target
    for delta in range(0, search_window + 1):
        for pos in (target + delta, target - delta):
            if lo <= pos <= hi and text[pos - 1] in _SENT_END:
                return int(pos)
    return target


def split_into_segments_uniform(text, num_segments, search_window=200):
    """
    Split text into EXACTLY num_segments pieces by proportional position.
    Piece i covers the interval [i*L/N, (i+1)*L/N) in character space.
    Interior boundaries are snapped to sentence terminators within the
    adjacent windows so the total piece count stays exactly num_segments.

    The returned pieces may be longer than MAX_SEGMENT_CHARS; downstream
    truncation happens in score_segment().
    """
    num_segments = int(num_segments)
    if num_segments < 1:
        num_segments = 1

    if not text:
        return [""] * num_segments
    text = text.strip()
    n = len(text)
    if n == 0:
        return [""] * num_segments

    boundaries = [0]
    for i in range(1, num_segments):
        boundaries.append(i * n // num_segments)
    boundaries.append(n)

    # Snap interior boundaries, keeping them strictly non-decreasing.
    for i in range(1, num_segments):
        lo = boundaries[i - 1] + 1
        hi = boundaries[i + 1] - 1
        if lo > hi:
            continue
        boundaries[i] = _snap_boundary(text, boundaries[i], lo, hi, search_window)

    segments = []
    for i in range(num_segments):
        segments.append(text[boundaries[i]:boundaries[i + 1]].strip())
    return segments


def normalize_result(result):
    if not isinstance(result, dict):
        result = {}
    for dim in DIMENSIONS:
        if dim not in result:
            result[dim] = {'score': 1, 'reason': 'Missing key', 'mentioned': False}
        elif not isinstance(result[dim], dict):
            try:
                result[dim] = {'score': int(result[dim]), 'reason': 'Flat score',
                               'mentioned': False}
            except Exception:
                result[dim] = {'score': 1, 'reason': 'Invalid value',
                               'mentioned': False}
        else:
            try:
                result[dim]['score'] = int(result[dim].get('score', 1))
            except Exception:
                result[dim]['score'] = 1
            result[dim]['score'] = max(1, min(10, result[dim]['score']))
            if not result[dim].get('reason'):
                result[dim]['reason'] = 'No reason given'
            if not isinstance(result[dim].get('mentioned'), bool):
                r = str(result[dim].get('reason', '')).lower()
                result[dim]['mentioned'] = not (
                    'no mention' in r or 'not mentioned' in r or r == ''
                )
    return result


# ================== LLM call ==================
# Anthropic SDK compatibility:
# - SDK < 1.0 accepts `temperature` as an explicit keyword.
# - SDK >= 1.0 removed the explicit parameter; pass it via `extra_body`.
_ANTHROPIC_TEMP_MODE = None  # "explicit" | "extra_body" | None (undecided)


async def _anthropic_create(system_prompt: str, user_prompt: str):
    global _ANTHROPIC_TEMP_MODE

    base_kwargs = {
        "model": MODEL_NAME,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": MAX_OUTPUT_TOKENS,
    }

    if _ANTHROPIC_TEMP_MODE == "extra_body":
        return await client.messages.create(
            extra_body={"temperature": TEMPERATURE}, **base_kwargs
        )
    if _ANTHROPIC_TEMP_MODE == "explicit":
        return await client.messages.create(
            temperature=TEMPERATURE, **base_kwargs
        )

    try:
        resp = await client.messages.create(temperature=TEMPERATURE, **base_kwargs)
        _ANTHROPIC_TEMP_MODE = "explicit"
        return resp
    except TypeError as e:
        msg = str(e).lower()
        if "temperature" not in msg or "unexpected keyword" not in msg:
            raise
        resp = await client.messages.create(
            extra_body={"temperature": TEMPERATURE}, **base_kwargs
        )
        _ANTHROPIC_TEMP_MODE = "extra_body"
        return resp


async def call_llm(system_prompt: str, user_prompt: str) -> str:
    if PROVIDER_KIND == "openai_compatible":
        kwargs = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        is_openai_reasoning = (
            PROVIDER == "openai"
            and any(tag in MODEL_NAME for tag in ("o1", "o3", "o4"))
        )

        if is_openai_reasoning:
            kwargs["max_completion_tokens"] = MAX_OUTPUT_TOKENS
        else:
            kwargs["temperature"] = TEMPERATURE
            kwargs["max_tokens"] = MAX_OUTPUT_TOKENS
            if PROVIDER != "ollama":
                kwargs["response_format"] = {"type": "json_object"}

        resp = await client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    elif PROVIDER_KIND == "anthropic":
        resp = await _anthropic_create(system_prompt, user_prompt)
        parts = []
        for block in resp.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts)

    else:
        raise RuntimeError("Unknown provider kind")


# ================== Preflight ==================
async def preflight_check():
    """Issue a minimal request to surface key / endpoint problems early.

    Raises AnalysisError on failure (no sys.exit).
    """
    print("[Analyzer] Preflight: testing provider connectivity...")
    try:
        content = await asyncio.wait_for(
            call_llm(
                "You are a test responder.",
                'Reply with exactly this JSON and nothing else: {"ok": true}',
            ),
            timeout=60,
        )
        print(f"[Analyzer] Preflight OK. Sample: {str(content)[:80]!r}")
        if PROVIDER == "anthropic":
            print(f"[Analyzer] Anthropic temperature mode: {_ANTHROPIC_TEMP_MODE}")
    except asyncio.TimeoutError:
        raise AnalysisError(
            "Preflight timed out. Check network / base URL / firewall."
        )
    except Exception as e:
        if is_auth_error(e):
            raise AnalysisError(
                f"Authentication failed for provider '{PROVIDER}': {e}. "
                "Check your API key in .env (not a placeholder) or switch PROVIDER."
            ) from e
        elif is_not_found_error(e):
            hint = ""
            if PROVIDER == "anthropic":
                hint = (
                    f" Common causes: (1) ANTHROPIC_MODEL={MODEL_NAME!r} not a "
                    f"valid model id (correct form: claude-sonnet-4-5, no "
                    f"provider prefix, no date); (2) ANTHROPIC_BASE_URL must "
                    f"NOT include a trailing /v1 — the SDK appends it. "
                    f"base_url_source={ANTHROPIC_BASE_URL_SOURCE}"
                )
            raise AnalysisError(
                f"404 Not found for provider '{PROVIDER}': {e}.{hint}"
            ) from e
        else:
            raise AnalysisError(
                f"Preflight failed for provider '{PROVIDER}': {e}"
            ) from e


# ================== Score one segment ==================
async def score_segment(char, round_name, seg_idx, seg_text):
    if not seg_text or len(seg_text) < 10:
        return {dim: {'score': 1, 'reason': 'Empty segment', 'mentioned': False}
                for dim in DIMENSIONS}

    # Truncate to the leading MAX_SEGMENT_CHARS characters.
    if len(seg_text) > MAX_SEGMENT_CHARS:
        seg_text = seg_text[:MAX_SEGMENT_CHARS] + "\n...[truncated]"

    user_prompt = USER_PROMPT_TEMPLATE.format(
        seg_idx=seg_idx + 1, round_name=round_name, character=char, text=seg_text
    )

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            content = await call_llm(SYSTEM_PROMPT, user_prompt)
            result = extract_json(content)
            result = normalize_result(result)
            result['_raw_response'] = content
            return result
        except Exception as e:
            last_error = e

            if is_auth_error(e):
                raise AnalysisError(
                    f"{char}/{round_name}/seg{seg_idx + 1}: {e}"
                ) from e

            if is_not_found_error(e):
                raise AnalysisError(
                    f"{char}/{round_name}/seg{seg_idx + 1} (404): {e}"
                ) from e

            is_rate = is_rate_error(e)
            wait = min(60, 2 ** attempt * 2) if is_rate else 1.5
            print(f"      Retry {attempt + 1}/{MAX_RETRIES} "
                  f"for {char}/{round_name}/seg{seg_idx + 1}: "
                  f"{type(e).__name__}: {e}")
            await asyncio.sleep(wait)

    print(f"    Segment {char}/{round_name}/{seg_idx + 1} failed: {last_error}")
    return {dim: {'score': 1, 'reason': 'API failed', 'mentioned': False}
            for dim in DIMENSIONS}


# ================== Score one round with state machine ==================
async def score_round_with_state(char, round_name, cot_text, dim_state, num_segments):
    if not cot_text or len(cot_text) < 50:
        segments = [""] * num_segments
    else:
        segments = split_into_segments_uniform(cot_text, num_segments)

    applied_segments = []
    for idx, seg_text in enumerate(segments):
        print(f"      Scoring {char}/{round_name} segment "
              f"{idx + 1}/{num_segments} (len={len(seg_text)})")
        raw = await score_segment(char, round_name, idx, seg_text)
        raw_response = raw.get('_raw_response', '')

        applied = {}
        for dim in DIMENSIONS:
            item = raw.get(dim, {})
            mentioned = bool(item.get('mentioned', False))
            raw_score = int(item.get('score', 1))
            reason = item.get('reason', '')

            state = dim_state[dim]

            if not state["seen"] and not mentioned:
                applied[dim] = {
                    "score": 0,
                    "reason": reason or "Not mentioned",
                    "source": "not_mentioned",
                }
            elif mentioned:
                state["seen"] = True
                state["last_score"] = raw_score
                applied[dim] = {
                    "score": raw_score,
                    "reason": reason,
                    "source": "actual",
                }
            else:
                applied[dim] = {
                    "score": state["last_score"],
                    "reason": reason or (
                        f"Carried forward from previous ({state['last_score']})"
                    ),
                    "source": "carry_forward",
                }

        applied['_raw_response'] = raw_response
        applied['_segment_text'] = seg_text
        applied_segments.append(applied)

    return applied_segments


# ================== Analyze one character ==================
async def analyze_character(char, cot_dict, num_segments):
    result = {}
    dim_state = {dim: {"seen": False, "last_score": 0} for dim in DIMENSIONS}

    for rnd in ROUNDS_ORDER:
        cot_text = cot_dict.get(rnd, {}).get('cot', '')
        print(f"  {char} - {rnd} (length {len(cot_text)}, segments {num_segments})")
        result[rnd] = await score_round_with_state(
            char, rnd, cot_text, dim_state, num_segments
        )
    return result


# ================== Report writers ==================
def write_progress_json(path, all_results, car1_members, car2_members,
                        num_segments=None):
    payload = {
        "updated_at": datetime.now().isoformat(),
        "provider": PROVIDER,
        "model": MODEL_NAME,
        "num_segments": num_segments,
        "max_segment_chars": MAX_SEGMENT_CHARS,
        "car1_members": sorted(car1_members),
        "car2_members": sorted(car2_members),
        "results": all_results,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_dataframe(all_results, car1_members, car2_members):
    rows = []
    for char, rounds in all_results.items():
        if char in car1_members:
            car = "car1"
        elif char in car2_members:
            car = "car2"
        else:
            car = "unknown"
        for rnd in ROUNDS_ORDER:
            segments = rounds.get(rnd, [])
            for seg_idx, seg_scores in enumerate(segments):
                row = {
                    'character': char,
                    'car': car,
                    'round': rnd,
                    'segment': seg_idx,
                }
                if SHOW_FULL_COT:
                    row['raw_response'] = seg_scores.get('_raw_response', '')
                    row['segment_text'] = seg_scores.get('_segment_text', '')
                for dim in DIMENSIONS:
                    d = seg_scores.get(dim, {})
                    row[f'{dim}_score'] = d.get('score', 0)
                    row[f'{dim}_source'] = d.get('source', 'not_mentioned')
                    row[f'{dim}_reason'] = d.get('reason', '')
                rows.append(row)
    return pd.DataFrame(rows)


def write_csv_incremental(path, df):
    df.to_csv(path, index=False, encoding='utf-8-sig')


def write_html_incremental(path, df, num_segments):
    import plotly.graph_objects as go
    label_map = {'initial': 'Init', 'debate1': 'D1', 'debate2': 'D2',
                 'discovery': 'Disc', 'debate3': 'D3', 'debate4': 'D4'}

    x_labels = []
    for rnd in ROUNDS_ORDER:
        for i in range(1, num_segments + 1):
            x_labels.append(f"{label_map[rnd]}{i}")

    chars = list(df['character'].unique()) if not df.empty else []
    html = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Trajectories</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body { font-family: 'Segoe UI', sans-serif; background: #f5f7fa; margin: 20px; }
.container { max-width: 1400px; margin: auto; background: white; padding: 20px;
             border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
h1 { color: #2c3e50; }
.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
</style></head><body>
<div class="container">
<h1>Cognitive Trajectories</h1>
<p>Provider: %s | Model: %s | Segments per round: %d</p>
<div class="chart-grid">
""" % (PROVIDER, MODEL_NAME, num_segments)

    for char in chars:
        char_df = df[df['character'] == char]
        if char_df.empty:
            continue
        fig = go.Figure()
        for dim in DIMENSIONS:
            ys = []
            for rnd in ROUNDS_ORDER:
                rnd_df = char_df[char_df['round'] == rnd]
                for seg in range(num_segments):
                    seg_df = rnd_df[rnd_df['segment'] == seg]
                    if seg_df.empty:
                        ys.append(None)
                    else:
                        val = seg_df.iloc[0][f'{dim}_score']
                        ys.append(None if pd.isna(val) else int(val))
            fig.add_trace(go.Scatter(
                x=x_labels, y=ys, mode='lines+markers',
                name=DIM_LABELS[dim], connectgaps=False
            ))
        fig.update_layout(
            title=f"{char}",
            xaxis_title="Time (segment)", yaxis_title="Score (1-10)",
            yaxis=dict(range=[0, 11]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            height=450, margin=dict(l=40, r=20, t=50, b=60)
        )
        html += f'<div>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>\n'
    html += "</div></div></body></html>"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)


# ================== Utilities ==================
def extract_timestamp(filename):
    m = re.search(r'run_(\d{8})_(\d{6})', filename)
    if m:
        return m.group(1) + m.group(2)
    return filename


def print_coverage_preview(characters, num_segments, max_chars):
    """Show, per character per round, total chars and the effective read budget."""
    print("[Analyzer] Coverage preview "
          "(total chars -> chars actually sent to the model):")
    for char in sorted(characters.keys()):
        cot_data = characters[char]
        if not isinstance(cot_data, dict):
            continue
        parts = []
        for rnd in ROUNDS_ORDER:
            text = (cot_data.get(rnd) or {}).get('cot', '') or ''
            total = len(text)
            avg_piece = total // num_segments if num_segments else 0
            per_piece_read = min(avg_piece, max_chars)
            total_read = per_piece_read * num_segments
            pct = (total_read / total * 100.0) if total > 0 else 0.0
            parts.append(f"{rnd}: {total}->{total_read} ({pct:.0f}%)")
        print(f"  {char}: " + " | ".join(parts))


# ================== Embeddable entry point ==================
async def run_analysis(progress_file: str) -> dict:
    """Run analyzer on a desert_progress_*.json file.

    Lazily initializes the provider on first call. Raises AnalysisError on
    any non-retryable failure. Returns dict of output paths:
        {"json": ..., "csv": ..., "html": ...}
    """
    _init_analyzer()

    with open(progress_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    characters = data['characters']
    final_decision = data.get('final_decision', {})
    car1_members = set(final_decision.get('car1_individual', {}).keys())
    car2_members = set(final_decision.get('car2_individual', {}).keys())
    if not car1_members or not car2_members:
        print("[Analyzer] Warning: cannot determine car membership "
              "from final_decision.")

    base = progress_file.replace('.json', '')
    suffix = f"_{PROVIDER}"
    progress_json = base + f'{suffix}_progress_results.json'
    csv_path = base + f'{suffix}_segmented_scores.csv'
    html_path = base + f'{suffix}_trajectory_60points.html'

    all_results = {}
    completed_chars = set()
    if os.path.exists(progress_json):
        try:
            with open(progress_json, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            all_results = saved.get('results', {})
            completed_chars = set(all_results.keys())
            print(f"[Analyzer] Resuming. Already completed: "
                  f"{sorted(completed_chars)}")
        except Exception as e:
            print(f"[Analyzer] Could not load existing progress: {e}")

    # Fixed segment count shared by every character and every round.
    num_segments = int(SEGMENTS_PER_ROUND)
    print(f"[Analyzer] Fixed segment count (all characters, all rounds): "
          f"{num_segments}")
    print(f"[Analyzer] MAX_SEGMENT_CHARS = {MAX_SEGMENT_CHARS} "
          f"(per-piece prefix kept, rest dropped)")
    print_coverage_preview(characters, num_segments, MAX_SEGMENT_CHARS)

    await preflight_check()

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    lock = asyncio.Lock()

    async def process(char, cot_data):
        if char in completed_chars:
            print(f"[Analyzer] Skipping {char} (already done)")
            return
        async with sem:
            print(f"\n[Analyzer] Analyzing {char}...")
            try:
                res = await analyze_character(char, cot_data, num_segments)
            except AnalysisError:
                raise
            except Exception as e:
                print(f"  {char} failed: {type(e).__name__}: {e}")
                return

            async with lock:
                all_results[char] = res
                try:
                    write_progress_json(progress_json, all_results,
                                        car1_members, car2_members, num_segments)
                    df = build_dataframe(all_results, car1_members, car2_members)
                    write_csv_incremental(csv_path, df)
                    write_html_incremental(html_path, df, num_segments)
                    print(f"  {char} done. Progress saved.")
                except Exception as e:
                    print(f"  {char} done but failed to write outputs: {e}")

    tasks = [asyncio.create_task(process(char, cot_data))
             for char, cot_data in characters.items()]

    try:
        await asyncio.gather(*tasks)
    except AnalysisError as e:
        # Cancel remaining tasks and flush whatever we have.
        for t in tasks:
            t.cancel()
        try:
            write_progress_json(progress_json, all_results,
                                car1_members, car2_members, num_segments)
            df = build_dataframe(all_results, car1_members, car2_members)
            write_csv_incremental(csv_path, df)
            write_html_incremental(html_path, df, num_segments)
        except Exception:
            pass
        raise AnalysisError(
            f"Fatal error during analysis: {e}. Partial results written."
        ) from e

    write_progress_json(progress_json, all_results,
                        car1_members, car2_members, num_segments)
    df = build_dataframe(all_results, car1_members, car2_members)
    write_csv_incremental(csv_path, df)
    write_html_incremental(html_path, df, num_segments)
    print("[Analyzer] All done.")
    print(f"[Analyzer] Progress JSON: {progress_json}")
    print(f"[Analyzer] CSV: {csv_path}")
    print(f"[Analyzer] HTML: {html_path}")

    return {"json": progress_json, "csv": csv_path, "html": html_path}


# ================== CLI ==================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        prog_file = sys.argv[1]
    else:
        files = [f for f in os.listdir('.')
                 if f.startswith('desert_progress_') and f.endswith('.json')]
        if not files:
            print("No desert_progress_*.json found.")
            sys.exit(1)
        prog_file = max(files, key=extract_timestamp)
        print(f"[Analyzer] Auto-selected: {prog_file}")

    try:
        asyncio.run(run_analysis(prog_file))
    except AnalysisError as e:
        print(f"\n[Analyzer] FATAL: {e}")
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n[Analyzer] Interrupted.")
        sys.exit(130)