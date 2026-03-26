# =====================================================
# Search25GPT_llm_upgrade.py
# LLM search planner for Search25GPT
# =====================================================

import json
import os
import re
import time
import hashlib
from typing import Dict, List, Optional

import streamlit as st
from openai import AzureOpenAI

from Search25GPTv2 import guard_ai_call, mark_ai_call_success

# -------------------------
# CONFIGURATION
# -------------------------
MAX_CALLS_PER_SESSION = 12
MIN_SECONDS_BETWEEN_CALLS = 15
MAX_PROMPT_CHARS = 800


# -------------------------
# SESSION STATE INIT
# -------------------------
if "api_call_count" not in st.session_state:
    st.session_state.api_call_count = 0

if "last_api_call_ts" not in st.session_state:
    st.session_state.last_api_call_ts = 0.0

if "seen_prompt_hashes" not in st.session_state:
    st.session_state.seen_prompt_hashes = set()


# -------------------------
# USER MESSAGE HANDLER
# -------------------------
def show_limit_message(reason, wait_seconds=None):
    if reason == "too_fast":
        if wait_seconds is not None and wait_seconds > 0:
            st.warning(
                f"Please wait {wait_seconds} seconds before running another AI-assisted search."
            )
        else:
            st.warning(
                "Please wait a few seconds before running another AI-assisted search."
            )
        st.info("You can still use Exact Word Search and filters while you wait.")

    elif reason == "session_limit":
        st.warning("Youve reached the AI-assisted search limit for this session.")
        st.info(
            "You can continue using Exact Word Search and filters, or come back later and try again."
        )

    elif reason == "prompt_too_long":
        st.warning("Your search is too long.")
        st.info(
            "Please shorten it to one main question or a few keywords (about 1 to 2 sentences)."
        )

    elif reason == "duplicate":
        st.warning("That search was already run recently.")
        st.info("Please revise the wording or wait a moment before trying again.")

    else:
        st.warning("AI-assisted search is temporarily unavailable.")
        st.info(
            "Please try again shortly, or use Exact Word Search in the meantime."
        )


# -------------------------
# GUARDRAIL FUNCTIONS
# -------------------------
def can_call_api(prompt: str) -> bool:
    now = time.time()
    prompt = prompt or ""

    if st.session_state.api_call_count >= MAX_CALLS_PER_SESSION:
        show_limit_message("session_limit")
        return False

    elapsed = now - st.session_state.last_api_call_ts
    if elapsed < MIN_SECONDS_BETWEEN_CALLS:
        remaining = int(MIN_SECONDS_BETWEEN_CALLS - elapsed)
        if remaining < 1:
            remaining = 1
        show_limit_message("too_fast", wait_seconds=remaining)
        return False

    if len(prompt) > MAX_PROMPT_CHARS:
        show_limit_message("prompt_too_long")
        return False

    h = hashlib.sha256(prompt.strip().lower().encode()).hexdigest()
    if h in st.session_state.seen_prompt_hashes:
        show_limit_message("duplicate")
        return False

    return True


def guard_ai_call(prompt: str):
    if not can_call_api(prompt):
        st.stop()


def mark_ai_call_success(prompt: str):
    h = hashlib.sha256(prompt.strip().lower().encode()).hexdigest()
    st.session_state.api_call_count += 1
    st.session_state.last_api_call_ts = time.time()
    st.session_state.seen_prompt_hashes.add(h)


def _get_azure_client() -> AzureOpenAI:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "").strip()
    api_key = (
        os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
        or os.environ.get("API_KEY", "").strip()
    )
    shortcode = os.environ.get("SHORTCODE", "").strip()

    if not endpoint or not api_version or not api_key:
        raise RuntimeError("Missing Azure OpenAI configuration")

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        api_key=api_key,
        organization=shortcode,
    )


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _clean_str_list(xs, maxn: int = 12, allow_phrases: bool = True) -> List[str]:
    out: List[str] = []
    if not isinstance(xs, list):
        return out

    for x in xs:
        if not isinstance(x, str):
            continue
        s = _norm(x)
        if not s:
            continue
        if not allow_phrases and " " in s:
            continue
        out.append(s)

    seen = set()
    deduped = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        deduped.append(x)

    return deduped[:maxn]


def llm_build_search_plan(user_query: str) -> Dict[str, object]:
    q = (user_query or "").strip()
    if not q:
        return {}

    try:
        guard_ai_call(q)

        client = _get_azure_client()
        chat_dep = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "").strip()
        if not chat_dep:
            return {}

        system = """
You are building a search plan for a survey codebook search engine.

Return ONLY valid JSON with these keys:
- scale
- entity
- role
- timeframe
- include_terms
- include_phrases
- exclude_terms
- subject_hints

Allowed values:
- scale: RISK_4, DISAPPROVAL, AVAILABILITY_4, FRIENDS_USE_5, EDU_7, INITIATION, or null
- role: MOTHER, FATHER, PARENT, or null
- timeframe: PAST_YEAR, PAST_30D, LIFETIME, or null

Rules:
- entity must be a list of likely topic/substance terms.
- include_terms must be short useful search terms.
- include_phrases must be short useful phrases likely to match survey wording.
- exclude_terms should only contain terms likely to create false positives.
- subject_hints should be short conceptual hints like school, peers, parents, discipline, progression.
- Return JSON only. No explanation.
"""

        user = f"""
User query:
{q}

Return JSON only.
"""

        resp = client.chat.completions.create(
            model=chat_dep,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=350,
        )

        mark_ai_call_success(q)

        txt = (resp.choices[0].message.content or "").strip()
        txt = re.sub(r"^```json\s*|\s*```$", "", txt, flags=re.I)

        plan = json.loads(txt)
        if not isinstance(plan, dict):
            return {}

        return {
            "scale": plan.get("scale"),
            "entity": _clean_str_list(
                plan.get("entity", []), maxn=10, allow_phrases=True
            ),
            "role": plan.get("role"),
            "timeframe": plan.get("timeframe"),
            "include_terms": _clean_str_list(
                plan.get("include_terms", []), maxn=12, allow_phrases=False
            ),
            "include_phrases": _clean_str_list(
                plan.get("include_phrases", []), maxn=12, allow_phrases=True
            ),
            "exclude_terms": _clean_str_list(
                plan.get("exclude_terms", []), maxn=12, allow_phrases=True
            ),
            "subject_hints": _clean_str_list(
                plan.get("subject_hints", []), maxn=8, allow_phrases=True
            ),
        }

    except Exception:
        return {}


def enhanced_parse(
    ai_query: str,
    entity_detector,
    role_parser,
    scale_parser,
    timeframe_parser,
) -> Dict[str, object]:
    plan = llm_build_search_plan(ai_query)

    role = plan.get("role")
    scale = plan.get("scale")
    timeframe = plan.get("timeframe")
    entity = plan.get("entity")

    if not entity:
        entity = entity_detector(ai_query)
    if not role:
        role = role_parser(ai_query)
    if not scale:
        scale = scale_parser(ai_query)
    if not timeframe:
        timeframe = timeframe_parser(ai_query)

    return {
        "role": role,
        "scale": scale,
        "timeframe": timeframe,
        "entity": entity or [],
        "include_terms": plan.get("include_terms", []),
        "include_phrases": plan.get("include_phrases", []),
        "exclude_terms": plan.get("exclude_terms", []),
        "subject_hints": plan.get("subject_hints", []),
    }
    
