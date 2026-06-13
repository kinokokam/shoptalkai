"""Module 1 core logic: Product → Content Scorer.

Pipeline: product image --BLIP-2--> caption --Llama 3.2 (Ollama)--> 5 hook
candidates --rubric--> scored & ranked hooks.

Everything runs locally. Each model stage degrades gracefully:
- no image            -> skip captioning, use the text description alone
- Ollama unreachable  -> silent template fallback (generate_hooks reports the
                         source so the UI can label output, never warn)
"""

import json
import re

import requests
import streamlit as st

from modules.shared import OLLAMA_MODEL, OLLAMA_URL

# BLIP-2 (Salesforce/blip2-opt-2.7b) is the spec'd captioner; it runs on CPU but
# weighs ~15 GB on disk. blip-base (~1 GB) is offered as a fast-demo alternative.
CAPTION_MODELS = {
    "BLIP-2 OPT-2.7B (full pipeline, ~15 GB download)": "Salesforce/blip2-opt-2.7b",
    "BLIP base (fast demo, ~1 GB download)": "Salesforce/blip-image-captioning-base",
}


@st.cache_resource(show_spinner=False)
def _load_captioner(model_id: str):
    """Load a BLIP/BLIP-2 captioner directly (CPU).

    We load the model classes by hand rather than via pipeline("image-to-text"):
    transformers 5.x dropped that pipeline task. use_fast=False keeps the slow
    PIL-based image processor so torchvision isn't a required dependency.
    """
    import torch
    from transformers import AutoProcessor

    if "blip2" in model_id:
        from transformers import Blip2ForConditionalGeneration as CaptionModel
    else:
        from transformers import BlipForConditionalGeneration as CaptionModel

    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    model = CaptionModel.from_pretrained(model_id, torch_dtype=torch.float32).to("cpu")
    return processor, model


def caption_image(image, model_id: str) -> str:
    """Caption a PIL image with the selected BLIP variant on CPU."""
    import torch

    processor, model = _load_captioner(model_id)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40)
    return processor.decode(out[0], skip_special_tokens=True).strip()


# ----------------------------------------------------------------- generation

_SYSTEM_PROMPT = (
    "You are a top-performing TikTok Shop Singapore content strategist. You write "
    "scroll-stopping first-3-second video hooks that make Singaporean viewers stop "
    "and watch. You know local context: HDB living, MRT commutes, humid weather, "
    "hawker culture, kiasu deal-hunting. Keep hooks under 15 words. Never use "
    "hashtags inside the hook."
)


def _llama_generate(prompt: str, system: str) -> str:
    """Stream a completion from the local quantised Llama model.

    Streaming matters on CPU: with stream=false no bytes arrive until the full
    generation finishes, so one slow run trips the HTTP read timeout. Streamed
    chunks reset the read clock token by token. keep_alive holds the model in
    memory so back-to-back demo runs skip the reload cost.
    """
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "system": system,
            "stream": True,
            "keep_alive": "30m",
            "options": {"temperature": 0.8, "num_predict": 256},
        },
        stream=True,
        timeout=(5, 180),  # (connect, per-chunk read) — first chunk includes model load
    )
    resp.raise_for_status()
    parts = []
    for line in resp.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        parts.append(chunk.get("response", ""))
        if chunk.get("done"):
            break
    return "".join(parts).strip()


def generate_hooks_llm(product_name: str, description: str, caption: str | None,
                       n: int = 5) -> list[str]:
    """Ask the local quantised Llama model for n hook candidates."""
    visual = f'\nWhat the product looks like (image caption): "{caption}"' if caption else ""
    prompt = (
        f'Product: "{product_name}"\n'
        f'Description: "{description}"{visual}\n\n'
        f"Write exactly {n} different TikTok video hooks for selling this product to "
        f"Singapore viewers. Vary the angle: curiosity, problem-agitation, social proof, "
        f"price anchor, local-lifestyle. Return them as a numbered list, one hook per "
        f"line, nothing else."
    )
    raw = _llama_generate(prompt, system=_SYSTEM_PROMPT)
    hooks = [re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip().strip('"')
             for line in raw.splitlines() if re.match(r"^\s*\d+[\.\)]", line)]
    return hooks[:n] if hooks else [raw.strip()[:120]]


def generate_hooks(product_name: str, description: str, caption: str | None,
                   n: int = 5) -> tuple[list[str], str]:
    """Llama 3 by default; silent template fallback only if Ollama is unreachable.

    Returns (hooks, source) where source is "llama3" or "template" so the UI
    can label the output without ever surfacing a warning state.
    """
    try:
        return generate_hooks_llm(product_name, description, caption, n), "llama3"
    except (requests.RequestException, json.JSONDecodeError, KeyError):
        return generate_hooks_fallback(product_name, description, n), "template"


def generate_hooks_fallback(product_name: str, description: str, n: int = 5) -> list[str]:
    """Deterministic template hooks used when Ollama is offline."""
    short = product_name.split(",")[0]
    templates = [
        f"I was today years old when I found out {short} exists",
        f"POV: you finally fixed this with a {short} — why did no one tell me",
        f"3 reasons every HDB home needs the {short}",
        f"Stop scrolling if you've ever wasted money on this problem — {short} review",
        f"This {short} sold out twice on TikTok Shop SG. Here's why",
        f"Don't buy the {short} until you watch this",
    ]
    return templates[:n]


# -------------------------------------------------------------------- scoring

# Transparent rubric: each dimension is observable in the hook text itself, so
# the score is explainable to a seller (unlike a black-box LLM judge).
_CURIOSITY = ["why", "secret", "nobody", "no one", "until", "found out", "truth",
              "wish i knew", "pov", "here's why", "what happens"]
_URGENCY = ["stop scrolling", "sold out", "don't buy", "before", "last", "now",
            "running out", "won't last", "today only"]
_SG_LOCAL = ["sg", "singapore", "hdb", "bto", "mrt", "hawker", "aircon", "humid",
             "shiok", "kiasu", "ez-link", "tampines", "jurong", "punggol"]
_CTA_PATTERN = re.compile(r"watch|here's|review|reasons|how to|let me show", re.I)
_NUMBER_PATTERN = re.compile(r"\d")


def score_hook(hook: str) -> dict:
    """Score one hook 0-100 across 5 explainable dimensions."""
    h = hook.lower()
    words = len(hook.split())

    curiosity = 20 if any(k in h for k in _CURIOSITY) else 5
    urgency = 20 if any(k in h for k in _URGENCY) else 5
    specificity = 20 if _NUMBER_PATTERN.search(hook) else 8
    sg_relevance = 20 if any(k in h for k in _SG_LOCAL) else 6
    # Sweet spot for a spoken 3-second hook is ~6-14 words.
    brevity = 20 if 6 <= words <= 14 else (12 if words <= 18 else 4)
    cta_bonus = 5 if _CTA_PATTERN.search(hook) else 0

    total = min(100, curiosity + urgency + specificity + sg_relevance + brevity + cta_bonus)
    return {
        "hook": hook,
        "curiosity": curiosity,
        "urgency": urgency,
        "specificity": specificity,
        "sg_relevance": sg_relevance,
        "brevity": brevity,
        "total": total,
    }


def score_hooks(hooks: list[str]) -> list[dict]:
    return sorted((score_hook(h) for h in hooks), key=lambda d: d["total"], reverse=True)
