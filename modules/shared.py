"""Shared helpers: data loading, Ollama detection, business-framing callouts."""

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OLLAMA_URL = "http://localhost:11434"
# Default to a 3B model: it fits ~8 GB machines without swapping and keeps
# llama.cpp + Metal acceleration. Override with the OLLAMA_MODEL env var.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")


@st.cache_data
def load_products() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "products.csv")


@st.cache_data
def load_creators() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "creators.csv")


@st.cache_data
def load_reviews() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sg_reviews.csv")


@st.cache_data
def load_hashtags() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "hashtag_volumes.csv")


@st.cache_data
def load_channel_revenue() -> pd.DataFrame:
    """Real category × channel net revenue, derived from the Kaggle sales.csv
    (rajhkumarr/e-commerce-and-retail-supply-chain) by data/adapt_kaggle.py."""
    return pd.read_csv(DATA_DIR / "channel_revenue.csv")


def ollama_status(timeout: float = 1.5) -> dict:
    """Three-state Ollama health check.

    Returns {server, model_ready, models}: the server can be up while the
    target model is still unpulled, and the two need different UI messages.
    """
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException:
        return {"server": False, "model_ready": False, "models": []}
    models = [m["name"] for m in resp.json().get("models", [])]
    # The configured tag and its fully-qualified variant both count, e.g.
    # "llama3.2:3b" and "llama3.2:3b-instruct-q4_K_M".
    ready = any(name == OLLAMA_MODEL or name.startswith(OLLAMA_MODEL + "-")
                for name in models)
    return {"server": True, "model_ready": ready, "models": models}


def ollama_available(timeout: float = 1.5) -> bool:
    """True only if the server is up AND the target model is pulled."""
    return ollama_status(timeout)["model_ready"]


def ollama_generate(prompt: str, system: str = "", temperature: float = 0.8,
                    timeout: int = 300) -> str:
    """Single-shot generation against the local quantised Llama model."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def business_callout(title: str, body: str) -> None:
    """Every module ends with one of these: the 'so what' for SG GMV."""
    st.divider()
    st.success(f"**💼 Business impact — {title}**\n\n{body}")


def fallback_banner(what: str, fix: str) -> None:
    st.warning(f"⚠️ **Fallback mode:** {what}\n\n*To enable the full pipeline:* {fix}")
