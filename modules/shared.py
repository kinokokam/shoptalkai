"""Shared helpers: data loading, Ollama detection, business-framing callouts."""

from pathlib import Path

import pandas as pd
import requests
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3:8b"  # 4-bit quantised by default in Ollama


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


def ollama_available(timeout: float = 1.5) -> bool:
    """True if a local Ollama server is reachable."""
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout).ok
    except requests.RequestException:
        return False


def ollama_generate(prompt: str, system: str = "", temperature: float = 0.8,
                    timeout: int = 300) -> str:
    """Single-shot generation against the local quantised Llama 3 8B."""
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
