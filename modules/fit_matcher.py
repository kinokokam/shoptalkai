"""Module 2 core logic: Product–Creator Fit Matcher.

Embeds the catalog and a creator's niche with all-MiniLM-L6-v2 (CPU), then
ranks products by a blend of semantic fit and commission upside:

    fit_score = 0.7 * cosine_similarity_norm + 0.3 * commission_potential_norm

Commission potential = price * commission_rate * expected attributable units,
i.e. what the creator can actually earn, not just the percentage.
"""

import numpy as np
import pandas as pd
import streamlit as st

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIM_WEIGHT = 0.7
COMMISSION_WEIGHT = 0.3
# Conservative assumption: a niche-fit video converts ~2% of avg views into
# attributable monthly orders. Mock-level constant, surfaced in the UI.
ATTRIBUTION_RATE = 0.02


@st.cache_resource(show_spinner=False)
def _load_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL, device="cpu")


@st.cache_data(show_spinner=False)
def _embed_catalog(texts: tuple[str, ...]) -> np.ndarray:
    return _load_embedder().encode(list(texts), normalize_embeddings=True)


def rank_products(products: pd.DataFrame, niche_text: str,
                  avg_video_views: int, top_k: int = 5) -> pd.DataFrame:
    """Return top_k products ranked by blended fit score for a creator niche."""
    catalog_texts = tuple(
        f"{r['name']}. {r['category']}. {r['description']}" for _, r in products.iterrows()
    )
    catalog_emb = _embed_catalog(catalog_texts)
    niche_emb = _load_embedder().encode([niche_text], normalize_embeddings=True)[0]

    df = products.copy()
    df["similarity"] = catalog_emb @ niche_emb

    # Earnings a creator could attribute to one decent video per product.
    expected_units = avg_video_views * ATTRIBUTION_RATE
    df["est_monthly_commission_sgd"] = (
        df["price_sgd"] * df["commission_rate"] * expected_units
    ).round(0)

    sim_norm = _minmax(df["similarity"])
    comm_norm = _minmax(df["est_monthly_commission_sgd"])
    df["fit_score"] = (SIM_WEIGHT * sim_norm + COMMISSION_WEIGHT * comm_norm).round(3)

    return df.sort_values("fit_score", ascending=False).head(top_k)


def _minmax(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng else pd.Series(0.5, index=s.index)
