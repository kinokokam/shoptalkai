"""Module 3 core logic: SG Content-Commerce Gap Dashboard.

Two signals per category:
  demand  = DistilBERT positive-sentiment share x review volume (people want
            these products and mostly like them)
  supply gap = how far SG's per-capita shoppable-video volume trails the
            ID/TH per-capita average (content isn't being made)

  opportunity = demand_norm x supply_gap   -- high demand met by thin content
                                              is where GMV is being left on
                                              the table.
"""

import pandas as pd
import streamlit as st

SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


@st.cache_resource(show_spinner=False)
def _load_sentiment():
    from transformers import pipeline

    return pipeline("sentiment-analysis", model=SENTIMENT_MODEL, device="cpu")


@st.cache_data(show_spinner=False)
def score_review_sentiment(reviews: pd.DataFrame) -> pd.DataFrame:
    """Attach DistilBERT sentiment label/score to each review."""
    clf = _load_sentiment()
    results = clf(reviews["review_text"].tolist(), truncation=True, batch_size=16)
    out = reviews.copy()
    out["sentiment"] = [r["label"] for r in results]
    out["sentiment_score"] = [r["score"] for r in results]
    return out


def category_opportunity(scored_reviews: pd.DataFrame, hashtags: pd.DataFrame) -> pd.DataFrame:
    """Combine demand (sentiment x volume) and supply gap into one score."""
    demand = (
        scored_reviews.groupby("category")
        .agg(
            reviews=("review_id", "count"),
            positive_share=("sentiment", lambda s: (s == "POSITIVE").mean()),
        )
        .reset_index()
    )
    demand["demand_signal"] = demand["reviews"] * demand["positive_share"]

    df = hashtags.merge(demand, on="category")
    df["sg_per_capita"] = df["sg_monthly_videos"] / df["sg_population_m"]
    df["id_per_capita"] = df["id_monthly_videos"] / df["id_population_m"]
    df["th_per_capita"] = df["th_monthly_videos"] / df["th_population_m"]
    benchmark = (df["id_per_capita"] + df["th_per_capita"]) / 2
    # 0 = SG matches peers' per-capita content supply, 1 = SG produces none.
    df["supply_gap"] = (1 - df["sg_per_capita"] / benchmark).clip(0, 1)

    demand_norm = df["demand_signal"] / df["demand_signal"].max()
    df["opportunity_score"] = (100 * demand_norm * df["supply_gap"]).round(1)
    return df.sort_values("opportunity_score", ascending=False)
