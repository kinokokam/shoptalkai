"""Module 2: Product–Creator Fit Matcher.

A creator describes their niche (or picks a mock SG creator profile) and gets
the top-5 catalog products ranked by semantic fit + commission upside.
"""

import altair as alt
import streamlit as st

from modules.fit_matcher import ATTRIBUTION_RATE, COMMISSION_WEIGHT, SIM_WEIGHT, rank_products
from modules.shared import business_callout, load_creators, load_products

st.set_page_config(page_title="Product–Creator Fit Matcher", page_icon="🤝", layout="wide")
st.title("🤝 Product–Creator Fit Matcher")
st.markdown(
    "SG creators don't make shop content partly because **they can't see which products "
    "are worth their audience**. This module matches niche → product with "
    "`all-MiniLM-L6-v2` sentence embeddings, blended with real earning potential."
)

products = load_products()
creators = load_creators()

# ------------------------------------------------------------------ inputs
mode = st.radio("Creator input", ["Pick a mock SG creator", "Describe a niche"], horizontal=True)
if mode == "Pick a mock SG creator":
    handle = st.selectbox("Creator", creators["handle"])
    c = creators.loc[creators["handle"] == handle].iloc[0]
    niche_text = c["bio"]
    avg_views = int(c["avg_video_views"])
    st.caption(f"{c['primary_niche']} · {c['followers']:,} followers · {avg_views:,} avg views")
    st.markdown(f"> *{niche_text}*")
else:
    niche_text = st.text_area(
        "Describe the creator's niche and audience",
        placeholder="e.g. budget home gym workouts for HDB dwellers, no-equipment routines",
    )
    avg_views = st.number_input("Average video views", 500, 2_000_000, 15_000, step=500)

# ----------------------------------------------------------------- ranking
if st.button("Find my top-5 products", type="primary", disabled=not niche_text):
    with st.spinner("Embedding catalog with MiniLM on CPU (first run downloads ~90 MB)…"):
        top = rank_products(products, niche_text, avg_views)

    st.subheader("Top-5 product matches")
    show = top[["name", "category", "price_sgd", "commission_rate",
                "similarity", "est_monthly_commission_sgd", "fit_score"]].rename(columns={
        "price_sgd": "price (S$)",
        "commission_rate": "comm. rate",
        "similarity": "semantic fit",
        "est_monthly_commission_sgd": "est. monthly commission (S$)",
        "fit_score": "fit score",
    })
    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "fit score": st.column_config.ProgressColumn("fit score", min_value=0.0, max_value=1.0, format="%.2f"),
            "comm. rate": st.column_config.NumberColumn(format="percent"),
        },
    )

    chart = (
        alt.Chart(top)
        .mark_circle(size=200)
        .encode(
            x=alt.X("similarity", title="Semantic fit (cosine)"),
            y=alt.Y("est_monthly_commission_sgd", title="Est. monthly commission (S$)"),
            color=alt.Color("category", legend=None),
            tooltip=["name", "similarity", "est_monthly_commission_sgd", "fit_score"],
        )
        .properties(height=320, title="Fit vs earnings — aim top-right")
    )
    st.altair_chart(chart, use_container_width=True)

    st.caption(
        f"fit score = {SIM_WEIGHT:.0%} × semantic similarity + {COMMISSION_WEIGHT:.0%} × "
        f"commission potential (price × rate × {ATTRIBUTION_RATE:.0%} of avg views as "
        f"attributable monthly orders). Weights are tunable per campaign objective."
    )

business_callout(
    "why this moves SG GMV",
    "ID/TH solved creator supply with volume; SG must solve it with **efficiency** — its "
    "creator pool is small, so every creator-product pairing has to count. By showing a "
    "creator the 5 SKUs where niche fit *and* commission are jointly highest, this module "
    "raises expected earnings per video, which is the single strongest lever to convert "
    "SG's lifestyle creators into shop-tagging affiliates. More affiliates → more "
    "shoppable videos per SKU → higher discovery-to-purchase conversion → GMV.",
)
