"""Module 3: SG Content-Commerce Gap Dashboard.

DistilBERT sentiment over mock SG TikTok Shop reviews x per-capita hashtag
supply vs ID/TH -> a ranked category opportunity score.
"""

import altair as alt
import pandas as pd
import streamlit as st

from modules.gap_dashboard import category_opportunity, score_review_sentiment
from modules.shared import business_callout, load_hashtags, load_reviews

st.set_page_config(page_title="SG Gap Dashboard", page_icon="📊", layout="wide")
st.title("📊 SG Content-Commerce Gap Dashboard")
st.markdown(
    "Where is SG demand **not** being met by content? Demand comes from DistilBERT "
    "sentiment over SG shop reviews; supply comes from per-capita shoppable-video "
    "volume vs the ID/TH benchmark. Their product is the **opportunity score**."
)

reviews = load_reviews()
hashtags = load_hashtags()

with st.spinner("Scoring review sentiment with DistilBERT on CPU (first run downloads ~270 MB)…"):
    scored = score_review_sentiment(reviews)
opp = category_opportunity(scored, hashtags)

# ------------------------------------------------------------- headline KPIs
top_cat = opp.iloc[0]
k1, k2, k3 = st.columns(3)
k1.metric("Biggest opportunity", top_cat["category"], f"score {top_cat['opportunity_score']}")
k2.metric("Its positive sentiment", f"{top_cat['positive_share']:.0%}", "demand is healthy")
k3.metric("Its content supply gap vs ID/TH", f"{top_cat['supply_gap']:.0%}", "supply is missing", delta_color="inverse")

# --------------------------------------------------------------- main chart
st.subheader("Category opportunity ranking")
bar = (
    alt.Chart(opp)
    .mark_bar()
    .encode(
        x=alt.X("opportunity_score", title="Opportunity score (demand × supply gap)"),
        y=alt.Y("category", sort="-x", title=None),
        color=alt.Color("supply_gap", scale=alt.Scale(scheme="reds"), legend=alt.Legend(title="Supply gap", format=".0%")),
        tooltip=["category", "opportunity_score",
                 alt.Tooltip("positive_share", format=".0%"),
                 alt.Tooltip("supply_gap", format=".0%"), "reviews"],
    )
    .properties(height=300)
)
st.altair_chart(bar, use_container_width=True)

# ------------------------------------------------------- supporting visuals
c1, c2 = st.columns(2)
with c1:
    st.subheader("Per-capita content supply (videos/mo per M pop)")
    supply = opp.melt(
        id_vars="category",
        value_vars=["sg_per_capita", "id_per_capita", "th_per_capita"],
        var_name="market", value_name="per_capita",
    )
    supply["market"] = supply["market"].str[:2].str.upper()
    grouped = (
        alt.Chart(supply)
        .mark_bar()
        .encode(
            x=alt.X("market", title=None),
            y=alt.Y("per_capita", title="videos / mo / M pop"),
            color="market",
            column=alt.Column("category", header=alt.Header(labelAngle=-30, labelAlign="right"), title=None),
            tooltip=["category", "market", alt.Tooltip("per_capita", format=",.0f")],
        )
        .properties(height=200, width=70)
    )
    st.altair_chart(grouped)
with c2:
    st.subheader("Demand health by category (DistilBERT)")
    sent = (
        alt.Chart(opp)
        .mark_bar()
        .encode(
            x=alt.X("positive_share", title="Positive review share", axis=alt.Axis(format=".0%")),
            y=alt.Y("category", sort="-x", title=None),
            tooltip=["category", alt.Tooltip("positive_share", format=".0%"), "reviews"],
        )
        .properties(height=300)
    )
    st.altair_chart(sent, use_container_width=True)

with st.expander("Scored reviews (DistilBERT output)"):
    st.dataframe(
        scored[["category", "review_text", "stars", "sentiment", "sentiment_score"]],
        use_container_width=True, hide_index=True,
    )

st.caption(
    "opportunity = normalised(demand) × supply_gap, where demand = review volume × "
    "positive share, and supply_gap = 1 − SG per-capita videos ÷ mean(ID, TH per-capita). "
    "Mock data stands in for blocked TikTok endpoints; the pipeline is scrape-ready."
)

business_callout(
    "why this moves SG GMV",
    f"This dashboard turns 'SG underperforms' into a **prioritised action list**. "
    f"Today it says **{top_cat['category']}** has healthy demand "
    f"({top_cat['positive_share']:.0%} positive sentiment) but a "
    f"{top_cat['supply_gap']:.0%} content supply gap vs ID/TH — meaning purchase intent "
    f"exists with almost no shoppable video to capture it. Pointing Modules 1–2 (seller "
    f"hook generation + creator matching) at the top-ranked categories concentrates "
    f"scarce SG creator supply exactly where each new video unlocks the most incremental "
    f"GMV, instead of spreading incentives evenly across categories that are already saturated.",
)
