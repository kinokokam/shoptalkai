"""shoptalkai — AI-powered content-commerce gap analyser for TikTok Shop SG.

Entrypoint / landing page. Run with:  streamlit run app.py
"""

import streamlit as st

from modules.shared import OLLAMA_MODEL, load_hashtags, ollama_status

st.set_page_config(
    page_title="shoptalkai — TikTok Shop SG Gap Analyser",
    page_icon="🛍️",
    layout="wide",
)

st.title("🛍️ shoptalkai")
st.subheader("AI-powered content-commerce gap analyser for TikTok Shop Singapore")

st.markdown(
    """
**The thesis:** TikTok Shop's flywheel is *content → discovery → purchase*. In Indonesia
and Thailand that flywheel spins because creators flood the platform with shoppable
videos. In Singapore the flywheel is starved at the first step — **demand exists, but
the content supply that converts demand into GMV is thin.** Sellers list products no
creator talks about; creators don't know which products fit their niche; nobody can
see where the gaps are.

**shoptalkai attacks all three failure points with local, CPU-only AI:**
"""
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 1️⃣ Product → Content Scorer")
    st.markdown(
        "Turn any product into ranked TikTok video hooks. "
        "**BLIP-2** captions the product image, **Llama 3.2 3B** (quantised, via Ollama) "
        "writes the hooks, a transparent rubric scores them."
    )
    st.page_link("pages/1_Product_to_Content_Scorer.py", label="Open module →", icon="🎬")
with col2:
    st.markdown("### 2️⃣ Product–Creator Fit Matcher")
    st.markdown(
        "Rank the top-5 products for any creator niche using **MiniLM sentence "
        "embeddings** (semantic fit) blended with estimated commission upside."
    )
    st.page_link("pages/2_Product_Creator_Fit_Matcher.py", label="Open module →", icon="🤝")
with col3:
    st.markdown("### 3️⃣ SG Gap Dashboard")
    st.markdown(
        "Quantify the content-commerce gap per category: **DistilBERT** sentiment on SG "
        "review data × hashtag supply vs ID/TH → a category opportunity score."
    )
    st.page_link("pages/3_SG_Gap_Dashboard.py", label="Open module →", icon="📊")

st.divider()

# Quick health/status strip so the demo environment is legible at a glance.
hashtags = load_hashtags()
sg_per_capita = hashtags["sg_monthly_videos"].sum() / hashtags["sg_population_m"].iloc[0]
id_per_capita = hashtags["id_monthly_videos"].sum() / hashtags["id_population_m"].iloc[0]
th_per_capita = hashtags["th_monthly_videos"].sum() / hashtags["th_population_m"].iloc[0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("SG shoppable videos / mo / M pop", f"{sg_per_capita:,.0f}")
m2.metric("ID equivalent", f"{id_per_capita:,.0f}", delta=f"{id_per_capita / sg_per_capita:.1f}× SG")
m3.metric("TH equivalent", f"{th_per_capita:,.0f}", delta=f"{th_per_capita / sg_per_capita:.1f}× SG")
_ollama = ollama_status()
if _ollama["model_ready"]:
    _llm_state = "🟢 ready"
elif _ollama["server"]:
    _llm_state = "🟡 pulling model"
else:
    _llm_state = "🔴 offline"
m4.metric(f"Ollama ({OLLAMA_MODEL})", _llm_state)

st.caption(
    "All models run locally on CPU. No paid APIs. Datasets are mock (live scraping of "
    "TikTok endpoints is blocked) but shaped to real SEA market dynamics."
)
