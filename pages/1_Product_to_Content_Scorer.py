"""Module 1: Product → Content Scorer.

Give it a product (catalog pick or free text, optionally with a photo) and it
returns ranked TikTok hook candidates with an explainable score breakdown.
"""

import pandas as pd
import streamlit as st

from modules.content_scorer import (
    CAPTION_MODELS,
    caption_image,
    generate_hooks_fallback,
    generate_hooks_llm,
    score_hooks,
)
from modules.shared import (
    OLLAMA_MODEL,
    business_callout,
    fallback_banner,
    load_products,
    ollama_status,
)

st.set_page_config(page_title="Product → Content Scorer", page_icon="🎬", layout="wide")
st.title("🎬 Product → Content Scorer")
st.markdown(
    "Most SG sellers list products and wait. This module turns a **product listing into "
    "scored, ready-to-shoot TikTok hooks** — the missing first step of the content flywheel."
)

products = load_products()

# ------------------------------------------------------------------ inputs
left, right = st.columns([3, 2])

with left:
    source = st.radio("Product input", ["Pick from catalog", "Describe my own"], horizontal=True)
    if source == "Pick from catalog":
        pick = st.selectbox("Product", products["name"])
        row = products.loc[products["name"] == pick].iloc[0]
        product_name, description = row["name"], row["description"]
        st.caption(f"{row['category']} · S${row['price_sgd']} · {row['commission_rate']:.0%} commission")
    else:
        product_name = st.text_input("Product name", placeholder="e.g. Portable Neck Fan")
        description = st.text_area("Short description", placeholder="What it does, who it's for")

with right:
    image_file = st.file_uploader("Product photo (optional)", type=["png", "jpg", "jpeg", "webp"])
    caption_model_label = st.selectbox(
        "Image captioning model",
        list(CAPTION_MODELS.keys()),
        index=1,
        help="BLIP-2 is the full spec'd pipeline; BLIP base keeps the first demo run fast. Both are CPU-only.",
    )
    if image_file:
        st.image(image_file, width=220)

status = ollama_status()
llm_online = status["model_ready"]
if not status["server"]:
    fallback_banner(
        "Ollama is not running — hooks below come from deterministic SG templates, not Llama 3.",
        "`brew install ollama && ollama serve`, then `ollama pull llama3:8b` (4-bit quantised, ~4.7 GB).",
    )
elif not status["model_ready"]:
    others = f" (pulled so far: {', '.join(status['models'])})" if status["models"] else ""
    fallback_banner(
        f"Ollama is running but `{OLLAMA_MODEL}` isn't pulled yet{others} — using template hooks meanwhile.",
        f"`ollama pull {OLLAMA_MODEL}` (4-bit quantised, ~4.7 GB), then rerun this page.",
    )
else:
    st.caption(f"🟢 Llama 3 hooks via local Ollama (`{OLLAMA_MODEL}`)")

# ---------------------------------------------------------------- pipeline
if st.button("Generate & score hooks", type="primary", disabled=not (product_name and description)):
    caption = None
    if image_file:
        from PIL import Image

        with st.spinner("Captioning product image on CPU (first run downloads the model)…"):
            caption = caption_image(Image.open(image_file).convert("RGB"),
                                    CAPTION_MODELS[caption_model_label])
        st.caption(f"🖼️ Vision model sees: *{caption}*")

    if llm_online:
        with st.spinner("Llama 3 8B (quantised, local) is writing hooks…"):
            hooks = generate_hooks_llm(product_name, description, caption)
    else:
        hooks = generate_hooks_fallback(product_name, description)

    scored = score_hooks(hooks)
    best = scored[0]

    st.subheader("Ranked hooks")
    st.metric("Best hook score", f"{best['total']} / 100")
    st.markdown(f"> **{best['hook']}**")

    df = pd.DataFrame(scored)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "total": st.column_config.ProgressColumn("total", min_value=0, max_value=100, format="%d"),
        },
    )
    st.caption(
        "Scoring rubric (explainable, not a black box): curiosity, urgency, specificity "
        "(concrete numbers), SG cultural relevance, and spoken-hook brevity, each 0–20, "
        "+5 for a clear watch-on cue."
    )

business_callout(
    "why this moves SG GMV",
    "TikTok Shop converts when products have native video, and SG's bottleneck is that "
    "most listed SKUs have **zero** creator content. This module collapses the seller's "
    "cold-start cost from 'hire an agency' to 'pick the 85-point hook and shoot it on a "
    "phone'. Every seller it activates adds shoppable supply at the top of the "
    "content → discovery → purchase funnel — the exact stage where SG underperforms ID/TH.",
)
