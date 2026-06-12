# 🛍️ shoptalkai

> TikTok Shop may be #2 in SEA, but in Singapore, it's all talk and no shop.
> **shoptalkai** closes the content-commerce gap for SG sellers, creators, and analysts.

## The insight

TikTok Shop's growth engine is the **content → discovery → purchase** flywheel. In
Indonesia and Thailand it spins because creators flood the feed with shoppable video.
Singapore's flywheel is starved at the first step: **demand exists** (healthy review
sentiment, real purchase volume) **but per-capita shoppable-content supply trails ID/TH
by an order of magnitude**. Sellers list SKUs nobody talks about; creators can't see
which products are worth their audience; nobody quantifies where the gaps are.

shoptalkai is a full-stack diagnostic tool that attacks all three failure points —
running **entirely locally on CPU**, with no paid APIs.

## Modules

| # | Module | Models | What it answers |
|---|--------|--------|-----------------|
| 1 | 🎬 Product → Content Scorer | BLIP-2 (image caption) + quantised Llama 3 8B via Ollama + explainable rubric | *"I'm a seller — what video do I shoot for this product?"* |
| 2 | 🤝 Product–Creator Fit Matcher | `sentence-transformers/all-MiniLM-L6-v2` | *"I'm a creator — which 5 products should I promote?"* |
| 3 | 📊 SG Gap Dashboard | DistilBERT (SST-2 sentiment) + per-capita hashtag benchmarks | *"I'm an analyst — which categories leave the most GMV on the table?"* |

Every module ends with a **business framing callout** tying its output to SG GMV.

## Architecture

```mermaid
flowchart TB
    subgraph UI["Streamlit multi-page app (app.py)"]
        P1["pages/1 · Product → Content Scorer"]
        P2["pages/2 · Product–Creator Fit Matcher"]
        P3["pages/3 · SG Gap Dashboard"]
    end

    subgraph LOGIC["modules/ (pure-Python core logic)"]
        CS["content_scorer.py<br/>caption → hooks → rubric"]
        FM["fit_matcher.py<br/>embed → cosine → blend"]
        GD["gap_dashboard.py<br/>sentiment × supply gap"]
        SH["shared.py<br/>loaders · Ollama client · callouts"]
    end

    subgraph MODELS["Local CPU models (no GPU, no paid APIs)"]
        BLIP["BLIP-2 OPT-2.7B<br/>(or BLIP-base fast mode)"]
        LLAMA["Llama 3 8B, 4-bit quantised<br/>via Ollama :11434"]
        MINI["all-MiniLM-L6-v2"]
        DB["DistilBERT SST-2"]
    end

    subgraph DATA["data/ (mock — TikTok endpoints blocked)"]
        PR["products.csv"]
        CR["creators.csv"]
        RV["sg_reviews.csv"]
        HT["hashtag_volumes.csv<br/>SG vs ID vs TH"]
    end

    P1 --> CS;  P2 --> FM;  P3 --> GD
    CS & FM & GD --> SH
    CS -->|image caption| BLIP
    CS -->|hook generation| LLAMA
    CS -.->|Ollama offline| FB["template fallback"]
    FM --> MINI
    GD --> DB
    SH --> PR & CR & RV & HT
```

## Quickstart

```bash
# 1. Python deps (CPU-only wheels are fine)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Local LLM (optional but recommended — Module 1 falls back to templates without it)
brew install ollama          # or https://ollama.com/download
ollama serve &
ollama pull llama3:8b        # 4-bit quantised, ~4.7 GB

# 3. Run
streamlit run app.py
```

First runs download models from Hugging Face into the local cache:
MiniLM ~90 MB, DistilBERT ~270 MB, BLIP-base ~1 GB, BLIP-2 ~15 GB (optional —
Module 1's model picker defaults to BLIP-base so the demo stays fast on CPU).

Regenerate mock data any time with `python data/generate_mock_data.py` (seeded,
reproducible).

## How each module works

### 1. Product → Content Scorer
Product text (and optionally a photo, captioned by BLIP-2 on CPU) is prompted into a
locally-served quantised Llama 3 8B, which writes five hook candidates across distinct
persuasion angles. A **transparent rubric** — curiosity, urgency, specificity, SG
cultural relevance, spoken-hook brevity (each 0–20, +5 CTA bonus) — ranks them, so a
seller can see *why* a hook scores 85, not just that it does. If Ollama is offline the
page says so and serves deterministic SG-flavoured templates instead of failing.

### 2. Product–Creator Fit Matcher
The catalog and the creator's niche description are embedded with MiniLM;
`fit = 0.7 × cosine similarity + 0.3 × commission potential`, where commission
potential = price × commission rate × an assumed 2% of average views converting to
attributable monthly orders. Output: top-5 table plus a fit-vs-earnings scatter
("aim top-right").

### 3. SG Gap Dashboard
DistilBERT scores every SG review; `demand = review volume × positive share`.
`supply_gap = 1 − (SG per-capita videos ÷ mean(ID, TH per-capita))`.
`opportunity = normalised demand × supply gap`, ranked per category — high demand met
by thin content is exactly where incremental GMV is cheapest to unlock.

## Honest limitations

- **Datasets are mock** (live TikTok scraping is blocked) but seeded, reproducible, and
  shaped to real SEA dynamics; every pipeline is scrape-ready — swap the CSVs.
- The 2% view→order attribution rate and the 0.7/0.3 fit weights are stated assumptions,
  surfaced in the UI and tunable.
- BLIP-2 on CPU is slow (~15 GB weights); the UI offers BLIP-base as a fast demo mode.

## Project layout

```
app.py                      # entrypoint + landing page with market metrics
pages/                      # one Streamlit page per module
modules/                    # pure-Python core logic (testable without UI)
data/                       # mock datasets + seeded generator
requirements.txt
```
