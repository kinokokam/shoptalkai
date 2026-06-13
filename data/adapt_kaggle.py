"""Adapt the real Kaggle 'E-Commerce & Retail Supply Chain' dataset into the
schema shoptalkai's Modules 1 & 2 expect, plus a real channel-penetration table.

Source (downloaded via the Kaggle MCP into data/raw/):
    rajhkumarr/e-commerce-and-retail-supply-chain
      - products.csv  : 80 real SKUs (name, category, brand, price, rating)
      - sales.csv     : 160k transactions 2016-2022, with a `channel` column
                        that includes TikTokShop alongside Amazon/Zalora/retail

What's real vs derived (surfaced honestly in the UI):
    name, category, price_sgd, rating       -> taken straight from products.csv
    monthly_units_sold                       -> REAL: avg monthly TikTokShop units
                                                from sales.csv (fallback: all-channel)
    description                              -> DERIVED: templated from brand /
                                                product_type / sub_category / size
    commission_rate                          -> DERIVED: category-typical creator
                                                affiliate rate + deterministic jitter
                                                (TikTok Shop affiliate rates are
                                                seller-set; these are placeholders)

Run from the repo root:  python data/adapt_kaggle.py
"""

import hashlib
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
TIKTOK_CHANNEL = "TikTokShop"

# Category-typical creator affiliate commission rates (stated assumption, not in
# the source data). Loosely reflects that beauty/personal-care carry richer
# affiliate margins than thin-margin electronics.
COMMISSION_BASE = {
    "Beauty": 0.18,
    "Personal Care": 0.16,
    "Home Care": 0.12,
    "Home & Living": 0.12,
    "Food & Beverage": 0.10,
    "Sports & Lifestyle": 0.14,
    "Electronics Accessories": 0.06,
}
DEFAULT_COMMISSION = 0.12


def _commission_rate(sku_id: str, category: str) -> float:
    """Deterministic category base + ±0.03 jitter keyed on the SKU id."""
    base = COMMISSION_BASE.get(category, DEFAULT_COMMISSION)
    h = int(hashlib.md5(sku_id.encode()).hexdigest(), 16)
    jitter = (h % 7 - 3) / 100.0  # -0.03 .. +0.03
    return round(min(0.30, max(0.04, base + jitter)), 2)


def _description(row: pd.Series) -> str:
    brand = str(row.get("brand", "")).strip()
    ptype = str(row.get("product_type", "")).strip()
    size = str(row.get("size_label", "")).strip()
    sub = str(row.get("sub_category", "")).strip()
    cat = str(row.get("category", "")).strip()
    size_part = f", {size}" if size and size.lower() != "nan" else ""
    lead = f"{brand} {ptype}".strip()
    return f"{lead}{size_part} — a {sub} item in the {cat} category."


def _real_monthly_tiktok_units(sales: pd.DataFrame) -> pd.Series:
    """Avg monthly units per SKU on the real TikTok Shop channel.

    Falls back to all-channel monthly units for SKUs with no TikTok sales, so no
    listed product shows an artificial zero.
    """
    def monthly(df: pd.DataFrame) -> pd.Series:
        g = df.groupby("sku_id").agg(units=("quantity", "sum"),
                                     months=("month", "nunique"))
        return (g["units"] / g["months"].clip(lower=1)).round().astype(int)

    tiktok = monthly(sales[sales["channel"] == TIKTOK_CHANNEL])
    overall = monthly(sales)
    return tiktok.reindex(overall.index).fillna(overall).astype(int)


def build_products(products_raw: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    units = _real_monthly_tiktok_units(sales)
    out = pd.DataFrame({
        "product_id": products_raw["sku_id"],
        "name": products_raw["product_name"],
        "category": products_raw["category"],
        "description": products_raw.apply(_description, axis=1),
        "price_sgd": products_raw["default_price"].round(2),
        "commission_rate": [
            _commission_rate(s, c)
            for s, c in zip(products_raw["sku_id"], products_raw["category"])
        ],
        "monthly_units_sold": products_raw["sku_id"].map(units).fillna(0).astype(int),
        "rating": products_raw["avg_rating"].round(1),
    })
    return out


def build_channel_revenue(products_raw: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """Tidy category × channel net revenue + units from real transactions."""
    cats = products_raw[["sku_id", "category"]]
    merged = sales.merge(cats, on="sku_id", how="left")
    grouped = (
        merged.groupby(["category", "channel"])
        .agg(net_revenue=("net_revenue", "sum"), units=("quantity", "sum"))
        .reset_index()
    )
    grouped["net_revenue"] = grouped["net_revenue"].round(2)
    grouped["units"] = grouped["units"].astype(int)
    return grouped.sort_values(["category", "net_revenue"], ascending=[True, False])


def main() -> None:
    products_raw = pd.read_csv(RAW_DIR / "products.csv")
    sales = pd.read_csv(RAW_DIR / "sales.csv")

    products = build_products(products_raw, sales)
    products.to_csv(DATA_DIR / "products.csv", index=False)
    print(f"wrote {DATA_DIR / 'products.csv'} ({len(products)} real SKUs)")

    channel = build_channel_revenue(products_raw, sales)
    channel.to_csv(DATA_DIR / "channel_revenue.csv", index=False)
    print(f"wrote {DATA_DIR / 'channel_revenue.csv'} ({len(channel)} category×channel rows)")


if __name__ == "__main__":
    main()
