"""Generate mock datasets for shoptalkai.

Live scraping of TikTok Shop / hashtag data is blocked, so these datasets are
synthetic but shaped to match the real SG market dynamics the tool diagnoses:
demand exists (reviews skew positive, sales are non-trivial) while per-capita
content supply lags ID/TH badly. Run from the repo root:

    python data/generate_mock_data.py
"""

import csv
import random
from pathlib import Path

random.seed(42)
DATA_DIR = Path(__file__).parent

CATEGORIES = [
    "Beauty & Skincare",
    "Home & Living",
    "Consumer Electronics",
    "Fashion & Accessories",
    "Food & Beverage",
    "Health & Wellness",
    "Baby & Kids",
]

# ---------------------------------------------------------------- products
PRODUCTS = [
    # name, category, description, price_sgd, commission_rate
    ("Glow Recipe-style Watermelon Dew Serum", "Beauty & Skincare",
     "Lightweight hydrating serum with watermelon extract and hyaluronic acid, suited for humid SG weather", 32.90, 0.18),
    ("SPF50+ Invisible Sunscreen Stick", "Beauty & Skincare",
     "Matte-finish sunscreen stick, no white cast, reapply over makeup, ideal for outdoor commutes", 24.50, 0.20),
    ("Snail Mucin Repair Essence 100ml", "Beauty & Skincare",
     "K-beauty essence for skin barrier repair and post-acne marks", 28.00, 0.17),
    ("Ceramide Barrier Moisturiser", "Beauty & Skincare",
     "Fragrance-free ceramide cream for sensitive aircon-dried skin", 36.00, 0.15),
    ("Vitamin C Brightening Ampoule", "Beauty & Skincare",
     "10% vitamin C ampoule targeting dullness and uneven tone", 42.00, 0.16),
    ("Heatless Curling Ribbon Set", "Beauty & Skincare",
     "Overnight heatless curls kit with satin ribbon and scrunchies", 12.90, 0.22),
    ("Foldable Laundry Basket with Wheels", "Home & Living",
     "Slim foldable laundry basket designed for HDB bathroom corners", 19.90, 0.12),
    ("Aromatherapy Diffuser with Rain Sounds", "Home & Living",
     "Ultrasonic essential-oil diffuser with white-noise mode for small bedrooms", 45.00, 0.14),
    ("Space-Saving Magnetic Knife Strip", "Home & Living",
     "Wall-mounted stainless knife strip for compact BTO kitchens", 16.50, 0.13),
    ("Anti-Mould Bathroom Sealant Tape", "Home & Living",
     "Waterproof self-adhesive tape that resists mould in humid bathrooms", 8.90, 0.25),
    ("Compact Air Fryer 3.5L", "Home & Living",
     "Small-footprint air fryer sized for 1-2 pax households", 79.00, 0.10),
    ("Blackout Magnetic Curtain Set", "Home & Living",
     "No-drill magnetic blackout curtains for rental-friendly installs", 34.90, 0.15),
    ("Wireless Earbuds with ENC Mic", "Consumer Electronics",
     "Budget true-wireless earbuds with call noise cancellation for MRT commutes", 39.90, 0.08),
    ("65W GaN Fast Charger 3-Port", "Consumer Electronics",
     "Palm-size GaN charger that powers laptop, phone and earbuds at once", 49.00, 0.09),
    ("Mini Phone Tripod with Remote", "Consumer Electronics",
     "Pocket tripod with bluetooth shutter remote, popular with content creators", 15.90, 0.18),
    ("Smart Door Sensor (No Hub)", "Consumer Electronics",
     "WiFi door sensor with phone alerts, renter-friendly, no hub needed", 22.00, 0.12),
    ("Portable Neck Fan 2025", "Consumer Electronics",
     "Bladeless rechargeable neck fan for outdoor queues and hawker dining", 26.90, 0.15),
    ("4K Action Camera Budget", "Consumer Electronics",
     "Entry 4K action cam with waterproof case for staycations and travel", 89.00, 0.07),
    ("Oversized UV-Cut Hoodie", "Fashion & Accessories",
     "UPF50 sun-protection hoodie in muted tones, breathable for tropical heat", 29.90, 0.20),
    ("Minimalist Crossbody Phone Bag", "Fashion & Accessories",
     "Slim crossbody bag that fits phone, EZ-link card and keys", 18.50, 0.22),
    ("Quick-Dry Gym Tee 3-Pack", "Fashion & Accessories",
     "Sweat-wicking tees bundle for daily gym and outdoor runs", 25.00, 0.18),
    ("Foldable Sun Visor Cap", "Fashion & Accessories",
     "Packable wide-brim visor for school pickups and park walks", 13.90, 0.24),
    ("Jelly Sandals SG Edition", "Fashion & Accessories",
     "Waterproof jelly sandals that survive sudden downpours", 21.90, 0.21),
    ("Salted Egg Fish Skin Family Pack", "Food & Beverage",
     "Crispy salted egg fish skin snack, halal-certified, family share pack", 14.90, 0.16),
    ("Premium Pandan Kaya Spread Duo", "Food & Beverage",
     "Small-batch kaya made with fresh pandan and gula melaka", 16.80, 0.15),
    ("Cold Brew Coffee Concentrate 1L", "Food & Beverage",
     "SG-roasted cold brew concentrate, 10 servings per bottle", 19.90, 0.14),
    ("Mala Instant Noodle Bundle", "Food & Beverage",
     "Numbing-spicy instant noodles with real Sichuan peppercorn oil pack", 12.50, 0.17),
    ("Low-Sugar Bubble Tea Kit", "Food & Beverage",
     "DIY brown-sugar boba kit with stevia option, makes 6 cups", 22.90, 0.18),
    ("Collagen Peptide Sachets 30-Day", "Health & Wellness",
     "Unflavoured marine collagen sachets that dissolve in kopi or water", 49.90, 0.20),
    ("Ergonomic Lumbar Support Cushion", "Health & Wellness",
     "Memory-foam lumbar cushion for WFH chairs and office use", 35.00, 0.15),
    ("Magnesium Sleep Gummies", "Health & Wellness",
     "Sugar-free magnesium gummies for better sleep quality", 28.90, 0.21),
    ("Massage Gun Mini", "Health & Wellness",
     "Palm-size percussive massage gun with 4 heads, USB-C charging", 59.00, 0.12),
    ("Posture Corrector Band", "Health & Wellness",
     "Discreet posture band wearable under office shirts", 17.90, 0.23),
    ("Montessori Busy Board for Toddlers", "Baby & Kids",
     "Quiet-play sensory busy board for 1-3 year olds, flight-friendly", 32.00, 0.18),
    ("Leak-Proof Toddler Water Bottle", "Baby & Kids",
     "One-touch straw bottle that survives being thrown from strollers", 18.90, 0.19),
    ("Baby Sleep Sack 0.5 TOG", "Baby & Kids",
     "Breathable bamboo sleep sack rated for aircon nursery temps", 26.50, 0.17),
    ("Washable Play Mat XL", "Baby & Kids",
     "Machine-washable padded play mat sized for HDB living rooms", 55.00, 0.13),
    ("Kids Water Shoes Anti-Slip", "Baby & Kids",
     "Quick-dry water shoes for water playgrounds and beach trips", 15.50, 0.20),
    ("UV Steriliser Box for Baby Items", "Baby & Kids",
     "Compact UV-C steriliser for pacifiers, teethers and small toys", 68.00, 0.11),
]


def write_products() -> None:
    rows = []
    for i, (name, cat, desc, price, comm) in enumerate(PRODUCTS, start=1):
        rows.append({
            "product_id": f"P{i:03d}",
            "name": name,
            "category": cat,
            "description": desc,
            "price_sgd": price,
            "commission_rate": comm,
            "monthly_units_sold": random.randint(40, 900),
            "rating": round(random.uniform(3.9, 4.9), 1),
        })
    _write_csv("products.csv", rows)


# ---------------------------------------------------------------- creators
CREATORS = [
    ("@sgskincarediary", "Beauty & Skincare",
     "Honest skincare reviews for oily skin in humid Singapore weather, ingredient deep dives, drugstore dupes"),
    ("@hdbhomehacks", "Home & Living",
     "Small-space organisation hacks, BTO renovation tips, rental-friendly home upgrades"),
    ("@techuncle.sg", "Consumer Electronics",
     "Budget gadget reviews, is-it-worth-it teardowns, commuter tech for MRT life"),
    ("@ootd.kopi", "Fashion & Accessories",
     "Affordable everyday outfits for tropical weather, office-to-dinner styling under $50"),
    ("@makanwithmel", "Food & Beverage",
     "Hawker food tours, snack taste tests, home cafe recipes with local flavours"),
    ("@strongaunty", "Health & Wellness",
     "Fitness for busy office workers, desk stretches, supplement myth-busting"),
    ("@mumlifesg", "Baby & Kids",
     "Real-talk parenting, toddler product tests, surviving HDB life with two kids"),
]


def write_creators() -> None:
    rows = []
    for i, (handle, niche, bio) in enumerate(CREATORS, start=1):
        rows.append({
            "creator_id": f"C{i:03d}",
            "handle": handle,
            "primary_niche": niche,
            "bio": bio,
            "followers": random.randint(8_000, 220_000),
            "avg_video_views": random.randint(3_000, 90_000),
        })
    _write_csv("creators.csv", rows)


# ---------------------------------------------------------------- reviews
POSITIVE_TEMPLATES = [
    "Quality is better than expected, {detail}. Delivery to {town} took only 2 days.",
    "Bought after seeing one video, no regrets. {detail}",
    "Cheaper than retail and works great. {detail}",
    "Repurchased twice already, {detail}. Shiok!",
    "Legit good buy, {detail}. Recommended to my colleagues.",
    "Works exactly as shown, {detail}. Packaging was secure.",
]
NEGATIVE_TEMPLATES = [
    "Hard to find any review video before buying, took a gamble. {detail}",
    "Product okay but no SG creator covers how to use it properly. {detail}",
    "Took long to decide because zero local content about this. {detail}",
    "Quality so-so, {detail}. Wish there were real demo videos first.",
    "Not as described, {detail}. Returning it.",
    "Disappointed, {detail}. The listing photos were misleading.",
]
DETAILS = {
    "Beauty & Skincare": ["texture absorbs fast in our humidity", "no breakouts after two weeks", "white cast was an issue", "bottle smaller than expected"],
    "Home & Living": ["fits my HDB bathroom corner perfectly", "assembly took 5 minutes", "material feels flimsy", "magnet grip is strong"],
    "Consumer Electronics": ["battery lasts my whole commute", "pairing was instant", "mic quality is average", "charger runs a bit warm"],
    "Fashion & Accessories": ["fabric is breathable for our weather", "sizing runs small", "colour faded after washing", "strap length is adjustable"],
    "Food & Beverage": ["taste is close to hawker standard", "expiry date was generous", "portion smaller than photos", "packaging kept it fresh"],
    "Health & Wellness": ["my back pain improved within days", "easy to mix into kopi", "effect is subtle so far", "strap digs in after an hour"],
    "Baby & Kids": ["survived my toddler throwing it daily", "easy to wash and dry", "edges could be softer", "my kid is obsessed with it"],
}
TOWNS = ["Tampines", "Jurong East", "Punggol", "Bedok", "Woodlands", "Ang Mo Kio", "Sengkang", "Clementi"]

# Demand is real but content-starved categories skew differently:
# share of positive reviews per category (demand quality signal).
POSITIVE_SHARE = {
    "Beauty & Skincare": 0.80,
    "Home & Living": 0.75,
    "Consumer Electronics": 0.60,
    "Fashion & Accessories": 0.65,
    "Food & Beverage": 0.85,
    "Health & Wellness": 0.70,
    "Baby & Kids": 0.78,
}


def write_reviews() -> None:
    rows = []
    rid = 1
    for cat in CATEGORIES:
        n = random.randint(18, 26)
        for _ in range(n):
            positive = random.random() < POSITIVE_SHARE[cat]
            template = random.choice(POSITIVE_TEMPLATES if positive else NEGATIVE_TEMPLATES)
            text = template.format(detail=random.choice(DETAILS[cat]), town=random.choice(TOWNS))
            rows.append({
                "review_id": f"R{rid:04d}",
                "category": cat,
                "review_text": text,
                "stars": random.choice([4, 5]) if positive else random.choice([1, 2, 3]),
            })
            rid += 1
    _write_csv("sg_reviews.csv", rows)


# ---------------------------------------------------------------- hashtags
# Monthly shop-tagged video counts per category. Populations (millions):
# SG 5.9, ID 277, TH 71. SG's per-capita content supply trails badly --
# this gap is the core thesis of the tool.
HASHTAG_ROWS = [
    # category, sg_videos, id_videos, th_videos
    ("Beauty & Skincare", 1850, 412000, 96500),
    ("Home & Living", 620, 188000, 41200),
    ("Consumer Electronics", 980, 224000, 52800),
    ("Fashion & Accessories", 1420, 506000, 118000),
    ("Food & Beverage", 2300, 351000, 89400),
    ("Health & Wellness", 410, 142000, 30100),
    ("Baby & Kids", 280, 96000, 22400),
]


def write_hashtags() -> None:
    rows = []
    for cat, sg, idn, th in HASHTAG_ROWS:
        rows.append({
            "category": cat,
            "sg_monthly_videos": sg,
            "id_monthly_videos": idn,
            "th_monthly_videos": th,
            "sg_population_m": 5.9,
            "id_population_m": 277.0,
            "th_population_m": 71.0,
        })
    _write_csv("hashtag_volumes.csv", rows)


def _write_csv(filename: str, rows: list[dict]) -> None:
    path = DATA_DIR / filename
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    # NOTE: products.csv is no longer generated here. The real product catalog
    # now comes from the Kaggle dataset via `python data/adapt_kaggle.py`
    # (Modules 1 & 2 run on real SKUs). write_products()/PRODUCTS are kept only
    # as a fallback if you ever need a fully-synthetic catalog again.
    write_creators()
    write_reviews()
    write_hashtags()
