"""
Instacart-based Synthetic Retail POS Dataset Generator
Combines:
- Real Instacart catalog (products + departments)
- Real basket structure (orders + order_products)
- Synthetic POS noise (PDC simulation)

Output:
- catalogue.json
- synel_dataset.csv
- transactions.json
"""

import random
import json
import csv
import re
from pathlib import Path
import pandas as pd

random.seed(42)

# ─────────────────────────────────────────────────────
# 1. LOAD REAL INSTACART DATA
# ─────────────────────────────────────────────────────

def load_catalog(products_path, departments_path):
    products = pd.read_csv(products_path)
    departments = pd.read_csv(departments_path)

    df = products.merge(departments, on="department_id")

    catalogue = []
    catalogue_map = {}

    for _, row in df.iterrows():
        name = str(row["product_name"]).strip()
        if not name:
            continue

        brand = name.split()[0]  # simple heuristic

        sku = f"SKU-{row['product_id']}"

        entry = {
            "sku": sku,
            "name": name,
            "brand": brand,
            "category": row["department"]
        }

        catalogue.append(entry)
        catalogue_map[row["product_id"]] = entry

    print(f"[INFO] Loaded {len(catalogue)} products")
    return catalogue, catalogue_map


def load_baskets(orders_path, order_products_path, max_baskets=5000):
    orders = pd.read_csv(orders_path)
    order_products = pd.read_csv(order_products_path)

    merged = order_products.merge(orders, on="order_id")

    baskets = merged.groupby("order_id")["product_id"].apply(list).tolist()

    random.shuffle(baskets)
    baskets = baskets[:max_baskets]

    print(f"[INFO] Loaded {len(baskets)} baskets")
    return baskets


# ─────────────────────────────────────────────────────
# 2. NOISE ENGINE (UNCHANGED CORE)
# ─────────────────────────────────────────────────────

ABBREVS = {
    "Organic": ["Org"],
    "Chocolate": ["Choc"],
    "Chicken": ["Chkn"],
    "Cheese": ["Chs"],
    "Bottle": ["Btl"],
    "Pack": ["Pk"],
    "Greek": ["Grk"],
    "Yogurt": ["Ygt"],
    "Strawberry": ["Strbry"],
    "Vanilla": ["Vnl"],
}

def keyboard_typo(word):
    if len(word) < 3:
        return word
    idx = random.randint(0, len(word)-1)
    return word[:idx] + random.choice("abcdefghijklmnopqrstuvwxyz") + word[idx+1:]

def drop_vowels(word):
    if len(word) <= 3:
        return word
    return word[0] + ''.join(c for c in word[1:] if c.lower() not in "aeiou")

def apply_abbreviation(text):
    for full, abbrs in ABBREVS.items():
        if full.lower() in text.lower():
            text = re.sub(full, random.choice(abbrs), text, flags=re.IGNORECASE)
    return text

def add_noise(name, noise_level=0.4):
    text = apply_abbreviation(name)

    words = text.split()
    noisy = []

    for w in words:
        r = random.random()
        if r < noise_level * 0.4:
            w = keyboard_typo(w)
        elif r < noise_level * 0.7:
            w = drop_vowels(w)
        elif r < noise_level * 0.9:
            w = w.upper() if random.random() < 0.5 else w.lower()
        noisy.append(w)

    if len(noisy) > 3 and random.random() < 0.25:
        noisy.pop(random.randint(0, len(noisy)-1))

    if len(noisy) > 2 and random.random() < 0.2:
        i = random.randint(0, len(noisy)-2)
        noisy[i] = noisy[i] + noisy[i+1]
        noisy.pop(i+1)

    return " ".join(noisy)


# ─────────────────────────────────────────────────────
# 3. BASKET GENERATION (REAL + NOISE)
# ─────────────────────────────────────────────────────

def make_basket_from_real(product_ids, catalogue_map):
    basket = []

    for pid in product_ids[:8]:  # cap size
        item = catalogue_map.get(pid)
        if not item:
            continue

        noisy = add_noise(item["name"])

        basket.append({
            "description": noisy,
            "canonical_name": item["name"],
            "sku": item["sku"],
            "quantity": random.randint(1, 4),
            "price": round(random.uniform(1.0, 20.0), 2),
            "department": item["category"],
        })

    return basket


# ─────────────────────────────────────────────────────
# 4. DATASET GENERATION
# ─────────────────────────────────────────────────────

def generate_dataset(
    products_path,
    departments_path,
    orders_path,
    order_products_path,
    output_dir="data",
    n_transactions=400
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    catalogue, catalogue_map = load_catalog(products_path, departments_path)
    baskets = load_baskets(orders_path, order_products_path)

    all_items = []
    transactions = []

    txn_id = 1000

    for product_ids in baskets[:n_transactions]:
        basket = make_basket_from_real(product_ids, catalogue_map)

        if not basket:
            continue

        txn = {"transaction_id": f"TXN-{txn_id}", "items": basket}
        transactions.append(txn)

        for item in basket:
            item["transaction_id"] = txn["transaction_id"]
            all_items.append(item)

        txn_id += 1

    # ── SAVE FILES ─────────────────────────────

    with open(output_dir / "catalogue.json", "w") as f:
        json.dump(catalogue, f, indent=2)

    with open(output_dir / "transactions.json", "w") as f:
        json.dump(transactions, f, indent=2)

    fieldnames = ["transaction_id", "description", "canonical_name", "sku",
                  "quantity", "price", "department"]

    with open(output_dir / "synel_dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_items)

    print(f"\nDONE")
    print(f"Transactions: {len(transactions)}")
    print(f"Items: {len(all_items)}")
    print(f"Saved to: {output_dir}")


# ─────────────────────────────────────────────────────
# 5. RUN
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_dataset(
        products_path="products.csv",
        departments_path="departments.csv",
        orders_path="orders.csv",
        order_products_path="order_products__train.csv",
        output_dir="data",
        n_transactions=500
    )