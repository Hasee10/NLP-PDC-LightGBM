"""
real_data_loader.py  (Grocery Edition)
=======================================
Downloads and processes the Instacart Open Grocery Dataset into a format
compatible with the RetailEL pipeline.

Primary source  : Instacart Market Basket Analysis (2017 public release)
                  products.csv  — 49,688 US grocery products with real brand
                  names (Heinz, Tropicana, Ben & Jerry's, Organic Valley …)
Fallback source : arules Groceries CSV (GitHub, no auth required)
                  ~9,835 real grocery transactions, 169 unique item categories

Use-case alignment:
  This dataset models the exact problem RetailEL solves:
  A multi-store grocery chain (Walmart-style) collects POS transactions where
  the same product is described differently depending on the POS system's
  character limit (legacy 30-char, standard 40-char, modern 50-char).
  Examples for "Heinz Tomato Ketchup 32oz Bottle":
    Legacy  (30): "HNZ TMT KTCHP 32OZ"
    Standard(40): "Heinz Tomato Ketchup 32oz Btl"
    Modern  (50): "Heinz Tomato Ketchup 32oz Bottle"

What this script produces (written to data_real/):
  catalogue_real.json    — canonical SKU catalogue (up to MAX_SKUS products)
  synel_real.csv         — entity linking pairs (noisy POS desc → canonical SKU)
  transactions_real.json — basket transactions (for basket-context feature)
"""
from __future__ import annotations

import csv
import io
import json
import random
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────────
MAX_SKUS         = 500    # catalogue size target
N_TRANSACTIONS   = 2000   # synthetic baskets to generate
BASKET_SIZE_MIN  = 3
BASKET_SIZE_MAX  = 8
OUTPUT_DIR       = "data_real"

# ── Download URLs ──────────────────────────────────────────────────────────────
# Instacart products.csv (product_id, product_name, aisle_id, department_id)
INSTACART_PRODUCTS_URLS = [
    "https://raw.githubusercontent.com/nicholasjhana/short-long-term-memory/master/data/products.csv",
    "https://raw.githubusercontent.com/ramansah/ml-assignments/master/instacart-data/products.csv",
    "https://raw.githubusercontent.com/yasinkutuk/instacart/master/data/products.csv",
]
# Instacart departments.csv (department_id, department)
INSTACART_DEPTS_URLS = [
    "https://raw.githubusercontent.com/nicholasjhana/short-long-term-memory/master/data/departments.csv",
    "https://raw.githubusercontent.com/ramansah/ml-assignments/master/instacart-data/departments.csv",
]
# arules groceries fallback (each row = comma-separated items in one basket)
ARULES_URL = (
    "https://raw.githubusercontent.com/stedy/"
    "Machine-Learning-with-R-datasets/master/groceries.csv"
)

# ── Instacart department → category mapping ────────────────────────────────────
DEPT_CATEGORY = {
    "frozen":           "Frozen",
    "bakery":           "Bakery",
    "produce":          "Produce",
    "beverages":        "Beverages",
    "dry goods pasta":  "Pantry",
    "pantry":           "Pantry",
    "canned goods":     "Canned",
    "breakfast":        "Breakfast",
    "dairy eggs":       "Dairy",
    "household":        "Household",
    "snacks":           "Snacks",
    "deli":             "Deli",
    "personal care":    "Personal",
    "meat seafood":     "Meat",
    "international":    "International",
    "bulk":             "Bulk",
    "babies":           "Babies",
    "alcohol":          "Alcohol",
    "other":            "General",
    "missing":          "General",
}

# Departments we keep (skip alcohol, pets, babies, other/missing)
KEEP_DEPTS = {
    "frozen", "bakery", "produce", "beverages", "dry goods pasta",
    "pantry", "canned goods", "breakfast", "dairy eggs", "household",
    "snacks", "deli", "personal care", "meat seafood", "international",
}

# ── POS Noise: abbreviation table ─────────────────────────────────────────────
# Simulates how cashiers / legacy POS systems abbreviate product descriptions
ABBREVS: dict[str, list[str]] = {
    # Brands
    "Heinz":        ["HNZ"],
    "Campbell":     ["CMPBL", "CMPBLS"],
    "Tropicana":    ["TRPC", "TRPCA"],
    "Minute Maid":  ["MNMD", "MMAID"],
    "Coca-Cola":    ["CK", "CC", "COKE"],
    "Pepsi":        ["PPS", "PSI"],
    "Gatorade":     ["GTRD", "GAT"],
    "Kellogg":      ["KLGG", "KLG"],
    "General Mills":["GNML", "GM"],
    "Nabisco":      ["NBS", "NBSC"],
    "Kraft":        ["KFT"],
    "Nestle":       ["NSTL", "NST"],
    "Unilever":     ["UNLVR"],
    "Procter":      ["P&G"],
    "Colgate":      ["CGTE"],
    "Quaker":       ["QKR"],
    "Barilla":      ["BRL"],
    "Progresso":    ["PRGS", "PRGSO"],
    "Del Monte":    ["DLMT", "DLM"],
    "Skippy":       ["SKPY"],
    "Smucker":      ["SMKR"],
    "Organic":      ["ORG", "ORGC"],
    "Natural":      ["NATRL", "NAT"],
    "Original":     ["ORIG", "OG"],
    "Classic":      ["CLSC", "CLS"],
    "Premium":      ["PREM", "PRM"],
    "Reduced Fat":  ["RED FAT", "RDCD FAT", "RF"],
    "Fat Free":     ["FF", "FAT FR"],
    "Low Fat":      ["LF", "LO FAT"],
    "Whole Grain":  ["WHL GRN", "WG"],
    "Gluten Free":  ["GF", "GLU FR"],
    "Sugar Free":   ["SF", "SGR FR"],
    # Descriptors
    "Strawberry":   ["STRBRY", "STRWBRY", "STRBY"],
    "Blueberry":    ["BLUBRY", "BLBRY"],
    "Raspberry":    ["RASPBRY", "RSPBRY"],
    "Chocolate":    ["CHOC", "CHCLT"],
    "Vanilla":      ["VAN", "VANL"],
    "Cinnamon":     ["CINN", "CNNMN"],
    "Peanut Butter":["PNT BTR", "PB"],
    "Chicken":      ["CHKN", "CKN"],
    "Vegetable":    ["VEG", "VGTBL"],
    "Tomato":       ["TMT", "TOM"],
    "Ketchup":      ["KTCHP", "KCHP"],
    "Mustard":      ["MSTD", "MST"],
    "Mayonnaise":   ["MAYO", "MAY"],
    "Shampoo":      ["SHMP", "SHMPO"],
    "Conditioner":  ["COND", "CNDR"],
    "Toothpaste":   ["TPST", "TP"],
    "Detergent":    ["DTGT", "DTG"],
    "Disinfectant": ["DSINF", "DISIF"],
    "Laundry":      ["LNDRY", "LND"],
    "Softener":     ["SFTNR", "SFTR"],
    # Units / sizes
    "ounce":        ["oz", "OZ"],
    "ounces":       ["oz", "OZ"],
    "pound":        ["lb", "LB"],
    "pounds":       ["lb", "LB"],
    "gallon":       ["gal", "GAL"],
    "liter":        ["L", "LTR"],
    "milliliter":   ["ml", "ML"],
    "count":        ["ct", "CT"],
    "pack":         ["pk", "PK"],
    "bottle":       ["btl", "BTL"],
    "Bottle":       ["Btl", "BTL"],
    "can":          ["cn", "CN"],
    "Can":          ["Cn", "CN"],
    "box":          ["bx", "BX"],
    "Box":          ["Bx", "BX"],
    "bag":          ["bg", "BG"],
    "Bag":          ["Bg", "BG"],
    "rolls":        ["rls", "RLS"],
    "Rolls":        ["Rls", "RLS"],
}


# ── Noise helpers ──────────────────────────────────────────────────────────────

def apply_abbreviation(text: str) -> str:
    """Replace known long tokens with their POS abbreviations."""
    for full, abbrs in ABBREVS.items():
        pattern = re.compile(re.escape(full), re.IGNORECASE)
        if pattern.search(text):
            text = pattern.sub(random.choice(abbrs), text, count=1)
    return text


def pos_truncate(name: str, char_limit: int) -> str:
    """
    Truncate a product name to char_limit, respecting word boundaries.
    Simulates what a POS system does when the description field is too short.
    """
    if len(name) <= char_limit:
        return name
    truncated = name[:char_limit]
    last_space = truncated.rfind(" ")
    if last_space > int(char_limit * 0.55):
        return truncated[:last_space].rstrip()
    return truncated.rstrip()


def keyboard_typo(word: str) -> str:
    """Single keyboard-adjacent character substitution."""
    kmap = {
        'a': 'sq', 'b': 'vn', 'c': 'xv', 'd': 'sf', 'e': 'wr',
        'f': 'dg', 'g': 'fh', 'h': 'gj', 'i': 'uo', 'j': 'hk',
        'k': 'jl', 'l': 'k',  'm': 'n',  'n': 'mb', 'o': 'ip',
        'p': 'o',  'r': 'et', 's': 'ad', 't': 'ry', 'u': 'yi',
        'v': 'cb', 'w': 'qe', 'x': 'zc', 'y': 'tu', 'z': 'x',
    }
    if len(word) < 2:
        return word
    idx = random.randint(0, len(word) - 1)
    ch  = word[idx].lower()
    if ch in kmap and kmap[ch]:
        return word[:idx] + random.choice(kmap[ch]) + word[idx + 1:]
    return word


def make_pos_noise(canonical: str, pos_style: str) -> str:
    """
    Generate a noisy POS description for a canonical product name.

    pos_style : "legacy"   → 30-char ALL-CAPS (oldest systems)
                "standard" → 40-char Title Case (common mid-range POS)
                "modern"   → 50-char natural case (newer systems)
    """
    limits = {"legacy": 30, "standard": 40, "modern": 50}
    char_limit = limits.get(pos_style, 40)

    text = apply_abbreviation(canonical)

    if pos_style == "legacy":
        text = text.upper()
        # Legacy also adds occasional run-together words
        words = text.split()
        if len(words) > 2 and random.random() < 0.35:
            i = random.randint(0, len(words) - 2)
            words[i] = words[i] + words[i + 1]
            words.pop(i + 1)
        text = " ".join(words)

    elif pos_style == "standard":
        # Occasional keyboard typo
        words = text.split()
        words = [keyboard_typo(w) if random.random() < 0.15 else w for w in words]
        text = " ".join(words)

    # else modern: light changes, just abbreviation + truncation

    text = pos_truncate(text, char_limit)
    return text.strip()


def generate_variants(canonical: str) -> list[str]:
    """Return 3 POS-style noisy variants of a canonical product name."""
    return [
        make_pos_noise(canonical, "legacy"),
        make_pos_noise(canonical, "standard"),
        make_pos_noise(canonical, "modern"),
    ]


# ── Download helpers ───────────────────────────────────────────────────────────

def try_download_csv(url: str, label: str = "") -> list[dict] | None:
    """Download a CSV and return list of row dicts, or None on failure."""
    tag = f"  [{label}] " if label else "  "
    print(f"{tag}Trying {url} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            content = resp.read().decode("utf-8", errors="replace")
        rows = list(csv.DictReader(io.StringIO(content)))
        print(f"{tag}Loaded {len(rows):,} rows.")
        return rows
    except Exception as exc:
        print(f"{tag}Failed: {exc}")
        return None


# ── Instacart path ────────────────────────────────────────────────────────────

def load_instacart(max_skus: int = MAX_SKUS) -> list[dict] | None:
    """
    Try each Instacart mirror in turn.  Returns a catalogue list of
    {sku, name, brand, category} dicts, or None if all mirrors fail.
    """
    # -- products.csv
    products_rows = None
    for url in INSTACART_PRODUCTS_URLS:
        products_rows = try_download_csv(url, "Instacart products")
        if products_rows:
            break
    if not products_rows:
        return None

    # Detect column names (some mirrors rename them)
    sample = products_rows[0]
    pid_col  = next((k for k in sample if "product_id"   in k.lower()), None)
    name_col = next((k for k in sample if "product_name" in k.lower()), None)
    dept_col = next((k for k in sample if "department_id" in k.lower()), None)
    if not name_col:
        print("  products.csv column detection failed.")
        return None

    # -- departments.csv (optional; we fall back to numeric IDs)
    dept_map: dict[str, str] = {}
    for url in INSTACART_DEPTS_URLS:
        dept_rows = try_download_csv(url, "Instacart depts")
        if dept_rows:
            d_id_col   = next((k for k in dept_rows[0] if "department_id" in k.lower()), None)
            d_name_col = next((k for k in dept_rows[0] if k.lower() in ("department", "department_name")), None)
            if d_id_col and d_name_col:
                for r in dept_rows:
                    dept_map[str(r[d_id_col]).strip()] = str(r[d_name_col]).strip().lower()
            break

    # Fall back to known Instacart department IDs if download failed
    if not dept_map:
        dept_map = {
            "1": "frozen",      "2": "other",       "3": "bakery",
            "4": "produce",     "5": "alcohol",      "6": "international",
            "7": "beverages",   "8": "pets",         "9": "dry goods pasta",
            "10": "bulk",       "11": "personal care","12": "meat seafood",
            "13": "pantry",     "14": "breakfast",   "15": "canned goods",
            "16": "dairy eggs", "17": "household",   "18": "babies",
            "19": "snacks",     "20": "deli",        "21": "missing",
        }

    # -- Filter and build catalogue
    catalogue = []
    seen_names: set[str] = set()
    for row in products_rows:
        name = str(row.get(name_col, "") or "").strip()
        if not name or name.upper() in ("NAN", "NONE", ""):
            continue
        name_key = name.lower()
        if name_key in seen_names:
            continue

        dept_id = str(row.get(dept_col, "") or "").strip()
        dept_name = dept_map.get(dept_id, "other").lower()

        if dept_name not in KEEP_DEPTS:
            continue

        seen_names.add(name_key)

        # Heuristic: first capitalised word = brand if it looks like a proper noun
        words = name.split()
        brand = words[0].title() if words and words[0][0].isupper() else "Generic"

        sku_num = len(catalogue) + 1
        catalogue.append({
            "sku":      f"GRC-{sku_num:04d}",
            "name":     name,
            "brand":    brand,
            "category": DEPT_CATEGORY.get(dept_name, "General"),
        })
        if len(catalogue) >= max_skus:
            break

    print(f"  Instacart catalogue: {len(catalogue)} SKUs (after dept filter)")
    return catalogue if catalogue else None


# ── arules fallback path ───────────────────────────────────────────────────────

# Expand arules category tokens → realistic grocery product names
ARULES_EXPANSION: dict[str, list[tuple[str, str, str]]] = {
    "whole milk":           [("Horizon Organic Whole Milk 1gal", "Horizon", "Dairy"),
                             ("Lucerne Whole Milk 1gal", "Lucerne", "Dairy"),
                             ("Great Value Whole Milk 1gal", "GreatValue", "Dairy")],
    "other vegetables":     [("Birds Eye Mixed Vegetables 12oz", "BirdsEye", "Frozen"),
                             ("Green Giant Broccoli Florets 12oz", "GreenGiant", "Frozen")],
    "rolls/buns":           [("Martin's Potato Rolls 15ct", "Martins", "Bakery"),
                             ("Wonder Classic Hamburger Buns 8ct", "Wonder", "Bakery")],
    "soda":                 [("Coca-Cola Classic 12pk 12oz", "CocaCola", "Beverages"),
                             ("Pepsi Regular 12pk 12oz", "Pepsi", "Beverages"),
                             ("Sprite 12pk 12oz Cans", "Sprite", "Beverages")],
    "yogurt":               [("Chobani Greek Yogurt Plain 32oz", "Chobani", "Dairy"),
                             ("Yoplait Strawberry Yogurt 6oz", "Yoplait", "Dairy")],
    "root vegetables":      [("Bolthouse Farms Carrots 2lb Bag", "Bolthouse", "Produce"),
                             ("Sweet Potatoes 3lb Bag", "Generic", "Produce")],
    "tropical fruit":       [("Dole Pineapple Chunks 20oz Can", "Dole", "Canned"),
                             ("Mango Chunks 10oz Bag", "Generic", "Produce")],
    "bottled water":        [("Nestle Pure Life Water 24pk 16.9oz", "Nestle", "Beverages"),
                             ("Dasani Purified Water 24pk", "Dasani", "Beverages")],
    "pork":                 [("Oscar Mayer Bacon 16oz", "OscarMayer", "Meat"),
                             ("Smithfield Pork Chops 2lb", "Smithfield", "Meat")],
    "pastry":               [("Entenmann's Glazed Donuts 8ct", "Entenmanns", "Bakery"),
                             ("Little Debbie Honey Buns 6ct", "LittleDebbie", "Bakery")],
    "fruit/vegetable juice":  [("Tropicana OJ No Pulp 52oz", "Tropicana", "Beverages"),
                               ("Ocean Spray Cranberry Juice 64oz", "OceanSpray", "Beverages")],
    "whipped/sour cream":   [("Daisy Sour Cream 16oz", "Daisy", "Dairy"),
                             ("Land O Lakes Sour Cream 16oz", "LandOLakes", "Dairy")],
    "pip fruit":            [("Fuji Apple 3lb Bag", "Generic", "Produce"),
                             ("Gala Apples 5lb Bag", "Generic", "Produce")],
    "domestic eggs":        [("Eggland's Best Eggs 12ct Large", "Egglands", "Dairy"),
                             ("Vital Farms Eggs 12ct", "VitalFarms", "Dairy")],
    "brown bread":          [("Dave's Killer Bread 21 Grain", "Daves", "Bakery"),
                             ("Nature's Own Wheat Bread 20oz", "NaturesOwn", "Bakery")],
    "margarine":            [("I Can't Believe It's Not Butter 15oz", "ICBINB", "Dairy"),
                             ("Smart Balance Buttery Spread 15oz", "SmartBalance", "Dairy")],
    "beef":                 [("80/20 Ground Beef 1lb", "Generic", "Meat"),
                             ("Angus Beef Patties 4ct", "Generic", "Meat")],
    "frankfurter":          [("Oscar Mayer Wieners 16oz", "OscarMayer", "Meat"),
                             ("Ball Park Beef Franks 15oz", "BallPark", "Meat")],
    "bottled beer":         [("Budweiser Beer 12pk 12oz", "Budweiser", "Beverages"),
                             ("Coors Light Beer 12pk 12oz", "Coors", "Beverages")],
    "whole milk cheese":    [("Kraft Shredded Mozzarella 8oz", "Kraft", "Dairy"),
                             ("Sargento Sliced Cheddar 7oz", "Sargento", "Dairy")],
    "canned beer":          [("Miller Lite 12pk 12oz Cans", "Miller", "Beverages"),
                             ("Bud Light 12pk 12oz Cans", "BudLight", "Beverages")],
    "newspapers":           [("Bic Ballpoint Pens 10ct", "Bic", "General"),
                             ("Composition Notebook 100pg", "Generic", "General")],
    "chicken":              [("Tyson Chicken Breasts 3lb", "Tyson", "Meat"),
                             ("Perdue Whole Chicken 4lb", "Perdue", "Meat")],
    "curd":                 [("Kraft Velveeta Shells & Cheese 12oz", "Kraft", "Pantry"),
                             ("Horizon Organic Cottage Cheese 16oz", "Horizon", "Dairy")],
    "shopping bags":        [("Glad Kitchen Trash Bags 13gal 80ct", "Glad", "Household"),
                             ("Hefty Trash Bags 30gal 20ct", "Hefty", "Household")],
    "butter":               [("Land O Lakes Unsalted Butter 4ct", "LandOLakes", "Dairy"),
                             ("Kerrygold Grass-Fed Butter 8oz", "Kerrygold", "Dairy")],
    "fruit/vegetable":      [("Fresh Strawberries 1lb Clamshell", "Generic", "Produce"),
                             ("Broccoli Crowns 1.5lb", "Generic", "Produce")],
    "rice":                 [("Uncle Ben's Long Grain White Rice 5lb", "UncleBens", "Pantry"),
                             ("Mahatma Jasmine Rice 5lb", "Mahatma", "Pantry")],
    "abrasive cleaner":     [("Comet Powder Cleanser 21oz", "Comet", "Household"),
                             ("Bar Keepers Friend 21oz", "BKF", "Household")],
    "flour":                [("Gold Medal All Purpose Flour 5lb", "GoldMedal", "Pantry"),
                             ("King Arthur Bread Flour 5lb", "KingArthur", "Pantry")],
    "UHT-milk":             [("Parmalat Whole Milk 1L", "Parmalat", "Dairy"),
                             ("Horizon Organic 2% Milk 8pk", "Horizon", "Dairy")],
    "frozen vegetables":    [("Birds Eye Broccoli Florets 12oz", "BirdsEye", "Frozen"),
                             ("Green Giant Sweet Corn 12oz", "GreenGiant", "Frozen")],
    "herbs":                [("McCormick Garlic Powder 3.12oz", "McCormick", "Condiments"),
                             ("Spice Islands Basil 0.5oz", "SpiceIslands", "Condiments")],
    "cream cheese":         [("Philadelphia Cream Cheese 8oz", "Philadelphia", "Dairy"),
                             ("Kraft Cream Cheese Block 8oz", "Kraft", "Dairy")],
    "frozen meals":         [("Lean Cuisine Chicken Alfredo 9oz", "LeanCuisine", "Frozen"),
                             ("Marie Callender's Pot Pie 10oz", "MarieCallender", "Frozen")],
    "onions":               [("Yellow Onions 3lb Bag", "Generic", "Produce"),
                             ("Red Onions 2lb Bag", "Generic", "Produce")],
    "beverages":            [("Gatorade Lemon Lime 32oz", "Gatorade", "Beverages"),
                             ("Snapple Peach Tea 16oz", "Snapple", "Beverages")],
    "canned fish":          [("StarKist Chunk Light Tuna in Water 5oz", "StarKist", "Canned"),
                             ("Bumble Bee Solid White Albacore 5oz", "BumbleBee", "Canned")],
    "hygiene articles":     [("Dove Bar Soap 6pk 3.75oz", "Dove", "Personal"),
                             ("Colgate Toothpaste Total 4.8oz", "Colgate", "Personal")],
    "waffles/pancakes":     [("Eggo Homestyle Waffles 10ct", "Eggo", "Frozen"),
                             ("Aunt Jemima Buttermilk Pancake Mix 32oz", "AuntJemima", "Breakfast")],
    "frozen potato":        [("Ore-Ida Golden Fries 32oz", "OreIda", "Frozen"),
                             ("Alexia Waffle Fries 20oz", "Alexia", "Frozen")],
    "pasta":                [("Barilla Spaghetti 16oz", "Barilla", "Pantry"),
                             ("Ronzoni Penne Rigate 16oz", "Ronzoni", "Pantry")],
    "vinegar":              [("Heinz White Vinegar 32oz", "Heinz", "Condiments"),
                             ("Bragg Apple Cider Vinegar 32oz", "Bragg", "Condiments")],
    "sugar":                [("Domino Pure Cane Sugar 4lb", "Domino", "Pantry"),
                             ("C&H Pure Cane Sugar 4lb", "CH", "Pantry")],
    "cereals":              [("Kellogg's Corn Flakes 18oz", "Kelloggs", "Breakfast"),
                             ("Cheerios Original 18oz", "Cheerios", "Breakfast"),
                             ("Special K Original 12.2oz", "Kelloggs", "Breakfast")],
    "coffee":               [("Folgers Classic Roast 30.5oz", "Folgers", "Beverages"),
                             ("Maxwell House Original 30.6oz", "MaxwellHouse", "Beverages"),
                             ("Starbucks Pike Place 12oz", "Starbucks", "Beverages")],
    "oil":                  [("Crisco Vegetable Oil 48oz", "Crisco", "Pantry"),
                             ("Mazola Corn Oil 40oz", "Mazola", "Pantry")],
    "cat food":             [("Purina Cat Chow 3.15lb", "Purina", "Pets"),
                             ("Fancy Feast Gravy 3oz Can", "FancyFeast", "Pets")],
    "candy":                [("M&Ms Milk Chocolate 10.7oz", "MMs", "Snacks"),
                             ("Reese's Peanut Butter Cups 2pk", "Reeses", "Snacks"),
                             ("Snickers Bar 1.86oz", "Snickers", "Snacks")],
    "chocolate":            [("Hershey's Milk Chocolate Bar 1.55oz", "Hersheys", "Snacks"),
                             ("Ghirardelli Dark Chocolate 3.5oz", "Ghirardelli", "Snacks")],
    "misc. beverages":      [("Red Bull Energy 8.4oz Can", "RedBull", "Beverages"),
                             ("Monster Energy Original 16oz", "Monster", "Beverages")],
    "ice cream":            [("Ben & Jerry's Chocolate Fudge Brownie Pint", "BenJerrys", "Frozen"),
                             ("Haagen-Dazs Vanilla Ice Cream 14oz", "HaagenDazs", "Frozen"),
                             ("Breyers Natural Vanilla 48oz", "Breyers", "Frozen")],
    "napkins":              [("Bounty Select-A-Size 12 Rolls", "Bounty", "Household"),
                             ("Scott Comfort Plus 12 Rolls", "Scott", "Household")],
    "canned vegetables":    [("Del Monte Sweet Corn 15.25oz", "DelMonte", "Canned"),
                             ("Bush's Best Pinto Beans 15.5oz", "Bushs", "Canned")],
    "toilet cleaner":       [("Lysol Toilet Bowl Cleaner 24oz", "Lysol", "Household"),
                             ("Clorox Toilet Bowl Cleaner 24oz", "Clorox", "Household")],
    "dish cleaner":         [("Dawn Dish Soap Original 19.4oz", "Dawn", "Household"),
                             ("Palmolive Dish Soap 20oz", "Palmolive", "Household")],
    "detergent":            [("Tide Pods Laundry 42ct", "Tide", "Household"),
                             ("Gain Flings Laundry 42ct", "Gain", "Household")],
    "house keeping":        [("Swiffer Sweeper Dry Refills 32ct", "Swiffer", "Household"),
                             ("Pledge Furniture Spray 9.7oz", "Pledge", "Household")],
    "frozen fish":          [("Gorton's Fish Fillets 19oz", "Gortons", "Frozen"),
                             ("Van de Kamp's Fish Sticks 24ct", "VanddeKamps", "Frozen")],
    "photo/film":           [("Duracell AA Batteries 10ct", "Duracell", "General"),
                             ("Energizer AA Batteries 8ct", "Energizer", "General")],
    "liquor (spirits)":     [("Smirnoff Vodka 750ml", "Smirnoff", "Beverages"),
                             ("Jack Daniel's Tennessee 750ml", "JackDaniels", "Beverages")],
    "baby cosmetics":       [("Johnson's Baby Shampoo 13.6oz", "Johnsons", "Babies"),
                             ("Pampers Sensitive Wipes 216ct", "Pampers", "Babies")],
    "canned fruit":         [("Dole Sliced Peaches 15.25oz", "Dole", "Canned"),
                             ("Del Monte Fruit Cocktail 15.25oz", "DelMonte", "Canned")],
    "ham":                  [("Oscar Mayer Deli Fresh Ham 9oz", "OscarMayer", "Deli"),
                             ("Boar's Head Honey Ham 8oz", "BoarsHead", "Deli")],
    "processed cheese":     [("Kraft Singles American Cheese 24ct", "Kraft", "Dairy"),
                             ("Velveeta Original 32oz", "Velveeta", "Dairy")],
    "pickled vegetables":   [("Vlasic Kosher Dill Pickles 32oz", "Vlasic", "Condiments"),
                             ("Mt. Olive Pickles Bread & Butter 24oz", "MtOlive", "Condiments")],
    "fish":                 [("Atlantic Salmon Fillet 1lb", "Generic", "Meat"),
                             ("Tilapia Fillets 2lb", "Generic", "Meat")],
    "mustard":              [("French's Yellow Mustard 20oz", "Frenchs", "Condiments"),
                             ("Grey Poupon Dijon Mustard 10oz", "GreyPoupon", "Condiments")],
    "spread":               [("Skippy Peanut Butter Creamy 16.3oz", "Skippy", "Spreads"),
                             ("Jif Peanut Butter Creamy 16oz", "Jif", "Spreads"),
                             ("Smucker's Strawberry Jam 18oz", "Smuckers", "Spreads")],
    "soup":                 [("Campbell's Tomato Soup 10.75oz", "Campbells", "Canned"),
                             ("Progresso Chicken Noodle 18.5oz", "Progresso", "Canned"),
                             ("Knorr Chicken Noodle Soup Mix 1.8oz", "Knorr", "Canned")],
    "ketchup":              [("Heinz Tomato Ketchup 32oz Bottle", "Heinz", "Condiments"),
                             ("Hunt's Ketchup 24oz Bottle", "Hunts", "Condiments")],
    "potato chips":         [("Lay's Classic Chips 8oz Bag", "Lays", "Snacks"),
                             ("Pringles Original 5.96oz Can", "Pringles", "Snacks"),
                             ("Doritos Nacho Cheese 9.75oz", "Doritos", "Snacks")],
    "mayonnaise":           [("Hellmann's Real Mayonnaise 30oz", "Hellmanns", "Condiments"),
                             ("Duke's Mayonnaise 32oz", "Dukes", "Condiments")],
}


def load_arules(max_skus: int = MAX_SKUS) -> tuple[list[dict], list[list[str]]] | None:
    """
    Download arules groceries. Returns (catalogue, raw_basket_tokens) or None.
    """
    rows = try_download_csv(ARULES_URL, "arules groceries")
    if not rows:
        return None

    # arules CSV: no header — each row value is a comma-separated item list
    # csv.DictReader treats the first row as header, so we need raw lines
    try:
        req = urllib.request.Request(ARULES_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None

    raw_baskets: list[list[str]] = []
    for line in raw.splitlines():
        tokens = [t.strip().lower() for t in line.split(",") if t.strip()]
        if tokens:
            raw_baskets.append(tokens)

    # Build catalogue from expansion table
    catalogue: list[dict] = []
    seen_names: set[str] = set()
    sku_num = 1
    for token, products in ARULES_EXPANSION.items():
        for (name, brand, category) in products:
            if name.lower() in seen_names:
                continue
            seen_names.add(name.lower())
            catalogue.append({
                "sku":      f"GRC-{sku_num:04d}",
                "name":     name,
                "brand":    brand,
                "category": category,
                "_arules_token": token,
            })
            sku_num += 1
            if len(catalogue) >= max_skus:
                break
        if len(catalogue) >= max_skus:
            break

    print(f"  arules catalogue: {len(catalogue)} SKUs from expansion table")
    return catalogue, raw_baskets


# ── Transaction / pair builder ─────────────────────────────────────────────────

def build_transactions_and_pairs(
    catalogue: list[dict],
    n_transactions: int = N_TRANSACTIONS,
) -> tuple[list[dict], list[dict]]:
    """
    Generate synthetic baskets from the catalogue.
    For each item in each basket, generate one POS-noisy description.
    POS style is chosen randomly (legacy / standard / modern) so the dataset
    reflects the mixed-system reality of a multi-store chain.
    """
    sku_list  = [e["sku"] for e in catalogue]
    sku_index = {e["sku"]: e for e in catalogue}
    pos_styles = ["legacy", "standard", "modern"]

    transactions: list[dict] = []
    all_pairs:    list[dict] = []
    txn_id = 5000

    for _ in range(n_transactions):
        size   = random.randint(BASKET_SIZE_MIN, BASKET_SIZE_MAX)
        chosen = random.sample(sku_list, min(size, len(sku_list)))
        basket = []
        for sku in chosen:
            entry = sku_index[sku]
            style = random.choice(pos_styles)
            noisy = make_pos_noise(entry["name"], style)
            item  = {
                "transaction_id": f"TXN-{txn_id}",
                "description":    noisy,
                "canonical_name": entry["name"],
                "sku":            sku,
                "quantity":       random.randint(1, 4),
                "price":          round(random.uniform(0.99, 14.99), 2),
                "department":     entry["category"],
                "pos_style":      style,        # which POS system generated this
                "is_variant":     noisy.lower() != entry["name"].lower(),
            }
            basket.append(item)

        transactions.append({"transaction_id": f"TXN-{txn_id}", "items": basket})
        all_pairs.extend(basket)
        txn_id += 1

    variant_ct = sum(1 for p in all_pairs if p["is_variant"])
    print(f"  Generated {len(transactions)} baskets, {len(all_pairs)} pairs")
    print(f"  Noisy variants: {variant_ct:,} / {len(all_pairs):,} "
          f"({variant_ct / len(all_pairs) * 100:.1f}%)")
    return transactions, all_pairs


# ── Entry point ────────────────────────────────────────────────────────────────

def load_real_data(output_dir: str = OUTPUT_DIR,
                   force_download: bool = False) -> dict:
    """
    Download real grocery data, generate POS-noise pairs, and save outputs.

    Parameters
    ----------
    output_dir     : directory for catalogue_real.json, synel_real.csv,
                     transactions_real.json
    force_download : re-process even if outputs already exist

    Returns
    -------
    dict with catalogue_path, dataset_csv, transactions_json,
             num_skus, num_pairs, num_transactions
    """
    out      = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cat_path = out / "catalogue_real.json"
    csv_path = out / "synel_real.csv"
    txn_path = out / "transactions_real.json"

    if not force_download and cat_path.exists() and csv_path.exists():
        print(f"Grocery data already exists in '{output_dir}'. Skipping download.")
        cat  = json.loads(cat_path.read_text())
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return {
            "catalogue_path":    str(cat_path),
            "dataset_csv":       str(csv_path),
            "transactions_json": str(txn_path),
            "num_skus":          len(cat),
            "num_pairs":         len(rows),
        }

    print("=" * 60)
    print("Downloading Real Grocery Dataset (Instacart / arules)")
    print("=" * 60)

    catalogue: list[dict] | None = None

    # ── 1. Try Instacart ───────────────────────────────────────────────────────
    print("\n[1] Attempting Instacart Open Dataset ...")
    catalogue = load_instacart(MAX_SKUS)

    # ── 2. Fall back to arules ─────────────────────────────────────────────────
    if not catalogue:
        print("\n[2] Falling back to arules Groceries dataset ...")
        result = load_arules(MAX_SKUS)
        if result:
            catalogue, _ = result

    # ── 3. Hard-stop if nothing worked ────────────────────────────────────────
    if not catalogue:
        raise RuntimeError(
            "\nAll download attempts failed.\n"
            "Check your internet connection and try again, or manually supply\n"
            "a products CSV and call load_real_data_from_file()."
        )

    # Strip internal helper field
    for entry in catalogue:
        entry.pop("_arules_token", None)

    # ── 4. Generate transactions & entity-linking pairs ────────────────────────
    print(f"\nBuilding {N_TRANSACTIONS} synthetic POS transactions ...")
    transactions, pairs = build_transactions_and_pairs(catalogue, N_TRANSACTIONS)

    # ── 5. Save ────────────────────────────────────────────────────────────────
    cat_path.write_text(json.dumps(catalogue, indent=2))
    print(f"\nCatalogue saved    -> {cat_path}  ({len(catalogue)} SKUs)")

    fieldnames = ["transaction_id", "description", "canonical_name", "sku",
                  "quantity", "price", "department", "pos_style", "is_variant"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)
    print(f"Dataset CSV saved  -> {csv_path}  ({len(pairs)} pairs)")

    txn_path.write_text(json.dumps(transactions[:5000], indent=2))
    print(f"Transactions saved -> {txn_path}  ({min(len(transactions), 5000)} baskets)")

    return {
        "catalogue_path":    str(cat_path),
        "dataset_csv":       str(csv_path),
        "transactions_json": str(txn_path),
        "num_skus":          len(catalogue),
        "num_pairs":         len(pairs),
        "num_transactions":  len(transactions),
    }


def load_real_data_from_file(filepath: str,
                              output_dir: str = OUTPUT_DIR) -> dict:
    """
    Process a locally supplied grocery products file (CSV).
    Expected columns (flexible): product_id/sku, product_name/name,
                                  department/category, brand (optional)

    Example
    -------
    load_real_data_from_file("my_grocery_products.csv")
    """
    fp = Path(filepath)
    if not fp.exists():
        raise FileNotFoundError(fp)

    print(f"Loading from file: {fp}")
    with open(fp, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows):,} rows.")

    # Build catalogue
    sample = rows[0] if rows else {}
    name_col = next((k for k in sample if "name" in k.lower()), None)
    dept_col = next((k for k in sample if any(x in k.lower() for x in
                     ("department", "category", "dept"))), None)
    brand_col = next((k for k in sample if "brand" in k.lower()), None)

    if not name_col:
        raise ValueError("Could not detect product name column. "
                         f"Available columns: {list(sample.keys())}")

    catalogue = []
    seen: set[str] = set()
    for i, row in enumerate(rows):
        name = str(row.get(name_col, "") or "").strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        brand    = str(row.get(brand_col, "") or "").strip() or name.split()[0]
        category = str(row.get(dept_col, "") or "General").strip()
        catalogue.append({
            "sku":      f"GRC-{len(catalogue)+1:04d}",
            "name":     name,
            "brand":    brand,
            "category": category,
        })
        if len(catalogue) >= MAX_SKUS:
            break

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    transactions, pairs = build_transactions_and_pairs(catalogue, N_TRANSACTIONS)

    cat_path = out / "catalogue_real.json"
    csv_path = out / "synel_real.csv"
    txn_path = out / "transactions_real.json"

    cat_path.write_text(json.dumps(catalogue, indent=2))
    fieldnames = ["transaction_id", "description", "canonical_name", "sku",
                  "quantity", "price", "department", "pos_style", "is_variant"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)
    txn_path.write_text(json.dumps(transactions[:5000], indent=2))

    print(f"Saved {len(catalogue)} SKUs, {len(pairs)} pairs to '{output_dir}/'")
    return {
        "catalogue_path":    str(cat_path),
        "dataset_csv":       str(csv_path),
        "transactions_json": str(txn_path),
        "num_skus":          len(catalogue),
        "num_pairs":         len(pairs),
        "num_transactions":  len(transactions),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = load_real_data_from_file(sys.argv[1])
    else:
        result = load_real_data()

    print("\nSummary:")
    for k, v in result.items():
        print(f"  {k}: {v}")
