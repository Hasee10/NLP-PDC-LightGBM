"""
RetailEL — Full evaluation script.

Supports two evaluation modes:
  --mode synthetic   SynEL benchmark (50 SKUs, generated noise)     [default]
  --mode real        UCI Online Retail dataset (500-700 real SKUs,
                     natural description variation as noise)

Runs all experiments:
  1. Full pipeline Accuracy@1
  2. Ablation: no basket context
  3. Ablation: BM25-only retrieval
  4. Ablation: no reranker (RRF top-1)
  5. Neural bi-encoder baseline (all-MiniLM-L6-v2)
  6. LightGBM feature importance + leave-one-out ablation
  7. Noise robustness with 95% bootstrap CIs
  8. Category-level accuracy
  9. Full latency breakdown with overhead accounting

Usage:
  python C-evaluate_full.py             # real mode (default)
  python C-evaluate_full.py --mode real
  python evaluate.py --mode both      # run both and compare

Saves output to:
  results/metrics_synthetic.json   (synthetic mode)
  results/metrics_real.json        (real mode)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from pipeline import (
    Item, RetailELPipeline, SKUEntry,
    normalise_text, build_basket_contexts, extract_features,
    HybridRetriever, ConstrainedSelector,
)

random.seed(42)
np.random.seed(42)

FEATURE_NAMES = [
    "Levenshtein Distance",
    "Levenshtein Ratio",
    "Jaccard Token Overlap",
    "Bigram Overlap",
    "Char-level Jaccard",
    "Length Ratio",
    "RRF Score",
    "Prior SKU Frequency",
]


# Local fallback noise function so this evaluator does not depend on data_generator.py.
def add_noise(name: str, noise_level: float = 0.4) -> str:
    """Create simple POS-style noise for robustness testing."""
    text = str(name)
    abbrevs = {
        "Organic": "Org", "Chocolate": "Choc", "Chicken": "Chkn",
        "Cheese": "Chs", "Bottle": "Btl", "Pack": "Pk",
        "Greek": "Grk", "Yogurt": "Ygt", "Strawberry": "Strbry",
        "Vanilla": "Vnl",
    }
    for full, short in abbrevs.items():
        text = re.sub(full, short, text, flags=re.IGNORECASE)

    words = text.split()
    noisy = []
    for word in words:
        r = random.random()
        if len(word) > 3 and r < noise_level * 0.35:
            i = random.randint(0, len(word) - 1)
            word = word[:i] + random.choice("abcdefghijklmnopqrstuvwxyz") + word[i + 1:]
        elif len(word) > 3 and r < noise_level * 0.60:
            word = word[0] + ''.join(c for c in word[1:] if c.lower() not in "aeiou")
        elif r < noise_level * 0.85:
            word = word.upper() if random.random() < 0.5 else word.lower()
        noisy.append(word)

    if len(noisy) > 3 and random.random() < 0.20:
        noisy.pop(random.randint(0, len(noisy) - 1))
    return " ".join(noisy)


def safe_round(value, digits: int = 4):
    return None if value is None else round(float(value), digits)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_items(csv_path: str) -> list[Item]:
    items = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            items.append(Item(
                description=row["description"],
                sku=row["sku"],
                quantity=int(row["quantity"]),
                price=float(row["price"]),
                department=row["department"],
                transaction_id=row["transaction_id"],
            ))
    return items


def accuracy(items: list[Item]) -> float:
    correct = sum(1 for it in items if it.predicted_sku == it.sku)
    return correct / len(items) if items else 0.0


def bootstrap_ci(hits: list[int], n_boot: int = 2000, alpha: float = 0.95) -> tuple[float, float]:
    """95% bootstrap confidence interval for a 0/1 list."""
    rng = np.random.default_rng(42)
    arr = np.array(hits, dtype=float)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    lo = np.percentile(boots, (1 - alpha) / 2 * 100)
    hi = np.percentile(boots, (1 + alpha) / 2 * 100)
    return round(float(lo), 4), round(float(hi), 4)


def clone_test(test_items: list[Item]) -> list[Item]:
    """Return a fresh copy of test items (cleared predictions)."""
    return [Item(
        description=it.description, sku=it.sku,
        quantity=it.quantity, price=it.price,
        department=it.department, transaction_id=it.transaction_id,
    ) for it in test_items]


# ── Experiment helpers ────────────────────────────────────────────────────────

def run_neural_baseline(test_items: list[Item], catalogue: list[SKUEntry]) -> float:
    """Bi-encoder (all-MiniLM-L6-v2) nearest-neighbour baseline."""
    from sentence_transformers import SentenceTransformer
    print("  Loading all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")  # CPU-only
    cat_texts = [f"{e.name} {e.brand} {e.category}" for e in catalogue]
    cat_skus  = [e.sku for e in catalogue]
    cat_embs  = model.encode(cat_texts, convert_to_numpy=True,
                              show_progress_bar=False, normalize_embeddings=True)
    q_texts   = [normalise_text(it.basket_context) for it in test_items]
    q_embs    = model.encode(q_texts, convert_to_numpy=True,
                              show_progress_bar=False, normalize_embeddings=True)
    scores    = q_embs @ cat_embs.T
    for item, idx in zip(test_items, scores.argmax(axis=1)):
        item.predicted_sku = cat_skus[int(idx)]
    return accuracy(test_items)


def run_feature_ablation(train_items: list[Item], test_items: list[Item],
                         catalogue: list[SKUEntry]) -> dict:
    """LightGBM gain importance + leave-one-out accuracy drop."""
    import lightgbm as lgb

    cat_dict  = {e.sku: e for e in catalogue}
    retriever = HybridRetriever(catalogue)
    selector  = ConstrainedSelector({e.sku for e in catalogue})

    # Build basket contexts for training items
    by_txn: dict = {}
    for it in train_items:
        by_txn.setdefault(it.transaction_id, []).append(it)
    for basket in by_txn.values():
        build_basket_contexts(basket)

    # SKU priors
    sku_counts: dict[str, int] = {}
    for it in train_items:
        sku_counts[it.sku] = sku_counts.get(it.sku, 0) + 1
    total = len(train_items)
    sku_prior = {k: v / total for k, v in sku_counts.items()}

    def build_xy(items, zeroed: int = -1):
        X, y, groups = [], [], []
        for item in items:
            q = normalise_text(item.basket_context)
            cands = retriever.retrieve(q, top_k=10)
            if not cands:
                continue
            gf, gl = [], []
            for sku, rrf in cands:
                entry = cat_dict.get(sku)
                if entry is None:
                    continue
                feats = extract_features(q, entry.search_text(),
                                         rrf, sku_prior.get(sku, 0.0))
                if zeroed >= 0:
                    feats[zeroed] = 0.0
                gf.append(feats)
                gl.append(1 if sku == item.sku else 0)
            if sum(gl) == 0:
                gl[0] = 1
            X.extend(gf); y.extend(gl); groups.append(len(gf))
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), groups

    def train_lgb(X, y, groups):
        ds = lgb.Dataset(X, label=y, group=groups, feature_name=FEATURE_NAMES)
        params = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1],
                  "learning_rate": 0.05, "num_leaves": 31, "verbose": -1}
        return lgb.train(params, ds, num_boost_round=200)

    def eval_lgb(model, items, zeroed: int = -1) -> float:
        copied = clone_test(items)
        by_t: dict = {}
        for it in copied:
            by_t.setdefault(it.transaction_id, []).append(it)
        for basket in by_t.values():
            build_basket_contexts(basket)
        for item in copied:
            q = normalise_text(item.basket_context)
            cands = retriever.retrieve(q, top_k=10)
            if not cands:
                item.predicted_sku = "<UNKNOWN_SKU>"; continue
            rows = []
            for sku, rrf in cands:
                entry = cat_dict.get(sku)
                feats = extract_features(q, entry.search_text() if entry else "",
                                          rrf, sku_prior.get(sku, 0.0))
                if zeroed >= 0:
                    feats[zeroed] = 0.0
                rows.append(feats)
            preds = model.predict(np.array(rows, dtype=np.float32))
            best  = cands[int(np.argmax(preds))][0]
            item.predicted_sku = best if best in selector.valid_skus else "<UNKNOWN_SKU>"
        return accuracy(copied)

    # Full model
    print("  Training full model ...")
    Xf, yf, gf = build_xy(train_items)
    full_model  = train_lgb(Xf, yf, gf)
    full_acc    = eval_lgb(full_model, test_items)

    # Gain importance
    gain  = full_model.feature_importance(importance_type="gain")
    split = full_model.feature_importance(importance_type="split")
    tg, ts = gain.sum() + 1e-9, split.sum() + 1e-9
    feat_importance = {
        name: {"gain_pct": round(float(g / tg * 100), 2),
               "split_pct": round(float(s / ts * 100), 2)}
        for name, g, s in zip(FEATURE_NAMES, gain, split)
    }

    # Leave-one-out
    loo = {}
    for fi, fname in enumerate(FEATURE_NAMES):
        print(f"  LOO: removing '{fname}' ...")
        Xl, yl, gl = build_xy(train_items, zeroed=fi)
        loo_model  = train_lgb(Xl, yl, gl)
        loo_acc    = eval_lgb(loo_model, test_items, zeroed=fi)
        drop       = round((full_acc - loo_acc) * 100, 3)
        loo[fname] = {"acc_without": round(loo_acc, 4), "acc_drop_pp": drop}
        print(f"    Acc without: {loo_acc:.4f}  (drop: {drop:+.3f} pp)")

    return {"full_model_acc": round(full_acc, 4),
            "feature_importance": feat_importance,
            "leave_one_out": loo}


def run_noise_robustness_synthetic(pipeline: RetailELPipeline,
                                   catalogue_list: list[dict],
                                   n_per_level: int = 50) -> dict:
    """
    Accuracy@1 + 95% bootstrap CI at each synthetic noise level.
    Works for both synthetic SynEL catalogue and real UCI catalogue
    (applies artificial noise on top of canonical names).
    """
    results = {}
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        items = []
        per_sku = max(1, n_per_level // len(catalogue_list))
        for entry in catalogue_list:
            for _ in range(per_sku):
                items.append(Item(
                    description=add_noise(entry["name"], alpha),
                    sku=entry["sku"], quantity=1, price=1.0,
                    department=entry["category"],
                    transaction_id=f"NOISE_{alpha}_{entry['sku']}",
                ))
        items = items[:n_per_level]
        pipeline.predict_batch(items)
        hits = [1 if it.predicted_sku == it.sku else 0 for it in items]
        acc  = round(sum(hits) / len(hits), 4)
        lo, hi = bootstrap_ci(hits)
        results[str(alpha)] = {"acc": acc, "ci_lo": lo, "ci_hi": hi, "n": len(hits)}
        print(f"    alpha={alpha:.1f}: Acc={acc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  n={len(hits)}")
    return results


def run_noise_robustness_natural(pipeline: RetailELPipeline,
                                  test_items: list[Item]) -> dict:
    """
    For real UCI data: accuracy on items where the description IS a natural
    variant (is_variant=True) vs items where description == canonical name.
    Replaces the synthetic noise sweep for real-data evaluation.
    """
    variant_items   = [it for it in test_items if getattr(it, 'is_variant', False)]
    canonical_items = [it for it in test_items if not getattr(it, 'is_variant', False)]

    acc_variant  = accuracy(variant_items)  if variant_items  else None
    acc_canonical= accuracy(canonical_items) if canonical_items else None

    print(f"    Natural variant queries : {len(variant_items):,}  Acc={acc_variant:.4f}" if acc_variant is not None else "    No variant queries")
    print(f"    Canonical queries       : {len(canonical_items):,}  Acc={acc_canonical:.4f}" if acc_canonical is not None else "    No canonical queries")

    return {
        "variant_acc":    round(acc_variant, 4)   if acc_variant   is not None else None,
        "canonical_acc":  round(acc_canonical, 4) if acc_canonical is not None else None,
        "n_variant":      len(variant_items),
        "n_canonical":    len(canonical_items),
    }


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_synthetic_data(data_dir: str) -> tuple[list[Item], list[Item], list[Item]]:
    """Load an already-created SynEL dataset and return (train, val, test).

    This evaluator no longer imports data_generator.py because the project
    generator file is named A-generate_synthetic_data.py and uses a different
    function signature. For a smooth run, use --mode real, or place these files
    manually in data/: catalogue.json and synel_dataset.csv.
    """
    csv_path = Path(data_dir) / "synel_dataset.csv"
    catalogue_path = Path(data_dir) / "catalogue.json"
    if not csv_path.exists() or not catalogue_path.exists():
        raise FileNotFoundError(
            f"Synthetic dataset not found in {data_dir}.\n"
            "Use real mode instead:\n"
            "  python C-evaluate_full.py --mode real\n"
            "or create data/catalogue.json and data/synel_dataset.csv first."
        )
    all_items = load_items(str(csv_path))
    random.shuffle(all_items)
    n = len(all_items)
    return (all_items[:int(0.70 * n)],
            all_items[int(0.70 * n):int(0.85 * n)],
            all_items[int(0.85 * n):])


def load_real_data_items(data_dir: str = "data_real") -> tuple[list[Item], list[Item], list[Item]]:
    """
    Load UCI Online Retail data and return (train, val, test) item lists.
    The 'is_variant' field is carried as an attribute on each Item so the
    natural noise analysis can separate variant from canonical queries.
    """
    csv_path = Path(data_dir) / "synel_real.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Real dataset not found at {csv_path}.\n"
            "Run real_data_loader.py first:\n"
            "  python real_data_loader.py\n"
            "Or if automatic download fails, manually download the UCI Online\n"
            "Retail dataset and run:\n"
            "  python real_data_loader.py online_retail.xlsx"
        )
    all_items = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item = Item(
                description=row["description"],
                sku=row["sku"],
                quantity=int(float(row.get("quantity", 1) or 1)),
                price=float(row.get("price", 0) or 0),
                department=row.get("department", "General"),
                transaction_id=row["transaction_id"],
            )
            # Carry is_variant flag (not part of Item dataclass — attach dynamically)
            item.is_variant = row.get("is_variant", "False").lower() == "true"
            all_items.append(item)

    random.shuffle(all_items)
    n = len(all_items)
    return (all_items[:int(0.70 * n)],
            all_items[int(0.70 * n):int(0.85 * n)],
            all_items[int(0.85 * n):])


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_evaluation(mode: str = "synthetic",
                   data_dir: str | None = None,
                   results_dir: str = "results") -> dict:
    """
    Parameters
    ----------
    mode        : "synthetic" | "real"
    data_dir    : override default data directory
    results_dir : where to write metrics JSON
    """
    Path(results_dir).mkdir(exist_ok=True)

    is_real = (mode == "real")
    default_data_dir = "data_real" if is_real else "data"
    data_dir = data_dir or default_data_dir
    out_file = f"metrics_{'real' if is_real else 'synthetic'}.json"

    print("=" * 60)
    print(f"RetailEL — Full Evaluation  [mode={mode}]")
    print("=" * 60)

    # ── 1. Dataset ─────────────────────────────────────────────────────────────
    if is_real:
        train_items, val_items, test_items = load_real_data_items(data_dir)
        catalogue_path = str(Path(data_dir) / "catalogue_real.json")
        import json as _j
        catalogue_list = _j.loads(Path(catalogue_path).read_text())
    else:
        train_items, val_items, test_items = load_synthetic_data(data_dir)
        catalogue_path = str(Path(data_dir) / "catalogue.json")
        catalogue_list = json.loads(Path(catalogue_path).read_text())

    print(f"    Train: {len(train_items)}  Val: {len(val_items)}  Test: {len(test_items)}")

    # ── 2. Build & train main pipeline ───────────────────────────────────────
    print("\n[2] Building pipeline ...")
    pipeline = RetailELPipeline(catalogue_path)
    pipeline.train(train_items)

    # ── 3. Full pipeline evaluation ───────────────────────────────────────────
    print("\n[3] Full pipeline on test set ...")
    t0 = time.perf_counter()
    pipeline.predict_batch(test_items)
    wall = time.perf_counter() - t0

    acc1        = accuracy(test_items)
    throughput  = len(test_items) / wall
    wall_ms     = wall / len(test_items) * 1000

    stage_latencies: dict[str, list[float]] = defaultdict(list)
    llm_bypass = 0
    for item in test_items:
        for stage in ["normalisation", "retrieval", "reranking", "selection"]:
            stage_latencies[stage].append(item.stage_times.get(stage, 0) * 1000)
        if not item.stage_times.get("used_llm", True):
            llm_bypass += 1

    stage_sum_ms = sum(
        np.mean(stage_latencies[s])
        for s in ["normalisation", "retrieval", "reranking", "selection"]
    )
    overhead_ms = wall_ms - stage_sum_ms

    print(f"    Accuracy@1      : {acc1:.4f}")
    print(f"    Throughput      : {throughput:.1f} items/sec")
    print(f"    Wall-clock avg  : {wall_ms:.3f} ms")
    print(f"    Stage sum avg   : {stage_sum_ms:.3f} ms")
    print(f"    Overhead avg    : {overhead_ms:.3f} ms  (basket grouping / dispatch)")
    print(f"    LLM bypass rate : {llm_bypass / len(test_items):.2%}")

    # ── 4. Ablation: no basket context ────────────────────────────────────────
    print("\n[4] Ablation: no basket context ...")
    pipeline_noctx = RetailELPipeline(catalogue_path)
    for item in train_items:
        item.basket_context = normalise_text(item.description)
    pipeline_noctx.reranker.fit(train_items, pipeline_noctx.retriever)
    pipeline_noctx._trained = True
    abl_noctx = [Item(description=it.description, sku=it.sku,
                      quantity=it.quantity, price=it.price,
                      department=it.department, transaction_id=it.transaction_id,
                      basket_context=normalise_text(it.description))
                 for it in test_items]
    for item in abl_noctx:
        cands = pipeline_noctx.retriever.retrieve(item.basket_context, top_k=10)
        top3  = pipeline_noctx.reranker.rerank(item.basket_context, cands, top_k=3)
        sku, conf, _ = pipeline_noctx.selector.select(top3)
        item.predicted_sku, item.confidence = sku, conf
    acc_noctx = accuracy(abl_noctx)
    print(f"    Accuracy@1 (no basket context): {acc_noctx:.4f}")

    # ── 5. Ablation: BM25-only ────────────────────────────────────────────────
    print("\n[5] Ablation: BM25-only ...")
    import json as _json
    from rank_bm25 import BM25Okapi
    cat_data  = _json.loads(Path(catalogue_path).read_text())
    cat_texts = [f"{d['name']} {d['brand']} {d['category']}" for d in cat_data]
    cat_skus  = [d["sku"] for d in cat_data]
    bm25      = BM25Okapi([t.lower().split() for t in cat_texts])
    abl_bm25  = clone_test(test_items)
    by_t: dict = {}
    for it in abl_bm25:
        by_t.setdefault(it.transaction_id, []).append(it)
    for basket in by_t.values():
        build_basket_contexts(basket)
    for item in abl_bm25:
        q = normalise_text(item.basket_context).lower().split()
        item.predicted_sku = cat_skus[int(np.argmax(bm25.get_scores(q)))]
    acc_bm25 = accuracy(abl_bm25)
    print(f"    Accuracy@1 (BM25 only): {acc_bm25:.4f}")

    # ── 6. Ablation: RRF, no reranker ─────────────────────────────────────────
    print("\n[6] Ablation: RRF top-1, no reranker ...")
    abl_norr = clone_test(test_items)
    by_t2: dict = {}
    for it in abl_norr:
        by_t2.setdefault(it.transaction_id, []).append(it)
    for basket in by_t2.values():
        build_basket_contexts(basket)
    for item in abl_norr:
        cands = pipeline.retriever.retrieve(normalise_text(item.basket_context), top_k=1)
        item.predicted_sku = cands[0][0] if cands else "<UNKNOWN_SKU>"
    acc_norr = accuracy(abl_norr)
    print(f"    Accuracy@1 (no reranker): {acc_norr:.4f}")

    # ── 7. Neural bi-encoder baseline ─────────────────────────────────────────
    print("\n[7] Neural bi-encoder baseline (all-MiniLM-L6-v2) ...")
    # Basket contexts for test items are already built from step 3 above
    neural_items = clone_test(test_items)
    by_t3: dict = {}
    for it in neural_items:
        by_t3.setdefault(it.transaction_id, []).append(it)
    for basket in by_t3.values():
        build_basket_contexts(basket)
    try:
        acc_neural = run_neural_baseline(neural_items, pipeline.catalogue)
        print(f"    Accuracy@1 (bi-encoder): {acc_neural:.4f}")
    except Exception as exc:
        acc_neural = None
        print(f"    [WARN] Neural baseline skipped: {exc}")

    # ── 8. Feature importance & leave-one-out ─────────────────────────────────
    print("\n[8] Feature importance & leave-one-out ablation ...")
    ablation_results = run_feature_ablation(train_items, test_items, pipeline.catalogue)
    print(f"    Full model Acc@1: {ablation_results['full_model_acc']:.4f}")
    print("    Top-3 features by gain:")
    for k, v in sorted(ablation_results["feature_importance"].items(),
                       key=lambda x: -x[1]["gain_pct"])[:3]:
        print(f"      {k:<30} {v['gain_pct']:>6.2f}%")

    # ── 9. Noise robustness ───────────────────────────────────────────────────
    if is_real:
        # Real data: compare accuracy on natural variant vs canonical queries
        print("\n[9] Noise robustness — natural description variants ...")
        noise_results = run_noise_robustness_natural(pipeline, test_items)
    else:
        # Synthetic data: sweep artificial noise levels
        print("\n[9] Noise robustness (n=50 per level, 95% bootstrap CI) ...")
        noise_results = run_noise_robustness_synthetic(
            pipeline, catalogue_list, n_per_level=50
        )

    # ── 10. Category-level accuracy ───────────────────────────────────────────
    print("\n[10] Category-level accuracy ...")
    cat_correct: dict[str, int] = defaultdict(int)
    cat_total:   dict[str, int] = defaultdict(int)
    for item in test_items:
        cat_total[item.department] += 1
        if item.predicted_sku == item.sku:
            cat_correct[item.department] += 1
    category_acc = {
        cat: round(cat_correct[cat] / cat_total[cat], 4)
        for cat in sorted(cat_total)
    }
    for cat, acc in category_acc.items():
        print(f"    {cat:<20} {acc:.4f}  (n={cat_total[cat]})")

    # ── Compile & save ────────────────────────────────────────────────────────
    import json as _j
    import os as _os
    num_skus = len(_j.loads(Path(catalogue_path).read_text()))

    metrics = {
        "mode": mode,
        "main_results": {
            "full_pipeline_acc1":    round(acc1, 4),
            "no_basket_ctx_acc1":    round(acc_noctx, 4),
            "bm25_only_acc1":        round(acc_bm25, 4),
            "no_reranker_acc1":      round(acc_norr, 4),
            "neural_biencoder_acc1": safe_round(acc_neural, 4),
        },
        "performance": {
            "throughput_items_per_sec": round(throughput, 2),
            "wall_clock_ms":            round(wall_ms, 3),
            "stage_sum_ms":             round(stage_sum_ms, 3),
            "overhead_ms":              round(overhead_ms, 3),
            "overhead_pct":             round(overhead_ms / wall_ms * 100, 1),
            "llm_bypass_rate":          round(llm_bypass / len(test_items), 4),
            "stage_latency_ms": {
                stage: {
                    "mean": round(float(np.mean(v)),           3),
                    "p50":  round(float(np.percentile(v, 50)), 3),
                    "p95":  round(float(np.percentile(v, 95)), 3),
                    "p99":  round(float(np.percentile(v, 99)), 3),
                }
                for stage, v in stage_latencies.items()
            },
        },
        "feature_ablation":  ablation_results,
        "noise_robustness":  noise_results,
        "category_accuracy": category_acc,
        "dataset_stats": {
            "train_items": len(train_items),
            "val_items":   len(val_items),
            "test_items":  len(test_items),
            "num_skus":    num_skus,
            "data_dir":    data_dir,
        },
    }

    out_path = Path(results_dir) / out_file
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nAll metrics saved -> {out_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RetailEL evaluation")
    parser.add_argument(
        "--mode", choices=["synthetic", "real", "both"],
        default="real",
        help="synthetic: SynEL 50-SKU benchmark  |  "
             "real: UCI Online Retail 500-700 SKU  |  "
             "both: run and compare both"
    )
    parser.add_argument("--data-dir", default=None,
                        help="Override data directory path")
    parser.add_argument("--results-dir", default="results",
                        help="Where to save metrics JSON files")
    args = parser.parse_args()

    def print_summary(metrics: dict) -> None:
        print("\n" + "=" * 60)
        print(f"SUMMARY  [mode={metrics['mode']}]")
        print("=" * 60)
        m, p = metrics["main_results"], metrics["performance"]
        ds   = metrics["dataset_stats"]
        print(f"Catalogue size         : {ds['num_skus']} SKUs")
        print(f"Test items             : {ds['test_items']}")
        print(f"Full Pipeline Acc@1    : {m['full_pipeline_acc1']:.4f}")
        print(f"Neural Bi-encoder Acc@1: {m['neural_biencoder_acc1'] if m['neural_biencoder_acc1'] is not None else 'SKIPPED'}")
        print(f"BM25 Only Acc@1        : {m['bm25_only_acc1']:.4f}")
        print(f"No Reranker Acc@1      : {m['no_reranker_acc1']:.4f}")
        print(f"No Basket Ctx Acc@1    : {m['no_basket_ctx_acc1']:.4f}")
        print(f"Throughput             : {p['throughput_items_per_sec']:.1f} items/sec")
        print(f"Wall-clock avg latency : {p['wall_clock_ms']:.3f} ms")
        print(f"  Stage sum            : {p['stage_sum_ms']:.3f} ms")
        print(f"  Overhead             : {p['overhead_ms']:.3f} ms ({p['overhead_pct']}%)")
        print(f"LLM Bypass Rate        : {p['llm_bypass_rate']:.2%}")

    if args.mode == "both":
        m_syn  = run_evaluation("synthetic", args.data_dir, args.results_dir)
        m_real = run_evaluation("real",      args.data_dir, args.results_dir)
        print_summary(m_syn)
        print_summary(m_real)
        print("\n" + "=" * 60)
        print("COMPARISON: Synthetic vs Real")
        print("=" * 60)
        print(f"{'Metric':<30} {'Synthetic':>12} {'Real':>12}")
        print("-" * 56)
        for key in ["full_pipeline_acc1", "neural_biencoder_acc1",
                    "bm25_only_acc1", "no_reranker_acc1"]:
            syn_val = m_syn['main_results'][key]
            real_val = m_real['main_results'][key]
            syn_txt = f"{syn_val:.4f}" if syn_val is not None else "SKIPPED"
            real_txt = f"{real_val:.4f}" if real_val is not None else "SKIPPED"
            print(f"  {key:<28} {syn_txt:>12} {real_txt:>12}")
        print(f"  {'num_skus':<28} {m_syn['dataset_stats']['num_skus']:>12} "
              f"{m_real['dataset_stats']['num_skus']:>12}")
    else:
        metrics = run_evaluation(args.mode, args.data_dir, args.results_dir)
        print_summary(metrics)