"""
FastAPI REST interface for the RetailEL pipeline.

Run order:
    python B-real_data_loader.py        # only needed first time
    python D-api_server.py              # starts API on port 8000

This API uses the real dataset files:
    data_real/catalogue_real.json
    data_real/synel_real.csv
"""
from __future__ import annotations

import csv
import importlib.util
import random
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline import Item, RetailELPipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_real"
CATALOGUE_PATH = DATA_DIR / "catalogue_real.json"
DATASET_CSV = DATA_DIR / "synel_real.csv"

app = FastAPI(title="RetailEL API", version="1.0.0")

# Useful if you call the API from a local frontend/demo page.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: Optional[RetailELPipeline] = None


def _load_real_data_if_available() -> None:
    """
    Create data_real/ files if they are missing by auto-running B-real_data_loader.py.
    """
    if CATALOGUE_PATH.exists() and DATASET_CSV.exists():
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    loader_path = BASE_DIR / "B-real_data_loader.py"

    if not loader_path.exists():
        raise FileNotFoundError(
            "Dataset files are missing and B-real_data_loader.py was not found.\n"
            f"Expected: {CATALOGUE_PATH} and {DATASET_CSV}\n"
            "Fix: run B-real_data_loader.py first:\n"
            "  python B-real_data_loader.py"
        )

    spec = importlib.util.spec_from_file_location("B_real_data_loader_runtime", loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import loader from {loader_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "load_real_data"):
        raise AttributeError(f"{loader_path.name} does not contain load_real_data()")

    module.load_real_data(str(DATA_DIR))


def _read_training_items() -> list[Item]:
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Training CSV not found: {DATASET_CSV}")

    items: list[Item] = []
    with open(DATASET_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                quantity = int(float(row.get("quantity", 1) or 1))
            except ValueError:
                quantity = 1

            try:
                price = float(row.get("price", 0) or 0)
            except ValueError:
                price = 0.0

            items.append(
                Item(
                    description=row.get("description", ""),
                    sku=row.get("sku", ""),
                    quantity=quantity,
                    price=price,
                    department=row.get("department", "General"),
                    transaction_id=row.get("transaction_id", "TXN-TRAIN"),
                )
            )

    return items


def get_pipeline() -> RetailELPipeline:
    """
    Lazy-load and train the RetailEL pipeline on the first request.
    """
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    try:
        _load_real_data_if_available()

        if not CATALOGUE_PATH.exists():
            raise FileNotFoundError(f"Catalogue not found: {CATALOGUE_PATH}")

        pipeline = RetailELPipeline(str(CATALOGUE_PATH))
        train_items = _read_training_items()

        if not train_items:
            raise ValueError(f"No training rows found in {DATASET_CSV}")

        random.seed(42)
        random.shuffle(train_items)
        train_split = train_items[: max(1, int(0.70 * len(train_items)))]
        pipeline.train(train_split)

        _pipeline = pipeline
        return _pipeline

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Request / Response schemas ───────────────────────────────────────────────

class SingleItemRequest(BaseModel):
    description: str = Field(..., min_length=1)
    quantity: int = 1
    price: float = 0.0
    department: str = ""
    transaction_id: str = "TXN-LIVE"


class SingleItemResponse(BaseModel):
    input: str
    predicted_sku: str
    predicted_name: str
    confidence: float
    latency_ms: float


class BasketItem(BaseModel):
    description: str = Field(..., min_length=1)
    quantity: int = 1
    price: float = 0.0
    department: str = ""


class BasketRequest(BaseModel):
    transaction_id: str = "TXN-LIVE"
    items: list[BasketItem]


class BasketItemResult(BaseModel):
    description: str
    predicted_sku: str
    predicted_name: str
    confidence: float


class BasketResponse(BaseModel):
    transaction_id: str
    results: list[BasketItemResult]
    total_latency_ms: float


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "RetailEL API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline_loaded": _pipeline is not None,
        "catalogue_exists": CATALOGUE_PATH.exists(),
        "dataset_exists": DATASET_CSV.exists(),
        "data_dir": str(DATA_DIR),
    }


@app.post("/predict/item", response_model=SingleItemResponse)
def predict_item(req: SingleItemRequest):
    pipeline = get_pipeline()
    result = pipeline.predict_single(
        description=req.description,
        quantity=req.quantity,
        price=req.price,
        department=req.department,
        transaction_id=req.transaction_id,
    )
    return SingleItemResponse(**result)


@app.post("/predict/basket", response_model=BasketResponse)
def predict_basket(req: BasketRequest):
    if not req.items:
        raise HTTPException(status_code=400, detail="Basket must contain at least one item.")

    pipeline = get_pipeline()
    t0 = time.perf_counter()

    items = [
        Item(
            description=it.description,
            sku="",
            quantity=it.quantity,
            price=it.price,
            department=it.department,
            transaction_id=req.transaction_id,
        )
        for it in req.items
    ]

    pipeline.predict_batch(items)
    total_ms = (time.perf_counter() - t0) * 1000

    sku_to_name = {entry.sku: entry.name for entry in pipeline.catalogue}
    results = [
        BasketItemResult(
            description=item.description,
            predicted_sku=item.predicted_sku,
            predicted_name=sku_to_name.get(item.predicted_sku, "UNKNOWN"),
            confidence=round(item.confidence, 4),
        )
        for item in items
    ]

    return BasketResponse(
        transaction_id=req.transaction_id,
        results=results,
        total_latency_ms=round(total_ms, 2),
    )


@app.get("/catalogue/size")
def catalogue_size():
    pipeline = get_pipeline()
    return {"num_skus": len(pipeline.catalogue)}


@app.post("/reload")
def reload_pipeline():
    """Reload and retrain the pipeline after changing data_real files."""
    global _pipeline
    _pipeline = None
    pipeline = get_pipeline()
    return {"status": "reloaded", "num_skus": len(pipeline.catalogue)}


if __name__ == "__main__":
    import uvicorn

    # Pass the app object directly — avoids module-name issues with
    # filenames that contain dashes (D-api_server.py is not importable
    # as a module by name, so "api:app" string form would fail).
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)