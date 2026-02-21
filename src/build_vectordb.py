import re
import json
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# -------------------------
# 1) Helpers: normalization + parsing
# -------------------------
def norm(s: str) -> str:
    """Normalize ingredient tokens for matching."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).lower().strip()
    # remove common punctuation but keep spaces
    s = re.sub(r"[\u200b\ufeff]", "", s)  # zero-width/BOM
    s = re.sub(r"\([^)]*\)", " ", s)      # remove parenthetical (e.g., "Fragrance (Parfum)")
    s = re.sub(r"[^a-z0-9\s\-\/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_inci_list(inci_raw: str) -> list[str]:
    """
    Split product INCI list into tokens.
    Most lists are comma-separated.
    """
    if inci_raw is None or (isinstance(inci_raw, float) and pd.isna(inci_raw)):
        return []
    # Keep it simple: split by comma, normalize each token
    parts = [p.strip() for p in str(inci_raw).split(",")]
    toks = [norm(p) for p in parts]
    # drop empties
    return [t for t in toks if t]


def price_to_float(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return None


# -------------------------
# 2) Build a lookup from ingredientsList.csv
# -------------------------
def build_ingredient_lookup(inci_df: pd.DataFrame) -> dict[str, dict]:
    """
    Create mapping: normalized ingredient name -> ingredient record
    Uses:
      - name (primary)
      - scientific_name (secondary, if present)
    """
    lookup = {}

    for _, row in inci_df.iterrows():
        name = row.get("name", "")
        sci = row.get("scientific_name", "")

        rec = {
            "name": str(name) if not pd.isna(name) else "",
            "scientific_name": str(sci) if not pd.isna(sci) else "",
            "short_description": str(row.get("short_description", "")) if "short_description" in inci_df.columns else "",
            "what_is_it": str(row.get("what_is_it", "")) if "what_is_it" in inci_df.columns else "",
            "what_does_it_do": str(row.get("what_does_it_do", "")) if "what_does_it_do" in inci_df.columns else "",
            "who_is_it_good_for": str(row.get("who_is_it_good_for", "")) if "who_is_it_good_for" in inci_df.columns else "",
            "who_should_avoid": str(row.get("who_should_avoid", "")) if "who_should_avoid" in inci_df.columns else "",
            "url": str(row.get("url", "")) if "url" in inci_df.columns else "",
        }

        keys = set()
        if rec["name"]:
            keys.add(norm(rec["name"]))
        if rec["scientific_name"]:
            keys.add(norm(rec["scientific_name"]))

        for k in keys:
            if k and k not in lookup:
                lookup[k] = rec

    return lookup


# -------------------------
# 3) Match a product’s ingredients to KB rows
# -------------------------
def match_ingredients(tokens: list[str], lookup: dict[str, dict]) -> list[dict]:
    matched = []
    seen = set()

    for t in tokens:
        rec = lookup.get(t)
        if rec:
            cname = rec["name"]
            if cname not in seen:
                matched.append(rec)
                seen.add(cname)

    return matched


def shorten(s: str, max_chars: int = 140) -> str:
    s = "" if s is None else str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_chars] + ("…" if len(s) > max_chars else "")


# -------------------------
# 4) Build ONE enriched document per product
# -------------------------
def build_product_document(row: pd.Series, matched_recs: list[dict], max_ing_lines: int = 10) -> str:
    """
    Enriched product doc = product info + compressed ingredient function lines.
    Keep it short: vector search likes signal > noise.
    """
    name = row.get("name", "")
    brand = row.get("brand", "")
    label = row.get("Label", "")  # this is product type in your file
    price = row.get("price_usd", None)
    rank = row.get("rank", "")

    # a compact ingredient insight section
    lines = []
    for rec in matched_recs[:max_ing_lines]:
        lines.append(
            f"- {rec['name']}: {shorten(rec.get('what_does_it_do', ''), 140)}"
        )

    inci_raw = row.get("ingredients", "")

    doc = f"""Product: {name}
Brand: {brand}
Type: {label}
Price: {price}
Rank: {rank}

Ingredient insights (from ingredientsList.csv):
{chr(10).join(lines) if lines else "- No ingredient matches found in KB."}

Raw INCI list (from product dataset):
{inci_raw}
"""
    return doc


# -------------------------
# 5) Main: load → enrich → store in Chroma
# -------------------------
def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"

    products_path = data_dir / "cosmetic_p.csv"
    inci_path = data_dir / "ingredientsList.csv"

    # Products: use utf-8-sig to handle BOM seen in file header
    products = pd.read_csv(products_path, encoding="utf-8-sig")
    inci_kb = pd.read_csv(inci_path, encoding="utf-8-sig")

    # Basic normalization
    products["price_usd"] = products["price"].apply(price_to_float)

    # Build ingredient lookup
    lookup = build_ingredient_lookup(inci_kb)

    docs = []
    for i, row in products.iterrows():
        # Parse product ingredients
        tokens = parse_inci_list(row.get("ingredients", ""))

        # Match to ingredient KB
        matched = match_ingredients(tokens, lookup)

        # Build enriched text doc
        text = build_product_document(row, matched_recs=matched)

        # Metadata for filtering + traceability
        metadata = {
            "doc_type": "product",
            "row_id": int(i),
            "name": row.get("name", ""),
            "brand": row.get("brand", ""),
            "product_type": row.get("Label", ""),
            "price_usd": row.get("price_usd", None),
            "rank": row.get("rank", None),
            "skin_flags": {
                "Combination": int(row.get("Combination", 0)),
                "Dry": int(row.get("Dry", 0)),
                "Normal": int(row.get("Normal", 0)),
                "Oily": int(row.get("Oily", 0)),
                "Sensitive": int(row.get("Sensitive", 0)),
            },
            "matched_ingredients": [m["name"] for m in matched],
        }

        docs.append(Document(page_content=text, metadata=metadata))

    # Vector DB
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("📚 Creating FAISS vector store...")
    vectordb = FAISS.from_documents(docs, emb)

    # persist locally
    save_dir = repo_root / "faiss_db"
    vectordb.save_local(str(save_dir))
    print("✅ FAISS saved at:", save_dir)


if __name__ == "__main__":
    main()