from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

repo_root = Path(__file__).resolve().parents[1]
save_dir = repo_root / "faiss_db"

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(str(save_dir), emb, allow_dangerous_deserialization=True)

print("✅ Loaded FAISS DB")
print("Total vectors:", db.index.ntotal)

docs = list(db.docstore._dict.values())  # all stored Documents
n = len(docs)
with_match = sum(1 for d in docs if d.metadata.get("matched_ingredients"))
zero_match = n - with_match

print("\n--- MERGE COVERAGE ---")
print("Total products:", n)
print("Products with >=1 ingredient match:", with_match, f"({with_match/n:.1%})")
print("Products with 0 matches:", zero_match, f"({zero_match/n:.1%})")

print("\n--- SAMPLE ZERO-MATCH PRODUCTS ---")
count = 0
for d in docs:
    if not d.metadata.get("matched_ingredients"):
        print("-", d.metadata.get("brand"), "|", d.metadata.get("name"))
        count += 1
        if count == 10:
            break

lengths = [len(d.page_content) for d in docs]

print("\n--- DOC LENGTHS ---")
print("Min chars:", min(lengths))
print("Median chars:", sorted(lengths)[len(lengths)//2])
print("Max chars:", max(lengths))

# show unusually large docs
print("\n--- TOP 5 LONGEST DOCS ---")
top = sorted(docs, key=lambda d: len(d.page_content), reverse=True)[:5]
for d in top:
    print(len(d.page_content), "|", d.metadata.get("brand"), "|", d.metadata.get("name"))

def test_query(q, k=5):
    print("\n======================")
    print("QUERY:", q)
    results = db.similarity_search_with_score(q, k=k)
    for rank, (doc, score) in enumerate(results, start=1):
        md = doc.metadata
        print(f"\n#{rank} score={score:.4f}")
        print("Type:", md.get("product_type"), "| Price:", md.get("price_usd"))
        print("Brand:", md.get("brand"))
        print("Name:", md.get("name"))
        print("Matched ingredients (count):", len(md.get("matched_ingredients", [])))

# Try a few
test_query("cleanser for oily skin")
test_query("moisturizer for dry sensitive skin")
test_query("sunscreen for sensitive skin")
test_query("serum for hyperpigmentation")

def eval_skin_retrieval(skin="Oily", k=10, n_queries=50):
    # pick some products that are labeled for that skin type
    candidates = [d for d in docs if d.metadata.get("skin_flags", {}).get(skin, 0) == 1]
    if len(candidates) == 0:
        print("No candidates found for skin:", skin)
        return

    # use product names as pseudo-queries (simple baseline)
    # better: create templated queries, but this works as a quick signal
    samples = candidates[:n_queries]

    hits = 0
    for d in samples:
        q = f"best skincare product for {skin.lower()} skin"
        results = db.similarity_search(q, k=k)
        # compute fraction of retrieved results that have that skin flag
        retrieved = sum(1 for r in results if r.metadata.get("skin_flags", {}).get(skin, 0) == 1)
        if retrieved >= (k // 2):  # arbitrary threshold
            hits += 1

    print(f"\n--- RETRIEVAL EVAL: {skin} ---")
    print("Queries tested:", len(samples))
    print("Hit rate:", hits / len(samples))

eval_skin_retrieval("Oily", k=10, n_queries=30)
eval_skin_retrieval("Sensitive", k=10, n_queries=30)