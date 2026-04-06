# Import required libraries
from pathlib import Path
from dotenv import load_dotenv
from llm_summary import generate_product_explanation
import streamlit as st
import time # Import time for simulating loading

# LangChain tools for vector search
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Configure Streamlit page settings
st.set_page_config(page_title="Skincare Recommender", layout="wide")

# Main app title
st.title("✨ Personalized Skincare Routine Recommender")


# Cache the database loading so FAISS index loads only once
@st.cache_resource
def load_db():
    # Find project root folder
    repo_root = Path(__file__).resolve().parents[1]
    save_dir = repo_root / "faiss_db"

    # Load embedding model used during indexing
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load FAISS vector database
    db = FAISS.load_local(str(save_dir), emb, allow_dangerous_deserialization=True)

    return db


# Initialize FAISS database
db = load_db()


# Sidebar UI for collecting user preferences
st.sidebar.header("Your Preferences")

# Age group selection
age_group = st.sidebar.selectbox(
    "Your Age Group",
    ["Teens / 20s", "30s", "40s", "50s+", "Not specified"],
    index=4
)

# Skin type selection
skin_type = st.sidebar.selectbox(
    "Skin Type",
    ["Oily", "Dry", "Normal", "Combination", "Sensitive"]
)

# Budget filter
budget = st.sidebar.slider("Max Budget ($)", 5, 200, 50)

# Optional skincare concern input
concern = st.sidebar.text_input(
    "Main Concern (optional)",
    placeholder="acne, hyperpigmentation, dryness..."
)

# Choose retrieval strategy
retrieval_mode = st.sidebar.selectbox(
    "Retrieval strategy",
    [
        "Baseline: main + fallback",
        "Step-wise only",
        "Ingredient-boosted",
    ],
    index=0,
)

# Number of initial products retrieved
k = st.sidebar.slider("Initial retrieval K", 5, 50, 20)

# Number of results used for fallback search
fallback_k = st.sidebar.slider("Fallback retrieval K", 10, 80, 40)

# Toggle debug mode
show_debug = st.sidebar.checkbox("Show debug (raw retrieved results)")


# Routine configuration section
st.sidebar.divider()
st.sidebar.subheader("Routine Options")

# Choose AM, PM, or both routines
routine_choice = st.sidebar.radio(
    "Which routine?",
    ["Both", "AM only", "PM only"],
    index=0
)

st.sidebar.divider()
st.sidebar.subheader("Clean Beauty Filters")
include_fragrance_free = st.sidebar.checkbox("Fragrance-Free", value=False)
include_sulfate_paraben_free = st.sidebar.checkbox("Sulfate/Paraben-Free", value=False)
include_non_comedogenic = st.sidebar.checkbox("Non-Comedogenic", value=False)

st.sidebar.divider()
st.sidebar.caption("Choose which steps to include:")

include_cleanser = st.sidebar.checkbox("Cleanser", value=True)
include_toner = st.sidebar.checkbox("Toner", value=False)
include_treatment = st.sidebar.checkbox("Treatment/Serum", value=True)
include_moisturizer = st.sidebar.checkbox("Moisturizer", value=True)
include_sunscreen = st.sidebar.checkbox("Sunscreen (AM)", value=True)
include_mask = st.sidebar.checkbox("Mask (PM)", value=False)


# Build semantic search query based on user input
def build_query(skin: str, concern_text: str, age: str) -> str:
    q = f"best skincare products for {skin.lower()} skin"
    if concern_text.strip():
        q += f" for {concern_text.strip().lower()}"

    # Add age-specific keywords
    if age == "Teens / 20s":
        q += " for acne prone or oily skin"
    elif age == "30s":
        q += " for early signs of aging or sun damage"
    elif age == "40s":
        q += " for collagen loss or elasticity concerns"
    elif age == "50s+":
        q += " for hormonal dryness or thin skin"
    return q


# Normalize messy product type labels into standard categories
def normalize_type(ptype: str) -> str:
    p = (ptype or "").strip().lower()

    if "cleanser" in p or "face wash" in p or "wash" in p:
        return "cleanser"
    if "toner" in p:
        return "toner"
    if "serum" in p or "treatment" in p or "essence" in p:
        return "treatment"
    if "moist" in p or "cream" in p or "lotion" in p:
        return "moisturizer"
    if "sun" in p or "spf" in p:
        return "sunscreen"
    if "mask" in p:
        return "mask"

    return "other"


# Filter retrieved products by budget and skin compatibility
def filter_results(results, skin: str, max_budget: float, fragrance_free: bool = False, sulfate_free: bool = False, non_comedogenic: bool = False):

    out = []
    for doc, score in results:
        md = doc.metadata

        # Budget filtering
        price = md.get("price_usd")
        if price is not None and price != "" and price > max_budget:
            continue

        # Skin compatibility filtering
        skin_flags = md.get("skin_flags", {})
        flag = skin_flags.get(skin, None)
        if flag is not None and flag != 1:
            continue

        # Clean Beauty Filtering
        ingredients_text = md.get("ingredients", "").lower()
        
        if fragrance_free and any(f in ingredients_text for f in ["fragrance", "parfum", "linalool", "limonene"]):
            continue
            
        if sulfate_free and any(s in ingredients_text for s in ["sulfate", "paraben", "sls", "sles"]):
            continue
            
        if non_comedogenic and any(c in ingredients_text for c in ["coconut oil", "isopropyl myristate", "cocoa butter"]):
            continue

        out.append((doc, score))

    return out


# Re-rank products by boosting ingredient matches
def rerank_by_ingredient_boost(results, alpha: float = 0.1):

    scored = []

    for doc, score in results:
        matched = doc.metadata.get("matched_ingredients", []) or []

        # Boost score based on number of matched ingredients
        bonus = alpha * min(len(matched), 20)

        new_score = score - bonus

        scored.append((doc, new_score))

    # Sort by best score
    scored.sort(key=lambda x: x[1])

    return scored


# Human-readable labels for routine buckets (used in UI + LLM step text)
STEP_LABEL = {
    "cleanser": "Cleanser",
    "toner": "Toner",
    "treatment": "Treatment / Serum",
    "moisturizer": "Moisturizer",
    "sunscreen": "Sunscreen",
    "mask": "Mask",
}


@st.cache_data(show_spinner=False)
def cached_llm_explanation(
    skin_type,
    concern,
    age_group,
    product_name,
    product_type,
    step,
    price,
    matched_ingredients_tuple,
    excerpt, 
):
    return generate_product_explanation(
        skin_type=skin_type,
        concern=concern,
        age_group=age_group,
        product_name=product_name,
        product_type=product_type,
        step=step,
        price="",
        matched_ingredients=list(matched_ingredients_tuple),
        excerpt="",
    )

def product_card(
    doc,
    score,
    skin_type=None,
    concern=None,
    routine_phase=None,
    bucket=None,
    age_group=None,
):
    """routine_phase is 'AM' or 'PM'; bucket is e.g. cleanser, sunscreen (for LLM context)."""
    md = doc.metadata

    st.subheader(md.get("name", "Unknown product"))

    cols = st.columns(4)

    cols[0].write(f"**Brand:** {md.get('brand', '')}")
    cols[1].write(f"**Type:** {md.get('product_type', '')}")
    cols[2].write(f"**Price:** ${md.get('price_usd', 'N/A')}")
    cols[3].write(f"**Score:** {score:.4f}")

    matched = md.get("matched_ingredients", []) or []

    if routine_phase and bucket:
        llm_step = f"{routine_phase} routine – {STEP_LABEL.get(bucket, bucket.title())}"
    else:
        llm_step = bucket or routine_phase or ""

    try:
        explanation = cached_llm_explanation(
            skin_type=skin_type,
            concern=concern or "",
            age_group=age_group,
            product_name=md.get("name", ""),
            product_type=md.get("product_type", ""),
            step=llm_step,
            price=md.get("price_usd", "N/A"),
            matched_ingredients_tuple=tuple(matched[:12]),
            excerpt=doc.page_content[:500],
        )
        st.write(f"**Why this matches:** {explanation}")
    except Exception as e:
        if matched:
            st.write("**Why this matches:**", ", ".join(matched[:12]) + (" ..." if len(matched) > 12 else ""))
        else:
            st.write("**Why this matches:** Semantic match to your query.")
        st.caption(f"LLM explanation unavailable, showing fallback instead. Error: {e}")

    # Usage tips: need routine_phase AM/PM (not product bucket — that was the previous bug)
    product_type_normalized = normalize_type(md.get("product_type", ""))
    if routine_phase == "AM" and product_type_normalized == "sunscreen":
        st.info("☀️ **Usage Tip (AM):** Remember to reapply sunscreen every 2 hours, especially if outdoors or sweating!")
    elif routine_phase == "PM" and product_type_normalized == "treatment" and any(ing.lower() == "retinol" for ing in matched):
        st.info("🌙 **Usage Tip (PM):** If this treatment contains retinol, use it at night and always follow with SPF in the morning.")
    elif routine_phase == "PM" and product_type_normalized == "mask":
        st.info("🌙 **Usage Tip (PM):** Masks are great for targeted concerns. Follow product instructions for frequency and duration.")

    if matched:
        st.write("**Matched ingredients:** " + ", ".join(matched))

    # Expandable section for document text
    with st.expander("Show product text (doc excerpt)"):
        st.write(doc.page_content[:1200] + ("..." if len(doc.page_content) > 1200 else ""))

    st.markdown("---")


# Select the first product per skincare step
def pick_first_per_bucket(items, bucket_order):

    picked = []
    used = set()

    for bucket in bucket_order:
        for doc, score in items:

            b = normalize_type(doc.metadata.get("product_type", ""))

            if b == bucket and bucket not in used:
                picked.append((doc, score))
                used.add(bucket)
                break

    return picked


# Keywords used for targeted retrieval of each skincare step
STEP_QUERY = {
    "cleanser": "cleanser face wash",
    "toner": "toner",
    "treatment": "serum treatment",
    "moisturizer": "moisturizer cream",
    "sunscreen": "sunscreen spf sun protect",
    "mask": "face mask",
}


# Retrieve products for a specific routine step
def retrieve_for_bucket(db, bucket: str, skin_type: str, concern: str, k: int):

    phrase = STEP_QUERY.get(bucket, bucket)

    q = f"{phrase} for {skin_type.lower()} skin"

    if concern.strip() and bucket in ["cleanser", "treatment", "moisturizer"]:
        q += f" for {concern.strip().lower()}"

    return db.similarity_search_with_score(q, k=k), q


# Build routine and fill missing steps using fallback retrieval
def build_routine_with_fallback(db, base_filtered, step_order, skin_type, concern, budget, fallback_k):

    picks = pick_first_per_bucket(base_filtered, step_order)

    have = {normalize_type(d.metadata.get("product_type", "")) for d, _ in picks}

    missing = [b for b in step_order if b not in have]

    debug_fallback = []

    for bucket in missing:

        step_raw, q = retrieve_for_bucket(db, bucket, skin_type, concern, k=fallback_k)

        step_filtered = filter_results(step_raw, skin_type, budget, include_fragrance_free, include_sulfate_paraben_free, include_non_comedogenic)

        step_pick = pick_first_per_bucket(step_filtered, [bucket])

        if step_pick:
            picks.extend(step_pick)

        debug_fallback.append((bucket, q, len(step_raw), len(step_filtered), bool(step_pick)))

    return picks, debug_fallback


# Define AM routine order
def am_bucket_order():

    order = []

    if include_cleanser: order.append("cleanser")
    if include_toner: order.append("toner")
    if include_treatment: order.append("treatment")
    if include_moisturizer: order.append("moisturizer")
    if include_sunscreen: order.append("sunscreen")

    return order


# Define PM routine order
def pm_bucket_order():

    order = []

    if include_cleanser: order.append("cleanser")
    if include_toner: order.append("toner")
    if include_treatment: order.append("treatment")
    if include_moisturizer: order.append("moisturizer")
    if include_mask: order.append("mask")

    return order


def render_routine(step_order, picks, skin_type=None, concern=None, age_group=None, routine_phase=None):

    by_bucket = {}

    for doc, score in picks:
        b = normalize_type(doc.metadata.get("product_type", ""))
        if b in step_order and b not in by_bucket:
            by_bucket[b] = (doc, score)

    for step_num, bucket in enumerate(step_order, 1):
        st.markdown(f"### Step {step_num}: {STEP_LABEL.get(bucket, bucket.title())}")
        if bucket in by_bucket:
            doc, score = by_bucket[bucket]
            product_card(
                doc,
                score,
                skin_type,
                concern,
                routine_phase,
                bucket,
                age_group,
            )
        else:
            st.write("No product found for this step.")


# Main execution block
if st.button("Generate Recommendations"):
    with st.spinner("Curating your personalized routine..."):
        time.sleep(1) # Simulate some processing time
        main_query = build_query(skin_type, concern, age_group)
        st.write(f"**Main query (for applicable strategies):** {main_query}")

        if retrieval_mode == "Baseline: main + fallback":
            raw = db.similarity_search_with_score(main_query, k=k)
            filtered_results = filter_results(raw, skin_type, budget, include_fragrance_free, include_sulfate_paraben_free, include_non_comedogenic)
        elif retrieval_mode == "Step-wise only":
            raw = []
            filtered_results = []
        else:  # Ingredient-boosted
            raw = db.similarity_search_with_score(main_query, k=max(k, 40))
            filtered_results = filter_results(raw, skin_type, budget, include_fragrance_free, include_sulfate_paraben_free, include_non_comedogenic)
            filtered_results = rerank_by_ingredient_boost(filtered_results)

        if show_debug:
            with st.expander("Raw Retrieved Documents"):
                st.write(raw)

        am_picks, am_debug = build_routine_with_fallback(db, filtered_results, am_bucket_order(), skin_type, concern, budget, fallback_k)
        pm_picks, pm_debug = build_routine_with_fallback(db, filtered_results, pm_bucket_order(), skin_type, concern, budget, fallback_k)

    if routine_choice in ["Both", "AM only"]:
        st.header("☀️ AM Routine")
        if not am_picks:
            st.write("No AM products found. Try increasing budget or increasing K.")
        else:
            render_routine(am_bucket_order(), am_picks, skin_type, concern, age_group, routine_phase="AM")

        if show_debug:
            with st.expander("Fallback debug (AM)"):
                for bucket, q, n_raw, n_filt, picked in am_debug:
                    st.write(f"**{bucket}**: {q} -> {n_raw} raw -> {n_filt} filtered -> Picked: {picked}")

    if routine_choice in ["Both", "PM only"]:
        st.header("🌙 PM Routine")
        if not pm_picks:
            st.write("No PM products found. Try increasing budget or increasing K.")
        else:
            render_routine(pm_bucket_order(), pm_picks, skin_type, concern, age_group, routine_phase="PM")

        if show_debug:
            with st.expander("Fallback debug (PM)"):
                for bucket, q, n_raw, n_filt, picked in pm_debug:
                    st.write(f"**{bucket}**: {q} -> {n_raw} raw -> {n_filt} filtered -> Picked: {picked}")

# Ingredient Glossary (at the bottom of the sidebar or main page)
st.sidebar.divider()
st.sidebar.subheader("Ingredient Glossary")
with st.sidebar.expander("Common Skincare Ingredients"):
    st.markdown("**Niacinamide:** A form of Vitamin B3 that helps reduce inflammation, minimize pores, and improve skin tone.")
    st.markdown("**Hyaluronic Acid:** A powerful humectant that attracts and holds moisture, providing intense hydration to the skin.")
    st.markdown("**Retinol:** A derivative of Vitamin A, known for its anti-aging properties, promoting cell turnover and reducing fine lines.")
    st.markdown("**Vitamin C:** A potent antioxidant that brightens skin, reduces hyperpigmentation, and protects against environmental damage.")
    st.markdown("**Salicylic Acid:** A beta-hydroxy acid (BHA) that exfoliates inside the pore, effective for acne and oily skin.")
    st.markdown("**Glycolic Acid:** An alpha-hydroxy acid (AHA) that exfoliates the skin's surface, improving texture and brightness.")