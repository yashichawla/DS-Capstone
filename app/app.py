# Import required libraries
from pathlib import Path
from dotenv import load_dotenv
from llm_summary import generate_product_explanation
import streamlit as st

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

# Choose which steps to include
st.sidebar.caption("Choose which steps to include:")

include_cleanser = st.sidebar.checkbox("Cleanser", value=True)
include_toner = st.sidebar.checkbox("Toner", value=False)
include_treatment = st.sidebar.checkbox("Treatment/Serum", value=True)
include_moisturizer = st.sidebar.checkbox("Moisturizer", value=True)
include_sunscreen = st.sidebar.checkbox("Sunscreen (AM)", value=True)
include_mask = st.sidebar.checkbox("Mask (PM)", value=False)


# Build semantic search query based on user input
def build_query(skin: str, concern_text: str) -> str:
    q = f"best skincare products for {skin.lower()} skin"
    if concern_text.strip():
        q += f" for {concern_text.strip().lower()}"
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
def filter_results(results, skin: str, max_budget: float):

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

@st.cache_data(show_spinner=False)
def cached_llm_explanation(
    skin_type,
    concern,
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
        product_name=product_name,
        product_type=product_type,
        step=step,
        price="",
        matched_ingredients=list(matched_ingredients_tuple),
        excerpt="",
    )

def product_card(doc, score, skin_type=None, concern=None, step=None):
# Display product details in UI

    md = doc.metadata

    st.subheader(md.get("name", "Unknown product"))

    cols = st.columns(4)

    cols[0].write(f"**Brand:** {md.get('brand', '')}")
    cols[1].write(f"**Type:** {md.get('product_type', '')}")
    cols[2].write(f"**Price:** ${md.get('price_usd', 'N/A')}")
    cols[3].write(f"**Score:** {score:.4f}")

    matched = md.get("matched_ingredients", []) or []

    try:
        explanation = cached_llm_explanation(
            skin_type=skin_type,
            concern=concern or "",
            product_name=md.get("name", ""),
            product_type=md.get("product_type", ""),
            step=step,
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
    # Show ingredient explanation
    matched = md.get("matched_ingredients", [])

    if matched:
        st.write("**Why this matches:**", ", ".join(matched[:12]) + (" ..." if len(matched) > 12 else ""))
    else:
        st.write("**Why this matches:** Semantic match to your query.")

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

        step_filtered = filter_results(step_raw, skin_type, budget)

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


# Human readable labels for each routine step
STEP_LABEL = {
    "cleanser": "Cleanser",
    "toner": "Toner",
    "treatment": "Treatment / Serum",
    "moisturizer": "Moisturizer",
    "sunscreen": "Sunscreen",
    "mask": "Mask",
}

def render_routine(step_order, picks, skin_type=None, concern=None):

    by_bucket = {}

    for doc, score in picks:
        b = normalize_type(doc.metadata.get("product_type", ""))

        if b not in by_bucket:
            by_bucket[b] = (doc, score)

    step_num = 1

    for bucket in step_order:

        if bucket not in by_bucket:
            continue

        doc, score = by_bucket[bucket]
        st.markdown(f"### Step {step_num}: {STEP_LABEL.get(bucket, bucket.title())}")
        product_card(doc, score, skin_type=skin_type, concern=concern, step=bucket)
        step_num += 1

        st.markdown(f"### Step {step_num}: {STEP_LABEL.get(bucket, bucket.title())}")

if st.button("Generate Recommendations"):
    main_query = build_query(skin_type, concern)
    st.write(f"**Main query (for applicable strategies):** {main_query}")

    if retrieval_mode == "Baseline: main + fallback":
        raw = db.similarity_search_with_score(main_query, k=k)
        filtered = filter_results(raw, skin_type, budget)
    elif retrieval_mode == "Step-wise only":
        raw = []
        filtered = []
    else:  # Ingredient-boosted
        raw = db.similarity_search_with_score(main_query, k=max(k, 40))
        base_filtered = filter_results(raw, skin_type, budget)
        filtered = rerank_by_ingredient_boost(base_filtered)

    if show_debug:
        st.markdown("### Debug: Main Retrieval (top 10)")
        st.write(f"Strategy: {retrieval_mode}")
        st.write(f"Raw retrieved: {len(raw)} | After filters: {len(filtered)}")
        for d, s in raw[:10]:
            st.write(f"- {d.metadata.get('name')} | {d.metadata.get('product_type')} | ${d.metadata.get('price_usd')} | score={s:.4f}")

    show_am = routine_choice in ["Both", "AM only"]
    show_pm = routine_choice in ["Both", "PM only"]

    am_steps = am_bucket_order()
    pm_steps = pm_bucket_order()

    if show_am and len(am_steps) == 0:
        st.warning("Select at least one AM step in the sidebar (e.g., Cleanser / Moisturizer / Sunscreen).")
        st.stop()
    if show_pm and len(pm_steps) == 0:
        st.warning("Select at least one PM step in the sidebar (e.g., Cleanser / Treatment / Moisturizer).")
        st.stop()

    if show_am and show_pm:
        col_am, col_pm = st.columns(2)

        with col_am:
            st.markdown("## 🌞 AM Routine")
            am_picks, am_debug = build_routine_with_fallback(
                db, filtered, am_steps, skin_type, concern, budget, fallback_k
            )
            if not am_picks:
                st.write("No AM products found. Try increasing budget or increasing K.")
            else:
                render_routine(am_steps, am_picks, skin_type=skin_type, concern=concern)

            if show_debug:
                with st.expander("Fallback debug (AM)"):
                    for bucket, q, n_raw, n_filt, picked in am_debug:
                        st.write(f"- **{bucket}** | q='{q}' | raw={n_raw} filt={n_filt} picked={picked}")

        with col_pm:
            st.markdown("## 🌙 PM Routine")
            pm_picks, pm_debug = build_routine_with_fallback(
                db, filtered, pm_steps, skin_type, concern, budget, fallback_k
            )
            if not pm_picks:
                st.write("No PM products found. Try increasing budget or increasing K.")
            else:
                render_routine(pm_steps, pm_picks, skin_type=skin_type, concern=concern)

            if show_debug:
                with st.expander("Fallback debug (PM)"):
                    for bucket, q, n_raw, n_filt, picked in pm_debug:
                        st.write(f"- **{bucket}** | q='{q}' | raw={n_raw} filt={n_filt} picked={picked}")

    else:
        if show_am:
            st.markdown("## 🌞 AM Routine")
            am_picks, am_debug = build_routine_with_fallback(
                db, filtered, am_steps, skin_type, concern, budget, fallback_k
            )
            if not am_picks:
                st.write("No AM products found. Try increasing budget or increasing K.")
            else:
                render_routine(am_steps, am_picks, skin_type=skin_type, concern=concern)

            if show_debug:
                with st.expander("Fallback debug (AM)"):
                    for bucket, q, n_raw, n_filt, picked in am_debug:
                        st.write(f"- **{bucket}** | q='{q}' | raw={n_raw} filt={n_filt} picked={picked}")

        if show_pm:
            st.markdown("## 🌙 PM Routine")
            pm_picks, pm_debug = build_routine_with_fallback(
                db, filtered, pm_steps, skin_type, concern, budget, fallback_k
            )
            if not pm_picks:
                st.write("No PM products found. Try increasing budget or increasing K.")
            else:
                render_routine(pm_steps, pm_picks, skin_type=skin_type, concern=concern)

            if show_debug:
                with st.expander("Fallback debug (PM)"):
                    for bucket, q, n_raw, n_filt, picked in pm_debug:
                        st.write(f"- **{bucket}** | q='{q}' | raw={n_raw} filt={n_filt} picked={picked}")
        product_card(doc, score)

        step_num += 1
