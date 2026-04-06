# DS5500 Data Science Capstone - Skincare Product Recommendation System

## Project Overview

This project proposes a deep learning-based skincare product recommendation system that analyzes facial skin conditions and cosmetic ingredient lists to provide personalized product recommendations. Unlike traditional recommender systems that rely solely on user reviews or ratings, this approach focuses on objective skin features and ingredient efficacy to improve personalization and recommendation accuracy.

### Key Objectives
- **Ingredient Analysis**: Process cosmetic ingredient lists as sequential data using deep learning to extract meaningful product attributes
- **Personalized Recommendations**: Match detected skin concerns with ingredient efficacy to rank suitable products
- **Explainability**: Provide transparent recommendations by showing which ingredients address specific skin concerns

## Datasets

This project integrates multiple publicly available skincare datasets from Kaggle to construct a comprehensive, ingredient-aware product representation suitable for machine learning-based recommendation modeling.

### 1. Skincare Product Ingredients (Sephora)

**Source**: [Kaggle - Skincare Product Ingredients](https://www.kaggle.com/datasets/dominoweir/skincare-product-ingredients)

**Description**: Contains 1,472 skincare products, all explicitly categorized as skincare, with clearly defined product types.

**Key Features**:
- Product name
- Brand
- Product type (cleanser, serum, moisturizer, etc.)
- Ingredients (raw text)
- Price
- Rating/rank
- Skin type compatibility (Combination, Dry, Normal, Oily, Sensitive)

**File**: `data/cosmetic_p.csv`

### 2. Skin Care Product Ingredients – INCI List

**Source**: [Kaggle - Skin Care Product Ingredients INCI List](https://www.kaggle.com/datasets/amaboh/skin-care-product-ingredients-inci-list)

**Description**: Structured list of skincare ingredients using INCI (International Nomenclature of Cosmetic Ingredients) naming conventions.

**Key Features**:
- Standardized ingredient names
- Scientific names
- Ingredient descriptions
- What it is / What it does
- Who it's good for / Who should avoid

**File**: `data/ingredientsList.csv`

### Dataset Integration Strategy

The project uses two datasets with complementary roles:

- **Sephora Skincare Product Dataset**: Primary dataset providing product identity, product type, and raw ingredient lists
- **INCI Ingredient Dataset**: Reference dataset for ingredient normalization, standardization, and enrichment

The INCI dataset is not merged directly at the product row level but is used to enrich and normalize ingredient representations extracted from the product dataset.

## Project Structure

```
DS-Capstone/
├── app/
│   ├── app.py              # Streamlit UI: retrieval, routines, product cards
│   └── llm_summary.py      # Groq (OpenAI-compatible) API: per-product “why this matches” text
├── data/
│   ├── cosmetic_p.csv      # Sephora skincare products dataset
│   ├── ingredientsList.csv # INCI ingredient reference dataset
│   └── README.md           # Data directory documentation
├── notebook/
│   ├── cosmetic_brand_label.ipynb  # Product dataset EDA
│   ├── ingredients_eda.ipynb       # Ingredients dataset EDA
│   └── README.md
├── src/
│   ├── build_vectordb.py   # Build FAISS index + enriched product documents
│   └── eval_vectordb.py    # Optional checks / sanity queries on the index
├── faiss_db/               # Generated locally (not in git); see below
├── requirements.txt
└── README.md               # This file
```

`faiss_db/` is created when you run `python src/build_vectordb.py` and should stay out of version control (see **Important** at the end).

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DS-Capstone
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download datasets from Kaggle and place them in the `data/` directory:
   - `cosmetic_p.csv` → `data/cosmetic_p.csv`
   - `ingredientsList.csv` → `data/ingredientsList.csv`

### Exploratory Data Analysis

Start by exploring the datasets using the provided notebooks:

1. **Product Dataset EDA**: `notebook/cosmetic_brand_label.ipynb`
   - Dataset overview (shape, columns, missing values)
   - Product type and brand distributions
   - Price and rating analysis
   - Skin type compatibility analysis
   - Correlation analysis

2. **Ingredients Dataset EDA**: `notebook/ingredients_eda.ipynb`
   - Text analysis of ingredient descriptions
   - Word frequency analysis
   - Condition extraction ("who is it good for", "who should avoid")
   - Skincare term/buzzword analysis
  
# 🔎 Vector Database & Streamlit App

## Building the FAISS Vector Database

Before running the app, generate the semantic vector store:

```bash
pip install -r requirements.txt
pip install streamlit
python src/build_vectordb.py
```

This creates a local `faiss_db/` directory containing the vector index used for semantic product retrieval.

⚠️ Note: The `faiss_db/` folder is not committed to GitHub and must be generated locally.

---

## 🚀 Running the Streamlit Application

From the project root (with your virtual environment activated):

```bash
streamlit run app/app.py
```

### Optional: LLM explanations (Groq)

Per-product **“Why this matches”** text is generated by `app/llm_summary.py` using the [Groq](https://groq.com/) OpenAI-compatible API (`llama-3.1-8b-instant` by default). The prompt includes the user’s skin type, concern, age group, routine step (AM/PM), product name and type, and matched ingredients (the module can also include a short product excerpt when wired through).

1. Create a Groq API key and add it to a **local** `.env` in the project root (never commit this file):

   ```text
   GROQ_API_KEY=your_key_here
   ```

2. `app/app.py` loads `.env` via `python-dotenv`. If `GROQ_API_KEY` is missing or the call fails, the UI falls back to a short explanation from **matched ingredients** or a generic semantic-match message.

---

## ✨ Streamlit Features

The application supports:

- **User profile**: skin type, optional main concern, **age group** (teens/20s through 50s+), and age-aware wording in the main semantic query  
- **AM and/or PM** routine generation with configurable steps (cleanser, toner, treatment, moisturizer, sunscreen, mask)  
- **Budget** and **skin-type** filtering on retrieved products  
- **Clean beauty filters** (optional): fragrance-free, sulfate/paraben-free, and non-comedogenic heuristics based on ingredient text in metadata  
- **FAISS** semantic retrieval over enriched product documents (see `src/build_vectordb.py`)  
- **Multiple retrieval / RAG-style strategies** (sidebar) for comparing pipelines  
- **Fallback retrieval** per step when the main pool does not cover every selected routine step  
- **LLM explanations**: Groq-backed, 2-sentence explanations per product (cached in-session); fallback if the API is unavailable  
- **Matched ingredients** shown as a **comma-separated** list on each product card  
- **Contextual usage tips** (e.g., sunscreen reapplication, retinol at night) where relevant  
- **Debug mode**: optional raw retrieval listings for troubleshooting  

---

### Retrieval strategies (for comparison experiments)

To make it easy to compare different retrieval formulations, the Streamlit sidebar exposes three strategies under **“Retrieval strategy”**:

- **Baseline: main + fallback**  
  - Builds one natural-language query from your skin type, concern, and **age group** (extra phrases for teens, 30s, 40s, 50s+).  
  - Retrieves a candidate pool via FAISS, filters by budget and skin-type flags, then picks one product per routine step (cleanser, treatment, etc.).  
  - For any missing steps, issues **step-specific fallback queries** (e.g., *“cleanser face wash for oily skin for acne”*) so the routine is as complete as possible.

- **Step-wise only**  
  - Skips the global query entirely and instead issues **independent queries per step** (cleanser, toner, treatment, moisturizer, sunscreen/mask).  
  - Each query is tailored to the step plus your skin type and concern, which often improves coverage and diversity across steps.  
  - Useful for analyzing how well the vector store supports fine-grained, step-level retrieval without relying on an overall “best products” query.

- **Ingredient-boosted**  
  - Starts from the same global query as the baseline but retrieves a slightly larger pool of candidates.  
  - After budget and skin-type filtering, it **re-ranks results to favor products with richer ingredient matches** (based on the merged INCI metadata).  
  - This lets you study whether explicitly rewarding ingredient coverage leads to more clinically-aligned or interpretable recommendations compared to pure semantic similarity.


# 🔒 Important

- **Secrets**: Do not commit `.env` or any file containing `GROQ_API_KEY` (or other API keys). GitHub push protection may block pushes if secrets appear in history.

Make sure the following entries exist in `.gitignore`:

```text
faiss_db/
.venv/
__pycache__/
.env
```
