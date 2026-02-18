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
├── data/                    # Dataset files
│   ├── cosmetic_p.csv      # Sephora skincare products dataset
│   ├── ingredientsList.csv # INCI ingredient reference dataset
│   └── README.md           # Data directory documentation
├── notebook/                # Jupyter notebooks for EDA and analysis
│   ├── cosmetic_brand_label.ipynb  # Product dataset EDA
│   ├── ingredients_eda.ipynb       # Ingredients dataset EDA
│   └── README.md           # Notebooks directory documentation
├── src/                     # Source code (to be added)
├── models/                  # Trained models (to be added)
├── results/                 # Analysis results and visualizations (to be added)
└── README.md               # This file
```

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