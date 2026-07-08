# 🧳 Customer Behaviour Analysis — Tourism Dataset

## 1. Project Overview

This project is a comprehensive, hands-on demonstration of the **data preprocessing and feature engineering** skills that precede any machine learning workflow. Working with a real tourism customer-behavior dataset, it covers exploratory data analysis, missing value handling, visualization, outlier detection, and three distinct encoding and scaling strategies — each implemented as a reusable function rather than one-off code.

Unlike a typical end-to-end ML project, this notebook does not train a predictive model. Its purpose is to demonstrate mastery of the preprocessing toolkit itself: the techniques that determine whether a downstream model succeeds or fails.

---

## 2. Dataset

| Attribute | Value |
|---|---|
| File | `Customer behaviour Tourism.csv` |
| Rows | 11,760 |
| Columns | 17 |
| Domain | Tourism / travel-page engagement behavior |
| Total missing values | 1,432 (across 6 columns) |

**Key columns:**

| Column | Description |
|---|---|
| `Taken_product` | Target-like variable — whether the customer purchased the travel product (Yes/No) |
| `Yearly_avg_view_on_travel_page` | Average yearly views on the travel page |
| `total_likes_on_outstation_checkin_given` | Total likes given on outstation check-ins |
| `preferred_device` | Device used to access the platform |
| `member_in_family` | Number of family members |
| `preferred_location_type` | Preferred travel destination type |
| `working_flag` | Whether the customer is employed |
| `Adult_flag` | Whether the customer is an adult |
| `travelling_network_rating` | Self-reported network/social rating related to travel |

**Note:** The dataset file is not included in this repository. To run the notebook, place `Customer behaviour Tourism.csv` in the same directory.

---

## 3. Methodology

```
Raw Data (11,760 × 17)
     │
     ▼
Initial Data Overview (shape, dtypes, head/tail, describe)
     │
     ▼
Automatic Column Classification (categorical / numerical / cardinal)
     │
     ▼
Categorical Variable Analysis
     │
     ▼
Numerical Variable Analysis
     │
     ▼
Missing Value Analysis & Imputation
     │
     ▼
Data Visualization (10+ charts)
     │
     ▼
Outlier Detection (IQR method)
     │
     ▼
Encoding (Label / One-Hot / Rare)
     │
     ▼
Feature Scaling (Standard / Robust / MinMax)
```

---

## 4. Initial Data Overview

A custom `check_df()` function was written to produce a single, consistent diagnostic report for any DataFrame — shape, dtypes, head, tail, missing values, and descriptive statistics — avoiding the need to call five separate pandas methods every time the dataset needs to be inspected.

```python
def check_df(dataframe, head=7):
    print("Shape:", dataframe.shape)
    print("Types:", dataframe.dtypes)
    print("Head:", dataframe.head(head))
    print("Tail:", dataframe.tail(head))
    print("NA:", dataframe.isnull().sum())
    print("Quantiles:", dataframe.describe().T)
```

---

## 5. Automatic Column Classification

Rather than manually listing which columns are categorical vs. numerical, a reusable `grab_col_names()` function classifies every column automatically based on dtype and cardinality thresholds:

- **`cat_cols`** — object/category/bool dtype, **or** numeric with fewer than 10 unique values (e.g., a 1–5 rating stored as an integer is still conceptually categorical)
- **`num_cols`** — numeric dtype, excluding anything already classified as categorical
- **`cat_but_car`** — "categorical but cardinal": object-type columns with more than 20 unique values (effectively free-text/ID-like columns that shouldn't be treated as ordinary categories)

**Result on this dataset:**

| Type | Count | Columns |
|---|---|---|
| Categorical | 7 | `Taken_product`, `preferred_device`, `member_in_family`, `preferred_location_type`, `following_company_page`, `working_flag`, `Adult_flag` |
| Numerical | 4 | `Yearly_avg_view_on_travel_page`, `total_likes_on_outstation_checkin_given`, `Yearly_avg_comment_on_travel_page`, `Daily_Avg_mins_spend_on_traveling_page` |
| Categorical-but-cardinal | 1 | (high-cardinality column excluded from standard categorical analysis) |

---

## 6. Categorical Variable Analysis

A `cat_summary()` function reports the value counts and percentage share of every category, with an optional bar-chart visualization. Applied across all 7 categorical columns.

**Notable finding:** `Taken_product` (whether the customer purchased) shows a **class imbalance**: 9,864 "No" vs. 1,896 "Yes" — only **16.1%** of customers converted. This is an important consideration for anyone building a classifier on this data later — a naive "always predict No" baseline would already score ~84% accuracy, so accuracy alone would be a misleading metric here.

**Data quality note:** `working_flag` was found to contain not just `"Yes"` and `"No"`, but also a single stray value of `"0"` (84.62% No, 15.37% Yes, 0.01% the anomalous "0"). This kind of small, easy-to-miss inconsistency is exactly what a `cat_summary()` sweep across every column is designed to catch before it silently corrupts downstream encoding.

---

## 7. Numerical Variable Analysis

A `num_summary()` function prints an extended set of quantiles (5th through 99th percentile, not just the default quartiles) for every numerical column, with an optional histogram.

**Example — `total_likes_on_outstation_checkin_given`:** average ≈ 28,170 likes, with the median close to the mean — indicating a fairly balanced (not heavily skewed) distribution for this particular variable.

---

## 8. Missing Value Analysis & Imputation

| Column | Missing count |
|---|---|
| `Yearly_avg_view_on_travel_page` | 581 |
| `total_likes_on_outstation_checkin_given` | 381 |
| `Yearly_avg_comment_on_travel_page` | 206 |
| `following_company_page` | 103 |
| `yearly_avg_Outstation_checkins` | 75 |
| `preferred_device` | 53 |
| `preferred_location_type` | 31 |
| **Total** | **1,432** |

Two imputation strategies were compared for numeric columns — **mean** and **median** — before settling on a consistent rule: numeric columns are filled with their mean, categorical columns with their mode (most frequent value). This mirrors a standard, defensible imputation approach for a dataset of this size, where dropping rows with missing values would have discarded a meaningful fraction of the data.

---

## 9. Data Visualization

Over 10 visualizations were built with `matplotlib` and `seaborn`, including:

- **Scatter plots** — yearly page views vs. daily minutes spent traveling; total likes given vs. received; family size vs. yearly page views
- **Bar charts** — average travel-page views by preferred device; average views by preferred location type
- **Count plots** — working status split by product purchase (`Taken_product`); preferred device distribution
- **Histograms** — distribution of likes received on outstation check-ins (right-skewed — most users receive low-to-medium likes, a smaller number receive very high likes); family size distribution
- **Box plots** — outlier inspection for yearly page views, yearly comments, daily minutes spent, and location type vs. page views
- **Line plot** — travel-page views trend against weeks since last outstation check-in
- **Correlation heatmap** — relationships across all numeric variables

---

## 10. Outlier Detection

A full custom outlier-detection toolkit was built from scratch, rather than relying on a single library call:

| Function | Purpose |
|---|---|
| `outlier_thresholds()` | Computes IQR-based lower/upper bounds for any column |
| `check_outlier()` | Returns `True`/`False` — does this column contain at least one outlier? |
| `grab_outliers()` | Returns the actual outlier rows (optionally just their index positions) |

Applied to `total_likes_on_outstation_checkin_given` and `Yearly_avg_comment_on_travel_page`, both of which were confirmed to contain outliers using this toolkit.

---

## 11. Encoding Techniques

Three distinct encoding strategies were implemented and compared, each suited to a different type of categorical variable:

### 11.1 Label Encoding
Applied to **binary** columns (exactly 2 unique values) — in this dataset, `Taken_product` (Yes/No → 1/0). A reusable `label_encoder()` function automatically detects all binary columns and encodes them in a loop, rather than hardcoding each column by name.

### 11.2 One-Hot Encoding
Applied to columns with low-to-moderate cardinality (2–10 unique values, e.g. `preferred_device`, `member_in_family`, `working_flag`). Three variants were explicitly compared:
- Standard one-hot (`pd.get_dummies`)
- `drop_first=True` — avoids the dummy-variable trap (multicollinearity) by dropping one redundant category
- `dummy_na=True` — creates an explicit "missing" indicator column instead of silently dropping NaN rows

### 11.3 Rare Encoding
For categories that appear too infrequently to be statistically meaningful on their own, a `rare_encoder()` function groups any category below a chosen frequency threshold (e.g. 1%) into a single `"Rare"` bucket. This was paired with a `rare_analyser()` function that reports, for every categorical column, how each category's frequency relates to the target variable — useful for deciding *which* rare categories are safe to merge without losing predictive signal.

---

## 12. Feature Scaling

Three scaling methods were implemented and compared side-by-side on the same column, with their underlying formulas documented directly in the notebook:

| Method | Formula | Notes |
|---|---|---|
| **StandardScaler** | `z = (x − μ) / σ` | Centers to mean 0, unit variance — sensitive to outliers |
| **RobustScaler** | `r = (x − median) / IQR` | Uses median and interquartile range — much less sensitive to outliers |
| **MinMaxScaler** | `x_scaled = (x − min) / (max − min)` | Rescales to a fixed [0, 1] range |

Documenting all three side-by-side (rather than picking one) demonstrates an understanding of *when* each is appropriate — for instance, `RobustScaler` is generally preferable over `StandardScaler` on columns like `total_likes_on_outstation_checkin_given`, which contain confirmed outliers.

---

## 13. Key Insights

- **Class imbalance in the target-like variable:** only 16.1% of customers took the product, which would need to be accounted for (e.g. via class weighting or resampling) in any future predictive model built on this data
- **A data quality anomaly was caught, not missed:** the stray `"0"` value in `working_flag` demonstrates the value of running `cat_summary()` across *every* categorical column rather than spot-checking a few
- **Right-skewed engagement metrics:** likes received on check-ins are not normally distributed — most users cluster at low-to-medium engagement, with a long tail of highly engaged users
- **Missing data is systematic, not random:** the six columns with missing values are concentrated in engagement/behavior metrics rather than demographic fields, suggesting the missingness may be related to *how* certain users interact with the platform (worth investigating further before assuming missing-completely-at-random)

---

## 14. Repository Structure

```
data-analysis-project/
├── README.md
└── Project.ipynb      ← Full EDA, cleaning, and feature engineering notebook
```

**Note:** `Customer behaviour Tourism.csv` (the raw dataset) is not included in the repository.

---

## 15. Setup & Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook Project.ipynb
```

Place `Customer behaviour Tourism.csv` in the same directory as the notebook before running it.

---

## 16. Tech Stack

**Data processing:** Python, pandas, NumPy
**Visualization:** Matplotlib, Seaborn
**Preprocessing:** scikit-learn (`LabelEncoder`, `StandardScaler`, `RobustScaler`, `MinMaxScaler`)
**Environment:** Jupyter Notebook

---

## 17. Scope & Limitations

This notebook is intentionally scoped as a **preprocessing and feature engineering exercise**, not a full predictive modeling project:

- No model is trained or evaluated — the pipeline stops at the point where the data is fully cleaned, encoded, and scaled, ready to be fed into a model
- The three encoding methods (Label, One-Hot, Rare) and three scaling methods (Standard, Robust, MinMax) are demonstrated independently for comparison, rather than combined into a single final preprocessing pipeline — a natural next step would be selecting one consistent combination and applying it end-to-end
- Given the confirmed class imbalance in `Taken_product`, a logical extension of this work would be building a classifier (e.g. Logistic Regression, Random Forest, or XGBoost) to predict product purchase, using the cleaned and encoded features produced here
