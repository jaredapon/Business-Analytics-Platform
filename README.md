## Capstone Business Analytics Project

Pipeline for restaurant Excel sales data  
ETL + Market Basket Analysis + Holt-Winter's Forecasting Method with PED and Log-Log Model + Prescriptive Analytics

---

## Repository Structure

```
raw_sales/                # Raw Sales Transaction List Excel exports
raw_sales_by_product/     # Raw Sales Report by Product Excel exports
etl.py                    # ETL pipeline
mba.py                    # Market Basket Analysis (FP-Growth)
models/                   # Forecasting & descriptive analytics scripts
cleaned_data/             # Intermediate cleaned CSVs
etl_dimensions/           # Final output dimensions
mba_results/              # Market basket analysis results
```

---

## 1. ETL Pipeline (`etl.py`)

**Phases:** Extract → Transform → Load (with enhanced numeric coercion & outlier auditability)

- **Extract:**  
  Reads all `.xlsx` files in `raw_sales/` and `raw_sales_by_product/`.  
  Auto-detects header row (scans first 10 rows for tokens like `date`, `receipt`, `product`, etc.; defaults to row 4 if not found).  
  Normalizes columns across files; fills missing columns with `None`.  
  Standardizes "Take Out" flag (`Y` → `True`, blanks → `False`).

- **Transform:**  
  Cleans columns/rows (drops unused admin fields, removes blank `Date`/`Time`).  
  Normalizes product naming/IDs (first ID per canonical name).  
  Uppercases product names & standardizes receipt key.  
  Merges Sales Transaction List ↔ Sales by Product on `Receipt No`.  
  Performs numeric coercion on key metrics (`Qty`, `Price`, `Line Total`, `Net Total`) and logs coercion-induced NaNs.  
  Flags outliers using enhanced IQR logic (details below) before optional row removal.

**Outlier Detection & Removal (Overview):**

| Aspect | Logic |
| ------ | ----- |
| Method | IQR (Tukey fences) per metric (`Qty`, `Price`, `Line Total`, `Net Total` if present) |
| Per-Feature k | Custom k overrides (e.g. `{'Qty':1.2,'Price':2.0}`) fall back to global default (1.5) |
| Category-Aware | Optional segmentation: bounds computed within each `CATEGORY` to avoid cross-category skew |
| Log-Scale Price | Optional: Price bounds computed in log-space (multiplicative fence) to preserve legitimately high-priced premium items |
| Removal Rule | A row is removed only if ≥ 2 different metrics are individually flagged as outliers |
| Artifacts | Full, flagged, removed snapshot, value-level log, numeric coercion report |

**Generated Outlier & Coercion Artifacts (in `etl_dimensions/`):**

| File | Description |
| ---- | ----------- |
| `fact_transaction_full.csv` | Pre-removal fact snapshot (includes `__outlier_*` helper flag columns) |
| `fact_transaction_with_flags.csv` | Alias (currently same as full; future divergence placeholder) |
| `outliers_removed.csv` | Value-level log: one row per flagged value (row index, column, value, method, k, lower, upper, deviation) |
| `outlier_rows_snapshot.csv` | Entire rows meeting multi-metric removal rule (before they are dropped) |
| `numeric_coercion_report.csv` | Counts of newly introduced NaNs per coerced numeric column |

**Load:**
  - **Fact Transaction Dimension:** `fact_transaction_dimension.csv` (cleaned line-level fact after optional outlier row removal & duplicate pruning).  
  - **Product Dimension (SCD Type 4):**
    - `current_product_dimension.csv`: latest row per product with `parent_sku`, `CATEGORY`, `last_transaction_date`, cost inference, SCD metadata.
    - `history_product_dimension.csv`: all historical versions (SCD lineage).  
  - **Parent SKU Derivation:** Removes size/temperature tokens (`ICED`, `HOT`, `8OZ`, etc.), cleans noise, consolidates core product root (`parent_sku`).  
  - **Category Classification:** Heuristic keyword + pattern rules → `DRINK`, `FOOD`, `EXTRA`, `OTHERS`.  
  - **Transaction Records:** `transaction_records.csv` (one row per receipt; `SKU` is comma‑separated `parent_sku` list; reflects cleaned fact after removals).  
  - **Time Dimension:** `time_dimension.csv` (Year → Day hierarchy).

**Intermediate Cleaned Data:**

- `cleaned_data/sales_transactions.csv`
- `cleaned_data/sales_by_product.csv`

**Primary Output & Supporting Files:**

| File | Purpose |
| ---- | ------- |
| `fact_transaction_dimension.csv` | Cleaned line-level fact (post-removal) |
| `fact_transaction_full.csv` | Pre-removal snapshot with helper flags |
| `fact_transaction_with_flags.csv` | Same as full (reserved for future variant behavior) |
| `outliers_removed.csv` | Value-level outlier log (audit trail) |
| `outlier_rows_snapshot.csv` | Full rows that were removed (data provenance) |
| `numeric_coercion_report.csv` | Summary of coercion-induced NaNs |
| `current_product_dimension.csv` | Current SCD slice (Type 4) |
| `history_product_dimension.csv` | All historical product versions |
| `transaction_records.csv` | Receipt-level basket (post-removal) |
| `time_dimension.csv` | Calendar hierarchy for analytics |
| `cleaned_data/*.csv` | Pre-merge cleaned intermediary sources |

---

## 2. Market Basket Analysis (`mba.py`)

Enhanced to operate on multiple ETL outputs and provide a flexible CLI.

**Sources (`--source`):**
| Option | Description |
| ------ | ----------- |
| `transactions` (default) | Uses `transaction_records.csv` (receipt-level baskets, post-removal) |
| `fact` | Rebuilds baskets from `fact_transaction_dimension.csv` |
| `full` | Rebuilds baskets from `fact_transaction_full.csv` (pre-removal; includes flagged rows) |
| `flags` | Rebuilds baskets from `fact_transaction_with_flags.csv` |

If `transaction_records.csv` is missing and `--source transactions` is chosen, it falls back to `fact`.

**Key CLI Arguments:**
```
python mba.py \
  --source full \
  --exclude-flagged \
  --group-by parent_sku \
  --min-support 0.003 \
  --min-confidence 0.03 \
  --min-lift 1.0 \
  --exclude-cats EXTRA OTHERS \
  --no-category-runs
```

| Flag | Purpose |
| ---- | ------- |
| `--group-by parent_sku|product_id` | Choose item granularity for baskets |
| `--exclude-flagged` | Drop any rows with `__outlier_*` flags (only relevant for `full` / `flags` sources) |
| `--exclude-cats` | Skip categories (default: EXTRA, OTHERS) in itemsets & rules |
| `--no-category-runs` | Skip per-category FOOD / DRINK sub-analyses |
| `--min-support` | Minimum FP-Growth support |
| `--min-confidence` | Minimum rule confidence |
| `--min-lift` | Minimum rule lift |

**Processing Flow:**
1. Load or reconstruct baskets (`Receipt No`, `SKU`).
2. One-hot encode items (comma-separated SKUs).
3. Run FP-Growth → frequent itemsets.
4. Generate association rules (confidence filter) → filter by lift & exclusions.
5. De-duplicate reversed rule pairs & compute `combined_score = lift*0.7 + normalized_support*30`.
6. Export Excel + CSV to `mba_results/` (and category folders if enabled).

**Outputs:**
| File | Description |
| ---- | ----------- |
| `mba_results/frequent_itemsets.csv` | Frequent itemsets + support |
| `mba_results/association_rules.csv` | Ranked rules (support, confidence, lift, leverage, conviction, combined_score) |
| `mba_results/market_basket_analysis_results.xlsx` | Excel with both tabs |
| `mba_foods/*` / `mba_drinks/*` | Category-specific results (if enabled) |

**Choosing a Source:**
- Use `transactions` for production-quality basket mining (cleaned fact, consistent with reporting).
- Use `full` + `--exclude-flagged` to experiment with or without outlier impact.
- Use `fact` when you want baskets after removal but before receipt-level aggregation nuance changes.

**Grain Reference:**
| File | Grain |
| ---- | ----- |
| `fact_transaction_full.csv` | Line item (pre-removal) |
| `fact_transaction_dimension.csv` | Line item (post-removal) |
| `transaction_records.csv` | One row per receipt (basket) |

---

## 3. Forecasting & Prescriptive Analytics (`models/`)

- **Descriptive Analytics (`descriptive.py`):**  
  Visualizes sales trends (daily, monthly, quarterly, annual), top products by revenue/quantity, category breakdowns.

- **Holt-Winters Forecasting (`holtwinters.py`):**
  - Loads bundle pairs from association rules
  - Analyzes demand at different price points
  - Forecasts bundle sales per quarter using Holt-Winters
  - Estimates price elasticity of demand (PED) via regression
  - Simulates impact of new bundle pricing on demand and revenue
  - Plots historical vs. forecasted sales and revenue impact

---

## How to Run

1. Place raw Excel files in `raw_sales/` and `raw_sales_by_product/`
2. Run ETL:
   ```
   python etl.py
   ```
3. Run Market Basket Analysis (basic):
  ```
  python mba.py
  ```
  Advanced example (pre-removal data excluding flagged outliers, product_id granularity):
  ```
  python mba.py --source full --exclude-flagged --group-by product_id --min-support 0.002 --min-confidence 0.05
  ```
4. Run analytics/forecasting scripts in `models/` as needed:
   ```
   python models/descriptive.py
   python models/holtwinters.py
   ```

---

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, mlxtend, statsmodels, scikit-learn, openpyxl

---

## Notes

- All outputs are regenerated on each run; ensure raw files are up-to-date.
- For bundle pricing analysis, edit the bundle selection and price in `models/holtwinters.py`.
