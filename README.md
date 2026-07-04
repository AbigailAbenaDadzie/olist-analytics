# 🛒 Olist E-Commerce Analytics

Analytics engineering capstone on Olist's Brazilian marketplace dataset — from
raw relational data to a growth recommendation, with an interactive dashboard
and a predictive model.

**🔗 Live app:** [olist-analytics-8gwgsfkkw2kwnsckkbfgec.streamlit.app](https://olist-analytics-8gwgsfkkw2kwnsckkbfgec.streamlit.app/)

## What's here

| Path | What it is |
|---|---|
| `01_ecommerce_olist_SOLUTION.ipynb` | Full analysis notebook: load → SQL marts → charts → model → recommendation |
| `sql/` | Standalone `.sql` files for each mart (monthly revenue, top sellers, customer tiers) |
| `dashboard/app.py` | Interactive Streamlit app version of the analysis |
| `slides/olist_analytics.pptx` | Presentation deck summarizing findings |
| `requirements.txt` | Python dependencies |

## Setup

1. **Get the data.** Download the dataset from
   [kaggle.com/datasets/olistbr/brazilian-ecommerce](https://kaggle.com/datasets/olistbr/brazilian-ecommerce)
   and unzip it into a folder named `olist/` at the repo root, so you have:
   ```
   olist/
   ├── olist_customers_dataset.csv
   ├── olist_geolocation_dataset.csv
   ├── olist_order_items_dataset.csv
   ├── olist_order_payments_dataset.csv
   ├── olist_order_reviews_dataset.csv
   ├── olist_orders_dataset.csv
   ├── olist_products_dataset.csv
   ├── olist_sellers_dataset.csv
   └── product_category_name_translation.csv
   ```
   (This folder is gitignored — the data isn't committed to the repo.)

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Run the notebook

```bash
jupyter notebook 01_ecommerce_olist_SOLUTION.ipynb
```
Kernel ▸ Restart & Run All.

## Run the Streamlit app

```bash
cd dashboard
streamlit run app.py
```
Opens at `http://localhost:8501`. Use the sidebar to switch currency (BRL/GHS)
and navigate between Revenue, Sellers, Customers, and the Delivery Risk Model.

## Key findings

- **Revenue growth:** cumulative revenue grew ~18× from 2016 to peak in 2018.
- **Customer concentration:** top-quartile customers spend ~8× the bottom quartile.
- **Seller leadership:** each product category is dominated by a small number
  of top-ranked sellers.
- **Delivery risk model:** Logistic Regression catches 51% of genuinely late
  deliveries (recall), vs. 0% for a naive "always on-time" baseline. Raw
  accuracy favors the baseline only because of class imbalance (7.9% late
  rate) — ROC-AUC (0.638) is the fairer comparison and confirms real, if
  moderate, predictive signal.

## Recommendation

Focus retention spend on top-quartile customers, use the delivery-risk model
operationally to flag and proactively manage high-risk orders, and prioritize
category-leading sellers for co-marketing. See Section 7 of the notebook or
the Recommendation page in the Streamlit app for the full writeup.

## Data source

[Brazilian E-Commerce Public Dataset by Olist](https://kaggle.com/datasets/olistbr/brazilian-ecommerce)
(Kaggle, CC BY-NC-SA 4.0).
