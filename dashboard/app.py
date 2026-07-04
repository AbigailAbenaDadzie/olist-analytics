"""
Olist E-Commerce Analytics — Streamlit App
--------------------------------------------
Interactive version of the analytics notebook: revenue trends, seller
leaderboards, customer value tiers, and a late-delivery prediction model.

Run locally:
    streamlit run app.py

Expects a folder named `olist/` (containing the 9 Olist CSVs) in the same
directory as this file. See README.md for the dataset link.
"""

import os
import glob
import re

import duckdb
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Olist E-Commerce Analytics",
    page_icon="🛒",
    layout="wide",
)

TEAL = "#0F7173"
CORAL = "#FF6B5B"
NAVY = "#0B2545"
SLATE = "#5C7080"

# ----------------------------------------------------------------------
# Data loading (cached — runs once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def get_connection():
    con = duckdb.connect(":memory:")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_files = glob.glob(os.path.join(BASE_DIR, "olist", "*.csv"))
    if not csv_files:
        return None
    for f in csv_files:
        name = re.sub(r"olist_|_dataset", "", os.path.basename(f)[:-4])
        con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM read_csv_auto(\'{f}\')')
    return con


@st.cache_data
def load_monthly_revenue(_con):
    return _con.sql("""
        SELECT
            date_trunc('month', o.order_purchase_timestamp) AS month,
            SUM(oi.price + oi.freight_value)                AS revenue,
            SUM(SUM(oi.price + oi.freight_value)) OVER (
                ORDER BY date_trunc('month', o.order_purchase_timestamp)
            )                                                AS running_total
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.order_status NOT IN ('canceled', 'unavailable')
        GROUP BY 1
        ORDER BY 1
    """).df()


@st.cache_data
def load_top_sellers(_con):
    return _con.sql("""
        WITH seller_cat_revenue AS (
            SELECT
                p.product_category_name AS category,
                oi.seller_id,
                SUM(oi.price)           AS revenue,
                COUNT(*)                AS items_sold
            FROM order_items oi
            JOIN products p ON oi.product_id = p.product_id
            JOIN sellers  s ON oi.seller_id  = s.seller_id
            WHERE p.product_category_name IS NOT NULL
            GROUP BY 1, 2
        ),
        ranked AS (
            SELECT *,
                RANK() OVER (PARTITION BY category ORDER BY revenue DESC) AS rank_in_category
            FROM seller_cat_revenue
        )
        SELECT * FROM ranked WHERE rank_in_category <= 5
        ORDER BY category, rank_in_category
    """).df()


@st.cache_data
def load_customer_tiers(_con):
    return _con.sql("""
        WITH customer_spend AS (
            SELECT
                c.customer_unique_id,
                SUM(oi.price + oi.freight_value) AS total_spend,
                COUNT(DISTINCT o.order_id)       AS n_orders
            FROM orders o
            JOIN order_items oi ON o.order_id    = oi.order_id
            JOIN customers  c  ON o.customer_id  = c.customer_id
            WHERE o.order_status NOT IN ('canceled', 'unavailable')
            GROUP BY 1
        )
        SELECT *,
            NTILE(4) OVER (ORDER BY total_spend DESC) AS value_quartile
        FROM customer_spend
    """).df()


@st.cache_data
def load_model_data(_con):
    return _con.sql("""
        SELECT
            o.order_id,
            CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date
                 THEN 1 ELSE 0 END                                            AS is_late,
            oi.price,
            oi.freight_value,
            p.product_category_name                                          AS category,
            cu.customer_state,
            s.seller_state,
            DATE_PART('dow', o.order_purchase_timestamp)                     AS purchase_dow
        FROM orders o
        JOIN order_items oi ON o.order_id   = oi.order_id
        JOIN products    p  ON oi.product_id = p.product_id
        JOIN sellers      s ON oi.seller_id  = s.seller_id
        JOIN customers   cu ON o.customer_id = cu.customer_id
        WHERE o.order_status = 'delivered'
          AND o.order_delivered_customer_date IS NOT NULL
    """).df()


@st.cache_resource
def train_model(_model_df):
    model_df = _model_df.dropna(subset=["category", "price", "freight_value"])
    X = model_df[["price", "freight_value", "category", "customer_state", "seller_state", "purchase_dow"]]
    y = model_df["is_late"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    categorical = ["category", "customer_state", "seller_state"]
    numeric = ["price", "freight_value", "purchase_dow"]

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numeric),
    ])
    clf = Pipeline([
        ("prep", preprocess),
        ("logreg", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    y_base = baseline.predict(X_test)

    return {
        "y_test": y_test, "y_pred": y_pred, "y_proba": y_proba, "y_base": y_base,
        "accuracy": accuracy_score(y_test, y_pred),
        "baseline_accuracy": accuracy_score(y_test, y_base),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "report": classification_report(y_test, y_pred, target_names=["on-time", "late"], output_dict=True),
    }


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.title("🛒 Olist Analytics")
st.sidebar.caption("E-commerce marketplace insights")

currency = st.sidebar.radio("Currency", ["BRL (R$)", "GHS (GH₵)"], index=0)
fx_rate = 1.0
currency_symbol = "R$"
if currency.startswith("GHS"):
    fx_rate = st.sidebar.number_input("BRL → GHS rate", value=2.10, step=0.01, format="%.2f")
    currency_symbol = "GH₵"

page = st.sidebar.radio(
    "Section",
    ["Overview", "Revenue", "Sellers", "Customers", "Delivery Risk Model", "Recommendation"],
)

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
con = get_connection()

if con is None:
    st.error(
        "No data found. Create a folder named `olist/` next to this app "
        "containing the 9 Olist CSVs, then restart the app.\n\n"
        "Dataset: kaggle.com/datasets/olistbr/brazilian-ecommerce"
    )
    st.stop()

monthly_revenue = load_monthly_revenue(con)
top_sellers = load_top_sellers(con)
customer_tiers = load_customer_tiers(con)

monthly_revenue["revenue_disp"] = monthly_revenue["revenue"] * fx_rate
monthly_revenue["running_total_disp"] = monthly_revenue["running_total"] * fx_rate
customer_tiers["total_spend_disp"] = customer_tiers["total_spend"] * fx_rate

# ----------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------
if page == "Overview":
    st.title("Olist E-Commerce Analytics")
    st.caption("Online marketplace · operations & marketing · analytics dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"{currency_symbol}{monthly_revenue['revenue_disp'].sum():,.0f}")
    c2.metric("Customers", f"{customer_tiers.shape[0]:,}")
    c3.metric("Months of Data", f"{monthly_revenue.shape[0]}")
    c4.metric("Top Quartile Avg Spend", f"{currency_symbol}{customer_tiers[customer_tiers.value_quartile==1]['total_spend_disp'].mean():,.0f}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(monthly_revenue, x="month", y="running_total_disp",
                       title="Cumulative Revenue", color_discrete_sequence=[CORAL])
        fig.update_layout(yaxis_title=f"Revenue ({currency_symbol})", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        qsum = customer_tiers.groupby("value_quartile")["total_spend_disp"].mean().reset_index()
        fig = px.bar(qsum, x="value_quartile", y="total_spend_disp",
                     title="Avg Spend by Customer Quartile", color_discrete_sequence=[TEAL])
        fig.update_layout(yaxis_title=f"Avg spend ({currency_symbol})", xaxis_title="Quartile (1 = highest)")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Revenue":
    st.title("📈 Revenue")
    st.caption("Monthly revenue with running total — orders + order_items joined, SUM() OVER window function")

    fig = go.Figure()
    fig.add_bar(x=monthly_revenue["month"], y=monthly_revenue["revenue_disp"], name="Monthly revenue", marker_color=TEAL)
    fig.add_trace(go.Scatter(x=monthly_revenue["month"], y=monthly_revenue["running_total_disp"],
                              name="Running total", yaxis="y2", line=dict(color=CORAL, width=3)))
    fig.update_layout(
        yaxis=dict(title=f"Monthly revenue ({currency_symbol})"),
        yaxis2=dict(title=f"Cumulative ({currency_symbol})", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(monthly_revenue[["month", "revenue_disp", "running_total_disp"]].rename(
        columns={"revenue_disp": f"revenue ({currency_symbol})", "running_total_disp": f"running total ({currency_symbol})"}
    ), use_container_width=True)

elif page == "Sellers":
    st.title("🏆 Top Sellers by Category")
    st.caption("RANK() OVER (PARTITION BY category ORDER BY revenue DESC) — top 5 sellers per category")

    categories = sorted(top_sellers["category"].unique())
    selected = st.multiselect("Filter categories", categories, default=categories[:5])
    filtered = top_sellers[top_sellers["category"].isin(selected)] if selected else top_sellers
    st.dataframe(filtered, use_container_width=True)

elif page == "Customers":
    st.title("👥 Customer Value Tiers")
    st.caption("NTILE(4) OVER (ORDER BY spend DESC) — customers split into value quartiles")

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.histogram(customer_tiers, x="total_spend_disp", color="value_quartile",
                            nbins=50, title="Spend Distribution by Quartile",
                            color_discrete_sequence=px.colors.sequential.Teal)
        fig.update_layout(xaxis_title=f"Total spend ({currency_symbol})")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        summary = customer_tiers.groupby("value_quartile").agg(
            customers=("customer_unique_id", "count"),
            avg_spend=("total_spend_disp", "mean"),
            total_spend=("total_spend_disp", "sum"),
        ).reset_index()
        st.dataframe(summary, use_container_width=True)

elif page == "Delivery Risk Model":
    st.title("🚚 Late-Delivery Prediction")
    st.caption("Logistic Regression vs. majority-class baseline")

    with st.spinner("Training model..."):
        model_df = load_model_data(con)
        results = train_model(model_df)

    late_rate = model_df["is_late"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Late-delivery rate", f"{late_rate:.1%}")
    c2.metric("Model ROC-AUC", f"{results['roc_auc']:.3f}")
    c3.metric("Late orders caught (recall)", f"{results['report']['late']['recall']:.0%}")

    st.warning(
        "⚠️ The baseline's higher raw accuracy is misleading — it comes from always "
        "predicting 'on-time' and catches **0%** of late orders. Recall on the 'late' "
        "class is the metric that matters here."
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            x=["Baseline", "LogisticRegression"],
            y=[results["baseline_accuracy"], results["accuracy"]],
            title="Raw Accuracy (misleading alone)",
            color=["Baseline", "Model"], color_discrete_sequence=[SLATE, CORAL],
        )
        fig.update_layout(yaxis_range=[0, 1], showlegend=False, yaxis_title="Accuracy")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(
            x=["Baseline", "LogisticRegression"],
            y=[0.0, results["report"]["late"]["recall"]],
            title="Late Orders Actually Caught (recall)",
            color=["Baseline", "Model"], color_discrete_sequence=[SLATE, TEAL],
        )
        fig.update_layout(yaxis_range=[0, 1], showlegend=False, yaxis_title="Recall")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full classification report"):
        st.dataframe(pd.DataFrame(results["report"]).T, use_container_width=True)

elif page == "Recommendation":
    st.title("💡 Recommendation")
    st.markdown("""
> **Our recommendation:** Revenue and customer value are highly concentrated — the
> top spending quartile accounts for a disproportionate share of total revenue, so
> retention and loyalty spend should be targeted there rather than spread evenly
> across the customer base. On delivery risk, our logistic regression model
> identifies **51%** of genuinely late orders (recall), compared to **0%** for a
> naive baseline that just predicts every order arrives on time — even though the
> baseline shows higher raw accuracy (92.1% vs 68.7%), that number is an artifact
> of only 7.9% of orders being late, and offers zero actionable signal. We
> recommend using the model's late-delivery flag operationally: proactively notify
> customers and tighten delivery-date estimates on flagged high-risk orders, rather
> than relying on overall accuracy as the success metric. Because the model's
> ROC-AUC (0.638) shows only moderate predictive power, we'd also recommend
> enriching it with true shipping distance and carrier-level features before using
> it for anything beyond an early-warning signal.
    """)

    st.subheader("Three actions")
    c1, c2, c3 = st.columns(3)
    c1.info("**Retain top-quartile customers**\n\nQ1 customers spend ~8× the bottom quartile. Launch a targeted loyalty offer.")
    c2.info("**Flag high-risk deliveries**\n\nUse the model's late-delivery flag to trigger proactive notice and tighter estimates.")
    c3.info("**Back category-leading sellers**\n\nPrioritize top-ranked sellers per category for co-marketing and featured placement.")

st.sidebar.divider()
st.sidebar.caption("Built with Streamlit · DuckDB · scikit-learn")
