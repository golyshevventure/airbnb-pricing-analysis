#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
app.py — Streamlit Dashboard v2
==============================
5 вкладок:
1. Overview — базовая аналитика
2. Predictive Model — регрессия и важность признаков
3. Host Segmentation — профи vs любители
4. Geography — карты цен
5. Seasonality — временной анализ
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Airbnb Market Intelligence", page_icon="🏠", layout="wide")

# =============================================================================
# ЗАГРУЗКА ДАННЫХ
# =============================================================================

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base, "data", "processed", "airbnb_cleaned.csv"))
    return df

@st.cache_data
def load_calendar():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "data", "processed", "calendar_monthly_agg.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_raw():
    base = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base, "data", "processed", "airbnb_raw_combined.csv"), low_memory=False)
    df["price_clean"] = df["price"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).astype(float)
    df = df[(df["price_clean"] >= 10) & (df["price_clean"] <= 2000)]
    return df

df = load_data()
df_raw = load_raw()
df_cal = load_calendar()

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🔧 Filters")
selected_cities = st.sidebar.multiselect("Cities", sorted(df["city"].unique()), default=sorted(df["city"].unique()))
selected_rooms = st.sidebar.multiselect("Room Type", sorted(df["room_type"].unique()), default=sorted(df["room_type"].unique()))
min_p, max_p = st.sidebar.slider("Price", int(df["price_clean"].min()), int(df["price_clean"].max()), (10, 500))
min_r = st.sidebar.slider("Min Rating", 0.0, 5.0, 0.0, 0.1)

df_f = df[(df["city"].isin(selected_cities)) & (df["room_type"].isin(selected_rooms)) &
          (df["price_clean"] >= min_p) & (df["price_clean"] <= max_p) & (df["review_scores_rating"] >= min_r)].copy()

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🤖 Predictive", "👥 Hosts", "🗺️ Geography", "📅 Seasonality"
])

# =============================================================================
# TAB 1: OVERVIEW
# =============================================================================

with tab1:
    st.title("🏠 Airbnb Market Intelligence")
    st.markdown("*77,000+ listings across Austin, Dallas, Nashville, Seattle, NYC*")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Listings", f"{len(df_f):,}")
    c2.metric("Median Price", f"${df_f['price_clean'].median():.0f}")
    c3.metric("Mean Price", f"${df_f['price_clean'].mean():.0f}")
    c4.metric("Avg Rating", f"{df_f['review_scores_rating'].mean():.2f}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Price Distribution")
        fig = px.box(df_f[df_f["price_clean"] <= 500], x="city", y="price_clean", color="city")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Avg vs Median")
        stats = df_f.groupby("city")["price_clean"].agg(["mean", "median"]).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Mean", x=stats["city"], y=stats["mean"], marker_color="#3498db"))
        fig.add_trace(go.Bar(name="Median", x=stats["city"], y=stats["median"], marker_color="#e74c3c"))
        fig.update_layout(barmode="group", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price vs Rating")
    samp = df_f.sample(min(5000, len(df_f)), random_state=42)
    fig = px.scatter(samp, x="review_scores_rating", y="price_clean", color="city", opacity=0.5)
    fig.update_yaxes(range=[0, 500])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Price Categories")
    cat = df_f.groupby(["city", "price_category"]).size().unstack(fill_value=0)
    cat_pct = cat.div(cat.sum(axis=1), axis=0) * 100
    fig = px.bar(cat_pct.reset_index().melt(id_vars="city"), x="city", y="value", color="price_category",
                 color_discrete_map={"Budget": "#2ecc71", "Mid": "#f39c12", "Luxury": "#e74c3c"})
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2: PREDICTIVE MODEL
# =============================================================================

with tab2:
    st.title("🤖 Predictive Model")
    st.markdown("*Linear Regression: What drives Airbnb prices?*")

    from sklearn.linear_model import LinearRegression

    # Простая модель на лету
    d = df_raw[df_raw["city"].isin(selected_cities)].copy()
    d = d[d["price_clean"] <= 500]

    features = ["number_of_reviews", "review_scores_rating", "minimum_nights", "latitude", "longitude"]
    if "calculated_host_listings_count" in d.columns:
        features.append("calculated_host_listings_count")

    dummies = pd.get_dummies(d["room_type"], prefix="room")
    d = pd.concat([d, dummies], axis=1)
    features += list(dummies.columns)

    m = d[features + ["price_clean"]].dropna()
    if len(m) > 100:
        X = m[features]
        y = m["price_clean"]
        model = LinearRegression().fit(X, y)
        preds = model.predict(X)
        r2 = model.score(X, y)

        c1, c2 = st.columns(2)
        c1.metric("R² Score", f"{r2:.3f}")
        c2.metric("MAE", f"${np.mean(np.abs(y - preds)):.0f}")

        st.markdown("---")

        coefs = pd.Series(model.coef_, index=features).sort_values(key=abs, ascending=False)
        st.subheader("Feature Importance")
        fig = px.bar(x=coefs.head(10).values, y=coefs.head(10).index, orientation="h",
                     color=coefs.head(10).values, color_continuous_scale="RdBu_r")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Predicted vs Actual")
        fig = px.scatter(x=y, y=preds, opacity=0.3, labels={"x": "Actual", "y": "Predicted"})
        fig.add_shape(type="line", x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max(), line=dict(dash="dash", color="red"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough data for model. Adjust filters.")

# =============================================================================
# TAB 3: HOST SEGMENTATION
# =============================================================================

with tab3:
    st.title("👥 Host Segmentation")
    st.markdown("*Hobbyists vs Professionals*")

    if "calculated_host_listings_count" in df_raw.columns:
        d = df_raw[df_raw["city"].isin(selected_cities)].copy()
        d["segment"] = pd.cut(d["calculated_host_listings_count"], 
                              bins=[0, 1, 5, 20, 9999],
                              labels=["Hobbyist (1)", "Small (2-5)", "Manager (6-20)", "Professional (20+)"],
                              include_lowest=True)

        seg = d.groupby("segment").agg(
            count=("price_clean", "count"),
            median_price=("price_clean", "median"),
            mean_rating=("review_scores_rating", "mean")
        ).reset_index()

        c1, c2, c3 = st.columns(3)
        c1.metric("Hobbyists", f"{seg[seg['segment']=='Hobbyist (1)']['count'].sum():,}")
        c2.metric("Professionals", f"{seg[seg['segment']=='Professional (20+)']['count'].sum():,}")
        c3.metric("Prof. Median Price", f"${seg[seg['segment']=='Professional (20+)']['median_price'].sum():.0f}")

        st.markdown("---")

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Price by Segment")
            fig = px.box(d[d["price_clean"] <= 400], x="segment", y="price_clean", color="segment")
            st.plotly_chart(fig, use_container_width=True)
        with col_right:
            st.subheader("Rating by Segment")
            fig = px.box(d.dropna(subset=["review_scores_rating"]), x="segment", y="review_scores_rating", color="segment")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Host data not available.")

# =============================================================================
# TAB 4: GEOGRAPHY
# =============================================================================

with tab4:
    st.title("🗺️ Geographic Price Heatmap")
    st.markdown("*Where are the expensive listings?*")

    city_for_map = st.selectbox("Select City", sorted(df_f["city"].unique()))
    d_map = df_f[df_f["city"] == city_for_map].sample(min(3000, len(df_f[df_f["city"] == city_for_map])), random_state=42)

    fig = px.scatter_mapbox(d_map, lat="latitude", lon="longitude", color="price_clean",
                            color_continuous_scale="YlOrRd", size_max=8, zoom=10,
                            height=600, title=f"{city_for_map} — Price Heatmap")
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Top 10 Neighborhoods")
    top_neigh = df_f[df_f["city"] == city_for_map].groupby("neighbourhood_cleansed")["price_clean"].mean().sort_values(ascending=False).head(10)
    fig = px.bar(x=top_neigh.values, y=top_neigh.index, orientation="h", color=top_neigh.values,
                 color_continuous_scale="Viridis", labels={"x": "Avg Price ($)", "y": "Neighborhood"})
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 5: SEASONALITY
# =============================================================================

with tab5:
    st.title("📅 Seasonality & Occupancy")
    st.markdown("*28M+ calendar records analyzed*")

    if df_cal is not None:
        cal_f = df_cal[df_cal["city"].isin(selected_cities)].copy()

        c1, c2, c3 = st.columns(3)
        c1.metric("Records Analyzed", "28,429,579")
        c2.metric("Peak City (Price)", cal_f.loc[cal_f["avg_price"].idxmax(), "city"])
        c3.metric("Peak City (Occupancy)", cal_f.loc[cal_f["occupancy_rate"].idxmax(), "city"])

        st.markdown("---")

        st.subheader("Price Seasonality")
        fig = px.line(cal_f, x="year_month_str", y="avg_price", color="city", markers=True, height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Occupancy Rate (%)")
        cal_f["occ_pct"] = cal_f["occupancy_rate"] * 100
        fig = px.line(cal_f, x="year_month_str", y="occ_pct", color="city", markers=True, height=500)
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("City Comparison")
        summary = cal_f.groupby("city").agg(avg_price=("avg_price", "mean"), avg_occ=("occupancy_rate", "mean")).reset_index()
        summary["avg_occ_pct"] = summary["avg_occ"] * 100

        col_left, col_right = st.columns(2)
        with col_left:
            fig = px.bar(summary, x="city", y="avg_price", color="city", height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col_right:
            fig = px.bar(summary, x="city", y="avg_occ_pct", color="city", height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Calendar data not found. Run 05_temporal_analysis.py first.")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.caption("Built with Streamlit • 77K+ listings • 28M+ calendar records • By Nikita")
