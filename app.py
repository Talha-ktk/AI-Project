import numpy as np
import pandas as pd
from datetime import datetime

import plotly.express as px
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Laptop Price Analytics",
    layout="wide",
    page_icon="💻"
)

# =====================================================
# CLEAN & MINIMAL THEME
# =====================================================
PRIMARY_COLOR = "#2563EB"       # A clean, standard blue
BG_COLOR = "#FAFAFA"            # Almost white
CARD_BG = "#FFFFFF"             # Pure White
TEXT_COLOR = "#334155"          # Slate 700
HEADER_COLOR = "#0F172A"        # Slate 900

st.markdown(f"""
<style>
.stApp {{ background-color: {BG_COLOR}; }}
h1, h2, h3 {{ color: {HEADER_COLOR} !important; font-family: 'Inter', sans-serif; font-weight: 700; }}
.header-card {{
    background-color: {CARD_BG};
    padding: 2rem;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    margin-bottom: 2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}}
[data-testid="stMetric"] {{
    background-color: {CARD_BG};
    border: 1px solid #F1F5F9;
    border-radius: 10px;
    padding: 1.2rem;
}}
.stButton > button {{
    background-color: {PRIMARY_COLOR};
    color: white;
    border-radius: 8px;
    font-weight: 500;
    border: none;
}}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="header-card">
  <h1>💻 Laptop Price Analytics Dashboard</h1>
  <p>Market Analysis & Machine Learning Price Prediction</p>
  <p><b>Talha Bashir, Abdullah</b> | Roll No: 2430-0162, 2430-0067 | PAI Course Project</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    # UPDATED: Matches the specific filename with the trailing space
    file_path = "final_cleaned_laptop_data .csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File '{file_path}' not found. Please ensure it is in the same directory.")
        return pd.DataFrame()

    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Map columns to internal names used in the app
    if "OpSys_Category" in df.columns:
        df.rename(columns={"OpSys_Category": "OpSys"}, inplace=True)
    
    # The dataset already contains Ram (int) and Weight (float)
    df["Ram_GB"] = df["Ram"]
    df["Weight_kg"] = df["Weight"]

    # Create Price Category for visualization
    df["Price_Category"] = pd.cut(
        df["Price"],
        bins=[0, 35000, 75000, 120000, np.inf],
        labels=["Budget", "Mid-Range", "Premium", "Luxury"]
    )

    return df

df = load_data()

if df.empty:
    st.stop()

# =====================================================
# SIDEBAR FILTERS
# =====================================================
st.sidebar.header("🎛️ Filters")

companies = sorted(df["Company"].unique())
types = sorted(df["TypeName"].unique())

selected_companies = st.sidebar.multiselect("Company", companies, default=companies[:5])
selected_types = st.sidebar.multiselect("Laptop Type", types, default=types)

price_min, price_max = float(df["Price"].min()), float(df["Price"].max())
price_range = st.sidebar.slider("Price Range (₹)", price_min, price_max, (price_min, price_max))

filtered_df = df[
    (df["Company"].isin(selected_companies)) &
    (df["TypeName"].isin(selected_types)) &
    (df["Price"].between(price_range[0], price_range[1]))
]

if filtered_df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

# =====================================================
# TABS
# =====================================================
tabs = st.tabs(["🏠 Overview", "📊 Market", "💻 Specs", "📈 Trends", "🤖 ML Model", "💾 Export"])

# OVERVIEW
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Price", f"₹{filtered_df['Price'].mean():,.0f}")
    c2.metric("Total Laptops", len(filtered_df))
    c3.metric("Avg RAM", f"{filtered_df['Ram_GB'].mean():.1f} GB")
    c4.metric("Companies", filtered_df["Company"].nunique())

    fig = px.bar(
        filtered_df.groupby("Company")["Price"].mean().sort_values(ascending=False).head(10).reset_index(),
        x="Price", y="Company", orientation="h",
        title="Top 10 Companies by Average Price",
        color_discrete_sequence=[PRIMARY_COLOR]
    )
    st.plotly_chart(fig, use_container_width=True)

# MARKET
with tabs[1]:
    share_df = filtered_df["Company"].value_counts().head(8).reset_index()
    share_df.columns = ["Company", "Count"]
    fig = px.pie(share_df, names="Company", values="Count", hole=0.4, title="Market Share (Selected Data)")
    st.plotly_chart(fig, use_container_width=True)

# SPECS
with tabs[2]:
    fig = px.box(filtered_df, x="Ram_GB", y="Price", title="RAM Impact on Price")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        filtered_df, x="Inches", y="Price", size="Weight_kg", color="Company",
        title="Screen Size vs Price (Bubble size = Weight)"
    )
    st.plotly_chart(fig, use_container_width=True)

# TRENDS
with tabs[3]:
    fig = px.histogram(filtered_df, x="Price", color="Price_Category", title="Price Distribution")
    st.plotly_chart(fig, use_container_width=True)

# MACHINE LEARNING
with tabs[4]:
    st.markdown("## 🤖 Laptop Price Prediction Model")
    
    colA, colB, colC = st.columns(3)
    n_estimators = colA.slider("Trees", 50, 300, 150)
    max_depth = colB.slider("Max Depth", 5, 40, 20)
    test_size = colC.slider("Test Size (%)", 10, 40, 20) / 100

    if st.button("Train Model"):
        # Select relevant features from your CSV
        features = ["Ram_GB", "Inches", "Weight_kg", "Company", "TypeName", "OpSys", "Cpu_Speed_GHz", "SSD"]
        model_df = filtered_df.dropna(subset=features + ["Price"]).copy()

        # Encoding
        le = LabelEncoder()
        for col in ["Company", "TypeName", "OpSys"]:
            model_df[col] = le.fit_transform(model_df[col])

        X = model_df[features]
        y = model_df["Price"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE", f"₹{np.sqrt(mean_squared_error(y_test, preds)):,.0f}")
        m2.metric("MAE", f"₹{mean_absolute_error(y_test, preds):,.0f}")
        m3.metric("R² Score", f"{r2_score(y_test, preds):.4f}")

        fig = px.scatter(x=y_test, y=preds, labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted")
        fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

# EXPORT
with tabs[5]:
    st.dataframe(filtered_df, use_container_width=True)
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Filtered Results",
        csv,
        f"laptop_data_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv"
    )
