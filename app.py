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
PRIMARY_COLOR = "#2563EB"
BG_COLOR = "#FAFAFA"
CARD_BG = "#FFFFFF"
TEXT_COLOR = "#334155"
HEADER_COLOR = "#0F172A"

st.markdown(f"""
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
💻 Laptop Price Analytics Dashboard
Market Analysis & Machine Learning Price Prediction
Talha Bashir, Abdullah | Roll No: 2430-0162, 2430-0067 | PAI Course Project
 """, unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    file_path = "final_cleaned_laptop_data .csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File '{file_path}' not found.")
        return pd.DataFrame()
    
    df.columns = df.columns.str.strip()
    
    if "OpSys_Category" in df.columns:
        df.rename(columns={"OpSys_Category": "OpSys"}, inplace=True)
    
    df["Ram_GB"] = df["Ram"]
    df["Weight_kg"] = df["Weight"]
    
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

# MACHINE LEARNING - UPDATED
with tabs[4]:
    st.markdown("## 🤖 Laptop Price Prediction Model")
    
    # Model Training Parameters
    st.markdown("### ⚙️ Model Training Configuration")
    colA, colB, colC = st.columns(3)
    n_estimators = colA.slider("Trees", 50, 300, 150)
    max_depth = colB.slider("Max Depth", 5, 40, 20)
    test_size = colC.slider("Test Size (%)", 10, 40, 20) / 100
    
    if st.button("🔄 Train Model"):
        features = ["Ram_GB", "Inches", "Weight_kg", "Company", "TypeName", "OpSys", "Cpu_Speed_GHz", "SSD"]
        model_df = filtered_df.dropna(subset=features + ["Price"]).copy()
        
        # Encoding
        le_dict = {}
        categorical_cols = ["Company", "TypeName", "OpSys"]
        for col in categorical_cols:
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col])
            le_dict[col] = le
        
        X = model_df[features]
        y = model_df["Price"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Store model in session
        st.session_state.model = model
        st.session_state.le_dict = le_dict
        st.session_state.features = features
        st.session_state.model_trained = True
        
        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE", f"₹{np.sqrt(mean_squared_error(y_test, preds)):,.0f}")
        m2.metric("MAE", f"₹{mean_absolute_error(y_test, preds):,.0f}")
        m3.metric("R² Score", f"{r2_score(y_test, preds):.4f}")
        
        st.success("✅ Model trained successfully!")
        
        fig = px.scatter(x=y_test, y=preds, labels={'x': 'Actual', 'y': 'Predicted'}, 
                        title="Actual vs Predicted Prices")
        fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                     line=dict(dash="dash"))
        st.plotly_chart(fig, use_container_width=True)
    
    # Price Prediction Section
    st.markdown("---")
    st.markdown("### 📋 Predict Price for Your Laptop")
    
    if 'model_trained' in st.session_state and st.session_state.model_trained:
        # User Input Form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_company = st.selectbox("Company", sorted(df["Company"].unique()))
            selected_type = st.selectbox("Laptop Type", sorted(df["TypeName"].unique()))
            ram_gb = st.number_input("RAM (GB)", min_value=2, max_value=64, value=8, step=1)
        
        with col2:
            inches = st.number_input("Screen Size (Inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)
            weight_kg = st.number_input("Weight (kg)", min_value=1.0, max_value=4.0, value=1.8, step=0.1)
            cpu_speed = st.number_input("CPU Speed (GHz)", min_value=1.0, max_value=5.0, value=2.5, step=0.1)
        
        with col3:
            selected_os = st.selectbox("Operating System", sorted(df["OpSys"].unique()))
            ssd = st.number_input("SSD Storage (GB)", min_value=0, max_value=2048, value=512, step=128)
        
        if st.button("🚀 Predict Price"):
            try:
                model = st.session_state.model
                le_dict = st.session_state.le_dict
                features = st.session_state.features
                
                # Prepare input data
                input_data = pd.DataFrame({
                    "Ram_GB": [ram_gb],
                    "Inches": [inches],
                    "Weight_kg": [weight_kg],
                    "Company": [le_dict["Company"].transform([selected_company])[0]],
                    "TypeName": [le_dict["TypeName"].transform([selected_type])[0]],
                    "OpSys": [le_dict["OpSys"].transform([selected_os])[0]],
                    "Cpu_Speed_GHz": [cpu_speed],
                    "SSD": [ssd]
                })
                
                predicted_price = model.predict(input_data)[0]
                
                # Display Prediction Result
                st.markdown("---")
                st.markdown("### 💰 Prediction Result")
                
                pred_col1, pred_col2, pred_col3 = st.columns([1, 2, 1])
                with pred_col2:
                    st.metric("Predicted Price", f"₹{predicted_price:,.0f}", delta=None)
                
                # Show input summary
                st.markdown("#### Your Laptop Specifications:")
                summary_data = {
                    "Company": selected_company,
                    "Type": selected_type,
                    "RAM": f"{ram_gb} GB",
                    "Screen Size": f"{inches}\" inches",
                    "Weight": f"{weight_kg} kg",
                    "CPU Speed": f"{cpu_speed} GHz",
                    "SSD": f"{ssd} GB",
                    "OS": selected_os
                }
                summary_df = pd.DataFrame(list(summary_data.items()), columns=["Specification", "Value"])
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
    else:
        st.info("⚠️ Please train the model first by clicking the 'Train Model' button above!")

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
