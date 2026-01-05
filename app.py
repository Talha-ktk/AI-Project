import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
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
    page_icon="💻",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS STYLING
# =====================================================
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* HEADER STYLES */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 40px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        text-align: center;
    }
    
    .header-title {
        font-size: 2.5em;
        font-weight: 800;
        margin-bottom: 10px;
        letter-spacing: -1px;
    }
    
    .header-subtitle {
        font-size: 1.1em;
        opacity: 0.95;
        margin-bottom: 8px;
    }
    
    .header-authors {
        font-size: 0.95em;
        opacity: 0.85;
        font-style: italic;
    }
    
    /* TABS STYLING */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px 20px;
        font-weight: 600;
        color: #667eea;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f0f2ff;
        border-color: #667eea;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* METRIC CARDS */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .metric-label {
        font-size: 0.9em;
        opacity: 0.9;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 1.8em;
        font-weight: 800;
    }
    
    /* BUTTON STYLES */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1em;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* INPUT STYLES */
    .stNumberInput, .stSelectbox, .stSlider {
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        font-size: 1em;
        transition: border-color 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea;
        outline: none;
    }
    
    /* SECTION HEADERS */
    h1, h2, h3 {
        color: #2c3e50;
        margin-bottom: 15px;
    }
    
    h2 {
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
    
    /* SIDEBAR STYLING */
    .stSidebar {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* DATA FRAME */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* SUCCESS/ERROR/INFO MESSAGES */
    .stSuccess {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 8px;
        padding: 15px;
        color: #155724;
        font-weight: 600;
    }
    
    .stError {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 8px;
        padding: 15px;
        color: #721c24;
        font-weight: 600;
    }
    
    .stWarning {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 8px;
        padding: 15px;
        color: #856404;
        font-weight: 600;
    }
    
    .stInfo {
        background-color: #d1ecf1;
        border: 2px solid #17a2b8;
        border-radius: 8px;
        padding: 15px;
        color: #0c5460;
        font-weight: 600;
    }
    
    /* PREDICTION RESULT SECTION */
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .prediction-price {
        font-size: 2.5em;
        font-weight: 800;
        margin: 15px 0;
    }
    
    /* SPECIFICATION SUMMARY TABLE */
    .spec-table {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-top: 15px;
    }
    
    /* CHART CONTAINER */
    .chart-container {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* DIVIDER */
    hr {
        margin: 30px 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    
    /* RESPONSIVE */
    @media (max-width: 768px) {
        .header-title {
            font-size: 1.8em;
        }
        
        .prediction-price {
            font-size: 2em;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 10px 15px;
            font-size: 0.9em;
        }
    }
    
    /* SCROLLBAR STYLING */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER SECTION
# =====================================================
st.markdown("""
<div class="header-container">
    <div class="header-title">💻 Laptop Price Analytics Dashboard</div>
    <div class="header-subtitle">🚀 Market Analysis & Machine Learning Price Prediction</div>
    <div class="header-authors">👨‍💼 Talha Bashir, Abdullah | Roll No: 2430-0162, 2430-0067 | PAI Course Project</div>
</div>
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
st.sidebar.markdown("## 🎛️ Filters")
st.sidebar.markdown("---")

companies = sorted(df["Company"].unique())
types = sorted(df["TypeName"].unique())
selected_companies = st.sidebar.multiselect("🏢 Company", companies, default=companies[:5])
selected_types = st.sidebar.multiselect("💻 Laptop Type", types, default=types)

price_min, price_max = float(df["Price"].min()), float(df["Price"].max())
price_range = st.sidebar.slider("💰 Price Range (₹)", price_min, price_max, (price_min, price_max))

filtered_df = df[
    (df["Company"].isin(selected_companies)) &
    (df["TypeName"].isin(selected_types)) &
    (df["Price"].between(price_range[0], price_range[1]))
]

if filtered_df.empty:
    st.warning("⚠️ No data matches the selected filters.")
    st.stop()

# =====================================================
# TABS
# =====================================================
tabs = st.tabs(["🏠 Overview", "📊 Market", "💻 Specs", "📈 Trends", "🤖 ML Model", "💾 Export"])

# =====================================================
# TAB 1: OVERVIEW
# =====================================================
with tabs[0]:
    st.markdown("### 📊 Quick Overview")
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Price</div>
            <div class="metric-value">₹{filtered_df['Price'].mean():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Laptops</div>
            <div class="metric-value">{len(filtered_df)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg RAM</div>
            <div class="metric-value">{filtered_df['Ram_GB'].mean():.1f} GB</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Companies</div>
            <div class="metric-value">{filtered_df['Company'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📈 Top 10 Companies by Average Price")
    fig = px.bar(
        filtered_df.groupby("Company")["Price"].mean().sort_values(ascending=False).head(10).reset_index(),
        x="Price",
        y="Company",
        orientation="h",
        title="",
        color="Price",
        color_continuous_scale="Viridis",
        labels={"Price": "Average Price (₹)", "Company": ""}
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        hovermode="y unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 2: MARKET
# =====================================================
with tabs[1]:
    st.markdown("### 🏆 Market Share Analysis")
    
    c1, c2 = st.columns([1.2, 1])
    
    with c1:
        share_df = filtered_df["Company"].value_counts().head(8).reset_index()
        share_df.columns = ["Company", "Count"]
        fig = px.pie(
            share_df,
            names="Company",
            values="Count",
            hole=0.4,
            title="Market Share (Top 8 Companies)"
        )
        fig.update_layout(
            height=400,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown("#### 📊 Market Statistics")
        market_stats = {
            "Total Models": len(filtered_df),
            "Avg Price": f"₹{filtered_df['Price'].mean():,.0f}",
            "Price Range": f"₹{filtered_df['Price'].min():,.0f} - ₹{filtered_df['Price'].max():,.0f}",
            "Most Common Type": filtered_df['TypeName'].mode()[0] if not filtered_df['TypeName'].mode().empty else "N/A",
            "Most Popular Company": filtered_df['Company'].mode()[0] if not filtered_df['Company'].mode().empty else "N/A"
        }
        
        for stat, value in market_stats.items():
            st.markdown(f"**{stat}:** {value}")

# =====================================================
# TAB 3: SPECS
# =====================================================
with tabs[2]:
    st.markdown("### 🔧 Specifications Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### RAM Impact on Price")
        fig = px.box(
            filtered_df,
            x="Ram_GB",
            y="Price",
            title="",
            labels={"Ram_GB": "RAM (GB)", "Price": "Price (₹)"},
            color="Ram_GB",
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Screen Size vs Price")
        fig = px.scatter(
            filtered_df,
            x="Inches",
            y="Price",
            size="Weight_kg",
            color="Company",
            title="",
            labels={"Inches": "Screen Size (inches)", "Price": "Price (₹)", "Weight_kg": "Weight (kg)"},
            hover_data=["Company", "TypeName", "Ram_GB"]
        )
        fig.update_layout(
            height=400,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hovermode="closest"
        )
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 4: TRENDS
# =====================================================
with tabs[3]:
    st.markdown("### 📈 Price Distribution & Trends")
    
    fig = px.histogram(
        filtered_df,
        x="Price",
        color="Price_Category",
        title="Price Distribution by Category",
        labels={"Price": "Price (₹)", "count": "Number of Laptops"},
        nbins=30
    )
    fig.update_layout(
        height=450,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        budget_count = len(filtered_df[filtered_df['Price_Category'] == 'Budget'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Budget Laptops</div>
            <div class="metric-value">{budget_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        premium_count = len(filtered_df[filtered_df['Price_Category'] == 'Premium'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Premium Laptops</div>
            <div class="metric-value">{premium_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        luxury_count = len(filtered_df[filtered_df['Price_Category'] == 'Luxury'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Luxury Laptops</div>
            <div class="metric-value">{luxury_count}</div>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# TAB 5: ML MODEL
# =====================================================
with tabs[4]:
    st.markdown("### 🤖 Machine Learning Price Prediction")
    
    st.markdown("---")
    st.markdown("#### ⚙️ Model Training Configuration")
    
    colA, colB, colC = st.columns(3)
    
    with colA:
        n_estimators = st.slider("🌲 Number of Trees", 50, 300, 150, step=10)
    
    with colB:
        max_depth = st.slider("📊 Max Depth", 5, 40, 20, step=1)
    
    with colC:
        test_size = st.slider("🧪 Test Size (%)", 10, 40, 20, step=5) / 100
    
    if st.button("🔄 Train Model", use_container_width=True):
        with st.spinner("🔄 Training model... Please wait..."):
            features = ["Ram_GB", "Inches", "Weight_kg", "Company", "TypeName", "OpSys", "Cpu_Speed_GHz", "SSD"]
            model_df = filtered_df.dropna(subset=features + ["Price"]).copy()
            
            if len(model_df) < 10:
                st.error("❌ Not enough data to train the model.")
            else:
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
                
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                # Store in session
                st.session_state.model = model
                st.session_state.le_dict = le_dict
                st.session_state.features = features
                st.session_state.model_trained = True
                
                # Metrics
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                mae = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                
                st.success("✅ Model trained successfully!")
                
                m1, m2, m3 = st.columns(3)
                
                with m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">RMSE</div>
                        <div class="metric-value">₹{rmse:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">MAE</div>
                        <div class="metric-value">₹{mae:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">R² Score</div>
                        <div class="metric-value">{r2:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("#### 📊 Actual vs Predicted Prices")
                fig = px.scatter(
                    x=y_test,
                    y=preds,
                    labels={'x': 'Actual Price (₹)', 'y': 'Predicted Price (₹)'},
                    title="",
                    opacity=0.6,
                    color_discrete_sequence=['#667eea']
                )
                fig.add_shape(
                    type="line",
                    x0=y_test.min(),
                    y0=y_test.min(),
                    x1=y_test.max(),
                    y1=y_test.max(),
                    line=dict(dash="dash", color="red", width=2),
                    name="Perfect Prediction"
                )
                fig.update_layout(
                    height=400,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    hovermode="closest"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Price Prediction Section
    st.markdown("---")
    st.markdown("#### 📋 Predict Price for Your Laptop")
    
    if 'model_trained' in st.session_state and st.session_state.model_trained:
        st.info("✅ Model is ready! Fill in the specifications below to predict the price.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Select Company:**")
            selected_company = st.selectbox("Company", sorted(df["Company"].unique()), key="company_select")
            
            st.markdown("**Select Laptop Type:**")
            selected_type = st.selectbox("Laptop Type", sorted(df["TypeName"].unique()), key="type_select")
            
            st.markdown("**RAM Configuration:**")
            ram_gb = st.number_input("RAM (GB)", min_value=2, max_value=64, value=8, step=1, key="ram_input")
        
        with col2:
            st.markdown("**Display Size:**")
            inches = st.number_input("Screen Size (Inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1, key="inches_input")
            
            st.markdown("**Weight:**")
            weight_kg = st.number_input("Weight (kg)", min_value=1.0, max_value=4.0, value=1.8, step=0.1, key="weight_input")
            
            st.markdown("**Processor Speed:**")
            cpu_speed = st.number_input("CPU Speed (GHz)", min_value=1.0, max_value=5.0, value=2.5, step=0.1, key="cpu_input")
        
        with col3:
            st.markdown("**Operating System:**")
            selected_os = st.selectbox("Operating System", sorted(df["OpSys"].unique()), key="os_select")
            
            st.markdown("**Storage:**")
            ssd = st.number_input("SSD Storage (GB)", min_value=0, max_value=2048, value=512, step=128, key="ssd_input")
        
        st.markdown("---")
        
        if st.button("🚀 Predict Price", use_container_width=True):
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
                st.markdown("""
                <div class="prediction-result">
                    <div style="font-size: 1.2em; opacity: 0.9;">Predicted Laptop Price</div>
                    <div class="prediction-price">₹{:,.0f}</div>
                </div>
                """.format(predicted_price), unsafe_allow_html=True)
                
                # Show input summary
                st.markdown("---")
                st.markdown("#### 📋 Your Laptop Specifications")
                
                summary_data = {
                    "Company": selected_company,
                    "Laptop Type": selected_type,
                    "RAM": f"{ram_gb} GB",
                    "Screen Size": f"{inches}\" inches",
                    "Weight": f"{weight_kg} kg",
                    "CPU Speed": f"{cpu_speed} GHz",
                    "SSD Storage": f"{ssd} GB",
                    "Operating System": selected_os
                }
                
                summary_df = pd.DataFrame(
                    list(summary_data.items()),
                    columns=["Specification", "Value"]
                )
                
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.info("💡 Make sure all values are within the valid range.")
    else:
        st.warning("⚠️ Please train the model first by clicking the '🔄 Train Model' button above!")

# =====================================================
# TAB 6: EXPORT
# =====================================================
with tabs[5]:
    st.markdown("### 📥 Export Data")
    
    st.markdown(f"**Total Records:** {len(filtered_df)}")
    
    st.markdown("---")
    st.markdown("#### 📊 Filtered Data Preview")
    
    st.dataframe(filtered_df, use_container_width=True, height=400)
    
    st.markdown("---")
    st.markdown("#### 📥 Download Options")
    
    c1, c2 = st.columns(2)
    
    with c1:
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download as CSV",
            csv,
            f"laptop_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with c2:
        # Excel download
        try:
            import io
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, sheet_name='Laptops', index=False)
            excel_data = excel_buffer.getvalue()
            
            st.download_button(
                "📊 Download as Excel",
                excel_data,
                f"laptop_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except:
            st.info("Excel export requires openpyxl library")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #667eea; padding: 20px; font-size: 0.9em;">
    <p>💻 Laptop Price Analytics Dashboard | Developed for PAI Course Project</p>
    <p>© 2024 Talha Bashir & Abdullah | Roll No: 2430-0162, 2430-0067</p>
</div>
""", unsafe_allow_html=True)
