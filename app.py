import io
import base64
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

st.set_page_config(page_title="Laptop Price Analytics", layout="wide", page_icon="💻")

# Custom color scheme - Modern tech-inspired palette
PRIMARY_COLOR = "#00D9FF"
SECONDARY_COLOR = "#FF6B35"
ACCENT_COLOR = "#4ECDC4"
DARK_BG = "#1A1A2E"
LIGHT_TEXT = "#EAEAEA"

st.markdown(f"""
<style>
body {{
  background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%) !important;
}}

.block-container {{
  padding-top: 1.5rem;
}}

.main-header {{
  background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {SECONDARY_COLOR} 100%);
  padding: 2.5rem;
  border-radius: 25px;
  margin-bottom: 2rem;
  box-shadow: 0 15px 40px rgba(0,0,0,0.3);
  color: white;
  border: 2px solid rgba(255,255,255,0.1);
}}

.stTabs [data-baseweb="tab-list"] {{
  background: linear-gradient(90deg, rgba(0, 217, 255, 0.15) 0%, rgba(78, 205, 196, 0.15) 100%);
  border-radius: 15px;
  padding: 0.6rem;
  gap: 0.6rem;
  border: 1px solid rgba(255,255,255,0.1);
}}

.stTabs [data-baseweb="tab"] {{
  background: rgba(26, 26, 46, 0.6);
  border-radius: 12px;
  padding: 0.9rem 1.8rem;
  transition: all 0.3s ease;
  font-weight: 600;
  border: 2px solid transparent;
  color: #EAEAEA;
}}

.stTabs [data-baseweb="tab"]:hover {{
  background: rgba(0, 217, 255, 0.2);
  transform: translateY(-3px);
  border-color: {ACCENT_COLOR};
  box-shadow: 0 5px 15px rgba(0, 217, 255, 0.3);
}}

.stTabs [data-baseweb="tab"][aria-selected="true"] {{
  background: linear-gradient(135deg, {PRIMARY_COLOR}, {ACCENT_COLOR});
  color: white;
  box-shadow: 0 5px 20px rgba(0, 217, 255, 0.4);
}}

[data-testid="stMetric"] {{
  background: linear-gradient(135deg, rgba(0, 217, 255, 0.1), rgba(78, 205, 196, 0.1));
  padding: 1.2rem;
  border-radius: 18px;
  border: 2px solid rgba(0, 217, 255, 0.3);
  backdrop-filter: blur(15px);
  box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}}

section[data-testid="stSidebar"] > div {{
  background: linear-gradient(180deg, rgba(15, 32, 39, 0.95) 0%, rgba(32, 58, 67, 0.95) 100%);
  border-right: 2px solid rgba(0, 217, 255, 0.2);
}}

h1, h2, h3, h4, h5, h6, p, label, span, div {{
  color: #EAEAEA !important;
}}

.stButton > button {{
  background: linear-gradient(135deg, {PRIMARY_COLOR}, {ACCENT_COLOR});
  color: white;
  border: none;
  border-radius: 12px;
  padding: 0.6rem 1.8rem;
  font-weight: 700;
  transition: all 0.3s ease;
  box-shadow: 0 4px 12px rgba(0, 217, 255, 0.3);
}}

.stButton > button:hover {{
  transform: translateY(-3px);
  box-shadow: 0 8px 20px rgba(0, 217, 255, 0.5);
}}

.info-box {{
  background: linear-gradient(135deg, rgba(255, 107, 53, 0.15), rgba(0, 217, 255, 0.15));
  padding: 1.5rem;
  border-radius: 18px;
  border-left: 5px solid {ACCENT_COLOR};
  margin: 1rem 0;
  box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}}

.stat-card {{
  background: linear-gradient(135deg, rgba(26, 26, 46, 0.8), rgba(44, 83, 100, 0.6));
  padding: 1.5rem;
  border-radius: 18px;
  border: 2px solid rgba(0, 217, 255, 0.3);
  margin: 0.5rem 0;
  box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}}

</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="main-header">
  <h1 style='margin:0; font-size: 2.8rem;'>💻 Laptop Price Analytics Dashboard</h1>
  <p style='margin:0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.95;'>
    Comprehensive analysis and ML-powered price prediction for laptop market data
  </p>
  <p style='margin:0.5rem 0 0 0; font-size: 0.95rem; opacity: 0.75;'>
    Developed by <strong>Talha Bashir</strong> (Roll No: 2430-0162) | Programming for AI Course Project
  </p>
</div>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading laptop data...")
def load_laptop_data():
    try:
        df = pd.read_csv("laptopData.csv")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Drop unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Convert Price to numeric
        if 'Price' in df.columns:
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        
        # Extract numeric values from Ram (e.g., "8GB" -> 8)
        if 'Ram' in df.columns:
            df['Ram_GB'] = df['Ram'].str.extract('(\d+)').astype(float)
        
        # Extract screen size
        if 'Inches' in df.columns:
            df['Inches'] = pd.to_numeric(df['Inches'], errors='coerce')
        
        # Extract weight
        if 'Weight' in df.columns:
            df['Weight_kg'] = df['Weight'].str.extract('(\d+\.?\d*)').astype(float)
        
        # Create price categories
        if 'Price' in df.columns:
            df['Price_Category'] = pd.cut(df['Price'], 
                                         bins=[0, 30000, 60000, 100000, float('inf')],
                                         labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

with st.spinner('🔄 Loading laptop data...'):
    df = load_laptop_data()

if df.empty:
    st.error("⚠️ Unable to load laptopData.csv. Please ensure the file exists in the same directory.")
    st.stop()

# Sidebar
st.sidebar.markdown("### 🎛️ Control Panel")
st.sidebar.markdown("---")

# Company filter
companies = ['All'] + sorted(df['Company'].unique().tolist())
selected_companies = st.sidebar.multiselect(
    '🏢 Select Companies',
    options=companies[1:],
    default=companies[1:6]
)

# Type filter
laptop_types = ['All'] + sorted(df['TypeName'].unique().tolist())
selected_types = st.sidebar.multiselect(
    '💼 Select Laptop Types',
    options=laptop_types[1:],
    default=laptop_types[1:]
)

# Price range filter
price_min = float(df['Price'].min())
price_max = float(df['Price'].max())
price_range = st.sidebar.slider(
    '💰 Price Range (₹)',
    min_value=price_min,
    max_value=price_max,
    value=(price_min, price_max),
    format="₹%.0f"
)

# Apply filters
filtered_df = df.copy()
if selected_companies:
    filtered_df = filtered_df[filtered_df['Company'].isin(selected_companies)]
if selected_types:
    filtered_df = filtered_df[filtered_df['TypeName'].isin(selected_types)]
filtered_df = filtered_df[(filtered_df['Price'] >= price_range[0]) & 
                          (filtered_df['Price'] <= price_range[1])]

st.sidebar.markdown('---')
st.sidebar.markdown('### ⚡ Quick Actions')

col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button('🔥 Top 5', use_container_width=True):
        top_companies = df.groupby('Company').size().nlargest(5).index.tolist()
        st.rerun()

with col_b:
    if st.button('🔄 Reset', use_container_width=True):
        st.rerun()

st.sidebar.markdown('---')
st.sidebar.markdown('### 📊 Dataset Info')
st.sidebar.info(f"""
**Total Laptops**: {len(df):,}  
**Companies**: {df['Company'].nunique()}  
**Laptop Types**: {df['TypeName'].nunique()}  
**Price Range**: ₹{df['Price'].min():,.0f} - ₹{df['Price'].max():,.0f}
""")

st.sidebar.markdown('---')
st.sidebar.markdown('### 👨‍🎓 Student Info')
st.sidebar.success("""
**Talha Bashir**  
Roll No: **2430-0162**  
PAI Course Project
""")

tabs = st.tabs(["🏠 Overview", "📊 Market Analysis", "💻 Specifications", "📈 Price Trends", "🤖 ML Predictions", "💾 Export Data"])

# TAB 1: Overview
with tabs[0]:
    st.markdown("### 📊 Market Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        avg_price = filtered_df['Price'].mean()
        st.metric('💵 Avg Price', f'₹{avg_price:,.0f}')
    with col2:
        total_laptops = len(filtered_df)
        st.metric('💻 Total Laptops', f'{total_laptops:,}')
    with col3:
        avg_ram = filtered_df['Ram_GB'].mean()
        st.metric('🧠 Avg RAM', f'{avg_ram:.1f}GB')
    with col4:
        avg_screen = filtered_df['Inches'].mean()
        st.metric('📺 Avg Screen', f'{avg_screen:.1f}"')
    with col5:
        companies_count = filtered_df['Company'].nunique()
        st.metric('🏢 Companies', f'{companies_count}')
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💰 Price Distribution by Company")
        company_prices = filtered_df.groupby('Company')['Price'].mean().sort_values(ascending=False).head(10)
        fig = px.bar(company_prices, 
                     orientation='h',
                     labels={'value': 'Average Price (₹)', 'Company': 'Company'},
                     color=company_prices.values,
                     color_continuous_scale='Turbo')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Laptop Type Distribution")
        type_counts = filtered_df['TypeName'].value_counts().head(10)
        fig = px.pie(values=type_counts.values, 
                     names=type_counts.index,
                     hole=0.4,
                     color_discrete_sequence=px.colors.sequential.Turbo)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### 💻 Top 10 Most Expensive Laptops")
    top_expensive = filtered_df.nlargest(10, 'Price')[['Company', 'TypeName', 'Cpu', 'Ram', 'Price']]
    top_expensive['Price'] = top_expensive['Price'].apply(lambda x: f'₹{x:,.0f}')
    st.dataframe(top_expensive.reset_index(drop=True), use_container_width=True)

# TAB 2: Market Analysis
with tabs[1]:
    st.markdown("### 🏢 Market Share & Competition Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Market Share by Company")
        company_counts = filtered_df['Company'].value_counts().head(8)
        fig = px.pie(values=company_counts.values,
                     names=company_counts.index,
                     title='Top 8 Companies by Product Count',
                     color_discrete_sequence=px.colors.sequential.Teal)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Average Price by Company")
        company_avg_price = filtered_df.groupby('Company')['Price'].mean().sort_values(ascending=False).head(8)
        fig = px.bar(company_avg_price,
                     title='Top 8 Companies by Average Price',
                     labels={'value': 'Avg Price (₹)', 'Company': 'Company'},
                     color=company_avg_price.values,
                     color_continuous_scale='Plasma')
        fig.update_layout(showlegend=False, height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Operating System Distribution")
        os_counts = filtered_df['OpSys'].value_counts()
        fig = px.bar(os_counts,
                     labels={'value': 'Count', 'index': 'Operating System'},
                     color=os_counts.values,
                     color_continuous_scale='Viridis')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Price Range Distribution")
        fig = px.histogram(filtered_df, x='Price', nbins=30,
                          labels={'Price': 'Price (₹)'},
                          color_discrete_sequence=[PRIMARY_COLOR])
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

# TAB 3: Specifications
with tabs[2]:
    st.markdown("### 💻 Technical Specifications Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### RAM Distribution")
        ram_counts = filtered_df['Ram'].value_counts().sort_index()
        fig = px.bar(ram_counts,
                     labels={'value': 'Count', 'index': 'RAM'},
                     color=ram_counts.values,
                     color_continuous_scale='Blues')
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Screen Size Distribution")
        screen_counts = filtered_df['Inches'].value_counts().sort_index()
        fig = px.bar(screen_counts,
                     labels={'value': 'Count', 'index': 'Screen Size (inches)'},
                     color=screen_counts.values,
                     color_continuous_scale='Greens')
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("#### Weight Distribution")
        fig = px.histogram(filtered_df, x='Weight_kg', nbins=20,
                          labels={'Weight_kg': 'Weight (kg)'},
                          color_discrete_sequence=[SECONDARY_COLOR])
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Specification Impact on Price")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### RAM vs Price")
        fig = px.box(filtered_df, x='Ram', y='Price',
                     color='Ram',
                     labels={'Price': 'Price (₹)', 'Ram': 'RAM'},
                     color_discrete_sequence=px.colors.sequential.Turbo)
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Screen Size vs Price")
        fig = px.scatter(filtered_df, x='Inches', y='Price',
                        color='Company', size='Weight_kg',
                        labels={'Price': 'Price (₹)', 'Inches': 'Screen Size (inches)'},
                        hover_data=['Company', 'TypeName'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# TAB 4: Price Trends
with tabs[3]:
    st.markdown("### 📈 Price Analysis & Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Price by Type")
        type_price = filtered_df.groupby('TypeName')['Price'].agg(['mean', 'min', 'max']).sort_values('mean', ascending=False)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Average', x=type_price.index, y=type_price['mean'],
                            marker_color=PRIMARY_COLOR))
        fig.add_trace(go.Scatter(name='Max', x=type_price.index, y=type_price['max'],
                                mode='markers', marker=dict(size=10, color=SECONDARY_COLOR)))
        fig.update_layout(height=400, barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Price Distribution by Category")
        fig = px.box(filtered_df, x='Price_Category', y='Price',
                     color='Price_Category',
                     labels={'Price': 'Price (₹)', 'Price_Category': 'Category'},
                     color_discrete_sequence=px.colors.sequential.Plasma)
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("#### 🔥 Price Comparison Heatmap")
    
    # Create pivot table for heatmap
    heatmap_data = filtered_df.groupby(['Company', 'TypeName'])['Price'].mean().unstack(fill_value=0)
    
    if not heatmap_data.empty:
        fig = px.imshow(heatmap_data,
                       labels=dict(x="Laptop Type", y="Company", color="Avg Price (₹)"),
                       aspect="auto",
                       color_continuous_scale='Turbo')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# TAB 5: ML Predictions
with tabs[4]:
    st.markdown("### 🤖 Machine Learning - Price Prediction Model")
    st.markdown("**Predicting laptop prices using Random Forest Regressor**")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 🎛️ Model Configuration")
        
        n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
        max_depth = st.slider("Max Depth", 5, 50, 20, 5)
        min_samples_split = st.slider("Min Samples Split", 2, 20, 5, 1)
        test_size = st.slider("Test Size (%)", 10, 40, 20, 5) / 100
        
        train_button = st.button("🚀 Train Model", use_container_width=True, type="primary")
        
        st.markdown("---")
        st.info("""
        **Features:**
        - RAM (GB)
        - Screen Size (inches)
        - Weight (kg)
        - Company (encoded)
        - Type (encoded)
        - OS (encoded)
        """)
    
    with col2:
        if train_button or 'model_trained' not in st.session_state:
            with st.spinner('Training model...'):
                # Prepare features
                model_df = filtered_df.copy()
                
                # Encode categorical variables
                le_company = LabelEncoder()
                le_type = LabelEncoder()
                le_os = LabelEncoder()
                
                model_df['Company_encoded'] = le_company.fit_transform(model_df['Company'])
                model_df['Type_encoded'] = le_type.fit_transform(model_df['TypeName'])
                model_df['OS_encoded'] = le_os.fit_transform(model_df['OpSys'])
                
                # Select features
                feature_cols = ['Ram_GB', 'Inches', 'Weight_kg', 'Company_encoded', 
                               'Type_encoded', 'OS_encoded']
                
                # Remove rows with missing values
                model_df = model_df.dropna(subset=feature_cols + ['Price'])
                
                X = model_df[feature_cols].values
                y = model_df['Price'].values
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_pred = model.predict(X_test_scaled)
                
                # Metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Store in session
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['feature_cols'] = feature_cols
                st.session_state['model_trained'] = True
                st.session_state['metrics'] = {'rmse': rmse, 'mae': mae, 'r2': r2, 'mse': mse}
                st.session_state['predictions'] = {'y_test': y_test, 'y_pred': y_pred}
                st.session_state['encoders'] = {
                    'company': le_company,
                    'type': le_type,
                    'os': le_os
                }
        
        if 'model_trained' in st.session_state and st.session_state['model_trained']:
            st.markdown("#### ✅ Model Performance Metrics")
            
            metrics = st.session_state['metrics']
            
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
            with col_b:
                st.metric("RMSE", f"₹{metrics['rmse']:,.0f}")
            with col_c:
                st.metric("MAE", f"₹{metrics['mae']:,.0f}")
            with col_d:
                st.metric("MSE", f"₹{metrics['mse']:,.0f}")
            
            st.markdown("---")
            st.markdown("#### 📊 Predictions vs Actual Prices")
            
            preds = st.session_state['predictions']
            
            # Sample for visualization
            sample_size = min(500, len(preds['y_test']))
            indices = np.random.choice(len(preds['y_test']), sample_size, replace=False)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=preds['y_test'][indices],
                y=preds['y_pred'][indices],
                mode='markers',
                name='Predictions',
                marker=dict(size=8, color=preds['y_test'][indices],
                           colorscale='Turbo', showscale=True)
            ))
            
            max_val = max(preds['y_test'].max(), preds['y_pred'].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash', width=3)
            ))
            
            fig.update_layout(
                xaxis_title='Actual Price (₹)',
                yaxis_title='Predicted Price (₹)',
                height=450, hovermode='closest'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### 🌲 Feature Importance")
            
            model = st.session_state['model']
            feature_cols = st.session_state['feature_cols']
            
            importances = model.feature_importances_
            feature_imp_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(feature_imp_df, x='Importance', y='Feature',
                        orientation='h', color='Importance',
                        color_continuous_scale='Teal')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

# TAB 6: Export
with tabs[5]:
    st.markdown("### 💾 Data Export & Downloads")
    
    st.markdown("#### 📄 Preview Filtered Data")
    st.dataframe(filtered_df.head(20), use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### ⬇️ Download Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Filtered Data (CSV)",
            data=csv,
            file_name=f'laptops_filtered_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    with col2:
        full_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Complete Dataset (CSV)",
            data=full_csv,
            file_name=f'laptops_complete_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    with col3:
        if not filtered_df.empty:
            summary = filtered_df.groupby('Company').agg({
                'Price': ['mean', 'min', 'max', 'count'],
                'Ram_GB': 'mean',
                'Inches': 'mean'
            }).round(2)
            summary_csv = summary.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Company Summary (CSV)",
                data=summary_csv,
                file_name=f'company_summary_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
                mime='text/csv',
                use_container_width=True
            )
    
    st.markdown("---")
    st.markdown("#### 📊 Statistical Summary")
    
    if not filtered_df.empty:
        summary_stats = filtered_df[['Price', 'Ram_GB', 'Inches', 'Weight_kg']].describe()