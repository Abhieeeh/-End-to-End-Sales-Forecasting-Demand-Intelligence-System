import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Sales Demand Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM LOOK & FEEL ---
st.markdown("""
<style>
    /* Main container styling */
    .reportview-container {
        background: #0F0F1A;
    }
    
    /* Title styling */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(135deg, #00F5FF, #8A2BE2, #FF7F50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .section-subtitle {
        font-family: 'Inter', sans-serif;
        color: #B0B0C3;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphism KPI Card Styling */
    .kpi-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .kpi-card {
        flex: 1;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 245, 255, 0.3);
        box-shadow: 0 12px 40px 0 rgba(0, 245, 255, 0.15);
    }
    
    .kpi-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8A90A6;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        font-family: 'Outfit', sans-serif;
        line-height: 1.2;
    }
    
    .kpi-delta {
        font-size: 0.85rem;
        margin-top: 8px;
        font-weight: 500;
    }
    
    .delta-up {
        color: #00FF87;
    }
    
    /* Sidebar styling modifications */
    [data-testid="stSidebar"] {
        background-color: #0B0B13;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Tables and dataframes */
    [data-testid="stTable"] {
        background-color: transparent;
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA CACHING ---
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "..", "Dataset", "train.csv")
    df = pd.read_csv(train_path)
    # Parse dates with mixed formats
    df["Order Date"] = pd.to_datetime(df["Order Date"], format='mixed')
    df["Ordered Year"] = df["Order Date"].dt.year
    df["Ordered Month"] = df["Order Date"].dt.month
    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("<h2 style='text-align: center; color: #00F5FF; font-family: Outfit;'>Navigation</h2>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "Select Page",
    [
        "📊 Sales Overview",
        "🔮 Forecast Explorer",
        "🚨 Anomaly Report",
        "🧩 Demand Segments"
    ]
)

# Shared directories for charts
base_dir = os.path.dirname(os.path.abspath(__file__))
charts_dir = os.path.join(base_dir, "..", "Charts")

# ==========================================
# PAGE 1: SALES OVERVIEW DASHBOARD
# ==========================================
if "Sales Overview" in page:
    st.markdown("<h1 class='main-title'>Sales Overview Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='section-subtitle'>Analyze overall sales trends, growth patterns, and regional distributions with dynamic filtering.</p>", unsafe_allow_html=True)
    
    # Sidebar filters
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Interactive Filters")
    
    all_regions = sorted(df_raw["Region"].unique())
    selected_regions = st.sidebar.multiselect(
        "Select Region(s)",
        options=all_regions,
        default=all_regions
    )
    
    all_categories = sorted(df_raw["Category"].unique())
    selected_categories = st.sidebar.multiselect(
        "Select Category(s)",
        options=all_categories,
        default=all_categories
    )
    
    # Filter dataset
    if not selected_regions or not selected_categories:
        st.warning("Please select at least one Region and one Category from the sidebar.")
        st.stop()
        
    df_filtered = df_raw[
        (df_raw["Region"].isin(selected_regions)) & 
        (df_raw["Category"].isin(selected_categories))
    ]
    
    # Compute KPIs
    total_sales = df_filtered["Sales"].sum()
    total_orders = df_filtered["Order ID"].nunique()
    avg_sales_per_order = total_sales / total_orders if total_orders > 0 else 0.0
    
    # Display styled KPI Cards
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Total Revenue</div>
            <div class="kpi-value">${total_sales:,.2f}</div>
            <div class="kpi-delta delta-up">★ High Performance</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Total Volume (Orders)</div>
            <div class="kpi-value">{total_orders:,}</div>
            <div class="kpi-delta delta-up">✓ Unique transactions</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Avg Ticket Size</div>
            <div class="kpi-value">${avg_sales_per_order:,.2f}</div>
            <div class="kpi-delta delta-up">✦ Value per order</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dashboard Grid
    col1, col2 = st.columns(2)
    
    # Chart 1: Total Sales by Year (Bar Chart)
    with col1:
        yearly_sales = df_filtered.groupby("Ordered Year")["Sales"].sum().reset_index()
        yearly_sales["Ordered Year"] = yearly_sales["Ordered Year"].astype(str)
        
        fig_year = px.bar(
            yearly_sales,
            x="Ordered Year",
            y="Sales",
            title="Total Sales by Year",
            labels={"Ordered Year": "Year", "Sales": "Sales ($)"},
            color="Sales",
            color_continuous_scale="Viridis",
        )
        fig_year.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=40, t=50, b=40),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_year, use_container_width=True)
        
    # Chart 2: Monthly Sales Trend (Line Chart)
    with col2:
        monthly_sales = df_filtered.set_index("Order Date")["Sales"].resample("ME").sum().reset_index()
        
        fig_month = px.line(
            monthly_sales,
            x="Order Date",
            y="Sales",
            title="Monthly Sales Trend",
            labels={"Order Date": "Date", "Sales": "Sales ($)"},
            markers=True
        )
        fig_month.update_traces(
            line=dict(color="#00F5FF", width=3),
            marker=dict(size=6, color="#8A2BE2")
        )
        fig_month.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_month, use_container_width=True)
        
    # Section 3: Sales by Region and Category (Interactive Filter Breakdown)
    st.markdown("### Sales Distribution Breakdown")
    breakdown_df = df_filtered.groupby(["Region", "Category"])["Sales"].sum().reset_index()
    
    fig_breakdown = px.bar(
        breakdown_df,
        x="Region",
        y="Sales",
        color="Category",
        barmode="group",
        title="Sales by Region and Category",
        color_discrete_sequence=["#00F5FF", "#8A2BE2", "#FF7F50"],
        labels={"Sales": "Sales ($)"}
    )
    fig_breakdown.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_breakdown, use_container_width=True)

# ==========================================
# PAGE 2: FORECAST EXPLORER
# ==========================================
elif "Forecast Explorer" in page:
    st.markdown("<h1 class='main-title'>Forecast Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='section-subtitle'>Evaluate model errors and project future demand using the best-performing SARIMA model.</p>", unsafe_allow_html=True)
    
    # Secondary navigation / dropdown inputs
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        segment_type = st.selectbox(
            "Forecast Segment Level",
            options=["Category", "Region"]
        )
        
    with col_input2:
        if segment_type == "Category":
            options_val = sorted(df_raw["Category"].unique())
        else:
            options_val = sorted(df_raw["Region"].unique())
            
        selected_val = st.selectbox(
            f"Select {segment_type}",
            options=options_val
        )
        
    with col_input3:
        # Date range slider to select forecast horizon (1, 2, or 3 months ahead)
        horizon = st.slider(
            "Forecast Horizon (Months Ahead)",
            min_value=1,
            max_value=3,
            value=3
        )
        
    # Helper to prepare series
    @st.cache_data
    def get_aggregated_series(segment_type, value):
        filtered = df_raw[df_raw[segment_type] == value]
        series = filtered.set_index("Order Date")["Sales"].resample("ME").sum()
        return series

    # Helper for model evaluation & future forecasting
    @st.cache_resource
    def fit_and_forecast_sarima(segment_type, value, horizon):
        series = get_aggregated_series(segment_type, value)
        
        # 1. Validation Model (12-month holdout)
        train = series[:-12]
        test = series[-12:]
        val_model = SARIMAX(
            train,
            order=(0, 1, 1),
            seasonal_order=(0, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        val_fit = val_model.fit(disp=False)
        val_pred = val_fit.forecast(steps=12)
        
        # Validation metrics (calculated on the 12-month validation holdout)
        mae = mean_absolute_error(test, val_pred)
        rmse = np.sqrt(mean_squared_error(test, val_pred))
        
        # 2. Production Model (Entire Series)
        prod_model = SARIMAX(
            series,
            order=(0, 1, 1),
            seasonal_order=(0, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        prod_fit = prod_model.fit(disp=False)
        
        # Future predictions
        forecast_res = prod_fit.get_forecast(steps=horizon)
        forecast_mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int()
        
        return series, forecast_mean, conf_int, mae, rmse

    with st.spinner("Training best SARIMA model on selected segment..."):
        series, forecast_mean, conf_int, mae, rmse = fit_and_forecast_sarima(segment_type, selected_val, horizon)
        
    # Plotly visualization
    # We display the last 24 months of history + the future forecast horizon
    history_to_plot = series[-24:]
    
    fig_fc = go.Figure()
    
    # Historical Sales
    fig_fc.add_trace(go.Scatter(
        x=history_to_plot.index,
        y=history_to_plot.values,
        mode="lines+markers",
        name="Historical Sales",
        line=dict(color="#00F5FF", width=3),
        marker=dict(size=6)
    ))
    
    # Forecasted Sales
    fig_fc.add_trace(go.Scatter(
        x=forecast_mean.index,
        y=forecast_mean.values,
        mode="lines+markers",
        name="Predicted Sales",
        line=dict(color="#FF7F50", width=3, dash="dash"),
        marker=dict(size=6, symbol="diamond")
    ))
    
    # Confidence Interval Shading
    fig_fc.add_trace(go.Scatter(
        x=list(conf_int.index) + list(conf_int.index)[::-1],
        y=list(conf_int.iloc[:, 0]) + list(conf_int.iloc[:, 1])[::-1],
        fill="toself",
        fillcolor="rgba(255, 127, 80, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="95% Confidence Interval"
    ))
    
    fig_fc.update_layout(
        title=f"Sales Forecast for {segment_type}: {selected_val}",
        xaxis_title="Month",
        yaxis_title="Sales ($)",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    st.plotly_chart(fig_fc, use_container_width=True)
    
    # Model performance below the chart
    st.markdown("### Model Evaluation Metrics (12-Month Holdout Validation)")
    col_metric1, col_metric2 = st.columns(2)
    
    with col_metric1:
        st.metric(
            label="Mean Absolute Error (MAE)",
            value=f"${mae:,.2f}"
        )
    with col_metric2:
        st.metric(
            label="Root Mean Squared Error (RMSE)",
            value=f"${rmse:,.2f}"
        )
        
    st.info("The forecasting is powered by a SARIMA(0,1,1)x(0,1,1)₁₂ seasonal model, identified as the best-performing model for this sales dataset based on the validation process.")

# ==========================================
# PAGE 3: ANOMALY REPORT
# ==========================================
elif "Anomaly Report" in page:
    st.markdown("<h1 class='main-title'>Anomaly Detection Report</h1>", unsafe_allow_html=True)
    st.markdown("<p class='section-subtitle'>Review detected weekly sales anomalies identified using isolation forest machine learning models.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Anomaly Detection Chart (Task 5)")
        # Display the anomaly chart image from Task 5
        anomaly_path = os.path.join(charts_dir, "Isolation Forest - Weekly Sales Anomalies.png")
        if os.path.exists(anomaly_path):
            st.image(anomaly_path, use_container_width=True)
        else:
            st.warning("Anomaly chart image not found at 'Charts/Isolation Forest - Weekly Sales Anomalies.png'. Please ensure the task has run successfully.")
            
    with col2:
        st.markdown("### Detected Anomaly Weekly Sales")
        
        # Calculate weekly sales anomalies dynamically
        weekly_sales = df_raw.set_index("Order Date")["Sales"].resample("W").sum().reset_index()
        
        iso = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )
        weekly_sales["anomolies"] = iso.fit_predict(weekly_sales[["Sales"]])
        anomalies_df = weekly_sales[weekly_sales["anomolies"] == -1].copy()
        
        # Clean formatting
        anomalies_df["Order Date"] = anomalies_df["Order Date"].dt.strftime("%Y-%m-%d")
        anomalies_df = anomalies_df.rename(columns={"Order Date": "Week Start Date", "Sales": "Weekly Sales ($)"})
        anomalies_df = anomalies_df.sort_values(by="Week Start Date", ascending=False)
        
        # Display in table
        st.dataframe(
            anomalies_df[["Week Start Date", "Weekly Sales ($)"]],
            column_config={
                "Week Start Date": "Anomaly Week",
                "Weekly Sales ($)": st.column_config.NumberColumn(format="$%,.2f")
            },
            hide_index=True,
            use_container_width=True
        )

# ==========================================
# PAGE 4: PRODUCT DEMAND SEGMENTS
# ==========================================
elif "Demand Segments" in page:
    st.markdown("<h1 class='main-title'>Product Demand Segments</h1>", unsafe_allow_html=True)
    st.markdown("<p class='section-subtitle'>Analyze sub-category demand groupings segmented via unsupervised K-Means clustering.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Product Demand Segments Chart (Task 6)")
        # Display the cluster chart image from Task 6
        cluster_path = os.path.join(charts_dir, "demand_clusters.png")
        if os.path.exists(cluster_path):
            st.image(cluster_path, use_container_width=True)
        else:
            st.warning("Clustering chart image not found at 'Charts/demand_clusters.png'. Please ensure the K-Means script has run successfully.")
            
    with col2:
        st.markdown("### Sub-Category Demand Groupings")
        
        # Calculate sub-category clusters dynamically
        sub_data = df_raw.groupby("Sub-Category").agg(
            Total_Sales=("Sales", "sum"),
            Order_Count=("Sales", "count")
        ).reset_index()

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(sub_data[["Total_Sales", "Order_Count"]])

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        sub_data["Cluster"] = kmeans.fit_predict(scaled_features)

        # Label clusters logically based on average sales
        cluster_summary = sub_data.groupby("Cluster").mean(numeric_only=True)
        sorted_clusters = cluster_summary.sort_values(by="Total_Sales").index.tolist()
        cluster_mapping = {
            sorted_clusters[0]: "Low Demand",
            sorted_clusters[1]: "Moderate Demand",
            sorted_clusters[2]: "High Demand"
        }
        sub_data["Demand Cluster"] = sub_data["Cluster"].map(cluster_mapping)
        
        # Style and sort
        sub_display_df = sub_data.sort_values(by=["Demand Cluster", "Total_Sales"], ascending=[True, False])
        
        st.dataframe(
            sub_display_df[["Sub-Category", "Total_Sales", "Order_Count", "Demand Cluster"]],
            column_config={
                "Sub-Category": "Sub-Category",
                "Total_Sales": st.column_config.NumberColumn("Total Sales", format="$%,.2f"),
                "Order_Count": st.column_config.NumberColumn("Order Count", format="%,d"),
                "Demand Cluster": "Demand Segment"
            },
            hide_index=True,
            use_container_width=True
        )
