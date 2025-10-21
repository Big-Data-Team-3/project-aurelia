import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import json

# Page configuration
st.set_page_config(
    page_title="Project Aurelia",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Project Aurelia</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Data Explorer", "Analytics", "Settings"]
    )
    
    # Sidebar settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("Settings")
    
    # Theme selector
    theme = st.sidebar.selectbox(
        "Select Theme",
        ["Light", "Dark", "Auto"]
    )
    
    # Sample data toggle
    use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True)
    
    # Main content based on selected page
    if page == "Dashboard":
        show_dashboard(use_sample_data)
    elif page == "Data Explorer":
        show_data_explorer(use_sample_data)
    elif page == "Analytics":
        show_analytics(use_sample_data)
    elif page == "Settings":
        show_settings()

def show_dashboard(use_sample_data):
    st.header("📊 Dashboard")
    
    # Create sample data if needed
    if use_sample_data:
        data = generate_sample_data()
    else:
        data = load_user_data()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Records",
            value=len(data),
            delta=12
        )
    
    with col2:
        st.metric(
            label="Average Score",
            value=f"{data['score'].mean():.2f}",
            delta=0.5
        )
    
    with col3:
        st.metric(
            label="Active Users",
            value=data['active'].sum(),
            delta=-3
        )
    
    with col4:
        st.metric(
            label="Revenue",
            value=f"${data['revenue'].sum():,.0f}",
            delta=1500
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Score Distribution")
        fig_hist = px.histogram(
            data, 
            x='score', 
            nbins=20,
            title="Distribution of Scores"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("📊 Revenue by Category")
        fig_bar = px.bar(
            data.groupby('category')['revenue'].sum().reset_index(),
            x='category',
            y='revenue',
            title="Revenue by Category"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Time series chart
    st.subheader("📅 Revenue Over Time")
    data['date'] = pd.to_datetime(data['date'])
    daily_revenue = data.groupby(data['date'].dt.date)['revenue'].sum().reset_index()
    
    fig_line = px.line(
        daily_revenue,
        x='date',
        y='revenue',
        title="Daily Revenue Trend"
    )
    st.plotly_chart(fig_line, use_container_width=True)

def show_data_explorer(use_sample_data):
    st.header("🔍 Data Explorer")
    
    # Create sample data if needed
    if use_sample_data:
        data = generate_sample_data()
    else:
        data = load_user_data()
    
    # Data overview
    st.subheader("Data Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Shape:** {data.shape}")
        st.write(f"**Columns:** {list(data.columns)}")
    
    with col2:
        st.write(f"**Memory Usage:** {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        st.write(f"**Data Types:**")
        for col, dtype in data.dtypes.items():
            st.write(f"  - {col}: {dtype}")
    
    # Data table with filters
    st.subheader("Data Table")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        categories = st.multiselect(
            "Filter by Category",
            options=data['category'].unique(),
            default=data['category'].unique()
        )
    
    with col2:
        score_range = st.slider(
            "Score Range",
            min_value=float(data['score'].min()),
            max_value=float(data['score'].max()),
            value=(float(data['score'].min()), float(data['score'].max()))
        )
    
    with col3:
        active_filter = st.selectbox(
            "Active Status",
            ["All", "Active", "Inactive"]
        )
    
    # Apply filters
    filtered_data = data[
        (data['category'].isin(categories)) &
        (data['score'] >= score_range[0]) &
        (data['score'] <= score_range[1])
    ]
    
    if active_filter != "All":
        active_bool = active_filter == "Active"
        filtered_data = filtered_data[filtered_data['active'] == active_bool]
    
    # Display filtered data
    st.dataframe(
        filtered_data,
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def show_analytics(use_sample_data):
    st.header("📈 Analytics")
    
    # Create sample data if needed
    if use_sample_data:
        data = generate_sample_data()
    else:
        data = load_user_data()
    
    # Advanced analytics
    st.subheader("Statistical Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Numerical Columns:**")
        st.dataframe(data.describe())
    
    with col2:
        st.write("**Categorical Columns:**")
        for col in data.select_dtypes(include=['object']).columns:
            st.write(f"**{col}:**")
            st.write(data[col].value_counts().head())
    
    # Correlation matrix
    st.subheader("Correlation Matrix")
    numeric_data = data.select_dtypes(include=[np.number])
    if len(numeric_data.columns) > 1:
        corr_matrix = numeric_data.corr()
        fig_corr = px.imshow(
            corr_matrix,
            title="Correlation Matrix",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Advanced visualizations
    st.subheader("Advanced Visualizations")
    
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Scatter Plot", "Box Plot", "Violin Plot", "Heatmap"]
    )
    
    if viz_type == "Scatter Plot":
        x_col = st.selectbox("X-axis", numeric_data.columns)
        y_col = st.selectbox("Y-axis", numeric_data.columns)
        color_col = st.selectbox("Color by", data.columns)
        
        fig_scatter = px.scatter(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            title=f"{y_col} vs {x_col}"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    elif viz_type == "Box Plot":
        y_col = st.selectbox("Y-axis", numeric_data.columns)
        x_col = st.selectbox("X-axis (categorical)", data.select_dtypes(include=['object']).columns)
        
        fig_box = px.box(
            data,
            x=x_col,
            y=y_col,
            title=f"Box Plot: {y_col} by {x_col}"
        )
        st.plotly_chart(fig_box, use_container_width=True)

def show_settings():
    st.header("⚙️ Settings")
    
    st.subheader("Application Settings")
    
    # Configuration options
    auto_refresh = st.checkbox("Auto-refresh data", value=False)
    if auto_refresh:
        refresh_interval = st.slider("Refresh interval (seconds)", 5, 300, 30)
        st.info(f"Data will refresh every {refresh_interval} seconds")
    
    # Data source settings
    st.subheader("Data Source")
    data_source = st.radio(
        "Select data source",
        ["Sample Data", "Upload File", "Database Connection"]
    )
    
    if data_source == "Upload File":
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv"
        )
        if uploaded_file is not None:
            st.success("File uploaded successfully!")
    
    elif data_source == "Database Connection":
        st.text_input("Database URL")
        st.text_input("Username")
        st.text_input("Password", type="password")
    
    # Export settings
    st.subheader("Export Settings")
    export_format = st.selectbox(
        "Default export format",
        ["CSV", "Excel", "JSON", "Parquet"]
    )
    
    # Save settings
    if st.button("Save Settings"):
        settings = {
            "auto_refresh": auto_refresh,
            "data_source": data_source,
            "export_format": export_format
        }
        st.success("Settings saved successfully!")

def generate_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    n_records = 1000
    
    data = pd.DataFrame({
        'id': range(1, n_records + 1),
        'date': pd.date_range('2024-01-01', periods=n_records, freq='D'),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_records),
        'score': np.random.normal(75, 15, n_records).clip(0, 100),
        'revenue': np.random.exponential(1000, n_records),
        'active': np.random.choice([True, False], n_records, p=[0.7, 0.3]),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_records)
    })
    
    return data

def load_user_data():
    """Load user's actual data - implement based on your needs"""
    # Placeholder for actual data loading logic
    st.warning("No user data loaded. Using sample data.")
    return generate_sample_data()

# Cache data to improve performance
@st.cache_data
def load_cached_data():
    return generate_sample_data()

if __name__ == "__main__":
    main()
