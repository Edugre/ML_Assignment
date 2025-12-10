import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Page configuration
st.set_page_config(
    page_title="Product Sales ML Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        padding: 0.5rem 0;
        border-bottom: 3px solid #3498db;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        raw_data = pd.read_csv('data/product_sales.csv')
        preprocessed_data = pd.read_csv('results/preprocessed_data_outliers_capped.csv')
        cluster_data = pd.read_csv('results/data_with_clusters.csv')
        cluster_stats = pd.read_csv('results/cluster_analysis.csv')
        regression_comparison = pd.read_csv('results/regression_comparison.csv')
        
        return {
            'raw': raw_data,
            'preprocessed': preprocessed_data,
            'clustered': cluster_data,
            'cluster_stats': cluster_stats,
            'regression': regression_comparison
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">Product Sales ML Analysis</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Select Section:",
            ["🏠 Overview", "🔍 Data Analysis", "🎯 K-means Clustering", "📈 Regression Analysis", "💡 Insights & Recommendations"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        ML analysis:
        - Data preprocessing
        - K-means clustering
        - Regression modeling
        """)
    
    # Load data
    data = load_data()
    
    if data is None:
        st.error("⚠️ Please run the analysis scripts first!")
        st.code("""
        python source/preprocessing.py
        python source/kmeans.py
        python source/regression.py
        """)
        return
    
    # Page routing
    if page == "🏠 Overview":
        show_overview(data)
    elif page == "🔍 Data Analysis":
        show_data_analysis(data)
    elif page == "🎯 K-means Clustering":
        show_clustering(data)
    elif page == "📈 Regression Analysis":
        show_regression(data)
    elif page == "💡 Insights & Recommendations":
        show_insights(data)

def show_overview(data):
    st.markdown('<h2 class="section-header">🏠 Project Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📦 Total Products",
            value=len(data['raw']),
            delta="Analyzed"
        )
    
    with col2:
        st.metric(
            label="🎯 Clusters Identified",
            value=4,
            delta="Optimal K"
        )
    
    with col3:
        best_r2 = data['regression']['Test_R2'].max()
        st.metric(
            label="📈 Best Model R²",
            value=f"{best_r2:.2%}",
            delta="Excellent"
        )
    
    with col4:
        total_profit = data['raw']['profit'].sum()
        st.metric(
            label="💰 Total Profit",
            value=f"${total_profit:,.2f}",
            delta="Revenue"
        )
    
    st.markdown("---")
    
    # Project description
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Project Objectives")
        st.markdown("""
        <div class="info-box">
        <b>This analysis aims to:</b>
        <ul>
            <li>🔍 <b>Understand product sales patterns</b> through comprehensive data analysis</li>
            <li>🎯 <b>Segment products</b> using K-means clustering to identify distinct product groups</li>
            <li>📈 <b>Predict product profit</b> using advanced regression models</li>
            <li>💡 <b>Generate actionable insights</b> for business decision-making</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Analysis Pipeline")
        st.markdown("""
        <div class="success-box">
        <b>Completed Steps:</b>
        <ol>
            <li>✅ <b>Data Preprocessing:</b> Handling missing values, outlier detection and treatment</li>
            <li>✅ <b>K-means Clustering:</b> Product segmentation with elbow method optimization</li>
            <li>✅ <b>Regression Analysis:</b> Multiple models tested for profit prediction</li>
            <li>✅ <b>Visualization & Reporting:</b> Interactive dashboard and insights</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📈 Quick Stats")
        
        # Dataset summary
        st.markdown("**Dataset Information**")
        st.info(f"""
        - **Rows:** {len(data['raw'])}
        - **Columns:** {len(data['raw'].columns)}
        - **Categories:** {data['raw']['category'].nunique()}
        - **Price Range:** ${data['raw']['price'].min():.2f} - ${data['raw']['price'].max():.2f}
        """)
        
        st.markdown("**Model Performance**")
        best_model = data['regression'].iloc[0]
        st.success(f"""
        **Best Model:** {best_model['Model']}
        - **R² Score:** {best_model['Test_R2']:.4f}
        - **MAE:** ${best_model['Test_MAE']:.2f}
        """)
    
    # Dataset preview
    st.markdown("---")
    st.markdown("### 📋 Dataset Preview")
    st.dataframe(
        data['raw'].head(10),
        use_container_width=True,
        hide_index=True
    )

def show_data_analysis(data):
    st.markdown('<h2 class="section-header">🔍 Data Analysis & Preprocessing</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Statistics", "🧹 Preprocessing", "📈 Distributions"])
    
    with tab1:
        st.markdown("### 📊 Descriptive Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Numerical Features**")
            st.dataframe(
                data['raw'][['price', 'cost', 'units_sold', 'profit']].describe(),
                use_container_width=True
            )
        
        with col2:
            st.markdown("**Category Distribution**")
            category_counts = data['raw']['category'].value_counts()
            
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Products by Category",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 🧹 Preprocessing Summary")
        
        st.markdown("""
        <div class="info-box">
        <h4>Steps Performed:</h4>
        <ul>
            <li>✅ <b>Missing Value Imputation:</b> Filled missing product names and numerical values</li>
            <li>✅ <b>Outlier Detection:</b> Used IQR method to identify outliers</li>
            <li>✅ <b>Outlier Treatment:</b> Capped extreme values to reduce impact</li>
            <li>✅ <b>Feature Normalization:</b> Min-Max scaling and Z-score standardization</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Before Preprocessing**")
            st.metric("Missing Values", data['raw'].isnull().sum().sum())
            st.metric("Dataset Shape", f"{data['raw'].shape[0]} × {data['raw'].shape[1]}")
        
        with col2:
            st.markdown("**After Preprocessing**")
            st.metric("Missing Values", data['preprocessed'].isnull().sum().sum())
            st.metric("Dataset Shape", f"{data['preprocessed'].shape[0]} × {data['preprocessed'].shape[1]}")
    
    with tab3:
        st.markdown("### 📈 Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution
            fig = px.histogram(
                data['raw'],
                x='price',
                nbins=30,
                title="Price Distribution",
                labels={'price': 'Price ($)', 'count': 'Frequency'},
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Profit distribution
            fig = px.histogram(
                data['raw'],
                x='profit',
                nbins=30,
                title="Profit Distribution",
                labels={'profit': 'Profit ($)', 'count': 'Frequency'},
                color_discrete_sequence=['#764ba2']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Units sold distribution
            fig = px.histogram(
                data['raw'],
                x='units_sold',
                nbins=30,
                title="Units Sold Distribution",
                labels={'units_sold': 'Units Sold', 'count': 'Frequency'},
                color_discrete_sequence=['#f093fb']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Promotion frequency
            fig = px.histogram(
                data['raw'],
                x='promotion_frequency',
                title="Promotion Frequency Distribution",
                labels={'promotion_frequency': 'Promotion Frequency', 'count': 'Frequency'},
                color_discrete_sequence=['#4facfe']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def show_clustering(data):
    st.markdown('<h2 class="section-header">🎯 K-means Clustering Analysis</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📉 Elbow Method", "🎨 Cluster Visualization", "📊 Cluster Statistics"])
    
    with tab1:
        st.markdown("### 📉 Elbow Method for Optimal K")
        
        st.markdown("""
        <div class="info-box">
        The elbow method helps determine the optimal number of clusters by plotting WCSS 
        (Within-Cluster Sum of Squares) for different k values. The "elbow" point indicates 
        where adding more clusters provides diminishing returns.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display elbow curve
            if os.path.exists('results/elbow_curve.png'):
                st.image('results/elbow_curve.png', use_container_width=True)
            else:
                st.warning("Elbow curve image not found. Please run kmeans.py first.")
        
        with col2:
            st.markdown("### 🎯 Results")
            st.success("""
            **Optimal K: 4**
            
            The elbow occurs at k=4, providing:
            - Good cluster separation
            - Meaningful business segments
            - Manageable complexity
            """)
            
            # WCSS values
            wcss_data = {
                'k': [2, 3, 4, 5, 6, 7, 8],
                'WCSS': [399.34, 271.34, 223.41, 171.58, 151.78, 132.29, 111.52]
            }
            st.dataframe(pd.DataFrame(wcss_data), use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("### 🎨 Cluster Visualization")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            if os.path.exists('results/cluster_scatter.png'):
                st.image('results/cluster_scatter.png', use_container_width=True)
            else:
                # Create interactive scatter plot
                fig = px.scatter(
                    data['clustered'],
                    x='price',
                    y='units_sold',
                    color='cluster',
                    title="Product Clusters: Price vs Units Sold",
                    labels={'cluster': 'Cluster', 'price': 'Price ($)', 'units_sold': 'Units Sold'},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    hover_data=['product_name', 'profit']
                )
                fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📋 Cluster Names")
            
            for idx, row in data['cluster_stats'].iterrows():
                with st.expander(f"**Cluster {row['Cluster']}: {row['Cluster_Name']}**"):
                    st.write(f"**Products:** {row['Count']}")
                    st.write(f"**Avg Price:** ${row['Avg_Price']:.2f}")
                    st.write(f"**Avg Units Sold:** {row['Avg_Units_Sold']:.0f}")
                    st.write(f"**Avg Profit:** ${row['Avg_Profit']:.2f}")
                    st.write("---")
                    st.info(row['Business_Insight'])
    
    with tab3:
        st.markdown("### 📊 Cluster Statistics")
        
        # Format the cluster stats table
        display_stats = data['cluster_stats'][['Cluster', 'Cluster_Name', 'Count', 'Avg_Price', 
                                                'Avg_Units_Sold', 'Avg_Profit', 'Total_Profit']].copy()
        display_stats['Avg_Price'] = display_stats['Avg_Price'].apply(lambda x: f"${x:.2f}")
        display_stats['Avg_Units_Sold'] = display_stats['Avg_Units_Sold'].apply(lambda x: f"{x:.0f}")
        display_stats['Avg_Profit'] = display_stats['Avg_Profit'].apply(lambda x: f"${x:.2f}")
        display_stats['Total_Profit'] = display_stats['Total_Profit'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(display_stats, use_container_width=True, hide_index=True)
        
        # Cluster comparison charts
        st.markdown("### 📈 Cluster Comparisons")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                data['cluster_stats'],
                x='Cluster_Name',
                y='Count',
                title="Number of Products per Cluster",
                labels={'Cluster_Name': 'Cluster', 'Count': 'Number of Products'},
                color='Count',
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                data['cluster_stats'],
                x='Cluster_Name',
                y='Total_Profit',
                title="Total Profit by Cluster",
                labels={'Cluster_Name': 'Cluster', 'Total_Profit': 'Total Profit ($)'},
                color='Total_Profit',
                color_continuous_scale='Greens'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def show_regression(data):
    st.markdown('<h2 class="section-header">📈 Regression Analysis</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🏆 Model Comparison", "📊 Best Model Performance", "📉 Detailed Results"])
    
    with tab1:
        st.markdown("### 🏆 Model Comparison")
        
        # Format regression data
        display_reg = data['regression'].copy()
        display_reg['Test_R2'] = display_reg['Test_R2'].apply(lambda x: f"{x:.4f}")
        display_reg['Test_MSE'] = display_reg['Test_MSE'].apply(lambda x: f"{x:.2f}")
        display_reg['Test_MAE'] = display_reg['Test_MAE'].apply(lambda x: f"{x:.2f}")
        display_reg['Overfitting'] = display_reg['Overfitting'].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(
            display_reg[['Model', 'Test_R2', 'Test_MSE', 'Test_MAE', 'Overfitting']],
            use_container_width=True,
            hide_index=True
        )
        
        # Comparison chart
        if os.path.exists('results/model_comparison.png'):
            st.image('results/model_comparison.png', use_container_width=True)
        else:
            # Create interactive comparison
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Test R² Score', 'Test MSE', 'Test MAE', 'Overfitting Gap')
            )
            
            models = data['regression']['Model'].values
            
            fig.add_trace(
                go.Bar(x=models, y=data['regression']['Test_R2'], name='R²', marker_color='steelblue'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(x=models, y=data['regression']['Test_MSE'], name='MSE', marker_color='coral'),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(x=models, y=data['regression']['Test_MAE'], name='MAE', marker_color='lightgreen'),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(x=models, y=data['regression']['Overfitting'], name='Overfitting', marker_color='orange'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 📊 Best Model: Polynomial Regression (degree=3)")
        
        best_model = data['regression'].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="R² Score",
                value=f"{best_model['Test_R2']:.4f}",
                delta="93.2% accuracy"
            )
        
        with col2:
            st.metric(
                label="MSE",
                value=f"{best_model['Test_MSE']:.2f}",
                delta="Lower is better"
            )
        
        with col3:
            st.metric(
                label="MAE",
                value=f"${best_model['Test_MAE']:.2f}",
                delta="Avg error"
            )
        
        with col4:
            st.metric(
                label="Overfitting Gap",
                value=f"{best_model['Overfitting']:.4f}",
                delta="Minimal" if best_model['Overfitting'] < 0.1 else "High"
            )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Actual vs Predicted")
            if os.path.exists('results/actual_vs_predicted.png'):
                st.image('results/actual_vs_predicted.png', use_container_width=True)
            else:
                st.warning("Image not found")
        
        with col2:
            st.markdown("#### 📉 Residual Analysis")
            if os.path.exists('results/residual_plot.png'):
                st.image('results/residual_plot.png', use_container_width=True)
            else:
                st.warning("Image not found")
        
        st.markdown("---")
        st.markdown("#### 🎯 Feature Importance")
        if os.path.exists('results/feature_importance.png'):
            st.image('results/feature_importance.png', use_container_width=True)
        else:
            st.info("Feature importance available for linear regression model")
    
    with tab3:
        st.markdown("### 📉 Detailed Analysis")
        
        st.markdown("""
        <div class="success-box">
        <h4>🎯 Why Polynomial Regression (degree=3) Performs Best:</h4>
        <ul>
            <li><b>Captures Non-Linear Relationships:</b> Profit has multiplicative relationships with price and units sold</li>
            <li><b>Interaction Effects:</b> Polynomial features model interactions between variables (e.g., price × units_sold)</li>
            <li><b>Optimal Complexity:</b> Degree 3 provides best balance between accuracy and generalization</li>
            <li><b>No Significant Overfitting:</b> Small gap between train and test R² (6.4%)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Model Tradeoffs:</h4>
        <ul>
            <li><b>Linear Regression:</b> Simple and interpretable but poor accuracy (73.75%)</li>
            <li><b>Polynomial (degree=2):</b> Good balance, 92.64% accuracy</li>
            <li><b>Polynomial (degree=3):</b> Best accuracy (93.20%) with acceptable complexity</li>
            <li><b>Ridge Regression:</b> Prevents overfitting but doesn't capture non-linear patterns well</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Performance Metrics Explained")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **R² Score (0.9320)**
            
            Indicates that 93.2% of profit variance is explained by the model. 
            Excellent performance!
            """)
        
        with col2:
            st.info("""
            **MSE (2,312.64)**
            
            Mean Squared Error measures average squared prediction error.
            74% lower than linear regression.
            """)
        
        with col3:
            st.info("""
            **MAE ($19.74)**
            
            Average prediction error is only $19.74.
            Very accurate for business decisions.
            """)

def show_insights(data):
    st.markdown('<h2 class="section-header">💡 Key Insights & Recommendations</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🎯 Clustering Insights", "📈 Regression Insights", "💼 Business Recommendations"])
    
    with tab1:
        st.markdown("### 🎯 Product Segmentation Insights")
        
        st.markdown("""
        <div class="success-box">
        <h4>✅ Key Findings from K-means Clustering:</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for idx, row in data['cluster_stats'].iterrows():
            with st.expander(f"**Cluster {row['Cluster']}: {row['Cluster_Name']}** ({row['Count']} products)"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.metric("Avg Price", f"${row['Avg_Price']:.2f}")
                    st.metric("Avg Units Sold", f"{row['Avg_Units_Sold']:.0f}")
                    st.metric("Total Profit", f"${row['Total_Profit']:,.2f}")
                
                with col2:
                    st.markdown("**Business Insight:**")
                    st.info(row['Business_Insight'])
    
    with tab2:
        st.markdown("### 📈 Profit Prediction Insights")
        
        best_model = data['regression'].iloc[0]
        
        st.markdown("""
        <div class="success-box">
        <h4>✅ Regression Analysis Findings:</h4>
        <ul>
            <li>🎯 <b>Prediction Accuracy:</b> 93.2% - Excellent for profit forecasting</li>
            <li>📊 <b>Average Error:</b> Only $19.74 per product</li>
            <li>🔄 <b>Non-Linear Relationships:</b> Polynomial features essential for accuracy</li>
            <li>✅ <b>Model Reliability:</b> No significant overfitting detected</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🔍 Key Drivers of Profit")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Primary Factors:**
            1. 📦 **Units Sold** - Strongest positive impact
            2. 💰 **Price** - Positive correlation
            3. 💸 **Cost** - Negative impact (as expected)
            """)
        
        with col2:
            st.markdown("""
            **Secondary Factors:**
            4. 🎁 **Promotion Frequency** - Moderate positive effect
            5. 📍 **Shelf Level** - Minor impact on profit
            """)
        
        if os.path.exists('results/feature_importance.png'):
            st.image('results/feature_importance.png', use_container_width=True)
    
    with tab3:
        st.markdown("### 💼 Strategic Business Recommendations")
        
        st.markdown("""
        <div class="info-box">
        <h3>🎯 Cluster-Specific Strategies</h3>
        </div>
        """, unsafe_allow_html=True)
        
        recommendations = {
            "Budget Best-Sellers": {
                "icon": "🛒",
                "focus": "Volume & Availability",
                "actions": [
                    "Prioritize inventory management and stock availability",
                    "Optimize supply chain for high-volume products",
                    "Use as traffic drivers in store layout",
                    "Consider bulk purchasing to reduce costs",
                    "Bundle with higher-margin products"
                ]
            },
            "Mid-Range Steady Performers": {
                "icon": "⭐",
                "focus": "Maintain & Expand",
                "actions": [
                    "This is your profit sweet spot - maintain current strategy!",
                    "Monitor competition closely in this segment",
                    "Consider expanding product lines in this category",
                    "Test minor price optimizations",
                    "Ensure consistent quality and availability"
                ]
            },
            "Premium Specialty": {
                "icon": "💎",
                "focus": "Quality & Exclusivity",
                "actions": [
                    "Maintain premium positioning and quality standards",
                    "Implement targeted marketing to high-value customers",
                    "Consider exclusive launches or limited editions",
                    "Focus on brand storytelling and differentiation",
                    "Minimal need for promotions - leverage organic demand"
                ]
            }
        }
        
        for name, rec in recommendations.items():
            with st.expander(f"{rec['icon']} **{name}**"):
                st.markdown(f"**Strategic Focus:** {rec['focus']}")
                st.markdown("**Recommended Actions:**")
                for action in rec['actions']:
                    st.markdown(f"- {action}")
        
        st.markdown("---")
        
        st.markdown("""
        <div class="success-box">
        <h3>📊 Data-Driven Decision Making</h3>
        <ul>
            <li>🎯 <b>Use the prediction model</b> for profit forecasting of new products</li>
            <li>📈 <b>Test pricing strategies</b> with confidence using the regression model</li>
            <li>🔄 <b>Monitor cluster movements</b> - products may shift between segments</li>
            <li>📊 <b>Regular model updates</b> recommended as new data becomes available</li>
            <li>💡 <b>A/B test recommendations</b> before full implementation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Next Steps")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Short-term Actions (1-3 months):**
            - Implement inventory optimization for Budget Best-Sellers
            - Review pricing for underperforming products
            - Test targeted promotions on Mid-Range products
            - Audit supply chain efficiency
            """)
        
        with col2:
            st.markdown("""
            **Long-term Strategy (3-12 months):**
            - Expand Mid-Range Steady Performers category
            - Develop premium product lines
            - Implement dynamic pricing based on predictions
            - Build customer segmentation aligned with product clusters
            """)

if __name__ == "__main__":
    main()
