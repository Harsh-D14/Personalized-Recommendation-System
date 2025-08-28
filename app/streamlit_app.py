"""
Fixed Streamlit Web Application for InstaCart Recommendation System
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.models.user_based_cf import UserBasedCF
from src.models.item_based_cf import ItemBasedCF
from src.models.svd_model import SVDRecommender
from src.models.nmf_model import NMFRecommender

# Page configuration
st.set_page_config(
    page_title="InstaCart Recommendation System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_all_models():
    """Load all trained models"""
    models = {}
    models_dir = Path('data/models')

    # Model file mappings
    model_files = {
        'User-Based CF': 'user_cf.pkl',
        'Item-Based CF': 'item_cf.pkl',
        'SVD': 'svd.pkl',
        'NMF': 'nmf.pkl'
    }

    for name, filename in model_files.items():
        filepath = models_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    models[name] = pickle.load(f)
                st.sidebar.success(f"✓ Loaded {name}")
            except Exception as e:
                st.sidebar.warning(f"⚠ Could not load {name}: {str(e)[:30]}")
        else:
            st.sidebar.info(f"ℹ {name} not found")

    if not models:
        st.sidebar.error("No models found! Please run train_all_models.py first")

    return models


@st.cache_data
def load_all_data():
    """Load necessary data"""
    data_loader = DataLoader()

    try:
        # Load processed data
        user_item_matrix = data_loader.load_processed_data("user_item_matrix")

        # Load product information
        products_df = pd.read_csv('data/raw/products.csv')
        departments_df = pd.read_csv('data/raw/departments.csv')
        aisles_df = pd.read_csv('data/raw/aisles.csv')

        # Merge product info
        products_full = products_df.merge(departments_df, on='department_id')
        products_full = products_full.merge(aisles_df, on='aisle_id')

        product_names = dict(zip(products_df['product_id'], products_df['product_name']))

        return user_item_matrix, products_full, product_names
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


def display_recommendations(recommendations, scores, product_names):
    """Display recommendations in a nice format"""
    for i, (item_id, score) in enumerate(zip(recommendations, scores), 1):
        col1, col2, col3 = st.columns([5, 2, 1])

        with col1:
            product_name = product_names.get(item_id, f"Product {item_id}")
            st.write(f"**{i}. {product_name}**")

        with col2:
            # Star rating based on score
            if score > 2.0:
                stars = "⭐⭐⭐⭐⭐"
            elif score > 1.5:
                stars = "⭐⭐⭐⭐"
            elif score > 1.0:
                stars = "⭐⭐⭐"
            elif score > 0.5:
                stars = "⭐⭐"
            else:
                stars = "⭐"
            st.write(stars)

        with col3:
            st.metric("Score", f"{score:.2f}", label_visibility="collapsed")


def main():
    # Title
    st.title("🛒 InstaCart Recommendation System")
    st.markdown("### AI-Powered Grocery Shopping Recommendations")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Load models
        models = load_all_models()

        if models:
            selected_model = st.selectbox(
                "Select Model",
                options=list(models.keys()),
                help="Choose the recommendation algorithm"
            )

            n_recommendations = st.slider(
                "Number of Recommendations",
                min_value=5,
                max_value=20,
                value=10
            )

            st.markdown("---")
            st.header("📊 Model Info")

            if selected_model in models:
                model = models[selected_model]
                st.text(f"Model: {selected_model}")
                st.text(f"Fitted: ✓")
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    with st.expander("Parameters"):
                        for key, value in params.items():
                            if key not in ['name', 'is_fitted']:
                                st.text(f"{key}: {value}")

    # Load data
    user_item_matrix, products_full, product_names = load_all_data()

    if user_item_matrix is None or not models:
        st.error("⚠️ Please ensure data is loaded and models are trained")
        st.info("Run: `python scripts/train_all_models.py` to train models")
        return

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Get Recommendations",
        "🔍 Find Similar Products",
        "📈 Model Performance",
        "📊 Data Insights"
    ])

    # Tab 1: Recommendations
    with tab1:
        st.header("Get Personalized Recommendations")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Select User")

            # User selection
            if st.button("🎲 Random User"):
                st.session_state['selected_user_idx'] = np.random.choice(len(user_item_matrix))

            user_idx = st.number_input(
                "User Index",
                min_value=0,
                max_value=len(user_item_matrix)-1,
                value=st.session_state.get('selected_user_idx', 0)
            )
            st.session_state['selected_user_idx'] = user_idx

            # User info
            user_id = user_item_matrix.index[user_idx]
            purchased_items = user_item_matrix.columns[user_item_matrix.iloc[user_idx] > 0].tolist()

            st.info(f"**User {user_id}** | **{len(purchased_items)}** products purchased")

            with st.expander("View Purchase History"):
                for i, item_id in enumerate(purchased_items[:10], 1):
                    if item_id in product_names:
                        st.write(f"{i}. {product_names[item_id]}")
                if len(purchased_items) > 10:
                    st.write(f"... and {len(purchased_items)-10} more")

        with col2:
            st.subheader(f"Recommendations: {selected_model}")

            if st.button("🚀 Generate Recommendations", type="primary"):
                with st.spinner("Generating recommendations..."):
                    try:
                        model = models[selected_model]
                        recommendations, scores = model.recommend_items(
                            user_idx,
                            n_recommendations
                        )

                        # Map to product IDs
                        rec_product_ids = user_item_matrix.columns[recommendations].tolist()

                        st.success("✓ Recommendations generated!")
                        display_recommendations(rec_product_ids, scores, product_names)

                    except Exception as e:
                        st.error(f"Error: {e}")

    # Tab 2: Similar Products
    with tab2:
        st.header("Find Similar Products")

        if "Item-Based CF" in models:
            search_term = st.text_input("🔍 Search for a product", placeholder="e.g., Banana, Milk, Bread")

            if search_term:
                # Find matching products
                matching = [(pid, pname) for pid, pname in product_names.items()
                          if search_term.lower() in pname.lower()]

                if matching:
                    # Show matches
                    selected_product = st.selectbox(
                        "Select product:",
                        options=matching[:20],
                        format_func=lambda x: x[1]
                    )

                    if st.button("Find Similar Products"):
                        if selected_product[0] in user_item_matrix.columns:
                            try:
                                item_idx = user_item_matrix.columns.get_loc(selected_product[0])
                                model = models["Item-Based CF"]

                                similar_items, similarity_scores = model.find_similar_items(
                                    item_idx,
                                    n_recommendations
                                )

                                st.subheader(f"Products similar to: **{selected_product[1]}**")

                                similar_ids = user_item_matrix.columns[similar_items].tolist()
                                for i, (item_id, sim) in enumerate(zip(similar_ids, similarity_scores), 1):
                                    if item_id in product_names:
                                        col1, col2 = st.columns([4, 1])
                                        with col1:
                                            st.write(f"{i}. {product_names[item_id]}")
                                        with col2:
                                            st.metric("Similarity", f"{sim:.3f}", label_visibility="collapsed")

                            except Exception as e:
                                st.error(f"Error: {e}")
                        else:
                            st.warning("Product not in interaction matrix")
                else:
                    st.warning(f"No products found matching '{search_term}'")
        else:
            st.info("Item-Based CF model required for this feature")

    # Tab 3: Model Performance
    with tab3:
        st.header("Model Performance Metrics")

        # Load comparison data
        comparison_file = Path('data/models/model_comparison.csv')
        if comparison_file.exists():
            comparison_df = pd.read_csv(comparison_file)

            # Display metrics table
            st.subheader("📊 Performance Comparison")

            # Format the dataframe
            formatted_df = comparison_df.copy()
            for col in formatted_df.columns:
                if col != 'model' and 'time' not in col:
                    formatted_df[col] = formatted_df[col].round(4)

            st.dataframe(formatted_df, use_container_width=True)

            # Visualizations
            col1, col2 = st.columns(2)

            with col1:
                # RMSE and MAE
                fig_error = go.Figure()
                fig_error.add_trace(go.Bar(
                    name='RMSE',
                    x=comparison_df['model'],
                    y=comparison_df['rmse'],
                    marker_color='indianred'
                ))
                fig_error.add_trace(go.Bar(
                    name='MAE',
                    x=comparison_df['model'],
                    y=comparison_df['mae'],
                    marker_color='lightblue'
                ))
                fig_error.update_layout(
                    title="Error Metrics (Lower is Better)",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig_error, use_container_width=True)

            with col2:
                # Precision and Recall
                fig_pr = go.Figure()
                if 'precision@10' in comparison_df.columns:
                    fig_pr.add_trace(go.Bar(
                        name='Precision@10',
                        x=comparison_df['model'],
                        y=comparison_df['precision@10'],
                        marker_color='green'
                    ))
                if 'recall@10' in comparison_df.columns:
                    fig_pr.add_trace(go.Bar(
                        name='Recall@10',
                        x=comparison_df['model'],
                        y=comparison_df['recall@10'],
                        marker_color='purple'
                    ))
                fig_pr.update_layout(
                    title="Precision & Recall (Higher is Better)",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig_pr, use_container_width=True)

            # Best models
            st.subheader("🏆 Best Models")
            col1, col2 = st.columns(2)
            with col1:
                best_rmse = comparison_df.loc[comparison_df['rmse'].idxmin(), 'model']
                st.metric("Best RMSE", best_rmse)
            with col2:
                if 'precision@10' in comparison_df.columns:
                    best_prec = comparison_df.loc[comparison_df['precision@10'].idxmax(), 'model']
                    st.metric("Best Precision@10", best_prec)
        else:
            st.warning("No model comparison data found. Train models first!")

    # Tab 4: Data Insights
    with tab4:
        st.header("Dataset Insights")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Products", f"{len(products_full):,}")
        with col2:
            st.metric("Total Users", f"{len(user_item_matrix):,}")
        with col3:
            sparsity = (user_item_matrix == 0).sum().sum() / user_item_matrix.size
            st.metric("Matrix Sparsity", f"{sparsity:.1%}")

        # Department distribution
        st.subheader("Product Distribution")
        dept_counts = products_full['department'].value_counts().head(10)
        fig = px.bar(
            x=dept_counts.values,
            y=dept_counts.index,
            orientation='h',
            labels={'x': 'Number of Products', 'y': 'Department'},
            title="Top 10 Departments by Product Count"
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()