"""
Streamlit Web Application for InstaCart Recommendation System
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.models.user_based_cf import UserBasedCF
from src.models.item_based_cf import ItemBasedCF
from src.models.svd_model import SVDRecommender
from config.config import MODELS_DIR, PROCESSED_DATA_DIR

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
    .recommendation-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
def load_models():
    """Load trained models"""
    models = {}

    # Check if models exist
    model_files = {
        'User-Based CF': 'user_cf.pkl',
        'Item-Based CF': 'item_cf.pkl',
        'SVD': 'svd.pkl'
    }

    for name, filename in model_files.items():
        filepath = MODELS_DIR / filename
        if filepath.exists():
            try:
                if name == 'User-Based CF':
                    models[name] = UserBasedCF.load(filepath)
                elif name == 'Item-Based CF':
                    models[name] = ItemBasedCF.load(filepath)
                elif name == 'SVD':
                    models[name] = SVDRecommender.load(filepath)
            except Exception as e:
                st.error(f"Error loading {name}: {e}")

    return models


@st.cache_data
def load_data():
    """Load necessary data"""
    data_loader = DataLoader()

    # Load processed data
    try:
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
        if item_id in product_names:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{i}. {product_names[item_id]}**")
            with col2:
                st.metric("Score", f"{score:.2f}")


def main():
    # Title and description
    st.title("🛒 InstaCart Recommendation System")
    st.markdown("### Personalized Product Recommendations using Collaborative Filtering")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model selection
        models = load_models()
        if not models:
            st.error("No trained models found. Please run train_models.py first.")
            return

        selected_model = st.selectbox(
            "Select Model",
            options=list(models.keys()),
            help="Choose the recommendation algorithm"
        )

        n_recommendations = st.slider(
            "Number of Recommendations",
            min_value=5,
            max_value=20,
            value=10,
            help="How many products to recommend"
        )

        st.markdown("---")
        st.header("📊 Model Info")

        if selected_model in models:
            model = models[selected_model]
            params = model.get_params()
            for key, value in params.items():
                if key != 'name':
                    st.text(f"{key}: {value}")

    # Load data
    user_item_matrix, products_full, product_names = load_data()

    if user_item_matrix is None:
        st.error("Failed to load data. Please check data files.")
        return

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Get Recommendations",
        "🔍 Find Similar Products",
        "📈 Model Performance",
        "📊 Data Insights"
    ])

    with tab1:
        st.header("Get Personalized Recommendations")

        col1, col2 = st.columns([1, 1])

        with col1:
            # User selection
            st.subheader("Select User")

            # Random user button
            if st.button("🎲 Random User"):
                st.session_state['selected_user_idx'] = np.random.choice(len(user_item_matrix))

            # Manual user selection
            user_idx = st.number_input(
                "Or enter User Index",
                min_value=0,
                max_value=len(user_item_matrix) - 1,
                value=st.session_state.get('selected_user_idx', 0)
            )

            st.session_state['selected_user_idx'] = user_idx

            # Show user's purchase history
            st.subheader("Purchase History")
            user_id = user_item_matrix.index[user_idx]
            purchased_items = user_item_matrix.columns[user_item_matrix.iloc[user_idx] > 0].tolist()

            if purchased_items:
                st.info(f"User {user_id} has purchased {len(purchased_items)} different products")

                # Show sample of purchased items
                with st.expander("View Purchase History (First 10)"):
                    for item_id in purchased_items[:10]:
                        if item_id in product_names:
                            st.write(f"• {product_names[item_id]}")
            else:
                st.warning("This user has no purchase history")

        with col2:
            st.subheader(f"Recommendations using {selected_model}")

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

                        # Display recommendations
                        st.success("Recommendations generated!")
                        display_recommendations(rec_product_ids, scores, product_names)

                    except Exception as e:
                        st.error(f"Error generating recommendations: {e}")

    with tab2:
        st.header("Find Similar Products")

        if selected_model == "Item-Based CF":
            # Product search
            search_term = st.text_input("Search for a product", placeholder="e.g., Banana")

            if search_term:
                # Find matching products
                matching_products = [
                    (pid, pname) for pid, pname in product_names.items()
                    if search_term.lower() in pname.lower()
                ]

                if matching_products:
                    # Select product
                    selected_product = st.selectbox(
                        "Select a product",
                        options=matching_products,
                        format_func=lambda x: x[1]
                    )

                    if st.button("Find Similar Products"):
                        try:
                            # Get product index
                            if selected_product[0] in user_item_matrix.columns:
                                item_idx = user_item_matrix.columns.get_loc(selected_product[0])

                                model = models[selected_model]
                                similar_items, similarity_scores = model.find_similar_items(
                                    item_idx,
                                    n_recommendations
                                )

                                # Display similar products
                                st.subheader(f"Products similar to: {selected_product[1]}")

                                similar_product_ids = user_item_matrix.columns[similar_items].tolist()
                                for i, (item_id, sim_score) in enumerate(zip(similar_product_ids, similarity_scores),
                                                                         1):
                                    if item_id in product_names:
                                        col1, col2 = st.columns([4, 1])
                                        with col1:
                                            st.write(f"{i}. {product_names[item_id]}")
                                        with col2:
                                            st.metric("Similarity", f"{sim_score:.3f}")
                            else:
                                st.warning("Product not found in the interaction matrix")
                        except Exception as e:
                            st.error(f"Error finding similar products: {e}")
                else:
                    st.warning("No products found matching your search")
        else:
            st.info("Similar product search is only available for Item-Based CF model")

    with tab3:
        st.header("Model Performance Comparison")

        # Load model comparison results
        comparison_file = MODELS_DIR / "model_comparison.csv"
        if comparison_file.exists():
            comparison_df = pd.read_csv(comparison_file)

            # Display metrics table
            st.subheader("Performance Metrics")
            st.dataframe(comparison_df, width='stretch')

            # Visualize metrics
            col1, col2 = st.columns(2)

            with col1:
                # RMSE and MAE comparison
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
                    title="Error Metrics Comparison",
                    xaxis_title="Model",
                    yaxis_title="Error",
                    barmode='group'
                )
                st.plotly_chart(fig_error, width='stretch')

            with col2:
                # Precision and Recall comparison
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Bar(
                    name='Precision@10',
                    x=comparison_df['model'],
                    y=comparison_df['precision@10'],
                    marker_color='green'
                ))
                fig_pr.add_trace(go.Bar(
                    name='Recall@10',
                    x=comparison_df['model'],
                    y=comparison_df['recall@10'],
                    marker_color='purple'
                ))
                fig_pr.update_layout(
                    title="Precision & Recall @10",
                    xaxis_title="Model",
                    yaxis_title="Score",
                    barmode='group'
                )
                st.plotly_chart(fig_pr, width='stretch')
        else:
            st.warning("No model comparison data found. Please run training script first.")

    with tab4:
        st.header("Data Insights")

        if products_full is not None:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Products", f"{len(products_full):,}")
            with col2:
                st.metric("Total Users", f"{len(user_item_matrix):,}")
            with col3:
                sparsity = (user_item_matrix == 0).sum().sum() / user_item_matrix.size
                st.metric("Matrix Sparsity", f"{sparsity:.1%}")

            # Department distribution
            st.subheader("Top Departments by Product Count")
            dept_counts = products_full['department'].value_counts().head(10)
            fig_dept = px.bar(
                x=dept_counts.values,
                y=dept_counts.index,
                orientation='h',
                labels={'x': 'Number of Products', 'y': 'Department'},
                color=dept_counts.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_dept, width='stretch')

            # User activity distribution
            st.subheader("User Activity Distribution")
            user_activity = (user_item_matrix > 0).sum(axis=1)
            fig_activity = px.histogram(
                user_activity,
                nbins=30,
                labels={'value': 'Number of Products Purchased', 'count': 'Number of Users'},
                title="Distribution of Products Purchased per User"
            )
            st.plotly_chart(fig_activity, width='stretch')


if __name__ == "__main__":
    main()