# 🛒 InstaCart Market Basket Analysis & Recommendation System

A production-ready recommendation system for online grocery shopping using collaborative filtering techniques on the InstaCart dataset.

## 📋 Project Overview

This project implements a complete recommendation system pipeline including:
- **Data Processing**: EDA and feature engineering on 3M+ orders
- **Multiple Models**: User-based CF, Item-based CF, SVD, and NMF
- **Evaluation Framework**: Comprehensive metrics (RMSE, MAE, Precision@K, Recall@K)
- **Web Interface**: Interactive Streamlit application
- **Production Structure**: Modular Python codebase with proper separation of concerns

## 🏗️ Project Structure

```
instacart-recommendation/
│
├── config/
│   ├── __init__.py
│   └── config.py                 # Configuration settings
│
├── data/
│   ├── raw/                      # Original InstaCart CSV files
│   ├── processed/                # Preprocessed data
│   └── models/                   # Saved trained models
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Data loading utilities
│   ├── preprocessor.py          # Data preprocessing
│   ├── evaluator.py             # Model evaluation metrics
│   ├── utils.py                 # Helper functions
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py        # Base recommender class
│   │   ├── user_based_cf.py     # User-based CF
│   │   ├── item_based_cf.py     # Item-based CF
│   │   ├── svd_model.py         # SVD implementation
│   │   └── nmf_model.py         # NMF implementation
│   │
│   └── visualization/
│       ├── __init__.py
│       └── plots.py              # Visualization functions
│
├── scripts/
│   ├── run_eda.py               # EDA script
│   ├── train_models.py          # Model training script
│   └── evaluate_models.py       # Model evaluation
│
├── app/
│   ├── __init__.py
│   └── streamlit_app.py         # Web application
│
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/instacart-recommendation.git
cd instacart-recommendation
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download the InstaCart dataset from [Kaggle](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset) and place the CSV files in `data/raw/`:

```
data/raw/
├── aisles.csv
├── departments.csv
├── products.csv
├── orders.csv
├── order_products__prior.csv
└── order_products__train.csv
```

### 4. Run EDA (Optional)

```bash
python scripts/run_eda.py
```

### 5. Train Models

```bash
python scripts/train_models.py --sample-size 10000 --recommendations
```

Options:
- `--sample-size`: Number of users to sample (default: 10000)
- `--no-save`: Don't save models
- `--recommendations`: Generate sample recommendations

### 6. Launch Web Application

```bash
streamlit run app/streamlit_app.py
```

Open your browser to `http://localhost:8501`

## 📊 Models Implemented

### 1. User-Based Collaborative Filtering
- Finds similar users based on purchase patterns
- Recommends products that similar users bought
- Best for discovering diverse products

### 2. Item-Based Collaborative Filtering
- Identifies products frequently bought together
- "Customers who bought X also bought Y"
- Best for complementary product recommendations

### 3. SVD (Singular Value Decomposition)
- Matrix factorization with latent factors
- Handles sparse data effectively
- Includes user and item biases

### 4. NMF (Non-Negative Matrix Factorization)
- Interpretable latent features
- Works well with implicit feedback
- Non-negative factor decomposition

## 📈 Performance Metrics

The system evaluates models using:
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Precision@K**: Accuracy of top-K recommendations
- **Recall@K**: Coverage of relevant items in top-K
- **F1@K**: Harmonic mean of precision and recall
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Coverage**: Catalog coverage percentage
- **Diversity**: Recommendation diversity score

## 🖥️ Web Application Features

### Main Features:
1. **Get Recommendations**: Personalized product suggestions for users
2. **Find Similar Products**: Discover related items
3. **Model Comparison**: View performance metrics
4. **Data Insights**: Explore dataset statistics

### Interactive Elements:
- User selection (manual or random)
- Model selection (User-CF, Item-CF, SVD, NMF)
- Adjustable number of recommendations
- Real-time recommendation generation
- Purchase history viewing

## 🔧 Configuration

Edit `config/config.py` to modify:

```python
# Model parameters
MODEL_PARAMS = {
    'user_based_cf': {
        'k_neighbors': 30,
        'similarity_metric': 'cosine'
    },
    'svd': {
        'n_factors': 50,
        'n_epochs': 10,
        'learning_rate': 0.005
    }
    # ... more configurations
}

# Training parameters
TRAIN_PARAMS = {
    'sample_size': 10000,
    'test_size': 0.2,
    'min_items_user': 5
}
```

## 📝 Usage Examples

### Python API

```python
from src.data_loader import DataLoader
from src.models.item_based_cf import ItemBasedCF

# Load data
loader = DataLoader()
loader.load_raw_data()
matrix = loader.create_user_item_matrix(sample_size=5000)

# Train model
model = ItemBasedCF(k_neighbors=30)
model.fit(matrix.values)

# Get recommendations
user_idx = 0
items, scores = model.recommend_items(user_idx, n_items=10)

# Find similar products
item_idx = 100
similar_items, similarities = model.find_similar_items(item_idx)
```

### Command Line

```bash
# Train with custom parameters
python scripts/train_models.py --sample-size 20000

# Generate recommendations for specific users
python scripts/generate_recommendations.py --user-ids 1,2,3 --model item_cf
```

## 🧪 Testing

Run tests with:

```bash
pytest tests/
pytest tests/ --cov=src --cov-report=html
```

## 📊 Expected Results

Based on the InstaCart dataset:
- **Sparsity**: ~99% (typical for e-commerce)
- **RMSE**: 1.2-2.0
- **Precision@10**: 0.15-0.30
- **Recall@10**: 0.10-0.25
- **Training Time**: 10-60 seconds (10K users)

## 🔄 Next Steps

### Week 3 - Hybrid System
- [ ] Content-based filtering using TF-IDF
- [ ] Hybrid recommendation combining CF and content
- [ ] Context-aware recommendations

### Week 4 - Production Deployment
- [ ] FastAPI REST endpoints
- [ ] Docker containerization
- [ ] Model versioning and A/B testing
- [ ] Real-time recommendation serving

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 References

- [InstaCart Dataset](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset)
- [Collaborative Filtering Techniques](https://developers.google.com/machine-learning/recommendation/collaborative/basics)
- [Matrix Factorization Techniques](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- InstaCart for providing the dataset
- Kaggle for hosting the competition
- Streamlit for the amazing web framework