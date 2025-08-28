"""
Configuration settings for InstaCart Recommendation System
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = DATA_DIR / 'models'

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data files
DATA_FILES = {
    'aisles': 'aisles.csv',
    'departments': 'departments.csv',
    'products': 'products.csv',
    'orders': 'orders.csv',
    'order_products_prior': 'order_products__prior.csv',
    'order_products_train': 'order_products__train.csv'
}

# Model parameters
MODEL_PARAMS = {
    'user_based_cf': {
        'k_neighbors': 30,
        'similarity_metric': 'cosine'
    },
    'item_based_cf': {
        'k_neighbors': 30,
        'similarity_metric': 'cosine'
    },
    'svd': {
        'n_factors': 50,
        'n_epochs': 10,
        'learning_rate': 0.005,
        'regularization': 0.02
    },
    'nmf': {
        'n_components': 50,
        'max_iter': 100,
        'alpha_W': 0.1,  # Regularization for W matrix
        'alpha_H': 0.1,  # Regularization for H matrix
        'l1_ratio': 0.5,
        'random_state': 42
    }
}

# Training parameters
TRAIN_PARAMS = {
    'sample_size': 10000,  # Number of users to sample
    'test_size': 0.2,      # Train-test split ratio
    'min_items_user': 5,   # Minimum items per user
    'random_state': 42
}

# Evaluation parameters
EVAL_PARAMS = {
    'k_values': [5, 10],  # K values for Precision@K and Recall@K
    'metrics': ['rmse', 'mae', 'precision', 'recall']
}

# Visualization settings
VIZ_PARAMS = {
    'figure_size': (15, 8),
    'color_palette': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
    'style': 'seaborn-v0_8-darkgrid'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%Y-%m-%d %H:%M:%S'
}