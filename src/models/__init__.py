"""
Models Package for Recommendation System
"""

from .base_model import BaseRecommender
from .user_based_cf import UserBasedCF
from .item_based_cf import ItemBasedCF
from .svd_model import SVDRecommender
from .nmf_model import NMFRecommender

__all__ = [
    'BaseRecommender',
    'UserBasedCF',
    'ItemBasedCF',
    'SVDRecommender',
    'NMFRecommender'
]

# Model registry for easy access
MODEL_REGISTRY = {
    'user_cf': UserBasedCF,
    'item_cf': ItemBasedCF,
    'svd': SVDRecommender,
    'nmf': NMFRecommender
}

# Default parameters for each model
DEFAULT_PARAMS = {
    'user_cf': {'k_neighbors': 30, 'similarity_metric': 'cosine'},
    'item_cf': {'k_neighbors': 30, 'similarity_metric': 'cosine'},
    'svd': {'n_factors': 50, 'n_epochs': 10, 'learning_rate': 0.005, 'regularization': 0.02},
    'nmf': {'n_components': 50, 'max_iter': 100, 'alpha_W': 0.1, 'alpha_H': 0.1, 'l1_ratio': 0.5}
}


def get_model(model_name: str, **kwargs):
    """
    Factory function to get model by name

    Args:
        model_name: Name of the model
        **kwargs: Model parameters

    Returns:
        Model instance
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_name]

    # Merge default parameters with provided parameters
    params = DEFAULT_PARAMS.get(model_name, {}).copy()
    params.update(kwargs)

    return model_class(**params)