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
    return model_class(**kwargs)