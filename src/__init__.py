"""
InstaCart Recommendation System - Core Package
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .data_loader import DataLoader
from .evaluator import Evaluator
from .models.user_based_cf import UserBasedCF
from .models.item_based_cf import ItemBasedCF
from .models.svd_model import SVDRecommender

__all__ = [
    'DataLoader',
    'Evaluator',
    'UserBasedCF',
    'ItemBasedCF',
    'SVDRecommender'
]