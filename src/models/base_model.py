"""
Base Recommender Model Class
"""
from abc import ABC, abstractmethod
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    """
    Abstract base class for all recommendation models
    """

    def __init__(self, name: str = "BaseRecommender"):
        self.name = name
        self.is_fitted = False
        self.train_matrix = None

    @abstractmethod
    def fit(self, train_matrix: np.ndarray, **kwargs) -> 'BaseRecommender':
        """
        Train the recommendation model

        Args:
            train_matrix: User-item interaction matrix
            **kwargs: Additional model-specific parameters

        Returns:
            Self instance for method chaining
        """
        pass

    @abstractmethod
    def predict(self, user_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate predictions for users

        Args:
            user_idx: Specific user index or None for all users

        Returns:
            Predicted ratings matrix or vector
        """
        pass

    @abstractmethod
    def recommend_items(
            self,
            user_idx: int,
            n_items: int = 10,
            exclude_seen: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate top-N recommendations for a user

        Args:
            user_idx: User index
            n_items: Number of items to recommend
            exclude_seen: Whether to exclude already interacted items

        Returns:
            Tuple of (item_indices, scores)
        """
        pass

    def save(self, filepath: Path) -> None:
        """
        Save model to disk

        Args:
            filepath: Path to save the model
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Model {self.name} saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    @classmethod
    def load(cls, filepath: Path) -> 'BaseRecommender':
        """
        Load model from disk

        Args:
            filepath: Path to load the model from

        Returns:
            Loaded model instance
        """
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters

        Returns:
            Dictionary of model parameters
        """
        return {
            'name': self.name,
            'is_fitted': self.is_fitted
        }

    def __str__(self) -> str:
        return f"{self.name}(fitted={self.is_fitted})"

    def __repr__(self) -> str:
        return self.__str__()