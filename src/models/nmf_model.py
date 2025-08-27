"""
Non-Negative Matrix Factorization (NMF) Implementation
"""
import numpy as np
from sklearn.decomposition import NMF
from typing import Tuple, Optional
import logging
from .base_model import BaseRecommender

logger = logging.getLogger(__name__)


class NMFRecommender(BaseRecommender):
    """
    Non-Negative Matrix Factorization Recommender
    """

    def __init__(
            self,
            n_components: int = 50,
            max_iter: int = 100,
            alpha: float = 0.1,
            l1_ratio: float = 0.5,
            random_state: int = 42
    ):
        super().__init__(name="NMF")
        self.n_components = n_components
        self.max_iter = max_iter
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state

        self.model = None
        self.W = None  # User factors
        self.H = None  # Item factors

    def fit(self, train_matrix: np.ndarray, verbose: bool = True) -> 'NMFRecommender':
        """
        Train NMF model

        Args:
            train_matrix: User-item interaction matrix
            verbose: Whether to show training progress

        Returns:
            Fitted model instance
        """
        logger.info(f"Training {self.name} with {self.n_components} components")

        self.train_matrix = train_matrix

        # Ensure non-negative values
        train_matrix_positive = np.maximum(train_matrix, 0)

        # Initialize and fit NMF
        self.model = NMF(
            n_components=self.n_components,
            init='nndsvd',  # Better initialization than random
            random_state=self.random_state,
            max_iter=self.max_iter,
            solver='cd',  # Coordinate descent
            alpha=self.alpha,  # Regularization strength
            l1_ratio=self.l1_ratio,  # Balance between L1 and L2
            verbose=verbose
        )

        # Fit the model
        self.W = self.model.fit_transform(train_matrix_positive)
        self.H = self.model.components_

        self.is_fitted = True

        if verbose:
            reconstruction_error = self.model.reconstruction_err_
            logger.info(f"NMF training complete. Reconstruction error: {reconstruction_error:.4f}")

        return self

    def predict(self, user_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate rating predictions

        Args:
            user_idx: Specific user or None for all users

        Returns:
            Predicted ratings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if user_idx is not None:
            # Predict for single user
            predictions = self.W[user_idx].dot(self.H)
        else:
            # Predict for all users
            predictions = self.W.dot(self.H)

        # Clip predictions to valid range (0-5 for ratings)
        return np.clip(predictions, 0, 5)

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
            n_items: Number of recommendations
            exclude_seen: Whether to exclude already seen items

        Returns:
            Tuple of (item_indices, scores)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")

        predictions = self.W[user_idx].dot(self.H)

        if exclude_seen:
            # Mask already interacted items
            seen_items = np.where(self.train_matrix[user_idx] > 0)[0]
            predictions[seen_items] = -np.inf

        # Get top N items
        top_items = np.argsort(predictions)[-n_items:][::-1]

        return top_items, predictions[top_items]

    def get_user_factors(self) -> np.ndarray:
        """Get user factor matrix"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.W

    def get_item_factors(self) -> np.ndarray:
        """Get item factor matrix"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.H

    def explain_recommendation(self, user_idx: int, item_idx: int) -> dict:
        """
        Explain a recommendation by showing factor contributions

        Args:
            user_idx: User index
            item_idx: Item index

        Returns:
            Dictionary with factor contributions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        user_factors = self.W[user_idx]
        item_factors = self.H[:, item_idx]

        # Calculate contribution of each factor
        contributions = user_factors * item_factors

        # Sort by contribution
        factor_importance = np.argsort(np.abs(contributions))[::-1]

        return {
            'predicted_rating': self.W[user_idx].dot(self.H[:, item_idx]),
            'top_factors': factor_importance[:5].tolist(),
            'factor_contributions': contributions[factor_importance[:5]].tolist(),
            'user_factors': user_factors[factor_importance[:5]].tolist(),
            'item_factors': item_factors[factor_importance[:5]].tolist()
        }

    def get_params(self) -> dict:
        """Get model parameters"""
        params = super().get_params()
        params.update({
            'n_components': self.n_components,
            'max_iter': self.max_iter,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'random_state': self.random_state
        })
        return params