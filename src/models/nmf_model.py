"""
Non-Negative Matrix Factorization (NMF) Implementation
"""
import numpy as np
from sklearn.decomposition import NMF
from typing import Tuple, Optional
import logging
from tqdm import tqdm
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
        alpha_W: float = 0.1,
        alpha_H: float = 0.1,
        l1_ratio: float = 0.5,
        random_state: int = 42
    ):
        super().__init__(name="NMF")
        self.n_components = n_components
        self.max_iter = max_iter
        self.alpha_W = alpha_W  # Regularization for W matrix
        self.alpha_H = alpha_H  # Regularization for H matrix
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
        logger.info(f"Matrix shape: {train_matrix.shape} (users: {train_matrix.shape[0]}, items: {train_matrix.shape[1]})")

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
            alpha_W=self.alpha_W,  # Regularization for W matrix
            alpha_H=self.alpha_H,  # Regularization for H matrix
            l1_ratio=self.l1_ratio,  # Balance between L1 and L2
            verbose=verbose
        )

        # Fit the model with progress indication
        if verbose:
            logger.info(f"Fitting NMF on {train_matrix.shape[0]} users...")

        self.W = self.model.fit_transform(train_matrix_positive)
        self.H = self.model.components_

        self.is_fitted = True

        if verbose:
            reconstruction_error = self.model.reconstruction_err_
            logger.info(f"NMF training complete. Reconstruction error: {reconstruction_error:.4f}")
            logger.info(f"Factor matrices: W={self.W.shape}, H={self.H.shape}")

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
            # Predict for all users with progress bar
            from tqdm import tqdm
            predictions = np.zeros_like(self.train_matrix)

            # Use batch computation for efficiency
            batch_size = 100
            n_users = self.W.shape[0]

            for i in tqdm(range(0, n_users, batch_size), desc="Generating NMF predictions"):
                end_idx = min(i + batch_size, n_users)
                predictions[i:end_idx] = self.W[i:end_idx].dot(self.H)

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

        # Get predictions for this user
        predictions = self.W[user_idx].dot(self.H)

        if exclude_seen:
            # Mask already interacted items
            seen_items = np.where(self.train_matrix[user_idx] > 0)[0]
            predictions[seen_items] = -1  # Use -1 instead of -np.inf to avoid issues

        # Get top N items
        top_items = np.argsort(predictions)[-n_items:][::-1]

        # Ensure scores are positive and meaningful
        scores = predictions[top_items]
        scores = np.maximum(scores, 0)  # Ensure non-negative

        # If all scores are 0, use the raw prediction values
        if np.all(scores == 0):
            scores = self.W[user_idx].dot(self.H[: , top_items])
            scores = np.abs(scores)  # Take absolute values

        return top_items, scores

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
            'alpha_W': self.alpha_W,
            'alpha_H': self.alpha_H,
            'l1_ratio': self.l1_ratio,
            'random_state': self.random_state
        })
        return params