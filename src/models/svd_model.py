"""
SVD (Singular Value Decomposition) Matrix Factorization Implementation
"""
import numpy as np
from typing import Tuple, Optional
import logging
from .base_model import BaseRecommender

logger = logging.getLogger(__name__)


class SVDRecommender(BaseRecommender):
    """
    SVD Matrix Factorization Recommender using Gradient Descent
    """

    def __init__(
            self,
            n_factors: int = 50,
            n_epochs: int = 10,
            learning_rate: float = 0.005,
            regularization: float = 0.02,
            random_state: int = 42
    ):
        super().__init__(name="SVD")
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.random_state = random_state

        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = 0

        np.random.seed(random_state)

    def fit(self, train_matrix: np.ndarray, verbose: bool = True) -> 'SVDRecommender':
        """
        Train SVD model using Stochastic Gradient Descent

        Args:
            train_matrix: User-item interaction matrix
            verbose: Whether to show training progress

        Returns:
            Fitted model instance
        """
        logger.info(f"Training {self.name} with {self.n_factors} factors")

        self.train_matrix = train_matrix
        n_users, n_items = train_matrix.shape

        # Calculate global mean
        non_zero_mask = train_matrix > 0
        self.global_mean = train_matrix[non_zero_mask].mean()

        # Initialize factors and biases
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)

        # Get non-zero indices for training
        train_indices = np.array(np.where(train_matrix > 0)).T

        # Training loop
        train_errors = []
        for epoch in range(self.n_epochs):
            np.random.shuffle(train_indices)
            epoch_errors = []

            for user_idx, item_idx in train_indices:
                rating = train_matrix[user_idx, item_idx]

                # Predict rating
                prediction = self._predict_single(user_idx, item_idx)
                error = rating - prediction
                epoch_errors.append(error ** 2)

                # Cache old values for update
                user_factor_old = self.user_factors[user_idx].copy()
                user_bias_old = self.user_biases[user_idx]

                # Update biases
                self.user_biases[user_idx] += self.learning_rate * (
                        error - self.regularization * self.user_biases[user_idx]
                )
                self.item_biases[item_idx] += self.learning_rate * (
                        error - self.regularization * self.item_biases[item_idx]
                )

                # Update factors
                self.user_factors[user_idx] += self.learning_rate * (
                        error * self.item_factors[item_idx] -
                        self.regularization * self.user_factors[user_idx]
                )
                self.item_factors[item_idx] += self.learning_rate * (
                        error * user_factor_old -
                        self.regularization * self.item_factors[item_idx]
                )

            # Calculate and store RMSE
            rmse = np.sqrt(np.mean(epoch_errors))
            train_errors.append(rmse)

            if verbose and (epoch + 1) % max(1, self.n_epochs // 5) == 0:
                logger.info(f"  Epoch {epoch + 1}/{self.n_epochs}, RMSE: {rmse:.4f}")

        self.is_fitted = True
        self.training_errors = train_errors

        return self

    def _predict_single(self, user_idx: int, item_idx: int) -> float:
        """
        Predict a single rating

        Args:
            user_idx: User index
            item_idx: Item index

        Returns:
            Predicted rating
        """
        prediction = (
                self.global_mean +
                self.user_biases[user_idx] +
                self.item_biases[item_idx] +
                self.user_factors[user_idx].dot(self.item_factors[item_idx])
        )
        return prediction

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
            predictions = (
                    self.global_mean +
                    self.user_biases[user_idx] +
                    self.item_biases +
                    self.user_factors[user_idx].dot(self.item_factors.T)
            )
        else:
            # Predict for all users
            predictions = (
                    self.global_mean +
                    self.user_biases[:, np.newaxis] +
                    self.item_biases[np.newaxis, :] +
                    self.user_factors.dot(self.item_factors.T)
            )

        # Clip predictions to valid range
        return np.clip(predictions, 1, 5)

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

        predictions = self.predict(user_idx)

        if exclude_seen:
            # Mask already interacted items
            seen_items = np.where(self.train_matrix[user_idx] > 0)[0]
            predictions[seen_items] = -np.inf

        # Get top N items
        top_items = np.argsort(predictions)[-n_items:][::-1]

        return top_items, predictions[top_items]

    def get_user_embeddings(self) -> np.ndarray:
        """Get user factor matrix"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.user_factors

    def get_item_embeddings(self) -> np.ndarray:
        """Get item factor matrix"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.item_factors

    def get_params(self) -> dict:
        """Get model parameters"""
        params = super().get_params()
        params.update({
            'n_factors': self.n_factors,
            'n_epochs': self.n_epochs,
            'learning_rate': self.learning_rate,
            'regularization': self.regularization,
            'random_state': self.random_state
        })
        return params