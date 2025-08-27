"""
User-Based Collaborative Filtering Implementation
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional
import logging
from tqdm import tqdm
from .base_model import BaseRecommender

logger = logging.getLogger(__name__)


class UserBasedCF(BaseRecommender):
    """
    User-Based Collaborative Filtering Recommender
    """

    def __init__(self, k_neighbors: int = 30, similarity_metric: str = 'cosine'):
        super().__init__(name="UserBasedCF")
        self.k_neighbors = k_neighbors
        self.similarity_metric = similarity_metric
        self.user_similarity = None

    def fit(self, train_matrix: np.ndarray, verbose: bool = True) -> 'UserBasedCF':
        """
        Calculate user similarity matrix

        Args:
            train_matrix: User-item interaction matrix
            verbose: Whether to show progress

        Returns:
            Fitted model instance
        """
        logger.info(f"Training {self.name} with k={self.k_neighbors}")

        self.train_matrix = train_matrix

        # Calculate user similarity
        if self.similarity_metric == 'cosine':
            self.user_similarity = cosine_similarity(train_matrix)
        elif self.similarity_metric == 'pearson':
            # Center the data for Pearson correlation
            mean_centered = train_matrix - np.mean(train_matrix, axis=1, keepdims=True)
            self.user_similarity = cosine_similarity(mean_centered)
        elif self.similarity_metric == 'euclidean':
            # Euclidean distance converted to similarity
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(train_matrix)
            self.user_similarity = 1 / (1 + distances)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

        # Set diagonal to 0 (no self-similarity)
        np.fill_diagonal(self.user_similarity, 0)

        self.is_fitted = True
        logger.info(f"User similarity matrix computed: {self.user_similarity.shape}")

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
            return self._predict_for_user(user_idx)

        # Predict for all users
        predictions = np.zeros_like(self.train_matrix)

        for idx in tqdm(range(self.train_matrix.shape[0]), desc="Generating predictions"):
            predictions[idx] = self._predict_for_user(idx)

        return predictions

    def _predict_for_user(self, user_idx: int) -> np.ndarray:
        """
        Predict ratings for a single user

        Args:
            user_idx: User index

        Returns:
            Predicted ratings vector
        """
        # Get k most similar users
        similarities = self.user_similarity[user_idx]
        k_similar_users = np.argsort(similarities)[-self.k_neighbors:][::-1]
        k_similarities = similarities[k_similar_users]

        # Handle edge case
        if k_similarities.sum() == 0:
            return np.zeros(self.train_matrix.shape[1])

        # Weighted average of similar users' ratings
        weighted_sum = self.train_matrix[k_similar_users].T.dot(k_similarities)
        predictions = weighted_sum / k_similarities.sum()

        return predictions

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

        predictions = self._predict_for_user(user_idx)

        if exclude_seen:
            # Mask already interacted items
            seen_items = np.where(self.train_matrix[user_idx] > 0)[0]
            predictions[seen_items] = -np.inf

        # Get top N items
        top_items = np.argsort(predictions)[-n_items:][::-1]

        return top_items, predictions[top_items]

    def find_similar_users(self, user_idx: int, n_users: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find most similar users

        Args:
            user_idx: User index
            n_users: Number of similar users to find

        Returns:
            Tuple of (user_indices, similarity_scores)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        similarities = self.user_similarity[user_idx]
        similar_users = np.argsort(similarities)[-n_users:][::-1]

        return similar_users, similarities[similar_users]

    def get_params(self) -> dict:
        """Get model parameters"""
        params = super().get_params()
        params.update({
            'k_neighbors': self.k_neighbors,
            'similarity_metric': self.similarity_metric
        })
        return params