"""
Item-Based Collaborative Filtering Implementation
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Optional
import logging
from tqdm import tqdm
from .base_model import BaseRecommender

logger = logging.getLogger(__name__)


class ItemBasedCF(BaseRecommender):
    """
    Item-Based Collaborative Filtering Recommender
    """

    def __init__(self, k_neighbors: int = 30, similarity_metric: str = 'cosine'):
        super().__init__(name="ItemBasedCF")
        self.k_neighbors = k_neighbors
        self.similarity_metric = similarity_metric
        self.item_similarity = None

    def fit(self, train_matrix: np.ndarray, verbose: bool = True) -> 'ItemBasedCF':
        """
        Calculate item similarity matrix

        Args:
            train_matrix: User-item interaction matrix
            verbose: Whether to show progress

        Returns:
            Fitted model instance
        """
        logger.info(f"Training {self.name} with k={self.k_neighbors}")

        self.train_matrix = train_matrix

        # Calculate item similarity (transpose for item-item comparison)
        if self.similarity_metric == 'cosine':
            self.item_similarity = cosine_similarity(train_matrix.T)
        elif self.similarity_metric == 'jaccard':
            # Jaccard similarity for binary data
            binary_matrix = (train_matrix > 0).astype(int)
            self.item_similarity = self._jaccard_similarity(binary_matrix.T)
        elif self.similarity_metric == 'pearson':
            # Pearson correlation
            mean_centered = train_matrix.T - np.mean(train_matrix.T, axis=1, keepdims=True)
            self.item_similarity = cosine_similarity(mean_centered)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

        # Set diagonal to 0 (no self-similarity)
        np.fill_diagonal(self.item_similarity, 0)

        self.is_fitted = True
        logger.info(f"Item similarity matrix computed: {self.item_similarity.shape}")

        return self

    def _jaccard_similarity(self, matrix: np.ndarray) -> np.ndarray:
        """
        Calculate Jaccard similarity between items

        Args:
            matrix: Binary item matrix

        Returns:
            Jaccard similarity matrix
        """
        intersection = matrix.dot(matrix.T)
        row_sums = matrix.sum(axis=1)
        unions = row_sums[:, None] + row_sums[None, :] - intersection

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            similarity = intersection / unions
            similarity[np.isnan(similarity)] = 0

        return similarity

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
        user_ratings = self.train_matrix[user_idx]
        predictions = np.zeros(self.train_matrix.shape[1])

        # For each item, predict based on similar items
        for item_idx in range(self.train_matrix.shape[1]):
            if user_ratings[item_idx] > 0:
                # Keep existing rating
                predictions[item_idx] = user_ratings[item_idx]
            else:
                # Find similar items that user has rated
                rated_items = np.where(user_ratings > 0)[0]
                if len(rated_items) == 0:
                    continue

                # Get similarities to rated items
                similarities = self.item_similarity[item_idx, rated_items]

                # Get k most similar items
                k = min(self.k_neighbors, len(rated_items))
                if k == 0:
                    continue

                k_indices = np.argsort(similarities)[-k:][::-1]
                k_similar_items = rated_items[k_indices]
                k_similarities = similarities[k_indices]

                if k_similarities.sum() > 0:
                    # Weighted average
                    predictions[item_idx] = (
                                                    user_ratings[k_similar_items] * k_similarities
                                            ).sum() / k_similarities.sum()

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

    def find_similar_items(self, item_idx: int, n_items: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find most similar items

        Args:
            item_idx: Item index
            n_items: Number of similar items to find

        Returns:
            Tuple of (item_indices, similarity_scores)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        similarities = self.item_similarity[item_idx]
        similar_items = np.argsort(similarities)[-n_items:][::-1]

        return similar_items, similarities[similar_items]

    def get_frequently_bought_together(
            self,
            item_idx: int,
            n_items: int = 5,
            min_similarity: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get items frequently bought together

        Args:
            item_idx: Item index
            n_items: Number of items to return
            min_similarity: Minimum similarity threshold

        Returns:
            Tuple of (item_indices, similarity_scores)
        """
        similar_items, scores = self.find_similar_items(item_idx, n_items * 2)

        # Filter by minimum similarity
        mask = scores >= min_similarity
        similar_items = similar_items[mask][:n_items]
        scores = scores[mask][:n_items]

        return similar_items, scores

    def get_params(self) -> dict:
        """Get model parameters"""
        params = super().get_params()
        params.update({
            'k_neighbors': self.k_neighbors,
            'similarity_metric': self.similarity_metric
        })
        return params