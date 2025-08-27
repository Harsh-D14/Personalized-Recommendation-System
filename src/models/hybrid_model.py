"""
Hybrid Recommendation Model combining multiple approaches
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from .base_model import BaseRecommender

logger = logging.getLogger(__name__)


class HybridRecommender(BaseRecommender):
    """
    Hybrid recommender combining multiple recommendation models
    """

    def __init__(
            self,
            models: Dict[str, BaseRecommender],
            weights: Optional[Dict[str, float]] = None,
            combination_method: str = 'weighted'
    ):
        """
        Initialize hybrid recommender

        Args:
            models: Dictionary of model_name -> model instance
            weights: Dictionary of model_name -> weight (must sum to 1)
            combination_method: How to combine predictions ('weighted', 'switching', 'mixed')
        """
        super().__init__(name="HybridRecommender")
        self.models = models
        self.combination_method = combination_method

        # Set default equal weights if not provided
        if weights is None:
            n_models = len(models)
            self.weights = {name: 1.0 / n_models for name in models.keys()}
        else:
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            self.weights = {name: w / total_weight for name, w in weights.items()}

        # Validate weights match models
        if set(self.weights.keys()) != set(self.models.keys()):
            raise ValueError("Weight keys must match model keys")

    def fit(self, train_matrix: np.ndarray, **kwargs) -> 'HybridRecommender':
        """
        Fit all component models

        Args:
            train_matrix: User-item interaction matrix
            **kwargs: Additional parameters for models

        Returns:
            Fitted hybrid model
        """
        logger.info(f"Training {len(self.models)} models for hybrid recommender")

        self.train_matrix = train_matrix

        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(train_matrix)

        self.is_fitted = True
        logger.info("Hybrid model training complete")

        return self

    def predict(self, user_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate predictions using hybrid approach

        Args:
            user_idx: Specific user or None for all users

        Returns:
            Predicted ratings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if self.combination_method == 'weighted':
            return self._weighted_prediction(user_idx)
        elif self.combination_method == 'switching':
            return self._switching_prediction(user_idx)
        elif self.combination_method == 'mixed':
            return self._mixed_prediction(user_idx)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")

    def _weighted_prediction(self, user_idx: Optional[int]) -> np.ndarray:
        """
        Weighted average of all model predictions

        Args:
            user_idx: User index or None

        Returns:
            Weighted predictions
        """
        predictions = None

        for name, model in self.models.items():
            model_pred = model.predict(user_idx)
            weight = self.weights[name]

            if predictions is None:
                predictions = weight * model_pred
            else:
                predictions += weight * model_pred

        return predictions

    def _switching_prediction(self, user_idx: Optional[int]) -> np.ndarray:
        """
        Switch between models based on context

        Args:
            user_idx: User index or None

        Returns:
            Context-aware predictions
        """
        # Example switching logic based on user activity
        if user_idx is not None:
            # Single user prediction
            user_activity = (self.train_matrix[user_idx] > 0).sum()

            # Use different models based on user activity level
            if user_activity < 10:
                # New users - use popularity-based or SVD
                if 'svd' in self.models:
                    return self.models['svd'].predict(user_idx)
                else:
                    return self.models[list(self.models.keys())[0]].predict(user_idx)
            elif user_activity < 50:
                # Medium activity - use item-based CF
                if 'item_cf' in self.models:
                    return self.models['item_cf'].predict(user_idx)
                else:
                    return self._weighted_prediction(user_idx)
            else:
                # Heavy users - use user-based CF
                if 'user_cf' in self.models:
                    return self.models['user_cf'].predict(user_idx)
                else:
                    return self._weighted_prediction(user_idx)
        else:
            # For all users, use weighted average
            return self._weighted_prediction(user_idx)

    def _mixed_prediction(self, user_idx: Optional[int]) -> np.ndarray:
        """
        Mix predictions by taking best from each model

        Args:
            user_idx: User index or None

        Returns:
            Mixed predictions
        """
        all_predictions = {}

        for name, model in self.models.items():
            all_predictions[name] = model.predict(user_idx)

        # Take maximum prediction for each item (optimistic approach)
        if user_idx is not None:
            # Single user
            predictions = np.maximum.reduce(list(all_predictions.values()))
        else:
            # All users
            predictions = np.maximum.reduce(list(all_predictions.values()))

        return predictions

    def recommend_items(
            self,
            user_idx: int,
            n_items: int = 10,
            exclude_seen: bool = True,
            method: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate top-N recommendations

        Args:
            user_idx: User index
            n_items: Number of recommendations
            exclude_seen: Whether to exclude already seen items
            method: Override combination method for this recommendation

        Returns:
            Tuple of (item_indices, scores)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")

        # Use specified method or default
        if method:
            original_method = self.combination_method
            self.combination_method = method
            predictions = self.predict(user_idx)
            self.combination_method = original_method
        else:
            predictions = self.predict(user_idx)

        if exclude_seen:
            # Mask already interacted items
            seen_items = np.where(self.train_matrix[user_idx] > 0)[0]
            predictions[seen_items] = -np.inf

        # Get top N items
        top_items = np.argsort(predictions)[-n_items:][::-1]

        return top_items, predictions[top_items]

    def get_model_contributions(
            self,
            user_idx: int,
            item_idx: int
    ) -> Dict[str, float]:
        """
        Get each model's contribution to a specific prediction

        Args:
            user_idx: User index
            item_idx: Item index

        Returns:
            Dictionary of model contributions
        """
        contributions = {}

        for name, model in self.models.items():
            pred = model.predict(user_idx)
            if isinstance(pred, np.ndarray) and pred.ndim == 1:
                contributions[name] = {
                    'prediction': pred[item_idx],
                    'weighted_contribution': pred[item_idx] * self.weights[name]
                }

        return contributions

    def cross_validate_weights(
            self,
            train_matrix: np.ndarray,
            val_matrix: np.ndarray,
            weight_options: List[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Find optimal weights through cross-validation

        Args:
            train_matrix: Training data
            val_matrix: Validation data
            weight_options: List of weight combinations to try

        Returns:
            Optimal weights
        """
        if weight_options is None:
            # Generate default weight options
            weight_options = []
            step = 0.1
            for w1 in np.arange(0, 1.1, step):
                for w2 in np.arange(0, 1.1 - w1, step):
                    w3 = 1.0 - w1 - w2
                    if len(self.models) == 3 and abs(w3) < 1e-6:
                        continue
                    weights = {}
                    model_names = list(self.models.keys())
                    if len(model_names) >= 1:
                        weights[model_names[0]] = w1
                    if len(model_names) >= 2:
                        weights[model_names[1]] = w2
                    if len(model_names) >= 3:
                        weights[model_names[2]] = w3
                    weight_options.append(weights)

        best_weights = None
        best_score = float('inf')

        for weights in weight_options:
            self.weights = weights
            predictions = self.predict()

            # Calculate RMSE on validation set
            mask = val_matrix > 0
            if mask.sum() > 0:
                rmse = np.sqrt(np.mean((predictions[mask] - val_matrix[mask]) ** 2))

                if rmse < best_score:
                    best_score = rmse
                    best_weights = weights.copy()

        logger.info(f"Best weights found: {best_weights} with RMSE: {best_score:.4f}")
        self.weights = best_weights

        return best_weights

    def get_params(self) -> dict:
        """Get model parameters"""
        params = super().get_params()
        params.update({
            'models': list(self.models.keys()),
            'weights': self.weights,
            'combination_method': self.combination_method
        })
        return params

    def __str__(self) -> str:
        model_str = ", ".join([f"{name}({w:.2f})" for name, w in self.weights.items()])
        return f"HybridRecommender({model_str}, method={self.combination_method})"