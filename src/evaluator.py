"""
Model Evaluation Utilities
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Handles model evaluation and metrics calculation
    """

    @staticmethod
    def calculate_rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """
        Calculate Root Mean Square Error

        Args:
            predictions: Predicted values
            actuals: Actual values

        Returns:
            RMSE value
        """
        mask = actuals > 0
        if mask.sum() == 0:
            return np.inf

        return np.sqrt(mean_squared_error(actuals[mask], predictions[mask]))

    @staticmethod
    def calculate_mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error

        Args:
            predictions: Predicted values
            actuals: Actual values

        Returns:
            MAE value
        """
        mask = actuals > 0
        if mask.sum() == 0:
            return np.inf

        return mean_absolute_error(actuals[mask], predictions[mask])

    @staticmethod
    def precision_at_k(predictions: np.ndarray, actuals: np.ndarray, k: int = 10) -> float:
        """
        Calculate Precision@K

        Args:
            predictions: Predicted ratings matrix
            actuals: Actual ratings matrix
            k: Number of top items to consider

        Returns:
            Average Precision@K
        """
        precisions = []

        for user_idx in range(predictions.shape[0]):
            # Get top k predicted items
            top_k_items = np.argsort(predictions[user_idx])[-k:][::-1]

            # Get actual positive items
            actual_items = np.where(actuals[user_idx] > 0)[0]

            if len(actual_items) == 0:
                continue

            # Calculate precision
            hits = len(set(top_k_items) & set(actual_items))
            precisions.append(hits / k)

        return np.mean(precisions) if precisions else 0.0

    @staticmethod
    def recall_at_k(predictions: np.ndarray, actuals: np.ndarray, k: int = 10) -> float:
        """
        Calculate Recall@K

        Args:
            predictions: Predicted ratings matrix
            actuals: Actual ratings matrix
            k: Number of top items to consider

        Returns:
            Average Recall@K
        """
        recalls = []

        for user_idx in range(predictions.shape[0]):
            # Get top k predicted items
            top_k_items = np.argsort(predictions[user_idx])[-k:][::-1]

            # Get actual positive items
            actual_items = np.where(actuals[user_idx] > 0)[0]

            if len(actual_items) == 0:
                continue

            # Calculate recall
            hits = len(set(top_k_items) & set(actual_items))
            recalls.append(hits / len(actual_items))

        return np.mean(recalls) if recalls else 0.0

    @staticmethod
    def f1_at_k(predictions: np.ndarray, actuals: np.ndarray, k: int = 10) -> float:
        """
        Calculate F1 Score@K

        Args:
            predictions: Predicted ratings matrix
            actuals: Actual ratings matrix
            k: Number of top items to consider

        Returns:
            F1 Score@K
        """
        precision = Evaluator.precision_at_k(predictions, actuals, k)
        recall = Evaluator.recall_at_k(predictions, actuals, k)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def ndcg_at_k(predictions: np.ndarray, actuals: np.ndarray, k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K

        Args:
            predictions: Predicted ratings matrix
            actuals: Actual ratings matrix
            k: Number of top items to consider

        Returns:
            Average NDCG@K
        """

        def dcg_at_k(relevances: np.ndarray, k: int) -> float:
            """Calculate DCG@K"""
            relevances = relevances[:k]
            if relevances.size:
                return relevances[0] + np.sum(
                    relevances[1:] / np.log2(np.arange(2, relevances.size + 1))
                )
            return 0.0

        ndcgs = []

        for user_idx in range(predictions.shape[0]):
            # Get top k predicted items
            top_k_items = np.argsort(predictions[user_idx])[-k:][::-1]

            # Get relevance scores (actual ratings)
            relevances = actuals[user_idx, top_k_items]

            # Calculate DCG
            dcg = dcg_at_k(relevances, k)

            # Calculate IDCG (ideal DCG)
            ideal_relevances = np.sort(actuals[user_idx])[::-1]
            idcg = dcg_at_k(ideal_relevances, k)

            # Calculate NDCG
            if idcg > 0:
                ndcgs.append(dcg / idcg)

        return np.mean(ndcgs) if ndcgs else 0.0

    @staticmethod
    def coverage(
            recommendations: List[np.ndarray],
            n_items: int
    ) -> float:
        """
        Calculate catalog coverage

        Args:
            recommendations: List of recommended items for each user
            n_items: Total number of items in catalog

        Returns:
            Coverage percentage
        """
        recommended_items = set()
        for recs in recommendations:
            recommended_items.update(recs)

        return len(recommended_items) / n_items

    @staticmethod
    def diversity(recommendations: np.ndarray, item_similarity: np.ndarray) -> float:
        """
        Calculate recommendation diversity using item similarity

        Args:
            recommendations: Recommended items
            item_similarity: Item similarity matrix

        Returns:
            Average diversity score
        """
        if len(recommendations) < 2:
            return 0.0

        diversity_scores = []
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                similarity = item_similarity[recommendations[i], recommendations[j]]
                diversity_scores.append(1 - similarity)

        return np.mean(diversity_scores) if diversity_scores else 0.0

    def evaluate_model(
            self,
            predictions: np.ndarray,
            test_matrix: np.ndarray,
            k_values: List[int] = [5, 10],
            model_name: str = "Model"
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation

        Args:
            predictions: Predicted ratings
            test_matrix: Test set actual ratings
            k_values: K values for top-K metrics
            model_name: Name of the model

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")

        results = {
            'model': model_name,
            'rmse': self.calculate_rmse(predictions, test_matrix),
            'mae': self.calculate_mae(predictions, test_matrix)
        }

        for k in k_values:
            results[f'precision@{k}'] = self.precision_at_k(predictions, test_matrix, k)
            results[f'recall@{k}'] = self.recall_at_k(predictions, test_matrix, k)
            results[f'f1@{k}'] = self.f1_at_k(predictions, test_matrix, k)
            results[f'ndcg@{k}'] = self.ndcg_at_k(predictions, test_matrix, k)

        # Log results
        logger.info(f"{model_name} Results:")
        for metric, value in results.items():
            if metric != 'model':
                logger.info(f"  {metric}: {value:.4f}")

        return results

    def compare_models(
            self,
            results_list: List[Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Compare multiple model results

        Args:
            results_list: List of result dictionaries

        Returns:
            Comparison dataframe
        """
        df = pd.DataFrame(results_list)

        # Sort by RMSE (ascending)
        df = df.sort_values('rmse')

        # Format numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(4)

        return df

    def print_comparison(self, comparison_df: pd.DataFrame) -> None:
        """
        Print formatted model comparison

        Args:
            comparison_df: Comparison dataframe
        """
        print("\n" + "=" * 80)
        print("MODEL COMPARISON RESULTS")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        print("=" * 80)

        # Find best models
        for metric in comparison_df.columns:
            if metric == 'model':
                continue

            if metric in ['rmse', 'mae']:
                # Lower is better
                best_idx = comparison_df[metric].idxmin()
            else:
                # Higher is better
                best_idx = comparison_df[metric].idxmax()

            best_model = comparison_df.loc[best_idx, 'model']
            best_value = comparison_df.loc[best_idx, metric]

            print(f"Best {metric}: {best_model} ({best_value:.4f})")