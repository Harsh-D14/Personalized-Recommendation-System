"""
Unit tests for recommendation models
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.models import (
    UserBasedCF, ItemBasedCF, SVDRecommender,
    NMFRecommender, get_model
)
from src.evaluator import Evaluator


# Fixtures
@pytest.fixture
def sample_matrix():
    """Create a small sample user-item matrix for testing"""
    np.random.seed(42)
    matrix = np.random.rand(50, 100) * 5  # 50 users, 100 items
    matrix[matrix < 2] = 0  # Make it sparse
    return matrix


@pytest.fixture
def train_test_split(sample_matrix):
    """Create train-test split"""
    train = sample_matrix.copy()
    test = np.zeros_like(sample_matrix)

    # Hide 20% of interactions for testing
    for i in range(train.shape[0]):
        items = np.where(train[i] > 0)[0]
        if len(items) > 5:
            test_items = np.random.choice(items, size=len(items) // 5, replace=False)
            test[i, test_items] = train[i, test_items]
            train[i, test_items] = 0

    return train, test


# Test User-Based CF
class TestUserBasedCF:

    def test_initialization(self):
        """Test model initialization"""
        model = UserBasedCF(k_neighbors=10)
        assert model.k_neighbors == 10
        assert model.name == "UserBasedCF"
        assert not model.is_fitted

    def test_fit(self, sample_matrix):
        """Test model fitting"""
        model = UserBasedCF(k_neighbors=10)
        model.fit(sample_matrix)

        assert model.is_fitted
        assert model.user_similarity is not None
        assert model.user_similarity.shape == (50, 50)

    def test_predict(self, sample_matrix):
        """Test prediction generation"""
        model = UserBasedCF(k_neighbors=10)
        model.fit(sample_matrix)

        # Predict for single user
        predictions_single = model.predict(user_idx=0)
        assert predictions_single.shape == (100,)

        # Predict for all users
        predictions_all = model.predict()
        assert predictions_all.shape == sample_matrix.shape

    def test_recommend(self, sample_matrix):
        """Test recommendation generation"""
        model = UserBasedCF(k_neighbors=10)
        model.fit(sample_matrix)

        items, scores = model.recommend_items(user_idx=0, n_items=5)

        assert len(items) == 5
        assert len(scores) == 5
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))  # Descending order

    def test_save_load(self, sample_matrix, tmp_path):
        """Test model saving and loading"""
        model = UserBasedCF(k_neighbors=10)
        model.fit(sample_matrix)

        # Save model
        filepath = tmp_path / "user_cf.pkl"
        model.save(filepath)
        assert filepath.exists()

        # Load model
        loaded_model = UserBasedCF.load(filepath)
        assert loaded_model.is_fitted
        assert loaded_model.k_neighbors == 10


# Test Item-Based CF
class TestItemBasedCF:

    def test_initialization(self):
        """Test model initialization"""
        model = ItemBasedCF(k_neighbors=20)
        assert model.k_neighbors == 20
        assert model.name == "ItemBasedCF"

    def test_fit(self, sample_matrix):
        """Test model fitting"""
        model = ItemBasedCF(k_neighbors=20)
        model.fit(sample_matrix)

        assert model.is_fitted
        assert model.item_similarity is not None
        assert model.item_similarity.shape == (100, 100)

    def test_similar_items(self, sample_matrix):
        """Test finding similar items"""
        model = ItemBasedCF(k_neighbors=20)
        model.fit(sample_matrix)

        similar_items, scores = model.find_similar_items(item_idx=0, n_items=5)

        assert len(similar_items) == 5
        assert len(scores) == 5
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


# Test SVD
class TestSVDRecommender:

    def test_initialization(self):
        """Test model initialization"""
        model = SVDRecommender(n_factors=10, n_epochs=5)
        assert model.n_factors == 10
        assert model.n_epochs == 5

    def test_fit(self, sample_matrix):
        """Test model fitting"""
        model = SVDRecommender(n_factors=10, n_epochs=5)
        model.fit(sample_matrix, verbose=False)

        assert model.is_fitted
        assert model.user_factors.shape == (50, 10)
        assert model.item_factors.shape == (100, 10)

    def test_predict(self, sample_matrix):
        """Test prediction generation"""
        model = SVDRecommender(n_factors=10, n_epochs=5)
        model.fit(sample_matrix, verbose=False)

        predictions = model.predict()
        assert predictions.shape == sample_matrix.shape
        assert np.all(predictions >= 1) and np.all(predictions <= 5)  # Clipped to valid range


# Test NMF
class TestNMFRecommender:

    def test_initialization(self):
        """Test model initialization"""
        model = NMFRecommender(n_components=15)
        assert model.n_components == 15
        assert model.name == "NMF"

    def test_fit(self, sample_matrix):
        """Test model fitting"""
        model = NMFRecommender(n_components=15, max_iter=50)
        model.fit(sample_matrix, verbose=False)

        assert model.is_fitted
        assert model.W.shape == (50, 15)
        assert model.H.shape == (15, 100)

    def test_non_negative(self, sample_matrix):
        """Test that factors are non-negative"""
        model = NMFRecommender(n_components=15, max_iter=50)
        model.fit(sample_matrix, verbose=False)

        assert np.all(model.W >= 0)
        assert np.all(model.H >= 0)


# Test Model Factory
class TestModelFactory:

    def test_get_model(self):
        """Test model factory function"""
        model = get_model('user_cf', k_neighbors=15)
        assert isinstance(model, UserBasedCF)
        assert model.k_neighbors == 15

        model = get_model('svd', n_factors=20)
        assert isinstance(model, SVDRecommender)
        assert model.n_factors == 20

    def test_invalid_model(self):
        """Test invalid model name"""
        with pytest.raises(ValueError):
            get_model('invalid_model')


# Test Evaluator
class TestEvaluator:

    def test_rmse(self, train_test_split):
        """Test RMSE calculation"""
        train, test = train_test_split
        evaluator = Evaluator()

        # Use train as predictions for simple test
        rmse = evaluator.calculate_rmse(train, test)
        assert rmse >= 0

    def test_precision_at_k(self, train_test_split):
        """Test Precision@K calculation"""
        train, test = train_test_split
        evaluator = Evaluator()

        precision = evaluator.precision_at_k(train, test, k=5)
        assert 0 <= precision <= 1

    def test_recall_at_k(self, train_test_split):
        """Test Recall@K calculation"""
        train, test = train_test_split
        evaluator = Evaluator()

        recall = evaluator.recall_at_k(train, test, k=5)
        assert 0 <= recall <= 1

    def test_evaluate_model(self, train_test_split):
        """Test comprehensive model evaluation"""
        train, test = train_test_split
        evaluator = Evaluator()

        results = evaluator.evaluate_model(
            predictions=train,
            test_matrix=test,
            k_values=[5, 10],
            model_name="TestModel"
        )

        assert 'rmse' in results
        assert 'mae' in results
        assert 'precision@5' in results
        assert 'recall@10' in results


# Integration Tests
class TestIntegration:

    def test_full_pipeline(self, sample_matrix):
        """Test full training and evaluation pipeline"""
        # Create train-test split
        train = sample_matrix.copy()
        test = np.zeros_like(sample_matrix)

        for i in range(train.shape[0]):
            items = np.where(train[i] > 0)[0]
            if len(items) > 5:
                test_items = np.random.choice(items, size=2, replace=False)
                test[i, test_items] = train[i, test_items]
                train[i, test_items] = 0

        # Train model
        model = UserBasedCF(k_neighbors=10)
        model.fit(train)

        # Generate predictions
        predictions = model.predict()

        # Evaluate
        evaluator = Evaluator()
        results = evaluator.evaluate_model(predictions, test, model_name="UserCF")

        assert results['model'] == "UserCF"
        assert all(key in results for key in ['rmse', 'mae', 'precision@5'])

    def test_model_comparison(self, sample_matrix):
        """Test comparing multiple models"""
        models = [
            UserBasedCF(k_neighbors=10),
            ItemBasedCF(k_neighbors=10)
        ]

        evaluator = Evaluator()
        results = []

        for model in models:
            model.fit(sample_matrix)
            predictions = model.predict()

            # Use sample_matrix as both train and test for simplicity
            result = evaluator.evaluate_model(
                predictions,
                sample_matrix,
                model_name=model.name
            )
            results.append(result)

        # Compare models
        comparison = evaluator.compare_models(results)

        assert len(comparison) == 2
        assert 'rmse' in comparison.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])