"""
Main script to train all collaborative filtering models
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import time
from pathlib import Path
import pandas as pd
import numpy as np

from config.config import (
    MODEL_PARAMS, TRAIN_PARAMS, MODELS_DIR,
    PROCESSED_DATA_DIR, LOGGING_CONFIG
)
from src.data_loader import DataLoader
from src.evaluator import Evaluator
from src.models.user_based_cf import UserBasedCF
from src.models.item_based_cf import ItemBasedCF
from src.models.svd_model import SVDRecommender

# Configure logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def train_all_models(
        sample_size: int = None,
        save_models: bool = True
):
    """
    Train all collaborative filtering models

    Args:
        sample_size: Number of users to sample (None for all)
        save_models: Whether to save trained models
    """
    logger.info("=" * 80)
    logger.info("INSTACART RECOMMENDATION SYSTEM - MODEL TRAINING")
    logger.info("=" * 80)

    # Initialize data loader
    data_loader = DataLoader()
    evaluator = Evaluator()

    # Load data
    logger.info("\n1. Loading data...")
    data_loader.load_raw_data()

    # Create user-item matrix
    logger.info("\n2. Creating user-item matrix...")
    sample_size = sample_size or TRAIN_PARAMS['sample_size']
    user_item_matrix = data_loader.create_user_item_matrix(
        sample_size=sample_size,
        value_type='normalized'
    )

    # Save matrix info
    matrix_df = pd.DataFrame(user_item_matrix)
    logger.info(f"Matrix shape: {matrix_df.shape}")
    logger.info(f"Sparsity: {(matrix_df == 0).sum().sum() / matrix_df.size:.2%}")

    # Create train-test split
    logger.info("\n3. Creating train-test split...")
    train_matrix, test_matrix = data_loader.create_train_test_split(
        matrix_df,
        test_size=TRAIN_PARAMS['test_size'],
        min_items=TRAIN_PARAMS['min_items_user']
    )

    # Store results
    results = []
    models = {}

    # Train User-Based CF
    logger.info("\n4. Training User-Based Collaborative Filtering...")
    start_time = time.time()

    user_cf = UserBasedCF(**MODEL_PARAMS['user_based_cf'])
    user_cf.fit(train_matrix)
    user_cf_predictions = user_cf.predict()

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    # Evaluate
    user_cf_results = evaluator.evaluate_model(
        user_cf_predictions,
        test_matrix,
        k_values=[5, 10],
        model_name="User-Based CF"
    )
    user_cf_results['training_time'] = training_time
    results.append(user_cf_results)
    models['user_cf'] = user_cf

    # Train Item-Based CF
    logger.info("\n5. Training Item-Based Collaborative Filtering...")
    start_time = time.time()

    item_cf = ItemBasedCF(**MODEL_PARAMS['item_based_cf'])
    item_cf.fit(train_matrix)
    item_cf_predictions = item_cf.predict()

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    # Evaluate
    item_cf_results = evaluator.evaluate_model(
        item_cf_predictions,
        test_matrix,
        k_values=[5, 10],
        model_name="Item-Based CF"
    )
    item_cf_results['training_time'] = training_time
    results.append(item_cf_results)
    models['item_cf'] = item_cf

    # Train SVD
    logger.info("\n6. Training SVD Matrix Factorization...")
    start_time = time.time()

    svd_model = SVDRecommender(**MODEL_PARAMS['svd'])
    svd_model.fit(train_matrix)
    svd_predictions = svd_model.predict()

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    # Evaluate
    svd_results = evaluator.evaluate_model(
        svd_predictions,
        test_matrix,
        k_values=[5, 10],
        model_name="SVD"
    )
    svd_results['training_time'] = training_time
    results.append(svd_results)
    models['svd'] = svd_model

    # Compare models
    logger.info("\n7. Model Comparison")
    comparison_df = evaluator.compare_models(results)
    evaluator.print_comparison(comparison_df)

    # Save models if requested
    if save_models:
        logger.info("\n8. Saving models...")
        for name, model in models.items():
            model_path = MODELS_DIR / f"{name}.pkl"
            model.save(model_path)

        # Save comparison results
        comparison_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
        logger.info(f"Models saved to {MODELS_DIR}")

        # Save matrices for later use
        data_loader.save_processed_data("user_item_matrix", matrix_df)
        data_loader.save_processed_data("train_matrix", train_matrix)
        data_loader.save_processed_data("test_matrix", test_matrix)

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)

    return models, comparison_df


def generate_sample_recommendations(models, user_item_matrix, n_users=3, n_items=10):
    """
    Generate sample recommendations for demonstration

    Args:
        models: Dictionary of trained models
        user_item_matrix: User-item matrix
        n_users: Number of sample users
        n_items: Number of recommendations per user
    """
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE RECOMMENDATIONS")
    logger.info("=" * 80)

    # Load product names
    products_df = pd.read_csv(Path('data/raw/products.csv'))
    product_names = dict(zip(products_df['product_id'], products_df['product_name']))

    # Select random users
    sample_users = np.random.choice(len(user_item_matrix), n_users, replace=False)

    for user_idx in sample_users:
        user_id = user_item_matrix.index[user_idx]
        logger.info(f"\nUser {user_id}:")

        # Show user's purchase history
        purchased_items = user_item_matrix.columns[user_item_matrix.iloc[user_idx] > 0].tolist()
        logger.info(f"Previous purchases (sample):")
        for item_id in purchased_items[:5]:
            if item_id in product_names:
                logger.info(f"  - {product_names[item_id]}")

        # Get recommendations from each model
        for model_name, model in models.items():
            logger.info(f"\n{model_name} recommendations:")
            try:
                rec_items, scores = model.recommend_items(user_idx, n_items)

                # Map to product IDs and names
                rec_product_ids = user_item_matrix.columns[rec_items].tolist()
                for i, (item_id, score) in enumerate(zip(rec_product_ids, scores), 1):
                    if item_id in product_names:
                        logger.info(f"  {i}. {product_names[item_id]} (score: {score:.3f})")
            except Exception as e:
                logger.error(f"Error generating recommendations: {e}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Train InstaCart Recommendation Models')
    parser.add_argument(
        '--sample-size',
        type=int,
        default=10000,
        help='Number of users to sample for training'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save trained models'
    )
    parser.add_argument(
        '--recommendations',
        action='store_true',
        help='Generate sample recommendations'
    )

    args = parser.parse_args()

    try:
        # Train models
        models, comparison_df = train_all_models(
            sample_size=args.sample_size,
            save_models=not args.no_save
        )

        # Generate sample recommendations if requested
        if args.recommendations:
            # Load user-item matrix
            data_loader = DataLoader()
            user_item_matrix = data_loader.load_processed_data("user_item_matrix")
            generate_sample_recommendations(models, user_item_matrix)

        logger.info("\n✅ All tasks completed successfully!")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()