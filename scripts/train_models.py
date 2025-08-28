"""
Main script to train all collaborative filtering models including NMF
Fixed version with proper model selection
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
from src.models.nmf_model import NMFRecommender

# Configure logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def train_all_models(
        sample_size: int = None,
        save_models: bool = True,
        model_list: list = None
):
    """
    Train all collaborative filtering models including NMF

    Args:
        sample_size: Number of users to sample (None for all)
        save_models: Whether to save trained models
        model_list: List of models to train (default: all)
    """
    logger.info("=" * 80)
    logger.info("INSTACART RECOMMENDATION SYSTEM - MODEL TRAINING")
    logger.info("Including: User-CF, Item-CF, SVD, and NMF")
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

    # Define all available model configurations
    all_model_configs = {
        'user_cf': {
            'name': 'User-Based CF',
            'class': UserBasedCF,
            'params': MODEL_PARAMS.get('user_based_cf', {'k_neighbors': 30, 'similarity_metric': 'cosine'})
        },
        'item_cf': {
            'name': 'Item-Based CF',
            'class': ItemBasedCF,
            'params': MODEL_PARAMS.get('item_based_cf', {'k_neighbors': 30, 'similarity_metric': 'cosine'})
        },
        'svd': {
            'name': 'SVD',
            'class': SVDRecommender,
            'params': MODEL_PARAMS.get('svd', {'n_factors': 50, 'n_epochs': 10, 'learning_rate': 0.005, 'regularization': 0.02})
        },
        'nmf': {
            'name': 'NMF',
            'class': NMFRecommender,
            'params': MODEL_PARAMS.get('nmf', {
                'n_components': 50,
                'max_iter': 100,
                'alpha_W': 0.1,
                'alpha_H': 0.1,
                'l1_ratio': 0.5,
                'random_state': 42
            })
        }
    }

    # Determine which models to train
    if model_list and 'all' not in model_list:
        # Train only specified models
        models_to_train = {k: v for k, v in all_model_configs.items() if k in model_list}
        logger.info(f"Training selected models: {', '.join(models_to_train.keys())}")
    else:
        # Train all models
        models_to_train = all_model_configs
        logger.info("Training all models")

    # Train each model
    step = 4
    for model_key, config in models_to_train.items():
        logger.info(f"\n{step}. Training {config['name']}...")
        logger.info("-" * 40)

        try:
            start_time = time.time()

            # Initialize model
            model = config['class'](**config['params'])

            # Fit model
            model.fit(train_matrix)

            # Generate predictions
            predictions = model.predict()

            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")

            # Evaluate
            model_results = evaluator.evaluate_model(
                predictions,
                test_matrix,
                k_values=[5, 10],
                model_name=config['name']
            )
            model_results['training_time'] = training_time
            results.append(model_results)
            models[model_key] = model

            # Log key metrics
            logger.info(f"RMSE: {model_results['rmse']:.4f}")
            logger.info(f"MAE: {model_results['mae']:.4f}")
            logger.info(f"Precision@10: {model_results.get('precision@10', 0):.4f}")
            logger.info(f"Recall@10: {model_results.get('recall@10', 0):.4f}")

        except Exception as e:
            logger.error(f"Error training {config['name']}: {e}")
            import traceback
            traceback.print_exc()

        step += 1

    # Compare models if we have results
    if results:
        logger.info(f"\n{step}. Model Comparison")
        logger.info("=" * 80)

        comparison_df = evaluator.compare_models(results)
        evaluator.print_comparison(comparison_df)

        # Find best models
        best_rmse_idx = comparison_df['rmse'].idxmin()
        best_rmse_model = comparison_df.loc[best_rmse_idx, 'model']

        if 'precision@10' in comparison_df.columns:
            best_precision_idx = comparison_df['precision@10'].idxmax()
            best_precision_model = comparison_df.loc[best_precision_idx, 'model']
            logger.info(f"\n🏆 Best Precision@10: {best_precision_model}")

        logger.info(f"🏆 Best RMSE: {best_rmse_model}")
    else:
        comparison_df = None
        logger.warning("No models were successfully trained")

    # Save models if requested
    if save_models and models:
        logger.info(f"\n{step+1}. Saving models...")
        for name, model in models.items():
            model_path = MODELS_DIR / f"{name}.pkl"
            model.save(model_path)
            logger.info(f"  ✓ Saved {name}.pkl")

        # Save comparison results
        if results:
            comparison_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
            logger.info(f"  ✓ Saved model_comparison.csv")

        # Save matrices for later use
        data_loader.save_processed_data("user_item_matrix", matrix_df)
        data_loader.save_processed_data("train_matrix", train_matrix)
        data_loader.save_processed_data("test_matrix", test_matrix)
        logger.info(f"Models and data saved to {MODELS_DIR}")

    logger.info("\n" + "=" * 80)
    if models:
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"Trained {len(models)} model(s): {', '.join(models.keys())}")
    else:
        logger.info("NO MODELS WERE TRAINED")
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
    try:
        products_df = pd.read_csv(Path('data/raw/products.csv'))
        product_names = dict(zip(products_df['product_id'], products_df['product_name']))
    except Exception as e:
        logger.warning(f"Could not load product names: {e}")
        product_names = {}

    # Select random users
    sample_users = np.random.choice(len(user_item_matrix), min(n_users, len(user_item_matrix)), replace=False)

    for user_idx in sample_users:
        user_id = user_item_matrix.index[user_idx]
        logger.info(f"\n{'='*60}")
        logger.info(f"User {user_id} (Index: {user_idx})")
        logger.info(f"{'='*60}")

        # Show user's purchase history
        purchased_items = user_item_matrix.columns[user_item_matrix.iloc[user_idx] > 0].tolist()
        logger.info(f"Previous purchases: {len(purchased_items)} products")

        if product_names:
            logger.info("Sample purchases:")
            for item_id in purchased_items[:5]:
                if item_id in product_names:
                    logger.info(f"  • {product_names[item_id]}")

        if len(purchased_items) > 5:
            logger.info(f"  ... and {len(purchased_items) - 5} more products")

        # Get recommendations from each model
        for model_name, model in models.items():
            logger.info(f"\n{model_name.upper()} Recommendations:")
            try:
                rec_items, scores = model.recommend_items(user_idx, n_items)

                # Map to product IDs and names
                rec_product_ids = user_item_matrix.columns[rec_items].tolist()

                for i, (item_id, score) in enumerate(zip(rec_product_ids, scores), 1):
                    if product_names and item_id in product_names:
                        # Add confidence level based on score
                        if score > 1.5:
                            confidence = "⭐⭐⭐⭐⭐"
                        elif score > 1.0:
                            confidence = "⭐⭐⭐⭐"
                        elif score > 0.5:
                            confidence = "⭐⭐⭐"
                        else:
                            confidence = "⭐⭐"

                        logger.info(f"  {i:2d}. {product_names[item_id]} (score: {score:.3f}) {confidence}")
                    else:
                        logger.info(f"  {i:2d}. Product {item_id} (score: {score:.3f})")

            except Exception as e:
                logger.error(f"  Error generating recommendations: {e}")


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
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['user_cf', 'item_cf', 'svd', 'nmf', 'all'],
        default=['all'],
        help='Which models to train (default: all)'
    )

    args = parser.parse_args()

    try:
        # Train models
        models, comparison_df = train_all_models(
            sample_size=args.sample_size,
            save_models=not args.no_save,
            model_list=args.models
        )

        # Generate sample recommendations if requested
        if args.recommendations and models:
            # Load user-item matrix
            data_loader = DataLoader()
            user_item_matrix = data_loader.load_processed_data("user_item_matrix")

            if user_item_matrix is not None:
                generate_sample_recommendations(models, user_item_matrix)
            else:
                logger.warning("Could not load user-item matrix for recommendations")

        logger.info("\n✅ All tasks completed successfully!")

    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()