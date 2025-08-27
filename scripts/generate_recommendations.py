"""
Script to generate recommendations for specific users
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

from config.config import MODELS_DIR, PROCESSED_DATA_DIR, LOGGING_CONFIG
from src.data_loader import DataLoader
from src.models.user_based_cf import UserBasedCF
from src.models.item_based_cf import ItemBasedCF
from src.models.svd_model import SVDRecommender
from src.models.nmf_model import NMFRecommender
from src.utils import timer, create_submission

# Configure logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


@timer
def generate_recommendations(
        model_name: str = 'item_cf',
        user_ids: List[int] = None,
        n_recommendations: int = 10,
        output_format: str = 'display'
):
    """
    Generate recommendations for specified users

    Args:
        model_name: Name of the model to use
        user_ids: List of user IDs (None for random)
        n_recommendations: Number of items to recommend
        output_format: Output format ('display', 'csv', 'json')
    """
    logger.info("=" * 80)
    logger.info("GENERATING RECOMMENDATIONS")
    logger.info("=" * 80)

    # Load data
    data_loader = DataLoader()

    logger.info("\n1. Loading data...")
    user_item_matrix = data_loader.load_processed_data("user_item_matrix")

    # Load product names
    products_df = pd.read_csv(Path('data/raw/products.csv'))
    product_names = dict(zip(products_df['product_id'], products_df['product_name']))

    # Load model
    logger.info(f"\n2. Loading model: {model_name}")

    model_path = MODELS_DIR / f"{model_name}.pkl"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Available models:")
        for f in MODELS_DIR.glob("*.pkl"):
            logger.info(f"  - {f.stem}")
        return

    # Load appropriate model class
    if model_name == 'user_cf':
        model = UserBasedCF.load(model_path)
    elif model_name == 'item_cf':
        model = ItemBasedCF.load(model_path)
    elif model_name == 'svd':
        model = SVDRecommender.load(model_path)
    elif model_name == 'nmf':
        model = NMFRecommender.load(model_path)
    else:
        logger.error(f"Unknown model type: {model_name}")
        return

    # Select users
    if user_ids is None:
        # Random selection
        n_random = min(5, len(user_item_matrix))
        user_indices = np.random.choice(len(user_item_matrix), n_random, replace=False)
        user_ids = user_item_matrix.index[user_indices].tolist()
        logger.info(f"Selected {n_random} random users")
    else:
        # Map user IDs to indices
        user_indices = []
        for uid in user_ids:
            if uid in user_item_matrix.index:
                user_indices.append(user_item_matrix.index.get_loc(uid))
            else:
                logger.warning(f"User {uid} not found in matrix")

    # Generate recommendations
    logger.info(f"\n3. Generating {n_recommendations} recommendations per user")

    all_recommendations = {}

    for user_id, user_idx in zip(user_ids, user_indices):
        # Get user's purchase history
        purchased_items = user_item_matrix.columns[
            user_item_matrix.iloc[user_idx] > 0
            ].tolist()

        # Generate recommendations
        try:
            rec_items, scores = model.recommend_items(user_idx, n_recommendations)
            rec_product_ids = user_item_matrix.columns[rec_items].tolist()

            # Store recommendations
            all_recommendations[user_id] = {
                'recommendations': rec_product_ids,
                'scores': scores.tolist(),
                'purchased_count': len(purchased_items)
            }

            # Display if requested
            if output_format == 'display':
                logger.info(f"\n{'=' * 60}")
                logger.info(f"User {user_id}")
                logger.info(f"{'=' * 60}")
                logger.info(f"Purchase history: {len(purchased_items)} products")

                # Show sample of purchased items
                logger.info("Recent purchases (sample):")
                for item_id in purchased_items[:5]:
                    if item_id in product_names:
                        logger.info(f"  • {product_names[item_id]}")

                logger.info(f"\nTop {n_recommendations} Recommendations:")
                for i, (item_id, score) in enumerate(zip(rec_product_ids, scores), 1):
                    if item_id in product_names:
                        logger.info(f"  {i:2d}. {product_names[item_id]} (score: {score:.3f})")

        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            all_recommendations[user_id] = {
                'recommendations': [],
                'scores': [],
                'purchased_count': len(purchased_items),
                'error': str(e)
            }

    # Save results
    if output_format == 'csv':
        output_file = Path(f"recommendations_{model_name}.csv")

        # Convert to DataFrame
        rows = []
        for user_id, data in all_recommendations.items():
            for rank, (item_id, score) in enumerate(
                    zip(data['recommendations'], data['scores']), 1
            ):
                rows.append({
                    'user_id': user_id,
                    'rank': rank,
                    'product_id': item_id,
                    'product_name': product_names.get(item_id, 'Unknown'),
                    'score': score
                })

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        logger.info(f"\nRecommendations saved to: {output_file}")

    elif output_format == 'json':
        import json
        output_file = Path(f"recommendations_{model_name}.json")

        # Add product names
        for user_id, data in all_recommendations.items():
            data['product_names'] = [
                product_names.get(pid, 'Unknown')
                for pid in data['recommendations']
            ]

        with open(output_file, 'w') as f:
            json.dump(all_recommendations, f, indent=2)

        logger.info(f"\nRecommendations saved to: {output_file}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Model used: {model_name}")
    logger.info(f"Users processed: {len(all_recommendations)}")
    logger.info(f"Recommendations per user: {n_recommendations}")

    # Calculate statistics
    avg_score = np.mean([
        np.mean(data['scores'])
        for data in all_recommendations.values()
        if data['scores']
    ])
    logger.info(f"Average recommendation score: {avg_score:.3f}")

    return all_recommendations


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Generate Product Recommendations')

    parser.add_argument(
        '--model',
        type=str,
        default='item_cf',
        choices=['user_cf', 'item_cf', 'svd', 'nmf'],
        help='Model to use for recommendations'
    )

    parser.add_argument(
        '--user-ids',
        type=str,
        default=None,
        help='Comma-separated user IDs (leave empty for random)'
    )

    parser.add_argument(
        '--n-recommendations',
        type=int,
        default=10,
        help='Number of recommendations per user'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='display',
        choices=['display', 'csv', 'json'],
        help='Output format'
    )

    args = parser.parse_args()

    # Parse user IDs
    user_ids = None
    if args.user_ids:
        user_ids = [int(uid.strip()) for uid in args.user_ids.split(',')]

    try:
        generate_recommendations(
            model_name=args.model,
            user_ids=user_ids,
            n_recommendations=args.n_recommendations,
            output_format=args.output
        )
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise


if __name__ == "__main__":
    main()