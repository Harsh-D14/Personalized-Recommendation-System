"""
Exploratory Data Analysis Script for InstaCart Dataset
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGGING_CONFIG
from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.utils import timer, ensure_dir, save_pickle

# Configure logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


@timer
def run_eda(save_plots: bool = True, save_data: bool = True):
    """
    Run comprehensive EDA on InstaCart dataset

    Args:
        save_plots: Whether to save plots
        save_data: Whether to save processed data
    """
    logger.info("=" * 80)
    logger.info("INSTACART DATASET - EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 80)

    # Initialize loaders
    data_loader = DataLoader()
    preprocessor = DataPreprocessor()

    # Load data
    logger.info("\n1. Loading datasets...")
    datasets = data_loader.load_raw_data()

    # Dataset overview
    logger.info("\n2. Dataset Overview")
    logger.info("-" * 40)

    for name, df in datasets.items():
        logger.info(f"\n{name.upper()}:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"  Memory: {df.memory_usage().sum() / 1024 ** 2:.2f} MB")
        logger.info(f"  Missing values: {df.isnull().sum().sum()}")

    # Orders analysis
    logger.info("\n3. Orders Analysis")
    logger.info("-" * 40)

    orders = datasets['orders']
    logger.info(f"Total orders: {len(orders):,}")
    logger.info(f"Unique users: {orders['user_id'].nunique():,}")
    logger.info(f"Orders per user: {orders.groupby('user_id').size().describe()}")

    if save_plots:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Orders by day of week
        orders['order_dow'].value_counts().sort_index().plot(
            kind='bar', ax=axes[0, 0], color='skyblue'
        )
        axes[0, 0].set_title('Orders by Day of Week')
        axes[0, 0].set_xlabel('Day of Week')
        axes[0, 0].set_ylabel('Number of Orders')

        # Orders by hour
        orders['order_hour_of_day'].value_counts().sort_index().plot(
            kind='bar', ax=axes[0, 1], color='lightgreen'
        )
        axes[0, 1].set_title('Orders by Hour of Day')
        axes[0, 1].set_xlabel('Hour')
        axes[0, 1].set_ylabel('Number of Orders')

        # Days since prior order
        orders['days_since_prior_order'].dropna().hist(
            ax=axes[1, 0], bins=30, color='lightcoral'
        )
        axes[1, 0].set_title('Days Since Prior Order Distribution')
        axes[1, 0].set_xlabel('Days')
        axes[1, 0].set_ylabel('Frequency')

        # Order number distribution
        orders['order_number'].hist(
            ax=axes[1, 1], bins=30, color='lightsalmon'
        )
        axes[1, 1].set_title('Order Number Distribution')
        axes[1, 1].set_xlabel('Order Number')
        axes[1, 1].set_ylabel('Frequency')

        plt.suptitle('Order Patterns Analysis', fontsize=16)
        plt.tight_layout()

        plot_dir = Path('plots')
        ensure_dir(plot_dir)
        plt.savefig(plot_dir / 'order_patterns.png', dpi=100, bbox_inches='tight')
        plt.show()

    # Product analysis
    logger.info("\n4. Product Analysis")
    logger.info("-" * 40)

    products = datasets['products']
    departments = datasets['departments']
    aisles = datasets['aisles']

    # Merge product information
    products_full = products.merge(departments, on='department_id')
    products_full = products_full.merge(aisles, on='aisle_id')

    logger.info(f"Total products: {len(products):,}")
    logger.info(f"Total departments: {len(departments)}")
    logger.info(f"Total aisles: {len(aisles)}")

    # Top departments and aisles
    top_departments = products_full['department'].value_counts().head(10)
    top_aisles = products_full['aisle'].value_counts().head(10)

    if save_plots:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        top_departments.plot(kind='barh', ax=axes[0], color='steelblue')
        axes[0].set_title('Top 10 Departments by Product Count')
        axes[0].set_xlabel('Number of Products')

        top_aisles.plot(kind='barh', ax=axes[1], color='darkorange')
        axes[1].set_title('Top 10 Aisles by Product Count')
        axes[1].set_xlabel('Number of Products')

        plt.tight_layout()
        plt.savefig(plot_dir / 'product_distribution.png', dpi=100, bbox_inches='tight')
        plt.show()

    # Purchase behavior analysis
    logger.info("\n5. Purchase Behavior Analysis")
    logger.info("-" * 40)

    # Combine order products
    order_products = pd.concat([
        datasets.get('order_products_prior', pd.DataFrame()),
        datasets.get('order_products_train', pd.DataFrame())
    ])

    logger.info(f"Total order-product pairs: {len(order_products):,}")
    logger.info(f"Reorder rate: {order_products['reordered'].mean():.2%}")

    # Most popular products
    order_products_detailed = order_products.merge(products_full, on='product_id')
    top_products = order_products_detailed['product_name'].value_counts().head(20)

    logger.info("\nTop 10 Most Ordered Products:")
    for i, (product, count) in enumerate(top_products.head(10).items(), 1):
        logger.info(f"  {i}. {product}: {count:,} orders")

    if save_plots:
        plt.figure(figsize=(12, 8))
        top_products.plot(kind='barh', color='mediumseagreen')
        plt.title('Top 20 Most Ordered Products', fontsize=14)
        plt.xlabel('Number of Orders')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(plot_dir / 'top_products.png', dpi=100, bbox_inches='tight')
        plt.show()

    # Cart analysis
    logger.info("\n6. Cart Analysis")
    logger.info("-" * 40)

    cart_sizes = order_products.groupby('order_id').size()
    logger.info(f"Average cart size: {cart_sizes.mean():.2f} products")
    logger.info(f"Median cart size: {cart_sizes.median():.0f} products")
    logger.info(f"Max cart size: {cart_sizes.max()} products")

    if save_plots:
        plt.figure(figsize=(12, 6))
        plt.hist(cart_sizes, bins=50, edgecolor='black', alpha=0.7, color='dodgerblue')
        plt.axvline(cart_sizes.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {cart_sizes.mean():.1f}')
        plt.axvline(cart_sizes.median(), color='green', linestyle='--', linewidth=2,
                    label=f'Median: {cart_sizes.median():.1f}')
        plt.xlabel('Cart Size (Number of Products)')
        plt.ylabel('Number of Orders')
        plt.title('Distribution of Cart Sizes', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / 'cart_sizes.png', dpi=100, bbox_inches='tight')
        plt.show()

    # Create and save features
    if save_data:
        logger.info("\n7. Creating Features")
        logger.info("-" * 40)

        # User features
        user_features = data_loader.get_user_features()
        logger.info(f"User features created: {user_features.shape}")

        # Product features
        product_features = data_loader.get_product_features()
        logger.info(f"Product features created: {product_features.shape}")

        # Time features
        orders_with_time = preprocessor.create_time_features(orders)
        logger.info(f"Time features created: {orders_with_time.shape[1] - orders.shape[1]} new features")

        # Save processed data
        logger.info("\n8. Saving Processed Data")
        logger.info("-" * 40)

        ensure_dir(PROCESSED_DATA_DIR)

        data_loader.save_processed_data("user_features", user_features)
        data_loader.save_processed_data("product_features", product_features)
        data_loader.save_processed_data("products_full", products_full)
        data_loader.save_processed_data("orders_with_time", orders_with_time)

        logger.info(f"✅ Processed data saved to {PROCESSED_DATA_DIR}")

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("EDA SUMMARY")
    logger.info("=" * 80)

    summary = f"""
    Dataset Statistics:
    - Total Users: {orders['user_id'].nunique():,}
    - Total Orders: {len(orders):,}
    - Total Products: {len(products):,}
    - Total Departments: {len(departments)}
    - Total Aisles: {len(aisles)}

    Shopping Patterns:
    - Average Orders per User: {orders.groupby('user_id').size().mean():.1f}
    - Average Cart Size: {cart_sizes.mean():.1f} products
    - Overall Reorder Rate: {order_products['reordered'].mean():.1%}
    - Peak Shopping Hour: {orders['order_hour_of_day'].mode()[0]}
    - Peak Shopping Day: {orders['order_dow'].mode()[0]}

    Top Categories:
    - Most Popular Department: {top_departments.index[0]}
    - Most Popular Aisle: {top_aisles.index[0]}
    - Most Ordered Product: {top_products.index[0]}
    """

    logger.info(summary)

    logger.info("\n✅ EDA completed successfully!")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Run EDA on InstaCart Dataset')
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Do not save plots'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save processed data'
    )

    args = parser.parse_args()

    try:
        run_eda(
            save_plots=not args.no_plots,
            save_data=not args.no_save
        )
    except Exception as e:
        logger.error(f"Error during EDA: {e}")
        raise


if __name__ == "__main__":
    main()