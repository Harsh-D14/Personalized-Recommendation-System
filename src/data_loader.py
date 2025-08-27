"""
Standalone Data Loading and Preprocessing Utilities
This version works independently without requiring config module
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging
import os

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles data loading and preprocessing for the recommendation system
    """

    def __init__(self, data_dir: Optional[str] = None, processed_dir: Optional[str] = None):
        """
        Initialize DataLoader with configurable paths

        Args:
            data_dir: Path to raw data directory
            processed_dir: Path to processed data directory
        """
        # Set up paths - use current directory structure if not specified
        base_dir = Path.cwd()

        # Check if we're in a subdirectory and adjust accordingly
        if 'src' in str(base_dir):
            base_dir = base_dir.parent
        elif 'scripts' in str(base_dir):
            base_dir = base_dir.parent

        self.data_dir = Path(data_dir) if data_dir else base_dir / 'data' / 'raw'
        self.processed_dir = Path(processed_dir) if processed_dir else base_dir / 'data' / 'processed'

        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Data file names
        self.data_files = {
            'aisles': 'aisles.csv',
            'departments': 'departments.csv',
            'products': 'products.csv',
            'orders': 'orders.csv',
            'order_products_prior': 'order_products__prior.csv',
            'order_products_train': 'order_products__train.csv'
        }

        self.datasets = {}
        self.processed_data = {}

        logger.info(f"DataLoader initialized with data_dir={self.data_dir}")

    def load_raw_data(self, verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load all raw CSV files

        Args:
            verbose: Whether to print loading information

        Returns:
            Dictionary of dataframes
        """
        if verbose:
            logger.info("Loading raw data files...")

        for name, filename in self.data_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                try:
                    self.datasets[name] = pd.read_csv(filepath)
                    if verbose:
                        logger.info(f"✓ Loaded {name}: {self.datasets[name].shape}")
                except Exception as e:
                    logger.warning(f"✗ Error loading {name}: {e}")
            else:
                logger.warning(f"✗ File not found: {filepath}")

        if not self.datasets:
            logger.error("No data files were loaded. Please check your data directory.")
            logger.info(f"Looking for files in: {self.data_dir}")
            logger.info("Expected files:")
            for name, filename in self.data_files.items():
                logger.info(f"  - {filename}")

        return self.datasets

    def create_user_item_matrix(
        self,
        sample_size: Optional[int] = None,
        value_type: str = 'frequency'
    ) -> pd.DataFrame:
        """
        Create user-item interaction matrix

        Args:
            sample_size: Number of users to sample (None for all)
            value_type: Type of values ('frequency', 'binary', 'normalized')

        Returns:
            User-item interaction matrix
        """
        if not self.datasets:
            logger.warning("No datasets loaded. Loading raw data first...")
            self.load_raw_data()

        if 'orders' not in self.datasets or 'order_products_prior' not in self.datasets:
            logger.error("Required datasets (orders, order_products_prior) not loaded")
            return pd.DataFrame()

        logger.info(f"Creating user-item matrix (type: {value_type})...")

        # Merge order and product data
        orders = self.datasets['orders']

        # Combine prior and train order products if available
        order_products_list = []
        if 'order_products_prior' in self.datasets:
            order_products_list.append(self.datasets['order_products_prior'])
        if 'order_products_train' in self.datasets:
            order_products_list.append(self.datasets['order_products_train'])

        if not order_products_list:
            logger.error("No order_products data available")
            return pd.DataFrame()

        order_products = pd.concat(order_products_list, ignore_index=True)

        # Join orders with products
        user_products = orders.merge(order_products, on='order_id')

        # Sample users if specified
        if sample_size:
            unique_users = user_products['user_id'].unique()
            sampled_users = unique_users[:min(sample_size, len(unique_users))]
            user_products = user_products[user_products['user_id'].isin(sampled_users)]
            logger.info(f"Sampled {len(sampled_users)} users")

        # Create interaction matrix based on value_type
        if value_type == 'frequency':
            # Count of purchases
            matrix = user_products.groupby(['user_id', 'product_id']).size().unstack(fill_value=0)
        elif value_type == 'binary':
            # Binary (purchased or not)
            matrix = user_products.groupby(['user_id', 'product_id']).size().unstack(fill_value=0)
            matrix = (matrix > 0).astype(int)
        elif value_type == 'normalized':
            # Normalized frequency (0-5 scale)
            matrix = user_products.groupby(['user_id', 'product_id']).size().unstack(fill_value=0)
            max_val = matrix.max().max()
            if max_val > 0:
                matrix = (matrix / max_val * 5).clip(0, 5)
        else:
            raise ValueError(f"Unknown value_type: {value_type}")

        sparsity = (matrix == 0).sum().sum() / matrix.size
        logger.info(f"Matrix created: {matrix.shape}, Sparsity: {sparsity:.2%}")

        return matrix

    def create_train_test_split(
        self,
        matrix: pd.DataFrame,
        test_size: float = 0.2,
        min_items: int = 5,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split matrix into train and test sets

        Args:
            matrix: User-item matrix
            test_size: Proportion of interactions for test
            min_items: Minimum items per user to include in test
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_matrix, test_matrix)
        """
        np.random.seed(random_state)
        logger.info(f"Creating train-test split (test_size={test_size})...")

        train_matrix = matrix.values.copy()
        test_matrix = np.zeros_like(train_matrix)

        for user_idx in range(train_matrix.shape[0]):
            user_items = np.where(train_matrix[user_idx] > 0)[0]

            if len(user_items) >= min_items:
                n_test = max(1, int(len(user_items) * test_size))
                test_items = np.random.choice(user_items, size=n_test, replace=False)

                test_matrix[user_idx, test_items] = train_matrix[user_idx, test_items]
                train_matrix[user_idx, test_items] = 0

        train_interactions = (train_matrix > 0).sum()
        test_interactions = (test_matrix > 0).sum()

        logger.info(f"Train: {train_interactions} interactions")
        logger.info(f"Test: {test_interactions} interactions")

        return train_matrix, test_matrix

    def get_product_features(self) -> pd.DataFrame:
        """
        Create product features dataframe

        Returns:
            Product features dataframe
        """
        if not self.datasets:
            self.load_raw_data()

        logger.info("Creating product features...")

        if 'products' not in self.datasets:
            logger.error("Products dataset not loaded")
            return pd.DataFrame()

        # Start with basic product info
        products = self.datasets['products'].copy()

        # Add department names if available
        if 'departments' in self.datasets:
            products = products.merge(
                self.datasets['departments'],
                on='department_id',
                how='left'
            )

        # Add aisle names if available
        if 'aisles' in self.datasets:
            products = products.merge(
                self.datasets['aisles'],
                on='aisle_id',
                how='left'
            )

        # Add purchase statistics if order data is available
        if 'order_products_prior' in self.datasets or 'order_products_train' in self.datasets:
            order_products_list = []
            if 'order_products_prior' in self.datasets:
                order_products_list.append(self.datasets['order_products_prior'])
            if 'order_products_train' in self.datasets:
                order_products_list.append(self.datasets['order_products_train'])

            order_products = pd.concat(order_products_list, ignore_index=True)

            # Calculate statistics
            product_stats = order_products.groupby('product_id').agg({
                'order_id': 'count',
                'reordered': ['mean', 'sum'],
                'add_to_cart_order': 'mean'
            }).round(3)

            product_stats.columns = ['_'.join(col).strip() if col[1] else col[0]
                                    for col in product_stats.columns]
            product_stats.rename(columns={'order_id_count': 'total_orders'}, inplace=True)

            # Merge with products
            products = products.merge(
                product_stats,
                left_on='product_id',
                right_index=True,
                how='left'
            )

            # Fill missing values
            products.fillna(0, inplace=True)

        logger.info(f"Product features created: {products.shape}")

        return products

    def get_user_features(self) -> pd.DataFrame:
        """
        Create user features dataframe

        Returns:
            User features dataframe
        """
        if not self.datasets:
            self.load_raw_data()

        logger.info("Creating user features...")

        if 'orders' not in self.datasets:
            logger.error("Orders dataset not loaded")
            return pd.DataFrame()

        orders = self.datasets['orders']

        # Aggregate user features from orders
        user_features = orders.groupby('user_id').agg({
            'order_id': 'count',
            'order_dow': lambda x: x.mode()[0] if not x.empty else 0,
            'order_hour_of_day': 'mean',
            'days_since_prior_order': 'mean'
        }).rename(columns={
            'order_id': 'total_orders',
            'order_dow': 'favorite_day',
            'order_hour_of_day': 'avg_hour',
            'days_since_prior_order': 'avg_days_between_orders'
        })

        # Round numerical features
        user_features['avg_hour'] = user_features['avg_hour'].round(2)
        user_features['avg_days_between_orders'] = user_features['avg_days_between_orders'].round(2)

        logger.info(f"User features created: {user_features.shape}")

        return user_features

    def save_processed_data(self, name: str, data: any) -> None:
        """
        Save processed data to disk

        Args:
            name: Name for the saved file
            data: Data to save
        """
        filepath = self.processed_dir / f"{name}.pkl"

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"✓ Saved {name} to {filepath}")
        except Exception as e:
            logger.error(f"✗ Error saving {name}: {e}")

    def load_processed_data(self, name: str) -> any:
        """
        Load processed data from disk

        Args:
            name: Name of the saved file

        Returns:
            Loaded data
        """
        filepath = self.processed_dir / f"{name}.pkl"

        if not filepath.exists():
            logger.error(f"Processed data not found: {filepath}")
            return None

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"✓ Loaded {name} from {filepath}")
            return data
        except Exception as e:
            logger.error(f"✗ Error loading {name}: {e}")
            return None

    def check_data_availability(self) -> Dict[str, bool]:
        """
        Check which data files are available

        Returns:
            Dictionary of file availability
        """
        availability = {}

        logger.info("Checking data availability...")
        logger.info(f"Data directory: {self.data_dir}")

        for name, filename in self.data_files.items():
            filepath = self.data_dir / filename
            availability[name] = filepath.exists()
            status = "✓" if availability[name] else "✗"
            logger.info(f"  {status} {filename}: {availability[name]}")

        return availability


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize DataLoader
    loader = DataLoader()

    # Check data availability
    availability = loader.check_data_availability()

    if any(availability.values()):
        # Load data
        datasets = loader.load_raw_data()

        if datasets:
            # Create user-item matrix
            matrix = loader.create_user_item_matrix(sample_size=1000)

            if not matrix.empty:
                print(f"\nUser-Item Matrix Shape: {matrix.shape}")
                print(f"Sample of matrix:\n{matrix.iloc[:5, :5]}")

                # Create train-test split
                train, test = loader.create_train_test_split(matrix)
                print(f"\nTrain shape: {train.shape}")
                print(f"Test shape: {test.shape}")

                # Get features
                user_features = loader.get_user_features()
                print(f"\nUser features shape: {user_features.shape}")

                product_features = loader.get_product_features()
                print(f"Product features shape: {product_features.shape}")
    else:
        print("\nNo data files found. Please download the InstaCart dataset.")
        print("Expected location:", loader.data_dir)