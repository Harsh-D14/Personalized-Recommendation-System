"""
Data Preprocessing Utilities for Recommendation System
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles data preprocessing and feature engineering
    """

    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}

    def remove_outliers(
            self,
            df: pd.DataFrame,
            column: str,
            method: str = 'iqr',
            threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove outliers from a dataframe column

        Args:
            df: Input dataframe
            column: Column to process
            method: Method to use ('iqr' or 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            Dataframe with outliers removed
        """
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)

        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            mask = z_scores < threshold
        else:
            raise ValueError(f"Unknown method: {method}")

        n_removed = (~mask).sum()
        logger.info(f"Removed {n_removed} outliers from {column}")

        return df[mask]

    def encode_categorical(
            self,
            df: pd.DataFrame,
            columns: List[str],
            method: str = 'label'
    ) -> pd.DataFrame:
        """
        Encode categorical columns

        Args:
            df: Input dataframe
            columns: Columns to encode
            method: Encoding method ('label' or 'onehot')

        Returns:
            Encoded dataframe
        """
        df_encoded = df.copy()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in dataframe")
                continue

            if method == 'label':
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df[col])
                else:
                    df_encoded[col] = self.label_encoders[col].transform(df[col])

            elif method == 'onehot':
                dummies = pd.get_dummies(df[col], prefix=col)
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
            else:
                raise ValueError(f"Unknown encoding method: {method}")

        return df_encoded

    def scale_features(
            self,
            df: pd.DataFrame,
            columns: List[str],
            method: str = 'standard'
    ) -> pd.DataFrame:
        """
        Scale numerical features

        Args:
            df: Input dataframe
            columns: Columns to scale
            method: Scaling method ('standard' or 'minmax')

        Returns:
            Scaled dataframe
        """
        df_scaled = df.copy()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in dataframe")
                continue

            if col not in self.scalers:
                if method == 'standard':
                    self.scalers[col] = StandardScaler()
                elif method == 'minmax':
                    self.scalers[col] = MinMaxScaler()
                else:
                    raise ValueError(f"Unknown scaling method: {method}")

                df_scaled[col] = self.scalers[col].fit_transform(df[[col]])
            else:
                df_scaled[col] = self.scalers[col].transform(df[[col]])

        return df_scaled

    def create_time_features(
            self,
            orders_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create time-based features from orders data

        Args:
            orders_df: Orders dataframe

        Returns:
            Dataframe with time features
        """
        df = orders_df.copy()

        # Day of week features
        df['is_weekend'] = df['order_dow'].isin([0, 6]).astype(int)
        df['is_weekday'] = (~df['is_weekend'].astype(bool)).astype(int)

        # Hour of day features
        df['is_morning'] = ((df['order_hour_of_day'] >= 6) &
                            (df['order_hour_of_day'] < 12)).astype(int)
        df['is_afternoon'] = ((df['order_hour_of_day'] >= 12) &
                              (df['order_hour_of_day'] < 18)).astype(int)
        df['is_evening'] = ((df['order_hour_of_day'] >= 18) &
                            (df['order_hour_of_day'] < 24)).astype(int)
        df['is_night'] = ((df['order_hour_of_day'] >= 0) &
                          (df['order_hour_of_day'] < 6)).astype(int)

        # Peak hours
        df['is_peak_hour'] = df['order_hour_of_day'].isin([10, 11, 14, 15, 16]).astype(int)

        # Reorder patterns
        df['days_since_prior_order_log'] = np.log1p(df['days_since_prior_order'].fillna(0))

        logger.info(f"Created {len([c for c in df.columns if c not in orders_df.columns])} time features")

        return df

    def create_user_features(
            self,
            user_orders_df: pd.DataFrame,
            user_products_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create aggregated user features

        Args:
            user_orders_df: User orders data
            user_products_df: User product interactions

        Returns:
            User features dataframe
        """
        # Order-based features
        user_features = user_orders_df.groupby('user_id').agg({
            'order_id': 'count',
            'order_dow': lambda x: x.mode()[0] if not x.empty else 0,
            'order_hour_of_day': ['mean', 'std'],
            'days_since_prior_order': ['mean', 'std', 'min', 'max']
        })

        user_features.columns = ['_'.join(col).strip() for col in user_features.columns]
        user_features.rename(columns={'order_id_count': 'total_orders'}, inplace=True)

        # Product-based features
        product_features = user_products_df.groupby('user_id').agg({
            'product_id': 'nunique',
            'reordered': ['mean', 'sum'],
            'add_to_cart_order': ['mean', 'std']
        })

        product_features.columns = ['_'.join(col).strip() for col in product_features.columns]

        # Merge features
        user_features = user_features.merge(product_features, left_index=True, right_index=True, how='left')

        # Additional calculated features
        user_features['avg_basket_size'] = (
            user_products_df.groupby(['user_id', 'order_id']).size()
            .groupby('user_id').mean()
        )

        user_features['reorder_ratio'] = (
                user_features['reordered_sum'] /
                user_features['product_id_nunique']
        ).fillna(0)

        logger.info(f"Created {user_features.shape[1]} user features for {user_features.shape[0]} users")

        return user_features

    def create_product_features(
            self,
            products_df: pd.DataFrame,
            order_products_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create aggregated product features

        Args:
            products_df: Products metadata
            order_products_df: Order-product interactions

        Returns:
            Product features dataframe
        """
        # Basic product info
        product_features = products_df.set_index('product_id')

        # Aggregated statistics
        product_stats = order_products_df.groupby('product_id').agg({
            'order_id': 'count',
            'reordered': ['mean', 'sum'],
            'add_to_cart_order': ['mean', 'std', 'median']
        })

        product_stats.columns = ['_'.join(col).strip() for col in product_stats.columns]
        product_stats.rename(columns={'order_id_count': 'total_orders'}, inplace=True)

        # Merge features
        product_features = product_features.merge(
            product_stats,
            left_index=True,
            right_index=True,
            how='left'
        )

        # Popularity features
        product_features['popularity_score'] = (
                product_features['total_orders'] /
                product_features['total_orders'].max()
        )

        product_features['reorder_probability'] = product_features['reordered_mean'].fillna(0)

        # Cart position features
        product_features['is_impulse_buy'] = (
                product_features['add_to_cart_order_mean'] >
                product_features['add_to_cart_order_median']
        ).astype(int)

        logger.info(f"Created {product_features.shape[1]} product features for {product_features.shape[0]} products")

        return product_features

    def create_interaction_features(
            self,
            user_product_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create user-product interaction features

        Args:
            user_product_df: User-product interactions

        Returns:
            Enhanced interaction dataframe
        """
        df = user_product_df.copy()

        # Calculate recency
        df['orders_since_last_purchase'] = (
            df.groupby(['user_id', 'product_id'])['order_number']
            .transform(lambda x: x.max() - x + 1)
        )

        # Calculate frequency
        df['purchase_frequency'] = (
            df.groupby(['user_id', 'product_id'])['order_id']
            .transform('count')
        )

        # Calculate user-product affinity
        user_avg_reorder = df.groupby('user_id')['reordered'].transform('mean')
        product_avg_reorder = df.groupby('product_id')['reordered'].transform('mean')

        df['user_product_affinity'] = (
                df['reordered'] - (user_avg_reorder + product_avg_reorder) / 2
        )

        logger.info(f"Created interaction features: {df.shape}")

        return df

    def prepare_sequences(
            self,
            user_orders_df: pd.DataFrame,
            max_sequence_length: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for sequential models (RNN, LSTM)

        Args:
            user_orders_df: User order history
            max_sequence_length: Maximum sequence length

        Returns:
            Tuple of (sequences, labels)
        """
        sequences = []
        labels = []

        for user_id, user_data in user_orders_df.groupby('user_id'):
            # Sort by order number
            user_data = user_data.sort_values('order_number')

            # Get product sequences
            products = user_data['product_id'].values

            # Create sequences
            for i in range(2, min(len(products), max_sequence_length)):
                sequences.append(products[:i - 1])
                labels.append(products[i - 1])

        # Pad sequences
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='pre')
        labels = np.array(labels)

        logger.info(f"Created {len(sequences)} sequences")

        return sequences, labels