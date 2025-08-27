"""
Utility functions for the recommendation system
"""
import os
import json
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import time
from functools import wraps

logger = logging.getLogger(__name__)


def timer(func):
    """
    Decorator to time function execution
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result

    return wrapper


def ensure_dir(directory: Path) -> None:
    """
    Ensure directory exists, create if not

    Args:
        directory: Path to directory
    """
    directory = Path(directory)
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def save_pickle(obj: Any, filepath: Path) -> None:
    """
    Save object as pickle file

    Args:
        obj: Object to save
        filepath: Path to save file
    """
    ensure_dir(filepath.parent)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Saved pickle: {filepath}")


def load_pickle(filepath: Path) -> Any:
    """
    Load pickle file

    Args:
        filepath: Path to pickle file

    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    logger.info(f"Loaded pickle: {filepath}")
    return obj


def save_json(data: Dict, filepath: Path) -> None:
    """
    Save dictionary as JSON file

    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    ensure_dir(filepath.parent)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved JSON: {filepath}")


def load_json(filepath: Path) -> Dict:
    """
    Load JSON file

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded JSON: {filepath}")
    return data


def hash_dataframe(df: pd.DataFrame) -> str:
    """
    Generate hash for dataframe (useful for caching)

    Args:
        df: Input dataframe

    Returns:
        Hash string
    """
    return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()


def sample_users(
        user_item_matrix: pd.DataFrame,
        n_users: int,
        min_items: int = 5,
        random_state: int = 42
) -> List[int]:
    """
    Sample users with minimum number of items

    Args:
        user_item_matrix: User-item interaction matrix
        n_users: Number of users to sample
        min_items: Minimum items per user
        random_state: Random seed

    Returns:
        List of sampled user indices
    """
    np.random.seed(random_state)

    # Count items per user
    items_per_user = (user_item_matrix > 0).sum(axis=1)

    # Filter users with minimum items
    valid_users = items_per_user[items_per_user >= min_items].index

    # Sample users
    if len(valid_users) < n_users:
        logger.warning(f"Only {len(valid_users)} users have >= {min_items} items")
        sampled_users = valid_users.tolist()
    else:
        sampled_users = np.random.choice(valid_users, n_users, replace=False).tolist()

    logger.info(f"Sampled {len(sampled_users)} users")

    return sampled_users


def memory_usage(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate memory usage of dataframe

    Args:
        df: Input dataframe

    Returns:
        Dictionary with memory usage stats
    """
    memory = df.memory_usage(deep=True)
    total_mb = memory.sum() / 1024 / 1024

    return {
        'total_mb': total_mb,
        'per_column': {col: memory[col] / 1024 / 1024 for col in df.columns}
    }


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory usage of dataframe by downcasting

    Args:
        df: Input dataframe

    Returns:
        Optimized dataframe
    """
    start_mem = memory_usage(df)['total_mb']

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = memory_usage(df)['total_mb']

    logger.info(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB "
                f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")

    return df


def generate_experiment_id() -> str:
    """
    Generate unique experiment ID

    Returns:
        Experiment ID string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = np.random.randint(1000, 9999)
    return f"exp_{timestamp}_{random_suffix}"


def log_experiment(
        experiment_id: str,
        params: Dict,
        metrics: Dict,
        filepath: Optional[Path] = None
) -> None:
    """
    Log experiment parameters and results

    Args:
        experiment_id: Unique experiment identifier
        params: Model parameters
        metrics: Evaluation metrics
        filepath: Path to save log
    """
    if filepath is None:
        filepath = Path("experiments.json")

    experiment = {
        'id': experiment_id,
        'timestamp': datetime.now().isoformat(),
        'parameters': params,
        'metrics': metrics
    }

    # Load existing experiments
    if filepath.exists():
        experiments = load_json(filepath)
    else:
        experiments = []

    experiments.append(experiment)

    # Save updated experiments
    save_json(experiments, filepath)

    logger.info(f"Logged experiment: {experiment_id}")


def get_best_experiment(
        metric: str = 'rmse',
        ascending: bool = True,
        filepath: Path = Path("experiments.json")
) -> Optional[Dict]:
    """
    Get best experiment based on metric

    Args:
        metric: Metric to optimize
        ascending: Whether lower is better
        filepath: Path to experiments log

    Returns:
        Best experiment dictionary
    """
    if not filepath.exists():
        logger.warning("No experiments found")
        return None

    experiments = load_json(filepath)

    if not experiments:
        logger.warning("No experiments found")
        return None

    # Sort by metric
    sorted_exps = sorted(
        experiments,
        key=lambda x: x['metrics'].get(metric, float('inf') if ascending else float('-inf')),
        reverse=not ascending
    )

    best_exp = sorted_exps[0]
    logger.info(f"Best experiment: {best_exp['id']} with {metric}={best_exp['metrics'][metric]:.4f}")

    return best_exp


def create_submission(
        predictions: Dict[int, List[int]],
        filepath: Path
) -> None:
    """
    Create submission file for competition

    Args:
        predictions: Dictionary of user_id to list of product_ids
        filepath: Path to save submission
    """
    submission = []

    for user_id, product_ids in predictions.items():
        # Convert product IDs to space-separated string
        products_str = ' '.join(map(str, product_ids)) if product_ids else 'None'
        submission.append({
            'user_id': user_id,
            'products': products_str
        })

    submission_df = pd.DataFrame(submission)
    submission_df.to_csv(filepath, index=False)

    logger.info(f"Created submission: {filepath}")


def calculate_diversity(
        recommendations: List[List[int]],
        item_features: Optional[pd.DataFrame] = None
) -> float:
    """
    Calculate diversity of recommendations

    Args:
        recommendations: List of recommendation lists
        item_features: Optional item features for content diversity

    Returns:
        Diversity score
    """
    all_items = set()
    for rec_list in recommendations:
        all_items.update(rec_list)

    if not all_items:
        return 0.0

    # Catalog coverage
    catalog_coverage = len(all_items)

    # Average intra-list diversity
    if item_features is not None:
        # Content-based diversity
        diversity_scores = []
        for rec_list in recommendations:
            if len(rec_list) > 1:
                # Calculate pairwise distances
                features = item_features.loc[rec_list]
                distances = []
                for i in range(len(rec_list)):
                    for j in range(i + 1, len(rec_list)):
                        dist = np.linalg.norm(features.iloc[i] - features.iloc[j])
                        distances.append(dist)
                if distances:
                    diversity_scores.append(np.mean(distances))

        diversity = np.mean(diversity_scores) if diversity_scores else 0.0
    else:
        # Simple diversity based on unique items
        diversity = catalog_coverage / sum(len(rec) for rec in recommendations)

    return diversity