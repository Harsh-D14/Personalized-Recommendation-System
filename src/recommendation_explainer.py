"""
Recommendation Explanation System
Provides explanations for why certain products were recommended
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class RecommendationExplainer:
    """
    Explains recommendations by showing which user behaviors or items influenced each recommendation
    """

    def __init__(self, user_item_matrix: pd.DataFrame, product_names: Dict[int, str]):
        self.user_item_matrix = user_item_matrix
        self.product_names = product_names

    def explain_user_based_cf(
            self,
            model,
            user_idx: int,
            recommended_items: List[int],
            n_similar_users: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Explain User-Based CF recommendations
        """
        explanations = []

        # Get similar users who influenced recommendations
        if hasattr(model, 'user_similarity'):
            similarities = model.user_similarity[user_idx]
            top_similar_users = np.argsort(similarities)[-n_similar_users - 1:-1][::-1]

            for item_idx in recommended_items:
                item_id = self.user_item_matrix.columns[item_idx]

                # Find which similar users bought this item
                users_who_bought = []
                for similar_user in top_similar_users:
                    if model.train_matrix[similar_user, item_idx] > 0:
                        users_who_bought.append({
                            'user': similar_user,
                            'similarity': similarities[similar_user],
                            'rating': model.train_matrix[similar_user, item_idx]
                        })

                explanation = {
                    'item_id': item_id,
                    'item_name': self.product_names.get(item_id, f"Product {item_id}"),
                    'reason': f"Similar users also bought this",
                    'similar_users_count': len(users_who_bought),
                    'top_influence': users_who_bought[:2] if users_who_bought else []
                }
                explanations.append(explanation)

        return explanations

    def explain_item_based_cf(
            self,
            model,
            user_idx: int,
            recommended_items: List[int],
            n_similar_items: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Explain Item-Based CF recommendations
        """
        explanations = []

        # Get user's purchased items
        user_items = np.where(model.train_matrix[user_idx] > 0)[0]
        user_item_ids = self.user_item_matrix.columns[user_items].tolist()

        for item_idx in recommended_items:
            item_id = self.user_item_matrix.columns[item_idx]

            # Find which user items are similar to this recommendation
            similar_purchased = []
            if hasattr(model, 'item_similarity'):
                for user_item in user_items:
                    similarity = model.item_similarity[item_idx, user_item]
                    if similarity > 0:
                        similar_purchased.append({
                            'item_id': self.user_item_matrix.columns[user_item],
                            'item_name': self.product_names.get(
                                self.user_item_matrix.columns[user_item],
                                f"Product {self.user_item_matrix.columns[user_item]}"
                            ),
                            'similarity': similarity
                        })

                # Sort by similarity
                similar_purchased.sort(key=lambda x: x['similarity'], reverse=True)

            explanation = {
                'item_id': item_id,
                'item_name': self.product_names.get(item_id, f"Product {item_id}"),
                'reason': "Frequently bought together with your items",
                'based_on_items': similar_purchased[:n_similar_items],
                'connection_strength': np.mean(
                    [x['similarity'] for x in similar_purchased[:n_similar_items]]) if similar_purchased else 0
            }
            explanations.append(explanation)

        return explanations

    def explain_svd(
            self,
            model,
            user_idx: int,
            recommended_items: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Explain SVD recommendations using latent factors
        """
        explanations = []

        # Get user's latent factors
        user_factors = model.user_factors[user_idx] if hasattr(model, 'user_factors') else None

        for item_idx in recommended_items:
            item_id = self.user_item_matrix.columns[item_idx]

            if user_factors is not None and hasattr(model, 'item_factors'):
                # Find dominant factors
                item_factors = model.item_factors[item_idx]
                factor_contributions = user_factors * item_factors
                top_factors = np.argsort(np.abs(factor_contributions))[-3:][::-1]

                explanation = {
                    'item_id': item_id,
                    'item_name': self.product_names.get(item_id, f"Product {item_id}"),
                    'reason': "Matches your preference patterns",
                    'factor_strength': np.abs(factor_contributions[top_factors[0]]),
                    'pattern_match': self._interpret_factors(top_factors)
                }
            else:
                explanation = {
                    'item_id': item_id,
                    'item_name': self.product_names.get(item_id, f"Product {item_id}"),
                    'reason': "Discovered through latent pattern analysis",
                    'factor_strength': 0,
                    'pattern_match': "General preference alignment"
                }

            explanations.append(explanation)

        return explanations

    def explain_nmf(
            self,
            model,
            user_idx: int,
            recommended_items: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Explain NMF recommendations using non-negative factors
        """
        explanations = []

        # Get user and item factors
        user_factors = model.W[user_idx] if hasattr(model, 'W') else None

        for item_idx in recommended_items:
            item_id = self.user_item_matrix.columns[item_idx]

            if user_factors is not None and hasattr(model, 'H'):
                # Get item factors
                item_factors = model.H[:, item_idx]

                # Find which components contribute most
                contributions = user_factors * item_factors
                top_components = np.argsort(contributions)[-3:][::-1]

                explanation = {
                    'item_id': item_id,
                    'item_name': self.product_names.get(item_id, f"Product {item_id}"),
                    'reason': "Matches your shopping patterns",
                    'pattern_strength': contributions[top_components[0]],
                    'shopping_profile': self._interpret_nmf_components(top_components)
                }
            else:
                explanation = {
                    'item_id': item_id,
                    'item_name': self.product_names.get(item_id, f"Product {item_id}"),
                    'reason': "Identified through shopping behavior analysis",
                    'pattern_strength': 0,
                    'shopping_profile': "General preference"
                }

            explanations.append(explanation)

        return explanations

    def _interpret_factors(self, factor_indices: np.ndarray) -> str:
        """
        Interpret latent factors into human-readable patterns
        """
        # This is simplified - in reality, you'd analyze the factors
        patterns = [
            "Health-conscious choices",
            "Family shopping patterns",
            "Quick meal preferences",
            "Organic product interest",
            "Budget-conscious shopping",
            "Snack preferences",
            "Fresh produce focus",
            "Convenience items",
            "Breakfast essentials",
            "International cuisine"
        ]

        if len(factor_indices) > 0 and factor_indices[0] < len(patterns):
            return patterns[factor_indices[0] % len(patterns)]
        return "General shopping pattern"

    def _interpret_nmf_components(self, component_indices: np.ndarray) -> str:
        """
        Interpret NMF components into shopping profiles
        """
        profiles = [
            "Healthy lifestyle shopper",
            "Family meal planner",
            "Quick & convenient shopper",
            "Fresh food enthusiast",
            "Snack lover",
            "Baby/toddler parent",
            "Pet owner",
            "Party host",
            "Breakfast focused",
            "International food explorer"
        ]

        if len(component_indices) > 0 and component_indices[0] < len(profiles):
            return profiles[component_indices[0] % len(profiles)]
        return "Mixed shopping profile"

    def get_explanation(
            self,
            model_type: str,
            model: Any,
            user_idx: int,
            recommended_items: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Get explanations based on model type
        """
        model_type = model_type.lower().replace('-', '_').replace(' ', '_')

        if 'user' in model_type:
            return self.explain_user_based_cf(model, user_idx, recommended_items)
        elif 'item' in model_type:
            return self.explain_item_based_cf(model, user_idx, recommended_items)
        elif 'svd' in model_type:
            return self.explain_svd(model, user_idx, recommended_items)
        elif 'nmf' in model_type:
            return self.explain_nmf(model, user_idx, recommended_items)
        else:
            # Generic explanation
            generic_explanations = []
            for item_idx in recommended_items:
                item_id = self.user_item_matrix.columns[item_idx]
                generic_explanations.append({
                    'item_id': item_id,
                    'item_name': self.product_names.get(item_id, f"Product {item_id}"),
                    'reason': "Recommended based on your preferences",
                    'confidence': 'Medium'
                })
            return generic_explanations

    def format_explanation_text(self, explanation: Dict[str, Any], model_type: str) -> str:
        """
        Format explanation into readable text
        """
        item_name = explanation['item_name']
        reason = explanation['reason']

        if 'user' in model_type.lower():
            if explanation.get('similar_users_count', 0) > 0:
                return f"**{item_name}**: {reason} ({explanation['similar_users_count']} similar shoppers bought this)"
            return f"**{item_name}**: {reason}"

        elif 'item' in model_type.lower():
            if explanation.get('based_on_items'):
                top_item = explanation['based_on_items'][0] if explanation['based_on_items'] else None
                if top_item:
                    return f"**{item_name}**: Because you bought *{top_item['item_name']}* (similarity: {top_item['similarity']:.2f})"
            return f"**{item_name}**: {reason}"

        elif 'svd' in model_type.lower():
            pattern = explanation.get('pattern_match', 'preference pattern')
            return f"**{item_name}**: Matches your '{pattern}'"

        elif 'nmf' in model_type.lower():
            profile = explanation.get('shopping_profile', 'shopping profile')
            return f"**{item_name}**: Fits your '{profile}'"

        return f"**{item_name}**: {reason}"