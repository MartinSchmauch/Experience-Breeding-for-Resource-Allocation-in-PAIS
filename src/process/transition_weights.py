"""
Transition weight learning for probabilistic process branching.

Implements Padella et al.'s approach using logistic regression to predict
transition firing probabilities based on case context and execution history.

Based on: de Leoni et al. "Data-Aware Process Simulations with Petri Nets"
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import Counter
import logging
from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass
import pickle
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TransitionModelMetadata:
    """Metadata for transition probability models."""
    context_attributes: List[str]
    categorical_attributes: List[str]
    categorical_values: Dict[str, List[str]]
    history_mode: Optional[str]  # None, 'binary', or 'count'
    activity_labels: List[str]
    feature_names: List[str]


def count_activity_revisits(activities: List[str]) -> int:
    """Count how many activity executions are revisits within a trace."""
    counts = Counter(activities)
    return sum(max(count - 1, 0) for count in counts.values())


def has_excessive_loops(
    activities: List[str],
    max_activity_occurrences: Optional[int]
) -> bool:
    """Return whether a trace exceeds the allowed number of visits per activity."""
    if max_activity_occurrences is None:
        return False

    if max_activity_occurrences < 1:
        raise ValueError("max_activity_occurrences must be at least 1")

    counts = Counter(activities)
    return any(count > max_activity_occurrences for count in counts.values())


def trim_looping_activities(
    activities: List[str],
    max_activity_occurrences: Optional[int]
) -> List[str]:
    """Trim repeated visits beyond the allowed number of occurrences per activity."""
    if max_activity_occurrences is None:
        return list(activities)

    if max_activity_occurrences < 1:
        raise ValueError("max_activity_occurrences must be at least 1")

    trimmed_activities: List[str] = []
    counts: Counter[str] = Counter()

    for activity in activities:
        if counts[activity] >= max_activity_occurrences:
            continue
        trimmed_activities.append(activity)
        counts[activity] += 1

    return trimmed_activities


class TransitionWeightBuilder:
    """
    Build transition probability models from event logs.
    
    Creates logistic regression models per transition that predict
    firing probability based on case context and execution history.
    """
    
    def __init__(
        self,
        context_attributes: List[str],
        categorical_attributes: Optional[List[str]] = None,
        history_mode: Optional[str] = None,
        loop_handling: str = 'keep',
        max_activity_occurrences: Optional[int] = None
    ):
        """
        Initialize transition weight builder.
        
        Args:
            context_attributes: Case attributes to use as features (e.g., ['case:LoanGoal', 'case:ApplicationType'])
            categorical_attributes: Which context attributes are categorical (for one-hot encoding)
            history_mode: 'binary' (executed or not), 'count' (execution count), or None (no history)
            loop_handling: How to handle loop-heavy traces: 'keep', 'remove', or 'trim'
            max_activity_occurrences: Maximum allowed visits per activity before a trace is considered loop-heavy
        """
        self.context_attributes = context_attributes
        self.categorical_attributes = categorical_attributes or []
        self.history_mode = history_mode
        self.loop_handling = loop_handling
        self.max_activity_occurrences = max_activity_occurrences

        if self.loop_handling not in {'keep', 'remove', 'trim'}:
            raise ValueError("loop_handling must be one of: 'keep', 'remove', 'trim'")
        
        # Will be populated during training
        self.categorical_values: Dict[str, List[str]] = {}
        self.activity_labels: List[str] = []
    
    def build_training_datasets(
        self,
        log_df: pd.DataFrame,
        variants: List[Any],  # ProcessVariant objects
        case_column: str = 'case:concept:name',
        activity_column: str = 'concept:name',
        timestamp_column: str = 'time:timestamp'
    ) -> Dict[str, Dict[str, List]]:
        """
        Build training datasets for each transition from event log.
        
        Replays traces through process variants, labels enabled transitions
        as fired (1) or not fired (0), and extracts features.
        
        Args:
            log_df: Event log dataframe
            variants: List of ProcessVariant objects with activity sequences
            case_column: Column name for case ID
            activity_column: Column name for activity
            timestamp_column: Column name for timestamp
            
        Returns:
            Dictionary mapping activity names to training datasets:
            {
                'ActivityA': {
                    'case:LoanGoal': [...],
                    'case:ApplicationType': [...],
                    'ActivityA_history': [...],  # if history_mode enabled
                    'ActivityB_history': [...],
                    'class': [1, 0, 1, ...]  # target: fired or not
                },
                ...
            }
        """
        logger.info("Building training datasets for transition models...")
        
        # Sort by case and timestamp
        df = log_df.sort_values([case_column, timestamp_column]).copy()
        
        # Extract unique activity labels
        self.activity_labels = sorted(df[activity_column].unique().tolist())
        
        # Discover categorical values
        for attr in self.categorical_attributes:
            if attr in df.columns:
                self.categorical_values[attr] = sorted(df[attr].dropna().unique().tolist())
        
        # Initialize dataset structure for each activity
        activity_datasets: Dict[str, Dict[str, List]] = {}
        for activity in self.activity_labels:
            activity_datasets[activity] = {
                **{attr: [] for attr in self.context_attributes},
                'class': []
            }
            
            # Add history features if enabled
            if self.history_mode:
                for label in self.activity_labels:
                    activity_datasets[activity][f'{label}_history'] = []
        
        # Process each case
        total_cases = df[case_column].nunique()
        allowed_variants = None
        if variants is not None:
            allowed_variants = {tuple(variant.activities) for variant in variants}

        logger.info(f"Processing {total_cases} cases...")

        kept_cases = 0
        skipped_variant_cases = 0
        skipped_loop_cases = 0
        trimmed_cases = 0
        
        for case_id, case_df in tqdm(df.groupby(case_column), total=total_cases, desc="Building datasets"):
            # Extract case attributes (static for the case)
            case_attributes = {}
            for attr in self.context_attributes:
                if attr in case_df.columns:
                    values = case_df[attr].dropna()
                    case_attributes[attr] = values.iloc[0] if len(values) > 0 else None
            
            # Get activity sequence for this case
            activities = case_df[activity_column].tolist()

            if allowed_variants is not None and tuple(activities) not in allowed_variants:
                skipped_variant_cases += 1
                continue

            if self.loop_handling == 'remove' and has_excessive_loops(activities, self.max_activity_occurrences):
                skipped_loop_cases += 1
                continue

            if self.loop_handling == 'trim':
                trimmed_activities = trim_looping_activities(activities, self.max_activity_occurrences)
                if trimmed_activities != activities:
                    trimmed_cases += 1
                activities = trimmed_activities

            if not activities:
                skipped_loop_cases += 1
                continue

            kept_cases += 1
            
            # Initialize execution history
            if self.history_mode:
                history = {label: 0 for label in self.activity_labels}
            
            # Replay trace activity by activity
            for i, current_activity in enumerate(activities):
                # Get activities that could have been executed at this point
                # (all activities that appear after current position)
                remaining_activities = set(activities[i:])
                
                # Label all remaining activities
                for potential_activity in remaining_activities:
                    # Add case attributes as features
                    for attr in self.context_attributes:
                        activity_datasets[potential_activity][attr].append(
                            case_attributes.get(attr, None)
                        )
                    
                    # Add history features if enabled
                    if self.history_mode:
                        for label in self.activity_labels:
                            activity_datasets[potential_activity][f'{label}_history'].append(
                                history[label]
                            )
                    
                    # Label: 1 if this is the activity that was actually executed, 0 otherwise
                    is_fired = 1 if potential_activity == current_activity else 0
                    activity_datasets[potential_activity]['class'].append(is_fired)
                
                # Update history after firing current activity
                if self.history_mode:
                    if self.history_mode == 'binary':
                        history[current_activity] = 1
                    elif self.history_mode == 'count':
                        history[current_activity] += 1
        
        logger.info(f"Built training datasets for {len(activity_datasets)} activities")
        
        # Log dataset statistics
        for activity, dataset in activity_datasets.items():
            n_samples = len(dataset['class'])
            n_positive = sum(dataset['class'])
            if n_samples > 0:
                logger.debug(f"  {activity}: {n_samples} samples ({n_positive} positive, {n_samples - n_positive} negative)")

            logger.info(
                "Training dataset case usage: kept=%s, skipped_by_variant=%s, skipped_by_loops=%s, trimmed=%s",
                kept_cases,
                skipped_variant_cases,
                skipped_loop_cases,
                trimmed_cases,
            )
        
        return activity_datasets
    
    def train_models(
        self,
        activity_datasets: Dict[str, Dict[str, List]]
    ) -> Tuple[Dict[str, Optional[LogisticRegression]], pd.DataFrame]:
        """
        Train logistic regression models for each activity.
        
        Args:
            activity_datasets: Training datasets from build_training_datasets()
            
        Returns:
            Tuple of:
            - models: Dictionary mapping activity names to trained LogisticRegression models (or None if insufficient data)
            - coefficients: DataFrame showing model weights for interpretability
        """
        logger.info("Training transition probability models...")
        
        models = {}
        coefficients_list = []
        coeff_index = []
        
        for activity, dataset in activity_datasets.items():
            # Convert to DataFrame
            df = pd.DataFrame(dataset)
            
            # Skip if no data or only one class
            if len(df) == 0 or len(df['class'].unique()) < 2:
                logger.debug(f"  {activity}: Skipped (insufficient data or single class)")
                models[activity] = None
                continue
            
            # One-hot encode categorical attributes
            X_df = df.copy()
            for attr in self.categorical_attributes:
                if attr in X_df.columns:
                    if attr in self.categorical_values:
                        for value in self.categorical_values[attr]:
                            X_df[f'{attr}_{value}'] = (X_df[attr] == value).astype(int)
                    X_df = X_df.drop(columns=[attr])
            
            # Prepare training data
            X = X_df.drop(columns=['class'])
            y = df['class']
            
            # Handle missing values (fill with 0 for simplicity)
            X = X.fillna(0)
            
            try:
                # Train logistic regression
                clf = LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    solver='lbfgs'
                )
                clf.fit(X, y)
                
                models[activity] = clf
                
                # Store coefficients for interpretability
                coeff_index.append(activity)
                coefficients_list.append([clf.intercept_[0]] + list(clf.coef_[0]))
                
                logger.debug(f"  {activity}: Trained successfully ({len(df)} samples)")
                
            except Exception as e:
                logger.warning(f"  {activity}: Training failed - {e}")
                models[activity] = None
        
        # Create coefficients DataFrame
        if coefficients_list:
            feature_names = ['intercept'] + list(X.columns)
            coefficients = pd.DataFrame(
                coefficients_list,
                columns=feature_names,
                index=coeff_index
            )
        else:
            coefficients = pd.DataFrame()
        
        logger.info(f"Trained {sum(1 for m in models.values() if m is not None)}/{len(models)} models successfully")
        
        return models, coefficients
    
    def save_models(
        self,
        models: Dict[str, Optional[LogisticRegression]],
        coefficients: pd.DataFrame,
        output_path: Path
    ) -> None:
        """
        Save trained models and metadata to file.
        
        Args:
            models: Dictionary of trained models
            coefficients: DataFrame of model coefficients
            output_path: Path to save models (e.g., 'data/transition_models.pkl')
        """
        # Create metadata
        metadata = TransitionModelMetadata(
            context_attributes=self.context_attributes,
            categorical_attributes=self.categorical_attributes,
            categorical_values=self.categorical_values,
            history_mode=self.history_mode,
            activity_labels=self.activity_labels,
            feature_names=list(coefficients.columns) if len(coefficients) > 0 else []
        )
        
        # Package everything
        package = {
            'models': models,
            'coefficients': coefficients,
            'metadata': metadata
        }
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(package, f)
        
        logger.info(f"Saved transition models to {output_path}")
    
    @staticmethod
    def load_models(input_path: Path) -> Tuple[Dict[str, Optional[LogisticRegression]], pd.DataFrame, TransitionModelMetadata]:
        """
        Load trained models from file.
        
        Args:
            input_path: Path to load models from
            
        Returns:
            Tuple of (models, coefficients, metadata)
        """
        with open(input_path, 'rb') as f:
            package = pickle.load(f)
        
        return package['models'], package['coefficients'], package['metadata']


def compute_transition_probability(
    model: LogisticRegression,
    feature_vector: np.ndarray
) -> float:
    """
    Compute firing probability for a transition given features.
    
    Manual logistic regression implementation for clarity.
    
    Args:
        model: Trained LogisticRegression model
        feature_vector: Feature values
        
    Returns:
        Probability of transition firing (0-1)
    """
    coeff = model.coef_[0]
    intercept = model.intercept_[0]
    
    # Logistic regression: P(y=1|X) = 1 / (1 + exp(-(b + w·X)))
    logit = intercept + np.dot(coeff, feature_vector)
    probability = 1.0 / (1.0 + np.exp(-logit))
    
    return probability


def select_transition_weighted(
    transition_probabilities: Dict[str, float],
    enabled_transitions: Set[str],
    rng: Optional[np.random.Generator] = None
) -> Optional[str]:
    """
    Select transition using weighted random sampling (roulette wheel).
    
    Args:
        transition_probabilities: Dictionary mapping activity names to probabilities
        enabled_transitions: Set of activities that can be executed
        rng: Random number generator
        
    Returns:
        Selected activity name, or None if no valid transitions
    """
    if not enabled_transitions:
        return None
    
    if rng is None:
        rng = np.random.default_rng()
    
    # Filter to enabled transitions and their weights
    enabled_probs = {
        activity: transition_probabilities.get(activity, 1.0)
        for activity in enabled_transitions
    }
    
    # Normalize to probabilities
    total_weight = sum(enabled_probs.values())
    if total_weight == 0:
        # Fallback: uniform selection
        return rng.choice(list(enabled_transitions))
    
    # Weighted random selection (roulette wheel)
    activities = list(enabled_probs.keys())
    weights = np.array([enabled_probs[a] for a in activities])
    probabilities = weights / total_weight
    
    selected = rng.choice(activities, p=probabilities)
    
    return selected
