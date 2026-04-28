"""Process models defining case execution structure."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import itertools  # For fast task ID generation
import yaml
from pathlib import Path
import numpy as np
import pandas as pd  # For sklearn feature names
from ..entities import Case, Task

# Global counter for fast task ID generation (faster than UUID)
_task_id_counter = itertools.count()


@dataclass
class ProcessVariant:
    """
    A process variant - specific sequence of activities.
    
    Attributes:
        activities: Ordered list of activity names
        frequency: How often this variant occurs (for probability)
        context_filter: Optional context requirements (e.g., {'LoanGoal': 'Investment'})
    """
    activities: List[str]
    frequency: int = 1
    context_filter: Optional[Dict[str, Any]] = None


class ProcessModel(ABC):
    """
    Abstract base class for process models.
    
    Defines how cases progress through activities.
    """
    
    # Class-level cache for activity requirements (shared across all instances)
    _requirements_cache: Optional[Dict[str, float]] = None
    
    @classmethod
    def _load_activity_requirements(cls) -> Dict[str, float]:
        """Load activity requirements from configuration file.
        
        Uses class-level caching to avoid reloading on every instance.
        
        Returns:
            Dict mapping activity names to required capability levels
        """
        # Return cached value if available
        if cls._requirements_cache is not None:
            return cls._requirements_cache
        
        config_path = Path(__file__).parent.parent.parent / "config" / "activity_requirements.yaml"
        
        if not config_path.exists():
            # Cache empty dict if config doesn't exist
            cls._requirements_cache = {}
            return cls._requirements_cache
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            requirements = config.get('activity_requirements', {})
            default_level = config.get('default_requirement', {}).get('level', 50.0)
            
            # Store default level for activities not in config
            if requirements:
                requirements['__default__'] = default_level
            
            # Cache the result
            cls._requirements_cache = requirements
            return requirements
        except Exception as e:
            print(f"Warning: Could not load activity requirements: {e}")
            cls._requirements_cache = {}
            return cls._requirements_cache
    
    @abstractmethod
    def get_initial_tasks(self, case: Case) -> List[Task]:
        """
        Get initial task(s) for a new case.
        
        Args:
            case: The case that just arrived
            
        Returns:
            List of initial tasks to execute
        """
        pass
    
    @abstractmethod
    def get_next_tasks(self, case: Case, completed_task: Task) -> List[Task]:
        """
        Get next task(s) after completing a task.
        
        Args:
            case: The case being executed
            completed_task: Task that was just completed
            
        Returns:
            List of next tasks (empty if case is complete)
        """
        pass
    
    def has_precedence(
        self,
        activity_a: str,
        activity_b: str,
        case: 'Case'
    ) -> bool:
        """Check whether activity_a must complete before activity_b.

        Used by the CP-based batch scheduler (CONSTRAINT 3) to enforce
        task ordering within a case.

        The default implementation returns ``False`` (no precedence),
        which is appropriate for probabilistic models where transitions
        are determined dynamically.  Subclasses backed by a Petri net
        or other explicit ordering can override this.

        Args:
            activity_a: Name of the potential predecessor activity.
            activity_b: Name of the potential successor activity.
            case: The case being planned (may carry context that
                  influences the ordering).

        Returns:
            ``True`` if *activity_a* must finish before *activity_b*
            can start, ``False`` otherwise.
        """
        return False

    def _create_task(
        self,
        case: Case,
        activity_name: str,
        creation_time: float
    ) -> Task:
        """Helper to create a task for a case (OPTIMIZED).
        
        Args:
            case: Parent case
            activity_name: Activity to perform
            creation_time: Current simulation time
            
        Returns:
            New task instance with required capabilities
        """
        # Use fast counter instead of UUID (10x faster for 40k tasks)
        task_id = f"{case.id}_task_{next(_task_id_counter):08d}"
        
        # Get required capabilities for this activity
        required_capability_level = self._get_required_capability(activity_name)
        
        return Task(
            id=task_id,
            case_id=case.id,
            activity_name=activity_name,
            context=case.attributes,  # No copy needed - Task doesn't modify
            creation_time=creation_time,
            required_capability_level=required_capability_level
        )
    
    def _get_required_capability(self, activity_name: str) -> float:
        """Get required capability level for an activity.
        
        Args:
            activity_name: Name of the activity
            
        Returns:
            Required capability level (0-100)
        """
        # Load requirements (uses class-level cache)
        requirements = self._load_activity_requirements()
        
        if activity_name in requirements:
            return requirements[activity_name]
        else:
            # Use default level from config, or fallback to 35.0
            default_level = requirements.get('__default__', 35.0)
            return default_level


class ProbabilisticProcessModel(ProcessModel):
    """
    Probabilistic process model using transition weight models.
    
    Supports two modes:
    1) Simplified first-order transitions (state -> next activity probabilities)
    2) Legacy logistic-regression transition models
    """
    
    def __init__(
        self,
        transition_models: Dict[str, Any],
        metadata: Any,
        history_mode: Optional[str] = None,
        rng_seed: Optional[int] = None
    ):
        """
        Initialize probabilistic process model.
        
        Args:
            transition_models: Dictionary mapping activity names to trained LogisticRegression models
            metadata: TransitionModelMetadata with feature information
            history_mode: 'binary', 'count', or None for execution history tracking
            rng_seed: Random seed for reproducibility
        """
        self.transition_models = transition_models
        self.metadata = metadata
        self.history_mode = history_mode or metadata.history_mode
        self.rng = np.random.default_rng(rng_seed)
        self._simple_transition_mode = self._detect_simple_transition_mode()
        
        # Extract configuration from metadata
        self.context_attributes = metadata.context_attributes
        self.categorical_attributes = metadata.categorical_attributes
        self.categorical_values = metadata.categorical_values
        self.activity_labels = metadata.activity_labels # 'W_Assess potential fraud', 'W_Call after offers', ...
        
        # Build feature names for legacy mode
        self._build_feature_names()

    def _detect_simple_transition_mode(self) -> bool:
        """Detect whether transition_models stores state->next probability dicts."""
        if not self.transition_models:
            return False

        for value in self.transition_models.values():
            if isinstance(value, dict):
                return True
        return False
    
    def _build_feature_names(self):
        """Build ordered list of feature names matching training data."""
        features = []
        
        # Add context features (one-hot encoded categoricals)
        for attr in self.context_attributes:
            if attr in self.categorical_attributes:
                # One-hot encoded
                for value in self.categorical_values.get(attr, []):
                    features.append(f'{attr}_{value}')
            else:
                # Numeric
                features.append(attr)
        
        # Add history features if enabled
        if self.history_mode:
            for label in self.activity_labels:
                features.append(f'{label}_history')
        
        self.feature_names = features
    
    def _get_execution_history(self, case: Case) -> Dict[str, int]:
        """
        Get execution history for case (optimized with cache).
        
        Args:
            case: Case to get history for
            
        Returns:
            Dictionary mapping activity names to execution counts (binary or count)
        """
        history = {label: 0 for label in self.activity_labels}
        
        if not self.history_mode:
            return history
        
        # Use cached execution history (updated incrementally in Case.add_completed_activity)
        for activity_name, count in case._execution_history.items():
            if activity_name in history:
                if self.history_mode == 'binary':
                    history[activity_name] = 1 if count > 0 else 0
                elif self.history_mode == 'count':
                    history[activity_name] = count
        
        return history
    
    def _build_feature_vector(self, context: Dict[str, Any], history: Dict[str, int]) -> pd.DataFrame:
        """Build feature vector matching training format.
        
        Returns:
            DataFrame with single row containing features (maintains column names)
        """
        features = {}
        
        # Add context features (one-hot encoded)
        for attr in self.context_attributes:
            value = context.get(attr)
            if attr in self.categorical_values:
                for cat_value in self.categorical_values[attr]:
                    features[f'{attr}_{cat_value}'] = 1 if value == cat_value else 0
            else:
                features[attr] = value if value is not None else 0
        
        # Add history features if enabled
        if self.history_mode:
            for activity_label in self.activity_labels:
                features[f'{activity_label}_history'] = history.get(activity_label, 0)
        
        # Return as DataFrame (single row)
        return pd.DataFrame([features])
    
    def _compute_transition_probabilities(
        self,
        case: Case,
        enabled_activities: List[str]
    ) -> Dict[str, float]:
        """
        Compute firing probabilities for enabled transitions.
        
        Args:
            case: Current case
            enabled_activities: List of activities that can be executed
            
        Returns:
            Dictionary mapping activity names to probabilities
        """
        # Simplified mode: use direct relative transition probabilities
        if self._simple_transition_mode:
            if not case.trace:
                return dict(self.transition_models.get('__START__', {}))
            current_activity = case.trace[-1].activity_name
            return dict(self.transition_models.get(current_activity, {}))

        # Legacy mode: logistic models using context + history
        # Get execution history
        history = self._get_execution_history(case)
        
        # Build feature vector
        feature_vector = self._build_feature_vector(case.attributes, history)
        
        # Compute probabilities only for enabled activities
        probabilities = {}
        for activity in enabled_activities:
            model = self.transition_models.get(activity)
            if model is not None:
                try:
                    proba = model.predict_proba(feature_vector)[0][1]
                    probabilities[activity] = proba
                except Exception:
                    probabilities[activity] = 1.0
            else:
                probabilities[activity] = 1.0
        
        return probabilities
    
    def _select_next_activity(
        self,
        case: Case,
        enabled_activities: List[str]
    ) -> Optional[str]:
        """
        Select next activity using weighted random sampling.
        
        Args:
            case: Current case
            enabled_activities: List of activities that can be executed
            
        Returns:
            Selected activity name, or None if no valid activities
        """
        # In simplified mode, allowed next activities are exactly outgoing transitions
        # from the current state.
        if self._simple_transition_mode:
            probabilities = self._compute_transition_probabilities(case, enabled_activities=[])
            if not probabilities:
                return None
            activities = list(probabilities.keys())
            weights = np.array([probabilities[a] for a in activities], dtype=float)
            total_weight = weights.sum()
            if total_weight <= 0:
                return None
            probabilities_normalized = weights / total_weight
            return self.rng.choice(activities, p=probabilities_normalized)

        if not enabled_activities:
            return None

        # Legacy mode
        probabilities = self._compute_transition_probabilities(case, enabled_activities)
        
        # Normalize probabilities
        activities = list(probabilities.keys())
        weights = np.array([probabilities[a] for a in activities])
        
        total_weight = weights.sum()
        if total_weight == 0:
            # Fallback: uniform selection
            return self.rng.choice(activities)
        
        probabilities_normalized = weights / total_weight
        
        # Weighted random selection
        selected = self.rng.choice(activities, p=probabilities_normalized)
        
        return selected
    
    def _get_enabled_activities(self, case: Case) -> List[str]:
        """
        Determine which activities can be executed next (optimized with cache).
        
        For simplicity, we allow any activity that:
        1. Hasn't been executed yet (if in activity_labels)
        2. OR is a known activity type
        
        In a full Petri net implementation, this would check marking.
        
        Args:
            case: Current case
            
        Returns:
            List of enabled activity names
        """
        # Initialize enabled activities cache on first call
        if case._enabled_activities is None: # Initialize cache on first call: all activities enabled at start
            case._enabled_activities = set(self.activity_labels)
            # Remove already executed activities
            for activity_name in case._execution_history.keys():
                case._enabled_activities.discard(activity_name)
        
        # Return as list (cache is maintained incrementally in Case.add_completed_activity)
        enabled = list(case._enabled_activities)
        
        # If no activities enabled, case might be complete
        return enabled
    
    def get_initial_tasks(self, case: Case) -> List[Task]:
        """Get first task using probabilistic selection."""
        initial_activity = case.get_initial_activity()

        # Fallback to START distribution for robustness if initial activity is missing.
        if initial_activity is None and self._simple_transition_mode:
            start_probs = self.transition_models.get('__START__', {})
            if start_probs:
                activities = list(start_probs.keys())
                weights = np.array([start_probs[a] for a in activities], dtype=float)
                total = weights.sum()
                if total > 0:
                    initial_activity = self.rng.choice(activities, p=weights / total)

        if initial_activity is None:
            return []

        task = self._create_task(case, initial_activity, case.arrival_time)
        return [task]
    
    def get_next_tasks(self, case: Case, completed_task: Task) -> List[Task]:
        """Get next task using probabilistic selection."""
        if self._simple_transition_mode:
            next_activity = self._select_next_activity(case, enabled_activities=[])
            if next_activity is None:
                return []
            task = self._create_task(case, next_activity, completed_task.actual_end_time)
            return [task]

        # Determine enabled activities
        enabled_activities = self._get_enabled_activities(case)
        
        if not enabled_activities:
            # Case complete
            return []
        
        # Select next activity probabilistically
        next_activity = self._select_next_activity(case, enabled_activities)
        
        if next_activity is None:
            return []
        
        task = self._create_task(case, next_activity, completed_task.actual_end_time)
        return [task]

