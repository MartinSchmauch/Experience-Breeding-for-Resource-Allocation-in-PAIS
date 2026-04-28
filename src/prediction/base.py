"""Abstract base class for duration prediction."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DurationPredictor(ABC):
    """
    Abstract base class for task duration prediction.
    
    This interface allows different prediction strategies to be plugged into
    schedulers. The architecture is designed to support future extensions:
    - Offline learning (current): Train models on historical data
    - Online learning (future): Update models during simulation
    - Ensemble methods (future): Combine multiple predictors
    """
    
    def __init__(self, fallback_value: float = 1.0):
        """
        Initialize predictor with fallback configuration.
        
        Args:
            fallback_value: Default duration when prediction fails
        """
        self.fallback_value = fallback_value
        self._prediction_failures = 0
        self._total_predictions = 0
    
    @abstractmethod
    def predict(
        self,
        resource_id: str,
        activity_name: str,
        context: Dict[str, Any],
        experience_profile: Optional[Any] = None
    ) -> float:
        """
        Predict task duration for resource executing activity in context.
        
        Args:
            resource_id: ID of the resource
            activity_name: Name of the activity
            context: Case context attributes (e.g., LoanGoal, ApplicationType)
            experience_profile: Optional ExperienceProfile for feature extraction
        
        Returns:
            Predicted duration in hours
        """
        pass
    
    def predict_with_fallback(
        self,
        resource_id: str,
        activity_name: str,
        context: Dict[str, Any],
        experience_profile: Optional[Any] = None,
        fallback_duration: Optional[float] = None
    ) -> float:
        """
        Predict duration with automatic fallback and logging.
        
        Args:
            resource_id: ID of the resource
            activity_name: Name of the activity
            context: Case context attributes
            experience_profile: Optional ExperienceProfile
            fallback_duration: Specific fallback (if None, uses self.fallback_value)
        
        Returns:
            Predicted duration or fallback value
        """
        self._total_predictions += 1
        
        try:
            duration = self.predict(resource_id, activity_name, context, experience_profile)
            
            # Validate prediction
            if duration is None or duration <= 0:
                raise ValueError(f"Invalid prediction: {duration}")
            
            return duration
            
        except Exception as e:
            self._prediction_failures += 1
            fallback = fallback_duration if fallback_duration is not None else self.fallback_value
            
            logger.warning(
                f"Duration prediction failed for resource={resource_id}, "
                f"activity={activity_name}: {str(e)}. "
                f"Using fallback: {fallback:.2f}h. "
                f"Failure rate: {self._prediction_failures}/{self._total_predictions} "
                f"({100*self._prediction_failures/self._total_predictions:.1f}%)"
            )
            
            return fallback
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction performance statistics."""
        return {
            'total_predictions': self._total_predictions,
            'prediction_failures': self._prediction_failures,
            'failure_rate': self._prediction_failures / max(1, self._total_predictions),
            'success_rate': 1.0 - (self._prediction_failures / max(1, self._total_predictions))
        }
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'DurationPredictor':
        """Load model from disk."""
        pass
