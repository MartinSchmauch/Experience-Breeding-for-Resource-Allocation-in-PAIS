"""Update experience profiles during simulation based on observed performance."""

from enum import Enum
from typing import Optional, Dict, Any
from .store import ExperienceStore, ExperienceProfile
from .learning_curves import (
    LearningCurveParameters,
    RichardsCurveLearningCurve,
)


class LearningModel(Enum):
    RICHARDS = "richards"  # Generalized logistic (Richards curve)

class ExperienceUpdater:
    """
    Update resource experience profiles based on observed task completions.
    
    Implements various learning models to capture performance improvement over time,
    including the three breeding models from Małachowski & Korytkowski (2016).
    """
    
    def __init__(
        self,
        experience_store: ExperienceStore,
        learning_model: LearningModel = LearningModel.RICHARDS,
        breeding_params: Optional[Dict[str, float]] = None
    ):
        """
        Initialize experience updater.
        
        Args:
            experience_store: ExperienceStore to update
            learning_model: Learning model to use
            breeding_params: Parameters for breeding models (richards)
                           Can be a dict with keys matching LearningCurveParameters fields
        """
        self.store = experience_store
        self.learning_model = learning_model
        
        # Initialize learning curve parameters from breeding_params dict
        if breeding_params is None or not breeding_params:
            # Default parameters
            self.learning_params = LearningCurveParameters(
                learning_rate=0.95,
                A_i=0.0,
                K_i=99.0,
                v_i=0.1,
                Q_i=1.0,
                M_curve=1.0
            )
            self.learning_rate = 0.1  # For exponential smoothing
        else:
            # Convert dict to LearningCurveParameters
            self.learning_params = LearningCurveParameters(
                A_i=breeding_params.get('lower_asymptote', 0.0),
                K_i=breeding_params.get('upper_asymptote', 99.0),
                v_i=breeding_params.get('growth_rate', 0.1),
                Q_i=breeding_params.get('shape_param_Q', 1.0),
                M_curve=breeding_params.get('shape_param_M', 1.0)
            )        
        # Cache for learning curve objects (one per resource-activity pair)
        # Key: (resource_id, activity_name), Value: curve object
        self._curve_cache: Dict[tuple, Any] = {}
    
    def _get_or_create_curve(
        self, 
        resource_id: str, 
        activity_name: str, 
        model_type: LearningModel
    ):
        """Get cached learning curve or create new one for resource-activity pair.
        
        Args:
            resource_id: Resource identifier
            activity_name: Activity name
            model_type: Type of learning curve (RICHARDS)
            
        Returns:
            Cached or newly created learning curve object
        """
        cache_key = (resource_id, activity_name, model_type)
        
        if cache_key not in self._curve_cache:
            if model_type == LearningModel.RICHARDS:
                params = LearningCurveParameters(
                    A_i=self.learning_params.A_i,
                    K_i=self.learning_params.K_i,
                    v_i=self.learning_params.v_i,
                    Q_i=self.learning_params.Q_i,
                    M_curve=self.learning_params.M_curve
                )
                self._curve_cache[cache_key] = RichardsCurveLearningCurve(params)
        
        return self._curve_cache[cache_key]
    
    def update(
        self,
        resource_id: str,
        activity_name: str,
        observed_duration: float,
        context: Optional[Dict[str, Any]] = None,
        simulation_time: float = 0.0,
        success: bool = True
    ) -> None:
        """
        Update experience profile based on observed task completion.
        
        Args:
            resource_id: ID of resource that completed task
            activity_name: Name of activity
            observed_duration: Actual duration observed (hours)
            context: Context attributes for the task
            simulation_time: Current simulation time in hours
            success: Whether task was completed successfully
        """
        if context is None:
            context = {}
        
        # Get existing profile or create new one
        profile = self.store.get_profile(resource_id, activity_name, context)
        
        if profile is None:
            # Create new profile with this observation
            observed_int = round(observed_duration)
            profile = ExperienceProfile(
                resource_id=resource_id,
                activity_name=activity_name,
                context=context,
                mean_duration=observed_int,
                std_duration=0.0,
                median_duration=observed_int,
                min_duration=observed_int,
                max_duration=observed_int,
                count=1,
                success_rate=1.0 if success else 0.0,
                last_updated=simulation_time,
                experience_level=5.0,  # Starting experience level
            )
            self.store.add_profile(profile)
            return
        
        # Update based on learning model
        if self.learning_model == LearningModel.RICHARDS:
            # Richards (Generalized Logistic) curve with empirical mean blending
            # Get cached curve
            curve = self._get_or_create_curve(
                resource_id,
                activity_name,
                LearningModel.RICHARDS
            )
            
            # Update experience level based on new repetition count
            profile.experience_level = max(
                profile.capability_floor,
                curve.compute_experience_level(profile.count + 1),
            )
            
            # Blend observed duration with empirical running mean
            # This ensures the mean incorporates actual performance, not just curve prediction
            # new_mean = (old_mean * old_count + observed) / new_count
            new_count = profile.count + 1
            profile.mean_duration = (
                profile.mean_duration * profile.count + observed_duration
            ) / new_count
        
        # Update other statistics
        profile.min_duration = round(min(profile.min_duration, observed_duration))
        profile.max_duration = round(max(profile.max_duration, observed_duration))
        profile.count += 1
        profile.last_updated = simulation_time
        
        # Round computed durations to integer seconds
        profile.mean_duration = round(profile.mean_duration)
        profile.std_duration = round(profile.std_duration)
        
        # Update success rate
        if success:
            profile.success_rate = (profile.success_rate * (profile.count - 1) + 1) / profile.count
        else:
            profile.success_rate = (profile.success_rate * (profile.count - 1)) / profile.count
        
        # Store updated profile
        self.store.add_profile(profile)
    
    def update_from_task(self, task, simulation_time: float, resource_id: str) -> None:
        """
        Update experience from a completed task object.
        
        Args:
            task: Task object with completion information
            simulation_time: Current simulation time (hours)
            resource_id: ID of the resource that performed the task
        """
        if task.assigned_resource_id is None:
            return
        
        actual_duration_sec = task.get_actual_duration()
        if actual_duration_sec is None:
            return
        
        is_mentoring_task = getattr(task, 'is_mentoring_task', False)
        if is_mentoring_task:
            mentor_id = getattr(task, 'mentor_resource_id', None)
            if mentor_id:
                # Update experience for the mentor as well
                self.update(
                    resource_id=mentor_id,
                    activity_name=task.activity_name,
                    observed_duration=actual_duration_sec,
                    context=task.context,
                    simulation_time=simulation_time,
                    success=(task.status.value == 'completed')
                )

        # Experience store uses seconds as internal unit
        self.update(
            resource_id=resource_id,
            activity_name=task.activity_name,
            observed_duration=actual_duration_sec,
            context=task.context,
            simulation_time=simulation_time,
            success=(task.status.value == 'completed')
        )

        if getattr(task, 'bootstrap_assignment', False):
            required_level = float(getattr(task, 'required_capability_level', 0.0) or 0.0)
            if required_level > 0.0:
                self.store.grant_capability(
                    resource_id=resource_id,
                    activity_name=task.activity_name,
                    required_level=required_level,
                    context=task.context,
                    simulation_time=simulation_time,
                )
