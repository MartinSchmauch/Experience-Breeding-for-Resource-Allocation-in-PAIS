"""
Learning curve models for experience breeding based on Małachowski & Korytkowski (2016).

Implements the Richards (Generalized Logistic) learning model:
S-shaped curve with 5 parameters for precise modeling
"""

import math
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class LearningCurveParameters:
    """
    Parameters for learning curve models.
    
    Attributes:
        learning_rate: Learning rate coefficient (typically 0.8-0.95)
        A_i: Lower asymptote (for Richards curve)
        K_i: Upper asymptote/carrying capacity (for Richards curve)
        v_i: Growth rate (for Richards curve)
        Q_i: Affects near which asymptote maximum growth occurs (for Richards curve)
        M_curve: Parameter affecting growth shape (for Richards curve)
    """
    learning_rate: float = 0.95  # Learning coefficient (higher = slower learning)
        
    # Richards curve parameters
    A_i: float = 1.0   # Lower asymptote (minimum achievable time)
    K_i: float = 100.0  # Upper asymptote (maximum experience level)
    v_i: float = 0.5   # Growth rate
    Q_i: float = 1.0   # Affects curve shape
    M_curve: float = 1.0  # Affects curve shape

class RichardsCurveLearningCurve:
    """
    Richards (Generalized Logistic) learning curve for experience level:
    
    L(n) = A_i + (K_i - A_i) / (1 + Q_i * exp(-v_i * n))^(1/M)
    
    where:
    - L(n) = experience level at repetition n
    - A_i = lower asymptote (minimum experience level)
    - K_i = upper asymptote (maximum experience level)
    - v_i = growth rate
    - Q_i = affects near which asymptote maximum growth occurs
    - M = affects curve shape
    
    This is the most flexible model with S-shaped growth pattern.
    It can model:
    - Initial slow learning (getting started)
    - Rapid improvement phase
    - Plateau as expertise is reached
    """
    
    def __init__(self, params: LearningCurveParameters):
        """Initialize Richards curve learning model."""
        self.params = params
    
    def compute_experience_level(self, n: int) -> float:
        """
        Compute experience level (0-100) based on Richards curve.
        
        Args:
            n: Repetition number
            
        Returns:
            Experience level (0-100)
        """
        if n < 1:
            return self.params.A_i
        
        # Richards curve formula
        A = self.params.A_i
        K = self.params.K_i
        v = self.params.v_i
        Q = self.params.Q_i
        M = self.params.M_curve
        
        # Prevent numerical overflow in exp
        exponent = -v * n
        exponent = max(-50, min(50, exponent))  # Clamp to prevent overflow
        
        try:
            denominator = (1.0 + Q * np.exp(exponent)) ** (1.0 / M)
            level = A + (K - A) / denominator
        except (OverflowError, ZeroDivisionError):
            # If overflow, assume we're at the asymptote
            level = K if n > 10 else A
        
        return max(0.0, min(100.0, level))
    

    def repetitions_to_reach_level(self, current_n: int, target_level: float) -> int:
        """Compute additional repetitions needed to reach a target experience level.

        The Richards curve is analytically invertible for target levels strictly
        between the lower and upper asymptotes. We solve the inverse in closed
        form and then round up to the next integer repetition because the model
        itself is evaluated on integer repetition counts.

        Args:
            current_n: Current repetition count.
            target_level: Desired experience level (0-100).

        Returns:
            Additional repetitions needed. 0 if already reached or if the
            target exceeds the upper asymptote K_i.
        """
        A = self.params.A_i
        K = self.params.K_i
        v = self.params.v_i
        Q = self.params.Q_i
        M = self.params.M_curve

        if target_level >= K:
            return 0  # above asymptote — unreachable

        current_total_reps = max(current_n, 0)
        current_level = self.compute_experience_level(current_total_reps)
        if current_level >= target_level:
            return 0

        first_model_level = self.compute_experience_level(1)
        if current_total_reps < 1 and target_level <= first_model_level:
            return 1 - current_total_reps

        if v <= 0 or Q <= 0 or M <= 0:
            return 0

        # Richards inverse: solve for the repetition index n.
        # L(n) = A + (K - A) / (1 + Q * exp(-v*n))^(1/M)
        # => n = -(1/v) * ln( ( ((K-A)/(L-A))^M - 1 ) / Q )
        fraction = (K - A) / (target_level - A)
        log_argument = (fraction ** M - 1.0) / Q
        if log_argument <= 0 or not math.isfinite(log_argument):
            return 0

        n_target = -(1.0 / v) * math.log(log_argument)
        if not math.isfinite(n_target):
            return 0

        required_total_reps = max(0, int(math.ceil(n_target)))
        return max(0, required_total_reps - current_total_reps)


def create_learning_curve(
    model_type: str,
    params: Optional[LearningCurveParameters] = None
):
    """
    Factory function to create learning curve model.
    
    Args:
        model_type: One of 'richards'
        params: Learning curve parameters (uses defaults if None)
        
    Returns:
        Learning curve model instance
        
    Raises:
        ValueError: If model_type is not recognized
    """
    if params is None:
        params = LearningCurveParameters()
    
    model_type = model_type.lower()
    
    if model_type == 'richards':
        return RichardsCurveLearningCurve(params)
    else:
        raise ValueError(f"Unknown learning curve model: {model_type}. "
                        f"Choose from: richards")
