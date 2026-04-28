"""Concrete duration prediction models."""

from typing import Dict, Any, Optional, List
import pickle
from pathlib import Path
import logging
import numpy as np
from sklearn.linear_model import LinearRegression

from .base import DurationPredictor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature names for the per-activity regression (8 features)
# ---------------------------------------------------------------------------
PER_ACTIVITY_FEATURE_NAMES: List[str] = [
    "mean_duration",
    "std_duration",
    "trend_slope",
    "log_count",          # log1p(count)
    "experience_level",
    "benchmark_duration",
    "min_duration",
    "max_duration",
]


def _extract_per_activity_features(
    experience_profile: Any,
    benchmark_duration: float = 0.0,
) -> np.ndarray:
    """Extract the 8-feature vector from an ExperienceProfile.

    Args:
        experience_profile: ExperienceProfile instance (must not be None).
        benchmark_duration: Expert-level benchmark for this activity (seconds).

    Returns:
        numpy array of shape (8,).
    """
    return np.array([
        experience_profile.mean_duration or 0.0,
        experience_profile.std_duration or 0.0,
        experience_profile.trend_slope or 0.0,
        np.log1p(experience_profile.count) if experience_profile.count else 0.0,
        experience_profile.experience_level or 0.0,
        benchmark_duration,
        experience_profile.min_duration or 0.0,
        experience_profile.max_duration or 0.0,
    ], dtype=np.float64)


class PerActivityPredictor(DurationPredictor):
    """Per-activity linear-regression duration predictor.

    Trains one lightweight ``LinearRegression`` per activity using 8
    numeric features drawn directly from the ``ExperienceProfile``:

        mean_duration, std_duration, trend_slope, log1p(count),
        experience_level, benchmark_duration, min_duration, max_duration

    No one-hot encoding of activities, contexts, or roles — the model is
    already activity-specific.  This mirrors the information the simulation
    engine's ``sample_duration`` uses and keeps the solver's duration
    estimate well-aligned with execution.

    At prediction time the caller can request a **safety margin** via
    ``predict_with_safety()``:

        predicted + safety_multiplier × std_duration

    This replaces the previous ``type="max"`` worst-case estimate with a
    statistically grounded, configurable upper bound.
    """

    def __init__(
        self,
        models: Optional[Dict[str, LinearRegression]] = None,
        activity_std: Optional[Dict[str, float]] = None,
        activity_benchmarks: Optional[Dict[str, float]] = None,
        fallback_value: float = 3600.0,
        model_version: Optional[str] = None,
    ):
        super().__init__(fallback_value=fallback_value)
        # activity_name -> trained LinearRegression
        self.models: Dict[str, LinearRegression] = models or {}
        # activity_name -> mean std_duration across training profiles
        self.activity_std: Dict[str, float] = activity_std or {}
        # activity_name -> expert benchmark duration (seconds)
        self.activity_benchmarks: Dict[str, float] = activity_benchmarks or {}
        self.model_version = model_version
        self.is_trained = bool(self.models)

    # ------------------------------------------------------------------ #
    # Core prediction
    # ------------------------------------------------------------------ #

    def predict(
        self,
        resource_id: str,
        activity_name: str,
        context: Dict[str, Any],
        experience_profile: Optional[Any] = None,
    ) -> float:
        """Predict duration in **seconds** for a single resource-activity pair.

        Falls back to ``experience_profile.mean_duration`` when the
        activity has no trained sub-model, and to ``self.fallback_value``
        when no profile is available at all.
        """
        if experience_profile is None:
            logger.debug(
                f"No experience profile for {resource_id}/{activity_name} — "
                f"using fallback {self.fallback_value:.0f}s"
            )
            return self.fallback_value

        if activity_name not in self.models:
            # No sub-model for this activity → mean is the best we have
            return float(experience_profile.mean_duration or self.fallback_value)

        benchmark = self.activity_benchmarks.get(activity_name, 0.0)
        features = _extract_per_activity_features(experience_profile, benchmark)
        prediction = float(self.models[activity_name].predict(features.reshape(1, -1))[0])

        # Clamp to sane range
        if prediction <= 0:
            prediction = float(experience_profile.mean_duration or self.fallback_value)
        return prediction

    def predict_with_safety(
        self,
        resource_id: str,
        activity_name: str,
        context: Dict[str, Any],
        experience_profile: Optional[Any] = None,
        safety_multiplier: float = 0.5,
    ) -> float:
        """Predict duration with a configurable safety margin.

        Returns ``predicted + safety_multiplier × std_duration`` (seconds).
        """
        predicted = self.predict(resource_id, activity_name, context, experience_profile)

        # std from the specific profile (preferred) or activity average
        if experience_profile and experience_profile.std_duration:
            std = float(experience_profile.std_duration)
        else:
            std = self.activity_std.get(activity_name, 0.0)

        return predicted + safety_multiplier * std

    # ------------------------------------------------------------------ #
    # Training (single activity)
    # ------------------------------------------------------------------ #

    def train_activity(
        self,
        activity_name: str,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Train a sub-model for one activity.

        Args:
            activity_name: Activity this model predicts for.
            X: Feature matrix (n_samples, 8).
            y: Target durations in seconds (n_samples,).

        Returns:
            Metrics dict for this activity.
        """
        model = LinearRegression()
        model.fit(X, y)
        self.models[activity_name] = model
        self.is_trained = True

        y_pred = model.predict(X)
        mse = float(np.mean((y - y_pred) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y - y_pred)))
        r2 = float(model.score(X, y))

        return {
            "activity": activity_name,
            "n_samples": int(X.shape[0]),
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "models": self.models,
            "activity_std": self.activity_std,
            "activity_benchmarks": self.activity_benchmarks,
            "fallback_value": self.fallback_value,
            "model_version": self.model_version,
            "is_trained": self.is_trained,
            "predictor_type": "per_activity",
        }
        with open(path_obj, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"PerActivityPredictor saved to {path} (version: {self.model_version})")

    @classmethod
    def load(cls, path: str) -> "PerActivityPredictor":
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path_obj, "rb") as f:
            data = pickle.load(f)
        predictor = cls(
            models=data["models"],
            activity_std=data.get("activity_std", {}),
            activity_benchmarks=data.get("activity_benchmarks", {}),
            fallback_value=data.get("fallback_value", 3600.0),
            model_version=data.get("model_version"),
        )
        predictor.is_trained = data.get("is_trained", True)
        logger.info(f"PerActivityPredictor loaded from {path} (version: {predictor.model_version})")
        return predictor

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def get_feature_importance(self, activity_name: str) -> Dict[str, float]:
        """Feature importance (coefficients) for one activity."""
        if activity_name not in self.models:
            return {}
        coeffs = self.models[activity_name].coef_
        return dict(zip(PER_ACTIVITY_FEATURE_NAMES, coeffs))

    def get_all_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Feature importance for all activities."""
        return {act: self.get_feature_importance(act) for act in self.models}
