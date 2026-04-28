"""Experience storage and retrieval for resource performance data."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
import json
from pathlib import Path
import numpy as np
import pandas as pd
import math

# All durations (mean, std, min, max, etc.) are stored in **seconds**.


@dataclass
class ExperienceProfile:
    """
    Performance profile for a resource on a specific activity in a given context.
    
    Attributes:
        resource_id: ID of the resource
        activity_name: Name of the activity
        context: Context attributes (e.g., {'LoanGoal': 'Investment', 'ApplicationType': 'New'})
        mean_duration: Mean service time (seconds)
        std_duration: Standard deviation of service time (seconds)
        median_duration: Median service time (seconds)
        min_duration: Minimum observed duration (seconds)
        max_duration: Maximum observed duration (seconds)
        count: Number of observations (repetitions)
        success_rate: Proportion of successful completions (0-1)
        last_updated: Simulation time of last update
        trend_slope: Trend coefficient (seconds per occurrence, negative = improving)
        experience_level: Current experience level (0-100) for this activity
        capability_floor: Minimum experience level that must never be lost
    """
    resource_id: str
    activity_name: str
    context: Dict[str, Any] = field(default_factory=dict)
    mean_duration: float = 0.0
    std_duration: float = 0.0
    median_duration: float = 0.0
    min_duration: float = 0.0
    max_duration: float = 0.0
    count: int = 0
    success_rate: float = 1.0
    last_updated: float = 0.0
    trend_slope: float = 0.0
    experience_level: float = 0.0
    capability_floor: float = 0.0
    
    # Distribution fitting results (Padella et al. approach)
    best_distribution: str = "lognormal"  # Name of best-fit distribution
    distribution_params: Dict[str, float] = field(default_factory=dict)  # Parameters for the distribution
    fit_quality: float = 0.0  # Goodness-of-fit metric (KS statistic)
    
    def sample_duration(
        self,
        is_mentoring_task: bool = False,
        rng: Optional[np.random.Generator] = None,
        experience_level: Optional[float] = None,
        benchmark_duration: Optional[float] = None,
        beginner_duration: Optional[float] = None,
        mentoring_config: Optional[Dict[str, Any]] = None,
        mentor_experience_level: Optional[float] = None,
        required_capability_level: Optional[float] = None,
    ) -> int:
        """
        Sample a duration from the fitted distribution, scaled by experience level.
        
        If experience_level and benchmark_duration are provided, the sampled
        duration is smoothly interpolated between the raw distribution sample
        and a learning-informed target duration. The target combines the
        profile's current mean duration (updated online by the learning curve)
        with the benchmark floor.

        This keeps duration stochastic for all experience levels and avoids a
        hard discontinuity at an "expert" threshold.

        Args:
            is_mentoring_task: Whether this is a mentoring task (can be used to apply different logic if needed).        
            rng: Random number generator.
            experience_level: Current experience level of the resource (0-100).
            benchmark_duration: Expert-level duration for this activity (seconds).
                The minimum duration ever observed across all resources.
            beginner_duration: Optional duration to use for inexperienced resources (seconds).
                 If not provided, the raw distribution sample will be used even for beginners.
            mentoring_config: Optional dict with mentoring parameters (e.g., duration_multiplier, duration_summand_seconds). 
                
        Returns:
            Duration in integer seconds.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if self.count == 0 or self.mean_duration == 0 or self.experience_level <= 0:
            raw_duration = beginner_duration
        else:
            raw_duration = self._sample_raw_duration(rng)
        
        # --- Apply experience-level scaling ---
        if experience_level is not None and benchmark_duration is not None and benchmark_duration > 0:
            expert_threshold = 100.0
            progress = max(0.0, min(experience_level / expert_threshold, 1.0))

            # Learning-informed target duration:
            # - profile.mean_duration reflects online learning-curve updates
            # - benchmark_duration is an expert floor
            target_duration = max(float(benchmark_duration), float(self.mean_duration or 0.0))

            # Keep a small residual noise share even for highly experienced resources
            # so durations remain stochastic instead of becoming deterministic.
            noise_weight = max(0.35, 1.0 - progress) # maximum improvement of 65% at expert level
            duration_s = target_duration + (raw_duration - target_duration) * noise_weight

            # Ensure duration doesn't go below benchmark
            duration_s = max(duration_s, benchmark_duration)
        else:
            duration_s = raw_duration
        
        # Bound by observed max (seconds) — prevents extreme outliers
        if self.max_duration > 0:
            duration_s = min(duration_s, self.max_duration)
        
        # FINAL SAFEGUARD: Ensure strictly positive (minimum 1 second)
        duration_s = max(1.0, duration_s)
        
        if is_mentoring_task:
            duration_multiplier = (mentoring_config or {}).get('duration_multiplier', 1.5)
            duration_additive = (mentoring_config or {}).get('duration_summand_seconds', 300)

            if mentor_experience_level is not None:
                mentor_factor = 1.0 - 0.4 * min(mentor_experience_level / 100.0, 1.0)
            else:
                mentor_factor = 0.8

            if (
                required_capability_level is not None
                and required_capability_level > 0
                and experience_level is not None
            ):
                mentee_gap_ratio = max(0.0, min(1.0,
                    (required_capability_level - experience_level) / required_capability_level
                ))
            else:
                mentee_gap_ratio = 0.5

            overhead_factor = 1.0 + mentee_gap_ratio
            return round(duration_s * duration_multiplier * mentor_factor
                         + duration_additive * overhead_factor)
        else:
            return round(duration_s)
    
    def _sample_raw_duration(self, rng: np.random.Generator) -> float:
        """Sample raw duration in seconds from fitted distribution (no experience scaling)."""
        # Use fitted distribution if available (internally seconds)
        if self.best_distribution and self.distribution_params:
            try:
                if self.best_distribution == "lognormal":
                    mu = self.distribution_params['mu']
                    sigma = self.distribution_params['sigma']
                    duration = rng.lognormal(mu, sigma)
                elif self.best_distribution == "gamma":
                    shape = self.distribution_params['shape']
                    scale = self.distribution_params['scale']
                    duration = rng.gamma(shape, scale)
                elif self.best_distribution == "weibull":
                    shape = self.distribution_params['shape']
                    scale = self.distribution_params['scale']
                    duration = scale * rng.weibull(shape)
                elif self.best_distribution == "normal":
                    mu = self.distribution_params['mu']
                    sigma = self.distribution_params['sigma']
                    duration = rng.normal(mu, sigma)
                    duration = max(1.0, duration)  # Ensure positive
                else:
                    return self._fallback_sample_raw(rng)
                
                # Bound by observed min/max (seconds)
                if self.min_duration > 0:
                    duration = max(duration, self.min_duration)
                if self.max_duration > 0:
                    duration = min(duration, self.max_duration) 
                
                duration = max(1.0, duration)
                return duration
            except Exception:
                return self._fallback_sample_raw(rng)
        
        # Fallback to existing logic (lognormal with mean/std)
        return self._fallback_sample_raw(rng)
    
    def _fallback_sample(self, rng: np.random.Generator) -> int:
        """Fallback sampling using lognormal with mean/std (returns seconds)."""
        return round(self._fallback_sample_raw(rng))
    
    def _fallback_sample_raw(self, rng: np.random.Generator) -> float:
        """Fallback sampling using lognormal with mean/std.

        Returns:
            Duration in seconds (float).
        """
        if self.count == 0 or self.mean_duration <= 0:
            return 3600.0  # Default fallback (1 hour in seconds)
        
        elif self.std_duration > 0:
            # Ensure mean is positive before log
            safe_mean = max(1.0, self.mean_duration)
            mu = np.log(safe_mean**2 / np.sqrt(safe_mean**2 + self.std_duration**2))
            mu = max(0.0, mu)
            
            sigma = np.sqrt(np.log(1 + (self.std_duration**2 / safe_mean**2)))
            duration = rng.lognormal(mu, sigma)
            
            if self.min_duration > 0:
                duration = max(duration, self.min_duration)
            if self.max_duration > 0:
                duration = min(duration, self.max_duration)
            
            # Ensure positive (minimum 1 second)
            return max(1.0, duration)
        else:
            # Return mean, but ensure it's positive
            return max(1.0, self.mean_duration)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'resource_id': self.resource_id,
            'activity_name': self.activity_name,
            'context': self.context,
            'mean_duration': self.mean_duration,
            'std_duration': self.std_duration,
            'median_duration': self.median_duration,
            'min_duration': self.min_duration,
            'max_duration': self.max_duration,
            'count': self.count,
            'success_rate': self.success_rate,
            'last_updated': self.last_updated,
            'trend_slope': self.trend_slope,
            'experience_level': self.experience_level,
            'capability_floor': self.capability_floor,
            'best_distribution': self.best_distribution,
            'distribution_params': self.distribution_params,
            'fit_quality': self.fit_quality,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ExperienceProfile':
        """Create from dictionary."""
        # Handle backward compatibility for old profiles without distribution fields
        if 'best_distribution' not in data:
            data['best_distribution'] = 'lognormal'
        if 'distribution_params' not in data:
            data['distribution_params'] = {}
        if 'fit_quality' not in data:
            data['fit_quality'] = 0.0
        if 'capability_floor' not in data:
            data['capability_floor'] = float(data.get('experience_level', 0.0) or 0.0)
        return cls(**data)


class ExperienceStore:
    """
    Storage and retrieval system for resource experience profiles.
    
    Stores profiles indexed by (resource_id, activity_name, context_key).
    """
    
    def __init__(self):
        """Initialize empty experience store."""
        self._profiles: Dict[Tuple[str, str, str], ExperienceProfile] = {}
        self._context_keys = []  # Context attributes to use for indexing
        # Secondary indices for fast lookups
        self._resource_index: Dict[str, list[Tuple[str, str, str]]] = {}  # resource_id -> [keys]
        self._activity_index: Dict[str, list[Tuple[str, str, str]]] = {}  # activity_name -> [keys]
        
        # Performance optimization: Cache context key strings to avoid repeated operations
        self._context_key_cache: Dict[frozenset, str] = {}
    
    def set_context_keys(self, keys: list[str]) -> None:
        """Set which context attributes to use for experience lookup."""
        self._context_keys = keys
    
    def _make_context_key(self, context: Dict[str, Any]) -> str:
        """Create hashable context key from context dict (CACHED)."""
        if not self._context_keys:
            return ""
        
        # Create frozen set as cache key (order-independent)
        cache_key = frozenset(
            (key, context.get(key, "NONE"))
            for key in self._context_keys
        )
        
        # Check cache first
        if cache_key in self._context_key_cache:
            return self._context_key_cache[cache_key]
        
        # Build key string
        parts = []
        for key in sorted(self._context_keys):
            value = context.get(key, "NONE")
            parts.append(f"{key}={value}")
        result = "|".join(parts)
        
        # Cache it (limit cache size to prevent unbounded growth)
        if len(self._context_key_cache) < 10000:
            self._context_key_cache[cache_key] = result
        
        return result
    
    def add_profile(self, profile: ExperienceProfile) -> None:
        """Add or update an experience profile."""
        context_key = self._make_context_key(profile.context)
        key = (profile.resource_id, profile.activity_name, context_key)
        self._profiles[key] = profile
        
        # Update secondary indices
        if profile.resource_id not in self._resource_index:
            self._resource_index[profile.resource_id] = []
        if key not in self._resource_index[profile.resource_id]:
            self._resource_index[profile.resource_id].append(key)
        
        if profile.activity_name not in self._activity_index:
            self._activity_index[profile.activity_name] = []
        if key not in self._activity_index[profile.activity_name]:
            self._activity_index[profile.activity_name].append(key)
    
    def get_profile(
        self,
        resource_id: str,
        activity_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ExperienceProfile]:
        """
        Retrieve experience profile for resource × activity × context.
        
        Falls back to resource × activity if no context match found.
        """
        if context is None:
            context = {}
        
        context_key = self._make_context_key(context)
        key = (resource_id, activity_name, context_key)
        
        # Try exact match first
        if key in self._profiles:
            return self._profiles[key]
        
        # Fall back to resource × activity (empty context)
        fallback_key = (resource_id, activity_name, "")
        if fallback_key in self._profiles:
            return self._profiles[fallback_key]
        
        return None
    
    def get_all_profiles_for_resource(self, resource_id: str) -> list[ExperienceProfile]:
        """Get all experience profiles for a resource (O(1) lookup via index)."""
        keys = self._resource_index.get(resource_id, [])
        return [self._profiles[key] for key in keys if key in self._profiles]
    
    def get_all_profiles_for_activity(self, activity_name: str) -> list[ExperienceProfile]:
        """Get all experience profiles for an activity (O(1) lookup via index)."""
        keys = self._activity_index.get(activity_name, [])
        return [self._profiles[key] for key in keys if key in self._profiles]
    
    def get_all_resource_ids(self) -> set[str]:
        """Get all unique resource IDs in the store."""
        return set(p.resource_id for p in self._profiles.values())
    
    def get_all_activity_names(self) -> set[str]:
        """Get all unique activity names in the store."""
        return set(p.activity_name for p in self._profiles.values())
    
    def get_experience_count(
        self,
        resource_id: str,
        activity_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> int:
        """Get the count of observations for a resource-activity-context combination.
        
        Args:
            resource_id: Resource identifier
            activity_name: Activity name
            context: Optional context attributes
            
        Returns:
            Count of observations, or 0 if profile not found
        """
        profile = self.get_profile(resource_id, activity_name, context)
        return profile.count if profile else 0
    
    def get_duration(
        self,
        resource_id: str,
        activity_name: str,
        type: str = "mean",
        context: Optional[Dict[str, Any]] = None,
        is_mentoring: bool = False,
        mentor_experience_level: Optional[float] = None,
        mentee_experience_level: Optional[float] = None,
        required_capability_level: Optional[float] = None,
        mentoring_config: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Get duration estimate in seconds, with fallbacks.

        When ``is_mentoring=True`` and ``mentoring_config`` is provided, applies the
        same mentor-factor / mentee-gap-factor logic as ``sample_duration`` but
        deterministically (no stochastic sampling).
        """
        profile = self.get_profile(resource_id, activity_name, context)
        if profile and profile.count > 0:
            if type == "mean":
                base_duration = round(profile.mean_duration)
            elif type == "max":
                base_duration = round(profile.max_duration)
            elif type == "mean_plus_safety_margin":
                difference_to_max_duration = max(0, profile.max_duration - profile.mean_duration)
                base_duration = round(profile.mean_duration + 0.01 * difference_to_max_duration)
            else:
                raise ValueError(f"Unknown duration type: {type}")
        else:
            # Fallback: average across all resources for this activity
            activity_profiles = self.get_all_profiles_for_activity(activity_name)
            if activity_profiles:
                durations = [p.mean_duration for p in activity_profiles if p.count > 0]
                base_duration = round(float(np.mean(durations))) if durations else 3600
            else:
                base_duration = 3600

        if is_mentoring and mentoring_config:
            duration_multiplier = mentoring_config.get('duration_multiplier', 1.5)
            duration_additive = mentoring_config.get('duration_summand_seconds', 300)

            mentor_factor = (
                1.0 - 0.4 * min(mentor_experience_level / 100.0, 1.0)
                if mentor_experience_level is not None else 0.8
            )
            if required_capability_level and required_capability_level > 0 and mentee_experience_level is not None:
                gap = max(0.0, min(1.0,
                    (required_capability_level - mentee_experience_level) / required_capability_level
                ))
            else:
                gap = 0.5
            return round(base_duration * duration_multiplier * mentor_factor
                         + duration_additive * (1.0 + gap))

        return base_duration
    
    def sample_duration(
        self,
        resource_id: str,
        activity_name: str,
        is_mentoring_task: bool = False,
        context: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
        experience_level: Optional[float] = None,
        benchmark_duration: Optional[float] = None,
        beginner_duration: Optional[float] = None,
        mentoring_config: Optional[Dict[str, Any]] = None,
        mentor_experience_level: Optional[float] = None,
        required_capability_level: Optional[float] = None,
    ) -> int:
        """Sample a duration from experience profile, scaled by experience level.

        Args:
            resource_id: Resource ID.
            activity_name: Activity name.
            is_mentoring_task: Whether this is a mentoring task (can be used to apply different logic if needed).
            context: Case context attributes.
            rng: Random number generator.
            experience_level: Current experience level of the resource (0-100).
            benchmark_duration: Expert-level duration for this activity (seconds).
            beginner_duration: Duration to use for inexperienced resources (seconds).
            mentoring_config: Optional dict with mentoring parameters (e.g., duration_multiplier, duration_summand_seconds).
        Returns:
            Duration in integer seconds.
        """
        profile = self.get_profile(resource_id, activity_name, context)
        if profile:
            return profile.sample_duration(
                is_mentoring_task=is_mentoring_task,
                rng=rng,
                experience_level=experience_level,
                benchmark_duration=benchmark_duration,
                beginner_duration=beginner_duration,
                mentoring_config=mentoring_config,
                mentor_experience_level=mentor_experience_level,
                required_capability_level=required_capability_level,
            )
        
        # Fallback to mean duration
        return self.get_duration(resource_id, activity_name, context=context)

    def get_capability_level(
        self,
        resource_id: str,
        activity_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Get the capability/experience level for a resource on an activity.
        
        Args:
            resource_id: ID of the resource
            activity_name: Name of the activity
            context: Optional context attributes
            
        Returns:
            Experience level (0-100), or 0.0 if no profile found
        """
        profile = self.get_profile(resource_id, activity_name, context or {})
        if not profile:
            return 0.0
        return float(max(profile.experience_level, profile.capability_floor))
    
    def is_capable(
        self,
        resource_id: str,
        activity_name: str,
        required_level: float,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if a resource is capable of performing an activity at required level.
        
        Args:
            resource_id: ID of the resource
            activity_name: Name of the activity
            required_level: Minimum required experience level (0-100)
            context: Optional context attributes
            
        Returns:
            True if resource's capability level >= required_level, False otherwise
        """
        capability_level = self.get_capability_level(resource_id, activity_name, context)
        return capability_level >= float(required_level)
    
    def get_resource_capabilities_dict(self, resource_id: str) -> Dict[str, float]:
        """Get a dict of all activities and their experience levels for a resource.
        
        Used to enumerate capabilities.
        
        Args:
            resource_id: ID of the resource
            
        Returns:
            Dict mapping activity_name -> experience_level for all profiles of the resource
        """
        profiles = self.get_all_profiles_for_resource(resource_id)
        # Deduplicate by activity: keep only the first occurrence per activity
        result = {}
        for profile in profiles:
            if profile.activity_name not in result:
                result[profile.activity_name] = float(max(profile.experience_level, profile.capability_floor))
        return result

    def save(self, path):
        """
        Save experience store to JSON, converting dataclasses and sanitizing values
        (NaN, numpy types, pandas NA, inf) to JSON-compatible values.
        """
        def _sanitize_value(v):
            # dict
            if isinstance(v, dict):
                return {str(k): _sanitize_value(val) for k, val in v.items()}
            # list/tuple/set
            if isinstance(v, (list, tuple, set)):
                return [_sanitize_value(x) for x in v]
            # numpy scalar -> python native
            if isinstance(v, np.generic):
                return v.item()
            # pandas / numpy NA -> None
            try:
                if pd.isna(v):
                    return None
            except Exception:
                pass
            # floats: handle NaN / inf
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    return None
                return v
            # ints, bools, strings
            if isinstance(v, (int, bool, str)):
                return v
            # objects with to_dict
            if hasattr(v, "to_dict") and callable(v.to_dict):
                try:
                    return _sanitize_value(v.to_dict())
                except Exception:
                    pass
            # dataclass / objects
            if hasattr(v, "__dict__"):
                try:
                    return _sanitize_value(vars(v))
                except Exception:
                    pass
            # fallback: stringify
            return str(v)

        store_dict = {}
        profiles = getattr(self, "_profiles", None) or getattr(self, "profiles", {})

        for k, profile in profiles.items():
            # Prefer explicit to_dict conversion for ExperienceProfile
            try:
                if hasattr(profile, "to_dict") and callable(profile.to_dict):
                    prof_obj = profile.to_dict()
                else:
                    prof_obj = vars(profile)
            except Exception:
                prof_obj = str(profile)

            store_dict[str(k)] = _sanitize_value(prof_obj)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(store_dict, fh, ensure_ascii=False, indent=2)
            
    @classmethod
    def load(cls, filepath: Path) -> None:
        """
        Load an ExperienceStore from a JSON file and return an instance.
        The saved JSON maps string keys -> profile dicts (produced by save()).
        """
        inst = cls()
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Experience store file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        for key_str, prof_obj in data.items():
            # prof_obj should be a dict produced by ExperienceProfile.to_dict()
            if not isinstance(prof_obj, dict):
                continue
            # Ensure numeric fields are converted and Nones handled
            try:
                ep = ExperienceProfile.from_dict({
                    'resource_id': str(prof_obj.get('resource_id', prof_obj.get('resource_id', ''))),
                    'activity_name': str(prof_obj.get('activity_name', prof_obj.get('activity_name', ''))),
                    'context': prof_obj.get('context', {}) or {},
                    'mean_duration': float(prof_obj.get('mean_duration') or 0.0),
                    'std_duration': float(prof_obj.get('std_duration') or 0.0),
                    'median_duration': float(prof_obj.get('median_duration') or 0.0),
                    'min_duration': float(prof_obj.get('min_duration') or 0.0),
                    'max_duration': float(prof_obj.get('max_duration') or 0.0),
                    'count': int(prof_obj.get('count') or 0),
                    'success_rate': float(prof_obj.get('success_rate') or 1.0),
                    'last_updated': float(prof_obj.get('last_updated') or 0.0),
                    'trend_slope': float(prof_obj.get('trend_slope') or 0.0),
                    'experience_level': float(prof_obj.get('experience_level') or 0.0)
                })
            except Exception:
                # fallback: build ExperienceProfile conservatively
                ep = ExperienceProfile(
                    resource_id=str(prof_obj.get('resource_id', '')),
                    activity_name=str(prof_obj.get('activity_name', '')),
                    context=prof_obj.get('context', {}) or {},
                    mean_duration=float(prof_obj.get('mean_duration') or 0.0),
                    std_duration=float(prof_obj.get('std_duration') or 0.0),
                    median_duration=float(prof_obj.get('median_duration') or 0.0),
                    min_duration=float(prof_obj.get('min_duration') or 0.0),
                    max_duration=float(prof_obj.get('max_duration') or 0.0),
                    count=int(prof_obj.get('count') or 0),
                    success_rate=float(prof_obj.get('success_rate') or 1.0),
                    last_updated=float(prof_obj.get('last_updated') or 0.0),
                    trend_slope=float(prof_obj.get('trend_slope') or 0.0),
                    experience_level=float(prof_obj.get('experience_level') or 0.0)
                )
            inst.add_profile(ep)

        return inst
    
    def has_experience(self, resource_id: str, activity_name: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if a resource has experience for a given activity.
        
        Args:
            resource_id: ID of the resource
            activity_name: Name of the activity
            context: Optional context attributes
            
        Returns:
            True if profile exists, False otherwise
        """
        profile = self.get_profile(resource_id, activity_name, context or {})
        return profile is not None

    def grant_capability(
        self,
        resource_id: str,
        activity_name: str,
        required_level: float,
        context: Optional[Dict[str, Any]] = None,
        simulation_time: float = 0.0,
    ) -> ExperienceProfile:
        """Ensure a capability profile exists and floor its experience level.

        This is used by bootstrap logic to persist newly granted capabilities in the
        canonical experience store so later updates cannot drop below the granted
        threshold.
        """
        normalized_level = max(0.0, min(100.0, float(required_level)))
        profile = self.get_profile(resource_id, activity_name, context or {})

        if profile is None:
            profile = ExperienceProfile(
                resource_id=resource_id,
                activity_name=activity_name,
                context=context or {},
                mean_duration=0.0,
                std_duration=0.0,
                median_duration=0.0,
                min_duration=0.0,
                max_duration=0.0,
                count=0,
                success_rate=1.0,
                last_updated=float(simulation_time),
                trend_slope=0.0,
                experience_level=normalized_level,
                capability_floor=normalized_level,
            )
            self.add_profile(profile)
            return profile

        if float(profile.capability_floor) < normalized_level:
            profile.capability_floor = normalized_level
        if float(profile.experience_level) < normalized_level:
            profile.experience_level = normalized_level
        profile.last_updated = max(float(profile.last_updated or 0.0), float(simulation_time))
        self.add_profile(profile)
        return profile
    
    def __len__(self) -> int:
        """Number of profiles in store."""
        return len(self._profiles)
    
    def __repr__(self) -> str:
        return f"ExperienceStore(profiles={len(self._profiles)}, context_keys={self._context_keys})"
