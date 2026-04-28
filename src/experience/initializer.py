"""Initialize experience profiles from historical process logs."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import logging
from scipy import stats
from scipy.stats import lognorm, gamma, weibull_min, norm

from .store import ExperienceStore, ExperienceProfile
from .learning_curves import LearningCurveParameters, create_learning_curve

logger = logging.getLogger(__name__)


class ExperienceInitializer:
    """
    Build initial experience profiles from historical event logs.
    
    Reads service times per event (with resource, activity, context) and computes
    baseline performance statistics for each resource × activity × context combination.
    """
    
    def __init__(
        self,
        context_attributes: Optional[List[str]] = None,
        capability_mapping: Optional[List[dict]] = None,
        learning_model: str = 'richards',
        breeding_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the experience builder.
        
        Args:
            context_attributes: List of case attributes to use as context
                               (e.g., ['case:LoanGoal', 'case:ApplicationType'])
            capability_mapping: Optional list of dicts mapping count thresholds to experience levels.
                               Used as a fallback override if specified; the learning curve is primary.
            learning_model: Learning model to use for initial capability ('richards')
            breeding_params: Parameters for the learning curve model
        """
        self.context_attributes = context_attributes or []
        self.capability_mapping = capability_mapping  # Keep only as optional override
        
        # Initialize learning curve for startup experience estimation
        self.learning_model = learning_model
        self.breeding_params = breeding_params or {
            'lower_asymptote': 0.0,
            'upper_asymptote': 95.0,
            'growth_rate': 0.03,
            'shape_param_Q': 2.5,
            'shape_param_M': 0.8,
        }
        
        # Create the learning curve for initial capability computation
        params = LearningCurveParameters(
            A_i=self.breeding_params.get('lower_asymptote', 0.0),
            K_i=self.breeding_params.get('upper_asymptote', 95.0),
            v_i=self.breeding_params.get('growth_rate', 0.03),
            Q_i=self.breeding_params.get('shape_param_Q', 2.5),
            M_curve=self.breeding_params.get('shape_param_M', 0.8),
        )
        self._learning_curve = create_learning_curve(learning_model, params)
        
        # Activity-specific duration caps (seconds) to filter calendar elapsed time outliers
        # These caps represent realistic P95 maximum processing times excluding waiting periods
        self.activity_duration_caps = {
            'W_Handle leads': (30, 3600),
            'W_Complete application': (45, 7200),
            'W_Shortened completion': (10, 3600),
            'W_Validate application': (30, 7200),
            'W_Call incomplete files': (30, 7200),
            'W_Call after offers': (30, 7200),
            'W_Assess potential fraud': (30, 28800),
            'W_Personal Loan collection': (10, 120),
            '_default': (30, 3600)
        }
    
    def _compute_experience_level(self, count: int) -> float:
        """
        Compute initial experience level for a given observation count.
        
        Uses the configured learning curve by default. Falls back to capability_mapping
        if provided (for backward compatibility / manual override).
        
        Args:
            count: Number of observations
            
        Returns:
            Experience level (0-100)
        """
        # Fallback to capability_mapping if explicitly provided (for manual calibration)
        if self.capability_mapping:
            sorted_mapping = sorted(self.capability_mapping, key=lambda x: x['count'])
            for i in range(len(sorted_mapping) - 1, -1, -1):
                if count >= sorted_mapping[i]['count']:
                    return float(sorted_mapping[i]['level'])
            return float(sorted_mapping[0]['level']) if sorted_mapping else 0.0
        
        # Primary: use the learning curve
        if count < 1:
            return self._learning_curve.compute_experience_level(0)
        return self._learning_curve.compute_experience_level(int(count))
    
    def fit_duration_distribution(
        self, 
        durations: np.ndarray,
        min_samples: int = 10
    ) -> Tuple[str, Dict[str, float], float]:
        """
        Fit multiple distributions and select best based on goodness-of-fit.
        
        Uses Kolmogorov-Smirnov test to compare distributions.
        Follows Padella et al. approach: test multiple distributions per activity-resource pair.
        
        Args:
            durations: Array of observed durations
            min_samples: Minimum samples required for fitting
            
        Returns:
            Tuple of (distribution_name, parameters, ks_statistic)
        """        
        # return "lognormal", {"mu": 8.19, "sigma": 0.3}, 1.0
        if len(durations) < min_samples:
            # Not enough data - return default lognormal with mean/std
            mean_dur = np.mean(durations) if len(durations) > 0 else 3600.0
            std_dur = np.std(durations) if len(durations) > 1 else 0.3 * mean_dur
            
            if std_dur > 0 and mean_dur > 0:
                mu = np.log(mean_dur**2 / np.sqrt(mean_dur**2 + std_dur**2))
                mu = max(0.0, mu)
                sigma = np.sqrt(np.log(1 + (std_dur**2 / mean_dur**2)))
            else:
                mu = np.log(mean_dur) if mean_dur > 0 else 0.0
                mu = max(0.0, mu)
                sigma = 0.3
            
            return "lognormal", {"mu": mu, "sigma": sigma}, 1.0
        
        # Remove outliers (beyond 3 standard deviations)
        mean_dur = np.mean(durations)
        std_dur = np.std(durations)
        
        if std_dur > 0:
            filtered = durations[(durations > mean_dur - 3*std_dur) & (durations < mean_dur + 3*std_dur)]
        else:
            filtered = durations
        
        if len(filtered) < min_samples:
            filtered = durations  # Use all data if filtering removes too much
        
        # Candidate distributions to test
        distributions = {
            'lognormal': lognorm,
            'gamma': gamma,
            'weibull': weibull_min,
            'normal': norm
        }
        
        best_fit = None
        best_ks_stat = float('inf')
        best_params = {}
        
        for name, dist_class in distributions.items():
            try:
                # Fit distribution
                if name == 'lognormal':
                    # Fit lognormal (scipy uses shape, loc, scale parametrization)
                    shape, loc, scale = dist_class.fit(filtered, floc=0)
                    params = {"mu": max(0.0, np.log(scale)), "sigma": shape}
                    # KS test
                    ks_stat, _ = stats.kstest(filtered, lambda x: dist_class.cdf(x, shape, loc, scale))
                    
                elif name == 'gamma':
                    shape, loc, scale = dist_class.fit(filtered, floc=0)
                    params = {"shape": shape, "scale": scale}
                    ks_stat, _ = stats.kstest(filtered, lambda x: dist_class.cdf(x, shape, loc, scale))
                    
                elif name == 'weibull':
                    shape, loc, scale = dist_class.fit(filtered, floc=0)
                    params = {"shape": shape, "scale": scale}
                    ks_stat, _ = stats.kstest(filtered, lambda x: dist_class.cdf(x, shape, loc, scale))
                    
                elif name == 'normal':
                    loc, scale = dist_class.fit(filtered)
                    params = {"mu": loc, "sigma": scale}
                    ks_stat, _ = stats.kstest(filtered, lambda x: dist_class.cdf(x, loc, scale))
                
                # Update best if this is better
                if ks_stat < best_ks_stat:
                    best_fit = name
                    best_ks_stat = ks_stat
                    best_params = params
                    
            except Exception as e:
                logger.debug(f"Failed to fit {name} distribution: {e}")
                continue
        
        # Fallback if all fits failed
        if best_fit is None:
            mean_dur = np.mean(filtered)
            std_dur = np.std(filtered)
            
            if std_dur > 0 and mean_dur > 0:
                mu = np.log(mean_dur**2 / np.sqrt(mean_dur**2 + std_dur**2))
                sigma = np.sqrt(np.log(1 + (std_dur**2 / mean_dur**2)))
            else:
                mu = np.log(mean_dur) if mean_dur > 0 else 0.0
                sigma = 0.3
            
            return "lognormal", {"mu": mu, "sigma": sigma}, 1.0
        
        return best_fit, best_params, best_ks_stat
    
    def build_from_service_times(
        self,
        service_times_df: pd.DataFrame,
        resource_column: str = 'org:resource',
        activity_column: str = 'concept:name',
        duration_column: str = 'duration_seconds',
        start_time_column: str = 'start_time',
        complete_time_column: str = 'complete_time',
        all_resources: Optional[List[str]] = None,
        all_activities: Optional[List[str]] = None,
        default_durations: Dict[str, float] = None,
        default_std: float = 0.3,
        activity_requirements: Optional[Dict[str, float]] = None,
    ) -> ExperienceStore:
        """
        Build experience store from service times dataframe.
        
        Expected columns in service_times_df:
        - resource_column: resource identifier
        - activity_column: activity name
        - duration_column: service duration (seconds)
        - start_time_column: start timestamp (for trend analysis)
        - complete_time_column: completion timestamp
        - context attributes (e.g., 'case:LoanGoal')
        
        Args:
            service_times_df: DataFrame with per-event service times
            resource_column: Column name for resource ID
            activity_column: Column name for activity
            duration_column: Column name for duration
            start_time_column: Column name for start time
            complete_time_column: Column name for completion time
            all_resources: Optional list of all resources (including those not in training)
            all_activities: Optional list of all activities (including those not in training)
            default_durations: Default durations for missing resource-activity pairs
            default_std: Default std duration for missing resource-activity pairs
            activity_requirements: Optional dict of activity_name -> requirement level (0-100) to adjust experience levels for specific activities

        Returns:
            ExperienceStore with initialized profiles
        """
        store = ExperienceStore()
        store.set_context_keys(self.context_attributes)
        
        # Ensure timestamps are datetime
        df = service_times_df.copy()
        if start_time_column in df.columns:
            df[start_time_column] = pd.to_datetime(df[start_time_column])
        if complete_time_column in df.columns:
            df[complete_time_column] = pd.to_datetime(df[complete_time_column])
        
        # Group by resource × activity × context
        groupby_cols = [resource_column, activity_column] + self.context_attributes
        
        # Filter to only include columns that exist
        groupby_cols = [col for col in groupby_cols if col in df.columns]
        grouped = df.groupby(groupby_cols, dropna=False)
        
        for group_key, group_df in grouped:
            # Extract resource, activity, and context
            if isinstance(group_key, tuple):
                resource_id = group_key[0]
                activity_name = group_key[1]
                context = {}
                for i, attr in enumerate(self.context_attributes):
                    if attr in df.columns and i + 2 < len(group_key):
                        context[attr] = group_key[i + 2]
            else:
                resource_id = group_key
                activity_name = group_df[activity_column].iloc[0]
                context = {}
            
            # Compute statistics
            durations = group_df[duration_column].dropna()
            
            if len(durations) == 0:
                continue
            
            # Filter out non-positive durations (<=0)
            original_count_with_nonpos = len(durations)
            durations = durations[durations > 0]
            
            if len(durations) < original_count_with_nonpos:
                logger.info(
                    f"   Filtered out {original_count_with_nonpos - len(durations)} non-positive durations "
                    f"for resource={resource_id}, activity={activity_name}"
                )
            
            # Apply activity-specific duration cap to filter outliers
            lower_duration_cap, upper_duration_cap = self.activity_duration_caps.get(
                activity_name, 
                self.activity_duration_caps['_default']
            )
            
            # Count and log outliers before capping
            original_count = len(durations)
            outliers = durations[(durations < lower_duration_cap) | (durations > upper_duration_cap)]
            if len(outliers) > 0:
                logger.info(
                    f"   Capped {len(outliers)}/{original_count} outliers (>{upper_duration_cap:.0f}s) for "
                    f"resource={resource_id}, activity={activity_name}. "
                    f"Max before cap: {durations.max():.0f}s"
                )
            
            # Apply cap
            durations = durations[(durations >= lower_duration_cap) & (durations <= upper_duration_cap)]
            
            # Skip if no durations remain after capping
            if len(durations) == 0:
                logger.warning(
                    f"   !!! Skipping profile for resource={resource_id}, activity={activity_name} - "
                    f"all {original_count} durations exceeded cap of {upper_duration_cap:.0f}s"
                )
                continue

            # Fit distribution to capped durations
            best_dist, dist_params, ks_stat = self.fit_duration_distribution(
                durations.values,
                min_samples=10
            )
            
            # Compute trend (slope) if enough data points
            trend_slope = 0.0
            if len(durations) >= 3 and start_time_column in group_df.columns:
                sorted_df = group_df.sort_values(start_time_column)
                # Convert timestamps to numeric (days since first)
                time_numeric = (sorted_df[start_time_column] - sorted_df[start_time_column].min()).dt.total_seconds() / (24 * 3600)
                duration_values = sorted_df[duration_column].values
                
                # Simple linear regression
                if len(time_numeric) > 0 and time_numeric.std() > 0:
                    coeffs = np.polyfit(time_numeric, duration_values, 1)
                    trend_slope = float(coeffs[0])  # seconds per day
            
            # Create profile
            observation_count = int(len(durations))
            if observation_count < 3:
                logger.info(
                    f"   Resource={resource_id}, Activity={activity_name} has only {observation_count} observations after filtering, "
                    f"experience level may be unreliable"
                )
                continue
            experience_level = self._compute_experience_level(observation_count)
            
            mean_durr = None
            if experience_level < activity_requirements[activity_name]:
                mean_durr = default_durations.get(activity_name, 3600) if default_durations else 3600
                # heuristic to assign default duration for low-experience profiles to avoid unrealistic very low durations from small samples
                mean_durr = mean_durr * max(0.5, experience_level / activity_requirements[activity_name]) if activity_requirements and activity_name in activity_requirements else mean_durr
                mean_durr = max(mean_durr, durations.mean())
            mean_duration = round(float(durations.mean())) if not mean_durr else round(float(mean_durr))
            
            raw_std = durations.std()
            std_val = 0 if pd.isna(raw_std) else round(float(raw_std))
            
            profile = ExperienceProfile(
                resource_id=str(resource_id),
                activity_name=str(activity_name),
                context=context,
                mean_duration=mean_duration,
                std_duration=std_val,
                median_duration=round(float(durations.median())),
                min_duration=round(float(durations.min())),
                max_duration=round(float(durations.max())),
                count=observation_count,
                success_rate=1.0,  # Assume all completed tasks were successful
                last_updated=0.0,
                trend_slope=trend_slope,
                experience_level=experience_level,
                best_distribution=best_dist,
                distribution_params=dist_params,
                fit_quality=ks_stat
            )
            
            store.add_profile(profile)
        
        # Add default profiles for missing resource-activity combinations
        if all_resources is not None and all_activities is not None:
            print(f"   Adding default profiles for missing resource-activity combinations...")
            
            # Get resources and activities that exist in training data
            existing_combinations = set(
                (p.resource_id, p.activity_name) 
                for p in store._profiles.values()
            )
            
            # Create default profiles for missing combinations
            added_count = 0
            for resource_id in all_resources:
                for activity_name in all_activities:
                    if (str(resource_id), str(activity_name)) not in existing_combinations:
                        # Create default profile with minimal experience
                        default_count = 0
                        default_experience_level = self._compute_experience_level(default_count)
                        default_duration = default_durations.get(activity_name, 3600) if default_durations else 3600
                        # Default distribution params for lognormal
                        if default_std > 0 and default_duration > 0:
                            default_mu = np.log(default_duration**2 / np.sqrt(default_duration**2 + default_std**2))
                            default_mu = max(0.0, default_mu)  # Ensure non-negative
                            default_sigma = np.sqrt(np.log(1 + (default_std**2 / default_duration**2)))
                        else:
                            default_mu = max(0.0, np.log(default_duration)) if default_duration > 0 else 0.0
                            default_sigma = 0.3
                        
                        profile = ExperienceProfile(
                            resource_id=str(resource_id),
                            activity_name=str(activity_name),
                            context={},  # No specific context for default profiles
                            mean_duration=round(default_duration),
                            std_duration=round(default_std),
                            median_duration=round(default_duration),
                            min_duration=round(default_duration * 0.7),
                            max_duration=round(default_duration * 1.5),
                            count=default_count,
                            success_rate=1.0,
                            last_updated=0.0,
                            trend_slope=0.0,
                            experience_level=default_experience_level,
                            best_distribution="lognormal",
                            distribution_params={"mu": default_mu, "sigma": default_sigma},
                            fit_quality=1.0  # No fit performed for defaults
                        )
                        store.add_profile(profile)
                        added_count += 1
            
            print(f"   Added {added_count} default profiles for unseen resource-activity pairs")
        
        return store
    
    def build_from_log_file(
        self,
        log_filepath: Path,
        context_attributes: Optional[List[str]] = None,
        **kwargs
    ) -> ExperienceStore:
        """
        Build experience store from CSV log file.
        
        Args:
            log_filepath: Path to CSV file with service times
            context_attributes: Optional override for context attributes
            **kwargs: Additional arguments passed to build_from_service_times
            
        Returns:
            ExperienceStore with initialized profiles
        """
        if context_attributes is not None:
            self.context_attributes = context_attributes
        
        df = pd.read_csv(log_filepath)
        return self.build_from_service_times(df, **kwargs)
