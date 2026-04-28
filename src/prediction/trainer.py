"""Model training orchestration with versioning support."""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .models import PerActivityPredictor, _extract_per_activity_features

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Orchestrates model training with versioning and evaluation.
    
    Supports:
    - Train/test splitting
    - Model versioning with metadata
    - Performance evaluation
    - Future extensibility for online learning
    """
    
    def __init__(
        self,
        test_size: float = 0.3,
        random_state: int = 42,
        model_dir: Optional[Path] = None
    ):
        """
        Initialize model trainer.
        
        Args:
            test_size: Proportion of data for testing (0.0-1.0)
            random_state: Random seed for reproducibility
            model_dir: Directory to save models and metadata
        """
        self.test_size = test_size
        self.random_state = random_state
        self.model_dir = model_dir if model_dir else Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_training_data(
        self,
        timeline: pd.DataFrame,
        experience_store: Any,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Prepare training data from timeline and experience store.
        
        Args:
            timeline: Timeline dataframe with columns:
                - case_id, activity_name, resource_id, start_timestamp, complete_timestamp, duration_hours
                - context columns (if context_attributes specified)
            experience_store: ExperienceStore instance
            resource_metadata: Dictionary mapping resource_id to ResourceMetadata
            context_attributes: List of context attribute names
        
        Returns:
            Tuple of (train_df, test_df, experience_profiles_dict)
        """
        # Rename to standard column names if needed
        df = timeline.copy()
        if 'duration_seconds' in df.columns and 'duration' not in df.columns:
            df['duration'] = df['duration_seconds']  # Use seconds directly
        
        # Ensure timestamps are datetime (handle mixed formats)
        if 'start_timestamp' in df.columns:
            df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], format='mixed', utc=True)
        if 'complete_timestamp' in df.columns:
            df['complete_timestamp'] = pd.to_datetime(df['complete_timestamp'], format='mixed', utc=True)
        
        # Filter valid durations (0 < duration < 24 hours in seconds)
        valid_mask = (df['duration'] > 0) & (df['duration'] < 86400)
        df = df[valid_mask].copy()
        
        logger.info(
            f"Prepared {len(df)} valid training examples "
            f"(filtered {len(timeline) - len(df)} invalid durations)"
        )
        
        # Train/test split (skip if test_size is 0.0 or None - data is pre-split)
        if self.test_size is None or self.test_size == 0.0:
            logger.info("Using pre-split data (test_size=0.0)")
            train_df = df
            test_df = pd.DataFrame()  # Empty test set
        else:
            train_df, test_df = train_test_split(
                df,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=df['activity_name'] if len(df) > 100 else None
            )
            logger.info(
                f"Split: {len(train_df)} training, {len(test_df)} test samples"
            )
        
        # Extract experience profiles
        experience_profiles = {}
        if experience_store is not None:
            profiles = getattr(experience_store, '_profiles', {})
            for key, profile in profiles.items():
                experience_profiles[key] = profile
        
        return train_df, test_df, experience_profiles
    
    def train_per_activity(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        experience_profiles: Dict,
        activity_benchmarks: Optional[Dict[str, float]] = None,
        model_version: Optional[str] = None,
    ) -> Tuple[PerActivityPredictor, Dict[str, Any]]:
        """Train per-activity linear regression models.

        For each activity present in *train_df*, fits a lightweight
        ``LinearRegression`` on 8 features extracted from the matching
        ``ExperienceProfile`` (mean, std, trend_slope, log_count,
        experience_level, benchmark, min, max).

        Args:
            train_df: Training dataframe (must have ``activity_name``,
                ``resource_id``, ``duration`` / ``duration_seconds`` columns).
            test_df: Test dataframe (same schema; may be empty).
            experience_profiles: ``{(resource_id, activity, ctx_key): ExperienceProfile}``
                as extracted by ``prepare_training_data``.
            activity_benchmarks: ``{activity_name: benchmark_seconds}`` from
                ``config/activity_requirements.yaml`` (optional).
            model_version: Version string (auto-generated if *None*).

        Returns:
            ``(predictor, combined_metrics)``
        """
        if model_version is None:
            model_version = f"per_activity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        benchmarks = activity_benchmarks or {}

        predictor = PerActivityPredictor(
            activity_benchmarks=benchmarks,
            fallback_value=3600.0,
            model_version=model_version,
        )

        # Ensure 'duration' column exists
        df = train_df.copy()
        if 'duration' not in df.columns and 'duration_hours' in df.columns:
            df['duration'] = df['duration_hours'] * 3600  # Convert to seconds

        per_activity_metrics: list = []
        activity_std_map: Dict[str, float] = {}

        for activity_name, group in df.groupby('activity_name'):
            X_rows = []
            y_rows = []

            for _, row in group.iterrows():
                resource_id = row['resource_id']
                # Find matching experience profile
                profile = None
                for key, prof in experience_profiles.items():
                    if key[0] == resource_id and key[1] == activity_name:
                        profile = prof
                        break
                if profile is None:
                    continue  # skip rows without a profile

                bm = benchmarks.get(activity_name, 0.0)
                features = _extract_per_activity_features(profile, bm)
                X_rows.append(features)
                y_rows.append(float(row['duration']))

            if len(X_rows) < 3:
                logger.debug(
                    f"Skipping activity '{activity_name}': only {len(X_rows)} "
                    f"samples with profiles (need ≥3)"
                )
                continue

            X = np.array(X_rows)
            y = np.array(y_rows)

            metrics = predictor.train_activity(activity_name, X, y)
            per_activity_metrics.append(metrics)

            # Store mean std_duration for safety margin
            std_vals = [x[1] for x in X_rows]  # index 1 = std_duration
            activity_std_map[activity_name] = float(np.mean(std_vals)) if std_vals else 0.0

            logger.info(
                f"  {activity_name}: {metrics['n_samples']} samples, "
                f"R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.1f}s"
            )

        predictor.activity_std = activity_std_map

        # ---- Evaluate on test set ----
        test_metrics: Dict[str, Any] = {}
        if test_df is not None and len(test_df) > 0:
            tdf = test_df.copy()
            if 'duration' not in tdf.columns and 'duration_hours' in tdf.columns:
                tdf['duration'] = tdf['duration_hours'] * 3600  # Convert to seconds

            y_true_all = []
            y_pred_all = []
            for _, row in tdf.iterrows():
                activity_name = row['activity_name']
                resource_id = row['resource_id']
                if activity_name not in predictor.models:
                    continue
                profile = None
                for key, prof in experience_profiles.items():
                    if key[0] == resource_id and key[1] == activity_name:
                        profile = prof
                        break
                if profile is None:
                    continue
                pred = predictor.predict(resource_id, activity_name, {}, profile)
                y_true_all.append(float(row['duration']))
                y_pred_all.append(pred)

            if y_true_all:
                y_true = np.array(y_true_all)
                y_pred = np.array(y_pred_all)
                test_mse = float(np.mean((y_true - y_pred) ** 2))
                test_metrics = {
                    'test_samples': len(y_true),
                    'test_mse': test_mse,
                    'test_rmse': float(np.sqrt(test_mse)),
                    'test_mae': float(np.mean(np.abs(y_true - y_pred))),
                    'test_r2': float(1 - test_mse / max(np.var(y_true), 1e-9)),
                }
                logger.info(
                    f"Test evaluation — R²: {test_metrics['test_r2']:.4f}, "
                    f"RMSE: {test_metrics['test_rmse']:.4f}h, "
                    f"MAE: {test_metrics['test_mae']:.4f}h "
                    f"({test_metrics['test_samples']} samples)"
                )

        # ---- Combined summary ----
        total_train_samples = sum(m['n_samples'] for m in per_activity_metrics)
        mean_r2 = float(np.mean([m['r2'] for m in per_activity_metrics])) if per_activity_metrics else 0.0
        mean_rmse = float(np.mean([m['rmse'] for m in per_activity_metrics])) if per_activity_metrics else 0.0
        mean_mae = float(np.mean([m['mae'] for m in per_activity_metrics])) if per_activity_metrics else 0.0

        combined = {
            'model_version': model_version,
            'predictor_type': 'per_activity',
            'n_activities': len(per_activity_metrics),
            'n_samples': total_train_samples,
            'mean_r2': mean_r2,
            'mean_rmse': mean_rmse,
            'mean_mae': mean_mae,
            'per_activity': per_activity_metrics,
            **test_metrics,
            'timestamp': datetime.now().isoformat(),
        }

        return predictor, combined

    def save_model_with_metadata(
        self,
        predictor,
        metrics: Dict[str, Any],
        filename_prefix: str = "duration_model"
    ) -> Path:
        """
        Save model and metadata to disk.
        
        Args:
            predictor: Trained PerActivityPredictor predictor
            metrics: Evaluation metrics
            filename_prefix: Prefix for model filename
        
        Returns:
            Path to saved model file
        """
        model_version = predictor.model_version or "unknown"
        
        # Save model
        model_path = self.model_dir / f"{filename_prefix}_{model_version}.pkl"
        predictor.save(str(model_path))
        
        # Save metadata
        metadata_path = self.model_dir / f"{filename_prefix}_{model_version}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Metadata saved: {metadata_path}")
        
        return model_path
    
    def load_latest_model(
        self,
        filename_prefix: str = "duration_model"
    ):
        """Load the most recent model by timestamp.
        
        Automatically detects whether the saved model is a
        ``PerActivityPredictor`` loads accordingly.
        
        Args:
            filename_prefix: Prefix used when saving model
        
        Returns:
            Loaded predictor or None if no models found
        """
        model_files = list(self.model_dir.glob(f"{filename_prefix}_*.pkl"))
        
        if not model_files:
            logger.warning(f"No models found with prefix '{filename_prefix}'")
            return None
        
        # Sort by modification time
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
        
        logger.info(f"Loading latest model: {latest_model}")
        
        # Peek at the pickle to determine predictor type
        import pickle
        with open(latest_model, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and data.get('predictor_type') == 'per_activity':
            return PerActivityPredictor.load(str(latest_model))
        else:
            return TypeError(f"Unknown model format in {latest_model}")
