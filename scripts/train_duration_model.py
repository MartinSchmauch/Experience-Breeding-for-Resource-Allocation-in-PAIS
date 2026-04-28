"""
Train per-activity duration prediction models from historical event logs.

Each activity gets its own lightweight LinearRegression trained on 8 features
derived from ExperienceProfiles:

    mean_duration, std_duration, trend_slope, log1p(count),
    experience_level, benchmark_duration, min_duration, max_duration

Uses the same temporal split as ``initialize_simulation.py`` to prevent
data leakage.

Usage:
    python scripts/train_duration_model.py [--config CONFIG_PATH]
"""

import sys
from pathlib import Path
import logging
import argparse
import yaml
import json
import pandas as pd
import warnings

warnings.filterwarnings("ignore", message=".*rustxes.*", category=UserWarning)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.prediction import ModelTrainer
from src.experience.store import ExperienceStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_timeline_split(split_info_path: Path, timeline_df: pd.DataFrame):
    """Split timeline using the same temporal boundary as initialize_simulation."""
    if not split_info_path.exists():
        raise FileNotFoundError(
            f"Timeline split info not found: {split_info_path}\n"
            f"Please run: python scripts/initialize_simulation.py"
        )

    with open(split_info_path, 'r') as f:
        split_info = json.load(f)

    split_date = pd.to_datetime(split_info['split_date'], utc=True)
    timeline_df['start_timestamp'] = pd.to_datetime(
        timeline_df['start_timestamp'], format='mixed', utc=True
    )

    training = timeline_df[timeline_df['start_timestamp'] < split_date].copy()
    testing = timeline_df[timeline_df['start_timestamp'] >= split_date].copy()

    logger.info(f"Temporal split (matches experience store):")
    logger.info(f"  Split date : {split_date.date()}")
    logger.info(f"  Training   : {len(training):,} events ({len(training)/len(timeline_df)*100:.1f}%)")
    logger.info(f"  Testing    : {len(testing):,} events ({len(testing)/len(timeline_df)*100:.1f}%)")

    return training, testing, split_info


def load_activity_benchmarks(config_path: Path) -> dict:
    """Load activity benchmarks from activity_requirements.yaml."""
    if not config_path.exists():
        logger.warning(f"activity_requirements.yaml not found at {config_path} — no benchmarks used")
        return {}
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f) or {}
    benchmarks = data.get('activity_benchmarks', {})
    if benchmarks:
        logger.info(f"Loaded {len(benchmarks)} activity benchmarks")
    return {str(k): float(v) for k, v in benchmarks.items() if v is not None}


def main():
    parser = argparse.ArgumentParser(description='Train per-activity duration prediction models')
    parser.add_argument('--config', type=str, default='config/simulation_config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()

    config_path = project_root / args.config
    config = load_config(config_path)

    pred_config = config.get('duration_prediction', {})

    # ---- Paths ----
    timeline_path = project_root / config.get('case_arrival', {}).get('timeline_path', 'data/timeline.csv')
    split_info_path = project_root / config.get('case_arrival', {}).get('split_info_path', 'data/timeline_split_info.json')
    experience_store_path = project_root / config.get('experience', {}).get('experience_store_path', 'data/experience_store.json')
    activity_req_path = project_root / "config" / "activity_requirements.yaml"
    model_dir = project_root / pred_config.get('model_dir', 'models')
    random_state = pred_config.get('random_state', 42)

    logger.info("=" * 80)
    logger.info("Per-Activity Duration Model Training")
    logger.info("=" * 80)
    logger.info(f"Timeline          : {timeline_path}")
    logger.info(f"Split info        : {split_info_path}")
    logger.info(f"Experience store  : {experience_store_path}")
    logger.info(f"Activity benchmarks: {activity_req_path}")
    logger.info(f"Model directory   : {model_dir}")
    logger.info("=" * 80)

    # ---- Load data ----
    if not timeline_path.exists():
        logger.error(f"Timeline not found: {timeline_path}  — run initialize_simulation.py first")
        return

    timeline = pd.read_csv(timeline_path)
    logger.info(f"Loaded {len(timeline):,} timeline records")

    training_timeline, testing_timeline, split_info = load_timeline_split(split_info_path, timeline)

    experience_store = ExperienceStore.load(experience_store_path)
    logger.info(f"Loaded {len(experience_store)} experience profiles")

    activity_benchmarks = load_activity_benchmarks(activity_req_path)

    # ---- Prepare data ----
    trainer = ModelTrainer(test_size=0.0, random_state=random_state, model_dir=model_dir)

    train_df, _, experience_profiles = trainer.prepare_training_data(
        timeline=training_timeline,
        experience_store=experience_store,
    )
    test_df, _, _ = trainer.prepare_training_data(
        timeline=testing_timeline,
        experience_store=experience_store,
    )

    # ---- Train per-activity models ----
    logger.info("\nTraining per-activity regression models...")
    predictor, metrics = trainer.train_per_activity(
        train_df=train_df,
        test_df=test_df,
        experience_profiles=experience_profiles,
        activity_benchmarks=activity_benchmarks,
    )

    # ---- Results ----
    logger.info("\n" + "=" * 80)
    logger.info("Training Results")
    logger.info("=" * 80)
    logger.info(f"Model version     : {metrics['model_version']}")
    logger.info(f"Activities trained : {metrics['n_activities']}")
    logger.info(f"Total train samples: {metrics['n_samples']}")
    logger.info(f"Mean R² (train)   : {metrics['mean_r2']:.4f}")
    logger.info(f"Mean RMSE (train) : {metrics['mean_rmse']:.1f}s")
    logger.info(f"Mean MAE (train)  : {metrics['mean_mae']:.1f}s")
    if 'test_r2' in metrics:
        logger.info(f"Test R²           : {metrics['test_r2']:.4f}")
        logger.info(f"Test RMSE         : {metrics['test_rmse']:.1f}s")
        logger.info(f"Test MAE          : {metrics['test_mae']:.1f}s")
        logger.info(f"Test samples      : {metrics['test_samples']}")

    # Per-activity breakdown
    logger.info("\nPer-activity breakdown:")
    for m in metrics.get('per_activity', []):
        logger.info(f"  {m['activity']}: n={m['n_samples']}, R²={m['r2']:.4f}, RMSE={m['rmse']:.1f}s")

    # ---- Save ----
    model_path = trainer.save_model_with_metadata(
        predictor=predictor,
        metrics=metrics,
        filename_prefix="duration_model",
    )

    logger.info("\n" + "=" * 80)
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Training data : {len(train_df):,} samples (before {split_info['split_date']})")
    logger.info(f"Test data     : {len(test_df):,} samples (from {split_info['split_date']})")
    logger.info(f"No data leakage — temporal split matches experience store initialization")
    logger.info(f"Enable in config: duration_prediction.enabled = true")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
