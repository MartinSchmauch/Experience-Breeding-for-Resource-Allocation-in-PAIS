"""
Initialize simulation components from historical BPI 2017 data
"""
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import json
import random
import pandas as pd
import sys
import pickle
import yaml
import warnings
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# add path
sys.path.append('.')
# Import simulation components
from src.experience import ExperienceInitializer
from src.process.model import ProbabilisticProcessModel, ProcessVariant
from src.process.transition_weights import (
    TransitionModelMetadata,
    count_activity_revisits,
    has_excessive_loops,
    trim_looping_activities
)
from src.io import EventLogReader

DEFAULT_ABSENCE_PARAMS: Dict[str, int] = {
    "vacation_days_per_year": 25,
    "sick_days_per_year": 15,
    "vacation_min_consecutive": 3,
    "vacation_max_consecutive": 10,
}

def load_configuration(config_path: Path) -> Dict[str, Any]:
    """Load and parse simulation configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _cfg_absence_params(config: Dict[str, Any]) -> Dict[str, int]:
    return config.get("working_hours", {}).get("generate_absences", DEFAULT_ABSENCE_PARAMS)


def _cfg_calendar_start_year(config: Dict[str, Any]) -> int:
    return config.get("working_hours", {}).get("start_date", {}).get("year", 2016)


def _cfg_default_schedule(config: Dict[str, Any]) -> Dict[str, List[float]]:
    return config.get("working_hours", {}).get(
        "default_schedule",
        {
            "monday": [9.0, 17.0],
            "tuesday": [9.0, 17.0],
            "wednesday": [9.0, 17.0],
            "thursday": [9.0, 17.0],
            "friday": [9.0, 17.0],
        },
    )


def _user_num(resource_id: str) -> int:
    try:
        return int(resource_id.split("_")[1])
    except (IndexError, ValueError):
        return 0


def load_resource_ids(experience_store_path: Path) -> List[str]:
    """Return sorted unique resource IDs from the experience store JSON."""
    with open(experience_store_path, "r") as f:
        data: Dict[str, Any] = json.load(f)
    ids = {profile["resource_id"] for profile in data.values()}
    return sorted(ids)


def generate_working_hours(
    resource_id: str,
    default_schedule: Dict[str, List[float]],
) -> Dict[str, int]:
    """Generate per-resource working hours with early/late shift variation."""
    base = default_schedule.get("monday", [9.0, 17.0])
    base_start, base_end = int(base[0]), int(base[1])

    num = _user_num(resource_id)
    if num % 10 == 0:
        return {"start": base_start - 1, "end": base_end - 1}  # early shift
    if num % 10 == 1:
        return {"start": base_start + 1, "end": base_end + 1}  # late shift
    return {"start": base_start, "end": base_end}


def generate_absences(
    resource_id: str,
    start_year: int = 2016,
    num_years: int = 1,
    absence_params: Optional[Dict[str, int]] = None,
) -> List[Dict[str, str]]:
    """Generate deterministic vacation and sick-day absences for a resource."""
    params = absence_params or DEFAULT_ABSENCE_PARAMS
    vacation_days_total = params.get("vacation_days_per_year", 25)
    sick_days_target = params.get("sick_days_per_year", 15)
    vac_min = params.get("vacation_min_consecutive", 3)
    vac_max = params.get("vacation_max_consecutive", 10)

    num = _user_num(resource_id)
    rng = random.Random(num)

    absences: List[Dict[str, str]] = []

    for year in range(start_year, start_year + num_years):
        remaining = vacation_days_total
        num_periods = rng.randint(2, 4)

        for i in range(num_periods):
            if remaining <= 0:
                break
            if i == num_periods - 1:
                days = remaining
            else:
                max_this = min(vac_max, remaining - vac_min * (num_periods - i - 1))
                max_this = max(max_this, vac_min)
                days = rng.randint(vac_min, max_this)
            remaining -= days

            start_month = rng.randint(1, 12)
            start_day = rng.randint(1, 28)
            start_date = datetime(year, start_month, start_day)
            weekday = start_date.weekday()
            if weekday != 0:
                start_date += timedelta(days=(7 - weekday))
            end_date = start_date + timedelta(days=days)

            absences.append({
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "absence_type": "vacation",
                "description": f"Annual leave ({days} days)",
            })

        sick_remaining = sick_days_target
        max_episodes = rng.randint(2, 4)
        short_lengths = [1, 2, 2, 3, 3]
        long_lengths = [5, 7]

        for j in range(max_episodes):
            if sick_remaining <= 0:
                break

            episodes_left = max_episodes - j
            if episodes_left == 1:
                days = sick_remaining
            else:
                use_long_episode = sick_remaining >= 5 and rng.random() < 0.35
                candidate_lengths = long_lengths if use_long_episode else short_lengths
                valid_lengths = [d for d in candidate_lengths if d <= sick_remaining]

                if not valid_lengths:
                    valid_lengths = [d for d in (1, 2, 3, 5, 7) if d <= sick_remaining]

                days = rng.choice(valid_lengths)

            sick_remaining -= days

            month = rng.randint(1, 12)
            day = rng.randint(1, 28)
            start_date = datetime(year, month, day)
            end_date = start_date + timedelta(days=days)

            absences.append({
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "absence_type": "sick_leave",
                "description": f"Sick leave ({days} day{'s' if days > 1 else ''})",
            })

    absences.sort(key=lambda a: a["start_date"])
    return absences


def generate_calendar(
    resource_id: str,
    working_hours: Dict[str, int],
    start_year: int = 2016,
    absence_params: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Generate calendar data (schedule + absences) for one resource."""
    schedule = {
        "weekday_hours": {
            str(d): [working_hours["start"], working_hours["end"]]
            for d in range(5)
        },
        "timezone_offset": 0.0,
    }
    absences = generate_absences(
        resource_id,
        start_year=start_year,
        num_years=1,
        absence_params=absence_params,
    )
    return {"resource_id": resource_id, "schedule": schedule, "absences": absences}


def generate_calendars(
    experience_store_path: Path,
    output_path: Path,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate calendars.json from experience store resource IDs and config."""
    if config is None:
        config = {}

    absence_params = _cfg_absence_params(config)
    start_year = _cfg_calendar_start_year(config)
    default_schedule = _cfg_default_schedule(config)

    logger.info("4. Generating resource calendars...")
    resource_ids = load_resource_ids(experience_store_path)
    logger.info("   Found %s resources", len(resource_ids))

    calendars: Dict[str, Any] = {}
    shift_counts = {"early": 0, "late": 0, "normal": 0}

    for resource_id in resource_ids:
        hours = generate_working_hours(resource_id, default_schedule)
        calendars[resource_id] = generate_calendar(
            resource_id,
            hours,
            start_year=start_year,
            absence_params=absence_params,
        )

        base_start = int(default_schedule.get("monday", [9.0, 17.0])[0])
        if hours["start"] < base_start:
            shift_counts["early"] += 1
        elif hours["start"] > base_start:
            shift_counts["late"] += 1
        else:
            shift_counts["normal"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(calendars, f, indent=2)

    total_vac = 0
    total_sick = 0
    for cal in calendars.values():
        for absence in cal["absences"]:
            days = (
                datetime.fromisoformat(absence["end_date"])
                - datetime.fromisoformat(absence["start_date"])
            ).days
            if absence["absence_type"] == "vacation":
                total_vac += days
            else:
                total_sick += days

    n = len(calendars) or 1
    logger.info("   Saved %s (%s resources)", output_path, len(calendars))
    logger.info(
        "   Shift distribution: %s normal, %s early, %s late",
        shift_counts["normal"],
        shift_counts["early"],
        shift_counts["late"],
    )
    logger.info("   Avg vacation days/resource: %.1f", total_vac / n)
    logger.info("   Avg sick days/resource: %.1f", total_sick / n)

    return calendars


def create_timeline_from_xes(
    xes_path: Path,
    context_attributes: List[str],
    duration_assignment_strategy: str = 'longest_resource'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create timeline with durations from XES log.
    
    Processes start/suspend/resume/complete lifecycle transitions to compute
    active service durations (excluding suspension periods).
    
    Args:
        xes_path: Path to XES file
        context_attributes: Case attributes to include
        duration_assignment_strategy: Aggregation strategy for assigning each
            task occurrence to a resource and duration value.
        
    Returns:
        Tuple of (timeline_df, segments_df):
        - timeline_df: Aggregated timeline (one row per task occurrence)
        - segments_df: Detailed work segments (one row per active work period)
    """
    logger.info("1. Reading and preprocessing XES log...")
    reader = EventLogReader()
    timeline_df, segments_df = reader.preprocess_for_simulation(
        xes_path,
        filter_prefix='W_',
        context_attributes=context_attributes,
        compute_durations=True,
        return_segments=True,
        duration_assignment_strategy=duration_assignment_strategy,
    )
    
    logger.info(f"   Loaded {len(timeline_df):,} task occurrences")
    logger.info(f"   Cases: {timeline_df['case_id'].nunique():,}")
    logger.info(f"   Resources: {timeline_df['resource_id'].nunique()}")
    logger.info(f"   Activities: {timeline_df['activity_name'].nunique()}")
    if 'segment_count' in timeline_df.columns:
        multi_segment = (timeline_df['segment_count'] > 1).sum()
        multi_resource = (timeline_df['resource_count'] > 1).sum()
        logger.info(f"   Multi-segment tasks: {multi_segment:,} ({multi_segment/len(timeline_df)*100:.1f}%)")
        logger.info(f"   Multi-resource tasks: {multi_resource:,} ({multi_resource/len(timeline_df)*100:.1f}%)")
    
    return timeline_df, segments_df


def split_timeline_by_date(timeline_df: pd.DataFrame, training_split: float = 0.3):
    """
    Split timeline into training and testing sets based on temporal order.
    
    Finds the date where approximately training_split% of events occurred,
    ensuring temporal validity (no data leakage from future to past).
    
    Args:
        timeline_df: Timeline DataFrame with 'start_timestamp' column
        training_split: Fraction of events to use for training (0.0 to 1.0)
        
    Returns:
        tuple: (training_df, testing_df, split_date)
    """
    df = timeline_df.copy()
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
    df = df.sort_values('start_timestamp')
    
    # Find min and max dates
    min_date = df['start_timestamp'].min()
    max_date = df['start_timestamp'].max()
    total_events = len(df)
    target_training_events = int(total_events * training_split)
    
    logger.info(f"   Timeline date range: {min_date.date()} to {max_date.date()}")
    logger.info(f"   Total events: {total_events:,}")
    logger.info(f"   Target training events (~{training_split*100:.0f}%): {target_training_events:,}")
    
    # Calculate cumulative event counts by date
    df['date'] = df['start_timestamp'].dt.date
    daily_counts = df.groupby('date').size().reset_index(name='count')
    daily_counts['cumulative'] = daily_counts['count'].cumsum()
    
    # Find the date where cumulative count reaches ~training_split
    split_row = daily_counts[daily_counts['cumulative'] >= target_training_events].iloc[0]
    split_date = pd.Timestamp(split_row['date'], tz='UTC')  # Make timezone-aware
    actual_training_events = split_row['cumulative']
    
    logger.info(f"   Split date: {split_date.date()}")
    logger.info(f"   Actual training events: {actual_training_events:,} ({actual_training_events/total_events*100:.1f}%)")
    logger.info(f"   Testing events: {total_events - actual_training_events:,} ({(total_events - actual_training_events)/total_events*100:.1f}%)")
    
    # Split dataframe
    training_df = df[df['start_timestamp'] < split_date].copy()
    testing_df = df[df['start_timestamp'] >= split_date].copy()
    
    return training_df, testing_df, split_date


def filter_low_activity_resources(
    timeline_df: pd.DataFrame,
    min_avg_daily_hours: float = 1.0
) -> List[str]:
    """
    Filter out resources that average less than min_avg_daily_hours per calendar day.
    
    Computes total work seconds per resource across the entire timeline, then divides
    by the total number of calendar days in the timeline period.
    
    Args:
        timeline_df: Timeline DataFrame with 'resource_id', 'duration_seconds', 'start_timestamp'
        min_avg_daily_hours: Minimum average hours per calendar day to keep a resource
        
    Returns:
        List of resource IDs that meet the threshold
    """
    df = timeline_df.copy()
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
    
    # Compute total calendar days in timeline
    min_date = df['start_timestamp'].min()
    max_date = df['start_timestamp'].max()
    total_calendar_days = max((max_date - min_date).days, 1)
    
    # Total hours per resource
    resource_total_seconds = df.groupby('resource_id')['duration_seconds'].sum()
    
    # Average daily hours = total hours / calendar days
    resource_avg_daily = resource_total_seconds / total_calendar_days
    
    # Filter
    min_avg_daily_seconds = min_avg_daily_hours * 3600
    kept = resource_avg_daily[resource_avg_daily >= min_avg_daily_seconds]
    removed = resource_avg_daily[resource_avg_daily < min_avg_daily_seconds]
    
    logger.info(f"   Resource filtering (min {min_avg_daily_hours:.1f} h/day over {total_calendar_days} calendar days):")
    logger.info(f"     Total resources before filter: {len(resource_avg_daily)}")
    logger.info(f"     Resources kept: {len(kept)} (avg {kept.mean():.2f} h/day)")
    logger.info(f"     Resources removed: {len(removed)} (avg {removed.mean():.2f} h/day)")
    
    removed_seconds = resource_total_seconds[removed.index].sum()
    kept_seconds = resource_total_seconds[kept.index].sum()
    logger.info(f"     Total seconds kept: {kept_seconds:,.1f} s | removed: {removed_seconds:,.1f} s")
    
    return kept.index.tolist()


def build_experience_store(
    timeline_df: pd.DataFrame,
    training_split: float,
    split_date: Optional[pd.Timestamp],
    context_attributes: List[str],
    capability_mapping: List[Dict],
    learning_model: str,
    breeding_params: Dict[str, Any],
    default_durations: Dict[str, float],
    default_std: float,
    output_path: Path,
    min_avg_daily_hours: float = 0.0,
    activity_requirements: Optional[Dict[str, float]] = None
) -> Tuple[Any, pd.Timestamp]:
    """
    Build and save experience store from timeline.
    
    Args:
        timeline_df: Timeline with service times
        training_split: Fraction for training (if split_date is None)
        split_date: Explicit split date (overrides training_split)
        context_attributes: Case attributes for profiles
        capability_mapping: Resource capability configuration (optional override)
        learning_model: Learning model for initial capability computation
        breeding_params: Parameters for the learning curve model
        default_durations: Default durations for missing profiles
        default_std: Default standard deviation
        output_path: Path to save experience store
        min_avg_daily_hours: Exclude resources averaging fewer hours/day (0 = no filter)
        activity_requirements: Optional dict of activity_name -> requirement level (0-100) to adjust experience levels for specific activities
        
    Returns:
        Tuple of (experience_store, split_date)
    """
    logger.info("2. Building experience store...")
    logger.info("   Fitting duration distributions per resource-activity pair...")
    
    # Get all unique values for comprehensive profiles
    all_resources = timeline_df['resource_id'].unique().tolist()
    all_activities = timeline_df['activity_name'].unique().tolist()
    logger.info(f"   Total resources (before filter): {len(all_resources)}")
    logger.info(f"   Total activities: {len(all_activities)}")
    logger.debug(f"   All activities: {all_activities}")
    
    # Filter low-activity resources
    if min_avg_daily_hours > 0:
        active_resources = filter_low_activity_resources(timeline_df, min_avg_daily_hours)
        all_resources = [r for r in all_resources if r in active_resources]
        timeline_df = timeline_df[timeline_df['resource_id'].isin(active_resources)].copy()
        logger.info(f"   Resources after filter: {len(all_resources)}")
    else:
        logger.info(f"   Resource filtering disabled (min_avg_daily_hours={min_avg_daily_hours})")
    # Split timeline
    if split_date is None:
        training_timeline, testing_timeline, split_date = split_timeline_by_date(
            timeline_df, training_split=training_split
        )
    else:
        split_date = pd.to_datetime(split_date, utc=True)
        training_timeline = timeline_df[timeline_df['start_timestamp'] < split_date].copy()
        testing_timeline = timeline_df[timeline_df['start_timestamp'] >= split_date].copy()
        logger.info(f"   Using configured split date: {split_date.date()}")
        logger.info(f"   Training events: {len(training_timeline):,}")
        logger.info(f"   Testing events: {len(testing_timeline):,}")
    
    # Initialize experience builder with learning curve
    initializer = ExperienceInitializer(
        context_attributes=context_attributes,
        capability_mapping=capability_mapping if capability_mapping else None,
        learning_model=learning_model,
        breeding_params=breeding_params
    )
    
    # Build experience profiles from training data    
    experience_store = initializer.build_from_service_times(
        training_timeline,
        resource_column='resource_id',
        activity_column='activity_name',
        duration_column='duration_seconds',
        start_time_column='start_timestamp',
        complete_time_column='complete_timestamp',
        all_resources=all_resources,
        all_activities=all_activities,
        default_durations=default_durations,
        default_std=default_std,
        activity_requirements=activity_requirements
    )
    
    # Report statistics
    _report_experience_statistics(experience_store, training_timeline)
    
    # Save experience store
    experience_store.save(output_path)
    logger.info(f"   Saved to {output_path}")
    
    return experience_store, split_date


def compute_activity_benchmarks(experience_store: Any) -> Dict[str, float]:
    """
    Compute per-activity benchmark durations (in seconds) from the experience store.
    
    The benchmark for each activity is the minimum min_duration observed
    across all resources that have real experience (count > 0).
    This represents the fastest achievable time for an expert (level >= 95).
    
    Args:
        experience_store: ExperienceStore instance
        
    Returns:
        Dict mapping activity_name -> benchmark_duration_seconds
    """
    from collections import defaultdict
    
    logger.info("   Computing activity benchmarks (expert-level durations)...")
    
    # Collect min_duration per activity across all resources with real data
    activity_mins: Dict[str, float] = defaultdict(lambda: float('inf'))
    
    for profile in experience_store._profiles.values():
        if profile.count > 0 and profile.min_duration >= 0:
            if profile.min_duration < activity_mins[profile.activity_name]:
                activity_mins[profile.activity_name] = profile.min_duration
    # Convert to regular dict, drop any 'inf' (activities with no real data)
    lower_benchmarks = {
        act: dur for act, dur in activity_mins.items()
        if dur < float('inf')
    }

    logger.info(f"   Computed benchmarks for {len(lower_benchmarks)} activities:")
    for act, dur in sorted(lower_benchmarks.items()):
        logger.info(f"     {act}: {dur:.1f}s ({dur/3600:.4f}h)")
        
    return lower_benchmarks


def write_activity_benchmarks(benchmarks: Dict[str, float], config_path: Path, section: str = "activity_benchmarks") -> None:
    """
    Write activity benchmarks (in seconds) to activity_requirements.yaml.
    
    Preserves existing content (activity_requirements, default_requirement)
    and adds/updates the '{section}' section.
    
    Args:
        benchmarks: Dict mapping activity_name -> benchmark_duration_seconds
        config_path: Path to activity_requirements.yaml
        section: Section name for the benchmarks (default: "activity_benchmarks")
    """
    # Load existing config
    existing = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            existing = yaml.safe_load(f) or {}
    
    # Update benchmarks section (values in integer seconds)
    existing[section] = {
        str(k): round(float(v)) for k, v in sorted(benchmarks.items())
    }
    
    # Write back
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(existing, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    logger.info(f"   Saved activity benchmarks to {config_path}")


def _report_experience_statistics(experience_store: Any, training_timeline: pd.DataFrame) -> None:
    """Report experience store statistics via logger."""
    logger.info(f"   Created {len(experience_store)} experience profiles")
    logger.info(f"   Total events in Training Timeline: {len(training_timeline):,}")
    logger.info(f"   Unique Resources: {training_timeline['resource_id'].nunique()}")
    logger.info(f"   Unique Activities: {training_timeline['activity_name'].nunique()}")
    
    # Distribution fitting summary
    logger.info(f"   Distribution Fitting Summary:")
    dist_counts = {}
    profiles_list = list(experience_store._profiles.values())
    
    for profile in profiles_list:
        dist_name = profile.best_distribution
        dist_counts[dist_name] = dist_counts.get(dist_name, 0) + 1
    
    logger.info(f"   Total profiles: {len(experience_store)}")
    for dist_name, count in sorted(dist_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(experience_store) * 100
        logger.info(f"     {dist_name}: {count} profiles ({pct:.1f}%)")
    
    # Show best/worst fit examples
    fitted_profiles = [p for p in profiles_list if p.count >= 10]
    if fitted_profiles:
        best_fit = min(fitted_profiles, key=lambda p: p.fit_quality)
        worst_fit = max(fitted_profiles, key=lambda p: p.fit_quality)
        
        logger.info(f"   Best fit (lowest KS statistic):")
        logger.info(f"     Resource: {best_fit.resource_id}, Activity: {best_fit.activity_name}")
        logger.info(f"     Distribution: {best_fit.best_distribution}, Count: {best_fit.count}")
        logger.info(f"     KS: {best_fit.fit_quality:.4f}, Mean: {best_fit.mean_duration:.1f}s")
        
        logger.info(f"   Worst fit (highest KS statistic):")
        logger.info(f"     Resource: {worst_fit.resource_id}, Activity: {worst_fit.activity_name}")
        logger.info(f"     Distribution: {worst_fit.best_distribution}, Count: {worst_fit.count}")
        logger.info(f"     KS: {worst_fit.fit_quality:.4f}, Mean: {worst_fit.mean_duration:.1f}s")


def _build_transition_models_in_memory(
    xes_path: Path,
    context_attributes: List[str],
    history_mode: str = 'binary',
    variant_filter_config: Optional[Dict[str, Any]] = None
) -> Tuple[Dict, pd.DataFrame, Any]:
    """Build transition models from XES log and return in memory.
    
    Args:
        xes_path: Path to XES file
        context_attributes: Case attributes for context
        history_mode: History tracking mode ('binary' or 'count')
        variant_filter_config: Controls frequency and loop-aware filtering of process variants
        
    Returns:
        Tuple of (models_dict, coefficients_df, metadata)
    """
    # Read event log (no duration computation needed)
    variant_filter_config = variant_filter_config or {}
    min_frequency = int(variant_filter_config.get('min_frequency', 50))
    loop_handling = variant_filter_config.get('loop_handling', 'trim')
    max_activity_occurrences = variant_filter_config.get('max_activity_occurrences', 2)

    if loop_handling not in {'keep', 'remove', 'trim'}:
        raise ValueError("process_model.probabilistic.variant_filter.loop_handling must be 'keep', 'remove', or 'trim'")

    reader = EventLogReader()
    log_df = reader.preprocess_for_simulation(
        xes_path,
        filter_prefix='W_',
        context_attributes=context_attributes,
        compute_durations=False  # Just need event log
    )
    
    # Extract process variants for training
    logger.info("   Extracting process variants...")
    variants_list = []
    variant_counts = {}
    
    for case_id, case_df in log_df.groupby('case:concept:name'):
        case_df = case_df.sort_values('time:timestamp')
        activities = case_df['concept:name'].tolist()
        variant_str = ' -> '.join(activities)
        
        if variant_str not in variant_counts:
            variant = ProcessVariant(
                activities=activities,
                frequency=0
            )
            variants_list.append(variant)
            variant_counts[variant_str] = 0
        
        variant_counts[variant_str] += 1
    
    # Update frequencies
    for variant in variants_list:
        variant_str = ' -> '.join(variant.activities)
        variant.frequency = variant_counts[variant_str]
    
    logger.info(f"   Found {len(variants_list)} unique process variants")

    if min_frequency > 1:
        logger.info(
            "   Removing variants with frequency < %s (to ensure enough data for model training)",
            min_frequency,
        )
        variants_list = [v for v in variants_list if v.frequency >= min_frequency]
    else:
        logger.info("   Variant frequency filtering disabled (min_frequency=%s)", min_frequency)

    loop_heavy_variants = [
        variant for variant in variants_list
        if has_excessive_loops(variant.activities, max_activity_occurrences)
    ] if max_activity_occurrences is not None else []

    total_revisits = sum(count_activity_revisits(variant.activities) for variant in variants_list)
    logger.info(
        "   Variant loop filter: mode=%s, max_activity_occurrences=%s, loop_heavy_variants=%s, total_revisits=%s",
        loop_handling,
        max_activity_occurrences,
        len(loop_heavy_variants),
        total_revisits,
    )

    if loop_handling == 'remove' and max_activity_occurrences is not None:
        variants_list = [
            variant for variant in variants_list
            if not has_excessive_loops(variant.activities, max_activity_occurrences)
        ]
        logger.info(f"   {len(variants_list)} variants remain after frequency and loop filtering")
    else:
        logger.info(f"   {len(variants_list)} variants remain after frequency filtering")
        if loop_handling == 'trim' and loop_heavy_variants:
            logger.info("   Loop-heavy variants will be trimmed during training dataset construction")

    if not variants_list:
        logger.warning("   Variant filtering removed all variants; transition model training will receive no traces")
    
    # Build simple first-order transition probabilities from filtered variants.
    # State is current activity, next state is next activity in trace.
    # Also keep a START distribution for optional initial-task fallback.
    logger.info("   Building simplified transition matrix (relative next-activity frequencies)...")

    start_counts: Dict[str, int] = {}
    transition_counts: Dict[str, Dict[str, int]] = {}

    kept_variants = 0
    trimmed_variants = 0
    skipped_loop_variants = 0

    for variant in variants_list:
        activities = list(variant.activities)

        if loop_handling == 'remove' and has_excessive_loops(activities, max_activity_occurrences):
            skipped_loop_variants += 1
            continue

        if loop_handling == 'trim':
            trimmed = trim_looping_activities(activities, max_activity_occurrences)
            if trimmed != activities:
                trimmed_variants += 1
            activities = trimmed

        if not activities:
            continue

        freq = int(variant.frequency)
        kept_variants += 1

        start_activity = activities[0]
        start_counts[start_activity] = start_counts.get(start_activity, 0) + freq

        for src, dst in zip(activities[:-1], activities[1:]):
            if src not in transition_counts:
                transition_counts[src] = {}
            transition_counts[src][dst] = transition_counts[src].get(dst, 0) + freq

    # Convert counts to probabilities
    models: Dict[str, Dict[str, float]] = {}
    for src, dst_counts in transition_counts.items():
        total = float(sum(dst_counts.values()))
        if total <= 0:
            continue
        models[src] = {dst: count / total for dst, count in dst_counts.items()}

    start_total = float(sum(start_counts.values()))
    if start_total > 0:
        models['__START__'] = {dst: count / start_total for dst, count in start_counts.items()}

    # Keep coefficients artifact for compatibility with existing logging and interfaces
    coefficients = pd.DataFrame()

    activity_labels = sorted(set(log_df['concept:name'].unique().tolist()))
    metadata = TransitionModelMetadata(
        context_attributes=context_attributes,
        categorical_attributes=[],
        categorical_values={},
        history_mode=None,
        activity_labels=activity_labels,
        feature_names=['simple_transition_probabilities'],
    )

    logger.info(
        "   Built simplified transitions for %s states (kept_variants=%s, trimmed_variants=%s, skipped_loop_variants=%s)",
        len(models),
        kept_variants,
        trimmed_variants,
        skipped_loop_variants,
    )

    return models, coefficients, metadata


def build_probabilistic_process_model(
    xes_path: Path,
    context_attributes: List[str],
    prob_config: Dict[str, Any],
    output_path: Path
) -> ProbabilisticProcessModel:
    """Build probabilistic process model with transition weights.
    
    Builds transition models in memory (no intermediate files needed).
    
    Args:
        xes_path: Path to XES log
        context_attributes: Case attributes for context
        prob_config: Probabilistic model configuration
        output_path: Path to save process model
        
    Returns:
        ProbabilisticProcessModel instance
    """
    logger.info("3. Building probabilistic process model...")
    
    # Get configuration
    history_mode = prob_config.get('history_mode', 'binary')
    rng_seed = prob_config.get('random_seed', 42)
    variant_filter_config = prob_config.get('variant_filter', {})
    
    # Build transition models in memory
    logger.info("   Building transition models from XES log...")
    models, coefficients, metadata = _build_transition_models_in_memory(
        xes_path=xes_path,
        context_attributes=context_attributes,
        history_mode=history_mode,
        variant_filter_config=variant_filter_config,
    )
    
    logger.info(f"   Context attributes: {metadata.context_attributes}")
    logger.info(f"   History mode: {metadata.history_mode}")
    
    # Create probabilistic process model
    process_model = ProbabilisticProcessModel(
        transition_models=models,
        metadata=metadata,
        history_mode=history_mode,
        rng_seed=rng_seed
    )
    
    logger.info(f"   Created ProbabilisticProcessModel with {len(metadata.activity_labels)} activities")
    
    # Show top features
    if len(coefficients) > 0:
        logger.info("   Top 5 features by importance (mean absolute coefficient):")
        abs_coeffs = coefficients.abs()
        mean_importance = abs_coeffs.mean().sort_values(ascending=False)
        for i, (feature, importance) in enumerate(mean_importance.head(5).items(), 1):
            logger.info(f"     {i}. {feature}: {importance:.3f}")
    
    # Save process model
    with open(output_path, 'wb') as f:
        pickle.dump(process_model, f)
    logger.info(f"   Saved to {output_path}")
    
    return process_model

def main():
    """Initialize simulation components from XES log."""
    logger.info("=== Initializing Simulation Components ===")
    
    # Suppress pm4py warnings
    warnings.filterwarnings('ignore', message='.*rustxes.*', category=UserWarning)
    
    # Load configuration
    config_path = Path("config/simulation_config.yaml")
    logger.info("0. Loading configuration...")
    config = load_configuration(config_path)
    
    # --- Extract configuration parameters ---
    log_xes = Path(config.get('process_model', {}).get('log_path', 
                             'data/historical_logs/BPIC17/BPI_Challenge_2017.xes'))
    experience_store_path = Path(config.get('experience', {}).get('experience_store_path', 
                                           'data/experience_store.json'))
    process_model_path = Path(config.get('process_model', {}).get('model_path', 
                                        'data/process_model.pkl'))
    
    process_model_type = config.get('process_model', {}).get('type', 'variant')

    # Use the correct context_attributes section for each component.
    # process_model and experience sections each define their own list.
    pm_context_attributes = config.get('process_model', {}).get(
        'context_attributes', ['case:LoanGoal', 'case:ApplicationType'])
    exp_context_attributes = config.get('experience', {}).get(
        'context_attributes', ['case:LoanGoal', 'case:ApplicationType'])

    capability_mapping = config.get('experience', {}).get('capability_mapping', [])
    learning_model = config.get('experience', {}).get('learning_model', 'richards')
    breeding_params = config.get('experience', {}).get('breeding_params', {})
    default_std = config.get('experience', {}).get('default_std', 1080)  # config seconds
    min_avg_daily_hours = config.get('experience', {}).get('min_avg_daily_hours', 0.5)
    duration_assignment_strategy = config.get('experience', {}).get('duration_assignment_strategy', 'longest_resource')
    training_split = config.get('experience', {}).get('training_split', 0.4)
    prob_config = config.get('process_model', {}).get('probabilistic', {})
    
    # Create output directories
    Path("data").mkdir(exist_ok=True)
    Path("data/simulation_outputs").mkdir(parents=True, exist_ok=True)
    
    # Validate XES log exists
    if not log_xes.exists():
        logger.error(f"   XES log not found: {log_xes}")
        return
    
    # Create timeline from XES (use experience context attributes for duration profiles)
    timeline_df, segments_df = create_timeline_from_xes(
        log_xes,
        exp_context_attributes,
        duration_assignment_strategy=duration_assignment_strategy,
    )
    
    # Save timeline CSV for duration model training
    timeline_csv_path = Path(config.get('case_arrival', {}).get('timeline_path', 'data/timeline.csv'))
    timeline_csv_path.parent.mkdir(parents=True, exist_ok=True)
    timeline_df.to_csv(timeline_csv_path, index=False)
    logger.info(f"   Saved timeline to {timeline_csv_path}")
    
    # compute and save activity p99 durations
    activity_req_path = Path("config/activity_requirements.yaml")
    positive_durations_df = timeline_df[timeline_df["duration_seconds"] > 30].copy() # TODO change back to 0?
    positive_durations_df['duration_seconds'] = positive_durations_df['duration_seconds'].clip(upper=8*3600)  # cap at 8h to avoid extreme outliers
    percentile_98 = positive_durations_df.groupby("activity_name")["duration_seconds"].quantile(0.98)
    percentile_98 = percentile_98.to_dict()
    percentile_98["W_Assess potential fraud"] = 18000 # override for this activity which has some extreme outliers that distort the p99
    write_activity_benchmarks(percentile_98, activity_req_path, section="default_beginner_durations")
    
    # Build experience store (training_split from config, no hardcoded split_date)
    activity_req_path = Path("config/activity_requirements.yaml")
    with open(activity_req_path, 'r') as f:
        req_config = yaml.safe_load(f) or {}
        requirements = req_config.get('activity_requirements', {})
        activity_requirements = {
            str(k): float(v) for k, v in requirements.items()
            if v is not None
        }
    experience_store, split_date = build_experience_store(
        timeline_df=timeline_df,
        training_split=training_split,
        split_date=None,  # let the split be computed from training_split
        context_attributes=exp_context_attributes,
        capability_mapping=capability_mapping,
        learning_model=learning_model,
        breeding_params=breeding_params,
        default_durations=percentile_98,
        default_std=default_std,
        output_path=experience_store_path,
        min_avg_daily_hours=min_avg_daily_hours,
        activity_requirements=activity_requirements
    )

    # ---- Compute and save activity benchmarks ----
    benchmarks = compute_activity_benchmarks(experience_store)
    print(f"   Computed benchmarks for {len(benchmarks)} activities, saving to {activity_req_path}...")
    print(f"   Sample benchmarks:")
    for activity, benchmark in list(benchmarks.items()):
        print(f"     {activity}: {benchmark}")
    write_activity_benchmarks(benchmarks, activity_req_path, section="activity_benchmarks")

    calendar_path = Path(config.get('working_hours', {}).get('calendar_path', 'data/calendars.json'))
    generate_calendars(
        experience_store_path=experience_store_path,
        output_path=calendar_path,
        config=config,
    )
    
    # ---- Persist split info so run_simulation / dashboard can load it ----
    split_info_path = Path(
        config.get('case_arrival', {}).get('split_info_path', 'data/timeline_split_info.json')
    )
    timestamps = pd.to_datetime(timeline_df['start_timestamp'])
    training_events = int((timestamps < split_date).sum())
    testing_events = int((timestamps >= split_date).sum())
    split_info = {
        'split_date': split_date.strftime('%Y-%m-%d'),
        'training_events': training_events,
        'testing_events': testing_events,
        'training_split': training_split,
        'min_date': str(timestamps.min().date()),
        'max_date': str(timestamps.max().date()),
    }
    split_info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    logger.info(f"   Saved split info to {split_info_path}")

    # Build process model (uses process_model context attributes)
    process_model = build_probabilistic_process_model(
        xes_path=log_xes,
        context_attributes=pm_context_attributes,
        prob_config=prob_config,
        output_path=process_model_path
    )
    
    # Print summary
    logger.info('=' * 60)
    logger.info("=== Initialization Complete ===")
    logger.info('=' * 60)
    logger.info(" Saved files:")
    logger.info(f"  • Experience store: {experience_store_path}")
    logger.info(f"  • Process model ({process_model_type}): {process_model_path}")
    logger.info(f"  • Calendars: {calendar_path}")
    
    logger.info(f" Training split: {split_date.date()}")
    logger.info(" Next steps:")
    logger.info("  Run simulation: python scripts/run_simulation.py")
    
    return experience_store, process_model

if __name__ == "__main__":
    main()