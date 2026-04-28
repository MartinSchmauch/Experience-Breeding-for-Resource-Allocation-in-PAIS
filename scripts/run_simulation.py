"""Run business process simulation with different schedulers.

Reads ``config/simulation_config.yaml`` for all parameters, loads the
pre-built experience store and process model, then runs a single simulation
with the configured scheduler, learning model, and evaluation weights.
"""
from __future__ import annotations

import json
import logging
import math
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from tqdm import tqdm

# Ensure project root is on the path for ``src`` imports.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.simulation import SimulationEngine  # noqa: E402
from src.experience.updater import LearningModel  # noqa: E402
from src.scheduling import ExperienceBasedScheduler, RandomScheduler, GreedyScheduler
from src.experience import ExperienceStore  # noqa: E402
from src.io import EventLogWriter, LoggingConfigurator  # noqa: E402
from src.entities import ResourceFactory  # noqa: E402
from src.entities.calendar import ResourceCalendar  # noqa: E402
from src.utils.time_utils import create_default_converter, seconds_to_hours, seconds_to_days, SECONDS_PER_DAY  # noqa: E402
from src.evaluation import KPICalculator, DailySummaryLogger  # noqa: E402
from src.prediction import ModelTrainer  # noqa: E402
from src.simulation.case_generator import CaseGenerator  # noqa: E402

logger = logging.getLogger(__name__)

# Map learning model string to enum
LEARNING_MODEL_ENUM_MAP = {
    'richards': LearningModel.RICHARDS,
}

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path("config/simulation_config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_run_stats(output_dir: Path, output_name: str, stats: Dict, kpis: Dict) -> None:
    """Persist the serialisable portion of simulation stats + KPIs to JSON.

    This file is loaded by the Analysis & Comparison dashboard page so that
    batch-scheduling metrics (drain, deferral, overtime, queue peaks, …) can
    be compared across runs without re-running the simulation.
    """
    serialisable = {
        'simulation_time': stats.get('simulation_time'),
        'cases_completed': stats.get('cases_completed'),
        'total_cases': stats.get('total_cases'),
        'completion_rate': stats.get('completion_rate'),
        'queue_stats': stats.get('queue_stats', {}),
        'unfinished_work': stats.get('unfinished_work', {}),
        'drain_stats': stats.get('drain_stats', {}),
        'kpis': {k: v for k, v in kpis.items() if isinstance(v, (int, float, str, bool, type(None)))},
    }
    # Remove non-serialisable nested dicts (e.g. per-resource dicts with
    # large key sets are fine — they are plain str→number mappings).
    ot = stats.get('overtime_stats')
    if ot:
        serialisable['overtime_stats'] = {
            'total_overtime_hours': ot.get('total_overtime_hours', 0),
            'max_overtime_hours': ot.get('max_overtime_hours', 0),
            'max_overtime_resource_id': ot.get('max_overtime_resource_id'),
        }

    stats_path = output_dir / f"{output_name}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(serialisable, f, indent=2, default=str)
    logger.info(f"Run stats saved to {stats_path}")

def load_duration_predictor(pred_config: Dict[str, Any]):
    """Load duration predictor if enabled in config."""
    model_dir = Path(pred_config.get('model_dir', 'models'))
    filename_prefix = pred_config.get('filename_prefix', 'duration_model')
    
    trainer = ModelTrainer(model_dir=model_dir)
    
    try:
        predictor = trainer.load_latest_model(filename_prefix=filename_prefix)
        
        if predictor is None:
            logger.warning(f"No duration models found in {model_dir}")
            logger.warning("Run 'python scripts/train_duration_model.py' to train model")
            return None
        
        logger.info(f"Loaded duration predictor (version: {predictor.model_version})")
        return predictor
        
    except Exception as e:
        logger.warning(f"Failed to load duration predictor: {e}")
        logger.warning("Continuing without duration prediction...")
        return None

def load_components(config: Optional[Dict[str, Any]] = None):
    """Load pre-initialized simulation components from saved files."""
    logger.info("Loading simulation components...")

    # Define paths
    experience_store_path = Path(
        config.get('experience', {}).get('experience_store_path', 'data/experience_store.json')
    ) if config else Path("data/experience_store.json")
    process_model_path = Path(
        config.get('process_model', {}).get('model_path', 'data/process_model.pkl')
    ) if config else Path("data/process_model.pkl")
    
    # Check required files
    if not experience_store_path.exists():
        raise FileNotFoundError(f"Experience store not found: {experience_store_path}")
    if not process_model_path.exists():
        raise FileNotFoundError(f"Process model not found: {process_model_path}")
    
    # Load experience store
    experience_store = ExperienceStore.load(experience_store_path)
    logger.info(f"Loaded experience store with {len(experience_store)} profiles")
    
    # Load process model
    with open(process_model_path, 'rb') as f:
        process_model = pickle.load(f)
    
    # Print info based on model type
    if hasattr(process_model, 'transition_models'):
        logger.info(f"Loaded ProbabilisticProcessModel with {len(process_model.transition_models)} transition models")
        logger.info(f"  History mode: {process_model.history_mode}")
        logger.info(f"  Context attributes: {process_model.context_attributes}")
    else:
        logger.info(f"Loaded process model: {type(process_model).__name__}")
    
    # Generate resources from experience store
    logger.info("Generating resources from experience store...")
    resources = ResourceFactory().create_resources(experience_store)
    
    logger.info(f"Generated {len(resources)} resources from experience store")
    has_hr_data = any(r.name != r.id for r in resources)
    if has_hr_data:
        logger.info("Resources include HR data (names, roles, capabilities)")
    else:
        logger.info("Resources generated without HR data (using IDs only)")
    
    return experience_store, process_model, resources

def load_calendars(config: Dict[str, Any]) -> Optional[Dict[str, ResourceCalendar]]:
    """Load resource calendars from JSON file."""
    working_hours_config = config.get('working_hours', {})
    
    if not working_hours_config.get('enabled', False):
        return None
    
    calendar_path = Path(working_hours_config.get('calendar_path', 'data/calendars.json'))
    
    if not calendar_path.exists():
        logger.warning(f"Calendar file not found: {calendar_path}")
        logger.warning("Working hours will not be enforced. Run 'python scripts/initialize_simulation.py' to generate calendars and initialization artifacts.")
        return None
    
    try:
        with open(calendar_path, 'r') as f:
            calendars_data = json.load(f)
        
        calendars = {
            resource_id: ResourceCalendar.from_dict(calendar_data)
            for resource_id, calendar_data in calendars_data.items()
        }
        
        logger.info(f"Loaded {len(calendars)} resource calendars")
        total_absences = sum(len(cal.absences) for cal in calendars.values())
        logger.info(f"  Total absences across all resources: {total_absences}")
        
        return calendars
        
    except Exception as e:
        logger.warning(f"Error loading calendars: {e}")
        logger.warning("Working hours will not be enforced.")
        return None


def run_single_simulation(scheduler, process_model, resources, experience_store, 
                          cases, output_name, timestamp, learning_model='richards',
                          use_progress_bar=False, total_timeline_events=None, 
                          resource_calendars=None, time_converter=None,
                          starting_weekday=None, config=None, output_dir=None):
    """Run simulation with given scheduler using SimPy engine
    
    Args:
        scheduler: Scheduler instance
        process_model: Process model defining workflows
        resources: List of resources
        experience_store: Experience store
        cases: List of cases to simulate
        output_name: Output file name
        timestamp: Timestamp for output naming
        learning_model: Learning model name ('richards')
        use_progress_bar: Whether to show tqdm progress bar (when logging disabled)
        total_timeline_events: Total number of timeline events (for accurate progress tracking)
        resource_calendars: Optional dict of ResourceCalendar objects (loaded in main)
        time_converter: Optional SimulationTimeConverter (created in main)
        config: Configuration dictionary
    """
    
    if not use_progress_bar:
        print(f"\n{'='*60}")
        print(f"Running simulation: {output_name}")
        print(f"{'='*60}")
        print(f"Learning model: {learning_model}")
    
    learning_model_enum = LEARNING_MODEL_ENUM_MAP.get(
        learning_model.lower(), 
        LearningModel.RICHARDS
    )
    
    # Create log writer
    log_writer = EventLogWriter(
        output_path=output_dir / f"{output_name}.csv"
    )

    daily_summary_logger = None
    daily_summary_path = None
    daily_summary_enabled = bool(config.get('output', {}).get('daily_summary_logging', True)) if config else True
    if daily_summary_enabled:
        daily_summary_path = output_dir / f"{output_name}_daily_summary.jsonl"
        daily_summary_logger = DailySummaryLogger(daily_summary_path)
            
    log_writer.set_metadata(
        scheduler=scheduler.__class__.__name__,
        timestamp=timestamp,
        case_arrival_mode="Probabilistic (Context-Aware)",
        process_model_type=process_model.__class__.__name__,
        total_cases=len(cases),
        learning_model=learning_model,
        history_mode=process_model.history_mode,
        probabilistic_seed=42,
        mentoring_enabled=config.get('mentoring', {}).get('enabled', False),
        simulation_start_day=starting_weekday if time_converter else None,
        config=config,
    )
    
    # Convert resources dict to dict of Resource objects if needed
    resources_dict = resources if isinstance(resources, dict) else {r.id: r for r in resources}
    
    # Create progress bar if needed
    progress_bar = None
    if use_progress_bar:
        # Track simulation time progress (in days)
        total_days = config.get('simulation', {}).get('max_simulation_days', None)
        if total_days:
            progress_bar = tqdm(
                total=total_days,
                desc=f"{scheduler.__class__.__name__:<25}",
                unit="days",
                dynamic_ncols=True,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.1f}/{total:.1f} days [{elapsed}<{remaining}]'
            )
        else:
            # Fallback to task-based progress if no time limit
            progress_bar = tqdm(
                total=total_timeline_events if total_timeline_events else len(cases),
                desc=f"{scheduler.__class__.__name__:<25}",
                unit="tasks",
                dynamic_ncols=True,
                miniters=100,
                mininterval=0,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
    
    # Create SimPy-based simulation engine
    engine = SimulationEngine(
        process_model=process_model,
        scheduler=scheduler,
        experience_store=experience_store,
        resources=resources_dict,
        log_writer=log_writer,
        learning_model=learning_model_enum,
        progress_bar=progress_bar,  # Pass progress bar to engine
        resource_calendars=resource_calendars,
        time_converter=time_converter,
        config=config,
        daily_summary_logger=daily_summary_logger,
    )
    
    # Batch scheduler process is now started automatically in engine __init__
    # Schedule case arrivals
    for case in cases:
        engine.schedule_case_arrival(case, case.arrival_time)
    
    # Run simulation
    stats = engine.run()  # SimPy engine returns stats directly
    if daily_summary_path is not None:
        stats['daily_summary_path'] = str(daily_summary_path)
    # Close progress bar
    if progress_bar:
        progress_bar.close()
    
    if not use_progress_bar:
        logger.info(f"Simulation completed in {stats['simulation_time']:.2f} hours")
        # logger.info(f"Completion rate: {stats['completion_rate']:.2%}")
    
    # Get log dataframe before finalizing
    log_df = log_writer.to_dataframe()
    
    # Save log
    log_writer.save_with_metadata()
    if not use_progress_bar:
        logger.info(f"  Log saved to: {log_writer.output_path}")
    
    # Save experience tracker data if available
    if 'experience_tracker' in stats and stats['experience_tracker'] is not None:
        tracker_csv_path = output_dir / f"{output_name}_experience.csv"
        try:
            stats['experience_tracker'].save_to_csv(tracker_csv_path)
            if not use_progress_bar:
                logger.info(f"  Experience tracker saved to: {tracker_csv_path}")
        except Exception as e:
            if not use_progress_bar:
                logger.warning(f"  Warning: Could not save experience tracker: {e}")
    
    return log_df, stats

def evaluate_results(
    log_df, simulation_time, scheduler_type,
    use_progress_bar: bool = False,
    config: Optional[Dict[str, Any]] = None,
    daily_summary_path: Optional[str] = None,
):
    """Compute KPIs"""
    
    # Check if log is empty
    if log_df is None or len(log_df) == 0:
        if not use_progress_bar:
            logger.warning(f"\n{scheduler_type} - WARNING: Event log is empty! Cannot compute KPIs.")
        return {}, float('inf')
    
    if not use_progress_bar:
        logger.info(f"\n{scheduler_type} - Evaluating {len(log_df)} events...")
        logger.info(f"  Log columns: {list(log_df.columns)}")
    
    calculator = KPICalculator()
    kpis = calculator.compute_all(
        log_df,
        simulation_start=0.0,
        simulation_end=simulation_time
    )

    if daily_summary_path:
        daily_path = Path(daily_summary_path)
        if daily_path.exists():
            daily_kpis = calculator.compute_from_daily_summary_file(daily_path)
            kpis.update(daily_kpis)
    
    if not use_progress_bar:
        for kpi_name, kpi_value in kpis.items():
            logger.info(f"  {kpi_name}: {kpi_value}")
    
    
    if not use_progress_bar:
        logger.info(f"\n{scheduler_type} - KPIs:")
        logger.info(f"  Daily clearance rate: {kpis.get('mean_daily_clearance_rate', 0):.2%}")
        logger.info(f"  Days fully cleared: {kpis.get('pct_days_fully_cleared', 0):.2%}")
        logger.info(f"  Mean queue wait: {seconds_to_hours(kpis.get('mean_queue_wait_time', 0)):.2f} hours")
        logger.info(f"  Mean daily backlog: {kpis.get('mean_daily_backlog', 0):.1f} tasks")
        logger.info(f"  Overall completion: {kpis.get('overall_completion_rate', 0):.2%}")
        logger.info(f"  Mean service time: {seconds_to_hours(kpis.get('mean_service_time', 0)):.2f} hours")
    
    return kpis

def main():
    print("=== Business Process Simulation Runner ===\n")
    
    # Load configuration
    config = load_config()
    
    logging_enabled = LoggingConfigurator.configure(config)
    use_progress_bar = not logging_enabled
    
    # Extract experience and breeding configuration
    exp_config = config.get('experience', {})
    learning_model = exp_config.get('learning_model', 'richards')
    
    # Extract scheduling configuration
    scheduling_config = config.get('scheduling', {})
    scheduling_mode = scheduling_config.get('mode', 'immediate')
    scheduling_time = scheduling_config.get('scheduling_time', 8.0)
    planning_horizon_hours = scheduling_config.get('planning_horizon_hours', 24.0)
    
    if scheduling_mode == 'batch':
        print(f"  Daily scheduling at: {scheduling_time}:00")
        print(f"  Planning horizon: {planning_horizon_hours}h")
    if use_progress_bar:
        print("Progress bars: enabled (logging disabled)")
    
    # Load duration predictor (if enabled and available)
    pred_config = config.get('duration_prediction', {})
    if pred_config.get('enabled', False):
        duration_predictor = load_duration_predictor(pred_config)
    else:
        duration_predictor = None
    
    # Load components
    experience_store, process_model, resources = load_components(config)
    
    # ---- Load split info (needed for time_converter and case generation) ----
    use_test_split = config.get('case_arrival', {}).get('use_test_split', True)
    split_info: Optional[Dict] = None
    if use_test_split:
        split_info_path = Path(
            config.get('case_arrival', {}).get('split_info_path', 'data/timeline_split_info.json')
        )
        with open(split_info_path, 'r') as f:
                split_info = json.load(f)

    # Base start date for case generation (used for split filtering even when
    # working-hours calendars are disabled).
    if split_info and 'split_date' in split_info:
        starting_date = datetime.strptime(split_info['split_date'], '%Y-%m-%d')
    else:
        start_date_config = config.get('working_hours', {}).get('start_date', {})
        year = start_date_config.get('year', 2016)
        month = start_date_config.get('month', 1)
        day = start_date_config.get('day', 1)
        starting_date = datetime(year, month, day)
                
    # Load calendars if working hours enabled
    resource_calendars = None
    time_converter = None
    starting_weekday = None
    
    if config.get('working_hours', {}).get('enabled', False):
        resource_calendars = load_calendars(config)
        
        if resource_calendars:
            # When using test split, anchor time_converter to the split date
            # so sim_time=0.0 corresponds to the first test-set case arrival.
            if split_info and 'split_date' in split_info:
                print(
                    "Working hours enabled: time origin overridden to split date "
                    f"{starting_date.year}-{starting_date.month:02d}-{starting_date.day:02d}"
                )
            else:
                print(
                    "Working hours enabled: simulation starts "
                    f"{starting_date.year}-{starting_date.month:02d}-{starting_date.day:02d}"
                )
            time_converter = create_default_converter(
                year=starting_date.year,
                month=starting_date.month,
                day=starting_date.day,
            )
            print(f"Time converter created with starting date at {starting_date}")
            starting_weekday = time_converter.get_weekday(0.0)
            print(f"Starting WEEKDAY: {starting_weekday} (0=Monday, 6=Sunday)")
    
    # Generate cases using configured mode
    print("\nGenerating cases...")
    case_arrival_mode = config.get('case_arrival', {}).get('mode', 'probabilistic')
    case_arrival_config = dict(config.get('case_arrival', {}))  # shallow copy
    # Inject split_date so CaseGenerator filters to test split
    if split_info and 'split_date' in split_info:
        case_arrival_config['split_date'] = split_info['split_date']
    case_generator = CaseGenerator(case_arrival_config=case_arrival_config, starting_date=starting_date)
    cases, total_timeline_events = case_generator.generate_cases()
    print(f"\nGenerated {len(cases)} cases using mode: {case_arrival_mode}")
    print(f"Total timeline events (estimated): {total_timeline_events}")

    # ---- Derive simulation end time from last case arrival ----
    if cases:
        last_arrival_sec = cases[-1].arrival_time  # int seconds
        print(f"Last arrival second: {last_arrival_sec} (day {seconds_to_days(last_arrival_sec):.1f}, hour {seconds_to_hours(last_arrival_sec):.1f}h)")
        arrival_based_days = math.ceil(last_arrival_sec / SECONDS_PER_DAY)
        config_max_days = config.get('simulation', {}).get('max_simulation_days', None)
        if config_max_days is not None:
            effective_days = min(arrival_based_days, config_max_days)
        else:
            effective_days = arrival_based_days
        # Override so the engine uses the arrival-based end
        config.setdefault('simulation', {})['max_simulation_days'] = effective_days
        print(f"Simulation end: day {effective_days} (last arrival at {seconds_to_hours(last_arrival_sec):.1f}h = day {seconds_to_days(last_arrival_sec):.1f})")
    
    # Create scheduler from config
    scheduler_config = config.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'experience_based').lower()
    output_dir = Path(config.get('output', {}).get('directory', 'data/simulation_outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"sim_{scheduler_type}_{timestamp}"
    
    # Map scheduler type to scheduler instance
    if scheduler_type == 'experience_based':
        if duration_predictor is not None:
            scheduler_type += "_ml"

        scheduler = ExperienceBasedScheduler(
            duration_predictor=duration_predictor,
            config=config,
            time_converter=time_converter
        )
    elif scheduler_type == 'random':
        scheduler = RandomScheduler(
            duration_predictor=duration_predictor,
            config=config,
            time_converter=time_converter,
        )
    elif scheduler_type == 'greedy':
        scheduler = GreedyScheduler(
            duration_predictor=duration_predictor,
            config=config,
            time_converter=time_converter,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Valid options: 'random', 'greedy', 'experience_based'")
    
    print(f"Scheduler initialized: {scheduler.__class__.__name__}")
    if duration_predictor is not None:
        print(f"Duration predictor enabled: {duration_predictor.model_version}")
    
    # Run simulation
    
    if use_progress_bar:
        print(f"\n{'='*60}")
        print(f"Running simulation with {len(cases)} cases")
        print(f"{'='*60}\n")
    
    log_df, stats = run_single_simulation(
        scheduler=scheduler,
        process_model=process_model,
        resources=resources,
        experience_store=experience_store,
        cases=cases,
        output_name=output_name,
        timestamp=timestamp,
        learning_model=learning_model,
        use_progress_bar=use_progress_bar,
        total_timeline_events=total_timeline_events,
        resource_calendars=resource_calendars,
        time_converter=time_converter,
        starting_weekday=starting_weekday,
        config=config,
        output_dir=output_dir,
    )
    
    kpis = evaluate_results(
        log_df, 
        stats['simulation_time'], 
        scheduler_type,
        use_progress_bar=use_progress_bar,
        config=config,
        daily_summary_path=stats.get('daily_summary_path'),
    )
    
    # Persist run stats for dashboard comparison
    save_run_stats(output_dir, output_name, stats, kpis)
    
    # Display queue statistics
    if 'queue_stats' in stats:
        queue_stats = stats['queue_stats']
        print(f"\n  Queue Statistics:")
        print(f"    Tasks remaining in queues: {queue_stats['tasks_remaining_in_queues']}")
        print(f"    Peak queue length: {queue_stats['max_queue_length']}")
        if queue_stats['max_queue_resource_id']:
            print(f"    Resource with peak queue: {queue_stats['max_queue_resource_id']}")

    # Display drain / deferral statistics if available
    if 'drain_stats' in stats:
        ds = stats['drain_stats']
        print(f"\n  Drain & Deferral Statistics:")
        print(f"    Total tasks drained back: {ds['total_drained_tasks']}")
        print(f"    Days with drain: {ds['drain_days']}")
        print(f"    Total deferred (solver): {ds['total_deferred_tasks']}")
        print(f"    Dropped (max deferrals): {ds['total_dropped_tasks']}")
        if ds['tasks_in_unscheduled_pool'] > 0:
            print(f"    Tasks still in unscheduled pool: {ds['tasks_in_unscheduled_pool']}")
            
    # Display overtime statistics if available
    if 'overtime_stats' in stats:
        overtime_stats = stats['overtime_stats']
        print(f"\nOvertime Statistics:")
        print(f"  Total overtime hours: {overtime_stats['total_overtime_hours']:.2f}h")
        print(f"  Max overtime: {overtime_stats['max_overtime_hours']:.2f}h (resource: {overtime_stats['max_overtime_resource_id']})")
        print(f"  Overall the resource with the longest overtime had the following overtime hours: {overtime_stats['max_overtime_resource_hours']}")
        print(f"  The longest overtime section of that resource {overtime_stats['max_single_overtime_instance_resource_id']} was: {overtime_stats['max_single_overtime_instance_hours']}h for task {overtime_stats['max_single_overtime_instance_task_id']}")

    # Display summary
    print(f"\n{'='*60}")
    print("SIMULATION SUMMARY")
    print(f"{'='*60}")
    print(f"Scheduler: {scheduler_type}")
    print(f"  Daily Clearance Rate: {kpis.get('mean_daily_clearance_rate', 0):.2%}")
    print(f"  Days Fully Cleared: {kpis.get('pct_days_fully_cleared', 0):.2%}")
    print(f"  Queue Wait Time: {kpis.get('mean_queue_wait_time', 0):.2f}h")
    print(f"  Daily Backlog: {kpis.get('mean_daily_backlog', 0):.1f} tasks")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()