import streamlit as st
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import yaml
import pickle

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.simulation import SimulationEngine
from src.simulation.case_generator import CaseGenerator
from src.experience.updater import LearningModel
from src.scheduling import ExperienceBasedScheduler, GreedyScheduler, RandomScheduler
from src.experience import ExperienceStore
from src.io import EventLogWriter
from src.entities import ResourceFactory
from src.entities.calendar import ResourceCalendar
from src.utils.time_utils import create_default_converter, seconds_to_hours, seconds_to_days, SECONDS_PER_DAY
from src.evaluation import KPICalculator
from src.prediction import ModelTrainer
import json
import math

st.set_page_config(page_title="Run Simulation", page_icon="🚀", layout="wide")

st.title("🚀 Run Simulation")

# --- Load Config ---
@st.cache_data
def load_config():
    """Load configuration from YAML."""
    config_path = Path("config/simulation_config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Map learning model string to enum (same as run_simulation.py)
LEARNING_MODEL_ENUM_MAP = {
    'richards': LearningModel.RICHARDS,
}

def load_calendars(working_hours_config):
    """Load resource calendars from JSON file."""
    if not working_hours_config.get('enabled', False):
        return None
    
    calendar_path = Path(working_hours_config.get('calendar_path', 'data/calendars.json'))
    
    if not calendar_path.exists():
        st.warning(f"Calendar file not found: {calendar_path}")
        return None
    
    try:
        with open(calendar_path, 'r') as f:
            calendars_data = json.load(f)
        
        calendars = {}
        for resource_id, calendar_data in calendars_data.items():
            calendars[resource_id] = ResourceCalendar.from_dict(calendar_data)
        
        return calendars
    except Exception as e:
        st.error(f"Error loading calendars: {e}")
        return None


def load_duration_predictor(pred_config):
    """Load duration predictor if enabled in config."""
    model_dir = Path(pred_config.get('model_dir', 'models'))
    filename_prefix = pred_config.get('filename_prefix', 'duration_model')
    trainer = ModelTrainer(model_dir=model_dir)
    try:
        predictor = trainer.load_latest_model(filename_prefix=filename_prefix)
        return predictor
    except Exception:
        return None


def _save_run_stats(output_dir: Path, output_name: str, stats: dict, kpis: dict, score: float = 0.0) -> None:
    """Persist serialisable simulation stats + KPIs to JSON for later comparison."""
    serialisable = {
        'simulation_time': stats.get('simulation_time'),
        'cases_completed': stats.get('cases_completed'),
        'total_cases': stats.get('total_cases'),
        'completion_rate': stats.get('completion_rate'),
        'queue_stats': stats.get('queue_stats', {}),
        'unfinished_work': stats.get('unfinished_work', {}),
        'drain_stats': stats.get('drain_stats', {}),
        'kpis': {k: v for k, v in kpis.items() if isinstance(v, (int, float, str, bool, type(None)))},
        'objective_score': score,
    }
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


# --- Configuration ---
st.sidebar.header("⚙️ Configuration")

# ---- Simulation Parameters ----
st.sidebar.subheader("Simulation Parameters")
sim_config = config.get('simulation', {})

max_simulation_days = st.sidebar.number_input(
    "Max Simulation Days",
    min_value=0,
    max_value=3650,
    value=int(sim_config.get('max_simulation_days', 365) or 0),
    step=30,
    help="Maximum simulation time in days. 0 = run until all cases complete.",
)
# Convert 0 → None (unlimited)
max_simulation_days = max_simulation_days if max_simulation_days > 0 else None
max_simulation_time = max_simulation_days * SECONDS_PER_DAY if max_simulation_days else None

max_activities_per_case = st.sidebar.number_input(
    "Max Activities per Case",
    min_value=0,
    max_value=100,
    value=int(sim_config.get('max_activities_per_case', 10) or 0),
    step=1,
    help="Safety limit per case. 0 = unlimited.",
)
max_activities_per_case = max_activities_per_case if max_activities_per_case > 0 else None

prob_arrival_config = config.get('case_arrival', {}).get('probabilistic', {})
case_fraction = st.sidebar.number_input(
    "Case Fraction",
    min_value=0.0,
    max_value=1.0,
    value=float(prob_arrival_config.get('case_fraction', 1.0)),
    step=0.05,
    format="%.2f",
    help="Fraction of eligible timeline cases used for simulation.",
)

optimization_config = config.get('optimization', {})
max_task_deferrals = st.sidebar.number_input(
    "Max Task Deferrals",
    min_value=0,
    max_value=100,
    value=int(optimization_config.get('max_task_deferrals', 5)),
    step=1,
    help="Maximum number of times a task may be deferred before dropping.",
)

# Update simulation params in config
if 'simulation' not in config:
    config['simulation'] = {}
config['simulation']['max_simulation_days'] = max_simulation_days
config['simulation']['max_activities_per_case'] = max_activities_per_case
if 'optimization' not in config:
    config['optimization'] = {}
config['optimization']['max_task_deferrals'] = int(max_task_deferrals)

# ---- Fixed Modes ----
case_arrival_mode = "Probabilistic (Context-Aware)"
use_arrival_from_timeline = True
if 'case_arrival' not in config:
    config['case_arrival'] = {}
config['case_arrival']['mode'] = 'probabilistic'
config['case_arrival']['probabilistic'] = config['case_arrival'].get('probabilistic', {})
config['case_arrival']['probabilistic']['use_arrival_times_from_timeline'] = True
config['case_arrival']['probabilistic']['case_fraction'] = float(case_fraction)

if 'process_model' not in config:
    config['process_model'] = {}
config['process_model']['type'] = 'probabilistic'

st.sidebar.subheader("Experience Breeding")
selected_learning_model = 'richards'
st.sidebar.caption("Learning model fixed to Richards curve")

# Advanced breeding params (collapsible)
breeding_config = config.get('experience', {}).get('breeding_params', {})
with st.sidebar.expander("Advanced Breeding Parameters"):
    lower_asymptote = st.number_input(
        "Lower Asymptote",
        value=float(breeding_config.get('lower_asymptote', 0.0)),
        min_value=0.0,
        max_value=100.0,
        step=0.1,
    )
    upper_asymptote = st.number_input(
        "Upper Asymptote",
        value=float(breeding_config.get('upper_asymptote', 100.0)),
        min_value=0.0,
        max_value=100.0,
        step=0.1,
    )
    growth_rate = st.number_input(
        "Growth Rate",
        value=float(breeding_config.get('growth_rate', 0.08)),
        min_value=0.001,
        max_value=10.0,
        step=0.01,
        format="%.3f",
    )
    shape_param_Q = st.number_input(
        "Shape Parameter Q",
        value=float(breeding_config.get('shape_param_Q', 0.5)),
        min_value=0.0,
        max_value=10.0,
        step=0.05,
    )
    shape_param_M = st.number_input(
        "Shape Parameter M",
        value=float(breeding_config.get('shape_param_M', 0.1)),
        min_value=0.0,
        max_value=10.0,
        step=0.05,
    )

# Update breeding params in config
if 'experience' not in config:
    config['experience'] = {}
config['experience']['breeding_params'] = {
    "upper_asymptote": upper_asymptote,
    "lower_asymptote": lower_asymptote,
    "growth_rate": growth_rate,
    "shape_param_Q": shape_param_Q,
    "shape_param_M": shape_param_M,
}

track_experience = st.sidebar.checkbox("Track Experience Levels", value=True, help="Record experience curves during simulation")
config['experience']['track_experience_levels'] = track_experience
config['experience']['learning_model'] = selected_learning_model

st.sidebar.subheader("Working Hours")
if 'working_hours' not in config:
    config['working_hours'] = {}
config['working_hours']['enabled'] = True

cal_path = Path(config.get('working_hours', {}).get('calendar_path', '../data/calendars.json'))
if cal_path.exists():
    st.sidebar.success("✅ Calendars found")
else:
    st.sidebar.warning("⚠️ Calendars not found")

# ---- Scheduling Mode ----
st.sidebar.subheader("Scheduling")
scheduling_config = config.get('scheduling', {})
scheduling_mode_key = "batch"

scheduling_time = scheduling_config.get('scheduling_time', 8.0)
planning_horizon_hours = scheduling_config.get('planning_horizon_hours', 24.0)

with st.sidebar.expander("Batch Scheduling Settings"):
    scheduling_time = st.number_input("Daily Scheduling Hour", min_value=0.0, max_value=23.0, value=float(scheduling_time), step=1.0)
    planning_horizon_hours = st.number_input("Planning Horizon (hours)", min_value=1.0, max_value=168.0, value=float(planning_horizon_hours), step=1.0)

# Update scheduling config
if 'scheduling' not in config:
    config['scheduling'] = {}
config['scheduling']['mode'] = scheduling_mode_key
config['scheduling']['scheduling_time'] = scheduling_time
config['scheduling']['planning_horizon_hours'] = planning_horizon_hours

# ---- Mentoring ----
st.sidebar.subheader("Mentoring")
mentoring_yaml = config.get('mentoring', {})
mentoring_enabled = st.sidebar.checkbox(
    "Enable Mentoring",
    value=mentoring_yaml.get('enabled', True),
    help="Enable mentor-mentee pairing and mentoring-specific optimization terms.",
)
mentoring_mode_options = {
    "Objective Bonus": "objective_bonus",
    "Constraint": "constraint",
}
current_mentoring_mode = mentoring_yaml.get('severe_bottleneck_mode', 'objective_bonus')
mentoring_mode_index = list(mentoring_mode_options.values()).index(current_mentoring_mode) if current_mentoring_mode in mentoring_mode_options.values() else 0
if mentoring_enabled:
    selected_mentoring_mode = st.sidebar.selectbox(
        "Severe Bottleneck Mode",
        options=list(mentoring_mode_options.keys()),
        index=mentoring_mode_index,
        help="Objective Bonus: soft incentive | Constraint: hard lower-bound enforcement",
    )
else:
    selected_mentoring_mode = "Objective Bonus"

ba_yaml = mentoring_yaml.get('bottleneck_activity_strategy', {})
uu_yaml = mentoring_yaml.get('underutilization_strategy', {})
ss_yaml = mentoring_yaml.get('same_day_shortage_strategy', {})

with st.sidebar.expander("Mentoring Strategies"):
    st.markdown("**Bottleneck Activity Strategy**")
    ba_enabled = st.checkbox(
        "Enable Bottleneck Activity Strategy",
        value=ba_yaml.get('enabled', True),
        disabled=not mentoring_enabled,
    )
    ba_resource_availability_ratio = st.number_input(
        "Bottleneck: Resource Availability Ratio",
        min_value=0.0,
        max_value=1.0,
        value=float(ba_yaml.get('resource_availability_ratio', 0.15)),
        step=0.01,
        format="%.2f",
        disabled=not mentoring_enabled,
    )

    st.markdown("**Underutilization Strategy**")
    uu_enabled = st.checkbox(
        "Enable Underutilization Strategy",
        value=uu_yaml.get('enabled', True),
        disabled=not mentoring_enabled,
    )
    uu_mentor_pressure_threshold = st.number_input(
        "Underutilization: Mentor Pressure Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(uu_yaml.get('mentor_pressure_threshold', 0.95)),
        step=0.01,
        format="%.2f",
        disabled=not mentoring_enabled,
    )
    uu_target_pressure_threshold = st.number_input(
        "Underutilization: Target Pressure Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(uu_yaml.get('target_pressure_threshold', 0.55)),
        step=0.01,
        format="%.2f",
        disabled=not mentoring_enabled,
    )
    uu_min_spare_hours = st.number_input(
        "Underutilization: Min Spare Hours",
        min_value=0.0,
        max_value=24.0,
        value=float(uu_yaml.get('min_spare_hours', 0.5)),
        step=0.1,
        disabled=not mentoring_enabled,
    )
    uu_max_current_activities = st.number_input(
        "Underutilization: Max Current Activities",
        min_value=1,
        max_value=100,
        value=int(uu_yaml.get('max_current_activities', 5)),
        step=1,
        disabled=not mentoring_enabled,
    )
    uu_bonus_scale = st.number_input(
        "Underutilization: Bonus Scale",
        min_value=0.0,
        max_value=50.0,
        value=float(uu_yaml.get('bonus_scale', 1.75)),
        step=0.05,
        disabled=not mentoring_enabled,
    )

    st.markdown("**Same-Day Shortage Strategy**")
    ss_enabled = st.checkbox(
        "Enable Same-Day Shortage Strategy",
        value=ss_yaml.get('enabled', False),
        disabled=not mentoring_enabled,
    )
    ss_strong_shortage_ratio = st.number_input(
        "Same-Day Shortage: Strong Shortage Ratio",
        min_value=0.0,
        max_value=10.0,
        value=float(ss_yaml.get('strong_shortage_ratio', 1.1)),
        step=0.05,
        disabled=not mentoring_enabled,
    )
    ss_strategic_quota_per_activity = st.number_input(
        "Same-Day Shortage: Strategic Quota per Activity",
        min_value=1,
        max_value=1000,
        value=int(ss_yaml.get('strategic_quota_per_activity', 10)),
        step=1,
        disabled=not mentoring_enabled,
    )
    ss_objective_bonus_multiplier = st.number_input(
        "Same-Day Shortage: Objective Bonus Multiplier",
        min_value=0.0,
        max_value=10.0,
        value=float(ss_yaml.get('objective_bonus_multiplier', 0.5)),
        step=0.05,
        disabled=not mentoring_enabled,
    )

# Update mentoring config directly in config dict
if 'mentoring' not in config:
    config['mentoring'] = dict(mentoring_yaml)
config['mentoring']['enabled'] = bool(mentoring_enabled)
config['mentoring']['severe_bottleneck_mode'] = mentoring_mode_options[selected_mentoring_mode]
if 'bottleneck_activity_strategy' not in config['mentoring']:
    config['mentoring']['bottleneck_activity_strategy'] = {}
config['mentoring']['bottleneck_activity_strategy']['enabled'] = bool(ba_enabled)
config['mentoring']['bottleneck_activity_strategy']['resource_availability_ratio'] = float(ba_resource_availability_ratio)

if 'underutilization_strategy' not in config['mentoring']:
    config['mentoring']['underutilization_strategy'] = {}
config['mentoring']['underutilization_strategy']['enabled'] = bool(uu_enabled)
config['mentoring']['underutilization_strategy']['mentor_pressure_threshold'] = float(uu_mentor_pressure_threshold)
config['mentoring']['underutilization_strategy']['target_pressure_threshold'] = float(uu_target_pressure_threshold)
config['mentoring']['underutilization_strategy']['min_spare_hours'] = float(uu_min_spare_hours)
config['mentoring']['underutilization_strategy']['max_current_activities'] = int(uu_max_current_activities)
config['mentoring']['underutilization_strategy']['bonus_scale'] = float(uu_bonus_scale)

if 'same_day_shortage_strategy' not in config['mentoring']:
    config['mentoring']['same_day_shortage_strategy'] = {}
config['mentoring']['same_day_shortage_strategy']['enabled'] = bool(ss_enabled)
config['mentoring']['same_day_shortage_strategy']['strong_shortage_ratio'] = float(ss_strong_shortage_ratio)
config['mentoring']['same_day_shortage_strategy']['strategic_quota_per_activity'] = int(ss_strategic_quota_per_activity)
config['mentoring']['same_day_shortage_strategy']['objective_bonus_multiplier'] = float(ss_objective_bonus_multiplier)

# ---- Duration Prediction ----
st.sidebar.subheader("Duration Prediction")
pred_config = config.get('duration_prediction', {})
enable_prediction = st.sidebar.checkbox(
    "Enable ML Duration Prediction",
    value=pred_config.get('enabled', False),
    help="Use trained ML model for duration estimation (falls back to experience store)",
)

# Update duration prediction config
if 'duration_prediction' not in config:
    config['duration_prediction'] = {}
config['duration_prediction']['enabled'] = enable_prediction

# ---- Experience-Based Objective Weights ----
st.sidebar.subheader("Experience-Based Objective")
if 'optimization' not in config:
    config['optimization'] = {}
objective_weights = config.get('optimization', {}).get('objective_weights', {})
with st.sidebar.expander("Objective Weights"):
    w_pressure = st.number_input("Weight: Pressure", min_value=0.0, value=float(objective_weights.get('pressure', 1.0)), step=0.1)
    w_deferral_priority = st.number_input("Weight: Deferral Priority", min_value=0.0, value=float(objective_weights.get('deferral_priority', 1.0)), step=0.1)
    w_utilization = st.number_input("Weight: Utilization", min_value=0.0, value=float(objective_weights.get('utilization', 1.0)), step=0.1)
    w_bottleneck = st.number_input("Weight: Bottleneck", min_value=0.0, value=float(objective_weights.get('bottleneck', 1.0)), step=0.1)
    w_underutilization = st.number_input("Weight: Underutilization", min_value=0.0, value=float(objective_weights.get('underutilization', 1.0)), step=0.1)
    w_shortage = st.number_input("Weight: Shortage", min_value=0.0, value=float(objective_weights.get('shortage', 1.0)), step=0.1)

config['optimization']['objective_weights'] = {
    'pressure': w_pressure,
    'deferral_priority': w_deferral_priority,
    'utilization': w_utilization,
    'bottleneck': w_bottleneck,
    'underutilization': w_underutilization,
    'shortage': w_shortage,
}

st.sidebar.subheader("Schedulers to Test")
schedulers_map = {
    "Experience Based": "experience_based",
    "Greedy": "greedy",
    "Random": "random",
}

selected_schedulers = []
for display_name, key in schedulers_map.items():
    default_on = (display_name == "Experience Based")
    if st.sidebar.checkbox(display_name, value=default_on):
        selected_schedulers.append((display_name, key))

if not selected_schedulers:
    st.warning("Please select at least one scheduler.")
    st.stop()

# --- Helper Functions ---

class StreamlitProgressCallback:
    """Callback to update Streamlit progress bar from simulation.

    Supports two modes:
    - **day-based** (when ``max_simulation_time`` is set): tracks
      simulated days so the bar shows e.g. ``42.0 / 365.0 days``.
    - **event-based** (fallback): tracks completed events.
    """

    def __init__(self, progress_bar, total_items: float, desc: str = "", unit: str = "days"):
        self.progress_bar = progress_bar
        self.total_items = total_items
        self.completed_items = 0.0
        self.desc = desc
        self.unit = unit

    def update(self, n=1):
        self.completed_items += n
        progress = min(self.completed_items / self.total_items, 1.0)
        if self.unit == "days":
            text = f"{self.desc}: {self.completed_items:.1f}/{self.total_items:.1f} {self.unit}"
        else:
            text = f"{self.desc}: {int(self.completed_items)}/{int(self.total_items)} {self.unit}"
        self.progress_bar.progress(progress, text=text)

    def close(self):
        pass  # Streamlit handles cleanup

@st.cache_resource
def load_components(_config):
    """Load simulation components and generate resources (cached).

    Leading underscore on ``_config`` prevents Streamlit from hashing
    the dict (unhashable).
    """
    try:
        exp_path = Path(_config.get('experience', {}).get('experience_store_path', 'data/experience_store.json'))
        model_path = Path(_config.get('process_model', {}).get('model_path', 'data/process_model.pkl'))

        experience_store = ExperienceStore.load(exp_path)
        with open(model_path, 'rb') as f:
            process_model = pickle.load(f)
        factory = ResourceFactory()
        resources = factory.create_resources(experience_store)
        return experience_store, process_model, resources
    except Exception as e:
        st.error(f"Error loading components: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None

# --- Main Execution ---

if st.button("🚀 Start Simulation", type="primary"):
    experience_store, process_model, resources = load_components(config)
    resources_dict = resources if isinstance(resources, dict) else {r.id: r for r in resources}

    if experience_store is None:
        st.stop()

    st.success(f"Components loaded: {len(experience_store)} profiles, {len(resources_dict)} resources")

    # Load duration predictor if enabled
    duration_predictor = None
    if enable_prediction:
        duration_predictor = load_duration_predictor(pred_config)
        if duration_predictor:
            st.info(f"🧠 Duration predictor loaded")
        else:
            st.warning("Duration predictor not found – using experience store fallback")

    simulation_status = st.empty()
    simulation_progress = st.progress(0)

    results = []
    all_stats = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- Load split info (needed for both case generation and time_converter) ----
    split_info = None
    use_test_split = config.get('case_arrival', {}).get('use_test_split', True)
    if use_test_split:
        split_info_path = Path(
            config.get('case_arrival', {}).get('split_info_path', 'data/timeline_split_info.json')
        )
        if split_info_path.exists():
            with open(split_info_path, 'r') as f:
                split_info = json.load(f)

    case_arrival_config = {
        'mode': 'probabilistic',
        'split_info_path': config['case_arrival'].get('split_info_path', 'data/timeline_split_info.json'),
        'use_test_split': use_test_split,
        'use_arrival_times_from_timeline': use_arrival_from_timeline,
    }
    # Inject split_date so CaseGenerator filters to test split
    if split_info and 'split_date' in split_info:
        case_arrival_config['split_date'] = split_info['split_date']
    if split_info and 'split_date' in split_info:
        starting_date = datetime.strptime(split_info['split_date'], '%Y-%m-%d')
    else:
        start_date_config = config.get('working_hours', {}).get('start_date', {})
        year = start_date_config.get('year', 2016)
        month = start_date_config.get('month', 1)
        day = start_date_config.get('day', 1)
        starting_date = datetime(year, month, day)
    case_generator = CaseGenerator(case_arrival_config, starting_date=starting_date)
    cases, total_timeline_events = case_generator.generate_cases()
    st.info(f"Generated **{len(cases):,}** cases with probabilistic branching"
            + (f"  ({total_timeline_events:,} estimated events)" if total_timeline_events else ""))

    # ---- Load calendars ----
    resource_calendars = None
    time_converter = None
    starting_weekday = 0
    resource_calendars = load_calendars(config.get('working_hours', {}))
    if resource_calendars:
        # When using test split, anchor time_converter to split date
        if split_info and 'split_date' in split_info:
            sd_dt = datetime.strptime(split_info['split_date'], '%Y-%m-%d')
            year, month, day = sd_dt.year, sd_dt.month, sd_dt.day
            st.info(f"⏰ Working hours enabled – time origin: split date {year}-{month:02d}-{day:02d}")
        else:
            sd = config.get('working_hours', {}).get('start_date', {})
            year, month, day = sd.get('year', 2024), sd.get('month', 1), sd.get('day', 1)
            st.info(f"⏰ Working hours enabled – simulation starts {year}-{month:02d}-{day:02d}")
        time_converter = create_default_converter(year=year, month=month, day=day)
        starting_weekday = datetime(year, month, day).weekday()

    # ---- Derive simulation end from last case arrival ----
    if cases:
        last_arrival_sec = cases[-1].arrival_time  # int seconds
        arrival_based_days = math.ceil(last_arrival_sec / SECONDS_PER_DAY)
        if max_simulation_days is not None:
            effective_days = min(arrival_based_days, max_simulation_days)
        else:
            effective_days = arrival_based_days
        # Override so the engine uses the arrival-based end
        max_simulation_days = effective_days
        config['simulation']['max_simulation_days'] = effective_days

    # ---- Show simulation info ----
    info_cols = st.columns(4)
    info_cols[0].metric("Cases", f"{len(cases):,}")
    info_cols[1].metric("Resources", f"{len(resources_dict)}")
    info_cols[2].metric("Mode", scheduling_mode_key.title())
    info_cols[3].metric("Time limit", f"{max_simulation_days}d" if max_simulation_days else "∞")

    # ---- Helper: create scheduler ----
    def create_scheduler(key: str):
        """Instantiate a scheduler from its key, matching run_simulation.py logic."""
        solver_logging_enabled = config.get('optimization', {}).get('solver_logging', {}).get('enabled', True) if config else False
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"sim_{key}_{timestamp}"
        if solver_logging_enabled:
            print(f"Solver logging enabled: logs will be saved to data/simulation_outputs/{output_name}_solver.jsonl")
        if key == "experience_based":
            return ExperienceBasedScheduler(
                duration_predictor=duration_predictor,
                config=config,
                time_converter=time_converter
            )
        elif key == "greedy":
            return GreedyScheduler(
                duration_predictor=duration_predictor,
                config=config,
                time_converter=time_converter,
            )
        elif key == "random":
            return RandomScheduler(
                duration_predictor=duration_predictor,
                config=config,
                time_converter=time_converter,
            )
        else:
            raise ValueError(f"Unknown scheduler key: {key}")

    # ---- Simulation loop ----
    for i, (display_name, sched_key) in enumerate(selected_schedulers):
        simulation_status.text(f"Running simulation with \"{display_name}\" scheduler...")
        simulation_progress.progress(0.0, text=f"{display_name}: Initializing...")

        scheduler = create_scheduler(sched_key)

        output_name = f"sim_{sched_key}_{timestamp}"
        log_writer = EventLogWriter(output_path=Path(f"data/simulation_outputs/{output_name}.csv"))

        # Metadata
        metadata = {
            "scheduler": scheduler.__class__.__name__,
            "timestamp": timestamp,
            "case_arrival_mode": case_arrival_mode,
            "process_model_type": process_model.__class__.__name__,
            "total_cases": len(cases),
            "learning_model": config.get('experience', {}).get('learning_model', 'richards'),
            "working_hours_enabled": config.get('working_hours', {}).get('enabled', False),
            "scheduling_mode": config.get('scheduling', {}).get('mode', 'immediate'),
            "mentoring_enabled": config.get('mentoring', {}).get('enabled', False),
            "simulation_start_day": starting_weekday,
            "config": config,  # Store full config for reproducibility
        }
        if hasattr(process_model, 'history_mode'):
            metadata["history_mode"] = process_model.history_mode
        log_writer.set_metadata(**metadata)

        learning_model_enum = LEARNING_MODEL_ENUM_MAP.get(
            selected_learning_model.lower(), LearningModel.RICHARDS
        )

        # Progress callback – day-based when max_simulation_time is set
        if max_simulation_time:
            total_items = max_simulation_time / SECONDS_PER_DAY  # days
            unit = "days"
        else:
            total_items = float(total_timeline_events if total_timeline_events else len(cases))
            unit = "events" if total_timeline_events else "cases"

        progress_callback = StreamlitProgressCallback(
            progress_bar=simulation_progress,
            total_items=total_items,
            desc=display_name,
            unit=unit,
        )

        # Build engine (matches run_simulation.py exactly)
        engine = SimulationEngine(
            process_model=process_model,
            scheduler=scheduler,
            experience_store=experience_store,
            resources=resources_dict,
            log_writer=log_writer,
            learning_model=learning_model_enum,
            progress_bar=progress_callback,
            resource_calendars=resource_calendars,
            time_converter=time_converter,
            config=config,
        )

        # Schedule case arrivals
        for case in cases:
            engine.schedule_case_arrival(case, case.arrival_time)

        # Run
        stats = engine.run()
        log_writer.save_with_metadata()

        # Save experience tracker
        if track_experience and 'experience_tracker' in stats and stats['experience_tracker'] is not None:
            tracker_path = Path(f"data/simulation_outputs/{output_name}_experience.csv")
            try:
                stats['experience_tracker'].save_to_csv(tracker_path)
            except Exception:
                pass

        # KPIs
        log_df = log_writer.to_dataframe()
        calculator = KPICalculator()
        kpis = calculator.compute_all(log_df, simulation_start=0.0, simulation_end=stats['simulation_time'])

        # Objective score from scheduler metadata (if available)
        solver_meta = getattr(scheduler, '_last_solver_meta', {})
        objective_score = solver_meta.get('objective_value')
        objective_score = float(objective_score) if objective_score is not None else 0.0

        # Persist run stats for Analysis & Comparison dashboard page
        _save_run_stats(Path("data/simulation_outputs"), output_name, stats, kpis, objective_score)

        result_row = {
            "Scheduler": display_name,
            "Clearance Rate": kpis.get('mean_daily_clearance_rate', 0),
            "Days Cleared": kpis.get('pct_days_fully_cleared', 0),
            "Queue Wait (h)": seconds_to_hours(kpis.get('mean_queue_wait_time', 0)),
            "Daily Backlog": kpis.get('mean_daily_backlog', 0),
            "Completion Rate": stats.get('completion_rate', 0),
            "Completed Cases": f"{stats.get('cases_completed', 0)}/{stats.get('total_cases', len(cases))}",
            "Objective Score": objective_score,
            "Sim Time (d)": stats.get('simulation_time', 0) / 24.0,
            "Log File": output_name,
        }
        results.append(result_row)
        all_stats.append((stats, kpis))

        # Update progress
        simulation_progress.progress(1.0, text=f"{display_name}: Complete ✓")

    # ---- Results ----
    simulation_status.text("✅ Ready for analysis")

    st.subheader("📊 Simulation Results")
    results_df = pd.DataFrame(results)

    st.dataframe(
        results_df.style.format({
            "Clearance Rate": "{:.2%}",
            "Days Cleared": "{:.2%}",
            "Queue Wait (h)": "{:.2f}",
            "Daily Backlog": "{:.1f}",
            "Completion Rate": "{:.2%}",
            "Objective Score": "{:.2f}",
            "Sim Time (d)": "{:.1f}",
        }),
        width='stretch',
    )

    # ---- Detailed stats per scheduler ----
    for idx, (row, (stats, kpis)) in enumerate(zip(results, all_stats)):
        with st.expander(f"📋 {row['Scheduler']} – Detailed Statistics"):
            # ---- Primary batch-scheduling KPIs ----
            st.markdown("**Batch Scheduling KPIs**")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Daily Clearance Rate", f"{kpis.get('mean_daily_clearance_rate', 0):.2%}")
            k2.metric("Days Fully Cleared", f"{kpis.get('pct_days_fully_cleared', 0):.2%}")
            k3.metric("Mean Queue Wait", f"{seconds_to_hours(kpis.get('mean_queue_wait_time', 0)):.2f}h")
            k4.metric("Mean Daily Backlog", f"{kpis.get('mean_daily_backlog', 0):.1f} tasks")

            # ---- Classic KPIs ----
            st.markdown("**Classic KPIs**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean Cycle Time", f"{kpis.get('mean_cycle_time', 0):.2f}h")
            col2.metric("Throughput", f"{kpis.get('throughput', 0):.4f} cases/h")
            col3.metric("Utilization", f"{kpis.get('mean_resource_utilization', 0):.2%}")
            col4.metric("Objective Score", f"{row['Objective Score']:.2f}")

            # ---- Case / task completion ----
            st.markdown("**Case & Task Completion**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Completed Cases", row['Completed Cases'])
            c2.metric("Completion Rate", f"{stats.get('completion_rate', 0):.2%}")
            sim_hours = stats.get('simulation_time', 0)
            c3.metric("Sim Duration", f"{sim_hours:.1f}h ({sim_hours / 24:.1f} days)")

            # ---- Unfinished work breakdown ----
            if 'unfinished_work' in stats:
                uw = stats['unfinished_work']
                st.markdown("**Unfinished Work (end of simulation)**")
                u1, u2, u3, u4 = st.columns(4)
                u1.metric("Cases Not Started", uw.get('cases_not_started', 0))
                u2.metric("Cases In Progress", uw.get('cases_in_progress', 0))
                u3.metric("Tasks Completed", f"{uw.get('tasks_completed', 0):,}")
                u4.metric("Tasks Unfinished", f"{uw.get('total_unfinished_tasks', 0):,}")

            # ---- Queue stats ----
            if 'queue_stats' in stats:
                qs = stats['queue_stats']
                st.markdown("**Queue Statistics**")
                qcol1, qcol2, qcol3 = st.columns(3)
                qcol1.metric("Tasks Still In Queues", qs.get('tasks_remaining_in_queues', 0))
                qcol2.metric("Peak Queue Length", qs.get('max_queue_length', 0))
                qcol3.metric("Peak Queue Resource", qs.get('max_queue_resource_id', 'N/A'))

            # ---- Drain & deferral stats (batch scheduling) ----
            if 'drain_stats' in stats:
                ds = stats['drain_stats']
                st.markdown("**Drain & Deferral (Batch Scheduling)**")
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Total Drained Tasks", f"{ds.get('total_drained_tasks', 0):,}",
                          help="Tasks drained from queues back to the unscheduled pool for re-planning")
                d2.metric("Days With Drain", ds.get('drain_days', 0),
                          help="Number of scheduling days where leftover tasks were drained")
                d3.metric("Total Deferred", f"{ds.get('total_deferred_tasks', 0):,}",
                          help="Tasks the solver could not assign ― deferred to the next day")
                d4.metric("Dropped (Max Deferrals)", ds.get('total_dropped_tasks', 0),
                          help="Tasks dropped after exceeding max deferral limit")
                remaining = ds.get('tasks_in_unscheduled_pool', 0)
                if remaining > 0:
                    st.info(f"{remaining} task(s) still in the unscheduled pool at simulation end")

            # ---- Overtime stats ----
            if 'overtime_stats' in stats:
                ot = stats['overtime_stats']
                st.markdown("**Overtime Statistics**")
                ocol1, ocol2 = st.columns(2)
                ocol1.metric("Total Overtime", f"{ot.get('total_overtime_hours', 0):.2f}h")
                ocol2.metric("Max Overtime", f"{ot.get('max_overtime_hours', 0):.2f}h ({ot.get('max_overtime_resource_id', 'N/A')})")

    st.success(f"✅ Simulation run `{timestamp}` completed. Go to **Analysis & Comparison** to visualize details.")
