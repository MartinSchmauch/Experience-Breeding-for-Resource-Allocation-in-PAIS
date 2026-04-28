import streamlit as st
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict
import plotly.express as px
import plotly.io as pio

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.experience import ExperienceStore
import yaml


def _patch_narwhals_polars_detection() -> None:
    """Guard Plotly/Narwhals against broken optional polars imports.

    Some environments end up with a partially initialized `polars` module,
    which can crash Plotly figure creation when Narwhals probes for polars
    native objects. In that case we safely treat objects as non-polars.
    """
    try:
        import narwhals._native as nw_native
        import narwhals.translate as nw_translate
    except Exception:
        return

    original_native = getattr(nw_native, 'is_native_polars', None)
    if original_native is not None:
        def _safe_native(obj):
            try:
                return original_native(obj)
            except Exception:
                return False
        nw_native.is_native_polars = _safe_native

    original_translate = getattr(nw_translate, 'is_native_polars', None)
    if original_translate is not None:
        def _safe_translate(obj):
            try:
                return original_translate(obj)
            except Exception:
                return False
        nw_translate.is_native_polars = _safe_translate


_patch_narwhals_polars_detection()

st.set_page_config(page_title="Capability Overview", page_icon="🔍", layout="wide")

# --- Display mode (light mode support) ---
light_mode = st.sidebar.toggle("Light mode", value=False, help="White background, black text, and white plot backgrounds")

if light_mode:
    pio.templates.default = "plotly_white"
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #ffffff;
            color: #000000;
        }
        [data-testid="stSidebar"] {
            background-color: #ffffff;
        }
        [data-testid="stHeader"] {
            background-color: #ffffff;
        }
        h1, h2, h3, h4, h5, h6, p, li, label, span, div {
            color: #000000;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_plot(fig):
    """Render Plotly charts with consistent theming across display modes."""
    if light_mode:
        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font={"color": "black"},
        )
        st.plotly_chart(fig, width='stretch', theme=None)
    else:
        st.plotly_chart(fig, width='stretch')

st.title("🔍 Capability & Requirements Overview")

st.markdown("""
This page helps diagnose capability matching issues by showing:
- **Activity Requirements**: What capability levels are required for each activity
- **Resource Capabilities**: What capability levels each resource currently has (config-driven mapping)
- **Learning Opportunities**: Where resources have the best chance to learn
- **Resource Schedules**: Work hours and planned absences per resource
""")

# --- Load Data ---
EXPERIENCE_STORE_PATH = Path("data/experience_store.json")
ACTIVITY_REQ_PATH = Path("config/activity_requirements.yaml")
SIM_CONFIG_PATH = Path("config/simulation_config.yaml")
CALENDARS_PATH = Path("data/calendars.json")
TIMELINE_PATH = Path("data/timeline.csv")

if not EXPERIENCE_STORE_PATH.exists():
    st.error(f"Experience store not found: {EXPERIENCE_STORE_PATH}")
    st.info("Run `python scripts/initialize_simulation.py` first")
    st.stop()

if not ACTIVITY_REQ_PATH.exists():
    st.error(f"Activity requirements not found: {ACTIVITY_REQ_PATH}")
    st.stop()

# Load experience store
experience_store = ExperienceStore.load(EXPERIENCE_STORE_PATH)

# Load activity requirements
with open(ACTIVITY_REQ_PATH, 'r') as f:
    activity_config = yaml.safe_load(f)
    activity_requirements: Dict[str, float] = activity_config.get('activity_requirements', {})
    default_requirement = activity_config.get('default_requirement', {}).get('level', 50.0)

# Load simulation config for capability mapping
capability_mapping = [
    {'count': 0, 'level': 10.0},
    {'count': 20, 'level': 32.0},
    {'count': 50, 'level': 55.0},
    {'count': 100, 'level': 82.0},
    {'count': 150, 'level': 92.0},
    {'count': 200, 'level': 95.0},
]
if SIM_CONFIG_PATH.exists():
    with open(SIM_CONFIG_PATH, 'r') as f:
        sim_config = yaml.safe_load(f)
    exp_cfg = sim_config.get('experience', {})
    capability_mapping = exp_cfg.get('capability_mapping', capability_mapping)

# Load calendars
calendars = {}
if CALENDARS_PATH.exists():
    with open(CALENDARS_PATH, 'r') as f:
        calendars = json.load(f)

# Load timeline for activity volume analysis
timeline_df = None
if TIMELINE_PATH.exists():
    timeline_df = pd.read_csv(TIMELINE_PATH)

# --- Section 1: Activity Requirements ---
st.header("📋 Activity Requirements")

st.markdown("""
These are the **minimum capability levels required** for each activity (from `config/activity_requirements.yaml`).
Resources must have capability levels **greater than or equal** to these values.
""")

# Convert to DataFrame
req_data = []
for activity, level in activity_requirements.items():
    req_data.append({
        'Activity': activity,
        'Minimum Level': level
        })

if req_data:
    req_df = pd.DataFrame(req_data)
    
    # Show statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Activities", len(activity_requirements))
    with col2:
        st.metric("Default Level", f"{default_requirement:.0f}")
    with col3:
        avg_req = req_df['Minimum Level'].mean()
        st.metric("Average Required Level", f"{avg_req:.1f}")
    
    # Show table
    st.dataframe(
        req_df.style.background_gradient(subset=['Minimum Level'], cmap='YlOrRd'),
        width='stretch',
        height=400
    )
    
    # Show distribution
    fig = px.histogram(req_df, x='Minimum Level', nbins=20, 
                      title="Distribution of Required Capability Levels")
    render_plot(fig)
else:
    st.warning("No activity requirements found")

# --- Section 2: Resource Capabilities ---
st.header("👥 Resource Capabilities")

# Show the capability mapping thresholds from config
mapping_desc = " → ".join([f"{t['count']} obs → **{t['level']:.0f}**" for t in sorted(capability_mapping, key=lambda x: x['count'])])
st.markdown(f"""
These are the **current capability levels** of each resource (from `data/experience_store.json`).
Levels are mapped from experience count using the thresholds defined in `config/simulation_config.yaml`:

{mapping_desc}

Values between thresholds are assigned the level of the highest threshold not exceeded.
""")


resource_capabilities = {}
for profile in experience_store._profiles.values():
    resource_id = profile.resource_id
    activity = profile.activity_name
    level = profile.experience_level
    
    if resource_id not in resource_capabilities:
        resource_capabilities[resource_id] = {}
    resource_capabilities[resource_id][activity] = {
        'level': level,
        'count': profile.count
    }

# Convert to DataFrame
cap_data = []
for resource_id, capabilities in resource_capabilities.items():
    for activity, info in capabilities.items():
        cap_data.append({
            'Resource': resource_id,
            'Activity': activity,
            'Capability Level': info['level'],
            'Experience Count': info['count']
        })

if cap_data:
    cap_df = pd.DataFrame(cap_data)
    
    # Show statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Resources", len(resource_capabilities))
    with col2:
        avg_cap = cap_df['Capability Level'].mean()
        st.metric("Average Capability Level", f"{avg_cap:.1f}")
    with col3:
        total_caps = len(cap_df)
        st.metric("Total Capabilities", total_caps)
    with col4:
        avg_per_resource = total_caps / len(resource_capabilities)
        st.metric("Avg Capabilities/Resource", f"{avg_per_resource:.1f}")
    
    # Resource selector
    selected_resources = st.multiselect(
        "Filter by Resource",
        options=sorted(cap_df['Resource'].unique()),
        default=sorted(cap_df['Resource'].unique())[:5]  # Show first 5
    )
    
    if selected_resources:
        filtered_df = cap_df[cap_df['Resource'].isin(selected_resources)]
        st.dataframe(
            filtered_df.style.background_gradient(subset=['Capability Level'], cmap='YlGn'),
            width='stretch',
            height=400
        )
        
        # Show distribution
        fig2 = px.histogram(cap_df, x='Capability Level', nbins=20,
                           title="Distribution of Resource Capability Levels")
        render_plot(fig2)
    else:
        st.info("Select at least one resource to view details")
else:
    st.warning("No resource capabilities found")

# --- Section 3: Task Volume Timeline ---
st.header("📈 Task Volume per Activity Over Time")

st.markdown("""
This view shows the number of tasks per day for each activity across the full historical timeline
from `data/timeline.csv`.
""")

if timeline_df is None:
    st.warning("Timeline not found: `data/timeline.csv`. Run `python scripts/initialize_simulation.py` first.")
else:
    required_cols = {
        'activity_name',
        'start_timestamp',
        'segment_count',
        'resource_count',
        'active_ratio',
    }
    missing_cols = sorted(list(required_cols - set(timeline_df.columns)))
    if missing_cols:
        st.error(f"Timeline is missing required columns: {missing_cols}")
    else:
        timeline_plot_df = timeline_df.copy()
        timeline_plot_df['start_timestamp'] = pd.to_datetime(
            timeline_plot_df['start_timestamp'], errors='coerce'
        )
        timeline_plot_df = timeline_plot_df.dropna(subset=['start_timestamp'])
        timeline_plot_df['date'] = timeline_plot_df['start_timestamp'].dt.date

        daily_activity_counts = (
            timeline_plot_df
            .groupby(['date', 'activity_name'])
            .size()
            .reset_index(name='task_count')
            .sort_values(['date', 'activity_name'])
        )

        all_activities = sorted(daily_activity_counts['activity_name'].unique())
        default_activities = all_activities[:min(10, len(all_activities))]
        selected_activities = st.multiselect(
            "Filter activities",
            options=all_activities,
            default=default_activities,
            key="timeline_activity_filter",
        )

        if selected_activities:
            plot_df = daily_activity_counts[
                daily_activity_counts['activity_name'].isin(selected_activities)
            ]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Timeline Days", plot_df['date'].nunique())
            with col2:
                st.metric("Activities Shown", len(selected_activities))
            with col3:
                st.metric("Total Task Occurrences", int(plot_df['task_count'].sum()))

            fig_timeline = px.line(
                plot_df,
                x='date',
                y='task_count',
                color='activity_name',
                markers=False,
                title="Tasks per Day by Activity",
                labels={
                    'date': 'Date',
                    'task_count': 'Tasks per Day',
                    'activity_name': 'Activity',
                },
            )
            fig_timeline.update_layout(height=520)
            render_plot(fig_timeline)
        else:
            st.info("Select at least one activity to display the timeline.")

        with st.expander("What do segment_count, resource_count, and active_ratio mean?"):
            st.markdown("""
            These columns are generated from lifecycle-based preprocessing in `EventLogReader.preprocess_for_simulation`:

            - **segment_count**: number of active work segments for a task occurrence
              (task split by suspend/resume cycles).
            - **resource_count**: number of distinct resources involved in that task occurrence.
            - **active_ratio**: active service time divided by elapsed time between start and completion.
            """)

# --- Section 4: Capability Value by Workload ---
st.header("💎 Capability Value by Activity Workload")

st.markdown("""
This view combines **resource capabilities** with **activity workload demand** from the timeline.

- **Avg Tasks/Day**: average number of tasks per activity per calendar day (including zero-task days)
- **Avg Duration (h)**: average task duration for the activity
- **Avg Workload/Day (h)** = Avg Tasks/Day × Avg Duration (h)
- **Capability Coverage** = `min(resource_level / required_level, 1.0)`
- **Capability Value (h/day)** = Capability Coverage × Avg Workload/Day (h)

Higher value means a resource is more valuable for high-demand activities.
""")

if timeline_df is None:
    st.warning("Timeline not found: `data/timeline.csv`. Cannot compute workload-weighted capability value.")
elif not cap_data:
    st.warning("No resource capabilities found. Cannot compute capability value.")
else:
    timeline_val_df = timeline_df.copy()
    timeline_val_df['start_timestamp'] = pd.to_datetime(
        timeline_val_df['start_timestamp'], errors='coerce'
    )
    timeline_val_df = timeline_val_df.dropna(subset=['start_timestamp', 'activity_name'])

    # Resolve duration in hours from whichever duration column exists.
    if 'duration_seconds' in timeline_val_df.columns:
        timeline_val_df['duration_h'] = pd.to_numeric(
            timeline_val_df['duration_seconds'], errors='coerce'
        ) / 3600.0
    elif 'duration_hours' in timeline_val_df.columns:
        timeline_val_df['duration_h'] = pd.to_numeric(
            timeline_val_df['duration_hours'], errors='coerce'
        )
    else:
        timeline_val_df['duration_h'] = np.nan

    timeline_val_df['date'] = timeline_val_df['start_timestamp'].dt.normalize()

    # Average tasks/day per activity over all timeline days (including zero days).
    all_days = pd.date_range(
        timeline_val_df['date'].min(),
        timeline_val_df['date'].max(),
        freq='D'
    )
    all_acts = sorted(timeline_val_df['activity_name'].dropna().unique())
    day_activity_grid = pd.MultiIndex.from_product(
        [all_days, all_acts], names=['date', 'activity_name']
    )

    day_counts = (
        timeline_val_df
        .groupby(['date', 'activity_name'])
        .size()
        .reindex(day_activity_grid, fill_value=0)
        .reset_index(name='task_count')
    )

    avg_tasks_per_day = (
        day_counts
        .groupby('activity_name')['task_count']
        .mean()
        .to_dict()
    )

    avg_duration_h = (
        timeline_val_df
        .groupby('activity_name')['duration_h']
        .mean()
        .fillna(0.0)
        .to_dict()
    )

    activity_workload_rows = []
    for act in all_acts:
        avg_tasks = float(avg_tasks_per_day.get(act, 0.0))
        avg_dur = float(avg_duration_h.get(act, 0.0))
        activity_workload_rows.append({
            'Activity': act,
            'Avg Tasks/Day': avg_tasks,
            'Avg Duration (h)': avg_dur,
            'Avg Workload/Day (h)': avg_tasks * avg_dur,
        })
    activity_workload_df = pd.DataFrame(activity_workload_rows)

    # Capability value per (resource, activity)
    cap_value_rows = []
    for resource_id, capabilities in resource_capabilities.items():
        for activity, info in capabilities.items():
            req_level = float(activity_requirements.get(activity, default_requirement))
            level = float(info['level'])
            count = int(info['count'])
            coverage = min(level / req_level, 1.0) if req_level > 0 else 1.0
            avg_workload_hpd = float(
                activity_workload_df.loc[
                    activity_workload_df['Activity'] == activity,
                    'Avg Workload/Day (h)'
                ].iloc[0]
            ) if activity in set(activity_workload_df['Activity']) else 0.0

            cap_value_rows.append({
                'Resource': resource_id,
                'Activity': activity,
                'Capability Level': level,
                'Required Level': req_level,
                'Coverage': coverage,
                'Experience Count': count,
                'Avg Workload/Day (h)': avg_workload_hpd,
                'Capability Value (h/day)': coverage * avg_workload_hpd,
            })

    if cap_value_rows:
        cap_value_df = pd.DataFrame(cap_value_rows)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Resources", cap_value_df['Resource'].nunique())
        with col2:
            st.metric("Activities in Timeline", activity_workload_df['Activity'].nunique())
        with col3:
            st.metric(
                "Mean Capability Value (h/day)",
                f"{cap_value_df['Capability Value (h/day)'].mean():.2f}"
            )

        with st.expander("Activity Demand Baseline (from timeline)"):
            st.dataframe(
                activity_workload_df.sort_values('Avg Workload/Day (h)', ascending=False),
                width='stretch',
                height=320,
            )

        # Heatmap: resource x activity capability value
        st.subheader("Capability Value Heatmap (Resource × Activity)")
        top_n_activities = st.slider(
            "Top activities by workload to show",
            min_value=5,
            max_value=min(40, len(activity_workload_df)) if len(activity_workload_df) > 0 else 5,
            value=min(20, len(activity_workload_df)) if len(activity_workload_df) > 0 else 5,
            key="cap_value_top_n_activities",
        )

        top_activities = activity_workload_df.nlargest(top_n_activities, 'Avg Workload/Day (h)')['Activity'].tolist()

        # Build a full resource x activity matrix so missing capabilities are visible.
        # Non-capable cells are mapped to negative values to render them red.
        heat_rows = []
        all_resources_sorted = sorted(resource_capabilities.keys())
        workload_lookup = {
            r['Activity']: float(r['Avg Workload/Day (h)'])
            for _, r in activity_workload_df.iterrows()
        }

        for res in all_resources_sorted:
            caps = resource_capabilities.get(res, {})
            for act in top_activities:
                req_level = float(activity_requirements.get(act, default_requirement))
                res_level = float(caps.get(act, {}).get('level', 10.0))
                coverage = min(res_level / req_level, 1.0) if req_level > 0 else 1.0
                workload_hpd = workload_lookup.get(act, 0.0)
                cap_value = coverage * workload_hpd
                is_capable = res_level >= req_level

                display_value = cap_value if is_capable else -workload_hpd
                heat_rows.append({
                    'Resource': res,
                    'Activity': act,
                    'Display Value': display_value,
                    'Capability Value (h/day)': cap_value,
                    'Is Capable': is_capable,
                })

        heat_df = pd.DataFrame(heat_rows)
        pivot_value = heat_df.pivot(index='Resource', columns='Activity', values='Display Value')

        z_abs_max = float(np.abs(pivot_value.to_numpy()).max()) if not pivot_value.empty else 1.0
        z_abs_max = max(z_abs_max, 1.0)

        fig_value_heat = px.imshow(
            pivot_value,
            color_continuous_scale='RdYlGn',
            zmin=-z_abs_max,
            zmax=z_abs_max,
            title='Workload-Weighted Capability Value (red = not yet capable)',
            labels={'color': 'Value Score'},
            aspect='auto',
        )
        fig_value_heat.update_layout(height=max(360, len(pivot_value) * 24))
        render_plot(fig_value_heat)

        # Per-resource total portfolio value
        st.subheader("Total Capability Portfolio Value per Resource")
        portfolio_df = (
            cap_value_df
            .groupby('Resource', as_index=False)['Capability Value (h/day)']
            .sum()
            .sort_values('Capability Value (h/day)', ascending=False)
        )
        fig_portfolio = px.bar(
            portfolio_df,
            x='Resource',
            y='Capability Value (h/day)',
            color='Capability Value (h/day)',
            color_continuous_scale='YlGnBu',
            title='Resource Portfolio Value (sum across activities)',
        )
        fig_portfolio.update_layout(height=420)
        render_plot(fig_portfolio)

        # Drill-down for one resource
        st.subheader("Resource Drill-Down")
        selected_resource_for_value = st.selectbox(
            "Select resource",
            options=sorted(cap_value_df['Resource'].unique()),
            key='cap_value_resource_selector',
        )

        res_value_df = (
            cap_value_df[cap_value_df['Resource'] == selected_resource_for_value]
            .sort_values('Capability Value (h/day)', ascending=False)
        )

        fig_res_value = px.bar(
            res_value_df,
            x='Activity',
            y='Capability Value (h/day)',
            color='Coverage',
            color_continuous_scale='Viridis',
            hover_data=['Capability Level', 'Required Level', 'Avg Workload/Day (h)', 'Experience Count'],
            title=f'Capability Value by Activity — {selected_resource_for_value}',
        )
        fig_res_value.update_layout(height=420, xaxis_tickangle=-45)
        render_plot(fig_res_value)
    else:
        st.info("No capability-value rows could be computed from current data.")

# --- Section 5: Capability Matching Analysis ---
st.header("🔍 Capability Matching Analysis")

st.markdown("""
This shows which resources can perform which activities based on current capability levels vs. requirements.
""")

if req_data and cap_data:
    # Build matching matrix
    matching_data = []
    
    for activity, req_level in activity_requirements.items():
        # Find resources with this capability at sufficient level
        capable_resources = []
        for resource_id, capabilities in resource_capabilities.items():
            if activity in capabilities:
                resource_level = capabilities[activity]['level']
                if resource_level >= req_level:
                    capable_resources.append(resource_id)
        
        matching_data.append({
            'Activity': activity,
            'Required Level': req_level,
            'Capable Resources': len(capable_resources),
            'Resource IDs': ', '.join(capable_resources[:5]) + ('...' if len(capable_resources) > 5 else '')
        })
    
    match_df = pd.DataFrame(matching_data)
    
    # Show summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        no_resources = len(match_df[match_df['Capable Resources'] == 0])
        st.metric("Activities with NO Capable Resources", no_resources, 
                 delta=None, delta_color="inverse")
    with col2:
        avg_capable = match_df['Capable Resources'].mean()
        st.metric("Avg Capable Resources/Activity", f"{avg_capable:.1f}")
    with col3:
        max_capable = match_df['Capable Resources'].max()
        st.metric("Max Capable Resources", int(max_capable))
    
    # Show problematic activities
    st.subheader("⚠️ Activities with NO Capable Resources")
    problem_df = match_df[match_df['Capable Resources'] == 0]
    
    if not problem_df.empty:
        st.error(f"Found {len(problem_df)} activities that NO resource can perform!")
        st.dataframe(
            problem_df[['Activity', 'Required Level']],
            width='stretch',
        )
        
        st.markdown("**Recommended Actions:**")
        st.markdown("""
        1. **Lower requirements** in `config/activity_requirements.yaml`
        2. **Increase resource capabilities** by running more historical data through initialization
        3. **Add synthetic capabilities** to resources in `data/simulation_resources.json`
        """)
    else:
        st.success("✅ All activities have at least one capable resource!")
    
    # Show full matching table
    st.subheader("Full Matching Table")
    st.dataframe(
        match_df.style.background_gradient(subset=['Capable Resources'], cmap='RdYlGn'),
        width='stretch',
        height=400
    )
else:
    st.warning("Need both requirements and capabilities to perform matching analysis")

# --- Section 6: Learning Opportunities ---
st.header("📈 Learning Opportunities (Chance to Learn)")

st.markdown("""
The **learning score** measures how much a resource can develop by performing a given activity.
It uses the same model as the experience-based scheduler:

- **gap_ratio** = resource_level / required_level
- **Sweet spot** (gap_ratio 0.5–1.0): maximum learning → score 1.0
- **Too far below** (gap_ratio < 0.5): limited learning → score = gap_ratio
- **Diminishing returns** (gap_ratio 1.0–1.5): → score = 1.5 − gap_ratio
- **Already mastered** (gap_ratio ≥ 1.5): → score 0.0
- Modulated by **1 / (1 + count / 20)** — fewer reps means more room to learn
""")

if req_data and cap_data:
    def compute_learning_score(resource_level: float, required_level: float, count: int) -> float:
        """Compute learning opportunity score (same logic as experience_based scheduler)."""
        if required_level > 0:
            gap_ratio = resource_level / required_level
        else:
            gap_ratio = 1.5

        if gap_ratio < 0.5:
            v_learning = gap_ratio
        elif gap_ratio < 1.0:
            v_learning = 1.0
        elif gap_ratio < 1.5:
            v_learning = max(0.0, 1.5 - gap_ratio)
        else:
            v_learning = 0.0

        learn_modifier = 1.0 / (1.0 + count / 20.0)
        return v_learning * learn_modifier

    learn_data = []
    for resource_id, capabilities in resource_capabilities.items():
        for activity, req_level in activity_requirements.items():
            res_info = capabilities.get(activity, {'level': 10.0, 'count': 0})
            score = compute_learning_score(res_info['level'], req_level, res_info['count'])
            learn_data.append({
                'Resource': resource_id,
                'Activity': activity,
                'Resource Level': res_info['level'],
                'Required Level': req_level,
                'Experience Count': res_info['count'],
                'Learning Score': round(score, 3),
            })

    learn_df = pd.DataFrame(learn_data)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_learn = learn_df['Learning Score'].mean()
        st.metric("Avg Learning Score", f"{avg_learn:.3f}")
    with col2:
        high_learn = len(learn_df[learn_df['Learning Score'] >= 0.5])
        st.metric("High-Opportunity Pairs (≥0.5)", high_learn)
    with col3:
        zero_learn = len(learn_df[learn_df['Learning Score'] == 0])
        st.metric("Zero Learning Pairs", zero_learn)
    with col4:
        top_resource = learn_df.groupby('Resource')['Learning Score'].mean().idxmax()
        st.metric("Top Learner", top_resource)

    # Top learning opportunities
    st.subheader("🏆 Top Learning Opportunities")
    top_n = st.slider("Show top N pairs", 5, 50, 20, key="learn_top_n")
    top_learn = learn_df.nlargest(top_n, 'Learning Score')
    st.dataframe(
        top_learn.style.background_gradient(subset=['Learning Score'], cmap='YlGn'),
        width='stretch',
    )

    # Learning heatmap: resource × activity
    st.subheader("Learning Heatmap")
    import plotly.express as px
    pivot_learn = learn_df.pivot_table(index='Resource', columns='Activity', values='Learning Score', aggfunc='mean')
    fig_heat = px.imshow(
        pivot_learn,
        color_continuous_scale='YlGn',
        title="Learning Score by Resource × Activity",
        labels=dict(color="Learning Score"),
        aspect='auto',
    )
    fig_heat.update_layout(height=max(300, len(pivot_learn) * 25))
    render_plot(fig_heat)

    # Per-resource average learning
    st.subheader("Average Learning Score per Resource")
    resource_learn = learn_df.groupby('Resource')['Learning Score'].mean().sort_values(ascending=False).reset_index()
    resource_learn.columns = ['Resource', 'Avg Learning Score']
    fig_bar = px.bar(resource_learn, x='Resource', y='Avg Learning Score',
                     title="Average Learning Score per Resource",
                     color='Avg Learning Score', color_continuous_scale='YlGn')
    render_plot(fig_bar)
else:
    st.warning("Need both requirements and capabilities to compute learning scores")

# --- Section 7: Resource Work Schedules ---
st.header("📅 Resource Work Schedules")

st.markdown("""
Resource availability from `data/calendars.json` — work hours (Mon–Fri) and planned absences.
""")

if calendars:
    WEEKDAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    schedule_rows = []
    absence_rows = []
    for res_id, cal in calendars.items():
        sched = cal.get('schedule', {})
        weekday_hours = sched.get('weekday_hours', {})

        # Build schedule string
        hours_parts = []
        total_weekly = 0.0
        for day_idx in range(7):
            hours = weekday_hours.get(str(day_idx))
            if hours:
                start_h, end_h = hours
                hours_parts.append(f"{WEEKDAY_NAMES[day_idx]} {start_h}:00–{end_h}:00")
                total_weekly += (end_h - start_h)

        # Count activities & capabilities
        res_caps = resource_capabilities.get(res_id, {})
        n_activities = len(res_caps)
        avg_level = np.mean([v['level'] for v in res_caps.values()]) if res_caps else 0.0

        absences = cal.get('absences', [])
        total_absence_days = 0
        vacation_days = 0
        sick_days = 0
        for ab in absences:
            start = pd.Timestamp(ab['start_date'])
            end = pd.Timestamp(ab['end_date'])
            days = max(1, (end - start).days)
            total_absence_days += days
            if ab.get('absence_type') == 'vacation':
                vacation_days += days
            elif ab.get('absence_type') == 'sick_leave':
                sick_days += days

        schedule_rows.append({
            'Resource': res_id,
            'Weekly Hours': total_weekly,
            'Schedule': ' | '.join(hours_parts) if hours_parts else 'Not set',
            'Activities Known': n_activities,
            'Avg Capability': round(avg_level, 1),
            'Vacation Days': vacation_days,
            'Sick Days': sick_days,
            'Total Absence Days': total_absence_days,
        })

        for ab in absences:
            absence_rows.append({
                'Resource': res_id,
                'Start': ab['start_date'][:10],
                'End': ab['end_date'][:10],
                'Type': ab.get('absence_type', 'unknown'),
                'Description': ab.get('description', ''),
            })

    sched_df = pd.DataFrame(schedule_rows).sort_values('Resource')
    absence_df = pd.DataFrame(absence_rows)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Resources with Calendar", len(calendars))
    with col2:
        avg_weekly = sched_df['Weekly Hours'].mean()
        st.metric("Avg Weekly Hours", f"{avg_weekly:.0f}h")
    with col3:
        avg_absence = sched_df['Total Absence Days'].mean()
        st.metric("Avg Absence Days", f"{avg_absence:.0f}")
    with col4:
        avg_vac = sched_df['Vacation Days'].mean()
        st.metric("Avg Vacation Days", f"{avg_vac:.0f}")

    # Resource overview table
    st.subheader("Resource Overview")
    st.dataframe(
        sched_df.style.background_gradient(subset=['Avg Capability'], cmap='YlGn')
                      .background_gradient(subset=['Total Absence Days'], cmap='OrRd'),
        width='stretch',
        height=400,
    )

    # Absence timeline
    if not absence_df.empty:
        st.subheader("📋 Absence Calendar")
        selected_type = st.multiselect(
            "Filter by absence type",
            options=sorted(absence_df['Type'].unique()),
            default=sorted(absence_df['Type'].unique()),
            key="absence_type_filter",
        )
        filtered_abs = absence_df[absence_df['Type'].isin(selected_type)] if selected_type else absence_df

        import plotly.express as px
        fig_timeline = px.timeline(
            filtered_abs,
            x_start='Start',
            x_end='End',
            y='Resource',
            color='Type',
            hover_data=['Description'],
            title="Resource Absences Timeline",
            color_discrete_map={'vacation': '#1f77b4', 'sick_leave': '#d62728'},
        )
        fig_timeline.update_layout(height=max(300, len(filtered_abs['Resource'].unique()) * 30))
        render_plot(fig_timeline)

        # --- Activity Bottleneck Timeline ---
        st.subheader("🚨 Activity Bottleneck Timeline")
        st.markdown("""
        Shows periods where **no capable resource is available** for an activity
        (all resources that meet the capability requirement are absent).
        """)

        # Build capable-resource sets per activity
        capable_per_activity: dict[str, set[str]] = {}
        for activity, req_level in activity_requirements.items():
            capable = set()
            for res_id, caps in resource_capabilities.items():
                if activity in caps and caps[activity]['level'] >= req_level:
                    capable.add(res_id)
            capable_per_activity[activity] = capable

        # Build per-resource absence day sets
        from datetime import timedelta
        resource_absence_days: dict[str, set] = {}
        for res_id, cal in calendars.items():
            days_set: set = set()
            for ab in cal.get('absences', []):
                start = pd.Timestamp(ab['start_date']).normalize()
                end = pd.Timestamp(ab['end_date']).normalize()
                day = start
                while day <= end:
                    days_set.add(day)
                    day += timedelta(days=1)
            resource_absence_days[res_id] = days_set

        # Determine date range from absences
        all_absence_dates = set()
        for days_set in resource_absence_days.values():
            all_absence_dates |= days_set

        if all_absence_dates:
            date_min = min(all_absence_dates)
            date_max = max(all_absence_dates)
            all_days = pd.date_range(date_min, date_max, freq='D')

            # Find bottleneck periods per activity
            bottleneck_rows = []
            for activity, capable_set in capable_per_activity.items():
                if not capable_set:
                    # No resource can ever do this → permanent bottleneck, skip (already shown above)
                    continue

                # For each day check if ALL capable resources are absent
                bottleneck_start = None
                for day in all_days:
                    all_absent = all(
                        day in resource_absence_days.get(res_id, set())
                        for res_id in capable_set
                    )
                    if all_absent:
                        if bottleneck_start is None:
                            bottleneck_start = day
                    else:
                        if bottleneck_start is not None:
                            bottleneck_rows.append({
                                'Activity': activity,
                                'Start': bottleneck_start.strftime('%Y-%m-%d'),
                                'End': (day - timedelta(days=1)).strftime('%Y-%m-%d'),
                                'Days': (day - bottleneck_start).days,
                                'Capable Resources': len(capable_set),
                            })
                            bottleneck_start = None
                # Close open range at end
                if bottleneck_start is not None:
                    bottleneck_rows.append({
                        'Activity': activity,
                        'Start': bottleneck_start.strftime('%Y-%m-%d'),
                        'End': all_days[-1].strftime('%Y-%m-%d'),
                        'Days': (all_days[-1] - bottleneck_start).days + 1,
                        'Capable Resources': len(capable_set),
                    })

            if bottleneck_rows:
                bn_df = pd.DataFrame(bottleneck_rows)

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Bottleneck Periods", len(bn_df))
                with col_b:
                    affected_acts = bn_df['Activity'].nunique()
                    st.metric("Activities Affected", affected_acts)
                with col_c:
                    total_bn_days = bn_df['Days'].sum()
                    st.metric("Total Bottleneck-Days", total_bn_days)

                fig_bn = px.timeline(
                    bn_df,
                    x_start='Start',
                    x_end='End',
                    y='Activity',
                    color_discrete_sequence=['#d62728'],  # red
                    hover_data=['Days', 'Capable Resources'],
                    title="Activity Bottleneck Periods (no capable resource available)",
                )
                fig_bn.update_layout(
                    height=max(300, bn_df['Activity'].nunique() * 40),
                    showlegend=False,
                )
                render_plot(fig_bn)

                with st.expander("Bottleneck Details"):
                    st.dataframe(
                        bn_df.sort_values(['Activity', 'Start']),
                        width='stretch',
                    )
            else:
                st.success("✅ No bottleneck periods found — every activity has at least one capable resource available at all times.")
        else:
            st.info("No absence dates to analyse for bottlenecks.")

        # --- Mean Activity Fitness Score Timeline ---
        st.subheader("📊 Mean Activity Fitness Score Over Time")
        st.markdown("""
        Shows the **mean fitness score** per activity across **all** resources for each day.
        The fitness score is computed the same way as in the simulation scheduler:

        `fitness(r, a) = clamp(capability_level(r, a) / required_level(a),  0,  1)`

        Absent resources contribute a **fitness of 0** for that day, so the mean drops
        when capable resources are on leave.
        """)

        # Precompute static fitness scores per (resource, activity)
        # fitness = clamp(cap_level / req_level, 0, 1)
        all_resources_set = set(resource_capabilities.keys()) | set(calendars.keys())
        static_fitness: dict[tuple[str, str], float] = {}
        for activity, req_level in activity_requirements.items():
            for res_id in all_resources_set:
                caps = resource_capabilities.get(res_id, {})
                cap_level = caps.get(activity, {}).get('level', 10.0)  # default lowest level
                raw = cap_level / req_level if req_level > 0 else 1.0
                static_fitness[(res_id, activity)] = max(0.0, min(1.0, raw))

        # Build date range from calendar data (full year span)
        cal_dates = set()
        for cal in calendars.values():
            for ab in cal.get('absences', []):
                cal_dates.add(pd.Timestamp(ab['start_date']).normalize())
                cal_dates.add(pd.Timestamp(ab['end_date']).normalize())
        if cal_dates:
            cal_min = min(cal_dates)
            cal_max = max(cal_dates)
            # Extend to full months
            cal_min = cal_min.replace(day=1)
            fitness_days = pd.date_range(cal_min, cal_max, freq='D')

            fitness_rows = []
            for day in fitness_days:
                # Skip weekends (no work scheduled typically)
                if day.weekday() >= 5:
                    continue
                for activity in activity_requirements:
                    scores = []
                    for res_id in all_resources_set:
                        key = (res_id, activity)
                        if key not in static_fitness:
                            continue
                        # Absent resources contribute 0 fitness for that day
                        if day in resource_absence_days.get(res_id, set()):
                            scores.append(0.0)
                        else:
                            scores.append(static_fitness[key])
                    available_count = sum(
                        1 for res_id in all_resources_set
                        if (res_id, activity) in static_fitness
                        and day not in resource_absence_days.get(res_id, set())
                    )
                    if scores:
                        fitness_rows.append({
                            'Date': day,
                            'Activity': activity,
                            'Mean Fitness': np.mean(scores),
                            'Available Resources': available_count,
                        })
                    else:
                        fitness_rows.append({
                            'Date': day,
                            'Activity': activity,
                            'Mean Fitness': 0.0,
                            'Available Resources': 0,
                        })

            if fitness_rows:
                fitness_df = pd.DataFrame(fitness_rows)

                import plotly.express as px
                fig_fitness = px.line(
                    fitness_df,
                    x='Date',
                    y='Mean Fitness',
                    color='Activity',
                    hover_data=['Available Resources'],
                    title="Mean Activity Fitness Score Over Time (weekdays, absent resources = 0)",
                )
                fig_fitness.update_layout(
                    height=500,
                    yaxis=dict(range=[0, 1.05], title="Mean Fitness Score"),
                    xaxis_title="Date",
                    legend_title="Activity",
                )
                render_plot(fig_fitness)

                # Summary stats
                avg_fitness_per_act = fitness_df.groupby('Activity')['Mean Fitness'].agg(['mean', 'min']).reset_index()
                avg_fitness_per_act.columns = ['Activity', 'Avg Fitness', 'Min Fitness']
                avg_fitness_per_act = avg_fitness_per_act.sort_values('Avg Fitness')

                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    overall_avg = fitness_df['Mean Fitness'].mean()
                    st.metric("Overall Avg Fitness", f"{overall_avg:.3f}")
                with col_f2:
                    worst_act = avg_fitness_per_act.iloc[0]
                    st.metric("Lowest Avg Fitness Activity", f"{worst_act['Activity']}", delta=f"{worst_act['Avg Fitness']:.3f}")

                with st.expander("Fitness Score Summary per Activity"):
                    st.dataframe(
                        avg_fitness_per_act.style.background_gradient(subset=['Avg Fitness'], cmap='RdYlGn'),
                        width='stretch',
                    )
            else:
                st.info("No fitness data computed.")
        else:
            st.info("No calendar date range available for fitness timeline.")

        # Expandable per-resource detail
        with st.expander("Per-Resource Absence Details"):
            for res_id in sorted(absence_df['Resource'].unique()):
                res_abs = absence_df[absence_df['Resource'] == res_id]
                st.markdown(f"**{res_id}** — {len(res_abs)} absences")
                st.dataframe(res_abs[['Start', 'End', 'Type', 'Description']], width='stretch')
else:
    st.info("No calendar data found at `data/calendars.json`.")
