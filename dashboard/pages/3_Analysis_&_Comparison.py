import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import ast
import sys
import json
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.evaluation import KPICalculator
from src.experience.streamlit_viz import plot_learning_curve, plot_capability_heatmap

st.set_page_config(page_title="Analysis & Comparison", page_icon="📈", layout="wide")

st.title("📈 Analysis & Comparison")

# --- Load Data & Metadata ---
OUTPUT_DIR = Path("data/simulation_outputs")
if not OUTPUT_DIR.exists():
    st.warning("No simulation outputs found.")
    st.stop()

# Scan for PROCESS LOGS only (simulation event logs, excluding experience logs)
all_csv_files = OUTPUT_DIR.glob("sim_*.csv")
sim_log_files = sorted([f for f in all_csv_files if "_experience.csv" not in f.name], reverse=True)

if not sim_log_files:
    st.warning("No simulation log files found. Run a simulation first.")
    st.stop()

# Build index of runs from process logs
runs_data = []
for sim_file in sim_log_files:
    # Look for corresponding metadata
    meta_file = sim_file.parent / f"{sim_file.stem}_metadata.json"
    
    run_info = {
        "File": sim_file.name,
        "Path": str(sim_file),
        "Date": datetime.fromtimestamp(sim_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        "Scheduler": "Unknown",
        "Learning Model": "N/A",
        "Process Model": "Unknown",
        "Cases": "Unknown",
        "Mode": "Unknown",
        "Mentoring": "N/A",
        "Planning Horizon": "N/A",
        "Objective Weights": "N/A",
        "Bottleneck Mode": "N/A",
        "Max Tasks/Case": "N/A",
        "Duration Pred.": "N/A",
    }
    
    if meta_file.exists():
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                run_info["Scheduler"] = meta.get("scheduler", "Unknown")
                run_info["Learning Model"] = meta.get("learning_model", "N/A")
                run_info["Process Model"] = meta.get("process_model_type", "Unknown")
                
                # Handle both old and new metadata formats
                num_cases = meta.get("total_cases") or meta.get("num_cases", "Unknown")
                run_info["Cases"] = str(num_cases) if num_cases != "Unknown" else "Unknown"
                
                # Case arrival mode
                run_info["Mode"] = meta.get("case_arrival_mode", "Unknown")
                
                # Mentoring enabled (top-level or nested)
                mentoring_enabled = meta.get("mentoring_enabled")
                if mentoring_enabled is None:
                    mentoring_enabled = (meta.get("config", {}).get("mentoring", {}).get("enabled"))
                run_info["Mentoring"] = str(mentoring_enabled) if mentoring_enabled is not None else "N/A"

                # Planning horizon hours
                horizon = meta.get("config", {}).get("scheduling", {}).get("planning_horizon_hours")
                run_info["Planning Horizon"] = f"{horizon}h" if horizon is not None else "N/A"

                # Objective weights (compact string)
                obj_w = meta.get("config", {}).get("optimization", {}).get("objective_weights")
                if obj_w:
                    parts_ow = [f"{k[:4]}={v}" for k, v in obj_w.items()]
                    run_info["Objective Weights"] = ", ".join(parts_ow)
                
                # Severe bottleneck mode
                bn_mode = meta.get("config", {}).get("mentoring", {}).get("severe_bottleneck_mode")
                run_info["Bottleneck Mode"] = bn_mode if bn_mode else "N/A"
                
                # Maximum tasks per case
                max_tasks = meta.get("config", {}).get("simulation", {}).get("max_tasks_per_case")
                run_info["Max Tasks/Case"] = str(max_tasks) if max_tasks is not None else "N/A"

                # Duration prediction enabled
                dur_pred = meta.get("config", {}).get("duration_prediction", {}).get("enabled")
                run_info["Duration Pred."] = str(dur_pred) if dur_pred is not None else "N/A"
                
                if "timestamp" in meta:
                    try:
                        ts = datetime.strptime(meta["timestamp"], "%Y%m%d_%H%M%S")
                        run_info["Date"] = ts.strftime("%Y-%m-%d %H:%M")
                    except ValueError:
                        pass
        except Exception as e:
            st.warning(f"Could not load metadata for {sim_file.name}: {e}")
    else:
        # Fallback: parse filename
        parts = sim_file.stem.replace("sim_", "").split("_")
        if len(parts) > 2:
            run_info["Scheduler"] = " ".join(parts[:-2]).title()
    
    runs_data.append(run_info)

runs_df = pd.DataFrame(runs_data)

# --- Advanced Filtering ---
st.sidebar.header("Filter Runs")

# Scheduler Filter
all_schedulers = sorted(runs_df["Scheduler"].unique().tolist())
selected_schedulers = st.sidebar.multiselect("Filter by Scheduler", all_schedulers, default=all_schedulers)

# Learning Model Filter
all_models = sorted(runs_df["Learning Model"].unique().tolist())
selected_models = st.sidebar.multiselect("Filter by Learning Model", all_models, default=all_models)

# Process Model Filter
all_process_models = sorted(runs_df["Process Model"].unique().tolist())
selected_process_models = st.sidebar.multiselect("Filter by Process Model", all_process_models, default=all_process_models)

# Case Arrival Mode Filter
all_modes = sorted(runs_df["Mode"].unique().tolist())
selected_modes = st.sidebar.multiselect("Filter by Case Arrival Mode", all_modes, default=all_modes)

# Mentoring Filter
all_mentoring = sorted(runs_df["Mentoring"].unique().tolist())
selected_mentoring = st.sidebar.multiselect("Filter by Mentoring", all_mentoring, default=all_mentoring)

# Duration Prediction Filter
all_dur_pred = sorted(runs_df["Duration Pred."].unique().tolist())
selected_dur_pred = st.sidebar.multiselect("Filter by Duration Prediction", all_dur_pred, default=all_dur_pred)

# Bottleneck Mode Filter
all_bn_modes = sorted(runs_df["Bottleneck Mode"].unique().tolist())
selected_bn_modes = st.sidebar.multiselect("Filter by Bottleneck Mode", all_bn_modes, default=all_bn_modes)

# Apply filters
filtered_runs = runs_df[
    (runs_df["Scheduler"].isin(selected_schedulers)) &
    (runs_df["Learning Model"].isin(selected_models)) &
    (runs_df["Process Model"].isin(selected_process_models)) &
    (runs_df["Mode"].isin(selected_modes)) &
    (runs_df["Mentoring"].isin(selected_mentoring)) &
    (runs_df["Duration Pred."].isin(selected_dur_pred)) &
    (runs_df["Bottleneck Mode"].isin(selected_bn_modes))
]

# --- Run Selection Table ---
st.subheader("Select Runs to Compare")
st.info("Select rows in the table below to include them in the comparison.")

event = st.dataframe(
    filtered_runs[["Date", "Scheduler", "Max Tasks/Case", "Objective Weights", "Bottleneck Mode"]],
    width='stretch',
    hide_index=True,
    on_select="rerun",
    selection_mode="multi-row"
)

selected_indices = event.selection.rows
selected_rows = filtered_runs.iloc[selected_indices]

if selected_rows.empty:
    st.warning("Please select at least one run from the table above.")
    st.stop()

selected_files = selected_rows["File"].tolist()


def _daily_summary_candidates(file_name: str) -> list[Path]:
    base_name = str(file_name).strip()
    if not base_name:
        return []

    normalized = base_name[:-4] if base_name.endswith(".csv") else base_name
    candidates = [
        OUTPUT_DIR / f"{normalized}_daily_summary.jsonl",
        OUTPUT_DIR / f"{normalized}.jsonl",
    ]

    if normalized.endswith("_solver"):
        without_solver = normalized[:-7]
        candidates.insert(0, OUTPUT_DIR / f"{without_solver}_daily_summary.jsonl")

    return candidates


def _resolve_daily_summary_path(file_name: str) -> Path | None:
    for candidate in _daily_summary_candidates(file_name):
        if candidate.exists():
            return candidate
    return None


def _experience_candidates(file_name: str) -> list[Path]:
    base_name = str(file_name).strip()
    if not base_name:
        return []

    normalized = base_name[:-4] if base_name.endswith(".csv") else base_name
    candidates = [OUTPUT_DIR / f"{normalized}_experience.csv"]

    if normalized.endswith("_solver"):
        without_solver = normalized[:-7]
        candidates.insert(0, OUTPUT_DIR / f"{without_solver}_experience.csv")

    return candidates


def _resolve_experience_path(file_name: str) -> Path | None:
    for candidate in _experience_candidates(file_name):
        if candidate.exists():
            return candidate
    return None


def nested_column_prefixes(frame: pd.DataFrame, prefix: str) -> list[str]:
    return [column for column in frame.columns if column.startswith(f"{prefix}__")]


def _extract_nested_numeric(value: object, tuple_index: int = 0) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) > tuple_index:
            value = value[tuple_index]
        else:
            return 0.0
    elif isinstance(value, str):
        candidate = value.strip()
        if candidate.startswith("[") or candidate.startswith("("):
            try:
                parsed = ast.literal_eval(candidate)
                return _extract_nested_numeric(parsed, tuple_index=tuple_index)
            except (ValueError, SyntaxError):
                pass

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else 0.0


def nested_wide_frame(frame: pd.DataFrame, prefix: str, tuple_index: int = 0) -> pd.DataFrame:
    base = frame[["day_index", "sim_time_hours", "sim_daytime"]].copy() if "day_index" in frame.columns else frame.copy()
    for column in nested_column_prefixes(frame, prefix):
        label = column[len(prefix) + 2:]
        base[label] = frame[column].apply(lambda value: _extract_nested_numeric(value, tuple_index=tuple_index))
    return base


def series_numeric(frame: pd.DataFrame, column: str, multiplier: float = 1.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.zeros(len(frame)), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0) * float(multiplier)


def activity_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    return sorted({column[len(prefix) + 2:].split("__")[0] for column in nested_column_prefixes(frame, prefix)})


def per_activity_wide(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    wide = nested_wide_frame(frame, prefix)
    if "day_index" not in wide.columns:
        wide["day_index"] = np.arange(len(frame))
    return wide


def per_activity_sum(frame: pd.DataFrame, prefixes: list[str]) -> pd.DataFrame:
    out = pd.DataFrame()
    for prefix in prefixes:
        wide = per_activity_wide(frame, prefix)
        keep_cols = [c for c in wide.columns if c not in {"sim_time_hours", "sim_daytime"}]
        wide = wide[keep_cols].copy()
        if out.empty:
            out = wide
        else:
            out = out.merge(wide, on="day_index", how="outer", suffixes=("", "__dup"))
            dup_cols = [c for c in out.columns if c.endswith("__dup")]
            for dup_col in dup_cols:
                base_col = dup_col[:-5]
                out[base_col] = pd.to_numeric(out.get(base_col, 0.0), errors="coerce").fillna(0.0) + pd.to_numeric(out[dup_col], errors="coerce").fillna(0.0)
            out = out.drop(columns=dup_cols)

    if out.empty:
        out = pd.DataFrame({"day_index": np.arange(len(frame))})

    out["day_index"] = pd.to_numeric(out["day_index"], errors="coerce").fillna(0).astype(int)
    for col in out.columns:
        if col != "day_index":
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def as_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)


def per_activity_bounded_rate(
    frame: pd.DataFrame,
    numerator_prefix: str,
    denominator_prefixes: list[str],
) -> pd.DataFrame:
    numerator = per_activity_wide(frame, numerator_prefix)
    denominator = per_activity_sum(frame, denominator_prefixes)

    out = pd.DataFrame({"day_index": pd.to_numeric(denominator["day_index"], errors="coerce").fillna(0).astype(int)})
    activities = sorted(
        (set(numerator.columns) | set(denominator.columns)) - {"day_index", "sim_time_hours", "sim_daytime"}
    )

    for activity in activities:
        num = as_numeric_series(numerator, activity)
        den = as_numeric_series(denominator, activity)
        out[activity] = num.div(den.where(den > 0, np.nan)).fillna(0.0) * 100.0

    return out


def plot_per_activity_run_comparison(
    run_wide_by_label: dict[str, pd.DataFrame],
    title: str,
    y_label: str,
    run_color_map: dict[str, str] | None = None,
) -> go.Figure | None:
    from plotly.subplots import make_subplots as _make_subplots

    all_activities: set[str] = set()
    for run_wide in run_wide_by_label.values():
        all_activities.update([c for c in run_wide.columns if c != "day_index"])

    activities = sorted(all_activities)
    if not activities:
        return None

    n_cols = 2
    n_rows = int(np.ceil(len(activities) / n_cols))
    fig = _make_subplots(rows=n_rows, cols=n_cols, subplot_titles=activities, shared_xaxes=False, shared_yaxes=False)

    for idx, activity in enumerate(activities):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        for run_label, run_wide in run_wide_by_label.items():
            x = pd.to_numeric(run_wide["day_index"], errors="coerce").fillna(0).astype(int)
            y = as_numeric_series(run_wide, activity)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=run_label,
                    showlegend=(idx == 0),
                    line=dict(color=(run_color_map or {}).get(run_label)),
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text="Day Index", row=row, col=col)
        fig.update_yaxes(title_text=y_label, row=row, col=col)

    fig.update_layout(
        title=title,
        height=max(450, 320 * n_rows),
        legend_title_text="Run",
        legend=dict(orientation="h", yanchor="top", y=-0.10, xanchor="center", x=0.5),
    )
    return fig


def build_activity_pressure_heatmap(frame: pd.DataFrame) -> go.Figure | None:
    available_capacity_df = nested_wide_frame(frame, "available_capacity_hours_per_activity")
    task_demand_prefix = next(
        (prefix for prefix in ["task_demand_hours_per_activity", "estimated_task_hours_per_activity", "actual_task_hours_per_activity"] if nested_column_prefixes(frame, prefix)),
        None,
    )
    if task_demand_prefix is None:
        return None

    task_demand_df = nested_wide_frame(frame, task_demand_prefix)
    activities = sorted(
        set(activity_columns(frame, task_demand_prefix)) | set(activity_columns(frame, "available_capacity_hours_per_activity"))
    )
    if not activities:
        return None

    pressure_rows = []
    for activity in activities:
        demand_series = series_numeric(task_demand_df, activity)
        capacity_series = series_numeric(available_capacity_df, activity)
        pressure = demand_series.div(capacity_series.where(capacity_series > 0, np.nan)).fillna(0.0)
        pressure_rows.append(pd.DataFrame({
            "day_index": pd.to_numeric(frame["day_index"], errors="coerce").fillna(0).astype(int),
            "activity": activity,
            "pressure_ratio": pressure,
        }))

    pressure_df = pd.concat(pressure_rows, ignore_index=True) if pressure_rows else pd.DataFrame()
    if pressure_df.empty:
        return None

    pivot = pressure_df.pivot(index="activity", columns="day_index", values="pressure_ratio").fillna(0.0)
    zmax = float(np.nanmax(pivot.to_numpy())) if pivot.size else 1.0
    zmax = max(1.0, zmax)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(),
            x=[str(c) for c in pivot.columns.tolist()],
            y=pivot.index.tolist(),
            colorscale="RdYlGn_r",
            zmin=0.0,
            zmax=zmax,
            colorbar=dict(title="Demand / Capacity"),
        )
    )
    fig.update_layout(
        title="Activity Demand Pressure Heatmap",
        xaxis_title="Day Index",
        yaxis_title="Activity",
        height=max(450, 28 * len(pivot.index) + 220),
    )
    return fig


def first_existing_prefix(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for prefix in candidates:
        if nested_column_prefixes(frame, prefix):
            return prefix
    return None


def numeric_series_or_default(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(np.full(len(frame), default, dtype=float), index=frame.index, dtype=float)


def activity_resource_matrix(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    records = []
    for column in nested_column_prefixes(frame, prefix):
        remainder = column[len(prefix) + 2:]
        parts = remainder.split("__")
        if len(parts) < 2:
            continue
        activity = parts[0]
        resource = "__".join(parts[1:])
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        day_values = pd.to_numeric(frame.get("day_index", pd.Series(np.arange(len(frame)))), errors="coerce").fillna(0).astype(int)
        for day_idx, value in zip(day_values, values):
            records.append({"day_index": int(day_idx), "activity": activity, "resource": resource, "value": float(value)})

    if not records:
        return pd.DataFrame(columns=["day_index", "activity", "resource", "value"])
    return pd.DataFrame(records)


calculator = KPICalculator()
daily_data: dict[str, pd.DataFrame] = {}
daily_paths: dict[str, Path] = {}

# --- Process Data ---
data = {}
kpis_list = []
run_metadata = {}  # Store metadata for each run label

for file_name in selected_files:
    file_path = OUTPUT_DIR / file_name
    
    # Load PROCESS LOG (event log)
    df = pd.read_csv(file_path)
    
    # Verify it's a process log (has required columns)
    required_cols = ['case_id', 'activity', 'resource', 'timestamp']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Invalid log file format: {file_name}")
        st.write(f"Found columns: {df.columns.tolist()}")
        continue
    
    # Get metadata for label
    row = runs_df[runs_df["File"] == file_name].iloc[0]
    scheduler_name = row["Scheduler"]
    learning_model = row["Learning Model"]
    process_model = row["Process Model"]
    date_str = row["Date"]
    
    # Build descriptive label without PM/LM tags (fixed configuration)
    run_label = f"{scheduler_name} | {date_str}"
    
    data[run_label] = df
    run_metadata[run_label] = {
        'scheduler': scheduler_name,
        'learning_model': learning_model,
        'process_model': process_model,
        'file_name': file_name,
        'daily_summary_path': None,
    }
    
    # ---- Load persisted run stats (saved by simulation runners) ----
    stats_file = OUTPUT_DIR / f"{file_name.replace('.csv', '')}_stats.json"
    run_stats = None
    if stats_file.exists():
        try:
            with open(stats_file, 'r') as f:
                run_stats = json.load(f)
        except Exception:
            pass
    run_metadata[run_label]['stats'] = run_stats
    
    # Calculate KPIs (now works because df has 'case_id')
    sim_end = df['timestamp'].max()  # Changed from 'simulation_time'
    
    run_kpis = calculator.compute_all(df, simulation_start=0.0, simulation_end=sim_end)
    run_kpis['Run'] = run_label
    run_kpis['Scheduler'] = scheduler_name
    run_kpis['Process Model'] = process_model
    run_kpis['Learning Model'] = learning_model
    kpis_list.append(run_kpis)

    daily_summary_path = _resolve_daily_summary_path(file_name)
    if daily_summary_path is not None:
        try:
            daily_df = calculator.load_daily_summary_dataframe(daily_summary_path)
            if not daily_df.empty:
                daily_data[run_label] = daily_df
                daily_paths[run_label] = daily_summary_path
                run_metadata[run_label]['daily_summary_path'] = str(daily_summary_path)
        except Exception as e:
            st.warning(f"Could not load daily summary for {file_name}: {e}")

kpis_df = pd.DataFrame(kpis_list)

run_labels = kpis_df["Run"].tolist() if "Run" in kpis_df.columns else list(data.keys())
run_color_palette = px.colors.qualitative.Plotly
run_color_map = {
    run_label: run_color_palette[idx % len(run_color_palette)]
    for idx, run_label in enumerate(run_labels)
}

# Check if we have any valid data
if kpis_df.empty:
    st.error("No valid simulation logs were loaded. Please check that the selected files are process logs (not experience logs).")
    st.stop()

# --- Visualizations ---

# 1. KPI Comparison Table
st.subheader("KPI Comparison")
st.info("📊 **Daily Clearance Rate**: Fraction of tasks cleared each day. **Days Fully Cleared**: % of days with no backlog. Lower queue wait and backlog are better.")

# Batch-scheduling KPIs (primary)
batch_kpi_cols = [c for c in ['mean_daily_clearance_rate', 'pct_days_fully_cleared',
                               'mean_queue_wait_time', 'mean_daily_backlog',
                               'overall_completion_rate'] if c in kpis_df.columns]
classic_kpi_cols = [c for c in ['mean_cycle_time', 'throughput',
                                 'mean_resource_utilization', 'total_cases'] if c in kpis_df.columns]

if batch_kpi_cols:
    st.markdown("**Batch Scheduling KPIs**")
    fmt = {}
    if 'mean_daily_clearance_rate' in kpis_df.columns:
        fmt['mean_daily_clearance_rate'] = '{:.2%}'
    if 'pct_days_fully_cleared' in kpis_df.columns:
        fmt['pct_days_fully_cleared'] = '{:.2%}'
    if 'mean_queue_wait_time' in kpis_df.columns:
        fmt['mean_queue_wait_time'] = '{:.1f} s'
    if 'mean_daily_backlog' in kpis_df.columns:
        fmt['mean_daily_backlog'] = '{:.1f}'
    if 'overall_completion_rate' in kpis_df.columns:
        fmt['overall_completion_rate'] = '{:.2%}'
    st.dataframe(kpis_df.set_index('Run')[batch_kpi_cols].style.format(fmt), width='stretch')

if classic_kpi_cols:
    with st.expander("Classic KPIs", expanded=False):
        fmt2 = {}
        if 'mean_cycle_time' in kpis_df.columns:
            fmt2['mean_cycle_time'] = '{:.2f} h'
        if 'throughput' in kpis_df.columns:
            fmt2['throughput'] = '{:.2f} cases/h'
        if 'mean_resource_utilization' in kpis_df.columns:
            fmt2['mean_resource_utilization'] = '{:.2%}'
        if 'total_cases' in kpis_df.columns:
            fmt2['total_cases'] = '{:.0f}'
        st.dataframe(kpis_df.set_index('Run')[classic_kpi_cols].style.format(fmt2), width='stretch')

# 2. Bar Charts — batch KPIs
st.subheader("Batch Scheduling Performance")
col1, col2 = st.columns(2)

with col1:
    if 'mean_daily_clearance_rate' in kpis_df.columns:
        fig_cr = px.bar(kpis_df, x='Run', y='mean_daily_clearance_rate',
                        title="Daily Clearance Rate", color='Run', color_discrete_map=run_color_map)
        fig_cr.update_layout(yaxis_title="Clearance Rate", yaxis_tickformat=".0%", xaxis_title="")
        fig_cr.add_annotation(text="Higher is better", xref="paper", yref="paper",
                              x=0.5, y=1.05, showarrow=False, font=dict(size=10, color="gray"))
        st.plotly_chart(fig_cr, width='stretch')

with col2:
    if 'mean_daily_backlog' in kpis_df.columns:
        fig_bl = px.bar(kpis_df, x='Run', y='mean_daily_backlog',
                        title="Mean Daily Backlog (tasks)", color='Run', color_discrete_map=run_color_map)
        fig_bl.update_layout(yaxis_title="Backlog (tasks)", xaxis_title="")
        fig_bl.add_annotation(text="Lower is better", xref="paper", yref="paper",
                              x=0.5, y=1.05, showarrow=False, font=dict(size=10, color="gray"))
        st.plotly_chart(fig_bl, width='stretch')

col3, col4 = st.columns(2)
with col3:
    if 'pct_days_fully_cleared' in kpis_df.columns:
        fig_dc = px.bar(kpis_df, x='Run', y='pct_days_fully_cleared',
                        title="% Days Fully Cleared", color='Run', color_discrete_map=run_color_map)
        fig_dc.update_layout(yaxis_title="Fraction", yaxis_tickformat=".0%", xaxis_title="")
        fig_dc.add_annotation(text="Higher is better", xref="paper", yref="paper",
                              x=0.5, y=1.05, showarrow=False, font=dict(size=10, color="gray"))
        st.plotly_chart(fig_dc, width='stretch')

with col4:
    if 'mean_queue_wait_time' in kpis_df.columns:
        fig_qw = px.bar(kpis_df, x='Run', y='mean_queue_wait_time',
                        title="Mean Queue Wait Time (seconds)", color='Run', color_discrete_map=run_color_map)
        fig_qw.update_layout(yaxis_title="Wait (s)", xaxis_title="")
        fig_qw.add_annotation(text="Lower is better", xref="paper", yref="paper",
                              x=0.5, y=1.05, showarrow=False, font=dict(size=10, color="gray"))
        st.plotly_chart(fig_qw, width='stretch')

# ── NEW: Feasibility & Operational Metrics ────────────────────────
st.subheader("Feasibility & Operational Metrics")
st.markdown(
    "📋 Compare operational statistics across runs — drain behaviour, "
    "peak queue lengths, deferral rates, unfinished work, and overtime. "
    "These metrics indicate whether the schedule is **feasible** in practice."
)

# Collect stats from _stats.json files
feasibility_rows = []


def _safe_num(value, default=0.0):
    """Convert value to float; return default for None/invalid."""
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


for run_label in data.keys():
    rs = run_metadata[run_label].get('stats')
    if rs is None:
        continue
    qs = rs.get('queue_stats', {})
    ds = rs.get('drain_stats', {})
    uw = rs.get('unfinished_work', {})
    ot = rs.get('overtime_stats', {})
    kp = rs.get('kpis', {})

    # Backward/forward-compatible completion-rate key
    completion_rate = rs.get('completion_rate')
    if completion_rate is None:
        completion_rate = rs.get('case_completion_rate')

    feasibility_rows.append({
        'Run': run_label,
        'Completion Rate': _safe_num(completion_rate, 0.0),
        'Deferral Rate': _safe_num(kp.get('task_deferral_rate', 0.0), 0.0),
        'Incompletion Rate': _safe_num(kp.get('task_incompletion_rate', 0.0), 0.0),
        'Drop Rate': _safe_num(kp.get('task_drop_rate', 0.0), 0.0),
        'Cases Completed': _safe_num(rs.get('cases_completed', 0), 0),
        'Total Cases': _safe_num(rs.get('total_cases', 0), 0),
        'Peak Queue Length': _safe_num(qs.get('max_queue_length', 0), 0),
        'Tasks Still Queued': _safe_num(qs.get('tasks_remaining_in_queues', 0), 0),
        'Total Drained': _safe_num(ds.get('total_drained_tasks', 0), 0),
        'Drain Days': _safe_num(ds.get('drain_days', 0), 0),
        'Total Deferred': _safe_num(ds.get('total_deferred_tasks', 0), 0),
        'Dropped Tasks': _safe_num(ds.get('total_dropped_tasks', 0), 0),
        'Unscheduled Pool': _safe_num(ds.get('tasks_in_unscheduled_pool', 0), 0),
        'Tasks Unfinished': _safe_num(uw.get('total_unfinished_tasks', 0), 0),
        'Cases In Progress': _safe_num(uw.get('cases_in_progress', 0), 0),
        'Overtime (h)': _safe_num(ot.get('total_overtime_hours', 0), 0.0),
        'Max OT (h)': _safe_num(ot.get('max_overtime_hours', 0), 0.0),
    })

if feasibility_rows:
    feas_df = pd.DataFrame(feasibility_rows)

    # ---- Summary cards ----
    st.markdown("**Overview**")
    for _, frow in feas_df.iterrows():
        with st.expander(f"🔍 {frow['Run']}"):
            m1, m2, m3 = st.columns(3)
            m1.metric("Completion Rate", f"{frow['Completion Rate']:.2%}")
            m2.metric("Cases", f"{int(frow['Cases Completed'])}/{int(frow['Total Cases'])}")
            m3.metric("Tasks Unfinished", f"{int(frow['Tasks Unfinished']):,}")

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Total Drained", f"{int(frow['Total Drained']):,}",
                       help="Tasks moved back from queues for re-planning")
            d2.metric("Drain Days", int(frow['Drain Days']))
            d3.metric("Deferred", f"{int(frow['Total Deferred']):,}",
                       help="Tasks the solver couldn't assign — deferred to next day")
            d4.metric("Dropped", int(frow['Dropped Tasks']),
                       help="Tasks removed after exceeding max deferral limit")

            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Peak Queue Length", int(frow['Peak Queue Length']))
            q2.metric("Tasks Still Queued", int(frow['Tasks Still Queued']))
            q3.metric("Total Overtime", f"{frow['Overtime (h)']:.1f}h")
            q4.metric("Max Single OT", f"{frow['Max OT (h)']:.1f}h")

    # ---- Comparison bar charts ----
    st.markdown("**Cross-Run Comparison**")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_drain = px.bar(feas_df, x='Run', y='Total Drained',
                           title="Total Tasks Drained (re-scheduled)", color='Run', color_discrete_map=run_color_map)
        fig_drain.update_layout(yaxis_title="Tasks", xaxis_title="", showlegend=False)
        st.plotly_chart(fig_drain, width='stretch')

    with chart_col2:
        fig_defer = px.bar(feas_df, x='Run', y='Total Deferred',
                           title="Total Tasks Deferred by Solver", color='Run', color_discrete_map=run_color_map)
        fig_defer.update_layout(yaxis_title="Tasks", xaxis_title="", showlegend=False)
        st.plotly_chart(fig_defer, width='stretch')

    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        fig_peak = px.bar(feas_df, x='Run', y='Peak Queue Length',
                          title="Peak Queue Length (high-water mark)", color='Run', color_discrete_map=run_color_map)
        fig_peak.update_layout(yaxis_title="Tasks", xaxis_title="", showlegend=False)
        st.plotly_chart(fig_peak, width='stretch')

    with chart_col4:
        fig_ot = px.bar(feas_df, x='Run', y='Overtime (h)',
                        title="Total Overtime Hours", color='Run', color_discrete_map=run_color_map)
        fig_ot.update_layout(yaxis_title="Hours", xaxis_title="", showlegend=False)
        st.plotly_chart(fig_ot, width='stretch')

    # ---- Full comparison table ----
    with st.expander("Full Feasibility Table"):
        st.dataframe(
            feas_df.set_index('Run').style.format({
                'Completion Rate': '{:.2%}',
                'Deferral Rate': '{:.2%}',
                'Incompletion Rate': '{:.2%}',
                'Drop Rate': '{:.2%}',
                'Cases Completed': '{:.0f}',
                'Total Cases': '{:.0f}',
                'Peak Queue Length': '{:.0f}',
                'Tasks Still Queued': '{:.0f}',
                'Total Drained': '{:,.0f}',
                'Drain Days': '{:.0f}',
                'Total Deferred': '{:,.0f}',
                'Dropped Tasks': '{:.0f}',
                'Unscheduled Pool': '{:.0f}',
                'Tasks Unfinished': '{:,.0f}',
                'Cases In Progress': '{:.0f}',
                'Overtime (h)': '{:.1f}',
                'Max OT (h)': '{:.1f}',
            }),
            width='stretch',
        )
else:
    st.info(
        "No run stats files found for the selected runs. "
        "Re-run simulations to generate `_stats.json` files with feasibility data."
    )

# 2b. Daily Summary KPI Comparison (from notebook)
st.subheader("Daily Summary KPI Comparisons")
st.markdown(
    "Compare per-activity deferral and drop rates across all selected runs using the daily summary JSONL files. "
    "These plots match the notebook's multi-run comparison view and replace the older single-metric-only operational summary."
)

if daily_data:
    deferral_by_run = {
        label: per_activity_bounded_rate(
            frame,
            numerator_prefix='assigned_dummy_per_activity',
            denominator_prefixes=['assigned_real_per_activity', 'assigned_dummy_per_activity'],
        )
        for label, frame in daily_data.items()
    }
    drop_by_run = {
        label: per_activity_bounded_rate(
            frame,
            numerator_prefix='dropped_per_activity',
            denominator_prefixes=['assigned_real_per_activity', 'assigned_dummy_per_activity', 'dropped_per_activity'],
        )
        for label, frame in daily_data.items()
    }

    fig_deferral = plot_per_activity_run_comparison(
        deferral_by_run,
        title='Deferral Rate per Activity - Run Comparison (Bounded)',
        y_label='Rate (%)',
        run_color_map=run_color_map,
    )
    if fig_deferral is not None:
        st.plotly_chart(fig_deferral, width='stretch')
    else:
        st.info("No per-activity deferral data available in the selected daily summaries.")

    fig_drop = plot_per_activity_run_comparison(
        drop_by_run,
        title='Drop Rate per Activity - Run Comparison (Bounded)',
        y_label='Rate (%)',
        run_color_map=run_color_map,
    )
    if fig_drop is not None:
        st.plotly_chart(fig_drop, width='stretch')
    else:
        st.info("No per-activity drop data available in the selected daily summaries.")
else:
    st.info("No daily summary JSONL files were found for the selected runs.")

# 2c. Daily Summary KPI Trends (all selected runs)
st.subheader("Daily Summary KPI Trends")
st.markdown(
    "Shows one line per selected run for key daily-summary KPI rates from the notebook: "
    "mean utilization, utilization std dev, deferral rate, drop rate, and incompletion rate."
)

if daily_data:
    comparison_metrics = [
        ("mean_resource_utilization", "Mean Resource Utilization", "%", 100.0),
        ("std_resource_utilization", "Resource Utilization Std Dev", "%", 100.0),
        ("task_deferral_rate", "Deferral Rate", "%", 100.0),
        ("task_drop_rate", "Drop Rate", "%", 100.0),
        ("task_incompletion_rate", "Incompletion Rate", "%", 100.0),
    ]

    for metric, title, y_label, scale in comparison_metrics:
        fig = go.Figure()
        for run_label, frame in daily_data.items():
            x = pd.to_numeric(frame.get("day_index", pd.Series(np.arange(len(frame)))), errors="coerce").fillna(0).astype(int)
            y = series_numeric(frame, metric, multiplier=scale)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=run_label,
                    line=dict(color=run_color_map.get(run_label)),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Day Index",
            yaxis_title=y_label,
            hovermode="x unified",
            legend_title_text="Run",
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, width="stretch")
else:
    st.info("No daily summary JSONL files were found for the selected runs.")

# 2d. Single-run detailed analysis (notebook cell 16 + 19 style)
st.subheader("Single-Run Detailed Daily Summary Analysis")
st.markdown(
    "Select one run for detailed plots: per-activity task count vs demand/capacity, "
    "overall demand vs capacity, and task assignments per resource per activity."
)

if daily_data:
    main_run = st.selectbox(
        "Select run for detailed analysis",
        options=list(daily_data.keys()),
        index=0,
        key="single_run_detailed_analysis",
    )
    main_df = daily_data[main_run]

    # Main Run Activity Analysis: Task Count, Demand Hours vs Capacity
    st.markdown(f"**Main Run Activity Analysis ({main_run}): Task Count, Demand Hours vs Capacity**")

    main_counts_df = nested_wide_frame(main_df, "tasks_per_activity")
    main_demand_prefix = first_existing_prefix(
        main_df,
        [
            "task_demand_hours_per_activity",
            "estimated_task_hours_per_activity",
            "actual_task_hours_per_activity",
        ],
    )
    main_demand_df = nested_wide_frame(main_df, main_demand_prefix) if main_demand_prefix else pd.DataFrame()

    main_deferred_prefix = first_existing_prefix(main_df, ["deferred_task_demand_hours_per_activity"])
    main_deferred_df = nested_wide_frame(main_df, main_deferred_prefix) if main_deferred_prefix else pd.DataFrame()
    main_capacity_df = nested_wide_frame(main_df, "available_capacity_hours_per_activity")

    main_activities = sorted(
        set(activity_columns(main_df, "tasks_per_activity"))
        | set(activity_columns(main_df, main_demand_prefix or "task_demand_hours_per_activity"))
        | set(activity_columns(main_df, "available_capacity_hours_per_activity"))
    )

    if main_activities:
        n_cols = 2
        n_rows = int(np.ceil(len(main_activities) / n_cols))
        fig_activity = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=main_activities,
            specs=[[{"secondary_y": True} for _ in range(n_cols)] for _ in range(n_rows)],
        )

        days = pd.to_numeric(main_df.get("day_index", pd.Series(np.arange(len(main_df)))), errors="coerce").fillna(0).astype(int)

        for idx, activity in enumerate(main_activities):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            counts = numeric_series_or_default(main_counts_df, activity)
            demand_hours = numeric_series_or_default(main_demand_df, activity)
            deferred_hours = numeric_series_or_default(main_deferred_df, activity)
            capacity_hours = numeric_series_or_default(main_capacity_df, activity)

            fig_activity.add_trace(
                go.Scatter(x=days, y=counts, mode="lines+markers", name="Task Count", line=dict(color="steelblue"), marker=dict(size=4), showlegend=(idx == 0)),
                row=row,
                col=col,
                secondary_y=False,
            )
            fig_activity.add_trace(
                go.Scatter(x=days, y=demand_hours, mode="lines+markers", name="Task Demand Hours", line=dict(color="darkorange"), marker=dict(size=4), showlegend=(idx == 0)),
                row=row,
                col=col,
                secondary_y=True,
            )
            fig_activity.add_trace(
                go.Scatter(x=days, y=capacity_hours, mode="lines+markers", name="Workforce Capacity (h)", line=dict(color="green", dash="dash"), marker=dict(size=4), showlegend=(idx == 0)),
                row=row,
                col=col,
                secondary_y=True,
            )

            if not main_deferred_df.empty and activity in main_deferred_df.columns:
                fig_activity.add_trace(
                    go.Scatter(x=days, y=deferred_hours, mode="lines+markers", name="Deferred Task Demand Hours", line=dict(color="purple", dash="dot"), marker=dict(size=4), showlegend=(idx == 0)),
                    row=row,
                    col=col,
                    secondary_y=True,
                )

            shortage_mask = demand_hours > capacity_hours
            fig_activity.add_trace(
                go.Scatter(
                    x=days,
                    y=np.where(shortage_mask, demand_hours, np.nan),
                    mode="lines",
                    line=dict(width=0),
                    fill=None,
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
                secondary_y=True,
            )
            fig_activity.add_trace(
                go.Scatter(
                    x=days,
                    y=np.where(shortage_mask, capacity_hours, np.nan),
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(255,0,0,0.15)",
                    name="Demand Capacity Shortage",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
                secondary_y=True,
            )

            fig_activity.update_xaxes(title_text="Day Index", row=row, col=col)
            fig_activity.update_yaxes(title_text="Task Count", row=row, col=col, secondary_y=False)
            fig_activity.update_yaxes(title_text="Hours", row=row, col=col, secondary_y=True)

        fig_activity.update_layout(
            title=f"Main Run Activity Analysis ({main_run}): Task Count, Demand Hours vs Capacity",
            height=max(650, 360 * n_rows),
            legend_title_text="Series",
            legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_activity, width="stretch")
    else:
        st.info("No per-activity hours/capacity data available for the selected run.")

    # Main Run Overall Daily Task Demand vs Workforce Capacity
    st.markdown("**Main Run Overall Daily Task Demand vs Workforce Capacity**")

    if not main_demand_df.empty:
        overall_demand = main_demand_df.drop(
            columns=[c for c in ["day_index", "sim_time_hours", "sim_daytime"] if c in main_demand_df.columns],
            errors="ignore",
        ).sum(axis=1)
    else:
        overall_demand = pd.Series(np.zeros(len(main_df)), index=main_df.index, dtype=float)

    overall_capacity = pd.to_numeric(main_df.get("available_capacity_hours_total", 0.0), errors="coerce").fillna(0.0)
    if not main_deferred_df.empty:
        overall_deferred = main_deferred_df.drop(
            columns=[c for c in ["day_index", "sim_time_hours", "sim_daytime"] if c in main_deferred_df.columns],
            errors="ignore",
        ).sum(axis=1)
    else:
        overall_deferred = pd.Series(np.nan, index=main_df.index, dtype=float)

    x_time = pd.to_datetime(main_df.get("sim_daytime", pd.Series(np.arange(len(main_df)))), errors="coerce")
    x_label = "Simulation Day"
    if x_time.isna().all():
        x_time = pd.to_numeric(main_df.get("day_index", pd.Series(np.arange(len(main_df)))), errors="coerce").fillna(0).astype(int)
        x_label = "Day Index"

    fig_overall = go.Figure()
    fig_overall.add_trace(go.Scatter(x=x_time, y=overall_demand, mode="lines", name="Overall Task Demand Hours", line=dict(color="darkorange", width=3)))
    fig_overall.add_trace(go.Scatter(x=x_time, y=overall_capacity, mode="lines", name="Overall Workforce Capacity Hours (available_capacity_hours_total)", line=dict(color="seagreen", width=3)))
    if not overall_deferred.isna().all():
        fig_overall.add_trace(go.Scatter(x=x_time, y=overall_deferred, mode="lines", name="Overall Deferred Task Demand Hours", line=dict(color="purple", width=2, dash="dot")))

    shortage_mask = overall_demand > overall_capacity
    fig_overall.add_trace(go.Scatter(x=x_time, y=np.where(shortage_mask, overall_demand, np.nan), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig_overall.add_trace(go.Scatter(x=x_time, y=np.where(shortage_mask, overall_capacity, np.nan), mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(255,0,0,0.15)", name="Overall Demand Shortage", hoverinfo="skip"))

    surplus_mask = overall_demand <= overall_capacity
    fig_overall.add_trace(go.Scatter(x=x_time, y=np.where(surplus_mask, overall_demand, np.nan), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig_overall.add_trace(go.Scatter(x=x_time, y=np.where(surplus_mask, overall_capacity, np.nan), mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(0,128,0,0.10)", name="Overall Demand Surplus", hoverinfo="skip"))

    fig_overall.update_layout(
        title=f"Main Run Overall Daily Task Demand vs Workforce Capacity ({main_run})",
        xaxis_title=x_label,
        yaxis_title="Hours",
        hovermode="x unified",
    )
    st.plotly_chart(fig_overall, width="stretch")

    # Task Assignments per Resource per Activity
    st.markdown("**Task Assignments per Resource per Activity**")
    assignment_matrix = activity_resource_matrix(main_df, "assignment_count_per_activity_resource")
    if not assignment_matrix.empty:
        top_activities = sorted(assignment_matrix["activity"].unique())
        n_cols = 2
        n_rows = int(np.ceil(len(top_activities) / n_cols))
        fig_assign = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=top_activities)
        seen_resources: set[str] = set()

        for idx, activity in enumerate(top_activities):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            sub = assignment_matrix[assignment_matrix["activity"] == activity]
            pivot = sub.pivot_table(index="day_index", columns="resource", values="value", aggfunc="sum").fillna(0.0)
            for resource in pivot.columns:
                resource_name = str(resource)
                fig_assign.add_trace(
                    go.Bar(
                        x=pivot.index.astype(int),
                        y=pivot[resource],
                        name=resource_name,
                        showlegend=(resource_name not in seen_resources),
                    ),
                    row=row,
                    col=col,
                )
                seen_resources.add(resource_name)
            fig_assign.update_xaxes(title_text="Day", row=row, col=col)
            fig_assign.update_yaxes(title_text="# Assigned Tasks", row=row, col=col)

        fig_assign.update_layout(
            barmode="stack",
            title=f"Task Assignments per Resource per Activity ({main_run})",
            height=max(650, 360 * n_rows),
            legend_title_text="Resource",
            legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_assign, width="stretch")
    else:
        st.info("No assignment-by-activity resource matrix available for the selected run.")
else:
    st.info("No daily summary JSONL files were found for the selected runs.")

# 3. Task Count Distribution per Case
st.subheader("Tasks per Case Distribution")
st.markdown(
    "🧮 **Tasks per Case**: Histogram of how many tasks each case contains. "
    "X-axis = number of tasks in a case, Y-axis = number of cases with that task count."
)

tasks_per_case_frames = []
for label, df in data.items():
    if 'case_id' not in df.columns or 'task_id' not in df.columns:
        continue

    # Count unique tasks per case to avoid double counting start/complete events.
    case_task_counts = (
        df[['case_id', 'task_id']]
        .dropna(subset=['case_id', 'task_id'])
        .drop_duplicates()
        .groupby('case_id')['task_id']
        .nunique()
        .reset_index(name='task_count')
    )
    case_task_counts['Run'] = label
    tasks_per_case_frames.append(case_task_counts)

if tasks_per_case_frames:
    tasks_per_case_df = pd.concat(tasks_per_case_frames, ignore_index=True)

    fig_tasks_hist = px.histogram(
        tasks_per_case_df,
        x='task_count',
        color='Run',
        color_discrete_map=run_color_map,
        barmode='overlay',
        opacity=0.65,
        nbins=int(tasks_per_case_df['task_count'].max() - tasks_per_case_df['task_count'].min() + 1),
        title="Distribution of Tasks per Case"
    )
    fig_tasks_hist.update_layout(
        xaxis_title="Number of Tasks per Case",
        yaxis_title="Number of Cases"
    )
    st.plotly_chart(fig_tasks_hist, width='stretch')
else:
    st.info("No valid case/task data available to compute tasks-per-case distribution.")

# 4. Detailed Distributions
st.subheader("Cycle Time Distribution")
st.markdown("📈 **Case Cycle Time**: Total elapsed time from case arrival to completion, measured in hours. Includes both active processing time and waiting time in queues.")
combined_cycle_times = []

for label, df in data.items():
    # Calculate cycle time per case
    case_times = df.groupby('case_id')['timestamp'].agg(['min', 'max'])
    case_times['duration'] = case_times['max'] - case_times['min']
    case_times['Run'] = label
    combined_cycle_times.append(case_times)

if combined_cycle_times:
    all_cycle_times = pd.concat(combined_cycle_times)
    fig_dist = px.histogram(all_cycle_times, x='duration', color='Run', color_discrete_map=run_color_map, barmode='overlay', title="Cycle Time Histogram (hours)", opacity=0.7)
    fig_dist.update_layout(xaxis_title="Cycle Time (hours)", yaxis_title="Number of Cases")
    st.plotly_chart(fig_dist, width='stretch')
    
    fig_box = px.box(all_cycle_times, x='Run', y='duration', color='Run', color_discrete_map=run_color_map, title="Cycle Time Boxplot (hours)")
    fig_box.update_layout(yaxis_title="Cycle Time (hours)", xaxis_title="Simulation Run")
    st.plotly_chart(fig_box, width='stretch')

# 6. Experience Learning Curves (if available)
st.subheader("Experience Learning Curves")

# Look for experience tracker CSV files only for selected runs
experience_files: list[Path] = []
seen_experience_paths: set[Path] = set()
for selected_file in selected_files:
    selected_exp_path = _resolve_experience_path(selected_file)
    if selected_exp_path is not None and selected_exp_path not in seen_experience_paths:
        experience_files.append(selected_exp_path)
        seen_experience_paths.add(selected_exp_path)

if experience_files:
    st.write(f"Found {len(experience_files)} experience tracking files")
    
    # Select a file to visualize
    selected_exp_file = st.selectbox(
        "Select Experience Tracker File",
        options=[f.name for f in experience_files],
        format_func=lambda x: x.replace("_experience.csv", "")
    )
    
    if selected_exp_file:
        exp_file_path = OUTPUT_DIR / selected_exp_file
        try:
            exp_df = pd.read_csv(exp_file_path)
            
            if not exp_df.empty:
                # Resource filter
                available_resources = exp_df['resource_id'].unique()
                selected_resource = st.selectbox("Select Resource", options=sorted(available_resources))
                
                # Activity filter
                resource_activities = exp_df[exp_df['resource_id'] == selected_resource]['activity_name'].unique()
                selected_activity = st.selectbox("Select Activity", options=sorted(resource_activities))
                
                # Plot learning curve
                st.subheader(f"Learning Curve: {selected_resource} - {selected_activity}")
                
                resource_activity_df = exp_df[
                    (exp_df['resource_id'] == selected_resource) & 
                    (exp_df['activity_name'] == selected_activity)
                ]
                
                if not resource_activity_df.empty:
                    # Plot experience level over time
                    fig = plot_learning_curve(
                        resource_activity_df,
                        title=f"{selected_resource} - {selected_activity}"
                    )
                    st.plotly_chart(fig, width='stretch')
                    
                    # Show statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Initial Experience", f"{resource_activity_df['experience_level'].iloc[0]:.1f}")
                    with col2:
                        st.metric("Final Experience", f"{resource_activity_df['experience_level'].iloc[-1]:.1f}")
                    with col3:
                        improvement = resource_activity_df['experience_level'].iloc[-1] - resource_activity_df['experience_level'].iloc[0]
                        st.metric("Improvement", f"+{improvement:.1f}")
                    
                    # ── Capable Resources Over Time (per Activity) ──────────────
                    st.subheader("Capable Resources Over Time (per Activity)")
                    st.markdown(
                        "Shows how many resources have reached the **required capability level** "
                        "for each activity over simulation time. Resources already capable at "
                        "simulation start (from the experience store baseline) count from Day 0. "
                        "Each step-up marks a new resource crossing the threshold defined in `activity_requirements.yaml`."
                    )

                    # Load required levels from config
                    import yaml as _yaml
                    _req_path = Path(__file__).parent.parent.parent / "config" / "activity_requirements.yaml"
                    _activity_reqs: dict = {}
                    _default_req = 30.0
                    if _req_path.exists():
                        with open(_req_path) as _f:
                            _req_cfg = _yaml.safe_load(_f)
                        _activity_reqs = _req_cfg.get('activity_requirements', {})
                        _default_req = _req_cfg.get('default_requirement', {}).get('level', 30.0)

                    # Load experience store to get baseline capability levels (before simulation)
                    # Map: resource_id -> {activity_name: max experience_level across contexts}
                    _baseline_levels: dict = {}
                    _exp_store_path = Path("data/experience_store.json")
                    if _exp_store_path.exists():
                        try:
                            from src.experience.store import ExperienceStore as _ExperienceStore
                            _exp_store = _ExperienceStore.load(_exp_store_path)
                            
                            for _profile in _exp_store._profiles.values():
                                _rid, _act, _lvl = _profile.resource_id, _profile.activity_name, _profile.experience_level
                                if _rid not in _baseline_levels:
                                    _baseline_levels[_rid] = {}
                                # Keep max across contexts for this (resource, activity)
                                if _act not in _baseline_levels[_rid] or _lvl > _baseline_levels[_rid][_act]:
                                    _baseline_levels[_rid][_act] = _lvl
                        except Exception as _e:
                            st.caption(f"Could not load experience store baseline: {_e}")

                    # Build capable events:
                    # 1. Resources already capable at baseline → day 0
                    # 2. Resources that first cross the threshold during simulation (from exp_df)
                    _capable_events = []
                    _already_capable: set = set()

                    for _rid, _acts in _baseline_levels.items():
                        for _act, _lvl in _acts.items():
                            _req = _activity_reqs.get(_act, _default_req)
                            if _lvl >= _req:
                                _capable_events.append({'activity': _act, 'resource': _rid, 'sim_day': 0.0})
                                _already_capable.add((_rid, _act))

                    for (res_id, act_name), grp in exp_df.groupby(['resource_id', 'activity_name']):
                        if (res_id, act_name) in _already_capable:
                            continue  # already counted at day 0
                        req_level = _activity_reqs.get(act_name, _default_req)
                        crossed = grp[grp['experience_level'] >= req_level]
                        if not crossed.empty:
                            first_time_h = crossed['simulation_time'].min()
                            _capable_events.append({
                                'activity': act_name,
                                'resource': res_id,
                                'sim_day': first_time_h / 24.0,
                            })

                    if _capable_events:
                        _cap_df = pd.DataFrame(_capable_events)
                        _time_max_day = exp_df['simulation_time'].max() / 24.0
                        _fig_cap = go.Figure()
                        _cap_colors = px.colors.qualitative.Plotly

                        for i, act in enumerate(sorted(_cap_df['activity'].unique())):
                            act_events = _cap_df[_cap_df['activity'] == act].sort_values('sim_day')
                            req_level = _activity_reqs.get(act, _default_req)

                            # Separate baseline (day 0) from simulation crossings
                            baseline_count = int((act_events['sim_day'] == 0.0).sum())
                            later_events = act_events[act_events['sim_day'] > 0.0]

                            # Build step function that starts at baseline_count
                            xs = [0.0] + later_events['sim_day'].tolist() + [_time_max_day]
                            ys = [baseline_count] + list(range(baseline_count + 1, baseline_count + len(later_events) + 1)) + [baseline_count + len(later_events)]
                            _fig_cap.add_trace(go.Scatter(
                                x=xs,
                                y=ys,
                                mode='lines',
                                line=dict(shape='hv', color=_cap_colors[i % len(_cap_colors)]),
                                name=f"{act} (req ≥ {req_level:.0f})",
                                hovertemplate="Day %{x:.1f}: %{y} capable<extra>" + act + "</extra>",
                            ))

                        _fig_cap.update_layout(
                            title="Cumulative Capable Resources Per Activity Over Time",
                            xaxis_title="Simulation Day",
                            yaxis_title="Number of Capable Resources",
                            yaxis=dict(rangemode='tozero', dtick=1),
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                            hovermode='x unified',
                        )
                        st.plotly_chart(_fig_cap, width='stretch')

                    st.markdown("**Experience Development by Resource (Selected Run and Activity)**")
                    st.markdown(
                        "Select a run and activity to view all resource experience curves together. "
                        "Y-axis is experience level and X-axis is simulation datetime."
                    )

                    experience_run_file_map = {}
                    for run_label, meta in run_metadata.items():
                        exp_path = _resolve_experience_path(meta.get('file_name', ''))
                        if exp_path is not None:
                            experience_run_file_map[run_label] = exp_path

                    if experience_run_file_map:
                        experience_run_options = sorted(experience_run_file_map.keys())
                        selected_experience_run = st.selectbox(
                            "Select Run for Experience Curves",
                            options=experience_run_options,
                            index=0,
                            key="experience_curves_run",
                        )

                        selected_experience_path = experience_run_file_map[selected_experience_run]
                        try:
                            selected_exp_df = pd.read_csv(selected_experience_path)
                            required_cols = {"resource_id", "activity_name", "experience_level"}

                            if not required_cols.issubset(selected_exp_df.columns):
                                st.info("Selected run experience file is missing required columns for this chart.")
                            else:
                                selected_exp_df = selected_exp_df.dropna(subset=["resource_id", "activity_name", "experience_level"]).copy()
                                selected_exp_df["experience_level"] = pd.to_numeric(selected_exp_df["experience_level"], errors="coerce")
                                selected_exp_df = selected_exp_df.dropna(subset=["experience_level"])

                                available_activity_options = sorted(selected_exp_df["activity_name"].dropna().unique())
                                if available_activity_options:
                                    selected_curve_activity = st.selectbox(
                                        "Select Activity for Resource Curves",
                                        options=available_activity_options,
                                        index=0,
                                        key="experience_curves_activity",
                                    )

                                    curve_df = selected_exp_df[selected_exp_df["activity_name"] == selected_curve_activity].copy()

                                    if not curve_df.empty:
                                        if "sim_datetime" in curve_df.columns:
                                            curve_df["x_axis"] = pd.to_datetime(curve_df["sim_datetime"], errors="coerce")
                                        else:
                                            curve_df["x_axis"] = pd.NaT

                                        x_label = "sim_datetime"
                                        if curve_df["x_axis"].isna().all():
                                            if "simulation_time" in curve_df.columns:
                                                curve_df["x_axis"] = pd.to_numeric(curve_df["simulation_time"], errors="coerce")
                                                x_label = "simulation_time (hours)"
                                            else:
                                                curve_df["x_axis"] = pd.to_numeric(curve_df.get("repetition_count", 0), errors="coerce")
                                                x_label = "repetition_count"

                                        curve_df = curve_df.dropna(subset=["x_axis", "experience_level", "resource_id"])
                                        if not curve_df.empty:
                                            fig_resource_curves = go.Figure()
                                            for resource_id, resource_group in curve_df.groupby("resource_id"):
                                                resource_group = resource_group.sort_values("x_axis")
                                                fig_resource_curves.add_trace(
                                                    go.Scatter(
                                                        x=resource_group["x_axis"],
                                                        y=resource_group["experience_level"],
                                                        mode="lines",
                                                        name=str(resource_id),
                                                    )
                                                )

                                            required_level = float(_activity_reqs.get(selected_curve_activity, _default_req))
                                            fig_resource_curves.add_hline(
                                                y=required_level,
                                                line_dash="dash",
                                                line_color="firebrick",
                                                annotation_text=f"Required Level: {required_level:.1f}",
                                                annotation_position="top right",
                                            )

                                            fig_resource_curves.update_layout(
                                                title=f"Experience Curves by Resource - {selected_experience_run} - {selected_curve_activity}",
                                                xaxis_title=x_label,
                                                yaxis_title="experience_level",
                                                hovermode="x unified",
                                                legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
                                            )
                                            st.plotly_chart(fig_resource_curves, width='stretch')
                                        else:
                                            st.info("No valid curve points available for the selected run and activity.")
                                    else:
                                        st.info("No experience records found for the selected activity in this run.")
                                else:
                                    st.info("No activities found in the selected run experience file.")
                        except Exception as curve_error:
                            st.info(f"Could not load selected run experience data: {curve_error}")
                    else:
                        st.info(
                            "No experience files were found for the selected runs. "
                            "Please make sure the run has a matching *_experience.csv output."
                        )

                    if not _capable_events:
                        st.info(
                            "No resources reached the required capability level during this simulation run. "
                            "Resources may still be developing — check back after more simulation time."
                        )

                    # Activity demand pressure heatmap from daily summary data
                    st.subheader("Activity Demand Pressure Heatmap")
                    st.markdown(
                        "Shows the activity demand-to-capacity ratio over time for a selected run. "
                        "Values above 1.0 indicate demand exceeds same-day capacity."
                    )

                    if daily_data:
                        heatmap_run = st.selectbox(
                            "Select Run for Demand Pressure Heatmap",
                            options=list(daily_data.keys()),
                            index=0,
                            key="activity_demand_pressure_run",
                        )
                        heatmap_fig = build_activity_pressure_heatmap(daily_data[heatmap_run])
                        if heatmap_fig is not None:
                            st.plotly_chart(heatmap_fig, width='stretch')
                        else:
                            st.info("No activity demand pressure data available in the selected daily summary.")
                    else:
                        st.info("No daily summary JSONL files were found for the selected runs.")
                    
                    # Top 10 Capability Improvements
                    st.subheader("Top 10 Capability Improvements")
                    st.markdown("View the resource-activity combinations with the highest experience gains during this simulation run.")
                    
                    if st.button("Show Top 10 Learning Curves", key="top10_button"):
                        # Calculate improvement for each resource-activity combination
                        improvements = []
                        
                        for (resource, activity), group in exp_df.groupby(['resource_id', 'activity_name']):
                            if len(group) >= 2:
                                initial = group['experience_level'].iloc[0]
                                final = group['experience_level'].iloc[-1]
                                improvement = final - initial
                                
                                improvements.append({
                                    'resource_id': resource,
                                    'activity_name': activity,
                                    'initial': initial,
                                    'final': final,
                                    'improvement': improvement,
                                    'data': group
                                })
                        
                        # Sort by improvement and get top 10
                        improvements_sorted = sorted(improvements, key=lambda x: x['improvement'], reverse=True)[:10]
                        
                        if improvements_sorted:
                            st.success(f"Found top {len(improvements_sorted)} capability improvements")
                            
                            # Display in a 2-column grid
                            for i in range(0, len(improvements_sorted), 2):
                                col1, col2 = st.columns(2)
                                
                                # First chart in row
                                with col1:
                                    imp = improvements_sorted[i]
                                    rank = i + 1
                                    title = f"#{rank}: {imp['resource_id']} - {imp['activity_name']}"
                                    
                                    fig = plot_learning_curve(
                                        imp['data'],
                                        title=title
                                    )
                                    st.plotly_chart(fig, width='stretch')
                                    
                                    # Show metrics
                                    col_a, col_b, col_c = st.columns(3)
                                    with col_a:
                                        st.metric("Initial", f"{imp['initial']:.1f}")
                                    with col_b:
                                        st.metric("Final", f"{imp['final']:.1f}")
                                    with col_c:
                                        st.metric("Gain", f"+{imp['improvement']:.1f}", delta=f"+{imp['improvement']:.1f}")
                                
                                # Second chart in row (if exists)
                                if i + 1 < len(improvements_sorted):
                                    with col2:
                                        imp = improvements_sorted[i + 1]
                                        rank = i + 2
                                        title = f"#{rank}: {imp['resource_id']} - {imp['activity_name']}"
                                        
                                        fig = plot_learning_curve(
                                            imp['data'],
                                            title=title
                                        )
                                        st.plotly_chart(fig, width='stretch')
                                        
                                        # Show metrics
                                        col_a, col_b, col_c = st.columns(3)
                                        with col_a:
                                            st.metric("Initial", f"{imp['initial']:.1f}")
                                        with col_b:
                                            st.metric("Final", f"{imp['final']:.1f}")
                                        with col_c:
                                            st.metric("Gain", f"+{imp['improvement']:.1f}", delta=f"+{imp['improvement']:.1f}")
                        else:
                            st.warning("No improvement data available")
                else:
                    st.info("No data for this resource-activity combination")
            else:
                st.warning("Experience tracker file is empty")
                
        except Exception as e:
            st.error(f"Error loading experience data: {e}")
else:
    st.info("No experience tracking data available for the selected runs. Enable 'Track Experience Levels' when running simulations.")

# 7. Working Hours Analysis
st.subheader("Working Hours Distribution by Weekday")
st.markdown("⏰ **Working Hours Analysis**: Shows when work is actually being performed throughout the week. Each bar represents the total hours of work completed during that hour across all selected resources and simulation runs.")

if data:
    # Get all unique resources across selected runs
    all_resources_set = set()
    for df in data.values():
        all_resources_set.update(df['resource'].unique())
    
    all_resources = sorted(list(all_resources_set))
    
    # Resource selection
    st.write("**Select Resources to Include:**")
    col_select_all, col_filters = st.columns([1, 4])
    
    with col_select_all:
        select_all = st.checkbox("Select All Resources", value=True, key="select_all_resources")
    
    if select_all:
        selected_resources = all_resources
    else:
        with col_filters:
            selected_resources = st.multiselect(
                "Choose specific resources:",
                options=all_resources,
                default=all_resources[:min(5, len(all_resources))],
                key="resource_multiselect"
            )
    
    if selected_resources:
        # Process data to calculate working hours
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        # Aggregate data across all selected runs
        working_hours_data = {day: {hour: 0.0 for hour in range(24)} for day in range(5)}
        
        for run_label, df in data.items():
            # Get the actual starting weekday from the first log entry
            # Load metadata to get simulation start datetime
            file_name = run_metadata[run_label]['file_name']
            meta_file = OUTPUT_DIR / f"{file_name.replace('.csv', '')}_metadata.json"
            
            start_weekday = 0  # Default to Monday if metadata not available
            if meta_file.exists():
                try:
                    with open(meta_file, 'r') as f:
                        meta = json.load(f)
                        if 'simulation_start_day' in meta:
                            # Parse the start date and get weekday
                            start_weekday = meta['simulation_start_day']
                except Exception as e:
                    st.warning(f"Could not determine start weekday from metadata: {e}")
            
            # Filter by selected resources
            df_filtered = df[df['resource'].isin(selected_resources)].copy()
            
            # Find matching start and complete events
            start_events = df_filtered[df_filtered['lifecycle'] == 'start'].copy()
            complete_events = df_filtered[df_filtered['lifecycle'] == 'complete'].copy()
            
            # Merge to get task durations
            start_events = start_events.rename(columns={'timestamp': 'start_time'})
            complete_events = complete_events.rename(columns={'timestamp': 'complete_time'})
            
            tasks = pd.merge(
                start_events[['task_id', 'resource', 'start_time']],
                complete_events[['task_id', 'complete_time']],
                on='task_id',
                how='inner'
            )
            
            if not tasks.empty:
                # Calculate duration in hours
                tasks['duration'] = tasks['complete_time'] - tasks['start_time']
                
                # Convert simulation time to datetime accounting for actual starting weekday
                # Simulation time is in hours from start
                def sim_time_to_weekday_hour(sim_hours, start_weekday_offset):
                    # Calculate which day relative to simulation start
                    days_elapsed = sim_hours / 24.0
                    # Adjust for actual starting weekday
                    weekday = (int(days_elapsed) + start_weekday_offset) % 7
                    hour = int(sim_hours % 24)
                    return weekday, hour
                
                # For each task, distribute its duration across the hours it spans
                for _, task in tasks.iterrows():
                    start_time = task['start_time']
                    end_time = task['complete_time']
                    
                    # Get start weekday and hour (accounting for actual simulation start day)
                    start_weekday, start_hour = sim_time_to_weekday_hour(start_time, start_weekday)
                    end_weekday, end_hour = sim_time_to_weekday_hour(end_time, start_weekday)
                    
                    # Only count work during weekdays (Mon-Fri = 0-4)
                    if start_weekday < 5:
                        if start_weekday == end_weekday:
                            # Task completed within same day
                            # Distribute across hours
                            current_hour = start_hour
                            remaining_duration = task['duration']
                            
                            while remaining_duration > 0 and current_hour < 24:
                                hour_fraction = min(1.0, remaining_duration)
                                working_hours_data[start_weekday][current_hour] += hour_fraction
                                remaining_duration -= hour_fraction
                                current_hour += 1
                        else:
                            # Task spans multiple days - attribute to start day/hour for simplicity
                            working_hours_data[start_weekday][start_hour] += task['duration']
        
        # Create visualizations - 5 plots in a grid
        st.write(f"**Showing data for {len(selected_resources)} selected resource(s)**")
        
        # Display as 2 columns (3 plots top row, 2 plots bottom row)
        col1, col2 = st.columns(2)
        
        for day_idx in range(5):
            with col1 if day_idx % 2 == 0 else col2:
                day_name = weekday_names[day_idx]
                
                # Prepare data for plotting
                hours = list(range(24))
                work_hours = [working_hours_data[day_idx][h] for h in hours]
                
                # Create bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=hours,
                    y=work_hours,
                    name=day_name,
                    marker_color='steelblue'
                ))
                
                fig.update_layout(
                    title=f"{day_name}",
                    xaxis_title="Hour of Day",
                    yaxis_title="Total Working Hours",
                    xaxis=dict(
                        tickmode='linear',
                        tick0=0,
                        dtick=2,
                        range=[-0.5, 23.5]
                    ),
                    yaxis=dict(
                        rangemode='tozero'
                    ),
                    height=350,
                    showlegend=False
                )
                
                st.plotly_chart(fig, width='stretch')
        
        # Summary statistics
        st.subheader("Summary Statistics")
        
        # Calculate total hours per weekday
        total_per_day = {weekday_names[i]: sum(working_hours_data[i].values()) for i in range(5)}
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        for idx, (day, total) in enumerate(total_per_day.items()):
            col = [col1, col2, col3, col4, col5][idx]
            with col:
                st.metric(day, f"{total:.1f}h")
        
        # Peak hours analysis
        st.write("**Peak Working Hours:**")
        peak_data = []
        for day_idx in range(5):
            day_hours = working_hours_data[day_idx]
            peak_hour = max(day_hours.items(), key=lambda x: x[1])
            peak_data.append({
                'Weekday': weekday_names[day_idx],
                'Peak Hour': f"{peak_hour[0]:02d}:00 - {peak_hour[0]+1:02d}:00",
                'Working Hours': f"{peak_hour[1]:.2f}h"
            })
        
        peak_df = pd.DataFrame(peak_data)
        st.dataframe(peak_df, hide_index=True, width='stretch')
        
    else:
        st.warning("Please select at least one resource to analyze.")
else:
    st.info("No simulation data available for working hours analysis.")
