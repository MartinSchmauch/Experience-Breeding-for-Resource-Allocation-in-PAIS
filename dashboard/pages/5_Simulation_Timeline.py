"""Simulation Timeline – interactive Gantt-style deep dive into a single run."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Simulation Timeline", page_icon="📅", layout="wide")
st.title("📅 Simulation Timeline")

# ──────────────────────────────────────────────────────────────
# 1. Discover available runs (reuse logic from Analysis page)
# ──────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("data/simulation_outputs")
if not OUTPUT_DIR.exists():
    st.warning("No simulation outputs found.")
    st.stop()

all_csv_files = OUTPUT_DIR.glob("sim_*.csv")
sim_log_files = sorted(
    [f for f in all_csv_files if "_experience.csv" not in f.name], reverse=True
)
if not sim_log_files:
    st.warning("No simulation log files found. Run a simulation first.")
    st.stop()

runs_data = []
for sim_file in sim_log_files:
    meta_file = sim_file.parent / f"{sim_file.stem}_metadata.json"
    run_info = {
        "File": sim_file.name,
        "Date": datetime.fromtimestamp(sim_file.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M"
        ),
        "Scheduler": "Unknown",
        "Learning Model": "N/A",
        "Mentoring": "N/A",
        "Bottleneck Mode": "N/A",
    }
    if meta_file.exists():
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
            run_info["Scheduler"] = meta.get("scheduler", "Unknown")
            run_info["Learning Model"] = meta.get("learning_model", "N/A")
            mentoring_enabled = meta.get("mentoring_enabled") or meta.get(
                "config", {}
            ).get("mentoring", {}).get("enabled")
            run_info["Mentoring"] = (
                str(mentoring_enabled) if mentoring_enabled is not None else "N/A"
            )
            bn_mode = (
                meta.get("config", {}).get("mentoring", {}).get("severe_bottleneck_mode")
            )
            run_info["Bottleneck Mode"] = bn_mode if bn_mode else "N/A"
            if "timestamp" in meta:
                try:
                    ts = datetime.strptime(meta["timestamp"], "%Y%m%d_%H%M%S")
                    run_info["Date"] = ts.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    pass
        except Exception:
            pass
    runs_data.append(run_info)

runs_df = pd.DataFrame(runs_data)

# ──────────────────────────────────────────────────────────────
# 2. Single-run selector
# ──────────────────────────────────────────────────────────────
st.sidebar.header("Select Run")
event = st.sidebar.dataframe(
    runs_df[["Date", "Scheduler", "Learning Model", "Mentoring", "Bottleneck Mode"]],
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="timeline_run_select",
)

selected_indices = event.selection.rows
if not selected_indices:
    st.info("👈 Select a simulation run from the sidebar to visualise its timeline.")
    st.stop()

selected_file = runs_df.iloc[selected_indices[0]]["File"]
st.sidebar.success(f"Selected: **{selected_file}**")

# ──────────────────────────────────────────────────────────────
# 3. Load data for selected run
# ──────────────────────────────────────────────────────────────
csv_path = OUTPUT_DIR / selected_file
df_raw = pd.read_csv(csv_path, low_memory=False)

# Load solver JSONL (optional – for bottleneck data)
solver_path = OUTPUT_DIR / selected_file.replace(".csv", "_solver.jsonl")
solver_days = []
if solver_path.exists():
    with open(solver_path, "r") as f:
        for line in f:
            try:
                solver_days.append(json.loads(line))
            except json.JSONDecodeError:
                continue

# Load calendars (optional – for absence overlay)
calendars_path = Path("data/calendars.json")
calendars = {}
if calendars_path.exists():
    with open(calendars_path, "r") as f:
        calendars = json.load(f)

# ──────────────────────────────────────────────────────────────
# 4. Parse CSV into task intervals
# ──────────────────────────────────────────────────────────────
# Use a synthetic base date so timestamps look like real dates on the x-axis.
BASE_DATE = datetime(2025, 1, 1)


def hours_to_dt(h: float) -> datetime:
    """Convert simulation-hours to a synthetic datetime."""
    return BASE_DATE + timedelta(hours=h)


# Separate start and complete events
starts = df_raw[df_raw["lifecycle"] == "start"].copy()
completes = df_raw[df_raw["lifecycle"] == "complete"].copy()

# Merge on (task_id, resource) to get intervals
intervals = starts.merge(
    completes[["task_id", "resource", "timestamp"]],
    on=["task_id", "resource"],
    how="inner",
    suffixes=("_start", "_end"),
)

if intervals.empty:
    st.error("No start/complete event pairs found in the selected run.")
    st.stop()

# Build Gantt rows
gantt_rows = []
for _, row in intervals.iterrows():
    resource = row["resource"]
    task_type = row.get("task_type", "standard")
    is_mentoring = task_type == "mentoring"
    mentor_id = row.get("mentor") if pd.notna(row.get("mentor")) else None
    mentee_id = row.get("mentee") if pd.notna(row.get("mentee")) else None

    # For mentoring tasks the resource column is "Mentee+Mentor"
    # Create a row for the mentee in their own lane
    if is_mentoring and mentee_id:
        gantt_rows.append(
            {
                "Resource": mentee_id,
                "Start": hours_to_dt(row["timestamp_start"]),
                "End": hours_to_dt(row["timestamp_end"]),
                "Activity": row["activity"],
                "TaskType": "mentoring (mentee)",
                "CaseID": row["case_id"],
                "TaskID": row["task_id"],
                "Duration_h": round(row["timestamp_end"] - row["timestamp_start"], 2),
                "Mentor": mentor_id or "",
                "Mentee": mentee_id or "",
                "Category": f"🎓 {row['activity']}",
            }
        )
        # Also create a shadow row for the mentor
        if mentor_id:
            gantt_rows.append(
                {
                    "Resource": mentor_id,
                    "Start": hours_to_dt(row["timestamp_start"]),
                    "End": hours_to_dt(row["timestamp_end"]),
                    "Activity": row["activity"],
                    "TaskType": "mentoring (mentor)",
                    "CaseID": row["case_id"],
                    "TaskID": row["task_id"],
                    "Duration_h": round(
                        row["timestamp_end"] - row["timestamp_start"], 2
                    ),
                    "Mentor": mentor_id,
                    "Mentee": mentee_id or "",
                    "Category": f"🧑‍🏫 {row['activity']}",
                }
            )
    else:
        # Standard task – determine resource (strip "+" composites if present)
        res = resource.split("+")[0] if "+" in str(resource) else resource
        gantt_rows.append(
            {
                "Resource": res,
                "Start": hours_to_dt(row["timestamp_start"]),
                "End": hours_to_dt(row["timestamp_end"]),
                "Activity": row["activity"],
                "TaskType": "standard",
                "CaseID": row["case_id"],
                "TaskID": row["task_id"],
                "Duration_h": round(row["timestamp_end"] - row["timestamp_start"], 2),
                "Mentor": "",
                "Mentee": "",
                "Category": row["activity"],
            }
        )

gantt_df = pd.DataFrame(gantt_rows)
gantt_df["SimDay"] = ((gantt_df["Start"] - BASE_DATE).dt.total_seconds() / 3600 / 24).astype(int)

# ──────────────────────────────────────────────────────────────
# 5. Build absence overlay rows
# ──────────────────────────────────────────────────────────────
absence_rows = []
if calendars:
    sim_start_dt = BASE_DATE
    sim_end_dt = hours_to_dt(df_raw["timestamp"].max())
    for res_id, cal in calendars.items():
        for ab in cal.get("absences", []):
            ab_start = pd.Timestamp(ab["start_date"]).to_pydatetime()
            ab_end = pd.Timestamp(ab["end_date"]).to_pydatetime()
            # Only include absences that overlap the simulation period
            if ab_end < sim_start_dt or ab_start > sim_end_dt:
                continue
            absence_rows.append(
                {
                    "Resource": res_id,
                    "Start": max(ab_start, sim_start_dt),
                    "End": min(ab_end, sim_end_dt),
                    "Activity": "Absent",
                    "TaskType": ab.get("absence_type", "absent"),
                    "CaseID": "",
                    "TaskID": "",
                    "Duration_h": 0,
                    "Mentor": "",
                    "Mentee": "",
                    "Category": f"⛔ {ab.get('absence_type', 'Absent')}",
                    "SimDay": -1,
                }
            )
absence_df = pd.DataFrame(absence_rows) if absence_rows else pd.DataFrame()

# ──────────────────────────────────────────────────────────────
# 6. Build bottleneck data per day
# ──────────────────────────────────────────────────────────────
bottleneck_rows = []
for entry in solver_days:
    sim_day = entry.get("sim_day", 0)
    for bn in entry.get("input", {}).get("bottlenecks", []):
        bottleneck_rows.append(
            {
                "SimDay": sim_day,
                "Activity": bn["activity_name"],
                "Severity": bn["severity"],
                "DaysUntil": bn.get("days_until_bottleneck", "?"),
                "CapableCount": bn.get("capable_resource_count", "?"),
            }
        )
bottleneck_df = pd.DataFrame(bottleneck_rows) if bottleneck_rows else pd.DataFrame()

# ──────────────────────────────────────────────────────────────
# 7. Summary metrics
# ──────────────────────────────────────────────────────────────
total_tasks = len(gantt_df[gantt_df["TaskType"] == "standard"])
mentoring_mentee = gantt_df[gantt_df["TaskType"] == "mentoring (mentee)"]
total_mentoring = len(mentoring_mentee)
total_resources = gantt_df["Resource"].nunique()
sim_days = gantt_df["SimDay"].max() + 1 if not gantt_df.empty else 0
mentoring_pct = (total_mentoring / (total_tasks + total_mentoring) * 100) if (total_tasks + total_mentoring) > 0 else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Tasks", f"{total_tasks + total_mentoring:,}")
col2.metric("Mentoring Sessions", f"{total_mentoring}")
col3.metric("Mentoring %", f"{mentoring_pct:.1f}%")
col4.metric("Resources", total_resources)
col5.metric("Simulation Days", sim_days)

# ──────────────────────────────────────────────────────────────
# 8. Interactive filters
# ──────────────────────────────────────────────────────────────
st.sidebar.header("Timeline Filters")

# Day range slider
if sim_days > 0:
    day_range = st.sidebar.slider(
        "Simulation Day Range",
        min_value=0,
        max_value=max(sim_days - 1, 0),
        value=(0, min(4, max(sim_days - 1, 0))),
        key="day_range",
    )
else:
    day_range = (0, 0)

# Resource filter
all_resources = sorted(gantt_df["Resource"].unique())
selected_resources = st.sidebar.multiselect(
    "Resources",
    all_resources,
    default=all_resources,
    key="timeline_resources",
)

# Activity filter
all_activities = sorted(gantt_df["Activity"].unique())
selected_activities = st.sidebar.multiselect(
    "Activities",
    all_activities,
    default=all_activities,
    key="timeline_activities",
)

# Toggles
show_absences = st.sidebar.checkbox("Show Absences", value=True, key="show_abs")
show_bottlenecks = st.sidebar.checkbox("Show Bottleneck Markers", value=True, key="show_bn")
show_mentoring_highlight = st.sidebar.checkbox(
    "Highlight Mentoring", value=True, key="show_mentor_hl"
)

# ──────────────────────────────────────────────────────────────
# 9. Filter data
# ──────────────────────────────────────────────────────────────
mask = (
    (gantt_df["SimDay"] >= day_range[0])
    & (gantt_df["SimDay"] <= day_range[1])
    & (gantt_df["Resource"].isin(selected_resources))
    & (gantt_df["Activity"].isin(selected_activities))
)
filtered_df = gantt_df[mask].copy()

if filtered_df.empty:
    st.warning("No tasks match the current filters. Adjust day range, resources, or activities.")
    st.stop()

# Add absences if toggled on
plot_df = filtered_df.copy()
if show_absences and not absence_df.empty:
    # Filter absences to selected resources and day range
    range_start = hours_to_dt(day_range[0] * 24)
    range_end = hours_to_dt((day_range[1] + 1) * 24)
    abs_mask = (
        absence_df["Resource"].isin(selected_resources)
        & (absence_df["End"] >= range_start)
        & (absence_df["Start"] <= range_end)
    )
    abs_filtered = absence_df[abs_mask].copy()
    if not abs_filtered.empty:
        plot_df = pd.concat([plot_df, abs_filtered], ignore_index=True)

# ──────────────────────────────────────────────────────────────
# 10. Build colour map
# ──────────────────────────────────────────────────────────────
# Assign distinct colours per activity for standard tasks
ACTIVITY_COLOURS = px.colors.qualitative.Plotly + px.colors.qualitative.Set2
unique_activities = sorted(gantt_df["Activity"].unique())

colour_map = {}
for i, act in enumerate(unique_activities):
    base = ACTIVITY_COLOURS[i % len(ACTIVITY_COLOURS)]
    colour_map[act] = base  # standard task colour
    if show_mentoring_highlight:
        colour_map[f"🎓 {act}"] = "#FFD700"  # mentee = gold
        colour_map[f"🧑‍🏫 {act}"] = "#FF8C00"  # mentor = dark orange

# Absence colours
colour_map["⛔ vacation"] = "rgba(180,180,180,0.4)"
colour_map["⛔ sick_leave"] = "rgba(220,100,100,0.4)"
colour_map["⛔ Absent"] = "rgba(180,180,180,0.4)"

# ──────────────────────────────────────────────────────────────
# 11. Render the Gantt timeline
# ──────────────────────────────────────────────────────────────
st.subheader(
    f"Task Execution Timeline — Days {day_range[0]}–{day_range[1]}"
)

# Sort resources for consistent y-axis ordering
plot_df["Resource"] = pd.Categorical(
    plot_df["Resource"], categories=sorted(plot_df["Resource"].unique()), ordered=True
)

fig = px.timeline(
    plot_df,
    x_start="Start",
    x_end="End",
    y="Resource",
    color="Category",
    color_discrete_map=colour_map,
    hover_data={
        "Activity": True,
        "TaskType": True,
        "CaseID": True,
        "Duration_h": True,
        "Mentor": True,
        "Mentee": True,
        "Category": False,
        "Start": False,
        "End": False,
        "Resource": False,
    },
    labels={"Category": "Task Category"},
)

# Day boundary vertical lines
for d in range(day_range[0], day_range[1] + 2):
    day_dt = hours_to_dt(d * 24)
    day_ms = day_dt.timestamp() * 1000  # Plotly needs ms for datetime axes
    fig.add_shape(
        type="line",
        x0=day_ms, x1=day_ms,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(width=1, dash="dot", color="grey"),
    )
    fig.add_annotation(
        x=day_ms, y=1.02,
        xref="x", yref="paper",
        text=f"Day {d}",
        showarrow=False,
        font=dict(size=9, color="grey"),
    )

# Bottleneck markers as coloured background rectangles
if show_bottlenecks and not bottleneck_df.empty:
    bn_in_range = bottleneck_df[
        (bottleneck_df["SimDay"] >= day_range[0])
        & (bottleneck_df["SimDay"] <= day_range[1])
    ]
    for _, bn_row in bn_in_range.iterrows():
        d = bn_row["SimDay"]
        severity = bn_row["Severity"]
        fill = "rgba(255,60,60,0.10)" if severity == "severe" else "rgba(255,165,0,0.08)"
        border = "rgba(255,60,60,0.6)" if severity == "severe" else "rgba(255,165,0,0.5)"
        x0_ms = hours_to_dt(d * 24).timestamp() * 1000
        x1_ms = hours_to_dt((d + 1) * 24).timestamp() * 1000
        fig.add_shape(
            type="rect",
            x0=x0_ms, x1=x1_ms,
            y0=0, y1=1,
            xref="x", yref="paper",
            fillcolor=fill,
            line=dict(width=1, color=border, dash="dash"),
            layer="below",
        )
        fig.add_annotation(
            x=x0_ms, y=1.02,
            xref="x", yref="paper",
            text=f"⚠ {bn_row['Activity']} ({severity})",
            showarrow=False,
            font=dict(size=8, color="red" if severity == "severe" else "darkorange"),
            xanchor="left",
        )

n_resources = plot_df["Resource"].nunique()
fig.update_layout(
    height=max(400, n_resources * 60),
    xaxis_title="Simulation Time",
    yaxis_title="Resource",
    legend_title="Task Category",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=80, r=20, t=60, b=40),
)
fig.update_yaxes(autorange="reversed")  # top resource first

st.plotly_chart(fig, width='stretch')

# ──────────────────────────────────────────────────────────────
# 12. Mentoring detail panel
# ──────────────────────────────────────────────────────────────
mentoring_in_range = filtered_df[
    filtered_df["TaskType"].isin(["mentoring (mentee)", "mentoring (mentor)"])
]

if not mentoring_in_range.empty:
    with st.expander(f"🎓 Mentoring Sessions Detail ({len(mentoring_in_range[mentoring_in_range['TaskType'] == 'mentoring (mentee)'])} sessions)", expanded=False):
        display_cols = ["SimDay", "Activity", "Mentee", "Mentor", "Duration_h", "CaseID", "TaskID"]
        mentee_sessions = mentoring_in_range[mentoring_in_range["TaskType"] == "mentoring (mentee)"][display_cols].sort_values("SimDay")
        st.dataframe(mentee_sessions, width='stretch', hide_index=True)
else:
    st.info("No mentoring sessions in the selected day range.")

# ──────────────────────────────────────────────────────────────
# 13. Bottleneck detail panel
# ──────────────────────────────────────────────────────────────
if not bottleneck_df.empty:
    bn_in_range = bottleneck_df[
        (bottleneck_df["SimDay"] >= day_range[0])
        & (bottleneck_df["SimDay"] <= day_range[1])
    ]
    if not bn_in_range.empty:
        with st.expander(f"⚠️ Bottlenecks in Range ({len(bn_in_range)} detections)", expanded=False):
            st.dataframe(bn_in_range.sort_values(["SimDay", "Severity"]), width='stretch', hide_index=True)
elif solver_days:
    st.info("No bottlenecks detected in the selected day range.")
else:
    st.info("No solver log (JSONL) available for this run — bottleneck overlay disabled.")
