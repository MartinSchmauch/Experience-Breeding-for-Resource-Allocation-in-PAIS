import streamlit as st
import sys
import subprocess
from pathlib import Path
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Initialize Data", page_icon="🛠️", layout="wide")
st.title("🛠️ Initialize Simulation Data")
st.write(
    "Run the existing initialization pipeline from `scripts/initialize_simulation.py` "
    "to create/update experience store, probabilistic process model, and calendar data."
)

CONFIG_PATH = Path("config/simulation_config.yaml")


@st.cache_data
def load_config(config_path: Path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config(config_path: Path, config: dict):
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def _ensure_nested(config: dict, key: str):
    if key not in config or config[key] is None:
        config[key] = {}


if not CONFIG_PATH.exists():
    st.error(f"Configuration file not found: {CONFIG_PATH}")
    st.stop()

config = load_config(CONFIG_PATH)

# Sidebar controls requested by user
st.sidebar.header("Configuration")

variant_filter_cfg = (
    config.get("process_model", {})
    .get("probabilistic", {})
    .get("variant_filter", {})
)

min_frequency = st.sidebar.number_input(
    "Variant Filter: Min Frequency",
    min_value=1,
    max_value=10000,
    value=int(variant_filter_cfg.get("min_frequency", 1)),
    step=1,
)

loop_handling = st.sidebar.selectbox(
    "Variant Filter: Loop Handling",
    options=["keep", "remove", "trim"],
    index=["keep", "remove", "trim"].index(variant_filter_cfg.get("loop_handling", "trim"))
    if variant_filter_cfg.get("loop_handling", "trim") in ["keep", "remove", "trim"]
    else 2,
)

max_activity_occurrences = st.sidebar.number_input(
    "Variant Filter: Max Activity Occurrences",
    min_value=1,
    max_value=50,
    value=int(variant_filter_cfg.get("max_activity_occurrences", 3)),
    step=1,
)

experience_cfg = config.get("experience", {})
training_split = st.sidebar.number_input(
    "Experience: Training Split",
    min_value=0.01,
    max_value=0.99,
    value=float(experience_cfg.get("training_split", 0.3)),
    step=0.01,
    format="%.2f",
)

min_avg_daily_hours = st.sidebar.number_input(
    "Experience: Min Avg Daily Hours",
    min_value=0.0,
    max_value=24.0,
    value=float(experience_cfg.get("min_avg_daily_hours", 0.75)),
    step=0.05,
)

st.subheader("Current Run Configuration")
col1, col2 = st.columns(2)
col1.metric("Min Frequency", f"{int(min_frequency)}")
col1.metric("Loop Handling", loop_handling)
col1.metric("Max Activity Occurrences", f"{int(max_activity_occurrences)}")
col2.metric("Training Split", f"{training_split:.2f}")
col2.metric("Min Avg Daily Hours", f"{min_avg_daily_hours:.2f}")

if st.button("Run Initialization", type="primary"):
    # Update only the requested config keys, then call the existing script
    _ensure_nested(config, "process_model")
    _ensure_nested(config["process_model"], "probabilistic")
    _ensure_nested(config["process_model"]["probabilistic"], "variant_filter")
    _ensure_nested(config, "experience")

    config["process_model"]["probabilistic"]["variant_filter"]["min_frequency"] = int(min_frequency)
    config["process_model"]["probabilistic"]["variant_filter"]["loop_handling"] = str(loop_handling)
    config["process_model"]["probabilistic"]["variant_filter"]["max_activity_occurrences"] = int(max_activity_occurrences)
    config["experience"]["training_split"] = float(training_split)
    config["experience"]["min_avg_daily_hours"] = float(min_avg_daily_hours)

    save_config(CONFIG_PATH, config)
    st.cache_data.clear()

    with st.spinner("Running initialization script..."):
        cmd = [sys.executable, "scripts/initialize_simulation.py"]
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        st.success("Initialization completed successfully.")
    else:
        st.error(f"Initialization failed with exit code {result.returncode}.")

    if result.stdout:
        st.subheader("Output")
        st.code(result.stdout, language="text")

    if result.stderr:
        st.subheader("Errors")
        st.code(result.stderr, language="text")

st.info(
    "This page reuses the existing script logic and writes your selected values to "
    "`config/simulation_config.yaml` before execution."
)
