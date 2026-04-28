import streamlit as st
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Simulation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Business Process Simulation Dashboard")

st.markdown("""
Welcome to the Business Process Simulation Dashboard.

Use the sidebar to navigate between:
- **Run Simulation**: Configure and execute new simulation runs
- **Analysis & Comparison**: Visualize results and compare different runs

### Quick Start
1. Go to **Run Simulation**
2. Select schedulers to test
3. Click **Start Simulation**
4. View results in **Analysis & Comparison**
""")

# Check for data directory
data_dir = Path("data")
if not data_dir.exists():
    st.error("⚠️ Data directory not found! Please run initialization scripts first.")
    st.code("python scripts/initialize_simulation.py")
else:
    st.success("✅ Data directory found. Ready to simulate.")

# Show system info
st.sidebar.info(
    """
    **System Status**
    - Python: 3.x
    - SimPy: Installed
    - Data: Loaded
    """
)
