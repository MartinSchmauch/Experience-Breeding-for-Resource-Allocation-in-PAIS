#!/bin/bash
# Start the Streamlit dashboard

# Get the absolute path to the project root (parent of dashboard folder)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root directory
cd "$PROJECT_ROOT"

# Run streamlit headless to avoid trying to call system browser opener in restricted shells
.mt/bin/streamlit run dashboard/app.py --server.headless true
