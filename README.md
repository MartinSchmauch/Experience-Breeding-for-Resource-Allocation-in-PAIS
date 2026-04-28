# Business Process Simulation with Experience-Aware Scheduling

A discrete-event simulation framework for business process optimization with experience-aware resource scheduling, applied to the BPIC17 loan application dataset. Resources improve over time through a Richards learning curve, and the scheduler assigns tasks by solving a CP-SAT Multiple Knapsack Problem that accounts for experience levels, capability bottlenecks, and mentoring incentives.

---

## Project Structure

```
MasterThesis/
│
├── config/
│   ├── activity_requirements.yaml  # Required experience levels per activity
│   └── simulation_config.yaml    # Simulation parameters
│
├── dashboard/
│   ├── pages/
│   │   ├── 1_Run_Simulation.py
│   │   ├── 2_Analysis_&_Comparison.py
│   │   ├── 3_Capability_Overview.py
│   │   └── 4_Simulation_Timeline.py
│   └── app.py
│
├── data/                       # (Existing) Raw data and EDA outputs
│   ├── historical_logs/
│   │   └── BPIC17/
|   │       └── BPI_Challenge_2017.xes  # needs to be manually downloaded 
│   ├── calendars.json
│   ├── experience_store.json
│   ├── process_model.pkl
│   ├── timeline_split_info.json
│   ├── timeline.csv
│   └── simulation_outputs/     # Output: Simulation event logs
│
├── scripts/
│   ├── initialize_simulation.py  # offline simulation preperation
│   └── run_simulation.py       # online simulation 
│
├── src/
│   ├── entities/
│   │   ├── __init__.py
│   │   ├── resource.py         # Resource entity
│   │   ├── case.py             # Case entity
│   │   └── task.py             # Task entity
│   │
│   ├── experience/
│   │   ├── __init__.py
│   │   ├── store.py            # ExperienceStore & ExperienceProfile
│   │   ├── initializer.py      # Build from historical logs
│   │   ├── learning_curves.py  # learning curve logic
│   │   ├── level_tracker.py    # tracking experience development for ..._experience.csv log
│   │   ├── streamlit_viz.py    # learning curve visualization helper methods for dashboard
│   │   └── updater.py          # Learning models
│   │
│   ├── scheduling/
│   │   ├── __init__.py
│   │   ├── base.py             # Scheduler interface
│   │   ├── random_scheduler.py # Random assignment
│   │   ├── greedy_scheduler.py # Greedy assignment
│   │   ├── experience_based.py # Experience-based assignment
│   │   └── mkp_formulator.py   # Formulates MKP problem for experience_based scheduler
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── case_generator.py   # Case generation throughout simulation
│   │   ├── state.py            # SimulationState
│   │   └── engine.py           # SimulationEngine
│   │
│   ├── prediction/
│   │   ├── __init__.py
│   │   ├── base.py             # 
│   │   ├── features.py         # 
│   │   ├── models.py           # 
│   │   └── trainer.py          # 
│   │
│   ├── process/
│   │   ├── __init__.py
│   │   ├── model.py            # ProcessModel classes
│   │   └── loader.py           # Extract from logs
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── kpis.py             # KPICalculator
│   │   ├── daily_summary_logger.py
│   │   └── daily_summary_aggregator.py
│   │
│   ├── io/
│   │   ├── __init__.py
│   │   ├── logger.py           # abstract logger for simulation run
│   │   ├── log_reader.py       # Read XES/CSV
│   │   └── log_writer.py       # Write simulation logs
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── time_utils.py       # time conversion methods
│   │
│   └── __init__.py             # Package exports
│
├── Data_Analysis/              # (Existing) Data preprocessing notebooks
│   ├── 1_preprocessing/
│   ├── 2_process_model/
│   └── 3_results/
│
└── README.md                   # This file
```

---

## Getting Started

### 1. Download the BPIC17 event log

Download `BPI_Challenge_2017.xes` from the [4TU Research Data repository](https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884) and place it at:

```
data/historical_logs/BPIC17/BPI_Challenge_2017.xes
```

### 2. Set up the environment

```bash
python3 -m venv .mt
source .mt/bin/activate        # Mac/Linux
# .mt\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Initialize the simulation

This step reads the historical log and generates all required data artefacts under `data/`:

```bash
python scripts/initialize_simulation.py
```

It produces:
- `data/experience_store.json` — per-resource × activity × context duration profiles
- `data/process_model.pkl` — probabilistic process model extracted from the log
- `data/hr_data/calendars.json` — working-hour calendars with generated absences
- `data/timeline.csv` + `data/timeline_split_info.json` — case arrival timeline

Review these files before running a simulation to verify the extracted data looks sensible.

### 4. Configure the simulation

Edit `config/simulation_config.yaml` to select a scheduler, adjust the planning horizon, enable mentoring, etc. See the [Configuration Reference](#configuration-reference) below.

---

## Running a Simulation

```bash
python scripts/run_simulation.py
```

Each run writes a CSV log to `data/simulation_outputs/` with a name of the form:

```
SCHEDULERNAME__YYYY-MM-DD_HH-MM-SS.csv
```

---

## Evaluating Results

Open `DataAnalysis/3_results/kpi_analysis.ipynb`, set the run name variable at the top of the notebook to match your CSV filename, and execute all cells to compute KPIs and plots.

---

## Dashboard

An interactive Streamlit web app provides a UI for running simulations, inspecting the experience store, and evaluating results — without using the command line.

**Start the dashboard:**

```bash
# Option 1 — via the virtual environment directly
.mt/bin/streamlit run dashboard/app.py --server.headless true

# Option 2 — via the start script
bash dashboard/start_dashboard.sh
```

The app opens at `http://localhost:8501` and includes four pages:

| Page | Purpose |
|---|---|
| Run Simulation | Configure and launch a simulation run |
| Analysis & Comparison | Compare KPIs across multiple runs |
| Capability Overview | Inspect experience levels per resource and activity |
| Simulation Timeline | Visualize task assignments and resource utilization over time |

---

## How the Simulator Works

The engine is a **SimPy discrete-event simulation**. Cases arrive according to a timeline derived from the historical log (test split). Each case moves through a sequence of activities defined by a probabilistic process model extracted from the log.

At a configurable daily scheduling time, the scheduler collects all pending tasks and solves a **Multiple Knapsack Problem** (MKP) via Google OR-Tools CP-SAT. The objective maximises resource utilization and task pressure while penalising deferral to a dummy resource. When mentoring is enabled, the formulation also includes mentor–mentee assignment variables with bonuses for bottleneck risk reduction and underutilization.

After each task completes, the **ExperienceUpdater** advances the assignee's experience level along a Richards S-curve, which then influences future duration estimates and scheduling decisions.

**Scheduling algorithms:**

| Scheduler | Description |
|---|---|
| `experience_based` | CP-SAT MKP with experience, bottleneck detection, and optional mentoring |
| `greedy` | Assigns each task to the capable resource with the highest experience level |
| `random` | Assigns each task to a randomly chosen capable resource |

---

## Configuration Reference

All settings live in `config/simulation_config.yaml`. The most important parameters are:

### Simulation

| Key | Description |
|---|---|
| `simulation.max_simulation_days` | Total simulated calendar days to run |
| `simulation.max_tasks_per_case` | Hard cap on tasks per case (prevents runaway loops) |
| `case_arrival.probabilistic.case_fraction` | Fraction of historical cases to replay (1.0 = all) |

### Experience & Learning

| Key | Description |
|---|---|
| `experience.training_split` | Fraction of log used to initialise profiles (rest is test) |
| `experience.min_avg_daily_hours` | Minimum average daily activity hours for a resource to be included |
| `experience.breeding_params.growth_rate` | Richards curve growth rate — higher = faster learning |
| `experience.breeding_params.upper_asymptote` | Maximum achievable experience level |
| `experience.breeding_params.shape_param_Q` | Controls where on the curve growth starts |
| `experience.breeding_params.shape_param_M` | Controls the inflection point of the S-curve |

### Process Model

| Key | Description |
|---|---|
| `process_model.probabilistic.variant_filter.min_frequency` | Minimum trace frequency to include a process variant |
| `process_model.probabilistic.variant_filter.max_activity_occurrences` | Max times an activity may repeat within one case |

### Optimisation

| Key | Description |
|---|---|
| `optimization.objective_weights.pressure` | Weight for resource utilization pressure term |
| `optimization.objective_weights.utilization` | Weight for capacity-fill utilization term |
| `optimization.objective_weights.bottleneck` | Weight for bottleneck risk reduction (BRR) mentoring bonus |
| `optimization.objective_weights.underutilization` | Weight for proactive capacity-broadening mentoring bonus |
| `optimization.objective_weights.shortage` | Weight for same-day shortage urgency bonus |
| `optimization.max_task_deferrals` | Tasks deferred this many times receive maximum priority |

### Bottleneck Mitigation Strategies

| Key | Description |
|---|---|
| `mentoring.bottleneck_activity_strategy.enabled` | Detect forecast bottlenecks and incentivise mentoring for them |
| `mentoring.underutilization_strategy.enabled` | Identify underloaded resources and incentivise broadening their skills |
| `mentoring.same_day_shortage_strategy.enabled` | Trigger mentoring when same-day demand/supply ratio exceeds threshold |

### Working Hours & Absences

| Key | Description |
|---|---|
| `working_hours.generate_absences.vacation_days_per_year` | Mean vacation days generated per resource per year |
| `working_hours.generate_absences.sick_days_per_year` | Mean sick days generated per resource per year |

### Duration Prediction (experimental)

| Key | Description |
|---|---|
| `duration_prediction.enabled` | Enable ML-based duration prediction instead of experience store sampling — not recommended without a well-calibrated model |

---

## Contact

**Author:** Martin Schmauch  
**Institution:** Technical University of Munich (TUM)  
**Project:** Master Thesis — Business Process Optimization with Experience-Aware Scheduling
