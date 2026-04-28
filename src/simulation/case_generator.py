"""Generate simulation cases from historical data or synthetic distributions."""
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from ..entities import Case
from ..io import EventLogReader
from ..utils.time_utils import hours_to_seconds, seconds_to_hours, seconds_to_days


class CaseGenerator:
    """
    Generate simulation cases from timeline or synthetic distribution.
    
    Supports two modes:
    1. from_timeline: Use actual historical cases (default for validation)
    2. synthetic: Generate cases from distribution (for experimentation)
    """
    
    def __init__(self, case_arrival_config: dict, starting_date: datetime):
        """
        Initialize case generator with configuration.
        
        Args:
            config: Case arrival configuration from YAML
            starting_date: The date from which to start generating cases
        """
        self.config = case_arrival_config
        self.mode = case_arrival_config.get('mode', 'probabilistic')
        self.starting_date = starting_date

    def generate_cases(self) -> tuple[List[Case], Optional[int]]:
        """Generate cases based on configured mode.
        
        Returns:
            Tuple of (cases, total_events) where total_events is the number of 
            timeline events (None for synthetic mode)
        """
        if self.mode == 'synthetic':
            return self._generate_synthetic(), None
        elif self.mode == 'probabilistic':
            return self._generate_probabilistic()
        else:
            raise ValueError(f"Unknown case arrival mode: {self.mode}")
    
    
    def _generate_synthetic(self) -> List[Case]:
        """
        !!!DEPRECATED!!!
        Generate synthetic cases from distribution."""
        num_cases = self.config.get('total_cases', 100)
        arrival_pattern = self.config.get('pattern', 'poisson')
        rate = self.config.get('rate', 10.0)
        seed = self.config.get('random_seed', 42)
        
        rng = np.random.default_rng(seed)
        
        print(f"\n  Generating synthetic cases:")
        print(f"  Pattern: {arrival_pattern}, Rate: {rate} cases/hour, Total: {num_cases}")
        
        # Define distributions for case attributes
        loan_goals = ['Investment', 'Home improvement', 'Car', 'Existing loan takeover']
        app_types = ['New', 'Existing']
        
        cases = []
        arrival_time_sec = 0  # integer seconds
        
        for i in range(num_cases):
            # Generate inter-arrival time based on pattern (rate is in cases/hour)
            if arrival_pattern == 'poisson':
                inter_arrival_hours = rng.exponential(1.0 / rate)
            elif arrival_pattern == 'uniform':
                inter_arrival_hours = 1.0 / rate  # Constant inter-arrival
            else:
                raise ValueError(f"Unknown arrival pattern: {arrival_pattern}")
            
            arrival_time_sec += hours_to_seconds(inter_arrival_hours)
            
            case = Case(
                id=f"SIM_C{i:04d}",
                case_type='loan_application',
                arrival_time=arrival_time_sec,
                attributes={
                    'case:LoanGoal': rng.choice(loan_goals),
                    'case:ApplicationType': rng.choice(app_types),
                    'case:RequestedAmount': float(rng.integers(1000, 50000))
                }
            )
            cases.append(case)
        
        arrival_hours = seconds_to_hours(arrival_time_sec)
        print(f"  Generated {len(cases)} cases")
        print(f"  Time span: 0s to {arrival_time_sec}s ({arrival_hours:.2f}h / {seconds_to_days(arrival_time_sec):.1f} days)")
        
        return cases, len(cases)
    
    def _generate_probabilistic(self) -> tuple[List[Case], Optional[int]]:
        """
        Generate cases for probabilistic branching mode.
        
        Extracts case arrival times and context attributes directly from XES log,
        leaving activity sequences empty for dynamic determination by the
        probabilistic process model.
        
        Returns:
            Tuple of (cases, total_events)
        """
        xes_log_path = Path(self.config.get('xes_log_path', 'data/historical_logs/BPIC17/BPI_Challenge_2017.xes'))
        context_attributes = self.config.get('context_attributes', ['case:LoanGoal', 'case:ApplicationType'])
        probabilistic_cfg = self.config.get('probabilistic', {})
        case_fraction = float(probabilistic_cfg.get('case_fraction', 1.0))
        sampling_mode = probabilistic_cfg.get('sampling_mode', 'random')
        random_seed = int(self.config.get('random_seed', 42))

        if not (0.0 <= case_fraction <= 1.0):
            raise ValueError(f"case_arrival.probabilistic.case_fraction must be in [0, 1], got {case_fraction}")

        if sampling_mode not in {'random', 'earliest'}:
            raise ValueError(
                "case_arrival.probabilistic.sampling_mode must be one of: 'random', 'earliest'"
            )
        
        if not xes_log_path.exists():
            raise FileNotFoundError(f"XES log not found: {xes_log_path}")
        
        print(f"\nGenerating probabilistic cases from XES log: {xes_log_path}")
        
        # Use EventLogReader to read XES log
        reader = EventLogReader()
        log_df = reader.preprocess_for_simulation(filepath=xes_log_path)
                
        print(f"Total events in XES log: {len(log_df):,}")
        print(f"Total cases: {log_df['case:concept:name'].nunique():,}")
        
        # Filter to test split if split_date provided
        split_date = pd.to_datetime(self.starting_date, utc=True)
        log_df = log_df[log_df['time:timestamp'] >= split_date].copy()
        print(f"  Using test split (after {split_date.date()}): {len(log_df):,} events")
        print(f"  Test cases: {log_df['case:concept:name'].nunique():,}")
        
        # Get first event per case (arrival time)
        first_events = log_df.sort_values('time:timestamp').groupby('case:concept:name').first().reset_index()
        first_events = first_events.sort_values('time:timestamp')

        original_case_count = len(first_events)
        if original_case_count == 0:
            print("  No cases available after split filtering")
            return [], len(log_df)

        if case_fraction < 1.0:
            keep_n = int(np.floor(original_case_count * case_fraction))
            if case_fraction > 0 and keep_n == 0:
                keep_n = 1

            if keep_n == 0:
                first_events = first_events.iloc[0:0].copy()
            elif keep_n < original_case_count:
                if sampling_mode == 'earliest':
                    first_events = first_events.iloc[:keep_n].copy()
                else:
                    sampled = first_events.sample(n=keep_n, random_state=random_seed)
                    first_events = sampled.sort_values('time:timestamp').copy()

            print(
                f"  Case downsampling enabled: keeping {len(first_events):,}/{original_case_count:,} "
                f"cases ({case_fraction:.1%}) using sampling_mode={sampling_mode}"
            )
        
        cases = []
        first_arrival = None
        
        for _, row in first_events.iterrows():
            case_id = row['case:concept:name']
            arrival_time = row['time:timestamp']
            first_activity_name = row['concept:name']
            
            # Normalize to simulation time (0 = first case arrival, in seconds)
            if first_arrival is None:
                first_arrival = arrival_time
                arrival_time_sec = 0
            else:
                arrival_time_sec = int((arrival_time - first_arrival).total_seconds())
            
            # Extract case attributes for probabilistic branching
            attributes = {}
            
            # Add context attributes
            for attr in context_attributes:
                if attr in row and pd.notna(row[attr]):
                    # Store with clean name (remove 'case:' prefix if present)
                    clean_name = attr.replace('case:', '')
                    attributes[clean_name] = row[attr]
            
            # Add requested amount if available
            if 'case:RequestedAmount' in row and pd.notna(row['case:RequestedAmount']):
                attributes['RequestedAmount'] = row['case:RequestedAmount']
            
            # Create case WITHOUT pending_activity_types
            # Activities will be determined dynamically by ProbabilisticProcessModel
            case = Case(
                id=case_id,
                case_type=attributes.get('ApplicationType', 'unknown'),
                attributes=attributes,
                arrival_time=arrival_time_sec,
                pending_activity_types=[],  # Empty - activities determined by process model
                initial_activity=first_activity_name
            )
            
            cases.append(case)
        
        total_events = len(log_df)
        if not cases:
            print("  No cases generated after case fraction filtering")
            return [], total_events

        last_sec = cases[-1].arrival_time
        print(f"  Generated {len(cases):,} cases with probabilistic branching")
        print(f"  Activities will be determined dynamically by process model")
        print(f"  Time span: {cases[0].arrival_time}s to {last_sec}s")
        print(f"  Duration: {seconds_to_hours(last_sec):.2f} hours ({seconds_to_days(last_sec):.1f} days)")
        
        return cases, total_events