"""KPI calculation from simulation event logs.

All timestamps in the event log are in **hours** (converted from the
simulation's internal seconds at the logging boundary).  KPIs that
represent durations (cycle time, waiting time, service time, queue wait)
are therefore also expressed in hours.
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict
from typing import Dict, Optional, Any
from pathlib import Path

HOURS_PER_DAY = 24.0


class KPICalculator:
    """
    Calculate Key Performance Indicators from simulation event logs.
    
    Computes standard process mining metrics like throughput, cycle time,
    waiting time, resource utilization, etc.
    
    Expected log format: Each task has two rows (start and complete lifecycle events).
    """
    
    def __init__(self):
        """Initialize KPI calculator."""
        pass
    
    def compute_all(
        self,
        log_df: pd.DataFrame,
        case_column: str = 'case_id',
        task_column: str = 'task_id',
        activity_column: str = 'activity',
        resource_column: str = 'resource',
        timestamp_column: str = 'timestamp',
        lifecycle_column: str = 'lifecycle',
        simulation_start: float = 0.0,
        simulation_end: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute all KPIs from event log.
        
        Args:
            log_df: Event log dataframe with paired start/complete events per task
            case_column: Column name for case ID
            task_column: Column name for task ID
            activity_column: Column name for activity type
            resource_column: Column name for resource
            timestamp_column: Column name for timestamp
            lifecycle_column: Column name for lifecycle transition ("start" or "complete")
            simulation_start: Simulation start time
            simulation_end: Simulation end time (None = infer from log)
            
        Returns:
            Dictionary of KPI name -> value
        """
        kpis: Dict[str, float] = {}
        # Early exit for empty logs
        if log_df.empty:
            return self._empty_kpis()
        
        # Infer simulation end if not provided
        if simulation_end is None:
            simulation_end = log_df[timestamp_column].max()
        
        simulation_duration = simulation_end - simulation_start
        if simulation_duration <= 0:
            simulation_duration = 1.0
        
        # Create paired task dataframe (one row per task with start and complete times)
        tasks_df = self._create_task_pairs(
            log_df, task_column, timestamp_column, lifecycle_column,
            case_column, activity_column, resource_column
        )
        
        # Compute KPIs from paired tasks
        kpis['total_cases'] = log_df[case_column].nunique()
        kpis['completed_cases'] = tasks_df[case_column].nunique()
        kpis['throughput'] = kpis['completed_cases'] / simulation_duration
        
        # Cycle time metrics (per case)
        cycle_times = self._compute_cycle_times(tasks_df, case_column)
        if len(cycle_times) > 0:
            kpis['mean_cycle_time'] = cycle_times.mean()
            kpis['median_cycle_time'] = cycle_times.median()
            kpis['std_cycle_time'] = cycle_times.std()
            kpis['p95_cycle_time'] = cycle_times.quantile(0.95)
        else:
            kpis['mean_cycle_time'] = 0.0
            kpis['median_cycle_time'] = 0.0
            kpis['std_cycle_time'] = 0.0
            kpis['p95_cycle_time'] = 0.0
        
        # Service time metrics (per task)
        service_times = tasks_df['duration']
        if len(service_times) > 0:
            kpis['mean_service_time'] = service_times.mean()
            kpis['median_service_time'] = service_times.median()
        else:
            kpis['mean_service_time'] = 0.0
            kpis['median_service_time'] = 0.0
        
        # Waiting time metrics (between tasks in same case)
        waiting_times = self._compute_waiting_times(tasks_df, case_column)
        if len(waiting_times) > 0:
            kpis['mean_waiting_time'] = waiting_times.mean()
            kpis['median_waiting_time'] = waiting_times.median()
        else:
            kpis['mean_waiting_time'] = 0.0
            kpis['median_waiting_time'] = 0.0
        
        # Resource utilization metrics
        kpis['total_resources'] = tasks_df[resource_column].nunique()
        utilization = self._compute_resource_utilization(
            tasks_df, resource_column, simulation_duration
        )
        if len(utilization) > 0:
            kpis['mean_resource_utilization'] = utilization.mean()
            kpis['std_resource_utilization'] = utilization.std()
        else:
            kpis['mean_resource_utilization'] = 0.0
            kpis['std_resource_utilization'] = 0.0
        
        # Activity and event counts
        kpis['total_activities'] = tasks_df[activity_column].nunique()
        kpis['total_events'] = len(log_df)
        
        # Daily clearance KPIs
        clearance_kpis = self._compute_daily_clearance(
            log_df, tasks_df, case_column, task_column,
            timestamp_column, lifecycle_column, simulation_start, simulation_end
        )
        kpis.update(clearance_kpis)
        
        return kpis
    
    def _empty_kpis(self) -> Dict[str, float]:
        """Return empty KPIs for empty logs."""
        return {
            'total_cases': 0,
            'completed_cases': 0,
            'throughput': 0.0,
            'mean_cycle_time': 0.0,
            'median_cycle_time': 0.0,
            'std_cycle_time': 0.0,
            'p95_cycle_time': 0.0,
            'mean_service_time': 0.0,
            'median_service_time': 0.0,
            'mean_waiting_time': 0.0,
            'median_waiting_time': 0.0,
            'total_resources': 0,
            'mean_resource_utilization': 0.0,
            'std_resource_utilization': 0.0,
            'total_activities': 0,
            'total_events': 0,
            # Daily clearance KPIs
            'mean_daily_clearance_rate': 0.0,
            'min_daily_clearance_rate': 0.0,
            'pct_days_fully_cleared': 0.0,
            'mean_daily_backlog': 0.0,
            'max_daily_backlog': 0.0,
            'mean_queue_wait_time': 0.0,
            'median_queue_wait_time': 0.0,
            'p95_queue_wait_time': 0.0,
            'total_queued_tasks': 0,
            'total_completed_tasks': 0,
            'overall_completion_rate': 0.0,
        }
    
    def _create_task_pairs(
        self,
        log_df: pd.DataFrame,
        task_column: str,
        timestamp_column: str,
        lifecycle_column: str,
        case_column: str,
        activity_column: str,
        resource_column: str
    ) -> pd.DataFrame:
        """
        Create dataframe with one row per task containing start and complete times.
        
        Returns:
            DataFrame with columns: case_id, task_id, activity, resource, 
                                   start_time, complete_time, duration
        """
        # Separate start and complete events
        starts = log_df[log_df[lifecycle_column] == 'start'].copy()
        completes = log_df[log_df[lifecycle_column] == 'complete'].copy()
        
        # Merge on task_id to pair start and complete
        tasks = pd.merge(
            starts[[task_column, case_column, activity_column, resource_column, timestamp_column]],
            completes[[task_column, timestamp_column]],
            on=task_column,
            suffixes=('_start', '_complete')
        )
        
        # Rename columns for clarity
        tasks = tasks.rename(columns={
            f'{timestamp_column}_start': 'start_time',
            f'{timestamp_column}_complete': 'complete_time'
        })
        
        # Compute duration
        tasks['duration'] = tasks['complete_time'] - tasks['start_time']
        
        # Filter out invalid durations
        tasks = tasks[tasks['duration'] > 0].copy()
        
        # Sort by case and start time
        tasks = tasks.sort_values([case_column, 'start_time'])
        
        return tasks
    
    def _compute_cycle_times(self, tasks_df: pd.DataFrame, case_column: str) -> pd.Series:
        """
        Compute cycle time for each case (time from first start to last complete).
        
        Returns:
            Series of cycle times per case
        """
        # Use agg instead of apply to avoid FutureWarning
        cycle_times = (
            tasks_df.groupby(case_column).agg(
                first_start=('start_time', 'min'),
                last_complete=('complete_time', 'max')
            )
            .apply(lambda row: row['last_complete'] - row['first_start'], axis=1)
        )
        return cycle_times[cycle_times > 0]
    
    def _compute_waiting_times(self, tasks_df: pd.DataFrame, case_column: str) -> pd.Series:
        """
        Compute waiting time between consecutive tasks within each case.
        Waiting time = next_start_time - current_complete_time
        
        Returns:
            Series of waiting times
        """
        waiting_times = []
        
        for case_id, group in tasks_df.groupby(case_column):
            # Sort by complete time
            group_sorted = group.sort_values('complete_time')
            
            # Compute waiting times between consecutive tasks
            complete_times = group_sorted['complete_time'].values
            start_times = group_sorted['start_time'].values
            
            # For each task (except last), compute time until next task starts
            for i in range(len(group_sorted) - 1):
                wait = start_times[i + 1] - complete_times[i]
                if wait > 0:
                    waiting_times.append(wait)
        
        return pd.Series(waiting_times) if waiting_times else pd.Series([], dtype=float)
    
    def _compute_resource_utilization(
        self,
        tasks_df: pd.DataFrame,
        resource_column: str,
        simulation_duration: float
    ) -> pd.Series:
        """
        Compute utilization for each resource (busy time / simulation time).
        
        Returns:
            Series of utilization per resource
        """
        # Group by resource and sum durations
        busy_times = tasks_df.groupby(resource_column)['duration'].sum()
        
        # Utilization = busy time / total time
        utilization = busy_times / simulation_duration
        
        return utilization
    
    def _compute_daily_clearance(
        self,
        log_df: pd.DataFrame,
        tasks_df: pd.DataFrame,
        case_column: str,
        task_column: str,
        timestamp_column: str,
        lifecycle_column: str,
        simulation_start: float,
        simulation_end: float
    ) -> Dict[str, float]:
        """
        Compute daily clearance KPIs: how well does the scheduler clear each
        day's queued tasks within that day?

        A "day" is a 24-hour window starting at simulation_start.
        A task is "queued on day d" if its queued event falls in [d*24, (d+1)*24).
        A task is "cleared on day d" if its complete event also falls in that window.

        Returns dict of clearance KPIs.
        """
        kpis: Dict[str, float] = {}

        # --- queue wait time (queued -> start) -----------------------------------
        queued_events = log_df[log_df[lifecycle_column] == 'queued'].copy()
        if queued_events.empty:
            # No queued events – fall back to zeros
            kpis['mean_queue_wait_time'] = 0.0
            kpis['median_queue_wait_time'] = 0.0
            kpis['p95_queue_wait_time'] = 0.0
            kpis['mean_daily_clearance_rate'] = 0.0
            kpis['min_daily_clearance_rate'] = 0.0
            kpis['pct_days_fully_cleared'] = 0.0
            kpis['mean_daily_backlog'] = 0.0
            kpis['max_daily_backlog'] = 0.0
            kpis['total_queued_tasks'] = 0
            kpis['total_completed_tasks'] = int(len(tasks_df))
            kpis['overall_completion_rate'] = 1.0 if len(tasks_df) > 0 else 0.0
            return kpis

        # Join queued timestamp to the paired tasks_df
        queued_times = queued_events.set_index(task_column)[timestamp_column].rename('queued_time')
        tasks_with_queue = tasks_df.merge(
            queued_times, left_on=task_column, right_index=True, how='left'
        )

        # Queue wait = start_time - queued_time
        tasks_with_queue['queue_wait'] = (
            tasks_with_queue['start_time'] - tasks_with_queue['queued_time']
        )
        valid_waits = tasks_with_queue['queue_wait'].dropna()
        valid_waits = valid_waits[valid_waits >= 0]

        if len(valid_waits) > 0:
            kpis['mean_queue_wait_time'] = float(valid_waits.mean())
            kpis['median_queue_wait_time'] = float(valid_waits.median())
            kpis['p95_queue_wait_time'] = float(valid_waits.quantile(0.95))
        else:
            kpis['mean_queue_wait_time'] = 0.0
            kpis['median_queue_wait_time'] = 0.0
            kpis['p95_queue_wait_time'] = 0.0

        # --- daily clearance rate ------------------------------------------------
        # Assign each queued event to a day (day = 24-hour window)
        queued_events = queued_events.copy()
        queued_events['day'] = ((queued_events[timestamp_column] - simulation_start) / HOURS_PER_DAY).astype(int)

        # For completed tasks, map task_id -> complete_time (handle duplicates with groupby)
        complete_map = tasks_df.groupby(task_column)['complete_time'].last()

        # Per day: count queued and cleared
        total_days = int((simulation_end - simulation_start) / HOURS_PER_DAY) + 1
        daily_clearance_rates = []
        daily_backlogs = []

        for day in range(total_days):
            day_start = simulation_start + day * HOURS_PER_DAY
            day_end = day_start + HOURS_PER_DAY

            # Tasks queued on this day
            day_queued = queued_events[
                (queued_events[timestamp_column] >= day_start)
                & (queued_events[timestamp_column] < day_end)
            ]
            n_queued = len(day_queued)
            if n_queued == 0:
                continue  # skip days with no arrivals

            # How many of those were completed before end of day?
            queued_task_ids = day_queued[task_column].values
            completed_times = complete_map.reindex(queued_task_ids)
            n_cleared = int((completed_times.dropna() < day_end).sum())

            clearance_rate = n_cleared / n_queued
            daily_clearance_rates.append(clearance_rate)
            daily_backlogs.append(n_queued - n_cleared)

        if daily_clearance_rates:
            kpis['mean_daily_clearance_rate'] = float(np.mean(daily_clearance_rates))
            kpis['min_daily_clearance_rate'] = float(np.min(daily_clearance_rates))
            kpis['pct_days_fully_cleared'] = float(
                np.mean([1.0 if r >= 1.0 else 0.0 for r in daily_clearance_rates])
            )
        else:
            kpis['mean_daily_clearance_rate'] = 0.0
            kpis['min_daily_clearance_rate'] = 0.0
            kpis['pct_days_fully_cleared'] = 0.0

        if daily_backlogs:
            kpis['mean_daily_backlog'] = float(np.mean(daily_backlogs))
            kpis['max_daily_backlog'] = float(np.max(daily_backlogs))
        else:
            kpis['mean_daily_backlog'] = 0.0
            kpis['max_daily_backlog'] = 0.0

        # --- overall completion ---
        total_queued = len(queued_events)
        total_completed = len(tasks_df)
        kpis['total_queued_tasks'] = int(total_queued)
        kpis['total_completed_tasks'] = int(total_completed)
        kpis['overall_completion_rate'] = (
            total_completed / total_queued if total_queued > 0 else 0.0
        )

        return kpis

    def compute_from_file(self, log_filepath: Path, **kwargs) -> Dict[str, float]:
        """
        Compute KPIs from CSV log file.
        
        Args:
            log_filepath: Path to event log CSV
            **kwargs: Additional arguments for compute_all
            
        Returns:
            Dictionary of KPIs
        """
        log_df = pd.read_csv(log_filepath)
        return self.compute_all(log_df, **kwargs)

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        """Return numerator/denominator with zero-safe fallback."""
        if denominator <= 0:
            return 0.0
        return numerator / denominator

    def compute_from_daily_summary_file(self, summary_filepath: Path) -> Dict[str, Any]:
        """Compute scheduler KPIs from compact daily summary JSONL.

        This is the preferred source for scheduling KPIs because it includes
        both solver-time facts (dummy deferrals) and end-of-day facts
        (incomplete queued tasks and drop events).
        """
        rows: list[Dict[str, Any]] = []
        with open(summary_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

        if not rows:
            return {
                'daily_rows': 0,
                'task_deferral_rate': 0.0,
                'task_incompletion_rate': 0.0,
                'task_drop_rate': 0.0,
                'task_deferral_rate_per_activity': {},
                'task_incompletion_rate_per_activity': {},
                'task_drop_rate_per_activity': {},
                'mean_resource_utilization': 0.0,
                'std_resource_utilization': 0.0,
                'resource_utilization_cv': 0.0,
                'scheduler_composite_score': 0.0,
            }

        total_tasks = 0.0
        total_assigned_real = 0.0
        total_assigned_dummy = 0.0
        total_incomplete = 0.0
        total_dropped = 0.0

        activity_total_tasks: Dict[str, float] = defaultdict(float)
        activity_assigned_real: Dict[str, float] = defaultdict(float)
        activity_assigned_dummy: Dict[str, float] = defaultdict(float)
        activity_incomplete: Dict[str, float] = defaultdict(float)
        activity_dropped: Dict[str, float] = defaultdict(float)

        util_values: list[float] = []

        activity_actual_hours: Dict[str, float] = defaultdict(float)
        activity_capacity_hours: Dict[str, float] = defaultdict(float)

        for row in rows:
            tasks_total_day = float(row.get('tasks_total', 0.0) or 0.0)
            assigned_real_day = float(row.get('assigned_real_count', 0.0) or 0.0)
            assigned_dummy_day = float(row.get('assigned_dummy_count', 0.0) or 0.0)
            incomplete_day = float(row.get('incomplete_queued_count', 0.0) or 0.0)
            dropped_day = float(row.get('dropped_count', 0.0) or 0.0)

            total_tasks += tasks_total_day
            total_assigned_real += assigned_real_day
            total_assigned_dummy += assigned_dummy_day
            total_incomplete += incomplete_day
            total_dropped += dropped_day

            for activity, count in (row.get('tasks_per_activity') or {}).items():
                activity_total_tasks[str(activity)] += float(count or 0.0)
            for activity, count in (row.get('assigned_real_per_activity') or {}).items():
                activity_assigned_real[str(activity)] += float(count or 0.0)
            for activity, count in (row.get('assigned_dummy_per_activity') or {}).items():
                activity_assigned_dummy[str(activity)] += float(count or 0.0)
            for activity, count in (row.get('incomplete_queued_per_activity') or {}).items():
                activity_incomplete[str(activity)] += float(count or 0.0)
            for activity, count in (row.get('dropped_per_activity') or {}).items():
                activity_dropped[str(activity)] += float(count or 0.0)

            for util in (row.get('resource_utilization_per_resource') or {}).values():
                if isinstance(util, (list, tuple)):
                    util_values.append(float(util[0] if len(util) > 0 else 0.0))
                else:
                    util_values.append(float(util or 0.0))

            for activity, hours in (row.get('actual_task_hours_per_activity') or {}).items():
                activity_actual_hours[str(activity)] += float(hours or 0.0)
            for activity, hours in (row.get('available_capacity_hours_per_activity') or {}).items():
                activity_capacity_hours[str(activity)] += float(hours or 0.0)

        task_deferral_rate = self._safe_div(total_assigned_dummy, total_tasks)
        task_incompletion_rate = self._safe_div(total_incomplete, total_assigned_real)
        task_drop_rate = self._safe_div(total_dropped, total_tasks)

        all_activities = sorted(set(activity_total_tasks) | set(activity_assigned_real) | set(activity_assigned_dummy))
        task_deferral_rate_per_activity = {
            a: self._safe_div(activity_assigned_dummy.get(a, 0.0), activity_total_tasks.get(a, 0.0))
            for a in all_activities
        }
        task_incompletion_rate_per_activity = {
            a: self._safe_div(activity_incomplete.get(a, 0.0), activity_assigned_real.get(a, 0.0))
            for a in all_activities
        }
        task_drop_rate_per_activity = {
            a: self._safe_div(activity_dropped.get(a, 0.0), activity_total_tasks.get(a, 0.0))
            for a in all_activities
        }

        mean_resource_utilization = float(np.mean(util_values)) if util_values else 0.0
        std_resource_utilization = float(np.std(util_values)) if util_values else 0.0
        resource_utilization_cv = self._safe_div(std_resource_utilization, mean_resource_utilization)

        activity_utilization_actual = {
            a: self._safe_div(activity_actual_hours.get(a, 0.0), activity_capacity_hours.get(a, 0.0))
            for a in sorted(set(activity_actual_hours) | set(activity_capacity_hours))
        }

        norm_util_std = min(1.0, resource_utilization_cv)
        scheduler_composite_score = (
            0.25 * (1.0 - task_drop_rate)
            + 0.25 * (1.0 - task_deferral_rate)
            + 0.25 * mean_resource_utilization
            + 0.25 * (1.0 - norm_util_std)
        )

        return {
            'daily_rows': len(rows),
            'task_deferral_rate': float(task_deferral_rate),
            'task_incompletion_rate': float(task_incompletion_rate),
            'task_drop_rate': float(task_drop_rate),
            'task_deferral_rate_per_activity': task_deferral_rate_per_activity,
            'task_incompletion_rate_per_activity': task_incompletion_rate_per_activity,
            'task_drop_rate_per_activity': task_drop_rate_per_activity,
            'mean_resource_utilization': float(mean_resource_utilization),
            'std_resource_utilization': float(std_resource_utilization),
            'resource_utilization_cv': float(resource_utilization_cv),
            'activity_utilization_actual': activity_utilization_actual,
            'scheduler_composite_score': float(scheduler_composite_score),
        }

    def load_daily_summary_dataframe(self, summary_filepath: Path) -> pd.DataFrame:
        """Load daily summary JSONL into a plot-ready dataframe.

        The returned frame flattens nested day-summary dictionaries into
        columns so notebooks can plot directly without re-parsing JSONL or
        re-implementing KPI ratios.
        """
        rows: list[Dict[str, Any]] = []
        with open(summary_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

        if not rows:
            return pd.DataFrame()

        df = pd.json_normalize(rows, sep='__')

        if 'day_index' in df.columns:
            df = df.sort_values(['day_index'])
        elif 'sim_time_hours' in df.columns:
            df = df.sort_values(['sim_time_hours'])
        df = df.reset_index(drop=True)

        def numeric_column(column_name: str) -> pd.Series:
            if column_name not in df.columns:
                return pd.Series(0.0, index=df.index, dtype=float)
            return pd.to_numeric(df[column_name], errors='coerce').fillna(0.0)

        def safe_ratio(numerator: str, denominator: str) -> pd.Series:
            numerator_values = numeric_column(numerator)
            denominator_values = numeric_column(denominator)
            denominator_values = denominator_values.where(denominator_values > 0, np.nan)
            return numerator_values.div(denominator_values).fillna(0.0)

        df['task_deferral_rate'] = safe_ratio('assigned_dummy_count', 'tasks_total')
        df['task_incompletion_rate'] = safe_ratio('incomplete_queued_count', 'assigned_real_count')
        df['task_drop_rate'] = safe_ratio('dropped_count', 'tasks_total')
        df['resource_utilization_cv'] = safe_ratio('std_resource_utilization', 'mean_resource_utilization')

        if 'assigned_real_pct' not in df.columns:
            df['assigned_real_pct'] = safe_ratio('assigned_real_count', 'tasks_total') * 100.0
        if 'assigned_dummy_pct' not in df.columns:
            df['assigned_dummy_pct'] = safe_ratio('assigned_dummy_count', 'tasks_total') * 100.0

        return df
