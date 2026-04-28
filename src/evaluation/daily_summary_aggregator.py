"""Daily summary aggregation helpers for simulation KPI logging.

This module owns the pure aggregation logic used to build the compact daily
summary JSONL that feeds KPI calculation and plotting.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

from ..entities.resource import Resource
from ..entities.task import Task
from ..scheduling.base import SchedulingContext
from ..scheduling.experience_based import DUMMY_RESOURCE_ID
from ..utils.time_utils import SECONDS_PER_DAY, SimulationTimeConverter, seconds_to_hours
from .daily_summary_logger import DailySummaryLogger


class DailySummaryAggregator:
    """Build and finalize compact daily KPI summaries."""

    def __init__(
        self,
        *,
        scheduler: Any,
        resources: Dict[str, Resource],
        resource_calendars: Optional[Dict[str, Any]],
        time_converter: Optional[SimulationTimeConverter],
        enable_working_hours: bool,
        planning_horizon_seconds: float,
        max_task_deferrals: int,
        daily_task_durations: Dict[str, List[tuple]],
        state: Any,
        env: Any,
        daily_summary_logger: Optional[DailySummaryLogger],
    ) -> None:
        self.scheduler = scheduler
        self.resources: Dict[str, Resource] = resources
        self.resource_calendars = resource_calendars or {}
        self.time_converter = time_converter
        self.enable_working_hours = enable_working_hours
        self.planning_horizon_seconds = planning_horizon_seconds
        self.max_task_deferrals = max_task_deferrals
        self.daily_task_durations = daily_task_durations
        self.state = state
        self.env = env
        self.daily_summary_logger = daily_summary_logger

        self._open_daily_summary: Optional[Dict[str, Any]] = None
        self._open_daily_day_start: Optional[float] = None
        self._open_daily_day_end: Optional[float] = None

    def get_activity_requirements(self) -> Dict[str, float]:
        """Return activity requirements exposed by the active scheduler."""
        req = getattr(self.scheduler, 'activity_requirements', None)
        if isinstance(req, dict):
            return {str(k): float(v) for k, v in req.items()}
        return {}

    def build_activity_catalog(self, pending_tasks: List[Task]) -> List[str]:
        """Build stable activity catalog for day-level aggregates."""
        activity_set = set(self.get_activity_requirements().keys())
        for task in pending_tasks:
            if getattr(task, 'activity_name', None):
                activity_set.add(str(task.activity_name))
        for rid, resource in self.resources.items():
            if resource.experience_store is not None:
                capabilities_dict = resource.experience_store.get_resource_capabilities_dict(rid)
                for activity_name in capabilities_dict.keys():
                    if activity_name:
                        activity_set.add(str(activity_name))
        return sorted(activity_set)

    def _available_capacity_seconds(self, resource_id: str, day_start: float, day_end: float) -> float:
        """Return available capacity for one resource in the day window."""
        if self.enable_working_hours and self.time_converter:
            calendar = self.resource_calendars.get(resource_id)
            if calendar is not None:
                return float(
                    calendar.get_available_time_in_seconds(
                        start_sim=day_start,
                        end_sim=day_end,
                        time_converter=self.time_converter,
                        neglect_sick_leave=False,
                    )
                )
        return float(max(0.0, day_end - day_start))

    def build_capacity_snapshots(
        self,
        activity_catalog: List[str],
        day_start: float,
        day_end: float,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int], float]:
        """Build capacity aggregates needed by daily KPI summaries."""
        req = self.get_activity_requirements()
        default_req = 50.0

        capacity_hours_by_resource: Dict[str, float] = {}
        for rid in self.resources.keys():
            cap_seconds = self._available_capacity_seconds(rid, day_start, day_end)
            capacity_hours_by_resource[rid] = round(seconds_to_hours(cap_seconds), 6)

        overall_capacity_hours = round(sum(capacity_hours_by_resource.values()), 6)

        available_capacity_hours_by_activity: Dict[str, float] = {}
        capable_resources_by_activity: Dict[str, List[str]] = {}
        for activity in activity_catalog:
            required = float(req.get(activity, default_req))
            capable_ids = [
                rid for rid, resource in self.resources.items()
                if resource.can_perform(activity, required_level=required)
            ]
            capable_resources_by_activity[activity] = capable_ids
            available_capacity_hours_by_activity[activity] = round(
                sum(capacity_hours_by_resource.get(rid, 0.0) for rid in capable_ids),
                6,
            )

        return (
            capacity_hours_by_resource,
            available_capacity_hours_by_activity,
            capable_resources_by_activity,
            overall_capacity_hours,
        )

    def start_daily_summary(self, context: SchedulingContext) -> None:
        """Create a new open day summary before the solver is called."""
        if self.daily_summary_logger is None:
            return

        day_start = float(self.env.now)
        day_end = day_start + float(self.planning_horizon_seconds)
        pending_tasks = list(context.pending_tasks)
        activity_catalog = self.build_activity_catalog(pending_tasks)

        tasks_per_activity: Dict[str, int] = defaultdict(int)
        for task in pending_tasks:
            tasks_per_activity[str(task.activity_name)] += 1

        (
            capacity_hours_by_resource,
            available_capacity_hours_by_activity,
            capable_resources_by_activity,
            overall_capacity_hours,
        ) = self.build_capacity_snapshots(activity_catalog, day_start, day_end)

        self._open_daily_summary = {
            'day_index': int(day_start // SECONDS_PER_DAY),
            'sim_time_hours': round(seconds_to_hours(day_start), 6),
            'sim_daytime': self.time_converter.sim_time_to_datetime(day_start).isoformat() if self.time_converter else None,
            'is_partial_day': False,
            'scheduler_label': getattr(self.scheduler, 'name', self.scheduler.__class__.__name__),
            'solver_status': 'UNKNOWN',
            'solver_wall_time_seconds': 0.0,
            'solver_failed': False,
            'tasks_total': len(pending_tasks),
            'tasks_per_activity': dict(tasks_per_activity),
            'assigned_real_count': 0,
            'assigned_dummy_count': 0,
            'assigned_real_pct': 0.0,
            'assigned_dummy_pct': 0.0,
            'assigned_real_per_activity': {},
            'assigned_dummy_per_activity': {},
            'task_demand_count_total': 0,
            'task_demand_hours_total': 0.0,
            'deferred_task_demand_count_total': 0,
            'deferred_task_demand_hours_total': 0.0,
            'task_demand_count_per_activity': {},
            'task_demand_hours_per_activity': {},
            'deferred_task_demand_count_per_activity': {},
            'deferred_task_demand_hours_per_activity': {},
            'estimated_task_hours_per_activity': {},
            'actual_task_hours_per_activity': {},
            'actual_completed_tasks_per_activity': {},
            'incomplete_queued_count': 0,
            'incomplete_queued_per_activity': {},
            'dropped_count': 0,
            'dropped_per_activity': {},
            'available_capacity_hours_total': overall_capacity_hours,
            'available_capacity_hours_per_activity': available_capacity_hours_by_activity,
            'capable_resources_by_activity': capable_resources_by_activity,
            'assignment_count_per_activity_resource': {},
            'mentoring_pairs_per_activity': {},
            '_capacity_hours_by_resource': capacity_hours_by_resource,
            '_dropped_from_dummy_per_activity': {},
        }
        self._open_daily_day_start = day_start
        self._open_daily_day_end = day_end

    def summarize_assignments(self, assignments: Dict[str, List[Task]]) -> Dict[str, Any]:
        """Build compact assignment aggregates for the currently open day."""
        summary: Dict[str, Any] = {
            'assigned_real_count': 0,
            'assigned_dummy_count': 0,
            'assigned_real_per_activity': defaultdict(int),
            'assigned_dummy_per_activity': defaultdict(int),
            'task_demand_count_total': 0,
            'task_demand_hours_total': 0.0,
            'deferred_task_demand_count_total': 0,
            'deferred_task_demand_hours_total': 0.0,
            'task_demand_count_per_activity': defaultdict(int),
            'task_demand_hours_per_activity': defaultdict(float),
            'deferred_task_demand_count_per_activity': defaultdict(int),
            'deferred_task_demand_hours_per_activity': defaultdict(float),
            'estimated_task_hours_per_activity': defaultdict(float),
            'assignment_count_per_activity_resource': defaultdict(lambda: defaultdict(int)),
            'mentoring_pairs_per_activity': defaultdict(lambda: defaultdict(lambda: defaultdict(int))),
            'dropped_from_dummy_per_activity': defaultdict(int),
        }

        deferred_tasks = assignments.get(DUMMY_RESOURCE_ID, [])
        summary['assigned_dummy_count'] = len(deferred_tasks)
        for task in deferred_tasks:
            activity = str(task.activity_name)
            summary['assigned_dummy_per_activity'][activity] += 1
            summary['task_demand_count_total'] += 1
            summary['task_demand_count_per_activity'][activity] += 1
            summary['deferred_task_demand_count_total'] += 1
            summary['deferred_task_demand_count_per_activity'][activity] += 1
            est_hours = seconds_to_hours(float(task.estimated_duration or 0.0))
            summary['task_demand_hours_total'] += est_hours
            summary['task_demand_hours_per_activity'][activity] += est_hours
            summary['deferred_task_demand_hours_total'] += est_hours
            summary['deferred_task_demand_hours_per_activity'][activity] += est_hours
            summary['estimated_task_hours_per_activity'][activity] += est_hours
            summary['assignment_count_per_activity_resource'][activity][DUMMY_RESOURCE_ID] += 1
            if task.defer_count + 1 > self.max_task_deferrals:
                summary['dropped_from_dummy_per_activity'][activity] += 1

        for rid, tasks in assignments.items():
            if rid == DUMMY_RESOURCE_ID:
                continue
            for task in tasks:
                activity = str(task.activity_name)
                summary['assigned_real_count'] += 1
                summary['assigned_real_per_activity'][activity] += 1
                summary['task_demand_count_total'] += 1
                summary['task_demand_count_per_activity'][activity] += 1
                est_hours = seconds_to_hours(float(task.estimated_duration or 0.0))
                summary['task_demand_hours_total'] += est_hours
                summary['task_demand_hours_per_activity'][activity] += est_hours
                summary['estimated_task_hours_per_activity'][activity] += est_hours
                summary['assignment_count_per_activity_resource'][activity][rid] += 1
                if task.is_mentoring_task() and task.mentor_resource_id and task.mentee_resource_id:
                    summary['mentoring_pairs_per_activity'][activity][task.mentor_resource_id][task.mentee_resource_id] += 1

        summary['task_demand_count_per_activity'] = {
            k: int(v) for k, v in summary['task_demand_count_per_activity'].items()
        }
        summary['task_demand_hours_per_activity'] = {
            k: round(v, 6) for k, v in summary['task_demand_hours_per_activity'].items()
        }
        summary['task_demand_count_total'] = int(summary['task_demand_count_total'])
        summary['task_demand_hours_total'] = round(float(summary['task_demand_hours_total']), 6)
        summary['deferred_task_demand_count_per_activity'] = {
            k: int(v) for k, v in summary['deferred_task_demand_count_per_activity'].items()
        }
        summary['deferred_task_demand_hours_per_activity'] = {
            k: round(v, 6) for k, v in summary['deferred_task_demand_hours_per_activity'].items()
        }
        summary['deferred_task_demand_count_total'] = int(summary['deferred_task_demand_count_total'])
        summary['deferred_task_demand_hours_total'] = round(float(summary['deferred_task_demand_hours_total']), 6)
        summary['estimated_task_hours_per_activity'] = {
            k: round(v, 6) for k, v in summary['estimated_task_hours_per_activity'].items()
        }
        summary['assigned_real_per_activity'] = dict(summary['assigned_real_per_activity'])
        summary['assigned_dummy_per_activity'] = dict(summary['assigned_dummy_per_activity'])
        summary['assignment_count_per_activity_resource'] = {
            activity: dict(resource_counts)
            for activity, resource_counts in summary['assignment_count_per_activity_resource'].items()
        }
        summary['mentoring_pairs_per_activity'] = {
            activity: {
                mentor: dict(mentees)
                for mentor, mentees in mentors.items()
            }
            for activity, mentors in summary['mentoring_pairs_per_activity'].items()
        }
        summary['dropped_from_dummy_per_activity'] = dict(summary['dropped_from_dummy_per_activity'])
        return summary

    def merge_assignment_summary(self, assignment_summary: Dict[str, Any]) -> None:
        """Attach assignment aggregates to the open day summary."""
        if self._open_daily_summary is None:
            return
        self._open_daily_summary['assigned_real_count'] = int(assignment_summary.get('assigned_real_count', 0))
        self._open_daily_summary['assigned_dummy_count'] = int(assignment_summary.get('assigned_dummy_count', 0))
        self._open_daily_summary['assigned_real_per_activity'] = assignment_summary.get('assigned_real_per_activity', {})
        self._open_daily_summary['assigned_dummy_per_activity'] = assignment_summary.get('assigned_dummy_per_activity', {})
        self._open_daily_summary['task_demand_count_total'] = int(assignment_summary.get('task_demand_count_total', 0))
        self._open_daily_summary['task_demand_hours_total'] = float(assignment_summary.get('task_demand_hours_total', 0.0))
        self._open_daily_summary['deferred_task_demand_count_total'] = int(assignment_summary.get('deferred_task_demand_count_total', 0))
        self._open_daily_summary['deferred_task_demand_hours_total'] = float(assignment_summary.get('deferred_task_demand_hours_total', 0.0))
        self._open_daily_summary['task_demand_count_per_activity'] = assignment_summary.get('task_demand_count_per_activity', {})
        self._open_daily_summary['task_demand_hours_per_activity'] = assignment_summary.get('task_demand_hours_per_activity', {})
        self._open_daily_summary['deferred_task_demand_count_per_activity'] = assignment_summary.get('deferred_task_demand_count_per_activity', {})
        self._open_daily_summary['deferred_task_demand_hours_per_activity'] = assignment_summary.get('deferred_task_demand_hours_per_activity', {})
        self._open_daily_summary['estimated_task_hours_per_activity'] = assignment_summary.get('estimated_task_hours_per_activity', {})
        self._open_daily_summary['assignment_count_per_activity_resource'] = assignment_summary.get('assignment_count_per_activity_resource', {})
        self._open_daily_summary['mentoring_pairs_per_activity'] = assignment_summary.get('mentoring_pairs_per_activity', {})
        self._open_daily_summary['_dropped_from_dummy_per_activity'] = assignment_summary.get('dropped_from_dummy_per_activity', {})

        tasks_total = max(int(self._open_daily_summary.get('tasks_total', 0)), 1)
        self._open_daily_summary['assigned_real_pct'] = round(
            100.0 * self._open_daily_summary['assigned_real_count'] / tasks_total,
            4,
        )
        self._open_daily_summary['assigned_dummy_pct'] = round(
            100.0 * self._open_daily_summary['assigned_dummy_count'] / tasks_total,
            4,
        )

    def compute_actual_task_completion_stats(self, day_start: float, day_end: float) -> Tuple[Dict[str, float], Dict[str, int]]:
        """Compute actual completed task hours/counts for tasks ending in [day_start, day_end)."""
        hours_by_activity: Dict[str, float] = defaultdict(float)
        count_by_activity: Dict[str, int] = defaultdict(int)
        for tid, task in self.state.tasks.items():
            if task.actual_end_time is None or task.actual_start_time is None:
                continue
            if not (day_start <= task.actual_end_time < day_end):
                continue
            activity = str(task.activity_name)
            duration_seconds = max(0.0, float(task.actual_end_time - task.actual_start_time))
            hours_by_activity[activity] += seconds_to_hours(duration_seconds)
            count_by_activity[activity] += 1
        return (
            {k: round(v, 6) for k, v in hours_by_activity.items()},
            dict(count_by_activity),
        )

    def compute_resource_utilization_snapshot(self) -> Tuple[Dict[str, tuple], float, float]:
        """Compute per-resource utilization and direct estimated/actual hours.

        Returns a mapping per resource with tuple values:
        (utilization_ratio, estimated_task_hours, actual_task_hours).
        """
        if self._open_daily_summary is None:
            return {}, 0.0, 0.0

        capacity_hours_by_resource = self._open_daily_summary.get('_capacity_hours_by_resource', {})
        used_hours_by_resource: Dict[str, float] = {}
        estimated_hours_by_resource: Dict[str, float] = {}
        for rid in self.resources.keys():
            # daily task duration structure (tid, activity, estimated duration, actual duration)
            estimated_seconds = sum(entry[2] for entry in self.daily_task_durations.get(rid, []))
            used_seconds = sum(entry[3] for entry in self.daily_task_durations.get(rid, []))
            used_hours_by_resource[rid] = round(seconds_to_hours(float(used_seconds)), 6)
            estimated_hours_by_resource[rid] = round(seconds_to_hours(float(estimated_seconds)), 6)

        util_values: List[float] = []
        util_by_resource: Dict[str, tuple] = {}
        for rid, used_h in used_hours_by_resource.items():
            cap_h = float(capacity_hours_by_resource.get(rid, 0.0))
            util = (used_h / cap_h) if cap_h > 0 else 0.0
            util_rounded = round(util, 6)
            est_rounded = round(float(estimated_hours_by_resource[rid]), 6)
            used_rounded = round(float(used_h), 6)
            util_by_resource[rid] = (util_rounded, est_rounded, used_rounded)
            util_values.append(util_rounded)

        mean_util = float(mean(util_values)) if util_values else 0.0
        std_util = float(pstdev(util_values)) if len(util_values) > 1 else 0.0
        return util_by_resource, round(mean_util, 6), round(std_util, 6)

    def finalize_open_daily_summary(
        self,
        incomplete_by_activity: Optional[Dict[str, int]] = None,
        dropped_from_drain_by_activity: Optional[Dict[str, int]] = None,
        is_partial_day: bool = False,
    ) -> None:
        """Finalize the open day summary and append one compact JSONL row."""
        if self.daily_summary_logger is None or self._open_daily_summary is None:
            return

        incomplete_by_activity = incomplete_by_activity or {}
        dropped_from_drain_by_activity = dropped_from_drain_by_activity or {}

        day_start = float(self._open_daily_day_start or 0.0)
        day_end = float(self._open_daily_day_end or day_start)

        actual_hours_by_activity, completed_count_by_activity = self.compute_actual_task_completion_stats(day_start, day_end)
        util_by_resource, mean_util, std_util = self.compute_resource_utilization_snapshot()

        dropped_total = 0
        dropped_per_activity: Dict[str, int] = defaultdict(int)
        for activity, count in self._open_daily_summary.get('_dropped_from_dummy_per_activity', {}).items():
            dropped_per_activity[activity] += int(count)
            dropped_total += int(count)
        for activity, count in dropped_from_drain_by_activity.items():
            dropped_per_activity[activity] += int(count)
            dropped_total += int(count)

        self._open_daily_summary['is_partial_day'] = bool(is_partial_day)
        self._open_daily_summary['incomplete_queued_per_activity'] = {
            k: int(v) for k, v in incomplete_by_activity.items()
        }
        self._open_daily_summary['incomplete_queued_count'] = int(sum(incomplete_by_activity.values()))
        self._open_daily_summary['dropped_per_activity'] = dict(dropped_per_activity)
        self._open_daily_summary['dropped_count'] = int(dropped_total)
        self._open_daily_summary['actual_task_hours_per_activity'] = actual_hours_by_activity
        self._open_daily_summary['actual_completed_tasks_per_activity'] = completed_count_by_activity
        self._open_daily_summary['resource_utilization_per_resource'] = util_by_resource
        self._open_daily_summary['mean_resource_utilization'] = mean_util
        self._open_daily_summary['std_resource_utilization'] = std_util

        # Merge scheduling-context metrics from the last MKPFormulator run (if available)
        formulator_metrics: Dict[str, Any] = getattr(
            self.scheduler, '_last_formulator_metrics', {}
        ) or {}
        self._open_daily_summary['resource_pressure_per_resource'] = (
            formulator_metrics.get('resource_pressure_per_resource', {})
        )
        self._open_daily_summary['demand_supply_ratio_per_activity'] = (
            formulator_metrics.get('demand_supply_ratio_per_activity', {})
        )
        self._open_daily_summary['capable_resources_per_activity'] = (
            formulator_metrics.get('capable_resources_per_activity', {})
        )
        self._open_daily_summary['activity_demand_score_per_activity'] = (
            formulator_metrics.get('activity_demand_score_per_activity', {})
        )
        self._open_daily_summary['mentor_substitutability_per_activity'] = (
            formulator_metrics.get('mentor_substitutability_per_activity', {})
        )
        self._open_daily_summary['strategic_shortage_activities'] = (
            formulator_metrics.get('strategic_shortage_activities', [])
        )
        self._open_daily_summary['bottleneck_per_activity'] = (
            formulator_metrics.get('bottleneck_per_activity', {})
        )
        self._open_daily_summary['underutilized_resources'] = (
            formulator_metrics.get('underutilized_resources', {})
        )

        self._open_daily_summary.pop('_capacity_hours_by_resource', None)
        self._open_daily_summary.pop('_dropped_from_dummy_per_activity', None)

        self.daily_summary_logger.log_day(self._open_daily_summary)
        self._open_daily_summary = None
        self._open_daily_day_start = None
        self._open_daily_day_end = None
