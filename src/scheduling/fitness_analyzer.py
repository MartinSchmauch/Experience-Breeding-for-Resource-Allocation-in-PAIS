"""
Task-fitness analysis and bottleneck detection for batch scheduling.

Computes per-activity capacity scores across available resources for an
upcoming time horizon and generates emergency mentoring tasks when a
bottleneck is detected.

Scoring model
=============
For each resource *r* and activity *a* a **fitness score** is computed:

    fitness(r, a) = capability_level(r, a) / required_level(a)

where *capability_level* comes from ``resource.capabilities[a]`` and
*required_level* is a configurable baseline (default 50).

The score is clamped to [0, 1].  A resource counts as **fit** when its
score reaches or exceeds a configurable *fitness_threshold* (default 0.6).

For every activity the **capacity score** is the count of fit resources
that are actually available inside the lookahead window (not on vacation,
within working hours, etc.). 

where *avg_interarrival_hours* is derived from the experience store's
observation count and the simulation time elapsed so far.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Any, Set
import logging
import uuid
import yaml
from pathlib import Path

from ..entities.resource import Resource
from ..entities.calendar import ResourceCalendar
from ..experience.store import ExperienceStore
from ..utils.time_utils import hours_to_seconds

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BottleneckInfo:
    """Describes a detected capacity bottleneck for an activity.

    Severity levels:
        - 'severe':  at least one day in the next 3 days has NO capable resource available
        - 'medium':  at least one day in the next 3 days has >50 % of capable resources absent
    """
    activity_name: str
    severity: str               # 'medium' or 'severe'
    days_until_bottleneck: int  # 1–3 (earliest day triggering classification)
    capable_resource_count: int # total capable resources for this activity
    fit_resource_ids: List[str] # resources that are capable
    mentee_candidates: Optional[List[str]] = None
    mentor_candidates: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class TaskFitnessAnalyzer:
    """
    Analyzes resource fitness per activity and detects capacity bottlenecks.

    Usage (inside ``plan_tasks_to_resources``)::

        analyzer = TaskFitnessAnalyzer(config, experience_store, ...)
        bottlenecks = analyzer.detect_bottlenecks(pending_tasks, resources, ...)
        extra_tasks  = analyzer.generate_mentoring_tasks(bottlenecks, ...)
        pending_tasks.extend(extra_tasks)
    """

    def __init__(
        self,
        mentoring_config: Dict[str, Any],
        experience_store: ExperienceStore,
        activity_requirements: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            mentoring_config:  ``config['mentoring']``
            experience_store:  Experience store for capability / duration lookup
            activity_requirements: Pre-loaded activity requirements dict.
                If None, loads from config/activity_requirements.yaml (slower).
        """
        self.mentoring_config = mentoring_config
        self.experience_store = experience_store
        self.activity_requirements = activity_requirements if activity_requirements is not None else self._load_activity_requirements()

    def _load_activity_requirements(self) -> Dict[str, Any]:
        config_path = Path(__file__).parent.parent.parent / "config" / "activity_requirements.yaml"
        
        if not config_path.exists():
            return {}
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception:
            return {}
        return config.get('activity_requirements', {})

    def detect_bottlenecks(
        self,
        resources: List[Resource],
        resource_calendars: Dict[str, ResourceCalendar],
        current_sim_time: float,
        planning_horizon_hours: float,
        time_converter: Optional[Any] = None,
        active_activities: Optional[Set[str]] = None,
    ) -> List[BottleneckInfo]:
        """Detect activity-level capacity bottlenecks via 3-day calendar inspection.

        For each activity:
            1. Determine capable resources (cap_level >= required_level from config).
            2. For each of the next 3 days, check how many capable resources
               have available working time on that day.
            3. Classify:
               - **Severe**: any day where NO capable resource is available.
               - **Medium**: any day where >50 % of capable resources are unavailable.
            4. Record the earliest day that triggers the classification.

        Mentee selection: uncapable resources sorted by capability descending
        (the closest to the threshold — best candidates for mentoring).

        Mentor selection: capable resources with sufficient experience count,
        sorted by capability descending.

        Returns:
            List of BottleneckInfo for bottlenecked activities.
        """
        SECONDS_PER_PLANNING_HORIZON = planning_horizon_hours * 3600  # seconds

        # --- Build workday boundaries (skip weekends) ---
        # Each entry: (workday_index, day_start_sim, day_end_sim)
        workday_boundaries: list[tuple[int, float, float]] = []
        if time_converter:
            current_dt = time_converter.sim_time_to_datetime(current_sim_time)
            cal_offset = 0
            while len(workday_boundaries) < 4:
                day_dt = current_dt + timedelta(days=cal_offset)
                if day_dt.weekday() < 5:  # Monday–Friday
                    day_start = current_sim_time + cal_offset * SECONDS_PER_PLANNING_HORIZON
                    day_end = day_start + SECONDS_PER_PLANNING_HORIZON
                    workday_boundaries.append(
                        (len(workday_boundaries), day_start, day_end)
                    )
                cal_offset += 1
        else:
            # Fallback: consecutive calendar days (no weekend skipping)
            for i in range(4):
                day_start = current_sim_time + i * SECONDS_PER_PLANNING_HORIZON
                day_end = day_start + SECONDS_PER_PLANNING_HORIZON
                workday_boundaries.append((i, day_start, day_end))

        bottlenecks: List[BottleneckInfo] = []
        
        # --- Pre-compute calendar availability per (resource, day) ---
        # This avoids redundant hour-by-hour walks when checking the same
        # resource across multiple activities (~8× speedup for 8 activities).
        #   avail_cache[(rid, day_idx)] -> avail_seconds (float)
        avail_cache: Dict[tuple, float] = {}
        for r in resources:
            calendar = resource_calendars.get(r.id)
            for workday_idx, day_start, day_end in workday_boundaries:
                if calendar is None:
                    avail_cache[(r.id, workday_idx)] = float('inf')  # always available
                else:
                    avail_cache[(r.id, workday_idx)] = calendar.get_available_time_in_seconds(
                        start_sim=day_start, end_sim=day_end, time_converter=time_converter,
                        neglect_sick_leave=(workday_idx != 0),
                    )

        activity_set: Set[str] = set(self.activity_requirements.keys())
        if active_activities is not None:
            activity_set &= set(active_activities)

        cap_check: Dict[tuple, bool] = {}

        for activity in activity_set:
            required_level = self.activity_requirements.get(activity, 50.0)

            # --- Determine capable resources (memoized can_perform) ---
            capable_resources: List[Resource] = []
            for r in resources:
                key = (r.id, activity)
                cached = cap_check.get(key)
                if cached is None:
                    cached = r.can_perform(activity, required_level=required_level)
                    cap_check[key] = cached
                if cached:
                    capable_resources.append(r)
            if len(capable_resources) == 0:
                # No capable resource at all -> permanent severe bottleneck
                uncapable_sorted = sorted(
                    resources,
                    key=lambda r: r.get_experience_level(activity),
                    reverse=True,
                )
                bottlenecks.append(BottleneckInfo(
                    activity_name=activity,
                    severity='severe',
                    days_until_bottleneck=0,
                    capable_resource_count=0,
                    fit_resource_ids=[],
                    mentee_candidates=[r.id for r in uncapable_sorted],
                    mentor_candidates=[],
                )) # in case of tasks of that activity, this will trigger bootstrapping
                continue

            # --- Check each of the next 3 workdays ---
            severity: str | None = None
            days_until: int | None = None

            for workday_idx, day_start, day_end in workday_boundaries:
                available_capable_resources = 0
                for r in capable_resources:
                    if avail_cache[(r.id, workday_idx)] > 0:
                        available_capable_resources += 1

                unavailable_ratio = 1.0 - (available_capable_resources / len(capable_resources))
                availability_ratio = self.mentoring_config.get('bottleneck_activity_strategy', {}).get('resource_availability_ratio', 0.3)
                if available_capable_resources == 0:
                    # Severe: no capable resource available on this workday
                    severity = 'severe'
                    days_until = workday_idx
                    # print(f"Activity '{activity}' has a SEVERE bottleneck in {days_until} day(s) (unavailable ratio: {unavailable_ratio:.2f})")
                    break  # worst case found, no need to check further
                elif unavailable_ratio >= availability_ratio and severity is None: # Medium: >50 % of capable resources absent
                    severity = 'medium'
                    days_until = workday_idx
                    # print(f"Activity '{activity}' has a MEDIUM bottleneck in {days_until} day(s) (unavailable ratio: {unavailable_ratio:.2f})")

            if severity is None:
                continue  # No bottleneck for this activity

            bottlenecks.append(BottleneckInfo(
                activity_name=activity,
                severity=severity,
                days_until_bottleneck=days_until,
                capable_resource_count=len(capable_resources),
                fit_resource_ids=[r.id for r in capable_resources],
            ))
            
        return bottlenecks