"""Base scheduler interface, scheduling context, and shared batch-data helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..entities import Resource, ResourceCalendar, Task, Case
from ..experience import ExperienceStore
from ..utils.time_utils import hours_to_seconds


logger = logging.getLogger(__name__)

DUMMY_RESOURCE_ID = "User_00"  # Must match the dummy resource in the experience store.
DUMMY_DURATION = 1  # Near-zero duration to keep deferred assignments feasible.
DUMMY_CAPACITY = 100_000_000  # Effectively unlimited capacity for the dummy resource.

DEFAULT_OBJECTIVE_WEIGHTS = {
    "pressure": 1.0,
    "deferral_priority": 1.0,
    "bottleneck": 1.0,
    "utilization": 1.0,
    "underutilization": 1.0,
    "shortage": 1.0,
}


@dataclass
class SchedulingContext:
    """
    Context information provided to schedulers for decision making.

    Attributes:
        all_resources: All resources in the system (including busy ones)
        active_cases: All active cases
        pending_tasks: List of tasks waiting to be assigned
        experience_store: Access to experience profiles
        current_time: Current simulation time
        queue_lengths: Dict mapping resource_id to current task queue length of the resource
        resource_current_task_load: Dict mapping resource_id to list of tasks already in queue
        simulation_state: Optional reference to full simulation state
        resource_calendars: Optional dict of resource_id to calendar (if working hours enabled)
    """
    all_resources: List[Resource]
    active_cases: List[Case]
    pending_tasks: List[Task]
    experience_store: ExperienceStore
    current_time: float
    queue_lengths: Dict[str, int] = field(default_factory=dict)
    resource_current_task_load: Dict[str, List[Task]] = field(default_factory=dict)
    simulation_state: Optional[Any] = None
    resource_calendars: Optional[Dict[str, Any]] = None

    @property
    def all_cases(self) -> List[Case]:
        """Backward-compatible alias for code paths still using `all_cases`."""
        return self.active_cases


@dataclass
class KnapsackData:
    """Data model for the Multiple Knapsack Problem (MKP) formulation."""

    task_ids: List[str]
    resource_ids: List[str]
    capacities: Dict[str, float]
    durations: Dict[Tuple[str, str], Optional[float]]
    task_objects: Dict[str, Task]
    resource_objects: Dict[str, Resource]
    min_real_durations: Dict[str, float] = field(default_factory=dict)
    mean_real_durations: Dict[str, float] = field(default_factory=dict)
    bottlenecks: List[Any] = field(default_factory=list)
    capable_resources: Dict[str, List[str]] = field(default_factory=dict)


class Scheduler(ABC):
    """
    Abstract base class for task-resource scheduling algorithms.

    Provides shared infrastructure for all concrete schedulers:
    - Capability filtering with caching
    - Knapsack data construction (_build_knapsack_data)
    - Bootstrap capability grant logic
    - Mentoring stubs
    - Duration and experience profile caching
    """

    def __init__(
        self,
        name: Optional[str] = None,
        duration_predictor: Optional[Any] = None,
        sample_cache_size: int = 50,
        max_cache_entries: int = 10000,
        config: Optional[Dict[str, Any]] = None,
        time_converter: Optional[Any] = None,
    ):
        self.name = name or self.__class__.__name__
        self.duration_predictor = duration_predictor
        self.config = config or {}
        self.time_converter = time_converter if time_converter else None
        self._last_solver_meta: Dict[str, Any] = {}

        optimization_cfg = self.config.get("optimization", {})
        self.objective_weights = optimization_cfg.get(
            "objective_weights", DEFAULT_OBJECTIVE_WEIGHTS
        )

        # Capability filtering cache
        self._capability_cache: Dict[str, List[str]] = {}
        self._resources_by_id: Dict[str, Resource] = {}

        # Duration sampling cache
        self._duration_cache: Dict[str, List[float]] = {}
        self._sample_cache_size = sample_cache_size
        self._max_cache_entries = max_cache_entries
        self._cache_hits = 0
        self._cache_misses = 0

        # Experience profile cache
        self._profile_cache: "OrderedDict[tuple, Any]" = OrderedDict()
        self._profile_cache_hits = 0
        self._profile_cache_misses = 0

        self.activity_requirements = self._load_activity_requirements()
        self.max_task_deferrals = optimization_cfg.get("max_task_deferrals", 5)

        bootstrap_cfg = self._bootstrap_config()
        self.bootstrap_enabled = bootstrap_cfg["enabled"]
        self.bootstrap_max_per_activity = max(
            0,
            int(bootstrap_cfg.get("max_bootstrap_resources_per_activity_per_run", 1)),
        )
        self._bootstrap_grants_by_activity: Dict[str, set] = {}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def plan_tasks_to_resources(
        self,
        context: SchedulingContext,
        planning_horizon_hours: float = 24.0,
        enforce_working_hours: bool = True,
        duration_predictor: Optional[Any] = None,
        max_solver_time_seconds: float = 60.0,
        optimality_gap: float = 0.05,
    ) -> Dict[str, List[Task]]:
        """
        Plan assignments of pending tasks to resources over a planning horizon.

        Args:
            context: Current scheduling context
            planning_horizon_hours: How many hours into the future to plan
            enforce_working_hours: Whether to respect working hours in scheduling
            duration_predictor: Optional ML-based duration predictor for better estimates
            max_solver_time_seconds: Time limit for optimization solver (if used)
            optimality_gap: Acceptable optimality gap for solver solutions (if used)
        Returns:
            Dict mapping resource IDs to lists of assigned tasks for the planning horizon
        """
        pass

    # ------------------------------------------------------------------
    # Capability filtering
    # ------------------------------------------------------------------

    def filter_capable_resources(
        self,
        task: Task,
        resources: List[Resource],
    ) -> List[Resource]:
        """Filter resources to only those capable of performing the task."""
        if not task.required_capability_level:
            return resources

        if not self._resources_by_id and resources:
            self._resources_by_id = {r.id: r for r in resources}

        req_capability = task.activity_name
        req_level = task.required_capability_level
        cache_key = f"{req_capability}:{req_level}"
        if cache_key in self._capability_cache:
            cached_ids = self._capability_cache[cache_key]
            return [self._resources_by_id[rid] for rid in cached_ids if rid in self._resources_by_id]

        capable_ids = [r.id for r in resources if r.can_perform(req_capability, req_level)]
        self._capability_cache[cache_key] = capable_ids
        return [self._resources_by_id[rid] for rid in capable_ids]

    # ------------------------------------------------------------------
    # Task priority
    # ------------------------------------------------------------------

    def get_task_priority(self, task: Task, context: SchedulingContext) -> float:
        """Calculate priority score for a task (higher = more urgent)."""
        case = next((c for c in context.active_cases if c.id == task.case_id), None)
        if case is None:
            return 0.0

        waiting_time = context.current_time - task.creation_time
        priority = case.priority * 10 + waiting_time

        if case.deadline is not None:
            time_until_deadline = case.deadline - context.current_time
            if time_until_deadline < 0:
                priority += 1000
            elif time_until_deadline < 86400:
                priority += 100

        return priority

    # ------------------------------------------------------------------
    # Bootstrap capability helpers
    # ------------------------------------------------------------------

    def _bootstrap_config(self) -> Dict[str, Any]:
        """Get normalized bootstrap capability configuration."""
        raw = getattr(self, 'config', {}) or {}
        cfg = raw.get('bootstrap_capability', {}) or {}
        return {
            'enabled': bool(cfg.get('enabled', False)),
            'selection_heuristic': str(cfg.get('selection_heuristic', 'lowest_workload')),
            'default_onboarding_seconds': int(cfg.get('default_onboarding_seconds', 1800)),
            'activity_onboarding_seconds': dict(cfg.get('activity_onboarding_seconds', {}) or {}),
            'max_bootstrap_resources_per_activity_per_run': int(
                cfg.get('max_bootstrap_resources_per_activity_per_run', 1)
            ),
        }

    def _get_bootstrap_onboarding_seconds(self, activity_name: str) -> int:
        """Resolve additive onboarding penalty for a bootstrap assignment."""
        cfg = self._bootstrap_config()
        overrides = cfg.get('activity_onboarding_seconds', {})
        if activity_name in overrides:
            return int(overrides[activity_name])
        return int(cfg.get('default_onboarding_seconds', 1800))

    def _select_bootstrap_resource(
        self,
        candidates: List[Resource],
        queue_lengths: Optional[Dict[str, int]] = None,
        capacities: Optional[Dict[str, int]] = None,
    ) -> Optional[Resource]:
        """Pick bootstrap resource by lowest workload, then highest available capacity."""
        if not candidates:
            return None

        ql = queue_lengths or {}
        caps = capacities or {}

        def _rank(resource: Resource) -> Tuple[int, int, str]:
            queue_len = int(ql.get(resource.id, 0))
            cap = int(caps.get(resource.id, 0))
            return (queue_len, -cap, resource.id)

        return min(candidates, key=_rank)

    # ------------------------------------------------------------------
    # Mentoring stubs (overridden by ExperienceBasedScheduler)
    # ------------------------------------------------------------------

    def find_mentor_for_task(
        self,
        task: Task,
        context: SchedulingContext,
    ) -> Optional[str]:
        """Find a suitable mentor for a mentoring task. Default: no mentoring support."""
        return None

    def assign_mentor_to_task(
        self,
        task: Task,
        context: SchedulingContext,
    ) -> Optional[str]:
        """Assign a mentor to a mentoring task."""
        if not task.is_mentoring_task():
            return None
        if not task.mentee_resource_id:
            return None

        mentor_id = self.find_mentor_for_task(task, context)

        if mentor_id and mentor_id == task.mentee_resource_id:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                f"Mentor {mentor_id} is the same as mentee for task {task.id} – rejecting match"
            )
            task.mentor_resource_id = None
            return None

        if mentor_id:
            task.mentor_resource_id = mentor_id

        return mentor_id

    # ------------------------------------------------------------------
    # Activity requirements
    # ------------------------------------------------------------------

    def _load_activity_requirements(self) -> Dict[str, float]:
        """Load activity requirements from config/activity_requirements.yaml."""
        config_path = Path(__file__).parent.parent.parent / "config" / "activity_requirements.yaml"
        if not config_path.exists():
            return {}
        try:
            with open(config_path, "r") as f:
                req_config = yaml.safe_load(f)
            return req_config.get("activity_requirements", {})
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Knapsack data construction
    # ------------------------------------------------------------------

    def _build_knapsack_data(
        self,
        pending_tasks: List[Task],
        resources: List[Resource],
        resource_calendars: Dict[str, ResourceCalendar],
        experience_store: ExperienceStore,
        queue_lengths: Optional[Dict[str, int]],
        duration_predictor: Optional[Any],
        current_sim_time: float,
        planning_horizon_hours: float,
        enforce_working_hours: bool,
        bottlenecks: Optional[List[Any]] = None,
    ) -> KnapsackData:
        """Build the data model for the Multiple Knapsack Problem."""
        task_ids = [t.id for t in pending_tasks]
        resource_ids = [r.id for r in resources]
        task_objects = {t.id: t for t in pending_tasks}
        resource_objects = {r.id: r for r in resources}
        capable_resources: Dict[str, set] = {str(t.activity_name): set() for t in pending_tasks}

        capacities: Dict[str, int] = {}
        planning_horizon_seconds = hours_to_seconds(planning_horizon_hours)
        for resource in resources:
            if enforce_working_hours and resource.id in resource_calendars:
                calendar = resource_calendars[resource.id]
                available_seconds = calendar.get_available_time_in_seconds(
                    start_sim=current_sim_time,
                    end_sim=current_sim_time + planning_horizon_seconds,
                    time_converter=self.time_converter,
                )
                if available_seconds <= 0:
                    capacities[resource.id] = 0
                    continue
                if available_seconds > (10.0 * 3600.0):
                    logger.warning(
                        "Resource %s has unusually high available time (%.1fh) in the planning horizon - check calendar configuration",
                        resource.id,
                        available_seconds / 3600.0,
                    )
                available = available_seconds
            else:
                available = planning_horizon_seconds

            capacities[resource.id] = max(available, 0)

        durations: Dict[Tuple[str, str], Optional[int]] = {}
        safety_multiplier = self.config.get("duration_prediction", {}).get(
            "safety_margin_std_multiplier", 0.5
        )

        def _estimate_duration(task: Task, resource: Resource) -> int:
            if duration_predictor:
                profile = self._get_cached_profile(
                    experience_store, resource.id, task.activity_name, task.context
                )
                if hasattr(duration_predictor, "predict_with_safety"):
                    predicted = duration_predictor.predict_with_safety(
                        resource.id,
                        task.activity_name,
                        task.context,
                        experience_profile=profile,
                        safety_multiplier=safety_multiplier,
                    )
                else:
                    predicted = duration_predictor.predict(
                        resource.id,
                        task.activity_name,
                        task.context,
                        experience_profile=profile,
                    )
                return int(round(predicted))

            estimation_method = self.config.get("duration_prediction", {}).get(
                "duration_estimation_method",
                "mean_plus_safety_margin",
            )
            return int(
                experience_store.get_duration(
                    resource.id,
                    task.activity_name,
                    type=estimation_method,
                    context=task.context,
                )
            )

        def _has_global_capability(task: Task, required_level: float) -> bool:
            """Check whether capability exists globally, not only among today's available capacity."""
            activity_name = task.activity_name

            for resource in resources:
                if float(resource.get_experience_level(activity_name)) >= required_level:
                    return True

            if hasattr(experience_store, "get_all_profiles_for_activity"):
                profiles = experience_store.get_all_profiles_for_activity(activity_name)
                for profile in profiles:
                    if float(profile.experience_level) >= required_level and int(profile.count) > 0:
                        return True

            if hasattr(experience_store, "get_capability_level"):
                for resource in resources:
                    lvl = experience_store.get_capability_level(
                        resource_id=resource.id,
                        activity_name=activity_name,
                        context=task.context,
                    )
                    if float(lvl) >= required_level:
                        return True

            return False

        for tid, task in task_objects.items():
            required_level = float(task.required_capability_level)
            feasible_real_resources: List[str] = []
            capability_blocked_candidates: List[Resource] = []
            for resource in resources:
                if capacities.get(resource.id, 0) <= 0:
                    continue
                duration = _estimate_duration(task=task, resource=resource)
                durations[(tid, resource.id)] = duration

                cap_level = resource.get_experience_level(task.activity_name)
                if cap_level < required_level:
                    if capacities.get(resource.id, 0) > 0:
                        capability_blocked_candidates.append(resource)
                    continue

                capable_resources[task.activity_name].add(str(resource.id))
                feasible_real_resources.append(resource.id)

            if (
                self.bootstrap_enabled
                and not feasible_real_resources
                and capability_blocked_candidates
                and not _has_global_capability(task, required_level)
            ):
                granted_resources = self._bootstrap_grants_by_activity.get(task.activity_name, set())
                if len(granted_resources) < self.bootstrap_max_per_activity:
                    bootstrap_resource = self._select_bootstrap_resource(
                        candidates=capability_blocked_candidates,
                        queue_lengths=queue_lengths,
                        capacities=capacities,
                    )
                    if bootstrap_resource is not None:
                        bootstrap_duration = _estimate_duration(
                            task=task, resource=bootstrap_resource
                        )
                        durations[(tid, bootstrap_resource.id)] = int(bootstrap_duration)
                        capable_resources[task.activity_name].add(str(bootstrap_resource.id))

                        experience_store.grant_capability(
                            resource_id=bootstrap_resource.id,
                            activity_name=task.activity_name,
                            required_level=required_level,
                            context=task.context,
                            simulation_time=current_sim_time,
                        )
                        granted_resources.add(bootstrap_resource.id)
                        self._bootstrap_grants_by_activity[task.activity_name] = granted_resources

                        task.estimated_duration = int(bootstrap_duration)
                        task.bootstrap_assignment = True
                        task.bootstrap_activity = task.activity_name
                        task.required_capability_level = required_level
                        task.bootstrap_resource_id = bootstrap_resource.id
                        task.bootstrap_onboarding_seconds = self._get_bootstrap_onboarding_seconds(
                            task.activity_name
                        )
                        task.bootstrap_onboarding_applied = False
                        logger.info(
                            "Bootstrap capability grant: activity=%s resource=%s required=%.2f onboarding=%ss",
                            task.activity_name,
                            bootstrap_resource.id,
                            required_level,
                            task.bootstrap_onboarding_seconds,
                        )

        min_real_durations: Dict[str, int] = {}
        mean_real_durations: Dict[str, int] = {}
        for tid, task in task_objects.items():
            capable_rids = capable_resources.get(str(task.activity_name), set())
            real_durs = [
                durations[(tid, r.id)]
                for r in resources
                if r.id in capable_rids and durations.get((tid, r.id)) is not None
            ]
            min_real_durations[tid] = min(real_durs) if real_durs else 3600
            mean_real_durations[tid] = int(sum(real_durs) / len(real_durs)) if real_durs else 3600

        resource_ids.append(DUMMY_RESOURCE_ID)
        capacities[DUMMY_RESOURCE_ID] = DUMMY_CAPACITY
        for tid, task in task_objects.items():
            durations[(tid, DUMMY_RESOURCE_ID)] = DUMMY_DURATION
        capable_resources_lists: Dict[str, List[str]] = {
            k: sorted(v) for k, v in capable_resources.items()
        }
        return KnapsackData(
            task_ids=task_ids,
            resource_ids=resource_ids,
            capacities=capacities,
            durations=durations,
            task_objects=task_objects,
            resource_objects=resource_objects,
            min_real_durations=min_real_durations,
            mean_real_durations=mean_real_durations,
            bottlenecks=bottlenecks if bottlenecks is not None else [],
            capable_resources=capable_resources_lists,
        )

    # ------------------------------------------------------------------
    # Experience profile cache
    # ------------------------------------------------------------------

    def _get_cached_profile(
        self,
        experience_store: ExperienceStore,
        resource_id: str,
        activity_name: str,
        context: dict,
    ):
        """Get experience profile from cache or experience store."""
        context_key = experience_store._make_context_key(context) if context else ""
        cache_key = (resource_id, activity_name, context_key)

        if cache_key in self._profile_cache:
            self._profile_cache_hits += 1
            self._profile_cache.move_to_end(cache_key)
            return self._profile_cache[cache_key]

        self._profile_cache_misses += 1
        profile = experience_store.get_profile(resource_id, activity_name, context)

        self._profile_cache[cache_key] = profile
        if len(self._profile_cache) > self._max_cache_entries:
            self._profile_cache.popitem(last=False)

        return profile

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
