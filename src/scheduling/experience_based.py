"""
Experience-based scheduler — orchestrates CP-SAT-based daily task assignment.

This class is the public-facing scheduler.  All CP-SAT model logic (variables,
constraints, objective, solution extraction) and the associated pre-computation
live in :mod:`mkp_formulator`.  This class is responsible for:

* Loading and storing configuration.
* Detecting capability bottlenecks before each scheduling cycle.
* Building the knapsack input data (via the inherited ``Scheduler``).
* Creating and invoking :class:`MKPFormulator`.
* Handling the fallback retry when the solver fails.
* Logging the solver run.
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional, Tuple

from .base import Scheduler, SchedulingContext, KnapsackData, DUMMY_RESOURCE_ID
from .fitness_analyzer import TaskFitnessAnalyzer, BottleneckInfo
from .mkp_formulator import MKPFormulator
from ..entities import Task, Resource, ResourceCalendar
from ..experience.learning_curves import create_learning_curve, LearningCurveParameters
from ..experience.store import ExperienceStore
from ..utils.time_utils import SimulationTimeConverter

logger = logging.getLogger(__name__)


class ExperienceBasedScheduler(Scheduler):
    """
    Experience-based scheduler.

    Assigns tasks using CP-SAT optimization that incorporates resource experience,
    capability bottleneck forecasts, and mentoring incentives.  The formulation is
    delegated to :class:`MKPFormulator`; this class focuses on orchestration.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        duration_predictor: Optional[Any] = None,
        sample_cache_size: int = 50,
        max_cache_entries: int = 10000,
        config: Optional[Dict[str, Any]] = None,
        time_converter: Optional[SimulationTimeConverter] = None,
    ) -> None:
        super().__init__(
            name=name or "ExperienceBased",
            duration_predictor=duration_predictor,
            sample_cache_size=sample_cache_size,
            max_cache_entries=max_cache_entries,
            config=config,
            time_converter=time_converter,
        )

        self.mentoring_config = self.config.get("mentoring", {}) if self.config else {}
        self.mentoring_enabled = bool(self.mentoring_config.get("enabled", False))

        # Bottleneck-activity strategy toggle (with legacy config key fallback)
        _bn_strategy_cfg = self.mentoring_config.get("bottleneck_activity_strategy", {})
        _legacy_bn_enabled = bool(
            (self.config.get("bottleneck_detection", {}) or {}).get("enabled", True)
        )
        self.bottleneck_activity_strategy_enabled = bool(
            _bn_strategy_cfg.get("enabled", _legacy_bn_enabled)
        )

        # Lazy-initialized Richards learning curve for mentoring constraints
        self._learning_curve = None

        # Metrics snapshot from the last scheduling cycle (set by _solve_with_fallback)
        self._last_formulator_metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Learning curve (lazy property)
    # ------------------------------------------------------------------

    @property
    def learning_curve(self):
        """Lazy-initialized Richards curve for inverse repetition look-ups."""
        if self._learning_curve is None:
            exp_cfg = self.config.get("experience", {}) if self.config else {}
            bp = exp_cfg.get("breeding_params", {})
            params = LearningCurveParameters(
                A_i=bp.get("lower_asymptote", 0.0),
                K_i=bp.get("upper_asymptote", 99.0),
                v_i=bp.get("growth_rate", 0.1),
                Q_i=bp.get("shape_param_Q", 2.5),
                M_curve=bp.get("shape_param_M", 0.8),
            )
            self._learning_curve = create_learning_curve(
                exp_cfg.get("learning_model", "richards"), params
            )
        return self._learning_curve

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def plan_tasks_to_resources(
        self,
        context: SchedulingContext,
        planning_horizon_hours: float = 24.0,
        enforce_working_hours: bool = True,
        duration_predictor: Optional[Any] = None,
        max_solver_time_seconds: float = 60.0,
        optimality_gap: float = 0.05,
    ) -> Dict[str, List[Task]]:
        """Plan daily task-to-resource assignments as a Multiple Knapsack Problem.

        Steps:
        1. Detect capability bottlenecks for forward-looking mentoring.
        2. Build the knapsack input data.
        3. Invoke MKPFormulator; fall back without mentoring if the solver fails.
        4. Log the solver run.

        Returns
        -------
        Dict mapping resource_id → list of assigned Task objects.
        """
        active_activities = {t.activity_name for t in context.pending_tasks}

        # ---- Step 1: Bottleneck detection ----
        t0 = time.monotonic()
        bottlenecks: List[BottleneckInfo] = []
        if self.mentoring_enabled and self.bottleneck_activity_strategy_enabled:
            bottlenecks = self._detect_bottlenecks(
                resources=context.all_resources,
                experience_store=context.experience_store,
                resource_calendars=context.resource_calendars,
                current_sim_time=context.current_time,
                planning_horizon_hours=planning_horizon_hours,
                active_activities=active_activities,
            )
        t_bottleneck = time.monotonic() - t0

        # ---- Step 2: Build knapsack data ----
        t0 = time.monotonic()
        data = self._build_knapsack_data(
            pending_tasks=context.pending_tasks,
            resources=context.all_resources,
            resource_calendars=context.resource_calendars,
            experience_store=context.experience_store,
            queue_lengths=context.queue_lengths,
            duration_predictor=duration_predictor,
            current_sim_time=context.current_time,
            planning_horizon_hours=planning_horizon_hours,
            enforce_working_hours=enforce_working_hours,
            bottlenecks=bottlenecks,
        )
        t_build = time.monotonic() - t0

        # ---- Step 3: Solve (with fallback) ----
        assignments, solve_wall_time = self._solve_with_fallback(
            data=data,
            context=context,
            max_solver_time_seconds=max_solver_time_seconds,
            optimality_gap=optimality_gap,
        )

        # ---- Step 4: Log solver run ----
        t0 = time.monotonic()
        t_log = time.monotonic() - t0

        logger.info(
            "scheduler_cycle sim_time=%.1f n_tasks=%d n_resources=%d n_bottlenecks=%d "
            "t_bottleneck=%.3f t_build=%.3f t_solve=%.3f t_log=%.3f "
            "n_x_vars=%s n_mentor_triples=%s build_time=%s",
            context.current_time,
            len(context.pending_tasks),
            len(context.all_resources),
            len(bottlenecks),
            t_bottleneck,
            t_build,
            solve_wall_time,
            t_log,
            self._last_solver_meta.get("n_x_vars", "-"),
            self._last_solver_meta.get("n_mentor_triples", "-"),
            self._last_solver_meta.get("build_time_seconds", "-"),
        )

        return assignments

    # ------------------------------------------------------------------
    # Solve with fallback
    # ------------------------------------------------------------------

    def _solve_with_fallback(
        self,
        data: KnapsackData,
        context: SchedulingContext,
        max_solver_time_seconds: float,
        optimality_gap: float,
    ) -> Tuple[Dict[str, List[Task]], float]:
        """Attempt CP-SAT solve; retry without mentoring if the first attempt fails.

        Returns
        -------
        (assignments, total_wall_time)
        """
        def _make_formulator(mentoring_enabled: bool) -> MKPFormulator:
            return MKPFormulator(
                data=data,
                mentoring_config=self.mentoring_config,
                objective_weights=self.objective_weights,
                activity_requirements=self.activity_requirements,
                max_task_deferrals=self.max_task_deferrals,
                experience_store=context.experience_store,
                learning_curve=self.learning_curve if mentoring_enabled else None,
                profile_cache_fn=self._get_cached_profile,
                mentoring_enabled=mentoring_enabled,
            )

        assignments: Dict[str, List[Task]] = {}
        first_failed = False
        t0 = time.monotonic()
        active_formulator: Optional[MKPFormulator] = None

        try:
            formulator = _make_formulator(self.mentoring_enabled)
            assignments = formulator.solve(max_solver_time_seconds, optimality_gap)
            self._last_solver_meta = formulator.solver_meta
            active_formulator = formulator
        except Exception as exc:
            logger.warning("MKP solver failed: %s", exc)
            first_failed = True

        wall_time = time.monotonic() - t0
        status = self._last_solver_meta.get("status", "UNKNOWN")
        if status not in ("OPTIMAL", "FEASIBLE"):
            logger.warning(
                "Solver finished in %.2fs with status %s (objective=%s)",
                wall_time, status, self._last_solver_meta.get("objective_value"),
            )

        if first_failed and self.mentoring_enabled:
            logger.warning("Retrying without mentoring constraints.")
            t_fallback = time.monotonic()
            try:
                fallback = _make_formulator(mentoring_enabled=False)
                assignments = fallback.solve(max_solver_time_seconds, optimality_gap)
                self._last_solver_meta = fallback.solver_meta
                active_formulator = fallback
                logger.info(
                    "Fallback solver (no mentoring) finished with status %s",
                    self._last_solver_meta.get("status", "UNKNOWN"),
                )
            except Exception as exc:
                logger.error("Fallback solver also failed: %s", exc)
            wall_time += time.monotonic() - t_fallback

        # Expose metrics for the daily summary aggregator
        self._last_formulator_metrics = (
            active_formulator.get_metrics_snapshot() if active_formulator is not None else {}
        )

        return assignments, wall_time

    # ------------------------------------------------------------------
    # Bottleneck detection
    # ------------------------------------------------------------------

    def _detect_bottlenecks(
        self,
        resources: List[Resource],
        experience_store: ExperienceStore,
        resource_calendars: Dict[str, ResourceCalendar],
        current_sim_time: float,
        planning_horizon_hours: float,
        active_activities: Optional[set] = None,
    ) -> List[BottleneckInfo]:
        """Detect capability bottlenecks via TaskFitnessAnalyzer."""
        analyzer = TaskFitnessAnalyzer(
            mentoring_config=self.mentoring_config,
            experience_store=experience_store,
            activity_requirements=self.activity_requirements,
        )
        return analyzer.detect_bottlenecks(
            resources=resources,
            resource_calendars=resource_calendars,
            current_sim_time=current_sim_time,
            planning_horizon_hours=planning_horizon_hours,
            time_converter=self.time_converter,
            active_activities=active_activities,
        )

