"""
CP-SAT formulator for the experience-based Multiple Knapsack Problem.

This module is responsible for building and solving the daily task-to-resource
assignment problem.  It is separated from the scheduler orchestration layer so
that the CP-SAT model logic can be read, tested, and modified in one place.

All pre-computation (demand, pressure, BRR, bottleneck context, underutilization
context) and all model-building phases (variables, constraints, objective,
solution extraction) live here.  Intermediate results are stored as instance
attributes so they can be inspected for debugging.

Typical usage::

    formulator = MKPFormulator(data, mentoring_config, ...)
    assignments = formulator.solve(max_solver_time, optimality_gap)
    meta = formulator.solver_meta   # {'status', 'objective_value', ...}
"""

from __future__ import annotations

import math
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from ortools.sat.python import cp_model
from ortools.sat.python import cp_model_helper as cmh

IntVar = cmh.IntVar

from .base import KnapsackData, DUMMY_RESOURCE_ID, DEFAULT_OBJECTIVE_WEIGHTS
from .fitness_analyzer import BottleneckInfo
from ..entities import Task
from ..entities.task import TaskType
from ..experience.store import ExperienceStore
from ..utils.time_utils import seconds_to_hours

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (shared with experience_based.py via import)
# ---------------------------------------------------------------------------

DUMMY_PENALTY_BASE = 1_000   # base penalty per dummy assignment × task duration
OBJECTIVE_SCALE    = 100     # scales float objective terms to integers for CP-SAT
BOTTLENECK_BONUS   = 10      # large bonus for mentoring severe bottleneck activities
PRESSURE_FLOOR     = 0.15    # minimum pressure so x_normal retains some value
MAX_MENTORS_PER_PAIR = 3   # caps mentor candidates to bound variable count


class MKPFormulator:
    """Builds and solves the CP-SAT Multiple Knapsack Problem for one scheduling day.

    Instantiate once per scheduling cycle, call :meth:`solve`, then discard.

    Parameters
    ----------
    data:
        KnapsackData produced by ``BatchDataScheduler._build_knapsack_data``.
    mentoring_config:
        The ``mentoring`` sub-dict from the simulation YAML config.
    objective_weights:
        Weight dict for objective components (comes from config or defaults).
    activity_requirements:
        Mapping of activity name → required experience level.
    max_task_deferrals:
        Maximum deferrals before a task is dropped; used for deferral priority.
    experience_store:
        Required for learning-curve mentoring constraints (profile look-ups).
    learning_curve:
        Lazy-initialized Richards curve; may be ``None`` when mentoring is off.
    profile_cache_fn:
        Callable ``(experience_store, rid, activity, context) → profile`` that
        returns a cached experience profile from the parent scheduler.
    mentoring_enabled:
        Override flag — set to ``False`` for the fallback (no-mentoring) solve.
    """

    def __init__(
        self,
        data: KnapsackData,
        mentoring_config: dict,
        objective_weights: dict,
        activity_requirements: dict,
        max_task_deferrals: int,
        experience_store: ExperienceStore,
        learning_curve: Any,
        profile_cache_fn: Callable,
        mentoring_enabled: bool = True,
    ) -> None:
        self.data = data
        self.mentoring_config = mentoring_config
        self.objective_weights = objective_weights
        self.activity_requirements = activity_requirements
        self.max_task_deferrals = max_task_deferrals
        self.experience_store = experience_store
        self.learning_curve = learning_curve
        self._profile_cache_fn = profile_cache_fn

        # Resolve mentoring toggle (config + caller override)
        self.mentoring_enabled = mentoring_enabled and bool(mentoring_config.get("enabled", False))

        # Underutilization strategy sub-config
        _uutil = mentoring_config.get("underutilization_strategy", {})
        self.underutilization_enabled            = bool(_uutil.get("enabled", False))
        self.underutil_mentor_pressure_threshold = float(_uutil.get("mentor_pressure_threshold", 0.55))
        self.underutil_target_pressure_threshold = float(_uutil.get("target_pressure_threshold", 0.45))
        self.underutil_min_spare_hours           = float(_uutil.get("min_spare_hours", 2.0))
        self.underutil_max_current_activities    = int(_uutil.get("max_current_activities", 2))
        self.underutil_bonus_scale               = float(_uutil.get("bonus_scale", 0.15))

        # Same-day shortage strategy sub-config (with legacy key fallback)
        _shortage = mentoring_config.get(
            "same_day_shortage_strategy",
            mentoring_config.get("rolling_shortage_strategy", {}),
        )
        self.same_day_shortage_enabled        = bool(_shortage.get("enabled", False))
        self.same_day_shortage_debug          = bool(_shortage.get("debug", False))
        self.same_day_shortage_strong_ratio   = float(_shortage.get("strong_shortage_ratio", 2.0))
        self.same_day_shortage_quota          = max(1, int(_shortage.get("strategic_quota_per_activity", 1)))
        self.same_day_shortage_bonus_mult     = max(0.0, float(_shortage.get("objective_bonus_multiplier", 0.25)))

        # Bottleneck activity strategy sub-config
        _bn_strategy = mentoring_config.get("bottleneck_activity_strategy", {})
        self.bottleneck_activity_strategy_enabled = bool(_bn_strategy.get("enabled", True))
        self.severe_bottleneck_mode = mentoring_config.get("severe_bottleneck_mode", "objective_bonus")

        # Mentoring duration parameters
        self.duration_multiplier = float(mentoring_config.get("duration_multiplier", 1.5))
        self.duration_additive   = int(mentoring_config.get("duration_summand_seconds", 300))

        # Solver metadata populated after solve()
        self.solver_meta: Dict[str, Any] = {}

        # Pre-computation results — populated by _precompute()
        self._real_resource_ids: List[str] = [
            rid for rid in self.data.resource_ids if rid != DUMMY_RESOURCE_ID
        ]
        self._activity_roles: Dict[str, Tuple[set, List[Tuple[str, float]]]] = {}
        self._task_feasible_real_resources: Dict[str, List[str]] = {}
        self._activity_task_ids: Dict[str, List[str]] = {}
        self._activity_total_work_hours: Dict[str, float] = {}
        self._resource_capability_map: Dict[str, Dict[str, List[str]]] = {}
        self._activity_supply_hours: Dict[str, float] = {}
        self._direct_demand: Dict[str, float] = {}
        self._indirect_demand: Dict[str, float] = {}
        self._strategic_shortage_activities: set = set()
        self._resource_pressure: Dict[str, float] = {}
        self._resource_spare_hours: Dict[str, float] = {}
        self._bottleneck_lookup: Dict[str, BottleneckInfo] = {}
        self._severe_bottleneck_activities: set = set()
        self._detected_severe_bottleneck_activities: set = set()
        self._strategic_bottleneck_activities: set = set()

        # Underutilization context — populated by _precompute(), reused in objective builder
        self._activity_demand_score: Dict[str, float] = {}
        self._mentor_substitutability_norm: Dict[Tuple[str, str], float] = {}
        self._underutilized_targets: Dict[str, float] = {}
        self._growth_activity_scores: Dict[str, Dict[str, float]] = {}

        # CP-SAT model state — populated during model building
        self._x_normal: Dict[Tuple[str, str], IntVar] = {}
        self._x_mentor: Dict[Tuple[str, str, str], IntVar] = {}
        self._y: Dict[Tuple[str, str], IntVar] = {}
        self._mentor_vars_by_task: Dict[str, List[Tuple[str, str, IntVar]]] = {}
        self._mentor_vars_by_mentee: Dict[str, List[Tuple[str, str, IntVar]]] = {}
        self._mentor_vars_by_mentor: Dict[str, List[Tuple[str, str, IntVar]]] = {}
        self._mentor_dur: Dict[Tuple[str, str, str], int] = {}  # keyed by (tid, rid_mentee, mid_mentor)
        self._mentor_shadow_dur: Dict[Tuple[str, str, str], int] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(
        self,
        max_solver_time: float = 60.0,
        optimality_gap: float = 0.05,
    ) -> Dict[str, List[Task]]:
        """Build and solve the MKP; return resource → assigned-task mapping."""
        self._precompute()

        build_start = time.monotonic()
        model = cp_model.CpModel()
        self._create_normal_variables(model)
        self._create_mentoring_variables(model)
        self._add_assignment_constraints(model)
        self._add_capacity_constraints(model)
        self._add_mentoring_constraints(model)
        self._add_objective(model)
        build_time = time.monotonic() - build_start

        solve_start = time.monotonic()
        solver, status = self._run_solver(model, max_solver_time, optimality_gap)
        solve_time = time.monotonic() - solve_start

        assignments = self._extract_solution(solver, status)
        self.solver_meta["n_x_vars"] = len(self._x_normal)
        self.solver_meta["n_mentor_triples"] = len(self._x_mentor)
        self.solver_meta["build_time_seconds"] = round(build_time, 4)
        self.solver_meta["solve_time_seconds"] = round(solve_time, 4)
        return assignments

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    def _precompute(self) -> None:
        """Run all pre-computation required before model building."""
        self._activity_roles = self._populate_mentoring_cache()
        (
            self._task_feasible_real_resources,
            self._activity_task_ids,
            self._activity_total_work_hours,
            self._resource_capability_map,
            self._activity_supply_hours,
            self._direct_demand,
            self._indirect_demand,
        ) = self._build_diagnostics_and_demand()

        if self.mentoring_enabled and self.same_day_shortage_enabled:
            self._strategic_shortage_activities, _ = self._detect_same_day_activity_shortage(
                self._activity_total_work_hours,
                self._activity_supply_hours,
            )

        self._resource_pressure = self._calculate_resource_pressure()
        self._resource_spare_hours = self._calculate_resource_spare_hours()

        (
            self._bottleneck_lookup,
            self._severe_bottleneck_activities,
            self._detected_severe_bottleneck_activities,
            self._strategic_bottleneck_activities,
        ) = self._build_bottleneck_context()

        # Build activity demand scores and underutilization context once here so
        # both the objective builder and the metrics snapshot can reuse them.
        self._activity_demand_score = self._compute_activity_demand_scores()
        (
            self._mentor_substitutability_norm,
            self._underutilized_targets,
            self._growth_activity_scores,
        ) = self._build_underutilization_context()

    # ------------------------------------------------------------------
    # Variable creation
    # ------------------------------------------------------------------

    def _create_normal_variables(self, model: cp_model.CpModel) -> None:
        """Create x_normal[tid, rid] binary decision variables."""
        data = self.data
        for tid, task in data.task_objects.items():
            capable_rids = data.capable_resources.get(task.activity_name, [])
            for rid in data.resource_ids:
                if rid in capable_rids or rid == DUMMY_RESOURCE_ID:
                    self._x_normal[tid, rid] = model.NewBoolVar(f"x_{tid}_{rid}")

    def _create_mentoring_variables(self, model: cp_model.CpModel) -> None:
        """Create x_mentor[tid, rid, mid] variables and build reverse-index maps.

        Mentees are pre-filtered: only eligible mentees per activity are considered.
        A mentee is eligible if:
        - activity is in a bottleneck, or
        - activity is in a same-day shortage, or
        - mentee is an underutilized target and activity is in its growth scores.

        Bootstrap guardrail:
        - the bootstrap task itself is never modeled as mentoring
        - the bootstrap resource cannot be used in mentoring (as mentee or mentor)
          for the same bootstrap activity on that day.
        """
        if not self.mentoring_enabled:
            return

        data = self.data

        # Track (resource, activity) pairs that are currently in bootstrap mode.
        # Those pairs must not participate in same-day mentoring for that activity.
        bootstrap_pairs: set[tuple[str, str]] = set()
        for task in data.task_objects.values():
            if getattr(task, "bootstrap_assignment", False) and getattr(task, "bootstrap_resource_id", None):
                bootstrap_pairs.add((str(task.bootstrap_resource_id), str(task.activity_name)))

        # Build eligible mentee set per activity based on all three mentoring strategies.
        eligible_mentees_by_activity: Dict[str, set] = {}
        for activity_name, (mentees_all, _mentors_all) in self._activity_roles.items():
            eligible: set = set()
            if (
                activity_name in self._bottleneck_lookup
                or activity_name in self._strategic_shortage_activities
            ):
                eligible = set(mentees_all)
            else:
                for rid in mentees_all:
                    if rid not in self._underutilized_targets:
                        continue
                    growth_activities = self._growth_activity_scores.get(rid, {})
                    if activity_name in growth_activities:
                        eligible.add(rid)
            eligible_mentees_by_activity[activity_name] = eligible

        for tid in data.task_ids:
            task = data.task_objects[tid]
            activity_name = task.activity_name

            # Bootstrap task must execute as normal assignment, never as mentoring.
            if getattr(task, "bootstrap_assignment", False):
                continue

            _mentees_all, mentors = self._activity_roles[activity_name]
            eligible_mentees = eligible_mentees_by_activity.get(activity_name, set())
            if not eligible_mentees:
                continue

            # Prefer low-pressure mentors; fall back to full list if none qualify
            filtered_mentors = [
                (mid, cap)
                for mid, cap in mentors
                if self._resource_pressure.get(mid, 1.0) <= self.underutil_mentor_pressure_threshold
            ] or mentors

            for rid in eligible_mentees:
                if data.capacities.get(rid, 0) <= 0:
                    continue
                if data.durations.get((tid, rid)) is None:
                    continue

                # Resource that is currently bootstrapping this activity cannot be
                # used as mentee in same-day mentoring for the same activity.
                if (rid, activity_name) in bootstrap_pairs:
                    continue

                mentor_count = 0
                for mid, _ in filtered_mentors:
                    if mid == rid:
                        continue  # cannot mentor yourself

                    # Resource that is currently bootstrapping this activity cannot be
                    # used as mentor in same-day mentoring for the same activity.
                    if (mid, activity_name) in bootstrap_pairs:
                        continue

                    if (tid, mid) not in self._x_normal:
                        continue  # mentor needs a feasible duration
                    if rid == DUMMY_RESOURCE_ID or mid == DUMMY_RESOURCE_ID:
                        continue
                    self._x_mentor[tid, rid, mid] = model.NewBoolVar(f"xm_{tid}_{rid}_{mid}")
                    mentor_count += 1
                    if mentor_count >= MAX_MENTORS_PER_PAIR:
                        break

        # Build duration lookup and all three reverse indexes in one pass
        for tid, rid, mid in self._x_mentor:
            activity_name = data.task_objects[tid].activity_name
            mentor_dur = self.experience_store.get_duration(resource_id=rid, 
                                                            activity_name=activity_name, 
                                                            is_mentoring=True, 
                                                            mentor_experience_level=self.experience_store.get_capability_level(mid, activity_name),
                                                            mentee_experience_level=self.experience_store.get_capability_level(rid, activity_name),
                                                            required_capability_level=self.activity_requirements.get(activity_name, 0),
                                                            mentoring_config=self.mentoring_config)

            self._mentor_dur[(tid, rid, mid)] = mentor_dur
            self._mentor_shadow_dur[(tid, rid, mid)] = mentor_dur  # mentor shadow = full mentee duration

            var = self._x_mentor[tid, rid, mid]
            self._mentor_vars_by_task.setdefault(tid, []).append((rid, mid, var))
            self._mentor_vars_by_mentee.setdefault(rid, []).append((tid, mid, var))
            self._mentor_vars_by_mentor.setdefault(mid, []).append((tid, rid, var))
    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def _add_assignment_constraints(self, model: cp_model.CpModel) -> None:
        """Constraint 1: each task is assigned to exactly one execution mode."""
        data = self.data
        for tid in data.task_ids:
            task_obj = data.task_objects[tid]
            task_vars = [
                self._x_normal[tid, rid]
                for rid in data.resource_ids
                if (tid, rid) in self._x_normal
            ]
            if self.mentoring_enabled:
                task_vars += [var for _, _, var in self._mentor_vars_by_task.get(tid, [])]

            # Bootstrap tasks are hard-fixed to their designated resource for this day.
            if getattr(task_obj, "bootstrap_assignment", False):
                bootstrap_rid = getattr(task_obj, "bootstrap_resource_id", None)
                if not bootstrap_rid or (tid, bootstrap_rid) not in self._x_normal:
                    raise RuntimeError(
                        f"Bootstrap task {tid} has invalid bootstrap resource '{bootstrap_rid}' "
                        "or missing x_normal variable"
                    )
                model.Add(self._x_normal[tid, bootstrap_rid] == 1)

            if task_vars:
                model.add_exactly_one(task_vars)

    def _add_capacity_constraints(self, model: cp_model.CpModel) -> None:
        """Constraint 2: total work assigned to each resource must not exceed its capacity.

        For mentoring tasks the mentee pays the extended duration (multiplier + additive)
        and the mentor pays the same amount as shadow time.
        """
        data = self.data
        for rid in data.resource_ids:
            cap = data.capacities[rid]
            terms = []

            # Normal task execution
            for tid in data.task_ids:
                if (tid, rid) in self._x_normal:
                    terms.append(self._x_normal[tid, rid] * data.durations[tid, rid])

            if self.mentoring_enabled:
                # As mentee: extended duration (mentor-specific, since duration depends on mentor experience)
                for tid, mid, var in self._mentor_vars_by_mentee.get(rid, []):
                    m_dur = self._mentor_dur.get((tid, rid, mid), 0)
                    if m_dur > 0:
                        terms.append(var * m_dur)
                # As mentor: shadow time consumed supervising others
                for tid, mentee_rid, var in self._mentor_vars_by_mentor.get(rid, []):
                    # rid becomes the mentor
                    shadow = self._mentor_shadow_dur.get((tid, mentee_rid, rid), 0)
                    if shadow > 0:
                        terms.append(var * shadow)

            if terms:
                model.Add(sum(terms) <= cap)

    def _add_mentoring_constraints(self, model: cp_model.CpModel) -> None:
        """Constraint 3: learning-curve-driven minimum mentoring for bottleneck activities.

        For each detected bottleneck activity, selects the best mentee group (lowest
        remaining repetitions to reach required level) and enforces a minimum number
        of mentoring sessions today, scaled by days until the bottleneck arrives.
        """
        if not (
            self.mentoring_enabled
            and self.severe_bottleneck_mode == "constraint"
            and self._bottleneck_lookup
        ):
            return

        curve = self.learning_curve
        data = self.data

        for activity_name, bn in self._bottleneck_lookup.items():
            mentees_set, suitable_mentors = self._activity_roles[activity_name]
            required_level = self.activity_requirements.get(activity_name, 50.0)
            activity_task_count = len(self._activity_task_ids.get(activity_name, []))
            if activity_task_count == 0:
                continue

            # Build (rid, current_level, reps_needed) for each eligible mentee
            mentee_info: List[Tuple[str, float, int]] = []
            for rid in mentees_set:
                resource = data.resource_objects.get(rid)
                if resource is None:
                    continue
                cap_level = resource.get_experience_level(activity_name)
                profile = self._profile_cache_fn(self.experience_store, rid, activity_name, None)
                current_n = profile.count if profile else 0
                reps_needed = curve.repetitions_to_reach_level(current_n, required_level)
                if reps_needed <= 0:
                    continue
                mentee_info.append((rid, cap_level, reps_needed))

            if not mentee_info:
                logger.warning(
                    "Bottleneck activity '%s' has no eligible mentees — skipping constraint.",
                    activity_name,
                )
                continue

            # Focus on the mentee(s) closest to the required level (lowest reps remaining)
            lowest_reps = min(reps for _, _, reps in mentee_info)
            best_mentees = [
                (rid, cap, reps) for rid, cap, reps in mentee_info if reps == lowest_reps
            ]

            for rid, _, _ in best_mentees:
                self._y[(activity_name, rid)] = model.NewBoolVar(f"y_bn_{activity_name}_{rid}")

            group_vars: List[IntVar] = []
            per_mentee_feasible: Dict[str, int] = {}
            mentees_with_vars: List[str] = []

            for rid, _, _ in best_mentees:
                rid_entries: List[Tuple[int, str, int]] = []
                for tid in data.task_ids:
                    if data.task_objects[tid].activity_name != activity_name:
                        continue
                    for r, m, var in self._mentor_vars_by_task.get(tid, []):
                        if r != rid:
                            continue
                        # Link this var to the y-selector: if this session is active,
                        # rid must be the selected mentee for this activity.
                        model.AddImplication(var, self._y[(activity_name, rid)])
                        group_vars.append(var)
                        mentee_dur = self._mentor_dur.get((tid, rid, m), 0)
                        shadow = self._mentor_shadow_dur.get((tid, rid, m), mentee_dur)
                        rid_entries.append((mentee_dur, m, shadow))

                if not rid_entries:
                    logger.warning(
                        "Mentee %s in bottleneck activity '%s' has no feasible mentor pairings.",
                        rid, activity_name,
                    )
                    per_mentee_feasible[rid] = 0
                    continue

                mentees_with_vars.append(rid)
                avg_mentee_dur = sum(d for d, _, _ in rid_entries) / len(rid_entries)
                mentee_max = (
                    int(data.capacities.get(rid, 0) / avg_mentee_dur)
                    if avg_mentee_dur > 0 else 0
                )

                # Sum max sessions across all available mentors
                mentor_shadow_by_mid: Dict[str, List[int]] = {}
                for _, mid_c, shadow_c in rid_entries:
                    mentor_shadow_by_mid.setdefault(mid_c, []).append(shadow_c)
                total_mentor_max = 0
                for mid_c, shadows in mentor_shadow_by_mid.items():
                    avg_shadow = sum(shadows) / len(shadows)
                    if avg_shadow > 0:
                        total_mentor_max += int(data.capacities.get(mid_c, 0) / avg_shadow)

                per_mentee_feasible[rid] = min(mentee_max, total_mentor_max)

            if not group_vars:
                continue

            # Exactly one mentee receives all mentoring sessions today
            if mentees_with_vars:
                model.add_exactly_one(
                    self._y[(activity_name, rid)] for rid in mentees_with_vars
                )

            reps_required = max(min(lowest_reps, activity_task_count, len(group_vars)), 0)
            if reps_required == 0:
                continue

            # Scale daily target based on urgency
            days_left = bn.days_until_bottleneck
            if bn.severity == "severe" or days_left <= 1:
                daily_target = reps_required
            elif days_left == 2:
                daily_target = math.ceil(reps_required * 2 / 3)
            else:
                daily_target = math.ceil(reps_required / 2)

            if activity_name in self._strategic_bottleneck_activities:
                daily_target = min(daily_target, self.same_day_shortage_quota)
            daily_target = max(daily_target, 1)

            feasible_cap = min(per_mentee_feasible.values()) if per_mentee_feasible else 0
            if feasible_cap <= 0:
                continue
            daily_target = min(daily_target, feasible_cap)

            model.Add(sum(group_vars) >= daily_target)
            logger.info(
                "MKP: Learning-curve constraint for '%s': >= %d session(s) today "
                "(reps_needed=%d, tasks_avail=%d, severity=%s, days_to_bn=%d)",
                activity_name, daily_target, lowest_reps,
                activity_task_count, bn.severity, days_left,
            )

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _add_objective(self, model: cp_model.CpModel) -> None:
        """Assemble and set the maximization objective from all component terms."""
        terms: List[Any] = []
        terms += self._build_normal_value_terms()
        if self.mentoring_enabled:
            terms += self._build_mentoring_value_terms()
        terms += self._build_utilization_terms()
        model.Maximize(sum(terms))

    def _build_normal_value_terms(self) -> list:
        """Dummy-resource penalty and normal-execution value per (task, resource) pair.

        Normal value = w_pressure × pressure + w_priority × deferral_urgency
        """
        data = self.data
        terms = []
        w_pressure = self.objective_weights.get("pressure",          DEFAULT_OBJECTIVE_WEIGHTS["pressure"])
        w_priority = self.objective_weights.get("deferral_priority", DEFAULT_OBJECTIVE_WEIGHTS["deferral_priority"])

        for tid in data.task_ids:
            for rid in data.resource_ids:
                if (tid, rid) not in self._x_normal:
                    continue
                if rid == DUMMY_RESOURCE_ID:
                    min_dur = max(data.min_real_durations.get(tid, 1), 1)
                    penalty = int(DUMMY_PENALTY_BASE * min_dur * OBJECTIVE_SCALE)
                    terms.append(self._x_normal[tid, rid] * (-penalty))
                else:
                    pressure = self._resource_pressure.get(rid, 0.0)
                    if pressure <= 0:
                        continue
                    defer_count = getattr(data.task_objects[tid], "defer_count", 0)
                    v_priority = min(1.0, defer_count / max(self.max_task_deferrals, 1))
                    value = int((w_pressure * pressure + w_priority * v_priority) * OBJECTIVE_SCALE)
                    terms.append(self._x_normal[tid, rid] * value)
        return terms

    def _build_mentoring_value_terms(self) -> list:
        """BRR bonus, underutilization bonus, and shortage bonus for mentoring vars.

        Three components:
        (a) BRR term — forecast-oriented bottleneck risk reduction
        (b) Underutilization bonus — proactive capacity broadening
        (c) Shortage bonus — same-day or severe-bottleneck urgency
        """
        data = self.data
        terms = []
        w_bottleneck   = self.objective_weights.get("bottleneck",      DEFAULT_OBJECTIVE_WEIGHTS["bottleneck"])
        w_underutil    = self.objective_weights.get("underutilization", DEFAULT_OBJECTIVE_WEIGHTS["underutilization"])
        w_shortage     = self.objective_weights.get("shortage",         DEFAULT_OBJECTIVE_WEIGHTS["shortage"])

        brr = self._calculate_brr_values()
        # Use pre-computed underutilization context (set in _precompute)
        norm_sub         = self._mentor_substitutability_norm
        underutilized_targets = self._underutilized_targets
        growth_scores    = self._growth_activity_scores

        for (tid, rid, mid), var in self._x_mentor.items():
            base_activity = data.task_objects[tid].activity_name

            # Strategy 1 -> Term (a) in Equation 4.11: Underutilization bonus
            if self.underutilization_enabled and rid in underutilized_targets:
                activity_score = growth_scores.get(rid, {}).get(base_activity, 0.0)
                if activity_score > 0.0:
                    mentor_pressure = self._resource_pressure.get(mid, 1.0)
                    pressure_factor = max(
                        0.0,
                        (self.underutil_mentor_pressure_threshold - mentor_pressure)
                        / max(self.underutil_mentor_pressure_threshold, 1e-6),
                    )
                    substitutability = norm_sub.get((base_activity, mid), 0.0)
                    target_score = underutilized_targets[rid]
                    bonus = int(
                        self.underutil_bonus_scale * target_score * activity_score
                        * (0.3 + 0.7 * pressure_factor)
                        * (0.3 + 0.7 * substitutability)
                        * OBJECTIVE_SCALE
                    )
                    if bonus > 0:
                        terms.append(var * int(bonus * w_underutil))

            # Strategy 2a -> Term (b) in Equation 4.11: Shortage / severe-bottleneck bonus
            if (
                self.severe_bottleneck_mode == "objective_bonus"
                and base_activity in self._severe_bottleneck_activities
            ):
                bonus_units = BOTTLENECK_BONUS
                # Strategic (shortage-driven) activities get a reduced bonus compared
                # to activities with a hard-detected severe bottleneck.
                if (
                    base_activity in self._strategic_bottleneck_activities
                    and base_activity not in self._detected_severe_bottleneck_activities
                ):
                    bonus_units = max(
                        1, int(BOTTLENECK_BONUS * self.same_day_shortage_bonus_mult)
                    )
                terms.append(var * int(OBJECTIVE_SCALE * bonus_units * w_shortage))
                
            # Strategy 2b -> Term (c) in Equation 4.11: BRR bonus
            brr_scaled = int(brr.get((tid, rid), 0.0) * OBJECTIVE_SCALE)
            terms.append(var * int(brr_scaled * w_bottleneck))

        return terms

    def _build_utilization_terms(self) -> list:
        """Utilization maximization terms for all normal and mentoring assignments."""
        data = self.data
        terms = []
        w_util = self.objective_weights.get("utilization", DEFAULT_OBJECTIVE_WEIGHTS["utilization"])
        util_scale = self._calculate_utilization_scale()

        for rid in data.resource_ids:
            if rid == DUMMY_RESOURCE_ID:
                continue
            scale = util_scale[rid]
            if scale == 0.0:
                continue

            for tid in data.task_ids:
                if (tid, rid) in self._x_normal:
                    dur = data.durations.get((tid, rid), 0)
                    terms.append(self._x_normal[tid, rid] * int(scale * dur * w_util))

            if self.mentoring_enabled:
                for tid, mid, var in self._mentor_vars_by_mentee.get(rid, []):
                    m_dur = self._mentor_dur.get((tid, rid, mid), 0)
                    terms.append(var * int(scale * m_dur * w_util))
                for tid, mentee_rid, var in self._mentor_vars_by_mentor.get(rid, []):
                    # rid becomes the mentor
                    shadow = self._mentor_shadow_dur.get((tid, mentee_rid, rid), 0)
                    terms.append(var * int(scale * shadow * w_util))

        return terms

    # ------------------------------------------------------------------
    # Solver invocation and solution extraction
    # ------------------------------------------------------------------

    def _run_solver(
        self, model: cp_model.CpModel, max_solver_time: float, optimality_gap: float
    ) -> Tuple[cp_model.CpSolver, int]:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max_solver_time
        solver.parameters.relative_gap_limit = optimality_gap
        status = solver.Solve(model)
        return solver, status

    def _extract_solution(
        self, solver: cp_model.CpSolver, status: int
    ) -> Dict[str, List[Task]]:
        """Parse solver output into the resource → task assignment dict."""
        data = self.data
        assignments: Dict[str, List[Task]] = {rid: [] for rid in data.resource_ids}

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            self.solver_meta = {"status": f"FAILED_{status}", "objective_value": None}
            raise RuntimeError(f"MKP CP-SAT solver failed: status={status}")

        mentor_count = 0
        for tid in data.task_ids:
            assigned = False

            # Mentoring assignment takes reporting priority over normal assignment
            for rid, mid, var in self._mentor_vars_by_task.get(tid, []):
                if solver.Value(var) != 1:
                    continue
                task = data.task_objects[tid]
                m_dur = self._mentor_dur.get((tid, rid, mid), data.durations.get((tid, rid)))
                task.estimated_duration = m_dur
                task.mentee_resource_id = rid
                task.mentor_resource_id = mid
                task.task_type = TaskType.MENTORING
                task.learning_opportunity = True
                assignments[rid].append(task)
                mentor_count += 1
                assigned = True

                bn = self._bottleneck_lookup.get(task.activity_name)
                task.mentoring_against_bottleneck = (
                    (task.activity_name, rid) in self._y
                    and solver.Value(self._y[(task.activity_name, rid)]) == 1
                )
                task.is_emergency_mentoring = bn is not None and (
                    bn.severity == "severe"
                    or (bn.severity == "medium" and bn.days_until_bottleneck <= 1)
                )
                logger.info(
                    "MKP: Mentoring — %s mentee=%s mentor=%s dur=%.1fh emergency=%s",
                    task.activity_name, rid, mid,
                    seconds_to_hours(m_dur), task.is_emergency_mentoring,
                )
                break  # each task has exactly one active assignment

            if assigned:
                continue

            # Normal assignment
            for rid in data.resource_ids:
                if (tid, rid) not in self._x_normal:
                    continue
                if solver.Value(self._x_normal[tid, rid]) != 1:
                    continue
                task = data.task_objects[tid]
                # Defensively clear any stale mentoring metadata from previous cycles
                task.task_type = TaskType.STANDARD
                task.mentee_resource_id = None
                task.mentor_resource_id = None
                task.learning_opportunity = False
                task.mentoring_against_bottleneck = False
                task.is_emergency_mentoring = False
                task.estimated_duration = (
                    data.durations[(tid, rid)]
                    if rid != DUMMY_RESOURCE_ID
                    else data.mean_real_durations.get(tid, 3600)
                )
                assignments[rid].append(task)
                assigned = True
                break

            if not assigned:
                raise RuntimeError(
                    f"CP-SAT violated add_exactly_one: task {tid} has no active variable in solution"
                )

        status_name = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
        self.solver_meta = {
            "status": status_name,
            "objective_value": solver.ObjectiveValue(),
            "mentor_assignments": mentor_count,
        }
        return assignments

    # ------------------------------------------------------------------
    # Pre-computation helpers
    # ------------------------------------------------------------------

    def _populate_mentoring_cache(
        self,
    ) -> Dict[str, Tuple[set, List[Tuple[str, float]]]]:
        """Build mentee / mentor role lookup per activity from current experience levels."""
        activity_roles: Dict[str, Tuple[set, List[Tuple[str, float]]]] = {}
        activities_of_pending_tasks = set(
            task.activity_name for task in self.data.task_objects.values()
        )
        for activity in activities_of_pending_tasks:
            required_level = self.activity_requirements.get(activity, 50.0)
            mentee_set: set = set()
            mentor_list: List[Tuple[str, float]] = []
            for rid in self._real_resource_ids:
                resource = self.data.resource_objects.get(rid)
                if resource is None:
                    continue
                if self.data.capacities.get(rid, 0) <= 0:
                    continue
                cap = float(resource.get_experience_level(activity))
                if cap < required_level:
                    mentee_set.add(rid)
                else:
                    mentor_list.append((rid, cap))
            mentor_list.sort(key=lambda x: x[1], reverse=True)
            activity_roles[activity] = (mentee_set, mentor_list)
        return activity_roles

    def _build_diagnostics_and_demand(
        self,
    ) -> Tuple[
        Dict[str, List[str]],
        Dict[str, List[str]],
        Dict[str, float],
        Dict[str, Dict[str, List[str]]],
        Dict[str, float],
        Dict[str, float],
        Dict[str, float],
    ]:
        """Compute task feasibility, activity groupings, work hours, and per-resource demand."""
        data = self.data
        real_ids = self._real_resource_ids

        task_feasible: Dict[str, List[str]] = {}
        activity_task_ids: Dict[str, List[str]] = {}
        activity_work_sec: Dict[str, int] = {}
        resource_capability_map: Dict[str, Dict[str, List[str]]] = {rid: {} for rid in real_ids}
        direct_demand: Dict[str, float] = {rid: 0.0 for rid in real_ids}
        indirect_demand: Dict[str, float] = {rid: 0.0 for rid in real_ids}

        for tid in data.task_ids:
            activity = data.task_objects[tid].activity_name
            activity_task_ids.setdefault(activity, []).append(tid)
            mean_dur = int(data.mean_real_durations.get(tid, 0))
            activity_work_sec[activity] = activity_work_sec.get(activity, 0) + mean_dur

            feasible = [rid for rid in real_ids if (tid, rid) in self._x_normal]
            task_feasible[tid] = feasible
            n = len(feasible)
            if n > 0:
                for rid in feasible:
                    resource_capability_map[rid].setdefault(activity, []).append(tid)
                    dur = float(data.durations.get((tid, rid), 0))
                    if n == 1:
                        direct_demand[rid] += dur
                    else:
                        indirect_demand[rid] += dur / n

        activity_total_work_hours = {
            a: seconds_to_hours(s) for a, s in activity_work_sec.items()
        }

        # Activity supply: sum of calendar capacity of currently capable resources
        activity_supply_hours: Dict[str, float] = {}
        for activity in activity_task_ids:
            _, mentors = self._activity_roles.get(activity, (set(), []))
            supply_sec = sum(float(self.data.capacities.get(mid, 0.0)) for mid, _ in mentors)
            activity_supply_hours[activity] = seconds_to_hours(supply_sec)

        return (
            task_feasible,
            activity_task_ids,
            activity_total_work_hours,
            resource_capability_map,
            activity_supply_hours,
            direct_demand,
            indirect_demand,
        )

    def _detect_same_day_activity_shortage(
        self,
        activity_demand_hours: Dict[str, float],
        activity_supply_hours: Dict[str, float],
    ) -> Tuple[set, Dict[str, float]]:
        """Return activities whose demand/supply ratio exceeds the shortage threshold."""
        strategic: set = set()
        ratios: Dict[str, float] = {}
        for activity in activity_demand_hours:
            demand = float(activity_demand_hours.get(activity, 0.0))
            supply = float(activity_supply_hours.get(activity, 0.0))
            if demand <= 0.0:
                ratio = 0.0
            elif supply <= 0.0:
                ratio = 999.0
            else:
                ratio = demand / supply
            ratios[activity] = ratio
            if ratio >= self.same_day_shortage_strong_ratio:
                strategic.add(activity)

        if self.same_day_shortage_debug:
            print("\n=== SAME-DAY ACTIVITY SHORTAGE ===")
            for a in sorted(ratios):
                print(f"  {a}: ratio={ratios[a]:.3f}")
            print(
                f"  strategic={sorted(strategic)} "
                f"(threshold={self.same_day_shortage_strong_ratio:.2f})"
            )
        return strategic, ratios

    def _build_bottleneck_context(
        self,
    ) -> Tuple[Dict[str, BottleneckInfo], set, set, set]:
        """Build bottleneck lookup and merge detected + shortage-driven bottlenecks."""
        bottleneck_lookup: Dict[str, BottleneckInfo] = {}
        severe: set = set()
        detected_severe: set = set()
        strategic: set = set()

        if self.bottleneck_activity_strategy_enabled:
            for bn in self.data.bottlenecks:
                bottleneck_lookup[bn.activity_name] = bn
                if bn.severity == "severe":
                    severe.add(bn.activity_name)
                    detected_severe.add(bn.activity_name)

        for activity in self._strategic_shortage_activities:
            mentees, mentors = self._activity_roles.get(activity, (set(), []))
            mentor_ids = [mid for mid, _ in mentors]
            if not mentor_ids or not mentees:
                continue
            if activity not in bottleneck_lookup:
                bottleneck_lookup[activity] = BottleneckInfo(
                    activity_name=activity,
                    severity="severe",
                    days_until_bottleneck=0,
                    capable_resource_count=len(mentor_ids),
                    fit_resource_ids=mentor_ids,
                    mentee_candidates=sorted(mentees),
                    mentor_candidates=mentor_ids,
                )
            severe.add(activity)
            strategic.add(activity)

        return bottleneck_lookup, severe, detected_severe, strategic

    def _calculate_resource_pressure(self) -> Dict[str, float]:
        """Utilization pressure per resource: clamp(demand / capacity, FLOOR, 1).

        Unavailable resources (capacity = 0) get pressure = 0, which zeroes out
        their normal execution value and prevents the solver from assigning to them.
        """
        pressure: Dict[str, float] = {}
        data = self.data
        for rid in self._real_resource_ids:
            cap = data.capacities.get(rid, 0)
            if cap <= 0:
                pressure[rid] = 0.0
                continue
            demand = self._direct_demand.get(rid, 0.0) + self._indirect_demand.get(rid, 0.0)
            pressure[rid] = max(PRESSURE_FLOOR, min(1.0, demand / cap))
        pressure[DUMMY_RESOURCE_ID] = 0.0
        return pressure

    def _calculate_resource_spare_hours(self) -> Dict[str, float]:
        """Spare calendar hours per resource: max(0, capacity − estimated demand)."""
        spare: Dict[str, float] = {}
        data = self.data
        for rid in self._real_resource_ids:
            cap = float(data.capacities.get(rid, 0.0))
            demand = self._direct_demand.get(rid, 0.0) + self._indirect_demand.get(rid, 0.0)
            spare[rid] = max(0.0, seconds_to_hours(cap - demand)) if cap > 0.0 else 0.0
        return spare

    def _compute_activity_demand_scores(self) -> Dict[str, float]:
        """Per-activity demand score: task count + work hours, boosted near bottlenecks.

        Higher score → more strategically valuable to upskill resources for this activity.
        """
        scores: Dict[str, float] = {}
        for activity, task_ids in self._activity_task_ids.items():
            bn = self._bottleneck_lookup.get(activity)
            bn_mult = 1.0
            if bn is not None:
                bn_mult += (1.0 if bn.severity == "severe" else 0.5) / max(bn.days_until_bottleneck, 1)
            scores[activity] = (
                len(task_ids) + self._activity_total_work_hours.get(activity, 0.0)
            ) * bn_mult
        return scores

    def _build_underutilization_context(
        self,
    ) -> Tuple[
        Dict[Tuple[str, str], float],
        Dict[str, float],
        Dict[str, Dict[str, float]],
    ]:
        """Compute mentor substitutability and underutilized resource scores.

        Uses ``self._activity_demand_score`` (must be populated before this call).
        """
        activity_task_ids      = self._activity_task_ids
        activity_roles         = self._activity_roles
        task_feasible          = self._task_feasible_real_resources
        resource_capability    = self._resource_capability_map
        resource_pressure      = self._resource_pressure
        resource_spare         = self._resource_spare_hours
        activity_demand_score  = self._activity_demand_score

        # Mentor substitutability: how many other capable resources exist for each task
        raw_sub: Dict[Tuple[str, str], float] = {}
        for activity, (_, mentors) in activity_roles.items():
            for mid, _ in mentors:
                relevant = [
                    tid for tid in activity_task_ids.get(activity, [])
                    if mid in task_feasible.get(tid, [])
                ]
                if not relevant:
                    continue
                avg_other = sum(
                    max(len(task_feasible.get(tid, [])) - 1, 0) for tid in relevant
                ) / len(relevant)
                raw_sub[(activity, mid)] = avg_other

        max_sub: Dict[str, float] = {}
        for (activity, _), val in raw_sub.items():
            max_sub[activity] = max(max_sub.get(activity, 0.0), val)
        norm_sub: Dict[Tuple[str, str], float] = {
            (a, mid): min(max(val / max(max_sub.get(a, 0.0), 1e-6), 0.0), 1.0)
            for (a, mid), val in raw_sub.items()
        }

        # Identify underutilized resources and their preferred growth activities
        underutilized: Dict[str, float] = {}
        growth_scores: Dict[str, Dict[str, float]] = {}
        if self.mentoring_enabled and self.underutilization_enabled:
            for rid in self._real_resource_ids:
                pressure     = resource_pressure.get(rid, 0.0)
                spare        = resource_spare.get(rid, 0.0)
                act_count    = len(resource_capability.get(rid, {}))
                if (
                    pressure  > self.underutil_target_pressure_threshold
                    or spare  < self.underutil_min_spare_hours
                    or act_count > self.underutil_max_current_activities
                ):
                    continue
                pressure_score  = max(0.0, (self.underutil_target_pressure_threshold - pressure)
                                      / max(self.underutil_target_pressure_threshold, 1e-6))
                spare_score     = min(1.0, spare / max(self.underutil_min_spare_hours, 1e-6))
                narrowness      = max(0.0, (self.underutil_max_current_activities - act_count + 1)
                                      / max(self.underutil_max_current_activities, 1))
                target_score    = min(
                    max(0.45 * pressure_score + 0.35 * spare_score + 0.20 * narrowness, 0.0), 1.0
                )
                underutilized[rid] = target_score

                candidates = {
                    activity: activity_demand_score[activity]
                    for activity, (mentees, _) in activity_roles.items()
                    if rid in mentees and activity_demand_score.get(activity, 0.0) > 0.0
                }
                if candidates:
                    mx = max(candidates.values())
                    growth_scores[rid] = {a: s / mx for a, s in candidates.items()} if mx > 0.0 else {}

        return norm_sub, underutilized, growth_scores

    def _calculate_utilization_scale(self) -> Dict[str, float]:
        """Scale factor for the utilization objective term: OBJECTIVE_SCALE / capacity."""
        scale: Dict[str, float] = {}
        for rid in self.data.resource_ids:
            if rid == DUMMY_RESOURCE_ID:
                scale[rid] = 0.0
                continue
            cap = self.data.capacities.get(rid, 0)
            scale[rid] = (OBJECTIVE_SCALE / cap) if cap > 0 else 0.0
        return scale

    def _calculate_brr_values(self) -> Dict[Tuple[str, str], float]:
        """Pre-compute BRR (Bottleneck Risk Reduction) for every (task, mentee) pair.

        BRR = gap_ratio × severity_score × proximity × (1 / capable_count)

        where gap_ratio = mentee_level / required_level.  A mentee already at
        the required level contributes gap_ratio = 1; one at zero contributes 0.
        BRR is zero when no bottleneck exists for the activity (rho_a = 0).
        """
        brr: Dict[Tuple[str, str], float] = {}
        for (tid, rid, _mid) in self._x_mentor:
            if (tid, rid) in brr:
                continue
            task = self.data.task_objects[tid]
            bn = self._bottleneck_lookup.get(task.activity_name)
            if bn is None:
                brr[(tid, rid)] = 0.0
                continue
            resource = self.data.resource_objects.get(rid)
            required = self.activity_requirements.get(task.activity_name, 50.0)
            cap_level = resource.get_experience_level(task.activity_name) if resource else 0.0
            gap_ratio = cap_level / required if required > 0 else 0.0
            severity  = 1.0 if bn.severity == "severe" else 0.5
            proximity = 1.0 / bn.days_until_bottleneck if bn.days_until_bottleneck > 0 else 1.0
            brr[(tid, rid)] = gap_ratio * severity * proximity * (1.0 / max(bn.capable_resource_count, 1))
        return brr

    # ------------------------------------------------------------------
    # Metrics snapshot (called by the parent scheduler for JSONL logging)
    # ------------------------------------------------------------------

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """Return a flat dict of pre-computed scheduling-context metrics.

        Meant to be called after :meth:`solve` and merged into the daily JSONL row.
        All values reference the state as of the scheduling cycle that produced this
        formulator — they are not updated after ``solve()`` completes.

        Keys
        ----
        resource_pressure_per_resource : dict[str, float]
            Π_r for each real resource.
        demand_supply_ratio_per_activity : dict[str, float]
            ξ_a = demand_hours / supply_hours for each activity with pending tasks.
        capable_resources_per_activity : dict[str, int]
            |R^cap_a| — number of capable resources per activity.
        activity_demand_score_per_activity : dict[str, float]
            Combined task-count + work-hours score, boosted by bottleneck proximity.
        mentor_substitutability_per_activity : dict[str, float]
            Mean normalised substitutability score across all mentors per activity.
        strategic_shortage_activities : list[str]
            Activities flagged as same-day strategic shortage (ξ_a ≥ threshold).
        bottleneck_per_activity : dict[str, dict]
            Per-activity bottleneck summary {severity, days_until_bottleneck,
            capable_resource_count} for detected bottlenecks.
        underutilized_resources : dict[str, float]
            Resources identified as underutilized with their target score.
        """
        # demand/supply ratio — recompute ratios (strategic set is already stored)
        _, ratios = self._detect_same_day_activity_shortage(
            self._activity_total_work_hours,
            self._activity_supply_hours,
        )

        # capable resources per activity (from activity_roles)
        capable_counts: Dict[str, int] = {
            activity: len(mentors)
            for activity, (_, mentors) in self._activity_roles.items()
        }

        # mean normalised substitutability per activity
        sub_by_activity: Dict[str, List[float]] = {}
        for (activity, _mid), val in self._mentor_substitutability_norm.items():
            sub_by_activity.setdefault(activity, []).append(val)
        mean_sub: Dict[str, float] = {
            a: sum(vals) / len(vals) for a, vals in sub_by_activity.items()
        }

        # bottleneck summary
        bn_summary: Dict[str, Dict[str, Any]] = {
            activity: {
                "severity": bn.severity,
                "days_until_bottleneck": bn.days_until_bottleneck,
                "capable_resource_count": bn.capable_resource_count,
            }
            for activity, bn in self._bottleneck_lookup.items()
        }

        return {
            "resource_pressure_per_resource": dict(self._resource_pressure),
            "demand_supply_ratio_per_activity": ratios,
            "capable_resources_per_activity": capable_counts,
            "activity_demand_score_per_activity": dict(self._activity_demand_score),
            "mentor_substitutability_per_activity": mean_sub,
            "strategic_shortage_activities": sorted(self._strategic_shortage_activities),
            "bottleneck_per_activity": bn_summary,
            "underutilized_resources": dict(self._underutilized_targets),
        }
