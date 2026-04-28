"""Random batch scheduler with capability and capacity constraints."""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional

from .base import Scheduler, SchedulingContext, DUMMY_RESOURCE_ID


class RandomScheduler(Scheduler):
    """Assign tasks randomly among feasible resources.

    Feasible means:
    - Resource is capable of the activity (if capability constraints enabled)
    - Resource has enough remaining capacity within the planning horizon
    """

    def __init__(
        self,
        name: Optional[str] = None,
        duration_predictor: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        time_converter: Optional[Any] = None,
        random_seed: Optional[int] = None,
    ):
        super().__init__(
            name=name or "Random",
            duration_predictor=duration_predictor,
            config=config,
            time_converter=time_converter,
        )
        cfg_seed = None
        if config:
            cfg_seed = config.get("scheduler", {}).get(
                "random_seed", config.get("simulation", {}).get("random_seed", 42)
            )
        self._rng = random.Random(random_seed if random_seed is not None else cfg_seed)

    def plan_tasks_to_resources(
        self,
        context: SchedulingContext,
        planning_horizon_hours: float = 24.0,
        enforce_working_hours: bool = True,
        duration_predictor: Optional[Any] = None,
        max_solver_time_seconds: float = 60.0,
        optimality_gap: float = 0.05,
    ) -> Dict[str, List[Any]]:
        del max_solver_time_seconds, optimality_gap
        solve_start = time.monotonic()

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
            bottlenecks=[],
        )

        real_resource_ids = [rid for rid in data.resource_ids if rid != DUMMY_RESOURCE_ID]
        remaining_capacity: Dict[str, int] = {
            rid: int(data.capacities.get(rid, 0) or 0) for rid in real_resource_ids
        }

        assignments: Dict[str, List[Any]] = {DUMMY_RESOURCE_ID: []}
        task_ids = list(data.task_ids)
        self._rng.shuffle(task_ids)

        for tid in task_ids:
            task = data.task_objects[tid]
            capable = data.capable_resources.get(str(task.activity_name), [])
            feasible = [
                rid for rid in capable
                if int(data.durations.get((tid, rid), 0) or 0) <= remaining_capacity.get(rid, 0)
            ]

            if not feasible:
                task.estimated_duration = int(data.mean_real_durations.get(tid, 3600) or 3600)
                assignments[DUMMY_RESOURCE_ID].append(task)
                continue

            selected_rid = self._rng.choice(feasible)
            selected_duration = int(data.durations[(tid, selected_rid)] or 0)
            task.estimated_duration = selected_duration
            assignments.setdefault(selected_rid, []).append(task)
            remaining_capacity[selected_rid] = max(
                0, remaining_capacity[selected_rid] - selected_duration
            )

        self._last_solver_meta = {
            "status": "RANDOM",
            "objective_value": None,
        }

        return assignments