"""Capability-greedy batch scheduler with capacity constraints."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .base import Scheduler, SchedulingContext, DUMMY_RESOURCE_ID


class GreedyScheduler(Scheduler):
    """Assign each task to the feasible resource with highest capability.

    Tie-break order:
    1. Highest capability level for task activity
    2. Highest remaining capacity
    3. Lowest resource ID (deterministic)
    """

    def __init__(
        self,
        name: Optional[str] = None,
        duration_predictor: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        time_converter: Optional[Any] = None,
    ):
        super().__init__(
            name=name or "Greedy",
            duration_predictor=duration_predictor,
            config=config,
            time_converter=time_converter,
        )

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

        for tid in data.task_ids:
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

            def _key(rid: str) -> tuple:
                cap_level = float(data.resource_objects[rid].get_experience_level(task.activity_name))
                return (-cap_level, -remaining_capacity.get(rid, 0), rid)

            selected_rid = min(feasible, key=_key)
            selected_duration = int(data.durations[(tid, selected_rid)] or 0)
            task.estimated_duration = selected_duration
            assignments.setdefault(selected_rid, []).append(task)
            remaining_capacity[selected_rid] = max(
                0, remaining_capacity[selected_rid] - selected_duration
            )

        self._last_solver_meta = {
            "status": "GREEDY",
            "objective_value": None,
        }

        return assignments