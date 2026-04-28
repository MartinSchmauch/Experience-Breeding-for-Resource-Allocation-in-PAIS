"""Simulation state management."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from ..entities import Resource, Task
from ..entities.case import Case
from ..entities.case import Case, CompletedActivity
from ..scheduling.experience_based import DUMMY_RESOURCE_ID

@dataclass
class SimulationState:
    """
    Current state of the simulation with efficient set-based state tracking.
    
    Uses separate sets for each state (queued, active, completed) to enable
    O(1) state transitions and membership checks instead of O(n) list operations.
    
    Attributes:
        resources: Dictionary of all resources (id -> Resource)
        cases: Dictionary of all cases (id -> Case)
        tasks: Dictionary of all tasks (id -> Task)
        current_time: Current simulation clock time
        
        # Task states (mutually exclusive, set-based for O(1) operations)
        queued_tasks: Set of task IDs that have arrived but not started
        active_tasks: Set of task IDs currently being executed
        completed_tasks: Set of task IDs that have finished
        cancelled_tasks: Set of task IDs that were cancelled (e.g. due to max deferrals)
        
        # Case states
        active_cases: Set of case IDs currently in progress
        completed_cases: Set of case IDs that have finished
        cancelled_cases: Set of case IDs that were cancelled (e.g. due to max deferrals)
        
        # Resource utilization tracking
        resource_busy_time: Total time each resource has been busy
        resource_last_start: Timestamp when resource last became busy
    """
    # Entity storage (all entities ever in simulation)
    resources: Dict[str, Resource] = field(default_factory=dict)
    cases: Dict[str, Case] = field(default_factory=dict)
    tasks: Dict[str, Task] = field(default_factory=dict)
    current_time: float = 0.0
    
    # Task state tracking (sets for O(1) add/remove/check operations)
    queued_tasks: set = field(default_factory=set, init=False, repr=False)
    active_tasks: set = field(default_factory=set, init=False, repr=False)
    completed_tasks: set = field(default_factory=set, init=False, repr=False)
    cancelled_tasks: set = field(default_factory=set, init=False, repr=False)
    
    # Case state tracking
    active_cases: set = field(default_factory=set, init=False, repr=False)
    completed_cases: set = field(default_factory=set, init=False, repr=False)
    cancelled_cases: set = field(default_factory=set, init=False, repr=False)
    
    # Resource utilization tracking
    resource_busy_time: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    resource_last_start: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    
    def add_resource(self, resource: Resource) -> None:
        """Add a resource to the simulation."""
        self.resources[resource.id] = resource
        self.resource_busy_time[resource.id] = 0.0
        self.resource_last_start[resource.id] = 0.0
    
    def add_case(self, case: Case) -> None:
        """Add a case to the simulation."""
        self.cases[case.id] = case
        self.active_cases.add(case.id)
    
    def add_task(self, task: Task) -> None:
        """Add a task to the simulation in queued state."""
        self.tasks[task.id] = task
        self.queued_tasks.add(task.id)
    
    def get_all_resource_queue_lengths(self, resource_stores: Dict) -> Dict[str, int]:
        """Get the current queue lengths for all resources.
        
        Args:
            resource_stores: Dict of resource stores from engine
            
        Returns:
            Number of tasks waiting in resource's queue for each resource
        """
        queue_lengths = {resource_id: len(resource_stores[resource_id].items)
            for resource_id in self.resources.keys()}
        # queue_lengths = {}
        # for resource_id, store in resource_stores.items():
        #     queue_lengths[resource_id] = len(store.items)
        return queue_lengths
    
    def get_available_resources(self) -> List[Resource]:
        """Get list of currently available resources."""
        return [
            resource for resource in self.resources.values()
            if resource.is_available()
        ]
    
    def get_pending_tasks(self) -> List[Task]:
        """Get list of tasks in queued state (arrived but not started).
        
        O(n) where n = number of queued tasks.
        """
        return [self.tasks[task_id] for task_id in self.queued_tasks]
        
    def get_active_tasks(self) -> List[Task]:
        """Get list of tasks currently being executed.
        
        O(n) where n = number of active tasks.
        """
        return [self.tasks[task_id] for task_id in self.active_tasks]
        
    def get_finished_tasks(self) -> List[Task]:
        """Get list of completed tasks.
        
        O(n) where n = number of completed tasks.
        """
        return [self.tasks[task_id] for task_id in self.completed_tasks]
    
    def get_active_cases(self) -> List[Case]:
        """Get list of cases currently in progress.
        
        O(n) where n = number of active cases.
        """
        return [self.cases[case_id] for case_id in self.active_cases]
        
    def get_finished_cases(self) -> List[Case]:
        """Get list of completed cases.
        
        O(n) where n = number of completed cases.
        """
        return [self.cases[case_id] for case_id in self.completed_cases if case_id in self.cases]
    
    def get_all_cases(self) -> List[Case]:
        """Get list of all cases in the simulation."""
        return list(self.cases.values())
    
    def get_all_tasks(self) -> List[Task]:
        """Get list of all tasks in the simulation."""
        return list(self.tasks.values())
    
    def start_task(self, task_id: str, start_time: float, resource_id: str, duration: float) -> None:
        """
        Transition task from QUEUED to ACTIVE state.
        
        O(1) operation using set add/remove.
        
        Args:
            task_id: ID of task to start
            start_time: Start time
            resource_id: ID of resource to assign task to
            duration: Estimated duration of the task
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        task = self.tasks[task_id]
        task.assign_to_resource(resource_id, duration)
        
        # State transition: QUEUED -> ACTIVE
        self.queued_tasks.discard(task_id)
        self.active_tasks.add(task_id)
        
        # Update task entity
        task.start(start_time)
        
        if task.assigned_resource_id:
            resource_id = task.assigned_resource_id
            if resource_id not in self.resource_last_start:
                self.resource_last_start[resource_id] = start_time
    
    def complete_task(self, task_id: str, end_time: float) -> None:
        """
        Transition task from ACTIVE to COMPLETED state and release resource.
        
        O(1) operation using set add/remove.
        
        Args:
            task_id: ID of task to complete
            end_time: Completion time
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        
        # Update resource utilization before releasing
        if task.assigned_resource_id:
            resource_id = task.assigned_resource_id
            if resource_id in self.resource_last_start:
                busy_duration = end_time - self.resource_last_start[resource_id]
                self.resource_busy_time[resource_id] = self.resource_busy_time.get(resource_id, 0.0) + busy_duration
            
            # Release resource
            resource = self.resources.get(resource_id)
            if resource:
                resource.release_task(task_id)
                # Update last_start if resource has more tasks
                if not resource.is_available():
                    self.resource_last_start[resource_id] = end_time
        
        # State transition: ACTIVE -> COMPLETED
        self.active_tasks.discard(task_id)
        self.completed_tasks.add(task_id)
        
        # Update task entity
        task.complete(end_time)
        
        case = self.cases.get(task.case_id)
        completed_activity = CompletedActivity(
            activity_name=task.activity_name,
            resource_id=resource_id,
            start_time=task.actual_start_time,
            complete_time=task.actual_end_time
        )
        case.add_completed_activity(completed_activity)

    
    def cancel_task(self, task_id: str, cancel_time: float) -> None:
        """
        Cancel a task (e.g. dropped after exceeding max deferrals).
        
        Removes the task from queued/active tracking and marks it CANCELLED.
        Furthermore cancels the entire case to which the task belongs, 
        as we assume that if a task is cancelled due to max deferrals, the whole case is dropped.
        
        Args:
            task_id: ID of task to cancel
            cancel_time: Simulation time of cancellation
        """
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        
        # Release resource if one was assigned
        if task.assigned_resource_id:
            tid = task.assigned_resource_id
            if tid and tid != DUMMY_RESOURCE_ID:
                resource = self.resources.get(task.assigned_resource_id)
                if resource:
                    resource.release_task(task_id)
        
        # Remove from whichever state set it's in
        self.queued_tasks.discard(task_id)
        self.active_tasks.discard(task_id)
        self.cancelled_tasks.add(task_id)
        
        # Update task entity
        task.cancel(cancel_time)
        
        # Cancel task
        self.cancel_case(task.case_id, cancel_time)
    
    def complete_case(self, case_id: str, completion_time: float) -> None:
        """
        Transition case from ACTIVE to COMPLETED state.
        
        O(1) operation using set add/remove.
        
        Args:
            case_id: ID of case to complete
            completion_time: Completion time
        """
        if case_id not in self.cases:
            raise ValueError(f"Case {case_id} not found")
        
        # State transition: ACTIVE -> COMPLETED
        self.active_cases.discard(case_id)
        self.completed_cases.add(case_id)
        
        # Update case entity
        case = self.cases[case_id]
        case.complete(completion_time)
        
    def cancel_case(self, case_id: str, cancel_time: float) -> None:
        """
        Cancel a case (e.g. dropped after exceeding max deferrals).
        
        Removes the case from active tracking and marks it CANCELLED.
        
        Args:
            case_id: ID of case to cancel
            cancel_time: Simulation time of cancellation
        """
        if case_id not in self.cases:
            raise ValueError(f"Case {case_id} not found")
        
        # Remove from active tracking
        self.active_cases.discard(case_id)
        self.cancelled_cases.add(case_id)
                
        # Update case entity
        case = self.cases[case_id]
        case.cancel(cancel_time)
    
    def get_resource_utilization(self, current_time: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate resource utilization based on tracked busy time.
        
        Args:
            current_time: Current simulation time (uses self.current_time if None)
        
        Returns:
            Dictionary of resource_id -> utilization (0-1)
            Utilization = total_busy_time / elapsed_time
        """
        if current_time is None:
            current_time = self.current_time
        
        if current_time <= 0:
            return {rid: 0.0 for rid in self.resources.keys()}
        
        utilization = {}
        for resource_id in self.resources.keys():
            busy_time = self.resource_busy_time.get(resource_id, 0.0)
            
            # Add current active time if resource is busy
            resource = self.resources.get(resource_id)
            if resource and not resource.is_available():
                if resource_id in self.resource_last_start:
                    busy_time += current_time - self.resource_last_start[resource_id]
            
            utilization[resource_id] = busy_time / current_time
        
        return utilization
    
    def get_statistics(self) -> dict:
        """
        Get summary statistics of current state.
        
        All operations are O(1) thanks to set-based tracking.
        
        Returns:
            Dictionary with counts and metrics
        """
        tasks_per_activity = {}
        for task in self.tasks.values():
            activity_name = task.activity_name
            tasks_per_activity[activity_name] = tasks_per_activity.get(activity_name, 0) + 1
            
        cancelled_tasks_per_activity = {}
        for task_id in self.cancelled_tasks:
            task = self.tasks.get(task_id)
            if task:
                activity_name = task.activity_name
                cancelled_tasks_per_activity[activity_name] = cancelled_tasks_per_activity.get(activity_name, 0) + 1
        
        for activity, count in tasks_per_activity.items():
            cancelled_count = cancelled_tasks_per_activity.get(activity, 0)
            fcancelled_count = cancelled_count / count if count > 0 else 0.
            cancelled_tasks_per_activity[activity] = f"{cancelled_count}/{count} ({fcancelled_count:.2%})"
        return {
            'current_time': self.current_time,
            'total_resources': len(self.resources),
            'available_resources': len(self.get_available_resources()),
            'total_cases': len(self.cases),
            'active_cases': len(self.active_cases),
            'completed_cases': len(self.completed_cases),
            'cancelled_cases': len(self.cancelled_cases),
            'total_tasks': len(self.tasks),
            'queued_tasks': len(self.queued_tasks),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'cancelled_tasks': len(self.cancelled_tasks),
            'cancelled_tasks_per_activity': cancelled_tasks_per_activity,
        }
    
    def __repr__(self) -> str:
        return (
            f"SimulationState(time={self.current_time:.2f}, "
            f"cases={len(self.active_cases)}/{len(self.cases)}, "
            f"tasks: {len(self.queued_tasks)} queued, "
            f"{len(self.active_tasks)} active, "
            f"{len(self.completed_tasks)} completed, "
            f"{len(self.cancelled_tasks)} cancelled)"
        )
