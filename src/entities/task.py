"""Task entity representing an activity instance in the simulation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any


class TaskStatus(Enum):
    """Status of a task during simulation."""
    QUEUED = "queued"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """Type of task execution."""
    STANDARD = "standard"           # Normal single-resource task
    MENTORING = "mentoring"         # Dual-resource mentoring task


@dataclass
class Task:
    """
    Represents a task (activity instance) in the simulation.
    
    Attributes:
        id: Unique identifier for the task
        case_id: ID of the parent case
        activity_name: Name of the activity to perform
        status: Current status of the task
        assigned_resource_id: ID of assigned resource (None if unassigned)
        estimated_duration: Estimated duration based on experience in the scheduler(seconds)
        sampled_duration: Sample (in the execution engine) duration for this execution (seconds, set on assignment)
        actual_start_time: Actual start time in simulation clock (seconds)
        actual_end_time: Actual completion time in simulation clock (seconds)
        context: Additional context attributes (for experience lookup)
        creation_time: Simulation time when task was created (seconds)
    """
    id: str
    case_id: str
    activity_name: str
    status: TaskStatus = TaskStatus.QUEUED
    assigned_resource_id: Optional[str] = None
    estimated_duration: Optional[float] = 0.0
    sampled_duration: float = 0.0
    actual_start_time: Optional[float] = None
    actual_end_time: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    creation_time: float = 0.0
    queued_time: Optional[float] = None
    
    # Batch scheduling support
    deadline: Optional[float] = None  # Absolute simulation time deadline
    scheduled_start_time: Optional[float] = None  # Set by CP optimizer. TODO: evtl unused
    required_capability_level: float = 0.0  # Minimum capability level required
    arrival_time: float = 0.0  # Task arrival time (for waiting time calculation)
    defer_count: int = 0  # Number of times task was deferred (re-queued for next day)
    
    # Mentoring support
    task_type: 'TaskType' = None  # Will be set in __post_init__
    mentor_resource_id: Optional[str] = None
    mentee_resource_id: Optional[str] = None
    duration_multiplier: float = 1.3  # e.g., 1.5 for mentoring
    learning_opportunity: bool = False  # True if capability will be gained
    target_capability: Optional[str] = None  # Capability being learned
    mentoring_against_bottleneck: bool = False  # True if this mentoring task is specifically for bottleneck relief
    is_emergency_mentoring: bool = False  # Generated for bottleneck

    # Capability bootstrap support (for uncovered activities)
    bootstrap_assignment: bool = False
    bootstrap_activity: Optional[str] = None
    bootstrap_resource_id: Optional[str] = None
    bootstrap_onboarding_seconds: int = 0
    bootstrap_onboarding_applied: bool = False
    
    def __post_init__(self):
        """Initialize task_type if not set."""
        if self.task_type is None:
            self.task_type = TaskType.STANDARD
    
    def assign_to_resource(self, resource_id: str, sampled_duration: float) -> None:
        """Assign task to a resource with estimated duration."""
        self.assigned_resource_id = resource_id
        self.sampled_duration = sampled_duration
        self.status = TaskStatus.ASSIGNED
    
    def start(self, start_time: float) -> None:
        """Start the task execution."""
        self.actual_start_time = start_time
        self.status = TaskStatus.IN_PROGRESS
    
    def complete(self, end_time: float) -> None:
        """Complete the task execution."""
        self.actual_end_time = end_time
        self.status = TaskStatus.COMPLETED
    
    def cancel(self, cancel_time: float) -> None:
        """Cancel the task (e.g. dropped after exceeding max deferrals)."""
        self.actual_end_time = cancel_time
        self.status = TaskStatus.CANCELLED
        
    def reset_for_rescheduling(self) -> None:
        """Reset task state for rescheduling (e.g. after deferral)."""
        self.assigned_resource_id = None
        self.scheduled_start_time = None # TODO: evtl. ganzes property löschen
        self.estimated_duration = None
        self.task_type = TaskType.STANDARD  # Reset to standard for rescheduling
        self.learning_opportunity = False
        self.mentee_resource_id = None
        self.mentor_resource_id = None
        self.bootstrap_assignment = False
        self.bootstrap_activity = None
        self.bootstrap_resource_id = None
        self.bootstrap_onboarding_seconds = 0
        self.bootstrap_onboarding_applied = False
        self.defer_count += 1
        self.status = TaskStatus.QUEUED
    
    def get_actual_duration(self) -> Optional[float]:
        """Get actual duration if task is completed."""
        if self.actual_start_time is not None and self.actual_end_time is not None:
            return self.actual_end_time - self.actual_start_time
        return None
    
    def get_context_attribute(self, key: str) -> Any:
        """Get a context attribute value."""
        return self.context.get(key)
    
    def get_waiting_time(self) -> Optional[float]:
        """Get time between creation and start."""
        if self.actual_start_time is not None:
            return self.actual_start_time - self.creation_time
        return None
    
    def get_queue_waiting_time(self) -> Optional[float]:
        """Get time between being queued and actual start (time waiting in resource queue)."""
        if self.queued_time is not None and self.actual_start_time is not None:
            return self.actual_start_time - self.queued_time
        return None
    
    def is_mentoring_task(self) -> bool:
        """Check if this is a mentoring task."""
        return self.task_type == TaskType.MENTORING
    
    def requires_dual_resources(self) -> bool:
        """Check if task requires multiple resources."""
        return self.task_type in [TaskType.MENTORING, TaskType.COLLABORATIVE]
    
    def __repr__(self) -> str:
        return f"Task(id={self.id}, activity={self.activity_name}, status={self.status.value}, resource={self.assigned_resource_id})"
    
    def __lt__(self, other: 'Task') -> bool:
        """Compare tasks by priority (for priority queue)."""
        # Higher priority comes first (inverse comparison)
        return self.creation_time > other.creation_time
