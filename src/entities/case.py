"""Case entity representing a process instance in the simulation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class CaseStatus(Enum):
    """Status of a case during simulation."""
    WAITING = "waiting"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class CompletedActivity:
    """Record of a completed activity within a case."""
    activity_name: str
    resource_id: str
    start_time: float
    complete_time: float
    
    @property
    def duration(self) -> float:
        """Duration of the activity in simulation time units."""
        return self.complete_time - self.start_time


@dataclass
class Case:
    """
    Represents a process instance (case) in the simulation.
    
    Attributes:
        id: Unique identifier for the case
        case_type: Type of case (e.g., 'loan_application')
        status: Current status of the case
        attributes: Case-specific attributes (e.g., LoanGoal, ApplicationType, RequestedAmount)
        arrival_time: Simulation time when case arrived (seconds)
        current_activity_index: Index of current activity in the process sequence
        trace: List of completed activities
        pending_activity_types: Activity types that are pending for this case
        initial_activity: Optional name of the first activity (for probabilistic model)
        priority: Optional priority for scheduling (higher = more important)
        deadline: Optional deadline for completion (simulation time)
    """
    id: str
    case_type: str
    status: CaseStatus = CaseStatus.WAITING
    attributes: Dict[str, Any] = field(default_factory=dict)
    arrival_time: float = 0.0
    current_activity_index: int = 0
    trace: List[CompletedActivity] = field(default_factory=list)
    pending_activity_types: List[str] = field(default_factory=list)
    initial_activity: Optional[str] = None  # For probabilistic model
    priority: int = 0
    deadline: Optional[float] = None
    completion_time: Optional[float] = None
    
    # Performance caches for probabilistic model (prefix with _ to indicate internal)
    _execution_history: Dict[str, int] = field(default_factory=dict)  # Cached history counts
    _enabled_activities: Optional[set] = None  # Cached enabled activities set
    
    def add_completed_activity(self, activity: CompletedActivity) -> None:
        """Add a completed activity to the trace and update caches."""
        self.trace.append(activity)
        self.current_activity_index += 1
        
        # Update execution history cache for probabilistic model
        activity_name = activity.activity_name
        self._execution_history[activity_name] = self._execution_history.get(activity_name, 0) + 1
        
        # NOTE: Do NOT discard activities from _enabled_activities
        # Activities should be repeatable in realistic processes (loops, rework, etc.)
        # Completion is determined by max activity count or explicit end conditions
    
    def get_duration(self) -> Optional[float]:
        """Get total duration from arrival to completion."""
        if self.completion_time is not None:
            return self.completion_time - self.arrival_time
        return None
    
    def get_waiting_time(self) -> float:
        """Calculate total waiting time (time between activities)."""
        if len(self.trace) < 2:
            return 0.0
        
        waiting_time = 0.0
        for i in range(len(self.trace) - 1):
            gap = self.trace[i + 1].start_time - self.trace[i].complete_time
            waiting_time += max(0, gap)
        
        # Add initial waiting time
        if self.trace:
            waiting_time += self.trace[0].start_time - self.arrival_time
        
        return waiting_time
    
    def complete(self, completion_time: float) -> None:
        """Mark case as completed."""
        self.status = CaseStatus.COMPLETED
        self.completion_time = completion_time
        
    def cancel(self, cancel_time: float) -> None:
        """Mark case as cancelled."""
        self.status = CaseStatus.CANCELLED
        self.completion_time = cancel_time
    
    def get_initial_activity(self) -> Optional[str]:
        """Get the initial activity for this case (for probabilistic model)."""
        return self.initial_activity
    
    def __repr__(self) -> str:
        return f"Case(id={self.id}, type={self.case_type}, status={self.status.value}, activities={len(self.trace)})"
