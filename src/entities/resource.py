"""Resource entity representing a human worker in the simulation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
from ..experience.store import ExperienceStore

class ResourceStatus(Enum):
    """Status of a resource during simulation."""
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"


@dataclass
class Resource:
    """
    Represents a human resource (worker) in the simulation.
    
    Attributes:
        id: Unique identifier for the resource
        name: Human-readable name of the resource
        status: Current availability status
        experience_profile_id: Reference to experience data in ExperienceStore
        current_assigned_tasks: List of task IDs currently assigned
        working_hours: Optional dict defining work schedule (e.g., {'start': 8, 'end': 17})
        experience_store: Reference to ExperienceStore for capability queries
    """
    id: str
    name: str
    status: ResourceStatus = ResourceStatus.AVAILABLE
    experience_profile_id: str = field(default="")
    current_assigned_tasks: List[str] = field(default_factory=list)
    working_hours: Optional[dict] = None
    experience_store: Optional[ExperienceStore] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Initialize experience_profile_id to resource id if not set."""
        if not self.experience_profile_id:
            self.experience_profile_id = self.id
    
    def is_available(self) -> bool:
        """Check if resource is available for task assignment."""
        return self.status == ResourceStatus.AVAILABLE
    
    def can_perform(self, activity: str, required_level: float = 0.0) -> bool:
        """
        Check if resource has capability to perform an activity at the required level.
        
        Args:
            activity: Name of the activity
            required_level: Minimum experience level required (0-100)
            
        Returns:
            True if resource has the capability at or above the required level
        """
        if self.experience_store is None:
            return True
        
        return self.experience_store.is_capable(
            resource_id=self.id,
            activity_name=activity,
            required_level=required_level
        )
    
    def get_experience_level(self, activity: str) -> float:
        """Get experience level for a specific activity.
        
        Delegates to ExperienceStore if available.
        
        Args:
            activity: Name of the activity
            
        Returns:
            Experience level (0-100), or 0.0 if no profile found
        """
        if self.experience_store is None:
            return 0.0
        
        return self.experience_store.get_capability_level(
            resource_id=self.id,
            activity_name=activity
        )
    
    def assign_task(self, task_id: str) -> None:
        """Assign a task to this resource."""
        self.current_assigned_tasks.append(task_id)
        self.status = ResourceStatus.BUSY
    
    def release_task(self, task_id: str) -> None:
        """Release a task from this resource."""
        if task_id in self.current_assigned_tasks:
            self.current_assigned_tasks.remove(task_id)
        if not self.current_assigned_tasks:
            self.status = ResourceStatus.AVAILABLE
    
    def __repr__(self) -> str:
        return f"Resource(id={self.id}, name={self.name}, status={self.status.value})"
