"""
Experience Level Tracker for monitoring and visualizing experience curve development.

This module tracks how resource experience levels evolve over time during simulation,
enabling detailed analysis and visualization in Streamlit dashboards.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
import json


@dataclass
class ExperienceLevelSnapshot:
    """
    Snapshot of experience level at a specific point in time.
    
    Attributes:
        simulation_time: Simulation clock time in hours
        sim_datetime: Timestamp for the snapshot
        resource_id: ID of the resource
        activity_name: Name of the activity
        experience_level: Experience level (0-100)
        repetition_count: Number of times activity has been performed
        mean_duration: Current mean duration (hours)
        context: Context attributes (optional)
    """
    simulation_time: float
    sim_datetime: datetime
    resource_id: str
    activity_name: str
    experience_level: float
    repetition_count: int
    mean_duration: float
    context: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'simulation_time': self.simulation_time,
            'sim_datetime': self.sim_datetime,
            'resource_id': self.resource_id,
            'activity_name': self.activity_name,
            'experience_level': self.experience_level,
            'repetition_count': self.repetition_count,
            'mean_duration': self.mean_duration,
            'context': self.context
        }


class ExperienceLevelTracker:
    """
    Tracks experience level evolution during simulation.
    
    Collects snapshots of experience levels over time, enabling:
    - Learning curve visualization
    - Performance improvement analysis
    - Comparison across resources and activities
    - Longitudinal tracking for dashboards
    """
    
    def __init__(self):
        """Initialize experience level tracker."""
        self._snapshots: List[ExperienceLevelSnapshot] = []
        self._enabled = True
    
    def enable(self) -> None:
        """Enable tracking."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable tracking (for performance)."""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if tracking is enabled."""
        return self._enabled
    
    def record_snapshot(
        self,
        simulation_time: float,
        sim_datetime: datetime,
        resource_id: str,
        activity_name: str,
        experience_level: float,
        repetition_count: int,
        mean_duration: float,
        context: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a snapshot of experience level.
        
        Args:
            simulation_time: Current simulation time in hours
            sim_datetime: Current datetime
            resource_id: Resource ID
            activity_name: Activity name
            experience_level: Current experience level (0-100)
            repetition_count: Number of repetitions
            mean_duration: Current mean duration
            context: Optional context attributes
        """
        if not self._enabled:
            return
        
        if context is None:
            context = {}
        
        snapshot = ExperienceLevelSnapshot(
            simulation_time=simulation_time,
            sim_datetime=sim_datetime,
            resource_id=resource_id,
            activity_name=activity_name,
            experience_level=experience_level,
            repetition_count=repetition_count,
            mean_duration=mean_duration,
            context=context
        )
        
        self._snapshots.append(snapshot)
    
    def get_snapshots(self) -> List[ExperienceLevelSnapshot]:
        """Get all recorded snapshots."""
        return self._snapshots.copy()
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert snapshots to pandas DataFrame for analysis.
        
        Returns:
            DataFrame with columns: simulation_time, sim_datetime, resource_id, activity_name,
                                   experience_level, repetition_count, mean_duration
        """
        if not self._snapshots:
            return pd.DataFrame(columns=[
                'simulation_time', 'sim_datetime', 'resource_id', 'activity_name',
                'experience_level', 'repetition_count', 'mean_duration'
            ])
        
        data = [snapshot.to_dict() for snapshot in self._snapshots]
        df = pd.DataFrame(data)
        
        # Flatten context if present
        if 'context' in df.columns and len(df) > 0:
            # Extract context fields into separate columns
            context_df = pd.json_normalize(df['context'])
            df = pd.concat([df.drop('context', axis=1), context_df], axis=1)
        
        return df
    
    def get_resource_curve(
        self,
        resource_id: str,
        activity_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get experience curve for a specific resource.
        
        Args:
            resource_id: Resource ID
            activity_name: Optional activity name filter
            
        Returns:
            DataFrame with snapshots for the resource
        """
        df = self.to_dataframe()
        
        if df.empty:
            return df
        
        # Filter by resource
        df = df[df['resource_id'] == resource_id]
        
        # Optionally filter by activity
        if activity_name is not None:
            df = df[df['activity_name'] == activity_name]
        
        return df.sort_values('simulation_time')
    
    def get_activity_curves(self, activity_name: str) -> pd.DataFrame:
        """
        Get experience curves for all resources performing an activity.
        
        Args:
            activity_name: Activity name
            
        Returns:
            DataFrame with snapshots for all resources on this activity
        """
        df = self.to_dataframe()
        
        if df.empty:
            return df
        
        df = df[df['activity_name'] == activity_name]
        return df.sort_values(['resource_id', 'simulation_time'])
    
    def get_summary_statistics(self) -> Dict[str, pd.DataFrame]:
        """
        Get summary statistics for experience development.
        
        Returns:
            Dictionary with summary DataFrames:
            - by_resource: Average experience level per resource
            - by_activity: Average experience level per activity
            - by_resource_activity: Matrix of resource x activity levels
        """
        df = self.to_dataframe()
        
        if df.empty:
            return {
                'by_resource': pd.DataFrame(),
                'by_activity': pd.DataFrame(),
                'by_resource_activity': pd.DataFrame()
            }
        
        # Get latest snapshot for each resource-activity pair
        latest = df.sort_values('simulation_time').groupby(
            ['resource_id', 'activity_name']
        ).last().reset_index()
        
        # Summary by resource
        by_resource = latest.groupby('resource_id').agg({
            'experience_level': ['mean', 'min', 'max', 'count'],
            'repetition_count': 'sum'
        }).round(2)
        
        # Summary by activity
        by_activity = latest.groupby('activity_name').agg({
            'experience_level': ['mean', 'min', 'max', 'count'],
            'repetition_count': 'sum'
        }).round(2)
        
        # Matrix: resource x activity
        by_resource_activity = latest.pivot_table(
            index='resource_id',
            columns='activity_name',
            values='experience_level',
            aggfunc='mean'
        ).round(2)
        
        return {
            'by_resource': by_resource,
            'by_activity': by_activity,
            'by_resource_activity': by_resource_activity
        }
    
    def save_to_csv(self, filepath: str) -> None:
        """
        Save snapshots to CSV file.
        
        Args:
            filepath: Path to output CSV file
        """
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
    
    def save_to_json(self, filepath: str) -> None:
        """
        Save snapshots to JSON file.
        
        Args:
            filepath: Path to output JSON file
        """
        data = [snapshot.to_dict() for snapshot in self._snapshots]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_csv(self, filepath: str) -> None:
        """
        Load snapshots from CSV file.
        
        Args:
            filepath: Path to input CSV file
        """
        df = pd.read_csv(filepath)
        
        self._snapshots = []
        for _, row in df.iterrows():
            snapshot = ExperienceLevelSnapshot(
                simulation_time=row['simulation_time'],
                sim_datetime=pd.to_datetime(row['sim_datetime']),
                resource_id=row['resource_id'],
                activity_name=row['activity_name'],
                experience_level=row['experience_level'],
                repetition_count=row['repetition_count'],
                mean_duration=row['mean_duration'],
                context={}  # Context would need to be reconstructed from columns
            )
            self._snapshots.append(snapshot)
    
    def clear(self) -> None:
        """Clear all recorded snapshots."""
        self._snapshots.clear()
    
    def get_count(self) -> int:
        """Get number of recorded snapshots."""
        return len(self._snapshots)
    
    def __repr__(self) -> str:
        return f"ExperienceLevelTracker(snapshots={len(self._snapshots)}, enabled={self._enabled})"
