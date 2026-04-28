"""Write simulation output as event logs."""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime


class EventLogWriter:
    """
    Write simulation events to event log format.
    
    Supports CSV and can be extended to XES.
    """
    
    def __init__(self, output_path: Optional[Path] = None):
        """
        Initialize event log writer.
        
        Args:
            output_path: Path where log will be saved (None = buffered only)
        """
        self.output_path = output_path
        
        # Performance optimization: Batched event logging
        self.batch_size = 1000  # Flush buffer every N events
        self._event_buffer: List[tuple] = []  # Lightweight tuple buffer
        self.events: List[Dict[str, Any]] = []  # Pre-allocate for expected size
        
        self.metadata: Dict[str, Any] = {}
    
    def set_metadata(self, **kwargs) -> None:
        """
        Set metadata for the simulation run.
        
        Args:
            **kwargs: Metadata key-value pairs (e.g., scheduler='Random', seed=42)
        """
        self.metadata.update(kwargs)
    
    def log_task_start(
        self,
        case_id: str,
        task_id: str,
        activity_name: str,
        resource_id: str,
        timestamp: float,
        sim_datetime: Optional[datetime] = None,
        **additional_attributes
    ) -> None:
        """
        Log task start event (batched for performance).
        
        Args:
            case_id: Case ID
            task_id: Task ID
            activity_name: Activity name
            resource_id: Resource ID
            timestamp: Simulation timestamp in hours
            sim_datetime: Optional simulation datetime for the event
            **additional_attributes: Additional event attributes
        """
        # Store as lightweight tuple in buffer (faster than dict creation)
        self._event_buffer.append(
            ('start', case_id, task_id, activity_name, resource_id, timestamp, sim_datetime, additional_attributes)
        )
        
        # Flush buffer when batch size reached
        if len(self._event_buffer) >= self.batch_size:
            self._flush_buffer()
    
    def log_task_queued(
        self,
        case_id: str,
        task_id: str,
        activity_name: str,
        resource_id: str,
        timestamp: float,
        sim_datetime: Optional[datetime] = None,
        **additional_attributes
    ) -> None:
        """
        Log task queued event (when assigned to resource queue).
        
        Args:
            case_id: Case ID
            task_id: Task ID
            activity_name: Activity name
            resource_id: Resource ID assigned to
            timestamp: Simulation timestamp in hours
            sim_datetime: Optional simulation datetime for the event
            **additional_attributes: Additional event attributes
        """
        # Store as lightweight tuple in buffer
        self._event_buffer.append(
            ('queued', case_id, task_id, activity_name, resource_id, timestamp, sim_datetime, additional_attributes)
        )
        
        # Flush buffer when batch size reached
        if len(self._event_buffer) >= self.batch_size:
            self._flush_buffer()
    
    def log_task_complete(
        self,
        case_id: str,
        task_id: str,
        activity_name: str,
        resource_id: str,
        timestamp: str,
        sim_datetime: Optional[datetime] = None,
        **additional_attributes
    ) -> None:
        """
        Log task completion event (batched for performance).
        
        Args:
            case_id: Case ID
            task_id: Task ID
            activity_name: Activity name
            resource_id: Resource ID
            timestamp: Simulation timestamp in hours
            sim_datetime: Optional datetime for the event
            **additional_attributes: Additional event attributes
        """
        # Store as lightweight tuple in buffer (faster than dict creation)
        self._event_buffer.append(
            ('complete', case_id, task_id, activity_name, resource_id, timestamp, sim_datetime, additional_attributes)
        )
        
        # Flush buffer when batch size reached
        if len(self._event_buffer) >= self.batch_size:
            self._flush_buffer()
    
    def log_case_arrival(
        self,
        case_id: str,
        timestamp: float,
        sim_datetime: Optional[datetime] = None,
        **additional_attributes
    ) -> None:
        """
        Log case arrival (optional - for analysis).
        
        Args:
            case_id: Case ID
            timestamp: Arrival timestamp
            sim_datetime: Optional datetime for the event
            **additional_attributes: Additional attributes
        """
        event = {
            'case_id': case_id,
            'task_id': None,
            'activity': 'Case_Arrival',
            'resource': None,
            'lifecycle': 'start',
            'timestamp': timestamp,
            'sim_datetime': sim_datetime,
            **additional_attributes
        }
        self.events.append(event)
    
    def _flush_buffer(self) -> None:
        """
        Flush buffered events to main events list.
        Converts lightweight tuples to dicts in batch (more efficient).
        """
        for item in self._event_buffer:
            lifecycle, case_id, task_id, activity_name, resource_id, timestamp, sim_datetime, additional = item
            event = {
                'case_id': case_id,
                'task_id': task_id,
                'activity': activity_name,
                'resource': resource_id,
                'lifecycle': lifecycle,
                'timestamp': timestamp,
                'sim_datetime': sim_datetime,
                **additional
            }
            self.events.append(event)
        self._event_buffer.clear()
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert logged events to DataFrame.
        
        Returns:
            DataFrame with all logged events
        """
        # Flush any remaining buffered events
        if self._event_buffer:
            self._flush_buffer()
        
        if not self.events:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.events)
        
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        return df
    
    def save_csv(self, filepath: Optional[Path] = None) -> None:
        """
        Save event log to CSV file.
        
        Args:
            filepath: Output path (uses self.output_path if None)
        """
        output = filepath or self.output_path
        
        if output is None:
            raise ValueError("No output path specified")
        
        df = self.to_dataframe()
        
        # Create parent directory if needed
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output, index=False)
    
    def save_with_metadata(self, filepath: Optional[Path] = None) -> None:
        """
        Save event log with metadata file.
        
        Args:
            filepath: Output path for event log
        """
        self.save_csv(filepath)
        
        # Save metadata to adjacent file
        if filepath or self.output_path:
            log_path = Path(filepath or self.output_path)
            metadata_path = log_path.parent / f"{log_path.stem}_metadata.json"
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
    
    def finalize(self) -> None:
        """
        Finalize logging and save if output path is set.
        """
        if self.output_path:
            self.save_with_metadata()
    
    def clear(self) -> None:
        """Clear all logged events and buffer."""
        self.events.clear()
        self._event_buffer.clear()
    
    def __len__(self) -> int:
        """Number of logged events."""
        return len(self.events)
    
    def __repr__(self) -> str:
        return f"EventLogWriter(events={len(self.events)}, output={self.output_path})"
