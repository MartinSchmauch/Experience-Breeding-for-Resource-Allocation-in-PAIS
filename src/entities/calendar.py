"""
Calendar and working hours management for resources.

Defines working schedules, absences (vacation/sick days), and resource calendars
for realistic time modeling in business process simulation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum
from ..utils.time_utils import SimulationTimeConverter

class AbsenceType(Enum):
    """Types of resource absences."""
    VACATION = "vacation"
    SICK_LEAVE = "sick_leave"
    HOLIDAY = "holiday"
    OTHER = "other"


@dataclass
class WorkingSchedule:
    """
    Defines standard working hours for a resource.
    
    Attributes:
        weekday_hours: Dict mapping weekday (0=Monday, 6=Sunday) to (start_hour, end_hour)
                      Default is Mon-Fri 9am-5pm
        timezone_offset: Hours offset from UTC (not currently used, reserved for future)
    """
    weekday_hours: Dict[int, tuple[float, float]] = field(default_factory=lambda: {
        0: (9.0, 17.0),   # Monday
        1: (9.0, 17.0),   # Tuesday
        2: (9.0, 17.0),   # Wednesday
        3: (9.0, 17.0),   # Thursday
        4: (9.0, 17.0),   # Friday
        # 5 and 6 (weekend) not in dict = not working
    })
    timezone_offset: float = 0.0
    
    def is_working_day(self, weekday: int) -> bool:
        """Check if a weekday is a working day.
        
        Args:
            weekday: Day of week (0=Monday, 6=Sunday)
            
        Returns:
            True if working day, False otherwise
        """
        return weekday in self.weekday_hours
    
    def get_working_hours(self, weekday: int) -> Optional[tuple[float, float]]:
        """Get working hours for a specific weekday.
        
        Args:
            weekday: Day of week (0=Monday, 6=Sunday)
            
        Returns:
            Tuple of (start_hour, end_hour) or None if not a working day
        """
        return self.weekday_hours.get(weekday)
    
    def is_within_working_hours(self, weekday: int, hour: float) -> bool:
        """Check if a specific time is within working hours.
        
        Args:
            weekday: Day of week (0=Monday, 6=Sunday)
            hour: Hour of day (0.0-24.0, can be fractional)
            
        Returns:
            True if within working hours, False otherwise
        """
        hours = self.get_working_hours(weekday)
        if hours is None:
            return False
        start_hour, end_hour = hours
        return start_hour <= hour < end_hour


@dataclass
class Absence:
    """
    Represents a period when a resource is unavailable.
    
    Attributes:
        start_date: Start datetime of absence
        end_date: End datetime of absence (exclusive)
        absence_type: Type of absence (vacation, sick leave, etc.)
        description: Optional description or reason
    """
    start_date: datetime
    end_date: datetime
    absence_type: AbsenceType
    description: Optional[str] = None
    
    def overlaps_with(self, check_date: datetime) -> bool:
        """Check if a date falls within this absence period.
        
        Args:
            check_date: Datetime to check
            
        Returns:
            True if date is within absence period
        """
        return self.start_date <= check_date < self.end_date
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'absence_type': self.absence_type.value,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Absence':
        """Create from dictionary."""
        return cls(
            start_date=datetime.fromisoformat(data['start_date']),
            end_date=datetime.fromisoformat(data['end_date']),
            absence_type=AbsenceType(data['absence_type']),
            description=data.get('description')
        )


@dataclass
class ResourceCalendar:
    """
    Manages working schedule and absences for a resource.
    
    Combines standard working hours with specific absence periods
    to determine resource availability at any point in time.
    """
    resource_id: str
    schedule: WorkingSchedule
    absences: List[Absence] = field(default_factory=list)
    _cache: Dict[tuple, float] = field(default_factory=dict, init=False, repr=False)
    
    def clear_cache(self) -> None:
        """Clear the availability cache."""
        self._cache.clear()
    
    def is_available_at(self, dt: datetime, neglect_sick_leave: bool = False) -> bool:
        """Check if resource is available at a specific datetime.
        
        Considers both working schedule and absences.
        
        Args:
            dt: Datetime to check
            
        Returns:
            True if resource is available (working hours + not absent)
        """
        # Check if on absence
        for absence in self.absences:
            if absence.overlaps_with(dt):
                # If neglect_sick_leave is True and the absence is of type 'sick_leave', skip it
                if neglect_sick_leave and absence.absence_type == AbsenceType.SICK_LEAVE:
                    continue
                return False
        
        # Check working schedule
        weekday = dt.weekday()  # 0=Monday, 6=Sunday
        hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        
        return self.schedule.is_within_working_hours(weekday, hour)
    
    def calculate_overtime_duration(self, start_dt: datetime, end_dt: datetime) -> float:
        """Calculate overtime duration between two datetimes.
        
        Overtime is time spent outside working hours.
        
        Args:
            start_dt: Start datetime
            end_dt: End datetime
        Returns:
            Overtime duration in hours
        """
        total_overtime = 0.0
        current_dt = start_dt
        
        while current_dt < end_dt:
            if not self.is_available_at(current_dt):
                total_overtime += 1.0  # Count 1 hour of overtime
            current_dt += timedelta(hours=1)
        
        return total_overtime
    
    def get_next_available_time(self, from_dt: datetime, max_lookahead_days: int = 365) -> Optional[datetime]:
        """Find the next time when resource becomes available.
        
        Args:
            from_dt: Starting datetime to search from
            max_lookahead_days: Maximum days to search ahead
            
        Returns:
            Next available datetime, or None if not found within lookahead
        """
        current_dt = from_dt
        end_dt = from_dt + timedelta(days=max_lookahead_days)
        
        # Search in 1-hour increments (could be optimized to jump to next working day)
        while current_dt < end_dt:
            if self.is_available_at(current_dt):
                return current_dt
            
            # Check if currently in absence - skip to end of absence
            in_absence = False
            for absence in self.absences:
                if absence.overlaps_with(current_dt):
                    current_dt = absence.end_date
                    in_absence = True
                    break
            
            if not in_absence:
                # Not in absence, check next hour or jump to next working day
                weekday = current_dt.weekday()
                if not self.schedule.is_working_day(weekday):
                    # Jump to next Monday
                    days_until_monday = (7 - weekday) % 7
                    if days_until_monday == 0:
                        days_until_monday = 1  # It's Sunday, go to Monday
                    current_dt = current_dt + timedelta(days=days_until_monday)
                    # Set to start of working hours
                    hours = self.schedule.get_working_hours(current_dt.weekday())
                    if hours:
                        current_dt = self._align_to_hour_float(current_dt, float(hours[0]))
                else:
                    # It's a working day, check if before/during/after working hours
                    hours = self.schedule.get_working_hours(weekday)
                    if hours:
                        current_hour = current_dt.hour + current_dt.minute / 60.0
                        start_hour, end_hour = hours
                        
                        if current_hour < start_hour:
                            # Before working hours - jump to start
                            current_dt = self._align_to_hour_float(current_dt, float(start_hour))
                        elif current_hour >= end_hour:
                            # After working hours - jump to next working day
                            current_dt = current_dt + timedelta(days=1)
                            current_dt = self._align_to_hour_float(current_dt, float(start_hour))
                        else:
                            # During working hours - increment by 1 hour
                            current_dt = current_dt + timedelta(hours=1)
                    else:
                        # No working hours found (shouldn't happen)
                        current_dt = current_dt + timedelta(hours=1)
        
        return None  # No available time found within lookahead period

    @staticmethod
    def _align_to_hour_float(dt: datetime, hour_float: float) -> datetime:
        """Align datetime to a possibly fractional hour (e.g., 7.5 -> 07:30:00)."""
        base_hour = int(hour_float)
        minute_float = (hour_float - base_hour) * 60.0
        minute = int(minute_float)
        second = int(round((minute_float - minute) * 60.0))
        if second >= 60:
            second = 0
            minute += 1
        if minute >= 60:
            minute = 0
            base_hour += 1
        base_hour = max(0, min(base_hour, 23))
        return dt.replace(hour=base_hour, minute=minute, second=second, microsecond=0)
    
    def get_available_slots(
        self, 
        start_sim: int | float, 
        end_sim: int | float,
        time_converter: SimulationTimeConverter = None,
        neglect_sick_leave: bool = False,
        reference_dt: datetime = None
    ) -> list[tuple[int, int]]:
        """Get available working-hour slots within a simulation-time window.

        Walks the window hour-by-hour, merging consecutive available hours
        into contiguous (start, end) slots expressed in simulation seconds.

        Args:
            start_sim: Start of window in simulation seconds
            end_sim: End of window in simulation seconds
            time_converter: Optional converter from sim-seconds to datetime.
                           If None, a simple epoch of 2016-01-01 00:00 is assumed.
            neglect_sick_leave: If True, ignore sick leave absences (for bottleneck detection)
            reference_dt: Optional reference datetime for time-0.
                         Ignored when *time_converter* is supplied.

        Returns:
            List of (slot_start, slot_end) tuples in simulation seconds.
        """
        from datetime import datetime as _dt, timedelta as _td

        STEP = 3600  # 1-hour granularity in seconds

        if time_converter is not None:
            _to_dt = time_converter.sim_time_to_datetime
        else:
            ref = reference_dt or _dt(2016, 1, 1)
            _to_dt = lambda s: ref + _td(seconds=int(s))  # noqa: E731

        slots: list[tuple[int, int]] = []
        current = int(start_sim)
        slot_start: int | None = None

        while current < end_sim:
            dt = _to_dt(current)
            if self.is_available_at(dt=dt, neglect_sick_leave=neglect_sick_leave):
                if slot_start is None:
                    slot_start = current
            else:
                if slot_start is not None:
                    slots.append((slot_start, current))
                    slot_start = None
            current += STEP

        # Close trailing slot
        if slot_start is not None:
            slots.append((slot_start, int(end_sim)))

        return slots

    def get_available_time_in_seconds(
        self,
        start_sim: int | float,
        end_sim: int | float,
        time_converter: SimulationTimeConverter = None,
        neglect_sick_leave: bool = False,
        reference_dt: datetime = None
    ) -> float:
        """Total available working seconds within a simulation-time window.

        Args:
            start_sim: Start of window in simulation seconds
            end_sim: End of window in simulation seconds
            time_converter: Optional converter (see *get_available_slots*).
            neglect_sick_leave: If True, ignore sick leave absences (for bottleneck detection)
            reference_dt: Optional reference datetime for time-0.

        Returns:
            Total available seconds (float).
        """
        cache_key = (start_sim, end_sim, neglect_sick_leave)
        if cache_key in self._cache:
            return self._cache[cache_key]

        slots = self.get_available_slots(
            start_sim, end_sim,
            time_converter=time_converter,
            neglect_sick_leave=neglect_sick_leave,
            reference_dt=reference_dt,
        )
        total_seconds = sum(end - start for start, end in slots)
        
        # Simple cache management: prevent unbounded growth
        if len(self._cache) < 1000:
            self._cache[cache_key] = total_seconds
            
        return total_seconds

    def is_absent_in_range(
        self,
        start_dt: 'datetime',
        end_dt: 'datetime'
    ) -> bool:
        """Check whether the resource has any absence overlapping a date range.

        Args:
            start_dt: Start datetime
            end_dt: End datetime

        Returns:
            True if at least one absence overlaps.
        """
        return len(self.get_absences_in_range(start_dt, end_dt)) > 0

    def add_absence(self, absence: Absence) -> None:
        """Add an absence period to the calendar."""
        self.absences.append(absence)
        # Keep absences sorted by start date for efficiency
        self.absences.sort(key=lambda a: a.start_date)
    
    def get_absences_in_range(self, start_dt: datetime, end_dt: datetime) -> List[Absence]:
        """Get all absences that overlap with a date range.
        
        Args:
            start_dt: Start of range
            end_dt: End of range
            
        Returns:
            List of absences overlapping with range
        """
        return [
            absence for absence in self.absences
            if absence.start_date < end_dt and absence.end_date > start_dt
        ]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'resource_id': self.resource_id,
            'schedule': {
                'weekday_hours': {str(k): v for k, v in self.schedule.weekday_hours.items()},
                'timezone_offset': self.schedule.timezone_offset
            },
            'absences': [absence.to_dict() for absence in self.absences]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ResourceCalendar':
        """Create from dictionary."""
        schedule = WorkingSchedule(
            weekday_hours={int(k): tuple(v) for k, v in data['schedule']['weekday_hours'].items()},
            timezone_offset=data['schedule'].get('timezone_offset', 0.0)
        )
        absences = [Absence.from_dict(a) for a in data.get('absences', [])]
        return cls(
            resource_id=data['resource_id'],
            schedule=schedule,
            absences=absences
        )
