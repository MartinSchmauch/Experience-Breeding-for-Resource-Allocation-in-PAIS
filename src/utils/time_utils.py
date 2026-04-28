"""
Time conversion utilities for simulation.

The simulation uses **integer seconds** as its internal time unit.
All SimPy ``env.now`` values, durations, arrival times and timeouts are
expressed in whole seconds.

Helper functions convert between seconds and hours for external
interfaces (config files, log output, KPI reports) where hours are the
more natural unit.
"""

from datetime import datetime, timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# Conversion helpers (module-level, usable without a converter instance)
# ---------------------------------------------------------------------------

SECONDS_PER_HOUR: int = 3600
SECONDS_PER_DAY: int = 86_400


def hours_to_seconds(hours: float) -> int:
    """Convert a duration in hours to whole seconds (rounded)."""
    return round(hours * SECONDS_PER_HOUR)


def seconds_to_hours(seconds: int | float) -> float:
    """Convert a duration in seconds to fractional hours."""
    return seconds / SECONDS_PER_HOUR


def days_to_seconds(days: float) -> int:
    """Convert a duration in days to whole seconds (rounded)."""
    return round(days * SECONDS_PER_DAY)


def seconds_to_days(seconds: int | float) -> float:
    """Convert a duration in seconds to fractional days."""
    return seconds / SECONDS_PER_DAY


class SimulationTimeConverter:
    """
    Converts between simulation time (integer seconds) and real calendar datetime.
    
    Simulation time is measured in whole seconds from the simulation start.
    Calendar time allows mapping to actual dates for working hours/calendar logic.
    """
    
    def __init__(self, start_datetime: datetime):
        """
        Initialize converter with simulation start datetime.
        
        Args:
            start_datetime: Real-world datetime when simulation starts (time 0)
        """
        self.start_datetime = start_datetime
    
    def sim_time_to_datetime(self, sim_time: int | float) -> datetime:
        """
        Convert simulation time to calendar datetime.
        
        Args:
            sim_time: Simulation time in seconds from start
            
        Returns:
            Corresponding calendar datetime
        """
        return self.start_datetime + timedelta(seconds=int(sim_time))
    
    def datetime_to_sim_time(self, dt: datetime) -> int:
        """
        Convert calendar datetime to simulation time.
        
        Args:
            dt: Calendar datetime
            
        Returns:
            Simulation time in seconds from start
        """
        delta = dt - self.start_datetime
        return int(delta.total_seconds())
    
    def get_weekday(self, sim_time: int | float) -> int:
        """
        Get weekday for a simulation time.
        
        Args:
            sim_time: Simulation time in seconds
            
        Returns:
            Weekday (0=Monday, 6=Sunday)
        """
        dt = self.sim_time_to_datetime(sim_time)
        return dt.weekday()
    
    def get_hour_of_day(self, sim_time: int | float) -> float:
        """
        Get hour of day for a simulation time.
        
        Args:
            sim_time: Simulation time in seconds
            
        Returns:
            Hour of day (0.0-24.0, can be fractional)
        """
        dt = self.sim_time_to_datetime(sim_time)
        return dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    
    def is_working_hours(
        self,
        sim_time: int | float,
        start_hour: float = 9.0,
        end_hour: float = 17.0,
        working_weekdays: Optional[set[int]] = None
    ) -> bool:
        """
        Check if simulation time falls within standard working hours.
        
        Args:
            sim_time: Simulation time in seconds
            start_hour: Start of working day (default 9am)
            end_hour: End of working day (default 5pm)
            working_weekdays: Set of working weekdays (default Mon-Fri: {0,1,2,3,4})
            
        Returns:
            True if within working hours
        """
        if working_weekdays is None:
            working_weekdays = {0, 1, 2, 3, 4}  # Monday-Friday
        
        dt = self.sim_time_to_datetime(sim_time)
        weekday = dt.weekday()
        
        # Check if working day
        if weekday not in working_weekdays:
            return False
        
        # Check if within hours
        hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        return start_hour <= hour < end_hour


def create_default_converter(year: int = 2024, month: int = 1, day: int = 1) -> SimulationTimeConverter:
    """
    Create a time converter with default start date.
    
    Args:
        year: Start year (default 2024)
        month: Start month (default January)
        day: Start day (default 1st)
        
    Returns:
        Configured SimulationTimeConverter
    """
    start_dt = datetime(year, month, day, 0, 0, 0)  # Midnight on start date
    return SimulationTimeConverter(start_dt)
