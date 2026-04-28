"""
Simulation engine using SimPy for event management.

This engine combines SimPy's discrete-event simulation framework with
custom scheduling logic, experience-based learning, process models,
and realistic working hours with calendar support.

SimPy handles:
- Time management and event scheduling
- Resource request/release semantics
- Generator-based process definitions

Custom code handles:
- Scheduling algorithms (5 schedulers)
- Experience store and learning
- Process model navigation
- KPI calculation
- Working hours and calendar-based availability
"""

import simpy
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

from ..entities.case import Case
from ..entities.task import Task
from ..entities.resource import Resource
from ..entities.calendar import ResourceCalendar, WorkingSchedule
from ..scheduling.base import Scheduler, SchedulingContext
from ..scheduling.experience_based import DUMMY_RESOURCE_ID
from ..experience.store import ExperienceStore
from ..experience.updater import ExperienceUpdater, LearningModel
from ..experience.level_tracker import ExperienceLevelTracker
from ..process.model import ProcessModel
from ..io.log_writer import EventLogWriter
from ..evaluation.daily_summary_logger import DailySummaryLogger
from ..evaluation.daily_summary_aggregator import DailySummaryAggregator
from ..utils.time_utils import (
    SimulationTimeConverter,
    hours_to_seconds,
    seconds_to_hours,
    SECONDS_PER_HOUR,
    SECONDS_PER_DAY,
)
from .state import SimulationState

logger = logging.getLogger(__name__)


@dataclass
class _MentorSentinel:
    """Lightweight placeholder pushed into the mentor's priority queue.

    When the daily solver assigns a mentoring task to mentee→mentor pair,
    a sentinel is simultaneously enqueued into the *mentor's* resource store
    at the same composite priority.  This serves two purposes:

    1. **Diagnostic**: The mentor's worker logs/prints the sentinel on pickup,
       making it easy to verify in simulation output that the mechanism fired.
    2. **Future extensibility**: Provides a hook for more sophisticated
       coordination (e.g. yielding to give the mentee priority on the lock).

    The sentinel is silently discarded during the daily drain so it never
    contaminates the next scheduling cycle as a phantom task.
    """
    mentor_resource_id: str   # resource that will mentor
    mentee_task_id: str       # task.id of the corresponding mentoring task
    activity_name: str        # for readable log output
    sequence: int             # monotonic tie-breaker (same role as _queue_sequence)


class SimulationEngine:
    """
    Discrete-event simulation engine using SimPy for event management.
    
    Combines SimPy's mature event scheduling framework with custom
    scheduling logic, experience-based learning, and process model navigation.
    """
    
    def __init__(
        self,
        process_model: ProcessModel,
        scheduler: Scheduler,
        experience_store: ExperienceStore,
        resources: Dict[str, Resource],
        log_writer: Optional[EventLogWriter] = None,
        learning_model: LearningModel = LearningModel.RICHARDS,
        progress_bar: Optional[Any] = None,
        resource_calendars: Optional[Dict[str, ResourceCalendar]] = None,
        time_converter: Optional[SimulationTimeConverter] = None,
        config: Optional[Dict[str, Any]] = None,
        daily_summary_logger: Optional[DailySummaryLogger] = None,
    ):
        """
        Initialize the hybrid SimPy simulation engine.
        
        Args:
            process_model: Process model defining case workflows
            scheduler: Scheduling algorithm for task-resource assignment
            experience_store: Performance profiles for duration sampling
            resources: Available resources keyed by resource_id
            log_writer: Optional event logger for XES output
            learning_model: Learning curve model for experience breeding
            progress_bar: Optional tqdm progress bar for tracking completion
            resource_calendars: Optional calendars for working hours/absences
            time_converter: Optional time converter for calendar datetime mapping
        """
        self.config = config or {}
        
        self.process_model = process_model
        self.scheduler = scheduler
        self.experience_store = experience_store
        self.resources = resources
        self.log_writer = log_writer
        self.progress_bar = progress_bar
        self.daily_summary_logger = daily_summary_logger
        self._open_daily_summary: Optional[Dict[str, Any]] = None
        self._open_daily_day_start: Optional[float] = None
        self._open_daily_day_end: Optional[float] = None
        
        max_simulation_days = self.config.get('simulation', {}).get('max_simulation_days', None)  # None = run until completion
        self.max_simulation_time = max_simulation_days * SECONDS_PER_DAY if max_simulation_days else None  # Convert days to seconds
        self.max_tasks_per_case = self.config.get('simulation', {}).get('max_tasks_per_case', 100)  # Default to 100 if not set
        
        # Working hours configuration
        self.enable_working_hours = self.config.get('working_hours', {}).get('enabled', False) 
        self.resource_calendars = resource_calendars or {}
        self.time_converter = time_converter
        self.default_working_schedule = self._build_default_working_schedule()
        self.overtime_hours: Dict[str, Tuple[float, float]] = {rid: (0.0, 0.0) for rid in resources.keys()} # resource_id -> (total_overtime, max_single_instance_overtime)
        
        # Overtime configuration for worker processes
        constraints = self.config.get('optimization', {}).get('constraints', {})
        self.allow_overtime = constraints.get('allow_overtime', False)
        self.max_overtime_hours_per_day = constraints.get('max_overtime_hours', 0.0)
        
        # Initialize experience updater for breeding
        self.experience_updater = ExperienceUpdater(
            experience_store=self.experience_store,
            learning_model=learning_model,
            breeding_params=self.config.get('experience', {}).get('breeding_params', {})
        )
        
        # Initialize experience level tracker
        if self.config.get('experience', {}).get('track_experience_levels', True):
            self.experience_tracker = ExperienceLevelTracker()
        else:
            self.experience_tracker = None
        
        # Performance optimization: Batching and throttling
        self.max_queue_lengths: Dict[str, int] = {rid: 0 for rid in resources.keys()}
        
        # Drain statistics: cumulative counts across the whole simulation
        self._total_drained_tasks = 0  # Tasks drained back from queues across all days
        self._total_drain_days = 0     # Number of days where drain occurred
        self._total_deferred_tasks = 0 # Tasks deferred by solver across all days
        self._total_dropped_tasks = 0  # Tasks dropped after max deferrals
        
        # Duration divergence tracking: per-resource list of (task_id, activity, estimated, sampled)
        # Reset each scheduling cycle; used to diagnose solver vs execution mismatches
        self._daily_task_durations: Dict[str, list] = {rid: [] for rid in resources.keys()}
        
        # Activity benchmarks: minimum observed duration per activity (expert floor)
        self.activity_benchmarks: Dict[str, float] = {}  # activity_name -> benchmark_duration (seconds)
        self.beginner_durations: Dict[str, float] = {}  # activity_name -> duration for inexperienced resources (seconds) p99 duration
        self._load_activity_benchmarks()
        
        # Mentoring support
        self.mentoring_config = self.config.get('mentoring', {}) if self.config else {}
        self.mentoring_config['bottleneck_detection'] = self.config.get('bottleneck_detection', {}) if self.config else {}

        # Capability bootstrap support
        self.bootstrap_config = self.config.get('bootstrap_capability', {}) if self.config else {}
        self.bootstrap_enabled = self.bootstrap_config.get('enabled', False)
        self._bootstrap_assignments_executed = 0
        self._bootstrap_total_penalty_seconds = 0
        self._bootstrap_activities_executed: set[str] = set()
        
        # BATCH SCHEDULING: Configuration
        self.scheduling_mode = self.config.get('scheduling', {}).get('mode', 'batch')  # 'immediate' or 'batch'
        self.scheduling_time = self.config.get('scheduling', {}).get('scheduling_time', 8)  # Hour of day for daily scheduling (e.g. 0 for midnight)
        self.planning_horizon_seconds = hours_to_seconds(
            self.config.get('scheduling', {}).get('planning_horizon_hours', 24.0)
        )  # Planning horizon in seconds
        self.unscheduled_tasks = []  # Tasks waiting for batch scheduling
        self.max_task_deferrals = self.config.get('optimization', {}).get('max_task_deferrals', 5)  # Max days a task can be deferred before being dropped
        self._queue_sequence = 0  # Monotonic counter: guarantees unique priority tuples → Task.__lt__ never called
        
        # ====== SIMPY OBJECTS ======
        # SimPy environment for time management
        self.env = simpy.Environment()
        
        # Per-resource task queues using SimPy PriorityStore for mentoring priority support
        # Each store holds (priority, Task) tuples
        self.resource_stores: Dict[str, simpy.PriorityStore] = {}
        # Per-resource mutual-exclusion locks (simpy.Resource with capacity=1).
        # A worker acquires its own lock before executing any task.
        # blocking the mentor's worker from picking up new tasks until the
        # mentoring session ends.
        self.resource_locks: Dict[str, simpy.Resource] = {}
        for resource_id in resources.keys():
            self.resource_stores[resource_id] = simpy.PriorityStore(self.env)
            self.resource_locks[resource_id] = simpy.Resource(self.env, capacity=1)
        
        # Start resource worker processes
        for resource_id in resources.keys():
            self.env.process(self._resource_worker_process(resource_id))
        
        # Start daily progress bar update process if progress bar provided
        if self.progress_bar is not None:
            self.env.process(self._progress_bar_update_process())
        
        # Start daily batch scheduler process if in batch mode
        if self.scheduling_mode == 'batch':
            self.env.process(self._daily_scheduler_process())
            logger.info(f"Batch scheduling enabled (daily planning at {self.scheduling_time}:00)")
        
        # Simulation state tracking (preserves custom logic)
        self.state = SimulationState()
        # Single source of truth for resources: keep SimulationState pointing
        # to the same resource dictionary used by the engine.
        self.state.resources = self.resources
        for resource_id in self.resources.keys():
            self.state.resource_busy_time[resource_id] = 0.0
            self.state.resource_last_start[resource_id] = 0.0

        self.daily_summary_aggregator = DailySummaryAggregator(
            scheduler=self.scheduler,
            resources=self.resources,
            resource_calendars=self.resource_calendars,
            time_converter=self.time_converter,
            enable_working_hours=self.enable_working_hours,
            planning_horizon_seconds=float(self.planning_horizon_seconds),
            max_task_deferrals=int(self.max_task_deferrals),
            daily_task_durations=self._daily_task_durations,
            state=self.state,
            env=self.env,
            daily_summary_logger=self.daily_summary_logger,
        )
    
    def _load_activity_benchmarks(self) -> None:
        """Load per-activity benchmark durations from activity_requirements.yaml.
        
        Benchmarks represent the expert-level (experience >= 95) duration floor
        for each activity — the minimum duration ever observed across all resources.
        """
        from pathlib import Path
        import yaml
        
        config_path = Path(__file__).parent.parent.parent / "config" / "activity_requirements.yaml"
        if not config_path.exists():
            logger.warning("activity_requirements.yaml not found — no activity benchmarks loaded")
            return
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            
            benchmarks = config.get('activity_benchmarks', {})
            if benchmarks:
                self.activity_benchmarks = {
                    str(k): float(v) for k, v in benchmarks.items()
                    if v is not None
                }
                logger.info(f"Loaded {len(self.activity_benchmarks)} activity benchmarks")
            else:
                logger.info("No activity_benchmarks section found in activity_requirements.yaml")
            
            beginner_durations = config.get('default_beginner_durations', {})
            if beginner_durations:
                self.beginner_durations = {
                    str(k): float(v) for k, v in beginner_durations.items()
                    if v is not None
                }
                logger.info(f"Loaded {len(self.beginner_durations)} beginner durations")
            else:
                logger.info("No beginner_durations section found in activity_requirements.yaml")

        except Exception as e:
            logger.warning(f"Failed to load activity benchmarks: {e}")
    
    def _build_scheduling_context(self) -> SchedulingContext:
        """Build scheduling context for resource selection.
        
        Context now includes queue lengths for workload-aware scheduling.
        Queue data is computed in a single pass over resource_stores.
        """
        # Build queue data in single pass: O(n*m) where n=resources, m=avg queue size
        queue_lengths = self._get_resource_queue_lengths()
        pending_tasks = self._collect_pending_tasks()
        active_cases = self.state.get_active_cases()  # Get active cases from state
        return SchedulingContext(
            all_resources=list(self.resources.values()),
            active_cases=active_cases,
            pending_tasks=pending_tasks,
            experience_store=self.experience_store,
            current_time=self.env.now,
            queue_lengths=queue_lengths,  # {resource_id: queue length}
            simulation_state=self.state,  # Pass entire state for maximum flexibility
            resource_calendars=self.resource_calendars if self.enable_working_hours else None
        )
    
    def _get_resource_queue_lengths(self) -> Dict[str, int]:
        """Get queue lengths in a single pass.
        
        Returns:
            Dict[str, int] - resource_id -> queue length
        """
        return {
            resource_id: len(store.items)
            for resource_id, store in self.resource_stores.items()
        }
    
    
    def schedule_case_arrival(self, case: Case, arrival_time: float):
        """
        Schedule a case to arrive at the simulation at a specific time.
        
        Args:
            case: The case to schedule
            arrival_time: Simulation time when case should arrive
        """
        self.env.process(self._case_process(case, arrival_time))
    
    def _case_process(self, case: Case, arrival_time: float):
        """
        Generator process for a case's lifecycle.
        
        Handles case arrival, task creation, and completion tracking.
        This is a SimPy generator process that yields events.
        
        Args:
            case: The case being processed
            arrival_time: Simulation time when case arrives
            
        Yields:
            SimPy events (timeout for arrival delay)
        """
        # Register case in state
        self.state.add_case(case)
        
        # Wait until arrival time
        if arrival_time > self.env.now:
            yield self.env.timeout(arrival_time - self.env.now)

        # Get initial tasks from process model
        initial_tasks = self.process_model.get_initial_tasks(case)
        # Start task processes for initial tasks
        for task in initial_tasks:
            self.env.process(self._task_process(task))
    
    def _task_process(self, task: Task):
        """Generator process for task queueing (push-based allocation).
        
        Assigns task to a resource queue immediately via scheduler,
        then terminates. Actual execution happens in resource worker process.
        
        Args:
            task: The task being processed
            
        Yields:
            None (returns immediately after queuing)
        """
        self.state.add_task(task)
        
        # Add task to unscheduled pool for daily planning
        task.queued_time = self.env.now
        self.unscheduled_tasks.append(task)
        
        # Task waits here - daily scheduler will assign it
        return 
        yield
    
    def _is_within_overtime_window(self, calendar: ResourceCalendar, dt) -> bool:
        """Check if datetime falls within the overtime window after regular working hours.
        
        The overtime window extends from the end of regular working hours
        (e.g. 17:00) by max_overtime_hours_per_day (e.g. +2h → until 19:00),
        but only on working days.
        
        Args:
            calendar: ResourceCalendar for the resource
            dt: Datetime to check
            
        Returns:
            True if within the overtime window (past regular hours but
            within max_overtime_hours), False otherwise.
        """
        weekday = dt.weekday()
        if not calendar.schedule.is_working_day(weekday):
            return False
        
        hours = calendar.schedule.get_working_hours(weekday)
        if hours is None:
            return False
        
        _, end_hour = hours
        current_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        
        # Overtime window: [end_hour, end_hour + max_overtime_hours)
        return end_hour <= current_hour < end_hour + self.max_overtime_hours_per_day

    def _build_default_working_schedule(self) -> WorkingSchedule:
        """Build fallback working schedule from config `working_hours.default_schedule`."""
        default_week = {
            'monday': 0,
            'tuesday': 1,
            'wednesday': 2,
            'thursday': 3,
            'friday': 4,
            'saturday': 5,
            'sunday': 6,
        }
        raw = self.config.get('working_hours', {}).get('default_schedule', {})
        weekday_hours: Dict[int, tuple[float, float]] = {}
        for day_name, day_idx in default_week.items():
            hours = raw.get(day_name)
            if isinstance(hours, (list, tuple)) and len(hours) == 2:
                try:
                    start_h = float(hours[0])
                    end_h = float(hours[1])
                    if end_h > start_h:
                        weekday_hours[day_idx] = (start_h, end_h)
                except (TypeError, ValueError):
                    continue
        if not weekday_hours:
            return WorkingSchedule()
        return WorkingSchedule(weekday_hours=weekday_hours)

    def _get_calendar_for_resource(self, resource_id: str) -> Optional[ResourceCalendar]:
        """Resolve calendar for a resource, falling back to default schedule."""
        if not self.enable_working_hours:
            return None
        calendar = self.resource_calendars.get(resource_id)
        if calendar is not None:
            return calendar
        return ResourceCalendar(resource_id=resource_id, schedule=self.default_working_schedule)

    def _is_resource_start_allowed(self, resource_id: str, dt) -> bool:
        """Return whether a resource is allowed to start a task at datetime ``dt``."""
        if not self.enable_working_hours:
            return True

        calendar = self._get_calendar_for_resource(resource_id)
        if calendar is None:
            return True

        available = calendar.is_available_at(dt)
        if not available and self.allow_overtime and self.max_overtime_hours_per_day > 0:
            available = self._is_within_overtime_window(calendar, dt)
        return available

    def _next_resource_start_time(self, resource_id: str, from_dt):
        """Return next datetime when ``resource_id`` may start work, or None."""
        if not self.enable_working_hours:
            return from_dt

        calendar = self._get_calendar_for_resource(resource_id)
        if calendar is None:
            return from_dt

        return calendar.get_next_available_time(from_dt)

    def _next_joint_start_time(self, resource_ids: List[str], from_dt):
        """Find earliest datetime where all given resources can start work."""
        candidate = from_dt
        for _ in range(16):
            moved = False
            for resource_id in resource_ids:
                if self._is_resource_start_allowed(resource_id, candidate):
                    continue
                nxt = self._next_resource_start_time(resource_id, candidate)
                if nxt is None:
                    return None
                if nxt > candidate:
                    candidate = nxt
                    moved = True
            if not moved:
                return candidate
        return None

    def _resource_worker_process(self, resource_id: str):
        """Generator process that continuously processes tasks from a resource's queue.
        
        This worker runs for the entire simulation, pulling tasks from the resource's
        queue store, executing them sequentially with experience-based durations,
        updating experience, and triggering next tasks in the process.
        
        With working hours enabled, the worker checks calendar availability before
        starting tasks and waits until available if outside working hours.
        When overtime is allowed, the worker can also start tasks in the overtime
        window (past regular hours, up to max_overtime_hours after end of day).
        
        Args:
            resource_id: ID of the resource this worker manages
            
        Yields:
            SimPy events (store.get(), timeout for execution and waiting)
        """
        calendar = self._get_calendar_for_resource(resource_id) if self.enable_working_hours else None
        
        while True:
            # Wait for next task in queue (PriorityStore returns tuple)
            priority, task = yield self.resource_stores[resource_id].get()

            # ---- Sentinel short-circuit ----
            # When a mentoring task was assigned (mentee side), the solver also
            # pushed a _MentorSentinel into THIS resource's queue at the same
            # priority.  On pickup we just discard it and loop — no lock acquired.
            if isinstance(task, _MentorSentinel):
                continue

            # Acquire this resource's lock before executing any task.
            # - Standard tasks: usually immediate.
            # - Mentoring tasks: the mentee lock is held while mentor synchronization runs.
            # Re-validate start permissions *after* lock acquisition because simulation
            # time may have advanced while waiting for the lock.
            lock_req = None
            try:
                while True:
                    lock_req = self.resource_locks[resource_id].request()
                    yield lock_req

                    if self.enable_working_hours and self.time_converter:
                        current_dt = self.time_converter.sim_time_to_datetime(self.env.now)
                        if not self._is_resource_start_allowed(resource_id, current_dt):
                            next_available_dt = self._next_resource_start_time(resource_id, current_dt)
                            self.resource_locks[resource_id].release(lock_req)
                            lock_req = None

                            if next_available_dt is None:
                                logger.warning(
                                    f"Resource {resource_id} has no available start time in forecast period - skipping task"
                                )
                                break

                            wait_time = self.time_converter.datetime_to_sim_time(next_available_dt) - self.env.now
                            if wait_time > 0:
                                logger.debug(
                                    f"Resource {resource_id} waiting {seconds_to_hours(wait_time):.2f}h until next allowed start"
                                )
                                yield self.env.timeout(wait_time)
                            continue
                    break

                if lock_req is None:
                    continue

                # Dispatch to appropriate execution method based on task type
                if task.is_mentoring_task():
                    yield from self._execute_mentoring_task(task, resource_id, calendar)
                else:
                    yield from self._execute_standard_task(task, resource_id, calendar)
            finally:
                if lock_req is not None:
                    self.resource_locks[resource_id].release(lock_req)

    def _execute_standard_task(self, task: Task, resource_id: str, calendar: ResourceCalendar = None):
        """Execute a standard (non-mentoring) task.
        
        Args:
            task: Task to execute
            resource_id: Resource executing the task
            calendar: ResourceCalendar for the resource (optional)
        Yields:
            SimPy timeout events for task execution
        """        
        # Get the case this task belongs to
        case = self.state.cases.get(task.case_id)
        if case is None:
            logger.error(f"Case {task.case_id} not found for task {task.id}")
            return
        
        # get resource instance to look up resource's current experience level and activity benchmark
        resource = self.resources.get(resource_id)
        # Sample duration from experience store (scaled by experience level)
        duration = self.experience_store.sample_duration(
            resource_id=resource_id,
            activity_name=task.activity_name,
            is_mentoring_task=False,
            context=case.attributes,
            experience_level=resource.get_experience_level(task.activity_name) if resource else None,
            benchmark_duration=self.activity_benchmarks.get(task.activity_name),
            beginner_duration=self.beginner_durations.get(task.activity_name),
            mentoring_config=self.mentoring_config
        )

        # Add one-time onboarding penalty for bootstrap assignments.
        if task.bootstrap_assignment and not task.bootstrap_onboarding_applied:
            onboarding_seconds = max(0, int(task.bootstrap_onboarding_seconds or 0))
            if onboarding_seconds > 0:
                duration += onboarding_seconds
                task.bootstrap_onboarding_applied = True
                self._bootstrap_assignments_executed += 1
                self._bootstrap_total_penalty_seconds += onboarding_seconds
                self._bootstrap_activities_executed.add(str(task.activity_name))

        if duration > hours_to_seconds(8.0):
            logger.warning(f"   WARNING: Task {task.id} ({task.activity_name}) assigned to resource {resource_id} has sampled duration of {seconds_to_hours(duration):.2f}h (>8h)")
        
        self.state.start_task(task.id, self.env.now, resource_id, duration)
        
        # Track duration divergence (estimated from solver vs sampled for execution)
        if resource_id in self._daily_task_durations:
            self._daily_task_durations[resource_id].append((
                task.id,
                task.activity_name,
                task.estimated_duration or 0,  # solver's max-based estimate (seconds)
                duration,                       # sampled duration (seconds)
            ))
        
        # Track start time for overtime calculation
        task_start_time = self.env.now
        
        # Log task start
        if self.log_writer is not None:
            self.log_writer.log_task_start(
                case_id=case.id,
                task_id=task.id,
                activity_name=task.activity_name,
                resource_id=resource_id,
                timestamp=round(seconds_to_hours(task_start_time), 4),
                sim_datetime=self.time_converter.sim_time_to_datetime(task_start_time) if self.time_converter else None,
                task_type=task.task_type.value if task.task_type else "unknown",
            )
        
        # Execute task (wait for duration)
        yield self.env.timeout(duration)
        
        # Check if task completed outside working hours (overtime)
        if self.enable_working_hours and calendar and self.time_converter:
            start_dt = self.time_converter.sim_time_to_datetime(task_start_time)
            completion_dt = self.time_converter.sim_time_to_datetime(self.env.now)
            
            overtime = calendar.calculate_overtime_duration(start_dt, completion_dt)
            self.overtime_hours[resource_id] = (self.overtime_hours[resource_id][0] + overtime, max(self.overtime_hours[resource_id][1], overtime))
        
        self.state.complete_task(task.id, self.env.now)

        # Update experience store based on actual performance
        self._update_experience_after_completion(task, resource_id)
        # Log task completion
        if self.log_writer is not None:
            self.log_writer.log_task_complete(
                case_id=case.id,
                task_id=task.id,
                activity_name=task.activity_name,
                resource_id=resource_id,
                timestamp=round(seconds_to_hours(self.env.now), 4),
                sim_datetime=self.time_converter.sim_time_to_datetime(self.env.now) if self.time_converter else None,
                task_type=task.task_type.value if task.task_type else "unknown",
            )
        
        # Check if case has reached activity limit
        case_complete = False
        if self.max_tasks_per_case is not None:
            if len(case.trace) >= self.max_tasks_per_case:
                # Case reached activity limit
                case_complete = True
                logger.debug(f"Case {case.id} reached max activities limit ({self.max_tasks_per_case})")
        
        if not case_complete:
            # Check for next tasks in the process
            next_tasks = self.process_model.get_next_tasks(case, task)
            if next_tasks:
                # Queue next tasks
                for next_task in next_tasks:
                    self.state.add_task(next_task)
                    self.env.process(self._task_process(next_task))
            else:
                # No more tasks from process model, case is complete
                case_complete = True
        
        if case_complete:
            # Case is complete
            self.state.complete_case(case.id, self.env.now)
    
    def _update_experience_after_completion(
        self,
        task: Task,
        resource_id: str
    ) -> None:
        """Update experience store and track experience levels after task completion.
        
        Args:
            task: Completed task
            resource_id: Resource that performed the task
        """
        self.experience_updater.update_from_task(
                task=task,
                simulation_time=seconds_to_hours(self.env.now),
                resource_id=resource_id
            )
        
        # Record experience level snapshot for tracking (throttled to reduce overhead)
        # Only record every Nth task to reduce DataFrame append operations
        if self.experience_tracker is not None:
            resource = self.resources.get(resource_id)
            if resource:
                # Query the actual profile from store (updated by the experience updater)
                profile = self.experience_store.get_profile(
                    resource_id=resource_id,
                    activity_name=task.activity_name,
                    context=task.context
                )
                
                # Use real data from the experience profile
                experience_level = profile.experience_level
                task_count = profile.count
                mean_duration = profile.mean_duration
                
                self.experience_tracker.record_snapshot(
                    simulation_time=round(seconds_to_hours(self.env.now), 4),
                    sim_datetime=self.time_converter.sim_time_to_datetime(self.env.now) if self.time_converter else None,
                    resource_id=resource_id,
                    activity_name=task.activity_name,
                    experience_level=experience_level,
                    repetition_count=task_count,
                    mean_duration=mean_duration,
                    context=task.context
                )

    def _get_activity_requirements(self) -> Dict[str, float]:
        """Return activity requirements exposed by the active scheduler."""
        return self.daily_summary_aggregator.get_activity_requirements()

    def _build_activity_catalog(self, pending_tasks: List[Task]) -> List[str]:
        """Build stable activity catalog for day-level aggregates."""
        return self.daily_summary_aggregator.build_activity_catalog(pending_tasks)

    def _available_capacity_seconds(self, resource_id: str, day_start: float, day_end: float) -> float:
        """Return available capacity for one resource in the day window."""
        return self.daily_summary_aggregator._available_capacity_seconds(resource_id, day_start, day_end)

    def _build_capacity_snapshots(
        self,
        activity_catalog: List[str],
        day_start: float,
        day_end: float,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int], float]:
        """Build capacity aggregates needed by daily KPI summaries."""
        return self.daily_summary_aggregator.build_capacity_snapshots(activity_catalog, day_start, day_end)

    def _start_daily_summary(self, context: SchedulingContext) -> None:
        """Create a new open day summary before the solver is called."""
        self.daily_summary_aggregator.start_daily_summary(context)
        self._open_daily_summary = self.daily_summary_aggregator._open_daily_summary
        self._open_daily_day_start = self.daily_summary_aggregator._open_daily_day_start
        self._open_daily_day_end = self.daily_summary_aggregator._open_daily_day_end

    def _summarize_assignments(self, assignments: Dict[str, List[Task]]) -> Dict[str, Any]:
        """Build compact assignment aggregates for the currently open day."""
        return self.daily_summary_aggregator.summarize_assignments(assignments)

    def _merge_assignment_summary(self, assignment_summary: Dict[str, Any]) -> None:
        """Attach assignment aggregates to the open day summary."""
        self.daily_summary_aggregator.merge_assignment_summary(assignment_summary)
        self._open_daily_summary = self.daily_summary_aggregator._open_daily_summary

    def _compute_actual_task_completion_stats(self, day_start: float, day_end: float) -> Tuple[Dict[str, float], Dict[str, int]]:
        """Compute actual completed task hours/counts for tasks ending in [day_start, day_end)."""
        return self.daily_summary_aggregator.compute_actual_task_completion_stats(day_start, day_end)

    def _compute_resource_utilization_snapshot(self) -> Tuple[Dict[str, float], float, float]:
        """Compute per-resource and aggregate utilization from daily sampled durations."""
        return self.daily_summary_aggregator.compute_resource_utilization_snapshot()

    def _finalize_open_daily_summary(
        self,
        incomplete_by_activity: Optional[Dict[str, int]] = None,
        dropped_from_drain_by_activity: Optional[Dict[str, int]] = None,
        is_partial_day: bool = False,
    ) -> None:
        """Finalize the open day summary and append one compact JSONL row."""
        self.daily_summary_aggregator.finalize_open_daily_summary(
            incomplete_by_activity=incomplete_by_activity,
            dropped_from_drain_by_activity=dropped_from_drain_by_activity,
            is_partial_day=is_partial_day,
        )
        self._open_daily_summary = self.daily_summary_aggregator._open_daily_summary
        self._open_daily_day_start = self.daily_summary_aggregator._open_daily_day_start
        self._open_daily_day_end = self.daily_summary_aggregator._open_daily_day_end
    
    def _progress_bar_update_process(self):
        """Update progress bar once per simulation day.
        
        Yields:
            SimPy timeout events
        """
        if self.progress_bar is None:
            return
        
        last_update_day = 0
        
        while True:
            # Wait 24 hours (one simulation day)
            yield self.env.timeout(SECONDS_PER_DAY)
            
            # Calculate current day
            current_day = self.env.now // SECONDS_PER_DAY
            
            # Update progress bar by the number of days passed
            days_passed = current_day - last_update_day
            if days_passed > 0:
                self.progress_bar.update(days_passed)
                last_update_day = current_day
    
    def _execute_mentoring_task(self, task: Task, mentee_resource_id: str, calendar: ResourceCalendar = None):
        """Execute a mentoring task with dual-resource coordination.

        The mentor is matched and reserved via the scheduler (not by
        pushing the same Task into the mentor's queue).  Both resources
        are kept busy for the duration, after which the mentor is
        released back to its own worker loop.

        Args:
            task: Mentoring task to execute
            mentee_resource_id: ID of the mentee resource (already assigned)

        Yields:
            SimPy events for mentor matching and dual execution
        """
        # Get case
        case = self.state.cases.get(task.case_id)

        if task.mentee_resource_id != mentee_resource_id: # e.g. 89 (task.mentee_resource_id) statt 41 für mentee_resource_id
            print(f"WARNING: Task {task.id} has mentee {task.mentee_resource_id}, but expected {mentee_resource_id}")
        if task.mentee_resource_id == task.mentor_resource_id:
            print(f"WARNING: Task {task.id} has same mentee and mentor {task.mentee_resource_id} - this is likely a configuration error")

        # ---------- acquire mentor's SimPy lock ----------
        # This blocks until the mentor finishes its current task (if any),
        # ensuring both resources start the mentoring session simultaneously.
        # The mentor's worker loop also acquires this lock before executing
        # its own tasks, so the lock provides true mutual exclusion.
        mentor_lock_req = None
        mentor_resource = self.resources.get(task.mentor_resource_id)

        while True:
            mentor_lock_req = self.resource_locks[task.mentor_resource_id].request()
            yield mentor_lock_req  # wait until mentor is free

            # Re-check start permissions after waiting for mentor lock.
            # Simulation time can advance during the lock wait.
            if self.enable_working_hours and self.time_converter:
                current_dt = self.time_converter.sim_time_to_datetime(self.env.now)
                mentee_ok = self._is_resource_start_allowed(mentee_resource_id, current_dt)
                mentor_ok = self._is_resource_start_allowed(task.mentor_resource_id, current_dt)

                if not (mentee_ok and mentor_ok):
                    self.resource_locks[task.mentor_resource_id].release(mentor_lock_req)
                    mentor_lock_req = None

                    next_joint = self._next_joint_start_time(
                        [mentee_resource_id, task.mentor_resource_id],
                        current_dt,
                    )
                    if next_joint is None:
                        logger.warning(
                            f"No joint available start time for mentoring task {task.id}"
                        )
                        return

                    wait_time = self.time_converter.datetime_to_sim_time(next_joint) - self.env.now
                    if wait_time > 0:
                        logger.debug(
                            f"Mentoring task {task.id} waiting {seconds_to_hours(wait_time):.2f}h for joint availability"
                        )
                        yield self.env.timeout(wait_time)
                    continue
            break

        # Book-keeping: mark mentor as busy (status flag)
        if mentor_resource:
            mentor_resource.assign_task(task.id) # TODO: also assign the task to the mentee?

        try:
            # Look up mentee's current experience level and activity benchmark
            mentee_resource = self.resources.get(mentee_resource_id)
            mentor_experience_level = (
                mentor_resource.get_experience_level(task.activity_name)
                if mentor_resource else None
            )
            # Get base duration from experience store (for mentee, scaled by experience)
            duration = self.experience_store.sample_duration(
                resource_id=mentee_resource_id,
                activity_name=task.activity_name,
                is_mentoring_task=True,
                experience_level=mentee_resource.get_experience_level(task.activity_name) if mentee_resource else None,
                benchmark_duration=self.activity_benchmarks.get(task.activity_name),
                beginner_duration=self.beginner_durations.get(task.activity_name),
                mentoring_config=self.mentoring_config,
                mentor_experience_level=mentor_experience_level,
                required_capability_level=float(task.required_capability_level) if task.required_capability_level is not None else None,
            )

            # Apply mentoring duration multiplier
            # duration = duration * task.duration_multiplier

            if duration > hours_to_seconds(8.0):
                logger.warning(
                    f"  WARNING: Mentoring task {task.id} ({task.activity_name}) "
                    f"with mentee {mentee_resource_id} has duration of {seconds_to_hours(duration):.2f}h (>8h)"
                )

            # Assign both resources — start time reflects when BOTH are available
            task_start_time = self.env.now
            self.state.start_task(task.id, task_start_time, mentee_resource_id, duration)

            # Track sampled execution for daily utilization diagnostics.
            # Record one entry for mentee and one shadow entry for mentor.
            if mentee_resource_id in self._daily_task_durations:
                self._daily_task_durations[mentee_resource_id].append((
                    task.id,
                    task.activity_name,
                    task.estimated_duration or 0,
                    duration,
                ))
            if task.mentor_resource_id in self._daily_task_durations:
                self._daily_task_durations[task.mentor_resource_id].append((
                    task.id,
                    task.activity_name,
                    task.estimated_duration or 0,
                    duration,
                ))

            # Log task start (mentoring)
            if self.log_writer is not None:
                self.log_writer.log_task_start(
                    case_id=case.id,
                    task_id=task.id,
                    activity_name=task.activity_name,
                    resource_id=f"{mentee_resource_id}+{task.mentor_resource_id}",
                    timestamp=round(seconds_to_hours(task_start_time), 4),
                    sim_datetime=self.time_converter.sim_time_to_datetime(task_start_time) if self.time_converter else None,
                    task_type=task.task_type.value if task.task_type else "unknown",
                    mentor=task.mentor_resource_id if task.is_mentoring_task() else None,
                    mentee=task.mentee_resource_id if task.is_mentoring_task() else None,
                    mentoring_against_bottleneck=task.mentoring_against_bottleneck if task.is_mentoring_task() else None, 
                    emergency_mentoring=task.is_emergency_mentoring if task.is_mentoring_task() else None,
                )

            # Execute task (both resources occupied, both locks held)
            yield self.env.timeout(duration)

        finally:
            # ---------- release mentor ----------
            if mentor_resource:
                mentor_resource.release_task(task.id)
            # Release the mentor's SimPy lock so its worker can resume
            if mentor_lock_req is not None:
                self.resource_locks[task.mentor_resource_id].release(mentor_lock_req)
        
        # Check if task completed outside working hours (overtime)
        if self.enable_working_hours and calendar and self.time_converter:
            start_dt = self.time_converter.sim_time_to_datetime(task_start_time)
            completion_dt = self.time_converter.sim_time_to_datetime(self.env.now)
            
            overtime = calendar.calculate_overtime_duration(start_dt, completion_dt)
            self.overtime_hours[mentee_resource_id] = (self.overtime_hours[mentee_resource_id][0] + overtime, max(self.overtime_hours[mentee_resource_id][1], overtime))
        
        # Complete task in state (also calls task.complete() internally)
        self.state.complete_task(task.id, self.env.now)

        # Log task completion
        if self.log_writer is not None:
            self.log_writer.log_task_complete(
                case_id=case.id,
                task_id=task.id,
                activity_name=task.activity_name,
                resource_id=f"{mentee_resource_id}+{task.mentor_resource_id}",
                timestamp=round(seconds_to_hours(self.env.now), 4),
                sim_datetime=self.time_converter.sim_time_to_datetime(self.env.now) if self.time_converter else None,
                task_type=task.task_type.value if task.task_type else "unknown",
                mentor=task.mentor_resource_id if task.is_mentoring_task() else None,
                mentee=task.mentee_resource_id if task.is_mentoring_task() else None,
                mentoring_against_bottleneck=task.mentoring_against_bottleneck if task.is_mentoring_task() else None, 
                emergency_mentoring=task.is_emergency_mentoring if task.is_mentoring_task() else None,
            )

        # Update standard experience for both resources
        self._update_experience_after_completion(task, mentee_resource_id)

        # Continue process flow (same as standard task)
        case_complete = False
        if self.max_tasks_per_case is not None:
            if len(case.trace) >= self.max_tasks_per_case:
                case_complete = True
                logger.debug(f"Case {case.id} reached max activities limit ({self.max_tasks_per_case})")

        if not case_complete:
            next_tasks = self.process_model.get_next_tasks(case, task)
            if next_tasks:
                for next_task in next_tasks:
                    self.state.add_task(next_task)
                    self.env.process(self._task_process(next_task))
            else:
                case_complete = True

        if case_complete:
            self.state.complete_case(case.id, self.env.now)
    
    # ========== BATCH SCHEDULING METHODS ==========
    
    def _daily_scheduler_process(self):
        """
        Run daily batch scheduling at configured time (e.g., 8:00 AM).
        
        This process:
        1. Waits until scheduled time each day
        2. Collects all pending tasks from unscheduled pool
        3. Calls scheduler.plan_tasks_to_resources() for optimization
        4. Distributes optimized assignments to resource queues
        
        Yields:
            SimPy timeout events
        """
        logger.info(f"Daily batch scheduler started (scheduling at {self.scheduling_time}:00 each day)")
        
        while True:
            # Calculate time until next scheduling (e.g., 8:00 AM)
            
            if self.env.now > 0:
                yield self.env.timeout(SECONDS_PER_DAY)
            else:
                current_hour = (self.env.now % SECONDS_PER_DAY) / SECONDS_PER_HOUR
                hours_until_scheduling = (self.scheduling_time - current_hour) % 24
                if hours_until_scheduling > 0:
                    yield self.env.timeout(hours_to_seconds(hours_until_scheduling))
            # ---- Skip non-working days (weekends / holidays) ----
            # The solver assigns up to 8h of work per resource per day, but
            # workers only process during calendar working hours + potential overtime.
            if self.enable_working_hours and self.time_converter:
                current_dt = self.time_converter.sim_time_to_datetime(self.env.now)
                weekday = current_dt.weekday()  # 0=Mon … 6=Sun
                # Check against the first available calendar (all resources
                # share the same Mon-Fri schedule in our setup).
                any_working = False
                for cal in self.resource_calendars.values():
                    if cal.schedule.is_working_day(weekday):
                        any_working = True
                        break
                if not any_working:
                    continue

            # ---- Drain leftover queue items back into unscheduled pool ----
            # Workers may not have finished all assigned tasks from the
            # previous day (e.g. when sampled durations exceeded predictions).
            # Draining these back ensures the solver sees the FULL picture
            # and can optimally redistribute all work for today.
            drained_count = 0
            drained_by_resource: Dict[str, list] = {}  # rid -> [leftover tasks]
            incomplete_by_activity: Dict[str, int] = defaultdict(int)
            dropped_from_drain_by_activity: Dict[str, int] = defaultdict(int)
            for rid, store in self.resource_stores.items():
                while store.items:
                    _priority, leftover_task = store.items.pop(0)
                    # Sentinels are transient coordination artifacts — silently
                    # discard them rather than treating them as real tasks
                    if isinstance(leftover_task, _MentorSentinel):
                        continue
                    incomplete_by_activity[str(leftover_task.activity_name)] += 1
                    # Reset task so it can be re-planned
                    leftover_task.reset_for_rescheduling() # reset task properties, increment defer count
                    if leftover_task.defer_count > self.max_task_deferrals:
                        self._total_dropped_tasks += 1
                        dropped_from_drain_by_activity[str(leftover_task.activity_name)] += 1
                        self.state.cancel_task(leftover_task.id, self.env.now)  # Drop: exceeded max deferrals
                    else:
                        self._total_deferred_tasks += 1
                        self.unscheduled_tasks.append(leftover_task)
                    drained_count += 1
                    drained_by_resource.setdefault(rid, []).append(leftover_task)
            if drained_count > 0:
                self._total_drained_tasks += drained_count
                self._total_drain_days += 1

            # Finalize previous scheduling day here: this is the canonical point
            # where incompletion is observable (leftovers in resource queues).
            self._finalize_open_daily_summary(
                incomplete_by_activity=dict(incomplete_by_activity),
                dropped_from_drain_by_activity=dict(dropped_from_drain_by_activity),
                is_partial_day=False,
            )

            # Reset daily tracking for the new scheduling cycle
            for rid in self._daily_task_durations:
                self._daily_task_durations[rid] = []
                        
            # Build scheduling context
            context = self._build_scheduling_context()
            self._start_daily_summary(context)
            
            try:
                # Call batch scheduler's plan_tasks_to_resources method
                solve_start = time.monotonic()
                assignments = self.scheduler.plan_tasks_to_resources(
                    context=context,
                    planning_horizon_hours=seconds_to_hours(self.planning_horizon_seconds),
                    enforce_working_hours=self.enable_working_hours,
                    duration_predictor=self.scheduler.duration_predictor if hasattr(self.scheduler, 'duration_predictor') else None,
                    max_solver_time_seconds=60.0,
                    optimality_gap=0.05
                )
                solve_wall_time = time.monotonic() - solve_start
                assignment_summary = self._distribute_assignments(assignments)
                self._merge_assignment_summary(assignment_summary)
                if self._open_daily_summary is not None:
                    meta = getattr(self.scheduler, '_last_solver_meta', {}) if self.scheduler else {}
                    self._open_daily_summary['solver_wall_time_seconds'] = round(solve_wall_time, 6)
                    self._open_daily_summary['solver_status'] = str(meta.get('status', 'UNKNOWN'))
            except Exception as e:
                print(f"\n{'='*80}\nERROR during daily scheduling at {seconds_to_hours(self.env.now):.1f}h: {e}\n{'='*80}\n")
                logger.error(f"Daily scheduling failed at {seconds_to_hours(self.env.now):.1f}h: {e}", exc_info=True)
                if self._open_daily_summary is not None:
                    self._open_daily_summary['solver_failed'] = True
                    self._open_daily_summary['solver_status'] = f"FAILED_{type(e).__name__}"
                # Keep all tasks for retry next cycle (don't lose them)
                logger.warning(
                    f"Scheduling failed — keeping {len(self.unscheduled_tasks)} tasks "
                    f"for next scheduling cycle"
                )
            
    def _collect_pending_tasks(self):
        """
        Collect all pending tasks from unscheduled pool.
        
        Returns:
            List of tasks waiting to be scheduled
        """
        return list(self.unscheduled_tasks)
    
    def _get_active_cases(self):
        """
        Get all active cases (not completed).
        
        Returns:
            List of active Case objects
        """
        from ..entities.case import CaseStatus
        return [case for case in self.state.cases.values() if case.status != CaseStatus.COMPLETED]
    
    def _distribute_assignments(self, assignments):
        """
        Distribute optimized task assignments to resource queues.

        Priority is a composite tuple so that:
          1. Mentoring tasks (type_rank=0) are dequeued before standard
             tasks (type_rank=1) at similar start times.
          2. Within each type, earlier ``scheduled_start_time`` wins.
          3. Ties are broken by the task's configured ``priority`` (lower = first).

        Args:
            assignments: Dict mapping resource_id to list of tasks
        """
        assignment_summary = self._summarize_assignments(assignments)
        # _daily_scheduler_process overriding context.pending_tasks).
        requeued: List[Task] = []
        dropped: List[Task] = []
        # Keep only deferred tasks for the next scheduling cycle.
        # This is finalized after processing all assignment buckets.
        deferred_tasks = assignments.get(DUMMY_RESOURCE_ID, [])
        for resource_id, tasks in assignments.items():
            current_len = None
            # --- Handle deferred tasks (assigned to dummy by solver) --- 
            if resource_id == DUMMY_RESOURCE_ID:
                if not deferred_tasks:
                    continue
                for task in deferred_tasks:
                    task.reset_for_rescheduling()  # helper method to reset task state for re-planning and increment defer count
                    if task.defer_count > self.max_task_deferrals: # exceeded max deferrals, drop the task
                        dropped.append(task)
                    else: # Reset task state so it can be re-planned tomorrow
                        requeued.append(task)
                # Track deferred/dropped counts
                self._total_deferred_tasks += len(requeued)
                self._total_dropped_tasks += len(dropped)
                if dropped:
                    for task in dropped:
                        self.state.cancel_task(task.id, self.env.now)  # Drop: exceeded max deferrals
                
            # --- Handle tasks assigned to UNKOWN/INVALID resources ---             
            elif resource_id not in self.resource_stores:
                logger.warning(f"Unknown resource {resource_id} in assignments")
                continue

            # --- Handle normal & mentoring tasks assigned to actual resources --- 
            else:
                for task in tasks:
                    if self.log_writer is not None:
                        self.log_writer.log_task_queued(
                            case_id=task.case_id,
                            task_id=task.id,
                            activity_name=task.activity_name,
                            resource_id=resource_id,
                            timestamp=round(seconds_to_hours(self.env.now), 4),
                            sim_datetime=self.time_converter.sim_time_to_datetime(self.env.now) if self.time_converter else None,
                            task_type=task.task_type.value if task.task_type else "unknown",
                            mentor=task.mentor_resource_id if task.is_mentoring_task() else None,
                            mentee=task.mentee_resource_id if task.is_mentoring_task() else None,
                            mentoring_against_bottleneck=task.mentoring_against_bottleneck if task.is_mentoring_task() else None, 
                            emergency_mentoring=task.is_emergency_mentoring if task.is_mentoring_task() else None,
                            bootstrap_assignment=task.bootstrap_assignment,
                            bootstrap_onboarding_seconds=task.bootstrap_onboarding_seconds if task.bootstrap_assignment else 0,
                            estimated_duration=task.estimated_duration if task.estimated_duration else 0,
                        )
                    if task.is_mentoring_task():
                        if task.mentee_resource_id != resource_id:
                            print(f"  WARNING ENGINE: Task {task.id} is a mentoring task but assigned to {resource_id} instead of mentee {task.mentee_resource_id}")
                    
                    if task.is_emergency_mentoring:
                        type_rank = 0
                    elif task.bootstrap_assignment:
                        type_rank = 1
                    elif task.is_mentoring_task():
                        type_rank = 2
                    else:
                        type_rank = 3

                    # Deferral urgency: higher defer_count -> smaller value -> dequeued sooner
                    deferral_priority = self.max_task_deferrals - task.defer_count

                    # Composite priority tuple — PriorityStore picks the smallest first:
                    #   1st: tier (emergency mentoring -> mentoring -> normal)
                    #   2nd: deferral urgency (more deferrals -> higher urgency)
                    #   3rd: monotonic sequence — guarantees uniqueness so Task.__lt__ is never invoked
                    self._queue_sequence += 1
                    composite_priority = (type_rank,
                                          deferral_priority,
                                          self._queue_sequence
                                          )
                    self.resource_stores[resource_id].put((composite_priority, task))

                    # Push a matching sentinel into the MENTOR's queue so the mentor's worker can log/react when it reaches this slot.
                    # The sentinel carries the same composite_priority so it sits at the same position in the mentor's queue as the
                    # mentoring task sits in the mentee's queue.
                    if task.is_mentoring_task() and task.mentor_resource_id in self.resource_stores:
                        self._queue_sequence += 1
                        sentinel = _MentorSentinel(
                            mentor_resource_id=task.mentor_resource_id,
                            mentee_task_id=task.id,
                            activity_name=task.activity_name,
                            sequence=self._queue_sequence,
                        )
                        sentinel_priority = (type_rank, deferral_priority, self._queue_sequence)
                        self.resource_stores[task.mentor_resource_id].put((sentinel_priority, sentinel))
            
                # Update max queue length high-water mark for this resource
                current_len = len(self.resource_stores[resource_id].items)
                
            if current_len is not None and current_len > self.max_queue_lengths.get(resource_id, 0):
                self.max_queue_lengths[resource_id] = current_len

        # Clear all assigned tasks from unscheduled pool; keep only deferred ones.
        # This must happen even when there are zero dummy assignments.
        self.unscheduled_tasks = requeued

        total_assigned = sum(
            len(tasks) for rid, tasks in assignments.items() if rid != DUMMY_RESOURCE_ID
        )
        log_parts = [f"Daily scheduling complete at {seconds_to_hours(self.env.now):.1f}h - assigned {total_assigned} tasks"]
        if requeued:
            log_parts.append(f", {len(requeued)} deferred to next day")
        if dropped:
            log_parts.append(f", {len(dropped)} dropped (max deferrals exceeded)")
        logger.info("".join(log_parts))
        return assignment_summary

    # ========== END BATCH SCHEDULING METHODS ==========
    
    def run(self, until: Optional[float] = None) -> Dict[str, Any]:
        """
        Run the simulation until completion or time limit.
        
        Args:
            until: Optional simulation time limit (overrides max_simulation_time)
            
        Returns:
            Dictionary with simulation statistics
        """
        # Determine time limit
        time_limit = until if until is not None else self.max_simulation_time
        
        logger.info(f"Starting SimPy simulation (time_limit={time_limit})")
        
        # Run SimPy environment
        try:
            if time_limit is not None:
                self.env.run(until=time_limit)
            else:
                # Run until all processes complete
                self.env.run()
        except Exception as e:
            logger.error(f"Simulation error: {e}", exc_info=True)
            raise
        
        # Update progress bar to completion if enabled
        if self.progress_bar is not None:
            # Calculate final day count
            final_day = self.env.now // SECONDS_PER_DAY
            # Handle both tqdm (has .n attribute) and custom progress bars (have .completed_items)
            current_progress = getattr(self.progress_bar, 'n', None) or getattr(self.progress_bar, 'completed_items', 0)
            remaining_days = final_day - int(current_progress)
            if remaining_days > 0:
                self.progress_bar.update(remaining_days)
                
        logger.info(f"Simulation completed at time {self.env.now}s ({seconds_to_hours(self.env.now):.1f}h)")
        logger.info(f"Cases completed: {len(self.state.get_finished_cases())}/{len(self.state.get_all_cases())}")
        
        # Save experience tracker data if enabled
        if self.experience_tracker is not None:
            tracker_df = self.experience_tracker.to_dataframe()
            if not tracker_df.empty:
                logger.info(f"Experience tracker recorded {len(tracker_df)} snapshots")
            else:
                logger.warning("Experience tracker has no data recorded")
        
        # Calculate queue statistics
        total_tasks_in_queues = sum(len(store.items) for store in self.resource_stores.values())
        max_queue_resource = max(self.max_queue_lengths.items(), key=lambda x: x[1]) if self.max_queue_lengths else (None, 0)
        state_statistics = self.state.get_statistics()

        # Build statistics dictionary
        stats = {
            'simulation_time': seconds_to_hours(self.env.now),
            'total_cases': state_statistics["total_cases"],
            'cases_completed': state_statistics["completed_cases"],
            'cases_active': state_statistics["active_cases"],
            'case_completion_rate': state_statistics["completed_cases"] / state_statistics["total_cases"] if state_statistics["total_cases"] > 0 else 0.0,
            'total_tasks': state_statistics["total_tasks"],
            'tasks_completed': state_statistics["completed_tasks"],
            'tasks_active': state_statistics["active_tasks"],
            'task_completion_rate': state_statistics["completed_tasks"] / state_statistics["total_tasks"] if state_statistics["total_tasks"] > 0 else 0.0,
            'experience_tracker': self.experience_tracker,
            'queue_stats': {
                'tasks_remaining_in_queues': total_tasks_in_queues,
                'max_queue_length': max_queue_resource[1],
                'max_queue_resource_id': max_queue_resource[0],
                'max_queue_lengths_per_resource': dict(self.max_queue_lengths)
            },
            'unfinished_work': {
                'cases_not_started': state_statistics["total_cases"] - state_statistics["active_cases"] - state_statistics["completed_cases"] - state_statistics["cancelled_cases"],
                'cases_in_progress': state_statistics["active_cases"],
                'cases_cancelled': state_statistics["cancelled_cases"],
                'total_unfinished_cases': state_statistics["total_cases"] - state_statistics["completed_cases"],
                'tasks_unscheduled': state_statistics["total_tasks"] - state_statistics["queued_tasks"] - state_statistics["completed_tasks"] - state_statistics["active_tasks"] - state_statistics["cancelled_tasks"],
                'tasks_in_queues': state_statistics["queued_tasks"],
                'tasks_in_execution': state_statistics["active_tasks"],
                'tasks_cancelled': state_statistics["cancelled_tasks"],
                'total_unfinished_tasks': state_statistics["total_tasks"] - state_statistics["completed_tasks"] - state_statistics["cancelled_tasks"],
                'cancelled_tasks_per_activity': state_statistics["cancelled_tasks_per_activity"]
            },
            'drain_stats': {
                'total_drained_tasks': self._total_drained_tasks,
                'drain_days': self._total_drain_days,
                'total_deferred_tasks': self._total_deferred_tasks,
                'total_dropped_tasks': self._total_dropped_tasks,
                'tasks_in_unscheduled_pool': len(self.unscheduled_tasks),
            },
            'bootstrap_stats': {
                'enabled': bool(self.bootstrap_enabled),
                'bootstrap_assignments_executed': self._bootstrap_assignments_executed,
                'bootstrap_total_penalty_hours': seconds_to_hours(self._bootstrap_total_penalty_seconds),
                'bootstrap_activities_executed': sorted(self._bootstrap_activities_executed),
            },
        }
        
        # Print unfinished work statistics using stats dict -> DICT MADNESS
        print("\n=== Statistics ===")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        print(f"  {sub_key}:")
                        for sub_sub_key, sub_sub_value in sub_value.items():
                            print(f"    {sub_sub_key}: {sub_sub_value}")
                    else:
                        print(f"  {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        # Add overtime statistics to stats dict if working hours enabled
        if self.enable_working_hours:
            stats['overtime_stats'] = {
                'total_overtime_hours': sum(ov[0] for ov in self.overtime_hours.values()),
                'overtime_per_resource': dict(self.overtime_hours),
                'max_overtime_hours': max(ov[0] for ov in self.overtime_hours.values()) if self.overtime_hours else 0.0,
                'max_overtime_resource_id': max(self.overtime_hours.items(), key=lambda x: x[1][0])[0] if self.overtime_hours else None,
                'max_overtime_resource_hours': max(ov[0] for ov in self.overtime_hours.values()) if self.overtime_hours else 0.0,
                'max_single_overtime_instance_hours': max(self.overtime_hours.items(), key=lambda x: x[1][1])[1] if self.overtime_hours else 0.0,
                'max_single_overtime_instance_resource_id': max(self.overtime_hours.items(), key=lambda x: x[1][1])[0] if self.overtime_hours else None,
                'max_single_overtime_instance_task_id': max(self.overtime_hours.items(), key=lambda x: x[1][1])[0] if self.overtime_hours else None
            }
            
            # Log overtime statistics using stats dict
            if stats['overtime_stats']['total_overtime_hours'] > 0:
                logger.info(f"Total overtime hours: {stats['overtime_stats']['total_overtime_hours']:.2f}h")
                logger.info(f"Max overtime: {stats['overtime_stats']['max_overtime_hours']:.2f}h (resource: {stats['overtime_stats']['max_overtime_resource_id']})")

        # Persist final partial day summary (if an open day exists) so no KPI data is lost
        # when simulation stops before the next scheduling-cycle drain point.
        if self._open_daily_summary is not None:
            incomplete_now: Dict[str, int] = defaultdict(int)
            for store in self.resource_stores.values():
                for _priority, queued_item in store.items:
                    if isinstance(queued_item, _MentorSentinel):
                        continue
                    incomplete_now[str(queued_item.activity_name)] += 1
            self._finalize_open_daily_summary(
                incomplete_by_activity=dict(incomplete_now),
                dropped_from_drain_by_activity={},
                is_partial_day=True,
            )
        if self.daily_summary_logger is not None:
            self.daily_summary_logger.close()
        
        return stats
    
    def get_state(self) -> SimulationState:
        """Get the current simulation state."""
        return self.state
    
    def get_current_time(self) -> float:
        """Get the current simulation time."""
        return self.env.now
