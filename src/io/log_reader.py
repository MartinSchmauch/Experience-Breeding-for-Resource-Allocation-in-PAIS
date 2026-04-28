"""Read historical event logs from XES or CSV files."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import pm4py
import warnings
import logging

logger = logging.getLogger(__name__)

# Suppress pm4py rustxes warning (non-critical deprecation notice)
warnings.filterwarnings('ignore', message='.*rustxes.*', category=UserWarning)


class EventLogReader:
    """
    Read historical event logs.
    
    Supports XES and CSV formats.
    """
    
    def __init__(self):
        """Initialize event log reader."""
        pass
    
    def read_xes(self, filepath: Path) -> pd.DataFrame:
        """
        Read XES event log file.
        
        Args:
            filepath: Path to XES file
            
        Returns:
            DataFrame with event log
        """
        log = pm4py.read_xes(str(filepath))
        df = pm4py.convert_to_dataframe(log)
        
        # Ensure timestamps are datetime
        if 'time:timestamp' in df.columns:
            df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
        
        return df
    
    def read_csv(
        self,
        filepath: Path,
        timestamp_column: Optional[str] = None,
        parse_dates: bool = True
    ) -> pd.DataFrame:
        """
        Read CSV event log file.
        
        Args:
            filepath: Path to CSV file
            timestamp_column: Column name for timestamps
            parse_dates: Whether to parse timestamp columns
            
        Returns:
            DataFrame with event log
        """
        if parse_dates and timestamp_column:
            df = pd.read_csv(filepath, parse_dates=[timestamp_column])
        else:
            df = pd.read_csv(filepath)
        
        return df
    
    def read_auto(self, filepath: Path, **kwargs) -> pd.DataFrame:
        """
        Automatically detect format and read log.
        
        Args:
            filepath: Path to log file
            **kwargs: Additional arguments for read methods
            
        Returns:
            DataFrame with event log
        """
        filepath = Path(filepath)
        
        if filepath.suffix.lower() == '.xes':
            return self.read_xes(filepath)
        elif filepath.suffix.lower() == '.csv':
            return self.read_csv(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def standardize_columns(
        self,
        df: pd.DataFrame,
        case_col: str = 'case:concept:name',
        activity_col: str = 'concept:name',
        timestamp_col: str = 'time:timestamp',
        resource_col: str = 'org:resource',
        lifecycle_col: str = 'lifecycle:transition'
    ) -> pd.DataFrame:
        """
        Standardize column names to internal format.
        
        Args:
            df: Input dataframe
            case_col: Source column for case ID
            activity_col: Source column for activity
            timestamp_col: Source column for timestamp
            resource_col: Source column for resource
            lifecycle_col: Source column for lifecycle
            
        Returns:
            DataFrame with standardized columns
        """
        column_mapping = {}
        
        if case_col in df.columns:
            column_mapping[case_col] = 'case_id'
        if activity_col in df.columns:
            column_mapping[activity_col] = 'activity'
        if timestamp_col in df.columns:
            column_mapping[timestamp_col] = 'timestamp'
        if resource_col in df.columns:
            column_mapping[resource_col] = 'resource'
        if lifecycle_col in df.columns:
            column_mapping[lifecycle_col] = 'lifecycle'
        
        df_standardized = df.rename(columns=column_mapping)
        return df_standardized
    
    def preprocess_for_simulation(
        self,
        filepath: Path,
        filter_prefix: str = 'W_',
        context_attributes: Optional[list] = None,
        compute_durations: bool = False,
        return_segments: bool = False,
        duration_assignment_strategy: str = 'longest_resource'
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Preprocess XES log for simulation initialization.
        
        Handles two use cases:
        1. Transition model training: Event log with XES columns (compute_durations=False)
        2. Experience store building: Timeline with durations (compute_durations=True)
        
        Args:
            filepath: Path to XES file
            filter_prefix: Activity name prefix to keep (e.g., 'W_' for work activities)
            context_attributes: Case attributes to include (e.g., ['case:LoanGoal', 'case:ApplicationType'])
            compute_durations: If True, match start/complete and compute service durations
            return_segments: If True and compute_durations=True, return (timeline_df, segments_df) tuple
            duration_assignment_strategy: How to map a multi-resource occurrence to one timeline row.
                Options:
                - 'longest_resource': assign to resource with highest active contribution (default)
                - 'completing_resource': assign to resource that completed the occurrence
                - 'segment_level': keep one row per segment (legacy behavior)
            
        Returns:
            DataFrame with preprocessed events/timeline, or tuple of (timeline_df, segments_df) if return_segments=True
        """
        if context_attributes is None:
            context_attributes = ['case:LoanGoal', 'case:ApplicationType']
        
        # Read XES log
        log_df = self.read_xes(filepath)
        
        # Filter activities
        if filter_prefix:
            log_df = log_df[log_df['concept:name'].str.startswith(filter_prefix)].copy()
        # rename "W_Shortened completion " to "W_Shortened completion"
        log_df['concept:name'] = log_df['concept:name'].str.strip()
        # exclude activities "W_Shortened completion" and "W_Personal Loan collection"
        # log_df = log_df[~log_df['concept:name'].isin(["W_Shortened completion", "W_Personal Loan collection"])].copy()
        
        if not compute_durations:
            # Simple preprocessing for transition models
            # Just ensure timestamp is datetime and return
            log_df['time:timestamp'] = pd.to_datetime(log_df['time:timestamp'])
            return log_df
        else:
            # Full preprocessing for experience store
            timeline_df, segments_df = self._compute_service_durations(
                log_df,
                context_attributes,
                duration_assignment_strategy=duration_assignment_strategy,
            )
            if return_segments:
                return timeline_df, segments_df
            return timeline_df

    def _extract_work_segments(
        self,
        log_df: pd.DataFrame,
        context_attributes: Optional[list] = None,
        min_segment_seconds: float = 30.0
    ) -> pd.DataFrame:
        """
        Extract active work segments from lifecycle events using a state machine.
        
        Processes start/suspend/resume/complete lifecycle transitions to decompose
        each task occurrence into individual active work segments. Segments shorter
        than min_segment_seconds are discarded.
        
        State machine transitions:
            IDLE --start--> ACTIVE
            ACTIVE --suspend--> SUSPENDED
            SUSPENDED --resume--> ACTIVE
            ACTIVE --complete--> IDLE
            IDLE --complete--> IDLE (instant activity)
        
        Args:
            log_df: Event log DataFrame (filtered to relevant activities)
            context_attributes: Case attributes to include in segments
            min_segment_seconds: Minimum segment duration in seconds (segments
                                 shorter than this are discarded as negligible)
            
        Returns:
            Segments DataFrame with columns:
            - case_id, activity_name, occurrence, resource_id
            - segment_start, segment_end, segment_duration_seconds
            - is_completing_segment (True for the segment that ends with complete)
            - context attributes
        """
        if context_attributes is None:
            context_attributes = ['case:LoanGoal', 'case:ApplicationType']
        
        # Filter to relevant lifecycle transitions
        relevant_transitions = ['start', 'suspend', 'resume', 'complete']
        df_lc = log_df[log_df['lifecycle:transition'].isin(relevant_transitions)].copy()
        df_lc['time:timestamp'] = pd.to_datetime(df_lc['time:timestamp'])
        
        # Sort chronologically within each (case, activity) group
        df_lc = df_lc.sort_values(
            ['case:concept:name', 'concept:name', 'time:timestamp']
        ).reset_index(drop=True)
        
        segments = []
        
        # Process each (case, activity) group with a state machine
        for (case_id, activity_name), group in df_lc.groupby(
            ['case:concept:name', 'concept:name'], sort=False
        ):
            # Extract context attributes from first row of group
            context_values = {}
            for attr in context_attributes:
                if attr in group.columns:
                    context_values[attr] = group.iloc[0][attr]
            
            state = 'IDLE'
            occurrence = 0
            segment_start = None
            segment_resource = None
            resumed_pending_start = False  # True when ACTIVE was entered via resume (not yet confirmed by start)
            
            for _, event in group.iterrows():
                transition = event['lifecycle:transition']
                timestamp = event['time:timestamp']
                resource = event['org:resource']
                
                if transition == 'start':
                    if state == 'ACTIVE' and resumed_pending_start:
                        # Resume→Start pattern: the resume was just a "pick up" signal,
                        # actual work begins at this start event. Discard the resume→start
                        # gap by moving segment_start forward. Same occurrence.
                        logger.debug(
                            f"Resume→Start pattern for case={case_id}, "
                            f"activity={activity_name}. Adjusting segment start from "
                            f"{segment_start} to {timestamp}."
                        )
                        segment_start = timestamp
                        segment_resource = resource
                        resumed_pending_start = False
                    elif state == 'ACTIVE':
                        # Unexpected: start while active (not from resume) — close current segment, start new occurrence
                        logger.debug(
                            f"Unexpected 'start' while ACTIVE for case={case_id}, "
                            f"activity={activity_name}. Closing current segment."
                        )
                        segments.append(self._make_segment(
                            case_id, activity_name, occurrence,
                            segment_resource, segment_start, timestamp,
                            is_completing=False, context_values=context_values
                        ))
                        # Begin new occurrence
                        occurrence += 1
                        state = 'ACTIVE'
                        segment_start = timestamp
                        segment_resource = resource
                        resumed_pending_start = False
                    else:
                        # Normal start from IDLE or SUSPENDED
                        occurrence += 1
                        state = 'ACTIVE'
                        segment_start = timestamp
                        segment_resource = resource
                        resumed_pending_start = False
                
                elif transition == 'suspend':
                    if state == 'ACTIVE':
                        # Close active segment
                        resumed_pending_start = False
                        segments.append(self._make_segment(
                            case_id, activity_name, occurrence,
                            segment_resource, segment_start, timestamp,
                            is_completing=False, context_values=context_values
                        ))
                        state = 'SUSPENDED'
                        segment_start = None
                        segment_resource = None
                    else:
                        logger.debug(
                            f"Unexpected 'suspend' while {state} for case={case_id}, "
                            f"activity={activity_name}. Ignoring."
                        )
                
                elif transition == 'resume':
                    if state == 'SUSPENDED':
                        # Open new active segment (potentially different resource)
                        state = 'ACTIVE'
                        segment_start = timestamp
                        segment_resource = resource
                        resumed_pending_start = True  # Mark: if a 'start' follows, use that as real start
                    else:
                        logger.debug(
                            f"Unexpected 'resume' while {state} for case={case_id}, "
                            f"activity={activity_name}. Ignoring."
                        )
                
                elif transition == 'complete':
                    if state == 'ACTIVE':
                        # Close final segment — this is the completing segment
                        resumed_pending_start = False
                        segments.append(self._make_segment(
                            case_id, activity_name, occurrence,
                            segment_resource, segment_start, timestamp,
                            is_completing=True, context_values=context_values
                        ))
                        state = 'IDLE'
                        segment_start = None
                        segment_resource = None
                    elif state == 'SUSPENDED':
                        # Complete from suspended — treat as instant completion
                        # (the resource that completes does so without a resume)
                        segments.append(self._make_segment(
                            case_id, activity_name, occurrence,
                            resource, timestamp, timestamp,
                            is_completing=True, context_values=context_values
                        ))
                        state = 'IDLE'
                        segment_start = None
                        segment_resource = None
                    elif state == 'IDLE':
                        # Complete without start — instant activity
                        occurrence += 1
                        segments.append(self._make_segment(
                            case_id, activity_name, occurrence,
                            resource, timestamp, timestamp,
                            is_completing=True, context_values=context_values
                        ))
                    else:
                        logger.debug(
                            f"Unexpected 'complete' while {state} for case={case_id}, "
                            f"activity={activity_name}. Ignoring."
                        )
            
            # Handle case where group ends in ACTIVE state (no complete event)
            if state == 'ACTIVE' and segment_start is not None:
                logger.debug(
                    f"Unclosed segment for case={case_id}, activity={activity_name}. "
                    f"Event sequence ends without 'complete'."
                )
                # Don't include — incomplete tasks should not be in the experience store
        
        if not segments:
            # Return empty DataFrame with expected columns
            cols = ['case_id', 'activity_name', 'occurrence', 'resource_id',
                    'segment_start', 'segment_end', 'segment_duration_seconds',
                    'is_completing_segment']
            for attr in context_attributes:
                cols.append(attr.replace('case:', '').replace(':', '_'))
            return pd.DataFrame(columns=cols)
        
        segments_df = pd.DataFrame(segments)
        
        # Compute segment durations
        segments_df['segment_duration_seconds'] = (
            segments_df['segment_end'] - segments_df['segment_start']
        ).dt.total_seconds()
        
        # Filter out negligible segments (≤ min_segment_seconds)
        # But keep instant activities (completing segments with 0 duration from IDLE→complete)
        is_instant = (
            segments_df['is_completing_segment'] & 
            (segments_df['segment_duration_seconds'] == 0)
        )
        is_meaningful = segments_df['segment_duration_seconds'] > min_segment_seconds
        segments_df = segments_df[
            # is_instant | # TODO: for now, we want to keep all segments including short ones, as they may be relevant for resource attribution and timeline construction. We can revisit this threshold later if needed.
            is_meaningful].copy()
        
        # Filter negative durations (shouldn't happen, but safety check)
        segments_df = segments_df[segments_df['segment_duration_seconds'] >= 0].copy()
        
        logger.info(
            f"   Extracted {len(segments_df):,} work segments from "
            f"{segments_df[['case_id', 'activity_name', 'occurrence']].drop_duplicates().shape[0]:,} "
            f"task occurrences"
        )
        
        return segments_df
    
    @staticmethod
    def _make_segment(
        case_id: str,
        activity_name: str,
        occurrence: int,
        resource_id: str,
        segment_start: pd.Timestamp,
        segment_end: pd.Timestamp,
        is_completing: bool,
        context_values: dict
    ) -> dict:
        """Create a segment record dict."""
        record = {
            'case_id': case_id,
            'activity_name': activity_name,
            'occurrence': occurrence,
            'resource_id': resource_id,
            'segment_start': segment_start,
            'segment_end': segment_end,
            'is_completing_segment': is_completing,
        }
        # Add context attributes with clean names
        for attr, value in context_values.items():
            clean_name = attr.replace('case:', '').replace(':', '_')
            record[clean_name] = value
        return record
    
    def _aggregate_segments_to_timeline(
        self,
        segments_df: pd.DataFrame,
        context_attributes: Optional[list] = None,
        duration_assignment_strategy: str = 'longest_resource'
    ) -> pd.DataFrame:
        """
        Build a timeline from active work segments.

        Default behavior is occurrence-level aggregation with resource assignment
        based on the longest active contributor.
        
        Args:
            segments_df: Segments DataFrame from _extract_work_segments()
            context_attributes: Case attributes (needed for clean name mapping)
            
        Returns:
            Timeline DataFrame with columns:
            - case_id, activity_name, resource_id
            - start_timestamp, complete_timestamp, duration_seconds
            - segment_count, resource_count, active_ratio
            - context attributes
        """
        if context_attributes is None:
            context_attributes = ['case:LoanGoal', 'case:ApplicationType']
        
        if segments_df.empty:
            cols = ['case_id', 'activity_name', 'resource_id',
                    'start_timestamp', 'complete_timestamp', 'duration_seconds',
                    'segment_count', 'resource_count', 'active_ratio']
            for attr in context_attributes:
                cols.append(attr.replace('case:', '').replace(':', '_'))
            return pd.DataFrame(columns=cols)

        strategy_aliases = {
            'segment': 'segment_level',
            'segment_level': 'segment_level',
            'completing': 'completing_resource',
            'completing_resource': 'completing_resource',
            'longest': 'longest_resource',
            'longest_resource': 'longest_resource',
        }
        strategy = strategy_aliases.get(duration_assignment_strategy)
        if strategy is None:
            raise ValueError(
                "duration_assignment_strategy must be one of "
                "{'longest_resource', 'completing_resource', 'segment_level'}"
            )
        
        # Clean context attribute names
        context_clean = [attr.replace('case:', '').replace(':', '_') for attr in context_attributes]

        group_cols = ['case_id', 'activity_name', 'occurrence']

        if strategy == 'segment_level':
            # Legacy behavior: keep one row per segment.
            occ_meta = (
                segments_df
                .groupby(group_cols, sort=False)
                .apply(
                    lambda g: pd.Series(
                        {
                            'segment_count': len(g),
                            'resource_count': g['resource_id'].nunique(),
                            'active_ratio': round(
                                (
                                    g['segment_duration_seconds'].sum()
                                    / (g['segment_end'].max() - g['segment_start'].min()).total_seconds()
                                )
                                if (g['segment_end'].max() - g['segment_start'].min()).total_seconds() > 0
                                else 1.0,
                                4,
                            ),
                        }
                    ),
                    include_groups=False,
                )
                .reset_index()
            )

            timeline_df = segments_df.rename(
                columns={
                    'segment_start': 'start_timestamp',
                    'segment_end': 'complete_timestamp',
                    'segment_duration_seconds': 'duration_seconds',
                }
            ).merge(occ_meta, on=group_cols, how='left')
        else:
            occ_meta = (
                segments_df
                .groupby(group_cols, sort=False)
                .agg(
                    start_timestamp=('segment_start', 'min'),
                    complete_timestamp=('segment_end', 'max'),
                    segment_count=('segment_duration_seconds', 'size'),
                    resource_count=('resource_id', 'nunique'),
                    total_active_seconds=('segment_duration_seconds', 'sum'),
                )
                .reset_index()
            )

            elapsed_seconds = (
                occ_meta['complete_timestamp'] - occ_meta['start_timestamp']
            ).dt.total_seconds()
            occ_meta['active_ratio'] = np.where(
                elapsed_seconds > 0,
                (occ_meta['total_active_seconds'] / elapsed_seconds).round(4),
                1.0,
            )

            contributor_stats = (
                segments_df
                .groupby(group_cols + ['resource_id'], sort=False)
                .agg(
                    resource_active_seconds=('segment_duration_seconds', 'sum'),
                    last_segment_end=('segment_end', 'max'),
                )
                .reset_index()
            )

            if strategy == 'longest_resource':
                selected = (
                    contributor_stats
                    .sort_values(
                        group_cols + ['resource_active_seconds', 'last_segment_end', 'resource_id'],
                        ascending=[True, True, True, False, False, False],
                    )
                    .drop_duplicates(subset=group_cols, keep='first')
                    [group_cols + ['resource_id', 'resource_active_seconds']]
                )
            else:
                completing_candidates = (
                    segments_df[segments_df['is_completing_segment']]
                    .sort_values(group_cols + ['segment_end'], ascending=[True, True, True, False])
                    .drop_duplicates(subset=group_cols, keep='first')
                    [group_cols + ['resource_id']]
                )
                selected = completing_candidates.merge(
                    contributor_stats[group_cols + ['resource_id', 'resource_active_seconds']],
                    on=group_cols + ['resource_id'],
                    how='left',
                )
                if len(selected) < len(occ_meta):
                    fallback = (
                        contributor_stats
                        .sort_values(
                            group_cols + ['resource_active_seconds', 'last_segment_end', 'resource_id'],
                            ascending=[True, True, True, False, False, False],
                        )
                        .drop_duplicates(subset=group_cols, keep='first')
                        [group_cols + ['resource_id', 'resource_active_seconds']]
                    )
                    selected = occ_meta[group_cols].merge(
                        selected,
                        on=group_cols,
                        how='left',
                    ).merge(
                        fallback,
                        on=group_cols,
                        how='left',
                        suffixes=('', '_fallback'),
                    )
                    selected['resource_id'] = selected['resource_id'].fillna(selected['resource_id_fallback'])
                    selected['resource_active_seconds'] = selected['resource_active_seconds'].fillna(
                        selected['resource_active_seconds_fallback']
                    )
                    selected = selected[group_cols + ['resource_id', 'resource_active_seconds']]

            timeline_df = occ_meta.merge(selected, on=group_cols, how='left')
            timeline_df = timeline_df.rename(columns={'resource_active_seconds': 'duration_seconds'})

            context_cols = [c for c in context_clean if c in segments_df.columns]
            if context_cols:
                context_df = (
                    segments_df[group_cols + context_cols]
                    .drop_duplicates(subset=group_cols, keep='first')
                )
                timeline_df = timeline_df.merge(context_df, on=group_cols, how='left')

        # Keep timeline schema stable for downstream usage.
        keep_cols = [
            'case_id',
            'activity_name',
            'resource_id',
            'start_timestamp',
            'complete_timestamp',
            'duration_seconds',
            'segment_count',
            'resource_count',
            'active_ratio',
        ] + [c for c in context_clean if c in timeline_df.columns]
        timeline_df = timeline_df[keep_cols].copy()
        
        logger.info(
            f"   Aggregated into {len(timeline_df):,} timeline entries "
            f"using strategy='{strategy}' "
            f"(multi-resource tasks: {(timeline_df['resource_count'] > 1).sum():,}, "
            f"multi-segment tasks: {(timeline_df['segment_count'] > 1).sum():,})"
        )
        
        return timeline_df
    
    def _compute_service_durations(
        self,
        log_df: pd.DataFrame,
        context_attributes: Optional[list] = None,
        duration_assignment_strategy: str = 'longest_resource'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Match lifecycle events and compute active service durations.
        
        Uses a state machine to process start/suspend/resume/complete lifecycle
        transitions and compute actual active working time (excluding suspension
        periods). Work segments ≤30 seconds are discarded as negligible.
        
        Timeline rows are occurrence-level by default and assigned to the
        longest-contributing resource (configurable).
        
        Args:
            log_df: Event log DataFrame (filtered to relevant activities)
            context_attributes: Case attributes to include in timeline
            duration_assignment_strategy: Rule for selecting which resource and
                duration to use for each occurrence-level timeline row.
            
        Returns:
            Tuple of (timeline_df, segments_df):
            
            timeline_df columns:
            - case_id, activity_name, resource_id
            - start_timestamp, complete_timestamp, duration_seconds (segment active time)
            - segment_count, resource_count, active_ratio
            - context attributes (e.g., LoanGoal, ApplicationType)
            
            segments_df columns:
            - case_id, activity_name, occurrence, resource_id
            - segment_start, segment_end, segment_duration_seconds
            - is_completing_segment
            - context attributes
        """
        if context_attributes is None:
            context_attributes = ['case:LoanGoal', 'case:ApplicationType']
        
        # Step 1: Extract individual work segments via state machine
        segments_df = self._extract_work_segments(
            log_df, context_attributes, min_segment_seconds=0.0
        )
        
        # Step 2: Aggregate segments into timeline (one row per task)
        timeline_df = self._aggregate_segments_to_timeline(
            segments_df,
            context_attributes,
            duration_assignment_strategy=duration_assignment_strategy,
        )
        
        # Select final columns for timeline
        final_cols = ['case_id', 'activity_name', 'resource_id',
                      'start_timestamp', 'complete_timestamp', 'duration_seconds',
                      'segment_count', 'resource_count', 'active_ratio']
        
        # Add context columns that exist
        for attr in context_attributes:
            clean_name = attr.replace('case:', '').replace(':', '_')
            if clean_name in timeline_df.columns:
                final_cols.append(clean_name)
        
        # Only select columns that exist (safety for empty DataFrames)
        final_cols = [c for c in final_cols if c in timeline_df.columns]
        
        return timeline_df[final_cols], segments_df
