"""
Streamlit visualization helpers for experience breeding.

Provides ready-to-use plotting functions for the Streamlit dashboard.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, List
from pathlib import Path


def plot_learning_curve(
    df: pd.DataFrame,
    resource_id: Optional[str] = None,
    activity_name: Optional[str] = None,
    title: Optional[str] = None
) -> go.Figure:
    """
    Plot experience level learning curve.
    
    Args:
        df: DataFrame from ExperienceLevelTracker.to_dataframe()
        resource_id: Filter by resource (optional)
        activity_name: Filter by activity (optional)
        title: Custom title (optional)
        
    Returns:
        Plotly figure
    """
    # Filter data
    plot_df = df.copy()
    if resource_id:
        plot_df = plot_df[plot_df['resource_id'] == resource_id]
    if activity_name:
        plot_df = plot_df[plot_df['activity_name'] == activity_name]
    
    # Generate title
    if title is None:
        parts = ["Learning Curve"]
        if resource_id:
            parts.append(f"Resource: {resource_id}")
        if activity_name:
            parts.append(f"Activity: {activity_name}")
        title = " - ".join(parts)
    
    # Create figure
    fig = px.line(
        plot_df,
        x='repetition_count',
        y='experience_level',
        color='resource_id' if not resource_id else None,
        title=title,
        labels={
            'repetition_count': 'Number of Repetitions',
            'experience_level': 'Experience Level (0-100)',
            'resource_id': 'Resource'
        },
        markers=True
    )
    
    fig.update_layout(
        hovermode='x unified',
        showlegend=True,
        height=500
    )
    
    # Add target level line at 80
    fig.add_hline(
        y=80, 
        line_dash="dash", 
        line_color="green",
        annotation_text="Expert Level (80)",
        annotation_position="right"
    )
    
    return fig


def plot_performance_improvement(
    df: pd.DataFrame,
    resource_id: Optional[str] = None,
    activity_name: Optional[str] = None
) -> go.Figure:
    """
    Plot performance time reduction over repetitions.
    
    Args:
        df: DataFrame from ExperienceLevelTracker.to_dataframe()
        resource_id: Filter by resource (optional)
        activity_name: Filter by activity (optional)
        
    Returns:
        Plotly figure
    """
    # Filter data
    plot_df = df.copy()
    if resource_id:
        plot_df = plot_df[plot_df['resource_id'] == resource_id]
    if activity_name:
        plot_df = plot_df[plot_df['activity_name'] == activity_name]
    
    # Create figure
    fig = px.line(
        plot_df,
        x='repetition_count',
        y='mean_duration',
        color='resource_id' if not resource_id else None,
        title='Performance Improvement: Duration Reduction',
        labels={
            'repetition_count': 'Number of Repetitions',
            'mean_duration': 'Mean Duration (hours)',
            'resource_id': 'Resource'
        },
        markers=True
    )
    
    fig.update_layout(
        hovermode='x unified',
        showlegend=True,
        height=500
    )
    
    return fig


def plot_capability_heatmap(
    data,
    title: str = "Resource Experience Levels by Activity"
) -> go.Figure:
    """
    Create heatmap of resource capabilities.
    
    Args:
        data: Either ExperienceLevelTracker instance or DataFrame with columns:
              ['resource_id', 'activity_name', 'experience_level']
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # Handle both tracker and DataFrame inputs
    if isinstance(data, pd.DataFrame):
        df = data
        # Get latest experience level for each resource-activity combination
        latest = df.sort_values('simulation_time').groupby(
            ['resource_id', 'activity_name']
        )['experience_level'].last().reset_index()
        
        # Pivot to create matrix
        matrix = latest.pivot(
            index='resource_id',
            columns='activity_name',
            values='experience_level'
        )
    else:
        # Assume it's a tracker object
        stats = data.get_summary_statistics()
        matrix = stats['by_resource_activity']
    
    if matrix.empty:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=matrix.columns,
        y=matrix.index,
        colorscale='RdYlGn',  # Red (low) to Green (high)
        zmid=50,  # Middle point at level 50
        text=matrix.values,
        texttemplate='%{text:.1f}',
        textfont={"size": 10},
        colorbar=dict(title='Experience Level')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Activity',
        yaxis_title='Resource',
        height=max(400, len(matrix.index) * 40),
        xaxis={'tickangle': -45}
    )
    
    return fig


def plot_experience_distribution(
    df: pd.DataFrame,
    by: str = 'resource_id'
) -> go.Figure:
    """
    Plot distribution of experience levels.
    
    Args:
        df: DataFrame from ExperienceLevelTracker.to_dataframe()
        by: Group by 'resource_id' or 'activity_name'
        
    Returns:
        Plotly figure
    """
    # Get latest snapshot for each group
    latest = df.sort_values('simulation_time').groupby(
        ['resource_id', 'activity_name']
    ).last().reset_index()
    
    # Create box plot
    fig = px.box(
        latest,
        x=by,
        y='experience_level',
        color=by,
        title=f'Experience Level Distribution by {by.replace("_", " ").title()}',
        labels={
            by: by.replace('_', ' ').title(),
            'experience_level': 'Experience Level (0-100)'
        },
        points='all'
    )
    
    fig.update_layout(
        showlegend=False,
        height=500
    )
    
    return fig


def plot_learning_model_comparison(
    trackers: dict,
    resource_id: str,
    activity_name: str
) -> go.Figure:
    """
    Compare different learning models.
    
    Args:
        trackers: Dict of {model_name: ExperienceLevelTracker}
        resource_id: Resource to compare
        activity_name: Activity to compare
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    for model_name, tracker in trackers.items():
        df = tracker.get_resource_curve(resource_id, activity_name)
        
        if not df.empty:
            fig.add_trace(go.Scatter(
                x=df['repetition_count'],
                y=df['experience_level'],
                name=model_name,
                mode='lines+markers',
                marker=dict(size=6),
                line=dict(width=2)
            ))
    
    fig.update_layout(
        title=f'Learning Model Comparison<br>{resource_id} - {activity_name}',
        xaxis_title='Number of Repetitions',
        yaxis_title='Experience Level (0-100)',
        hovermode='x unified',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig


def plot_experience_timeline(
    df: pd.DataFrame,
    resource_id: Optional[str] = None
) -> go.Figure:
    """
    Plot experience levels over simulation time.
    
    Args:
        df: DataFrame from ExperienceLevelTracker.to_dataframe()
        resource_id: Filter by resource (optional)
        
    Returns:
        Plotly figure
    """
    # Filter data
    plot_df = df.copy()
    if resource_id:
        plot_df = plot_df[plot_df['resource_id'] == resource_id]
    
    # Create figure
    fig = px.line(
        plot_df,
        x='simulation_time',
        y='experience_level',
        color='activity_name',
        title='Experience Development Over Time',
        labels={
            'simulation_time': 'Simulation Time (hours)',
            'experience_level': 'Experience Level (0-100)',
            'activity_name': 'Activity'
        },
        markers=True
    )
    
    fig.update_layout(
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_experience_summary_table(tracker) -> pd.DataFrame:
    """
    Create summary table for dashboard.
    
    Args:
        tracker: ExperienceLevelTracker instance
        
    Returns:
        DataFrame with summary statistics
    """
    stats = tracker.get_summary_statistics()
    
    # Resource summary
    resource_summary = stats['by_resource'].copy()
    
    if resource_summary.empty:
        return pd.DataFrame()
    
    # Flatten multi-level columns
    resource_summary.columns = ['_'.join(col).strip('_') for col in resource_summary.columns]
    
    # Rename for display
    rename_map = {
        'experience_level_mean': 'Avg Experience',
        'experience_level_min': 'Min Experience',
        'experience_level_max': 'Max Experience',
        'experience_level_count': 'Activities',
        'repetition_count_sum': 'Total Tasks'
    }
    
    resource_summary = resource_summary.rename(columns=rename_map)
    resource_summary.index.name = 'Resource'
    
    return resource_summary.reset_index()


# Streamlit component examples
def render_experience_dashboard(tracker, st):
    """
    Render complete experience dashboard in Streamlit.
    
    Args:
        tracker: ExperienceLevelTracker instance
        st: Streamlit module
    """
    st.title("🎯 Experience Breeding Dashboard")
    
    # Get data
    df = tracker.to_dataframe()
    
    if df.empty:
        st.warning("No experience data available yet.")
        return
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    latest = df.sort_values('simulation_time').groupby(
        ['resource_id', 'activity_name']
    ).last()
    
    with col1:
        st.metric("Resources", df['resource_id'].nunique())
    with col2:
        st.metric("Activities", df['activity_name'].nunique())
    with col3:
        st.metric("Avg Experience", f"{latest['experience_level'].mean():.1f}")
    with col4:
        st.metric("Total Tasks", int(df['repetition_count'].max()))
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Learning Curves",
        "🎨 Capability Matrix",
        "📊 Statistics",
        "⏱️ Timeline"
    ])
    
    with tab1:
        st.subheader("Learning Curves")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            resources = ['All'] + sorted(df['resource_id'].unique().tolist())
            selected_resource = st.selectbox("Resource", resources)
        with col2:
            activities = ['All'] + sorted(df['activity_name'].unique().tolist())
            selected_activity = st.selectbox("Activity", activities)
        
        # Plot
        resource_filter = None if selected_resource == 'All' else selected_resource
        activity_filter = None if selected_activity == 'All' else selected_activity
        
        fig = plot_learning_curve(df, resource_filter, activity_filter)
        st.plotly_chart(fig, width='stretch')
        
        fig2 = plot_performance_improvement(df, resource_filter, activity_filter)
        st.plotly_chart(fig2, width='stretch')
    
    with tab2:
        st.subheader("Resource Capability Matrix")
        fig = plot_capability_heatmap(tracker)
        st.plotly_chart(fig, width='stretch')
    
    with tab3:
        st.subheader("Summary Statistics")
        summary = create_experience_summary_table(tracker)
        if not summary.empty:
            st.dataframe(summary, width='stretch')
        
        st.subheader("Experience Distribution")
        fig = plot_experience_distribution(df, by='resource_id')
        st.plotly_chart(fig, width='stretch')
    
    with tab4:
        st.subheader("Experience Timeline")
        selected_res = st.selectbox(
            "Select Resource",
            sorted(df['resource_id'].unique().tolist()),
            key='timeline_resource'
        )
        fig = plot_experience_timeline(df, selected_res)
        st.plotly_chart(fig, width='stretch')
