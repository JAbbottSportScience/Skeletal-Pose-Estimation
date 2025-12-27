"""
================================================================================
SPRINT POSE ESTIMATION DASHBOARD (Enhanced)
================================================================================
Features:
  - Selectable metrics (1-3)
  - Synchronized zoom: plot zoom updates video range & kinogram
  - Dynamic kinogram from zoomed frame range
  - Butterworth filtering for joint angles

Run with: python app.py
Access at: http://localhost:8050
================================================================================
"""

import os
import sys
import json
import base64
import tempfile
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import cv2

# Butterworth filter
from scipy.signal import butter, filtfilt, sosfiltfilt
from scipy.ndimage import uniform_filter1d

# Import from existing project
from configs.config import PoseEstimationConfig, SmoothingConfig
from src.pose_estimator import PoseEstimator, KeypointResult, create_estimator
from src.video_processor import VideoProcessor
from src.visualizer import SkeletonVisualizer, KinogramGenerator
from src.smoothing import OneEuroFilter

from src.biomechanics import (
    get_leading_trailing_legs,
    calculate_joint_angles,
    calculate_stride_metrics,
    detect_gait_phase,
    analyze_frame,
)


# ==============================================================================
# BUTTERWORTH FILTER
# ==============================================================================

def butterworth_filter(
    data: np.ndarray, 
    cutoff_freq: float, 
    sample_rate: float, 
    order: int = 4,
    filter_type: str = 'low'
) -> np.ndarray:
    """
    Apply Butterworth filter to data.
    
    Args:
        data: 1D array of values to filter
        cutoff_freq: Cutoff frequency in Hz
        sample_rate: Sampling rate in Hz (video FPS)
        order: Filter order (higher = sharper cutoff)
        filter_type: 'low', 'high', or 'band'
    
    Returns:
        Filtered data array
    """
    if len(data) < 15:  # Need enough points for filtering
        return data
    
    # Nyquist frequency
    nyquist = sample_rate / 2.0
    
    # Normalize cutoff frequency
    normalized_cutoff = cutoff_freq / nyquist
    
    # Clamp to valid range (0, 1)
    normalized_cutoff = np.clip(normalized_cutoff, 0.01, 0.99)
    
    try:
        # Design filter using second-order sections (more stable)
        sos = butter(order, normalized_cutoff, btype=filter_type, output='sos')
        
        # Apply zero-phase filter (no phase shift)
        filtered = sosfiltfilt(sos, data)
        
        return filtered
    except Exception as e:
        print(f"Filter error: {e}")
        return data


def apply_filters_to_metrics(
    metrics_data: Dict[str, List],
    sample_rate: float,
    cutoff_freq: float,
    order: int,
    enabled: bool = True
) -> Dict[str, List]:
    """
    Apply Butterworth filter to all metrics.
    
    Args:
        metrics_data: Dict of metric_name -> values list
        sample_rate: Video FPS
        cutoff_freq: Filter cutoff frequency in Hz
        order: Filter order
        enabled: Whether filtering is enabled
    
    Returns:
        Dict of filtered metrics
    """
    if not enabled:
        return metrics_data
    
    filtered = {}
    for key, values in metrics_data.items():
        if len(values) > 0:
            arr = np.array(values, dtype=float)
            filtered[key] = butterworth_filter(arr, cutoff_freq, sample_rate, order).tolist()
        else:
            filtered[key] = values
    
    return filtered


# ==============================================================================
# AVAILABLE METRICS
# ==============================================================================

AVAILABLE_METRICS = {
    'lead_hip_flexion': {'label': 'Lead Hip (vs vertical)', 'color': '#00ff00', 'unit': '°'},
    'trail_hip_flexion': {'label': 'Trail Hip (vs vertical)', 'color': '#ff6600', 'unit': '°'},
    'lead_knee_flexion': {'label': 'Lead Knee Flexion', 'color': '#00ccff', 'unit': '°'},
    'trail_knee_flexion': {'label': 'Trail Knee Flexion', 'color': '#ff00ff', 'unit': '°'},
    'ankle_separation': {'label': 'Ankle Separation', 'color': '#ffff00', 'unit': 'px'},
    'lead_knee_height': {'label': 'Lead Knee Height', 'color': '#00ff88', 'unit': 'px'},
    'trail_knee_height': {'label': 'Trail Knee Height', 'color': '#ff8800', 'unit': 'px'},
    'lead_ankle_y': {'label': 'Lead Ankle Y', 'color': '#88ff00', 'unit': 'px'},
    'trail_ankle_y': {'label': 'Trail Ankle Y', 'color': '#ff0088', 'unit': 'px'},
}

DEFAULT_METRICS = ['lead_hip_flexion', 'trail_hip_flexion', 'lead_knee_flexion']


# ==============================================================================
# GLOBAL STATE
# ==============================================================================

class AppState:
    def __init__(self):
        self.estimator: Optional[PoseEstimator] = None
        self.current_video_path: Optional[Path] = None
        self.video_metadata = None
        self.results: List[KeypointResult] = []
        self.frames: List[np.ndarray] = []
        self.processing: bool = False
        self.progress: float = 0.0
        self.config = PoseEstimationConfig()
        
        # Computed metrics cache
        self.metrics_data: Dict[str, List] = {}
        self.metrics_frames: List[int] = []
        
        # Zoom state
        self.zoom_range: Optional[Tuple[int, int]] = None
        
    def reset(self):
        self.results = []
        self.frames = []
        self.progress = 0.0
        self.metrics_data = {}
        self.metrics_frames = []
        self.zoom_range = None
    
    def compute_metrics(self):
        """Compute all metrics from results."""
        self.metrics_data = {key: [] for key in AVAILABLE_METRICS}
        self.metrics_frames = []
        
        for r in self.results:
            if r.num_people > 0:
                person = r.get_primary_person()
                if person:
                    kps, scores = person
                    
                    # Skip low confidence frames
                    if any(scores[j] < 0.3 for j in [11, 12, 13, 14, 15, 16]):
                        continue
                    
                    self.metrics_frames.append(r.frame_idx)
                    
                    legs = get_leading_trailing_legs(kps)
                    angles = calculate_joint_angles(kps)
                    stride = calculate_stride_metrics(kps)
                    
                    self.metrics_data['lead_hip_flexion'].append(angles['lead_hip_flexion'])
                    self.metrics_data['trail_hip_flexion'].append(angles['trail_hip_flexion'])
                    self.metrics_data['lead_knee_flexion'].append(angles['lead_knee_flexion'])
                    self.metrics_data['trail_knee_flexion'].append(angles['trail_knee_flexion'])
                    self.metrics_data['ankle_separation'].append(stride['ankle_separation'])
                    self.metrics_data['lead_knee_height'].append(stride['lead_knee_height'])
                    self.metrics_data['trail_knee_height'].append(stride['trail_knee_height'])
                    self.metrics_data['lead_ankle_y'].append(legs['lead']['ankle'][1])
                    self.metrics_data['trail_ankle_y'].append(legs['trail']['ankle'][1])


state = AppState()

app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.DARKLY], 
    title="Sprint Pose Estimation",
    suppress_callback_exceptions=True
)


# ==============================================================================
# LAYOUT
# ==============================================================================

def create_metric_selector():
    """Create the metric selection checklist with filter options."""
    return dbc.Card([
        dbc.CardHeader(html.H6("📊 Metrics & Filtering", className="mb-0")),
        dbc.CardBody([
            # Metric selection
            dbc.Label("Select Metrics (1-3)", className="fw-bold"),
            dbc.Checklist(
                id='metric-selector',
                options=[
                    {'label': v['label'], 'value': k} 
                    for k, v in AVAILABLE_METRICS.items()
                ],
                value=DEFAULT_METRICS,
                inline=False,
                className="metric-checklist mb-3"
            ),
            
            html.Hr(),
            
            # Filter toggle
            dbc.Row([
                dbc.Col([
                    dbc.Label("Butterworth Filter", className="fw-bold"),
                ], width=8),
                dbc.Col([
                    dbc.Switch(
                        id='filter-enabled',
                        value=True,
                        className="float-end"
                    ),
                ], width=4),
            ], className="mb-2"),
            
            # Cutoff frequency
            dbc.Label("Cutoff Frequency (Hz)"),
            dcc.Slider(
                id='filter-cutoff',
                min=1, max=15, step=0.5, value=6,
                marks={1: '1', 6: '6', 10: '10', 15: '15'},
                tooltip={"placement": "bottom", "always_visible": True},
                className="mb-2"
            ),
            html.Small("Lower = smoother (typical: 6-12 Hz for running)", className="text-muted d-block mb-2"),
            
            # Filter order
            dbc.Label("Filter Order"),
            dcc.Slider(
                id='filter-order',
                min=2, max=8, step=2, value=4,
                marks={2: '2nd', 4: '4th', 6: '6th', 8: '8th'},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Small("Higher = sharper cutoff", className="text-muted d-block"),
            
            # Show raw toggle
            dbc.Checklist(
                id='show-raw-data',
                options=[{'label': ' Show raw data (faded)', 'value': 'show'}],
                value=[],
                className="mt-3"
            ),
        ])
    ], className="mb-3")


app.layout = dbc.Container([
    # Header
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("🏃 Sprint Pose Estimation Dashboard", className="ms-2"),
        ], fluid=True),
        color="primary", dark=True, className="mb-4"
    ),
    
    dbc.Row([
        # Left Column - Controls
        dbc.Col([
            # Upload
            dbc.Card([
                dbc.CardHeader(html.H5("📁 Video")),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-video',
                        children=html.Div(['Drag & Drop or ', html.A('Select', className="text-primary")]),
                        style={'width': '100%', 'height': '80px', 'lineHeight': '80px',
                               'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '10px',
                               'textAlign': 'center', 'cursor': 'pointer'},
                        accept='video/*'
                    ),
                    html.Div(id='video-info', className="mt-2"),
                ])
            ], className="mb-3"),
            
            # Settings
            dbc.Card([
                dbc.CardHeader(html.H5("⚙️ Settings")),
                dbc.CardBody([
                    dbc.Label("Model"),
                    dcc.Dropdown(
                        id='model-select',
                        options=[
                            {'label': 'YOLOv8n (Fast)', 'value': 'yolov8n-pose.pt'},
                            {'label': 'YOLOv8m (Medium)', 'value': 'yolov8m-pose.pt'},
                            {'label': 'YOLOv8x (Accurate)', 'value': 'yolov8x-pose.pt'},
                        ],
                        value='yolov8x-pose.pt', clearable=False, className="mb-3"
                    ),
                    
                    dbc.Label("Confidence"),
                    dcc.Slider(id='confidence-slider', min=0.1, max=0.9, step=0.05, value=0.5,
                               marks={0.1: '0.1', 0.5: '0.5', 0.9: '0.9'},
                               tooltip={"placement": "bottom", "always_visible": True}),
                    
                    html.H6("Smoothing", className="mt-3"),
                    dbc.Label("Min Cutoff (lower=smoother)"),
                    dcc.Slider(id='smoothing-cutoff', min=0.1, max=1.0, step=0.05, value=0.3,
                               tooltip={"placement": "bottom", "always_visible": True}),
                    
                    dbc.Label("Beta"),
                    dcc.Slider(id='smoothing-beta', min=0.05, max=0.5, step=0.05, value=0.1,
                               tooltip={"placement": "bottom", "always_visible": True}),
                ])
            ], className="mb-3"),
            
            # Metric Selector
            create_metric_selector(),
            
            # Process Buttons
            dbc.Button("▶️ Process Video", id='process-btn', color="success", size="lg", className="w-100 mb-2", disabled=True),
            dbc.Button("🔄 Reset Zoom", id='reset-zoom-btn', color="warning", className="w-100 mb-2", disabled=True),
            dbc.Button("💾 Export JSON", id='export-btn', color="secondary", className="w-100", disabled=True),
            
            dbc.Progress(id='progress-bar', value=0, className="mt-3"),
            html.Div(id='status-text', className="text-center mt-2"),
            
        ], md=3),
        
        # Right Column - Results
        dbc.Col([
            # Tabs for Video / Kinogram / Stats
            dbc.Tabs([
                # Video Preview Tab
                dbc.Tab(label="🎬 Video", children=[
                    dbc.Card([
                        dbc.CardHeader([
                            html.H6("Video Preview", className="mb-0 d-inline"),
                            html.Span(id='zoom-indicator', className="ms-2 badge bg-info"),
                        ]),
                        dbc.CardBody([
                            html.Div(id='frame-container', className="text-center", children=[
                                html.Img(id='frame-display', style={'maxHeight': '400px', 'maxWidth': '100%'}),
                            ]),
                            dcc.Slider(
                                id='frame-slider', 
                                min=0, max=100, step=1, value=0,
                                tooltip={"placement": "bottom", "always_visible": True},
                                className="mt-3"
                            ),
                            html.Div(id='frame-info', className="text-center small mt-1"),
                        ])
                    ], className="mt-3")
                ]),
                
                # Kinogram Tab
                dbc.Tab(label="🖼️ Kinogram", children=[
                    dbc.Card([
                        dbc.CardHeader([
                            html.H6("Kinogram", className="mb-0 d-inline"),
                            html.Small(" — Generated from zoomed frame range", className="text-muted"),
                        ]),
                        dbc.CardBody([
                            html.Div(id='kinogram-container', className="text-center", children=[
                                html.Img(id='kinogram-display', style={'maxHeight': '450px', 'maxWidth': '100%'}),
                            ]),
                            html.Hr(),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Number of Phases"),
                                    dcc.Slider(
                                        id='kinogram-phases', 
                                        min=4, max=16, step=1, value=8,
                                        marks={4: '4', 8: '8', 12: '12', 16: '16'},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    ),
                                ], md=8),
                                dbc.Col([
                                    dbc.Button("🖼️ Generate Kinogram", id='kinogram-btn', color="info", 
                                               className="w-100 mt-4", disabled=True),
                                ], md=4),
                            ]),
                        ])
                    ], className="mt-3")
                ]),
                
                # Stats Tab
                dbc.Tab(label="📊 Stats", children=[
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H2(id='stat-frames', className="mb-0 text-primary"),
                                            html.P("Total Frames", className="mb-0")
                                        ], className="text-center")
                                    ])
                                ], md=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H2(id='stat-detections', className="mb-0 text-success"),
                                            html.P("Detections", className="mb-0")
                                        ], className="text-center")
                                    ])
                                ], md=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H2(id='stat-fps', className="mb-0 text-info"),
                                            html.P("Processing FPS", className="mb-0")
                                        ], className="text-center")
                                    ])
                                ], md=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H2(id='stat-conf', className="mb-0 text-warning"),
                                            html.P("Avg Confidence", className="mb-0")
                                        ], className="text-center")
                                    ])
                                ], md=3),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H3(id='stat-zoom-range', className="mb-0"),
                                            html.P("Current Zoom Range", className="mb-0")
                                        ], className="text-center")
                                    ])
                                ], md=12, className="mt-3"),
                            ]),
                        ])
                    ], className="mt-3")
                ]),
            ], className="mb-3"),
            
            # Metrics Plot (always visible below tabs)
            dbc.Card([
                dbc.CardHeader([
                    html.H6("📈 Biomechanics Plot", className="mb-0 d-inline"),
                    html.Small(" — Zoom to sync video & kinogram | Yellow line = current frame", className="text-muted"),
                ]),
                dbc.CardBody([
                    dcc.Graph(
                        id='metrics-graph',
                        style={'height': '300px'},
                        config={
                            'displayModeBar': True,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                            'scrollZoom': True,
                        }
                    ),
                ])
            ]),
            
        ], md=9),
    ]),
    
    # Hidden stores
    dcc.Store(id='video-store'),
    dcc.Store(id='results-store'),
    dcc.Store(id='zoom-store', data={'start': None, 'end': None}),
    dcc.Interval(id='progress-interval', interval=500, disabled=True),
    dcc.Download(id='download-json'),
    
], fluid=True)


# ==============================================================================
# CALLBACKS
# ==============================================================================

@callback(
    [Output('video-info', 'children'),
     Output('video-store', 'data'),
     Output('process-btn', 'disabled')],
    Input('upload-video', 'contents'),
    State('upload-video', 'filename'),
    prevent_initial_call=True
)
def handle_upload(contents, filename):
    if contents is None:
        return "", None, True
    
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        temp_dir = Path(tempfile.gettempdir()) / "pose_estimation"
        temp_dir.mkdir(exist_ok=True)
        video_path = temp_dir / filename
        
        with open(video_path, 'wb') as f:
            f.write(decoded)
        
        processor = VideoProcessor()
        metadata = processor.get_metadata(video_path)
        
        state.current_video_path = video_path
        state.video_metadata = metadata
        state.reset()
        
        info = html.Div([
            html.Strong(filename), html.Br(),
            f"{metadata.width}x{metadata.height} | {metadata.fps:.0f}fps | {metadata.duration:.1f}s"
        ], className="small text-success")
        
        return info, str(video_path), False
        
    except Exception as e:
        return html.Div(f"Error: {e}", className="text-danger"), None, True


@callback(
    [Output('progress-bar', 'value'),
     Output('status-text', 'children'),
     Output('kinogram-btn', 'disabled'),
     Output('export-btn', 'disabled'),
     Output('reset-zoom-btn', 'disabled'),
     Output('frame-slider', 'max'),
     Output('results-store', 'data'),
     Output('progress-interval', 'disabled')],
    [Input('process-btn', 'n_clicks'),
     Input('progress-interval', 'n_intervals')],
    [State('video-store', 'data'),
     State('model-select', 'value'),
     State('confidence-slider', 'value'),
     State('smoothing-cutoff', 'value'),
     State('smoothing-beta', 'value')],
    prevent_initial_call=True
)
def process_video(n_clicks, n_intervals, video_path, model, confidence, cutoff, beta):
    triggered = ctx.triggered_id
    
    if triggered == 'progress-interval':
        if state.processing:
            return state.progress * 100, f"Processing... {state.progress*100:.0f}%", True, True, True, 0, None, False
        elif len(state.results) > 0:
            # Compute metrics when done
            state.compute_metrics()
            return 100, f"✓ Done! {len(state.results)} frames", False, False, False, len(state.results)-1, {'done': True}, True
        return dash.no_update
    
    if triggered == 'process-btn' and video_path:
        state.reset()
        state.processing = True
        
        state.config.model.model_name = model
        state.config.model.confidence_threshold = confidence
        state.config.smoothing.min_cutoff = cutoff
        state.config.smoothing.beta = beta
        
        def process_thread():
            try:
                if state.estimator is None or state.estimator.config.model.model_name != model:
                    state.config.model.model_name = model
                    state.estimator = create_estimator(state.config)
                
                state.estimator.smoother = OneEuroFilter(min_cutoff=cutoff, beta=beta)
                state.estimator.smoother.reset()
                
                cap = cv2.VideoCapture(str(video_path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    state.frames.append(frame)
                    result = state.estimator.estimate_single(frame, idx)
                    state.results.append(result)
                    
                    state.progress = (idx + 1) / total
                    idx += 1
                
                cap.release()
            finally:
                state.processing = False
        
        threading.Thread(target=process_thread).start()
        return 0, "Starting...", True, True, True, 0, None, False
    
    return dash.no_update


@callback(
    Output('metrics-graph', 'figure'),
    [Input('results-store', 'data'),
     Input('metric-selector', 'value'),
     Input('zoom-store', 'data'),
     Input('frame-slider', 'value'),
     Input('filter-enabled', 'value'),
     Input('filter-cutoff', 'value'),
     Input('filter-order', 'value'),
     Input('show-raw-data', 'value')],
    prevent_initial_call=True
)
def update_metrics_plot(data, selected_metrics, zoom_data, current_frame, 
                        filter_enabled, cutoff_freq, filter_order, show_raw):
    """Update the metrics plot with selected metrics, filtering, and current frame indicator."""
    if not state.metrics_frames or not selected_metrics:
        return go.Figure().update_layout(
            template='plotly_dark',
            annotations=[dict(text="Process a video to see metrics", showarrow=False, font=dict(size=16))]
        )
    
    # Limit to 3 metrics
    selected_metrics = selected_metrics[:3]
    
    # Get sample rate from video metadata
    sample_rate = 30.0  # Default
    if state.video_metadata:
        sample_rate = state.video_metadata.fps
    
    # Apply Butterworth filter if enabled
    if filter_enabled:
        filtered_data = apply_filters_to_metrics(
            state.metrics_data,
            sample_rate=sample_rate,
            cutoff_freq=cutoff_freq,
            order=filter_order,
            enabled=True
        )
    else:
        filtered_data = state.metrics_data
    
    fig = go.Figure()
    
    # Show raw data as faded lines if requested
    show_raw_lines = 'show' in (show_raw or [])
    
    for metric_key in selected_metrics:
        if metric_key in state.metrics_data and metric_key in AVAILABLE_METRICS:
            metric_info = AVAILABLE_METRICS[metric_key]
            
            # Raw data (faded) - if enabled
            if show_raw_lines and filter_enabled:
                fig.add_trace(go.Scatter(
                    x=state.metrics_frames,
                    y=state.metrics_data[metric_key],
                    name=f"{metric_info['label']} (raw)",
                    line=dict(color=metric_info['color'], width=1, dash='dot'),
                    opacity=0.3,
                    hoverinfo='skip',
                    showlegend=False
                ))
            
            # Filtered (or raw if filter disabled) data
            fig.add_trace(go.Scatter(
                x=state.metrics_frames,
                y=filtered_data[metric_key],
                name=metric_info['label'],
                line=dict(color=metric_info['color'], width=2),
                hovertemplate=f"{metric_info['label']}: %{{y:.1f}}{metric_info['unit']}<br>Frame: %{{x}}<extra></extra>"
            ))
    
    # Add bright yellow vertical line for current frame
    if current_frame is not None:
        fig.add_vline(
            x=current_frame,
            line=dict(color='#FFFF00', width=3, dash='solid'),
            annotation=dict(
                text=f"Frame {current_frame}",
                font=dict(color='#FFFF00', size=10),
                bgcolor='rgba(0,0,0,0.7)',
                yanchor='bottom'
            )
        )
    
    # Build title with filter info
    if filter_enabled:
        title_text = f"Butterworth Filter: {cutoff_freq} Hz, {filter_order}th order"
    else:
        title_text = "Filtering: OFF (raw data)"
    
    fig.update_layout(
        template='plotly_dark',
        showlegend=True,
        legend=dict(orientation='h', y=1.15, x=0.5, xanchor='center'),
        margin=dict(l=50, r=20, t=60, b=40),
        xaxis_title='Frame',
        yaxis_title='Value',
        hovermode='x unified',
        dragmode='zoom',
        title=dict(
            text=title_text,
            font=dict(size=12, color='#888'),
            x=0.5,
            xanchor='center'
        )
    )
    
    # Apply zoom if set
    if zoom_data and zoom_data.get('start') is not None:
        fig.update_xaxes(range=[zoom_data['start'], zoom_data['end']])
    
    return fig


@callback(
    [Output('zoom-store', 'data'),
     Output('frame-slider', 'min'),
     Output('frame-slider', 'value'),
     Output('zoom-indicator', 'children')],
    [Input('metrics-graph', 'relayoutData'),
     Input('reset-zoom-btn', 'n_clicks')],
    [State('zoom-store', 'data'),
     State('frame-slider', 'value')],
    prevent_initial_call=True
)
def handle_zoom(relayout_data, reset_clicks, current_zoom, current_frame):
    """Handle zoom events from the plot and reset button."""
    triggered = ctx.triggered_id
    
    if triggered == 'reset-zoom-btn':
        # Reset to full range
        if state.results:
            state.zoom_range = None
            return {'start': None, 'end': None}, 0, 0, ""
        return dash.no_update
    
    if triggered == 'metrics-graph' and relayout_data:
        # Check for zoom event
        if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
            start = int(relayout_data['xaxis.range[0]'])
            end = int(relayout_data['xaxis.range[1]'])
            
            # Clamp to valid range
            start = max(0, start)
            end = min(len(state.results) - 1 if state.results else 100, end)
            
            state.zoom_range = (start, end)
            
            # Update slider to be within zoom range
            new_frame = max(start, min(end, current_frame if current_frame else start))
            
            return (
                {'start': start, 'end': end}, 
                start, 
                new_frame, 
                f"Frames {start}-{end}"
            )
        
        # Check for autorange/reset
        if 'xaxis.autorange' in relayout_data:
            state.zoom_range = None
            return {'start': None, 'end': None}, 0, current_frame, ""
    
    return dash.no_update


@callback(
    [Output('frame-display', 'src'),
     Output('frame-info', 'children')],
    Input('frame-slider', 'value'),
    prevent_initial_call=True
)
def update_frame(frame_idx):
    """Update the displayed frame."""
    if frame_idx is None or frame_idx >= len(state.frames) or frame_idx >= len(state.results):
        return "", ""
    
    frame = state.frames[frame_idx].copy()
    result = state.results[frame_idx]
    
    viz = SkeletonVisualizer(state.config)
    if result.num_people > 0:
        frame = viz.draw_multi_person(frame, result)
    
    _, buffer = cv2.imencode('.jpg', frame)
    img_str = base64.b64encode(buffer).decode()
    
    # Build info string
    info_parts = [f"Frame {frame_idx}"]
    if result.num_people > 0:
        info_parts.append(f"{result.num_people} person(s)")
        person = result.get_primary_person()
        if person:
            _, scores = person
            valid_scores = scores[scores > 0]
            if len(valid_scores) > 0:
                info_parts.append(f"Conf: {np.mean(valid_scores):.0%}")
    
    return f"data:image/jpeg;base64,{img_str}", " | ".join(info_parts)


@callback(
    Output('frame-slider', 'value', allow_duplicate=True),
    Input('metrics-graph', 'clickData'),
    State('frame-slider', 'min'),
    State('frame-slider', 'max'),
    prevent_initial_call=True
)
def click_plot_to_jump(click_data, slider_min, slider_max):
    """Click on plot to jump video to that frame."""
    if click_data is None:
        return dash.no_update
    
    try:
        # Get the x value (frame) from click
        clicked_frame = int(click_data['points'][0]['x'])
        
        # Clamp to valid range
        clicked_frame = max(slider_min or 0, min(slider_max or 100, clicked_frame))
        
        return clicked_frame
    except (KeyError, IndexError, TypeError):
        return dash.no_update


@callback(
    Output('kinogram-display', 'src'),
    [Input('kinogram-btn', 'n_clicks'),
     Input('zoom-store', 'data')],
    State('kinogram-phases', 'value'),
    prevent_initial_call=True
)
def generate_kinogram(n_clicks, zoom_data, phases):
    """Generate kinogram from the zoomed frame range."""
    triggered = ctx.triggered_id
    
    # Only regenerate on button click
    if triggered != 'kinogram-btn':
        return dash.no_update
    
    if not state.frames or not state.results:
        return ""
    
    # Determine frame range
    if zoom_data and zoom_data.get('start') is not None:
        start_frame = int(zoom_data['start'])
        end_frame = int(zoom_data['end'])
    else:
        start_frame = 0
        end_frame = len(state.frames) - 1
    
    # Clamp to valid range
    start_frame = max(0, start_frame)
    end_frame = min(len(state.frames) - 1, end_frame)
    
    # Get frames and results for the range
    range_frames = state.frames[start_frame:end_frame + 1]
    range_results = state.results[start_frame:end_frame + 1]
    
    if len(range_frames) < phases:
        phases = max(2, len(range_frames))
    
    # Calculate evenly spaced phase indices within the range
    phase_indices = np.linspace(0, len(range_frames) - 1, phases, dtype=int).tolist()
    
    state.config.kinogram.num_phases = phases
    generator = KinogramGenerator(state.config)
    kinogram = generator.generate(range_frames, range_results, phase_indices)
    
    _, buffer = cv2.imencode('.png', kinogram)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode()}"


@callback(
    [Output('stat-frames', 'children'),
     Output('stat-detections', 'children'),
     Output('stat-fps', 'children'),
     Output('stat-conf', 'children'),
     Output('stat-zoom-range', 'children')],
    [Input('results-store', 'data'),
     Input('zoom-store', 'data')],
    prevent_initial_call=True
)
def update_stats(data, zoom_data):
    """Update statistics display."""
    if not state.results:
        return "0", "0", "0", "0%", "All"
    
    n = len(state.results)
    det = sum(1 for r in state.results if r.num_people > 0)
    fps = state.estimator.get_average_fps() if state.estimator else 0
    
    confs = [c for r in state.results for c in r.scores.flatten() if c > 0]
    avg = np.mean(confs) if confs else 0
    
    # Zoom range display
    if zoom_data and zoom_data.get('start') is not None:
        zoom_str = f"{zoom_data['start']}-{zoom_data['end']}"
    else:
        zoom_str = f"0-{n-1}"
    
    return str(n), str(det), f"{fps:.1f}", f"{avg:.0%}", zoom_str


@callback(
    Output('metric-selector', 'value'),
    Input('metric-selector', 'value'),
    prevent_initial_call=True
)
def limit_metric_selection(selected):
    """Limit selection to 3 metrics maximum."""
    if selected and len(selected) > 3:
        return selected[:3]
    return selected


@callback(
    Output('download-json', 'data'),
    Input('export-btn', 'n_clicks'),
    [State('filter-enabled', 'value'),
     State('filter-cutoff', 'value'),
     State('filter-order', 'value')],
    prevent_initial_call=True
)
def export_json(n, filter_enabled, cutoff_freq, filter_order):
    """Export results and metrics as JSON."""
    if not state.results:
        return None
    
    # Get sample rate
    sample_rate = 30.0
    if state.video_metadata:
        sample_rate = state.video_metadata.fps
    
    # Apply filter to metrics
    if filter_enabled:
        filtered_data = apply_filters_to_metrics(
            state.metrics_data,
            sample_rate=sample_rate,
            cutoff_freq=cutoff_freq,
            order=filter_order,
            enabled=True
        )
    else:
        filtered_data = state.metrics_data
    
    data = {
        'video': state.current_video_path.name if state.current_video_path else '',
        'model': state.config.model.model_name,
        'total_frames': len(state.results),
        'sample_rate_fps': sample_rate,
        'filter_settings': {
            'enabled': filter_enabled,
            'type': 'butterworth_lowpass',
            'cutoff_hz': cutoff_freq,
            'order': filter_order
        },
        'metrics': {
            'frames': state.metrics_frames,
            'raw': state.metrics_data,
            'filtered': filtered_data if filter_enabled else None
        },
        'pose_data': [r.to_dict() for r in state.results]
    }
    
    return dict(
        content=json.dumps(data, indent=2), 
        filename=f"sprint_analysis_{datetime.now():%Y%m%d_%H%M%S}.json"
    )


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏃 Sprint Pose Estimation Dashboard (Enhanced)")
    print("="*60)
    print("Features:")
    print("  • Select 1-3 metrics to visualize")
    print("  • Butterworth low-pass filtering (adjustable)")
    print("  • Zoom the plot to sync video slider range")
    print("  • Kinogram generates from zoomed frame range")
    print("  • Click plot to jump to frame")
    print("="*60)
    print("Open: http://localhost:8050")
    print("="*60 + "\n")
    app.run(debug=True, port=8050)