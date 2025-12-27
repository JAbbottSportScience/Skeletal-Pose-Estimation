"""
================================================================================
SPRINT POSE ESTIMATION DASHBOARD
================================================================================
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
from typing import Optional, List

import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import cv2

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
        
    def reset(self):
        self.results = []
        self.frames = []
        self.progress = 0.0

state = AppState()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], title="Sprint Pose Estimation")


# ==============================================================================
# LAYOUT
# ==============================================================================

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
                    
                    html.H6("Kinogram", className="mt-3"),
                    dbc.Label("Phases"),
                    dcc.Slider(id='kinogram-phases', min=4, max=20, step=1, value=10,
                               marks={4: '4', 10: '10', 20: '20'},
                               tooltip={"placement": "bottom", "always_visible": True}),
                ])
            ], className="mb-3"),
            
            # Process Button
            dbc.Button("▶️ Process Video", id='process-btn', color="success", size="lg", className="w-100 mb-2", disabled=True),
            dbc.Button("🖼️ Generate Kinogram", id='kinogram-btn', color="info", className="w-100 mb-2", disabled=True),
            dbc.Button("💾 Export JSON", id='export-btn', color="secondary", className="w-100", disabled=True),
            
            dbc.Progress(id='progress-bar', value=0, className="mt-3"),
            html.Div(id='status-text', className="text-center mt-2"),
            
        ], md=3),
        
        # Right Column - Results
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="Preview", children=[
                    html.Div(className="mt-3", children=[
                        dcc.Slider(id='frame-slider', min=0, max=100, step=1, value=0,
                                   tooltip={"placement": "bottom", "always_visible": True}),
                        html.Div(id='frame-container', className="text-center mt-2", children=[
                            html.Img(id='frame-display', style={'maxHeight': '500px', 'maxWidth': '100%'}),
                        ]),
                        html.Div(id='frame-info', className="mt-2"),
                    ])
                ]),
                dbc.Tab(label="Kinogram", children=[
                    html.Div(id='kinogram-container', className="text-center mt-3", children=[
                        html.Img(id='kinogram-display', style={'maxHeight': '600px', 'maxWidth': '100%'}),
                    ])
                ]),
                dbc.Tab(label="Plot", children=[
                    dcc.Graph(id='keypoint-graph', style={'height': '500px'})
                ]),
                dbc.Tab(label="Stats", children=[
                    html.Div(className="mt-3", children=[
                        dbc.Row([
                            dbc.Col(dbc.Card(dbc.CardBody([html.H3(id='stat-frames'), html.P("Frames")])), md=3),
                            dbc.Col(dbc.Card(dbc.CardBody([html.H3(id='stat-detections'), html.P("Detections")])), md=3),
                            dbc.Col(dbc.Card(dbc.CardBody([html.H3(id='stat-fps'), html.P("FPS")])), md=3),
                            dbc.Col(dbc.Card(dbc.CardBody([html.H3(id='stat-conf'), html.P("Avg Conf")])), md=3),
                        ])
                    ])
                ]),
            ]),
        ], md=9),
    ]),
    
    # Hidden stores
    dcc.Store(id='video-store'),
    dcc.Store(id='results-store'),
    dcc.Interval(id='progress-interval', interval=500, disabled=True),
    dcc.Download(id='download-json'),
    
], fluid=True)



# ==============================================================================
# JOINT ANGLE CALCULATIONS
# ==============================================================================

def calculate_angle(p1, p2, p3):
    """
    Calculate angle at p2 formed by p1-p2-p3.
    Returns angle in degrees.
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.degrees(np.arccos(cos_angle))
    
    return angle


def get_joint_angles(keypoints):
    """
    Calculate hip and knee flexion angles from COCO keypoints.
    
    COCO indices:
        5/6: L/R shoulder
        11/12: L/R hip
        13/14: L/R knee
        15/16: L/R ankle
    
    Returns dict with left/right hip and knee angles.
    """
    # Left side
    l_shoulder = keypoints[5]
    l_hip = keypoints[11]
    l_knee = keypoints[13]
    l_ankle = keypoints[15]
    
    # Right side
    r_shoulder = keypoints[6]
    r_hip = keypoints[12]
    r_knee = keypoints[14]
    r_ankle = keypoints[16]
    
    # Hip flexion: angle at hip (shoulder-hip-knee)
    # 180° = full extension, <180° = flexion
    l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
    r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
    
    # Knee flexion: angle at knee (hip-knee-ankle)
    # 180° = full extension, <180° = flexion
    l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
    r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
    
    return {
        'l_hip': l_hip_angle,
        'r_hip': r_hip_angle,
        'l_knee': l_knee_angle,
        'r_knee': r_knee_angle,
    }


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
            return state.progress * 100, f"Processing... {state.progress*100:.0f}%", True, True, 0, None, False
        elif len(state.results) > 0:
            return 100, f"✓ Done! {len(state.results)} frames", False, False, len(state.results)-1, {'done': True}, True
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
        return 0, "Starting...", True, True, 0, None, False
    
    return dash.no_update


@callback(
    [Output('frame-display', 'src'),
     Output('frame-info', 'children')],
    Input('frame-slider', 'value'),
    prevent_initial_call=True
)
def update_frame(frame_idx):
    if frame_idx >= len(state.frames) or frame_idx >= len(state.results):
        return "", ""
    
    frame = state.frames[frame_idx].copy()
    result = state.results[frame_idx]
    
    viz = SkeletonVisualizer(state.config)
    if result.num_people > 0:
        frame = viz.draw_multi_person(frame, result)
    
    _, buffer = cv2.imencode('.jpg', frame)
    img_str = base64.b64encode(buffer).decode()
    
    info = f"Frame {frame_idx} | {result.num_people} person(s)"
    if result.num_people > 0:
        person = result.get_primary_person()
        if person:
            _, scores = person
            info += f" | Conf: {np.mean(scores[scores>0]):.0%}"
    
    return f"data:image/jpeg;base64,{img_str}", info


@callback(
    Output('kinogram-display', 'src'),
    Input('kinogram-btn', 'n_clicks'),
    State('kinogram-phases', 'value'),
    prevent_initial_call=True
)
def generate_kinogram(n_clicks, phases):
    if not state.frames or not state.results:
        return ""
    
    state.config.kinogram.num_phases = phases
    generator = KinogramGenerator(state.config)
    kinogram = generator.generate(state.frames, state.results)
    
    _, buffer = cv2.imencode('.png', kinogram)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode()}"


@callback(
    Output('keypoint-graph', 'figure'),
    Input('results-store', 'data'),
    prevent_initial_call=True
)
def update_plot(data):
    if not state.results:
        return go.Figure()
    
    from plotly.subplots import make_subplots
    
    frames = []
    lead_ankle_y, trail_ankle_y = [], []
    lead_hip_flex, trail_hip_flex = [], []
    lead_knee_flex, trail_knee_flex = [], []
    lead_sides = []  # Track which side is leading
    
    for r in state.results:
        if r.num_people > 0:
            person = r.get_primary_person()
            if person:
                kps, scores = person
                
                # Skip if key joints are low confidence
                if any(scores[j] < 0.3 for j in [11, 12, 13, 14, 15, 16]):
                    continue
                
                frames.append(r.frame_idx)
                
                # Get lead/trail legs
                legs = get_leading_trailing_legs(kps)
                angles = calculate_joint_angles(kps)
                
                # Ankle Y positions
                lead_ankle_y.append(legs['lead']['ankle'][1])
                trail_ankle_y.append(legs['trail']['ankle'][1])
                
                # Joint angles
                lead_hip_flex.append(angles['lead_hip_flexion'])
                trail_hip_flex.append(angles['trail_hip_flexion'])
                lead_knee_flex.append(angles['lead_knee_flexion'])
                trail_knee_flex.append(angles['trail_knee_flexion'])
                
                # Track which side is leading
                lead_sides.append(angles['lead_side'])
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'Ankle Y Position (lower = ground contact)',
            'Hip Flexion (positive = thigh forward)',
            'Knee Flexion (positive = bent)',
            'Lead Leg Side'
        ),
        shared_xaxes=True,
        vertical_spacing=0.08
    )
    
    # Row 1: Ankle position
    fig.add_trace(go.Scatter(
        x=frames, y=lead_ankle_y, name='Lead Ankle',
        line=dict(color='#00ff00', width=2)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=frames, y=trail_ankle_y, name='Trail Ankle',
        line=dict(color='#ff6600', width=2)
    ), row=1, col=1)
    
    # Row 2: Hip flexion
    fig.add_trace(go.Scatter(
        x=frames, y=lead_hip_flex, name='Lead Hip',
        line=dict(color='#00ff00', width=2)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=frames, y=trail_hip_flex, name='Trail Hip',
        line=dict(color='#ff6600', width=2)
    ), row=2, col=1)
    
    # Row 3: Knee flexion
    fig.add_trace(go.Scatter(
        x=frames, y=lead_knee_flex, name='Lead Knee',
        line=dict(color='#00ff00', width=2)
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=frames, y=trail_knee_flex, name='Trail Knee',
        line=dict(color='#ff6600', width=2)
    ), row=3, col=1)
    
    # Row 4: Lead side indicator (Left=0, Right=1)
    lead_side_numeric = [0 if s == 'left' else 1 for s in lead_sides]
    fig.add_trace(go.Scatter(
        x=frames, y=lead_side_numeric, name='Lead Side',
        line=dict(color='#ffff00', width=2, shape='hv'),  # Step plot
        fill='tozeroy', fillcolor='rgba(255,255,0,0.2)'
    ), row=4, col=1)
    
    fig.update_layout(
        height=800,
        template='plotly_dark',
        showlegend=True,
        legend=dict(orientation='h', y=1.02),
        title='Sprint Biomechanics (Lead vs Trail Leg)'
    )
    
    # Axis labels
    fig.update_yaxes(title_text='Y (px)', row=1, col=1, autorange='reversed')
    fig.update_yaxes(title_text='Degrees', row=2, col=1)
    fig.update_yaxes(title_text='Degrees', row=3, col=1)
    fig.update_yaxes(
        title_text='Side', row=4, col=1,
        tickvals=[0, 1], ticktext=['Left', 'Right'],
        range=[-0.1, 1.1]
    )
    fig.update_xaxes(title_text='Frame', row=4, col=1)
    
    return fig



@callback(
    [Output('stat-frames', 'children'),
     Output('stat-detections', 'children'),
     Output('stat-fps', 'children'),
     Output('stat-conf', 'children')],
    Input('results-store', 'data'),
    prevent_initial_call=True
)
def update_stats(data):
    if not state.results:
        return "0", "0", "0", "0%"
    
    n = len(state.results)
    det = sum(1 for r in state.results if r.num_people > 0)
    fps = state.estimator.get_average_fps() if state.estimator else 0
    
    confs = [c for r in state.results for c in r.scores.flatten() if c > 0]
    avg = np.mean(confs) if confs else 0
    
    return str(n), str(det), f"{fps:.1f}", f"{avg:.0%}"


@callback(
    Output('download-json', 'data'),
    Input('export-btn', 'n_clicks'),
    prevent_initial_call=True
)
def export_json(n):
    if not state.results:
        return None
    
    data = {
        'video': state.current_video_path.name if state.current_video_path else '',
        'model': state.config.model.model_name,
        'frames': [r.to_dict() for r in state.results]
    }
    return dict(content=json.dumps(data, indent=2), filename=f"pose_{datetime.now():%Y%m%d_%H%M%S}.json")


# ==============================================================================
# RUN
# ==============================================================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🏃 Sprint Pose Estimation Dashboard")
    print("="*50)
    print("Open: http://localhost:8050")
    print("="*50 + "\n")
    app.run(debug=True, port=8050)
