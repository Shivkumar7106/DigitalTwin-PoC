import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def create_glucose_timeline(timestamps, actual_glucose, predicted_glucose=None, confidence_lower=None, confidence_upper=None, meal_flags=None) -> go.Figure:
    fig = go.Figure()

    # Background zones
    fig.add_hrect(y0=0, y1=70, fillcolor="red", opacity=0.1, layer="below", line_width=0)
    fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.1, layer="below", line_width=0)
    fig.add_hrect(y0=180, y1=250, fillcolor="yellow", opacity=0.1, layer="below", line_width=0)
    fig.add_hrect(y0=250, y1=400, fillcolor="red", opacity=0.1, layer="below", line_width=0)

    # Confidence bands
    if confidence_lower is not None and confidence_upper is not None and predicted_glucose is not None:
        fig.add_trace(go.Scatter(
            x=list(timestamps) + list(timestamps)[::-1],
            y=list(confidence_upper) + list(confidence_lower)[::-1],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Confidence Interval'
        ))

    # Actual glucose
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=actual_glucose,
        mode='lines',
        line=dict(color='blue', width=2),
        name='Actual Glucose'
    ))

    # Predicted glucose
    if predicted_glucose is not None:
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=predicted_glucose,
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Predicted Glucose'
        ))

    # Meal markers
    if meal_flags is not None:
        meal_times = [t for t, f in zip(timestamps, meal_flags) if f]
        meal_vals = [g for g, f in zip(actual_glucose, meal_flags) if f]
        if meal_times:
            fig.add_trace(go.Scatter(
                x=meal_times,
                y=meal_vals,
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='orange'),
                name='Meal Event'
            ))

    fig.update_layout(
        title='Glucose Timeline',
        xaxis_title='Time',
        yaxis_title='Glucose (mg/dL)',
        yaxis=dict(range=[0, min(400, max(actual_glucose)*1.2)]),
        template='plotly_white',
        hovermode='x unified'
    )
    return fig

def create_heart_rate_chart(timestamps, heart_rate, hrv_rmssd=None) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Normal resting HR band
    fig.add_hrect(y0=60, y1=100, fillcolor="green", opacity=0.1, layer="below", line_width=0, secondary_y=False)

    fig.add_trace(
        go.Scatter(x=timestamps, y=heart_rate, name="Heart Rate", line=dict(color='orange')),
        secondary_y=False,
    )

    if hrv_rmssd is not None:
        fig.add_trace(
            go.Scatter(x=timestamps, y=hrv_rmssd, name="HRV (RMSSD)", line=dict(color='green')),
            secondary_y=True,
        )

    fig.update_layout(
        title_text="Heart Rate & HRV",
        template='plotly_white',
        hovermode='x unified'
    )
    fig.update_yaxes(title_text="Heart Rate (bpm)", secondary_y=False)
    if hrv_rmssd is not None:
        fig.update_yaxes(title_text="HRV RMSSD (ms)", secondary_y=True)

    return fig

def create_risk_gauge(risk_score: float, title: str = 'Risk Score', thresholds: dict = None) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps' : [
                {'range': [0, 20], 'color': "lightgreen"},
                {'range': [20, 40], 'color': "yellow"},
                {'range': [40, 60], 'color': "orange"},
                {'range': [60, 100], 'color': "red"}
            ],
        }
    ))
    return fig

def create_shap_waterfall(shap_values: list, feature_names: list, feature_values: list, base_value: float, max_display: int = 10) -> go.Figure:
    fig = go.Figure(go.Waterfall(
        name = "20", orientation = "h",
        measure = ["relative"] * len(shap_values),
        y = [f"{n}={v}" for n, v in zip(feature_names, feature_values)],
        x = shap_values,
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))
    fig.update_layout(title="Risk Factor Contributions (SHAP)", waterfallgap=0.3)
    return fig

def create_sleep_chart(sleep_data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if sleep_data.empty: return fig
        
    dates = sleep_data['date']
    fig.add_trace(go.Bar(x=dates, y=sleep_data['deep'], name='Deep', marker_color='navy'))
    fig.add_trace(go.Bar(x=dates, y=sleep_data['rem'], name='REM', marker_color='purple'))
    fig.add_trace(go.Bar(x=dates, y=sleep_data['light'], name='Light', marker_color='skyblue'))
    fig.add_trace(go.Bar(x=dates, y=sleep_data['wake'], name='Wake', marker_color='gray'))
    
    if 'efficiency' in sleep_data.columns:
        fig.add_trace(go.Scatter(x=dates, y=sleep_data['efficiency'], name='Efficiency %', yaxis='y2', line=dict(color='orange')))
        
    fig.update_layout(
        title='Sleep Composition',
        barmode='stack',
        yaxis=dict(title='Minutes'),
        yaxis2=dict(title='Efficiency %', overlaying='y', side='right', range=[0, 100]),
        template='plotly_white'
    )
    return fig

def create_activity_chart(step_data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if step_data.empty: return fig
    
    colors = ['red' if s < 4000 else 'yellow' if s < 7000 else 'green' for s in step_data['steps']]
    fig.add_trace(go.Bar(x=step_data['date'], y=step_data['steps'], marker_color=colors, name='Steps'))
    
    fig.add_hline(y=7000, line_dash="dash", line_color="blue", annotation_text="Target (7000)")
    fig.update_layout(title='Daily Steps', yaxis_title='Steps', template='plotly_white')
    return fig

def create_risk_trend_chart(dates, risk_scores, risk_type: str) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=20, fillcolor="green", opacity=0.1, layer="below", line_width=0)
    fig.add_hrect(y0=20, y1=40, fillcolor="yellow", opacity=0.1, layer="below", line_width=0)
    fig.add_hrect(y0=40, y1=100, fillcolor="red", opacity=0.1, layer="below", line_width=0)
    
    fig.add_trace(go.Scatter(x=dates, y=risk_scores, mode='lines+markers', name=f'{risk_type} Risk', line=dict(color='black')))
    fig.update_layout(title=f'{risk_type} Risk Trend', yaxis_title='Risk %', yaxis=dict(range=[0, 100]), template='plotly_white')
    return fig

def create_metrics_comparison(before: dict, after: dict, metric_names: list) -> go.Figure:
    fig = go.Figure()
    before_vals = [before.get(m, 0) for m in metric_names]
    after_vals = [after.get(m, 0) for m in metric_names]
    
    fig.add_trace(go.Bar(x=metric_names, y=before_vals, name='Before', marker_color='gray'))
    fig.add_trace(go.Bar(x=metric_names, y=after_vals, name='After', marker_color='blue'))
    
    fig.update_layout(title='Metric Comparison', barmode='group', template='plotly_white')
    return fig
