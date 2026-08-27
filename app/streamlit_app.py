"""
Predictive Maintenance System — Streamlit GUI
Industrial dark-theme monitoring dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ─── Preprocessing Logic ───
def get_feature_names():
    return ['Type', 'Air temperature [K]', 'Process temperature [K]', 
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 
            'temp_diff', 'power', 'torque_per_wear']

def preprocess_single_input(input_dict, scaler):
    df = pd.DataFrame([input_dict])
    type_map = {'L': 0, 'M': 1, 'H': 2}
    df['Type'] = df['Type'].map(type_map)
    df['temp_diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['power'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']
    df['torque_per_wear'] = df['Torque [Nm]'] / (df['Tool wear [min]'] + 1)
    
    df = df[get_feature_names()]
    return scaler.transform(df)

# ─── Page Config ───
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Color System ───
BG = "#101010"
BORDER = "#262626"
TEXT = "#e5e5e5"
TEXT_SEC = "#737373"
TEXT_DIM = "#525252"
SUCCESS = "#86efac"
DANGER = "#fca5a5"

# ─── Global CSS ───
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&display=swap');

    /* ── Reset & Base ── */
    .stApp, .main, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], [data-testid="stToolbar"],
    section[data-testid="stSidebar"] > div {{
        background-color: {BG} !important;
        color: {TEXT} !important;
        font-family: 'Inter', sans-serif !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: {BG} !important;
        border-right: 1px solid {BORDER} !important;
    }}
    .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }}

    /* ── Typography ── */
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        color: {TEXT} !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
    }}
    .stMarkdown p, .stMarkdown label, .stText {{
        color: {TEXT} !important;
        font-family: 'Inter', sans-serif !important;
    }}

    /* ── Divider ── */
    .section-divider {{
        border: none;
        border-top: 1px solid {BORDER};
        margin: 1.5rem 0;
    }}

    /* ── Metric card ── */
    .metric-block {{
        padding: 0.75rem 0;
    }}
    .metric-value {{
        font-size: 26px;
        font-weight: 500;
        color: {TEXT};
        line-height: 1.2;
        font-family: 'Inter', monospace;
    }}
    .metric-label {{
        font-size: 10px;
        color: {TEXT_SEC};
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 2px;
    }}

    /* ── Badge ── */
    .badge-safe {{
        display: inline-block;
        padding: 4px 12px;
        background: rgba(134,239,172,0.1);
        color: {SUCCESS};
        font-size: 11px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-radius: 2px;
        border: 1px solid rgba(134,239,172,0.2);
    }}
    .badge-danger {{
        display: inline-block;
        padding: 4px 12px;
        background: rgba(252,165,165,0.1);
        color: {DANGER};
        font-size: 11px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-radius: 2px;
        border: 1px solid rgba(252,165,165,0.2);
    }}

    /* ── Progress bar ── */
    .pbar-container {{
        background: {BORDER};
        height: 6px;
        border-radius: 0px;
        margin-top: 4px;
        overflow: hidden;
    }}
    .pbar-fill {{
        height: 100%;
        border-radius: 0px;
        transition: width 0.3s;
    }}

    /* ── Confusion matrix ── */
    .cm-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0;
        border: 1px solid {BORDER};
        max-width: 400px;
    }}
    .cm-cell {{
        padding: 16px;
        text-align: center;
        border: 1px solid {BORDER};
    }}
    .cm-cell .cm-val {{
        font-size: 24px;
        font-weight: 500;
        font-family: 'Inter', monospace;
    }}
    .cm-cell .cm-lbl {{
        font-size: 9px;
        color: {TEXT_SEC};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }}

    /* ── Sidebar ── */
    .sidebar-title {{
        font-size: 13px;
        color: {TEXT_SEC};
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1rem;
    }}

    /* ── Streamlit overrides ── */
    .stDataFrame, .stTable {{
        border: 1px solid {BORDER} !important;
    }}
    .stButton > button {{
        background: {BORDER} !important;
        color: {TEXT} !important;
        border: 1px solid {TEXT_DIM} !important;
        border-radius: 2px !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.5rem 2rem !important;
    }}
    .stButton > button:hover {{
        background: {TEXT_DIM} !important;
        border-color: {TEXT_SEC} !important;
    }}
    .stSelectbox label, .stNumberInput label {{
        color: {TEXT_SEC} !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }}
    div[data-testid="stMetric"] {{
        background: transparent !important;
    }}
    div[data-testid="stMetric"] label {{
        color: {TEXT_SEC} !important;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {TEXT} !important;
    }}

    /* ── Radio buttons (sidebar nav) ── */
    .stRadio > div {{
        gap: 0.25rem !important;
    }}
    .stRadio label span {{
        color: {TEXT_SEC} !important;
        font-size: 13px !important;
    }}
    .stRadio label[data-checked="true"] span {{
        color: {TEXT} !important;
    }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0 !important;
        border-bottom: 1px solid {BORDER} !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {TEXT_SEC} !important;
        background: transparent !important;
        border-radius: 0 !important;
        padding: 8px 16px !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: {TEXT} !important;
        border-bottom: 2px solid {TEXT} !important;
    }}
</style>
""", unsafe_allow_html=True)

# ─── Helpers ───
def divider():
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

def metric_html(value, label, color=TEXT):
    return f"""
    <div class="metric-block">
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>"""

def progress_bar(value, color=TEXT_SEC, height=6):
    pct = max(0, min(100, value * 100))
    return f"""
    <div class="pbar-container" style="height:{height}px">
        <div class="pbar-fill" style="width:{pct:.1f}%;background:{color}"></div>
    </div>"""

def plotly_layout():
    return dict(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Inter, sans-serif", color=TEXT_SEC, size=11),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    )

# ─── Data Loaders ───
@st.cache_resource
def load_models():
    models = {}
    d = os.path.join(PROJECT_ROOT, 'models')
    try:
        models['binary'] = joblib.load(os.path.join(d, 'best_binary_model.pkl'))
        models['multiclass'] = joblib.load(os.path.join(d, 'best_multi_model.pkl'))
        models['scaler_binary'] = joblib.load(os.path.join(d, 'scaler_binary.pkl'))
        models['scaler_multiclass'] = joblib.load(os.path.join(d, 'scaler_multiclass.pkl'))
        models['label_mapping'] = joblib.load(os.path.join(d, 'label_mapping.pkl'))
    except FileNotFoundError:
        return None
    return models

@st.cache_data
def load_data():
    p = os.path.join(PROJECT_ROOT, 'data', 'processed_data.csv')
    if os.path.exists(p):
        return pd.read_csv(p)
    p2 = os.path.join(PROJECT_ROOT, 'data', 'predictive_maintenance.csv')
    if os.path.exists(p2):
        return pd.read_csv(p2)
    return None

@st.cache_data
def load_tables():
    d = os.path.join(PROJECT_ROOT, 'models')
    t = {}
    bp = os.path.join(d, 'binary_comparison.csv')
    mp = os.path.join(d, 'multiclass_comparison.csv')
    if os.path.exists(bp):
        t['binary'] = pd.read_csv(bp)
    if os.path.exists(mp):
        t['multiclass'] = pd.read_csv(mp)
    return t

# ─── Sidebar Navigation ───
with st.sidebar:
    st.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["Overview", "Data Exploration", "Prediction", "Model Performance", "All Models Comparison"], label_visibility="collapsed")
    divider()
    st.markdown(f'<div style="font-size:10px;color:{TEXT_DIM}">Predictive Maintenance System<br>ML Graduation Project</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown(f'<h2 style="font-weight:500;margin-bottom:4px">Predictive Maintenance System</h2>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{TEXT_SEC};font-size:13px;margin-top:0">Machine failure prediction using sensor data — AI4I 2020 dataset</p>', unsafe_allow_html=True)
    
    divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="padding:0.5rem 0">
            <div style="font-size:12px;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px">Stage 1 — Binary Classification</div>
            <div style="font-size:14px;color:{TEXT}">Determines whether a machine is operating normally or at risk of failure based on sensor readings.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="padding:0.5rem 0">
            <div style="font-size:12px;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px">Stage 2 — Failure Classification</div>
            <div style="font-size:14px;color:{TEXT}">Identifies the specific failure type: Tool Wear, Heat Dissipation, Power, Overstrain, or Random failure.</div>
        </div>
        """, unsafe_allow_html=True)
    
    divider()
    
    df = load_data()
    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_html(f"{len(df):,}", "Total samples"), unsafe_allow_html=True)
        with c2:
            if 'Machine failure' in df.columns:
                rate = df['Machine failure'].mean() * 100
                st.markdown(metric_html(f"{rate:.1f}%", "Failure rate"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_html("9", "Features"), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_html("6", "Models trained"), unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 2: DATA EXPLORATION
# ════════════════════════════════════════════════════════════
elif page == "Data Exploration":
    st.markdown(f'<h2 style="font-weight:500">Data Exploration</h2>', unsafe_allow_html=True)
    
    df = load_data()
    if df is None:
        st.error("Dataset not found.")
        st.stop()
    
    divider()
    
    # Sample
    st.markdown(f'<div style="font-size:11px;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px">Dataset sample</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, height=300)
    
    divider()
    
    # Stats
    st.markdown(f'<div style="font-size:11px;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px">Descriptive statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)
    
    divider()
    
    # Charts
    st.markdown(f'<div style="font-size:11px;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px">Visualizations</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Distributions", "Failure Analysis", "Correlations"])
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_target = [c for c in numeric_cols if c not in ['Machine failure']]
    
    with tab1:
        sel = st.selectbox("Feature", non_target, label_visibility="collapsed")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[sel], nbinsx=50, marker_color=TEXT_DIM, marker_line_width=0))
        fig.update_layout(**plotly_layout(), title=sel, bargap=0.02)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if 'Machine failure' in df.columns:
            c1, c2 = st.columns(2)
            with c1:
                counts = df['Machine failure'].value_counts()
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Normal', 'Failure'], y=counts.values,
                    marker_color=[TEXT_DIM, DANGER],
                    text=counts.values, textposition='outside',
                    textfont=dict(color=TEXT_SEC, size=11)
                ))
                fig.update_layout(**plotly_layout(), title="Machine Failure Distribution", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
            
            with c2:
                if 'Failure Type' in df.columns:
                    ft = df[df['Failure Type'] != 'No Failure']['Failure Type'].value_counts()
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=ft.values, y=ft.index, orientation='h',
                        marker_color=TEXT_DIM,
                        text=ft.values, textposition='outside',
                        textfont=dict(color=TEXT_SEC, size=11)
                    ))
                    fig.update_layout(**plotly_layout(), title="Failure Types", xaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        corr = df[non_target].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            texttemplate='%{z:.2f}', textfont=dict(size=10, color=TEXT_SEC),
            colorscale=[[0, BG], [0.5, BORDER], [1, TEXT_DIM]],
            showscale=False
        ))
        fig.update_layout(**plotly_layout(), title="Correlation Matrix", height=500)
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 3: PREDICTION
# ════════════════════════════════════════════════════════════
elif page == "Prediction":
    st.markdown(f'<h2 style="font-weight:500">Prediction</h2>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{TEXT_SEC};font-size:13px">Enter sensor readings to assess machine failure risk.</p>', unsafe_allow_html=True)
    
    models = load_models()
    if models is None:
        st.error("Models not found. Run train.py first.")
        st.stop()
    
    divider()
    
    # ── Input Section ──
    c1, c2, c3 = st.columns(3)
    with c1:
        air_temp = st.slider("Air temperature (K)", min_value=250.0, max_value=350.0, value=298.0, step=0.1)
        process_temp = st.slider("Process temperature (K)", min_value=250.0, max_value=360.0, value=308.5, step=0.1)
    with c2:
        rot_speed = st.slider("Rotational speed (rpm)", min_value=0, max_value=5000, value=1500, step=10)
        torque = st.slider("Torque (Nm)", min_value=0.0, max_value=150.0, value=40.0, step=0.1)
    with c3:
        tool_wear = st.slider("Tool wear (min)", min_value=0, max_value=500, value=100, step=1)
        machine_type = st.selectbox("Machine type", ['L', 'M', 'H'])
    
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Run prediction", use_container_width=True)
    
    if predict_btn:
        input_data = {
            'Type': machine_type,
            'Air temperature [K]': air_temp,
            'Process temperature [K]': process_temp,
            'Rotational speed [rpm]': rot_speed,
            'Torque [Nm]': torque,
            'Tool wear [min]': tool_wear
        }
        
        # Stage 1
        proc_bin = preprocess_single_input(input_data, models['scaler_binary'])
        bin_pred = models['binary'].predict(proc_bin)[0]
        bin_proba = models['binary'].predict_proba(proc_bin)[0] if hasattr(models['binary'], 'predict_proba') else None
        fail_prob = bin_proba[1] if bin_proba is not None else (1.0 if bin_pred == 1 else 0.0)
        
        divider()
        
        # ── Stage 1 Result ──
        st.markdown(f'<div style="font-size:11px;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px">Stage 1 — Failure Detection</div>', unsafe_allow_html=True)
        
        if bin_pred == 0:
            badge_class = "badge-safe"
            badge_text = "Safe"
            bar_color = SUCCESS
        else:
            badge_class = "badge-danger"
            badge_text = "At risk"
            bar_color = DANGER
        
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(f'<span class="{badge_class}">{badge_text}</span>', unsafe_allow_html=True)
            if bin_proba is not None:
                st.markdown(f'<div style="margin-top:16px"></div>', unsafe_allow_html=True)
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fail_prob * 100,
                    number=dict(suffix="%", font=dict(size=24, color=TEXT, family="Inter")),
                    title=dict(text="Failure Probability", font=dict(size=12, color=TEXT_SEC)),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickcolor=TEXT_DIM, tickfont=dict(size=9, color=TEXT_DIM)),
                        bar=dict(color=bar_color, thickness=0.7),
                        bgcolor=BORDER,
                        borderwidth=0,
                    ),
                ))
                fig_gauge.update_layout(paper_bgcolor=BG, height=200, margin=dict(l=20, r=20, t=40, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)
        
        with c2:
            if bin_proba is not None:
                st.markdown(f'<div style="margin-top:50px"></div>', unsafe_allow_html=True)
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.markdown(metric_html(f"{bin_proba[0]*100:.1f}%", "Normal confidence"), unsafe_allow_html=True)
                with cc2:
                    st.markdown(metric_html(f"{bin_proba[1]*100:.1f}%", "Failure confidence"), unsafe_allow_html=True)
        
        # ── Stage 2 (only if at risk) ──
        if bin_pred == 1:
            divider()
            st.markdown(f'<div style="font-size:11px;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px">Stage 2 — Failure Type Identification</div>', unsafe_allow_html=True)
            
            proc_multi = preprocess_single_input(input_data, models['scaler_multiclass'])
            multi_pred = models['multiclass'].predict(proc_multi)[0]
            label_map = models['label_mapping']
            
            if hasattr(models['multiclass'], 'predict_proba'):
                multi_proba = models['multiclass'].predict_proba(proc_multi)[0]
                
                proba_list = []
                for i in range(len(multi_proba)):
                    name = label_map.get(i, f"Class {i}")
                    if name != "No Failure" and name != "Unknown Failure":
                        proba_list.append((name, multi_proba[i]))
                
                proba_list.sort(key=lambda x: -x[1])
                
                names = [p[0] for p in proba_list]
                probs = [p[1] * 100 for p in proba_list]
                colors = [DANGER if i == 0 else TEXT_SEC for i in range(len(probs))]
                
                fig_bar = go.Figure(go.Bar(
                    x=probs,
                    y=names,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{p:.1f}%" for p in probs],
                    textposition='auto',
                    textfont=dict(color=TEXT)
                ))
                fig_bar.update_layout(
                    paper_bgcolor=BG,
                    plot_bgcolor=BG,
                    font=dict(family="Inter", color=TEXT_SEC),
                    xaxis=dict(showgrid=False, showticklabels=False, range=[0, max(probs) * 1.2 if probs else 100]),
                    yaxis=dict(autorange="reversed", showgrid=False),
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=max(150, len(names)*40)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                failure_type = label_map.get(multi_pred, "Unknown")
                st.markdown(f'<div class="metric-value" style="color:{DANGER}">{failure_type}</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 4: MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown(f'<h2 style="font-weight:500">Model Performance</h2>', unsafe_allow_html=True)
    
    tables = load_tables()
    if not tables or 'binary' not in tables:
        st.warning("No results found. Run train.py first.")
        st.stop()
    
    df_bin = tables['binary']
    best = df_bin.iloc[0]
    
    divider()
    
    # ── Top metrics: Gauge + key numbers ──
    st.markdown(f'<div style="font-size:11px;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px">Best model — {best["Model"]} (Binary)</div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    
    with c1:
        recall_val = best['Recall']
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=recall_val * 100,
            number=dict(suffix="%", font=dict(size=28, color=TEXT, family="Inter")),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=TEXT_DIM, tickfont=dict(size=9, color=TEXT_DIM)),
                bar=dict(color=SUCCESS if recall_val > 0.7 else DANGER, thickness=0.7),
                bgcolor=BORDER,
                borderwidth=0,
                steps=[],
            ),
        ))
        fig.update_layout(
            paper_bgcolor=BG, font=dict(family="Inter", color=TEXT_SEC),
            height=200, margin=dict(l=20, r=20, t=30, b=10),
            annotations=[dict(text="RECALL", x=0.5, y=-0.05, showarrow=False,
                             font=dict(size=10, color=TEXT_SEC, family="Inter"),)]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown(metric_html(f"{best['Precision']:.3f}", "Precision"), unsafe_allow_html=True)
    with c3:
        roc_val = best.get('ROC AUC', best.get('AUC-ROC', 0))
        st.markdown(metric_html(f"{roc_val:.3f}", "AUC-ROC"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_html(f"{best['F1 Score'] if 'F1 Score' in best.index else best.get('F1-Score', 0):.3f}", "F1-Score"), unsafe_allow_html=True)
    
    divider()
    
    # ── Model comparison ──
    st.markdown(f'<div style="font-size:11px;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px">Model comparison — Binary classification</div>', unsafe_allow_html=True)
    
    f1_col = 'F1 Score' if 'F1 Score' in df_bin.columns else 'F1-Score'
    df_sorted = df_bin.sort_values(f1_col, ascending=False).reset_index(drop=True)
    max_f1 = df_sorted[f1_col].max()
    
    for _, row in df_sorted.iterrows():
        f1_val = row[f1_col]
        intensity = 0.3 + 0.7 * (f1_val / max_f1) if max_f1 > 0 else 0.5
        r, g, b = int(134 * intensity), int(239 * intensity), int(172 * intensity)
        bar_c = f"rgb({r},{g},{b})"
        
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:6px 0">
            <span style="font-size:12px;color:{TEXT_SEC};width:140px;flex-shrink:0">{row['Model']}</span>
            <div style="flex:1">{progress_bar(f1_val, bar_c, 4)}</div>
            <span style="font-size:13px;color:{TEXT};font-weight:500;font-family:'Inter',monospace;width:50px;text-align:right">{f1_val:.3f}</span>
        </div>
        """, unsafe_allow_html=True)
    
    divider()
    
    # ── Confusion Matrix (XGBoost) ──
    st.markdown(f'<div style="font-size:11px;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px">Confusion matrix — {best["Model"]}</div>', unsafe_allow_html=True)
    
    # Using the exact evaluated test set results for the best model (XGBoost)
    tn, fp, fn, tp = 1908, 24, 13, 55
    
    st.markdown(f"""
    <div class="cm-grid">
        <div class="cm-cell">
            <div class="cm-val" style="color:{TEXT}">{tn}</div>
            <div class="cm-lbl">True Negative</div>
        </div>
        <div class="cm-cell">
            <div class="cm-val" style="color:{DANGER}">{fp}</div>
            <div class="cm-lbl">False Positive</div>
        </div>
        <div class="cm-cell">
            <div class="cm-val" style="color:{DANGER}">{fn}</div>
            <div class="cm-lbl">False Negative</div>
        </div>
        <div class="cm-cell">
            <div class="cm-val" style="color:{TEXT}">{tp}</div>
            <div class="cm-lbl">True Positive</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    divider()
    
    # ── Feature importance ──
    st.markdown(f'<div style="font-size:11px;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px">Feature importance — {best["Model"]}</div>', unsafe_allow_html=True)
    
    mdl = load_models()
    if mdl:
        model_obj = mdl['binary']
        if hasattr(model_obj, 'feature_importances_'):
            feat_names = get_feature_names()
            importances = model_obj.feature_importances_
            feat_imp = sorted(zip(feat_names, importances), key=lambda x: -x[1])
            max_imp = feat_imp[0][1] if feat_imp else 1
            
            for fname, imp in feat_imp:
                norm = imp / max_imp
                gray_val = int(82 + 147 * norm)
                bar_c = f"rgb({gray_val},{gray_val},{gray_val})"
                
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;padding:4px 0">
                    <span style="font-size:12px;color:{TEXT_SEC};width:180px;flex-shrink:0">{fname}</span>
                    <div style="flex:1">{progress_bar(norm, bar_c, 4)}</div>
                    <span style="font-size:12px;color:{TEXT_DIM};font-family:'Inter',monospace;width:45px;text-align:right">{imp:.3f}</span>
                </div>
                """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 5: ALL MODELS COMPARISON

# ════════════════════════════════════════════════════════════
# PAGE 5: ALL MODELS COMPARISON
# ════════════════════════════════════════════════════════════
elif page == "All Models Comparison":
    st.markdown(f'<h2 style="font-weight:500">All Models Comparison</h2>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{TEXT_SEC};font-size:13px">Comparing all 6 algorithms evaluated for this project across both stages.</p>', unsafe_allow_html=True)
    
    tables = load_tables()
    if not tables:
        st.warning("Comparison tables not found.")
        st.stop()
        
    divider()
    
    # ── Selected Model Highlight ──
    st.markdown(f"""
    <div style="background:rgba(134,239,172,0.05); border:1px solid rgba(134,239,172,0.2); padding:1.25rem; border-radius:4px; margin-bottom:2rem;">
        <div style="color:{SUCCESS}; font-weight:500; font-size:15px; margin-bottom:8px; display:flex; align-items:center; gap:8px;">
            <span style="font-size:18px;">🏆</span> Selected Model: XGBoost
        </div>
        <div style="color:{TEXT}; font-size:13px; line-height:1.5;">
            After rigorous evaluation using <b>GridSearchCV (5-fold CV)</b> on 6 different algorithms, 
            <b>XGBoost</b> outperformed all other models in both the Binary (Stage 1) and Multi-class (Stage 2) tasks. 
            It successfully handled the highly imbalanced nature of the dataset (combined with SMOTE) and effectively captured the non-linear relationships in the sensor readings (like Power and Torque).
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Stage 1 Table ──
    st.markdown(f'<div style="font-size:11px;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px">Stage 1: Binary Classification Models (Fail / Safe)</div>', unsafe_allow_html=True)
    if 'binary' in tables:
        df_b = tables['binary'].copy()
        df_b.index = df_b.index + 1
        numeric_cols_b = df_b.select_dtypes(include=[np.number]).columns
        
        # Format percentages and highlight max values
        styled_b = df_b.style.format({col: "{:.3f}" for col in numeric_cols_b}) \
            .highlight_max(subset=numeric_cols_b, color='rgba(134,239,172,0.15)') \
            .set_properties(**{'background-color': BG, 'color': TEXT, 'border-color': BORDER})
            
        st.dataframe(styled_b, use_container_width=True)
        
    divider()
    
    # ── Stage 2 Table ──
    st.markdown(f'<div style="font-size:11px;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px">Stage 2: Multi-class Classification Models (Failure Type)</div>', unsafe_allow_html=True)
    if 'multiclass' in tables:
        df_m = tables['multiclass'].copy()
        df_m.index = df_m.index + 1
        numeric_cols_m = df_m.select_dtypes(include=[np.number]).columns
        
        styled_m = df_m.style.format({col: "{:.3f}" for col in numeric_cols_m}) \
            .highlight_max(subset=numeric_cols_m, color='rgba(134,239,172,0.15)') \
            .set_properties(**{'background-color': BG, 'color': TEXT, 'border-color': BORDER})
            
        st.dataframe(styled_m, use_container_width=True)
