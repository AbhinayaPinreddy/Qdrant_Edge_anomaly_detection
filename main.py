import time
import math
import streamlit as st
import plotly.graph_objs as go
import numpy as np
from collections import deque

from core.qdrant_engine import QdrantEdgeEngine
from intelligence.anomaly_engine import AnomalyDetector
import config

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0.5rem; }

    /* Hide default metric styling */
    div[data-testid="stMetric"] { padding: 0; background: none; border: none; }
    div[data-testid="stMetricLabel"] { display: none; }
    div[data-testid="stMetricValue"] { display: none; }

    /* Small sensor value box */
    .sensor-card {
        background: #111118;
        border: 1px solid #222235;
        border-radius: 6px;
        padding: 7px 12px;
        text-align: left;
    }
    .sensor-label {
        font-size: 0.68rem;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 2px;
    }
    .sensor-value {
        font-size: 1.05rem;
        font-weight: 600;
        color: #ddd;
        font-family: 'Courier New', monospace;
    }

    /* Stats box */
    .stat-card {
        background: #111118;
        border: 1px solid #222235;
        border-radius: 6px;
        padding: 6px 12px;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .stat-label { font-size: 0.7rem; color: #555; text-transform: uppercase; letter-spacing: 0.6px; }
    .stat-value { font-size: 0.95rem; font-weight: 600; color: #ccc; font-family: monospace; }

    /* Phase banner */
    .phase-training {
        background: #0d1f0d; border-left: 3px solid #27ae60;
        border-radius: 4px; padding: 8px 14px;
        color: #2ecc71; font-size: 0.85rem; font-weight: 600;
    }
    .phase-stabilizing {
        background: #0d0d1f; border-left: 3px solid #2980b9;
        border-radius: 4px; padding: 8px 14px;
        color: #5dade2; font-size: 0.85rem; font-weight: 600;
    }
    .phase-detecting {
        background: #141414; border-left: 3px solid #444;
        border-radius: 4px; padding: 8px 14px;
        color: #999; font-size: 0.85rem; font-weight: 600;
    }

    /* Alert */
    .alert-normal {
        background: #0d1a0d; border-left: 3px solid #27ae60;
        border-radius: 4px; padding: 8px 14px;
        color: #2ecc71; font-size: 0.85rem; font-weight: 600;
    }
    .alert-anomaly {
        background: #1f0d0d; border-left: 3px solid #e74c3c;
        border-radius: 4px; padding: 8px 14px;
        color: #e74c3c; font-size: 0.85rem; font-weight: 700;
    }

    /* Feed + Log */
    .feed-box {
        background: #0a0a12; border: 1px solid #1a1a2a;
        border-radius: 6px; padding: 10px 12px;
        font-family: 'Courier New', monospace; font-size: 0.72rem;
        color: #556; height: 200px; overflow-y: auto;
    }
    .feed-normal  { color: #2ecc71; }
    .feed-anomaly { color: #e74c3c; font-weight: bold; }
    .feed-warmup  { color: #e67e22; }

    .anom-log {
        background: #0a0a12; border: 1px solid #2a1a1a;
        border-radius: 6px; padding: 10px 12px;
        font-family: 'Courier New', monospace; font-size: 0.72rem;
        color: #e74c3c; height: 200px; overflow-y: auto;
    }

    .section-title {
        font-size: 0.68rem; color: #444;
        text-transform: uppercase; letter-spacing: 1.2px;
        margin-bottom: 6px; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown(
    "<div style='font-size:1.1rem;font-weight:700;color:#ccc;"
    "letter-spacing:1px;margin-bottom:0.4rem'>"
    "REAL-TIME SENSOR ANOMALY DETECTION</div>",
    unsafe_allow_html=True
)
st.markdown("<hr style='border:none;border-top:1px solid #1e1e2e;margin:0 0 0.8rem 0'/>", unsafe_allow_html=True)

# =========================
# INIT ENGINE
# =========================
engine   = QdrantEdgeEngine(fresh=True)
detector = AnomalyDetector(engine)

# =========================
# TOP ROW: SENSORS (left) + STATS (right)
# =========================
top_cols = st.columns([4, 1])

with top_cols[0]:
    st.markdown("<div class='section-title'>Sensor Readings</div>", unsafe_allow_html=True)
    sc = st.columns(5)
    temp_box  = sc[0].empty()
    hum_box   = sc[1].empty()
    vib_box   = sc[2].empty()
    sim_box   = sc[3].empty()
    step_box  = sc[4].empty()

with top_cols[1]:
    st.markdown("<div class='section-title'>Session Stats</div>", unsafe_allow_html=True)
    stats_box = st.empty()

st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)

# Phase + Alert
pa_cols = st.columns(2)
phase_box = pa_cols[0].empty()
alert_box = pa_cols[1].empty()

st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)

# =========================
# CHARTS
# =========================
chart        = st.empty()
sensor_chart = st.empty()

st.markdown("<hr style='border:none;border-top:1px solid #1a1a2a;margin:0.6rem 0'/>", unsafe_allow_html=True)

# =========================
# BOTTOM: FEED + ANOMALY LOG
# =========================
bot_cols = st.columns(2)

with bot_cols[0]:
    st.markdown("<div class='section-title'>Live Data Feed</div>", unsafe_allow_html=True)
    feed_box = st.empty()

with bot_cols[1]:
    st.markdown("<div class='section-title'>Anomaly Log</div>", unsafe_allow_html=True)
    anom_log_box = st.empty()

# =========================
# DATA BUFFERS
# =========================
CHART_WINDOW = 200   # how many steps visible in rolling chart

scores  = deque(maxlen=CHART_WINDOW)
score_x = deque(maxlen=CHART_WINDOW)

temps  = deque(maxlen=CHART_WINDOW)
hums   = deque(maxlen=CHART_WINDOW)
vibs   = deque(maxlen=CHART_WINDOW)

anomaly_x = []
anomaly_y = []

# Feed lines (last 12 lines shown)
feed_lines = deque(maxlen=12)
# Anomaly log (last 12 entries)
anom_lines = deque(maxlen=12)

anomaly_count = 0
trained_count = 0

window = deque(maxlen=10)
t = 0
i = 0

try:
    while True:
        t += 0.1

        # =========================
        # SENSOR DATA
        # =========================
        temperature = 25 + 3 * math.sin(t) + np.random.normal(0, 0.7)
        humidity    = 60 + 8 * math.sin(t / 2) + np.random.normal(0, 1.5)
        vibration   = 0.02 + 0.02 * math.sin(t) + np.random.normal(0, 0.02)

        is_injected = False
        if np.random.rand() < 0.05:
            temperature += np.random.uniform(40, 80)
            vibration   += np.random.uniform(3, 8)
            is_injected  = True
        if np.random.rand() < 0.05:
            humidity    -= np.random.uniform(30, 60)
            is_injected  = True

        # =========================
        # FEATURE EXTRACTION
        # =========================
        window.append([temperature, humidity, vibration])
        temps.append(temperature)
        hums.append(humidity)
        vibs.append(vibration)

        if len(window) < 10:
            continue

        i += 1   # only count steps where detection actually runs

        window_np = np.array(window)
        features  = []
        for j in range(window_np.shape[1]):
            col = window_np[:, j]
            features.extend([col.mean(), col.std(), col.min(), col.max(), col[-1] - col[0]])

        vector = np.array(features)

        # =========================
        # DETECT
        # =========================
        prev_count = engine._count
        result     = detector.process(vector)
        stored     = engine._count > prev_count   # did engine store this vector?

        if stored:
            trained_count += 1

        scores.append(result.similarity)
        score_x.append(i)

        if result.is_anomaly:
            anomaly_count += 1
            anomaly_x.append(i)
            anomaly_y.append(result.similarity)

        # =========================
        # DETERMINE PHASE LABEL
        # =========================
        if result.reason == "WARMUP":
            phase_label = "WARMUP"
            phase_css   = "phase-training"
            phase_icon  = "TRAINING PHASE — Learning normal patterns"
        elif result.reason == "STABILIZING":
            phase_label = "STABILIZING"
            phase_css   = "phase-stabilizing"
            phase_icon  = "STABILIZING — Building baseline"
        else:
            phase_label = "DETECTING"
            phase_css   = "phase-detecting"
            phase_icon  = "LIVE DETECTION"

        # =========================
        # FEED LINE
        # =========================
        if result.reason == "WARMUP":
            if stored:
                tag = "feed-warmup"
                tag_label = "TRAINED"
            else:
                tag = "feed-anomaly"
                tag_label = "SKIPPED"
        elif result.is_anomaly:
            tag = "feed-anomaly"
            tag_label = "ANOMALY"
        else:
            tag = "feed-normal"
            tag_label = "NORMAL "

        feed_line = (
            f"<span class='{tag}'>"
            f"[{tag_label}] Step {i:>4} | "
            f"T={temperature:>6.1f} H={humidity:>5.1f} V={vibration:.3f} | "
            f"Sim={result.similarity:.4f}"
            f"</span>"
        )
        feed_lines.append(feed_line)

        # =========================
        # ANOMALY LOG ENTRY
        # =========================
        if result.is_anomaly:
            anom_lines.append(
                f"Step {i} | Sim={result.similarity:.4f} | {result.reason} | "
                f"T={temperature:.1f} H={humidity:.1f} V={vibration:.3f}"
            )

        # =========================
        # TOP METRICS (small clean cards)
        # =========================
        def sensor_card(label, value):
            return (
                f"<div class='sensor-card'>"
                f"<div class='sensor-label'>{label}</div>"
                f"<div class='sensor-value'>{value}</div>"
                f"</div>"
            )

        temp_box.markdown(sensor_card("Temp (°C)",  f"{temperature:.2f}"),  unsafe_allow_html=True)
        hum_box.markdown (sensor_card("Humidity (%)", f"{humidity:.2f}"),   unsafe_allow_html=True)
        vib_box.markdown (sensor_card("Vibration",  f"{vibration:.4f}"),    unsafe_allow_html=True)
        sim_box.markdown (sensor_card("Similarity", f"{result.similarity:.4f}"), unsafe_allow_html=True)
        step_box.markdown(sensor_card("Step",       str(i)),                unsafe_allow_html=True)

        # Phase banner
        phase_box.markdown(
            f"<div class='{phase_css}'>{phase_icon}</div>",
            unsafe_allow_html=True
        )

        # Alert
        if result.is_anomaly:
            alert_box.markdown(
                f"<div class='alert-anomaly'>"
                f"ANOMALY DETECTED &nbsp;|&nbsp; "
                f"Similarity = {result.similarity:.4f} &nbsp;|&nbsp; Step {result.step}"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            alert_box.markdown(
                f"<div class='alert-normal'>System Normal &nbsp;|&nbsp; {result.reason}</div>",
                unsafe_allow_html=True
            )

        # =========================
        # ROLLING CHART (moves right→left)
        # =========================
        x_list  = list(score_x)
        y_list  = list(scores)

        x_min = x_list[0]  if x_list else 0
        x_max = x_list[-1] if x_list else CHART_WINDOW

        vis_ax = [x for x in anomaly_x if x >= x_min]
        vis_ay = [anomaly_y[k] for k, x in enumerate(anomaly_x) if x >= x_min]

        # Background shade for warmup zone
        warmup_end_x = min(config.WARMUP_STEPS + 10, x_max)

        fig = go.Figure()

        # Warmup zone shading
        if x_min <= config.WARMUP_STEPS + 10:
            fig.add_vrect(
                x0=x_min, x1=min(warmup_end_x, x_max),
                fillcolor="rgba(39,174,96,0.06)",
                line_width=0,
                layer="below",
            )

        # Similarity area line
        fig.add_trace(go.Scatter(
            x=x_list,
            y=y_list,
            mode='lines',
            name='Similarity',
            line=dict(color='#5dade2', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(93,173,226,0.06)',
        ))

        # Anomaly markers
        fig.add_trace(go.Scatter(
            x=vis_ax,
            y=vis_ay,
            mode='markers',
            name='Anomaly',
            marker=dict(
                color='#e74c3c',
                size=11,
                symbol='x-thin',
                line=dict(color='#e74c3c', width=2.5),
            ),
        ))

        # Reference line
        fig.add_hline(
            y=config.CHART_SIMILARITY_REF_LINE,
            line_dash="dot",
            line_color="rgba(241,196,15,0.4)",
            line_width=1,
            annotation_text="Ref",
            annotation_font_color="rgba(241,196,15,0.6)",
            annotation_position="bottom right",
        )

        y_min = min(0.75, min(y_list) - 0.02) if y_list else 0.75

        fig.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(l=50, r=20, t=20, b=40),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#aaa", size=11),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.01,
                xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
            ),
            xaxis=dict(
                title="Step",
                gridcolor="#1a1a2a",
                range=[x_min, x_max],   # fixed window → chart scrolls left
                showgrid=True,
            ),
            yaxis=dict(
                title="Cosine Similarity",
                gridcolor="#1a1a2a",
                range=[y_min, 1.01],
                showgrid=True,
            ),
        )

        chart.plotly_chart(fig, use_container_width=True)

        # =========================
        # SENSOR LINES CHART
        # =========================
        sx_list = list(score_x)   # same x-axis as similarity chart

        # vibration scaled up so it's visible alongside temp/humidity
        vib_scaled = [v * 1000 for v in list(vibs)]

        sfig = go.Figure()

        sfig.add_trace(go.Scatter(
            x=sx_list, y=list(temps),
            mode='lines', name='Temperature (°C)',
            line=dict(color='#e74c3c', width=1.5),
        ))

        sfig.add_trace(go.Scatter(
            x=sx_list, y=list(hums),
            mode='lines', name='Humidity (%)',
            line=dict(color='#3498db', width=1.5),
        ))

        sfig.add_trace(go.Scatter(
            x=sx_list, y=vib_scaled,
            mode='lines', name='Vibration (×1000)',
            line=dict(color='#f39c12', width=1.5),
        ))

        # Mark anomaly steps on sensor chart too
        for ax in vis_ax:
            sfig.add_vline(
                x=ax,
                line_dash="dot",
                line_color="rgba(231,76,60,0.4)",
                line_width=1,
            )

        sfig.update_layout(
            template="plotly_dark",
            height=260,
            margin=dict(l=50, r=20, t=20, b=40),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#aaa", size=11),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.01,
                xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
            ),
            xaxis=dict(
                title="Step",
                gridcolor="#1a1a2a",
                range=[x_min, x_max],
                showgrid=True,
            ),
            yaxis=dict(
                title="Sensor Values",
                gridcolor="#1a1a2a",
                showgrid=True,
            ),
        )

        sensor_chart.plotly_chart(sfig, use_container_width=True)

        # =========================
        # FEED BOX
        # =========================
        feed_html = "<br>".join(reversed(list(feed_lines)))
        feed_box.markdown(
            f"<div class='feed-box'>{feed_html}</div>",
            unsafe_allow_html=True
        )

        # =========================
        # ANOMALY LOG BOX
        # =========================
        if anom_lines:
            anom_html = "<br>".join(reversed(list(anom_lines)))
        else:
            anom_html = "<span style='color:#555'>No anomalies detected yet...</span>"
        anom_log_box.markdown(
            f"<div class='anom-log'>{anom_html}</div>",
            unsafe_allow_html=True
        )

        # =========================
        # SESSION STATS (top right)
        # =========================
        rate = (anomaly_count / i * 100) if i > 0 else 0.0

        def stat_row(label, value):
            return (
                f"<div class='stat-card'>"
                f"<span class='stat-label'>{label}</span>"
                f"<span class='stat-value'>{value}</span>"
                f"</div>"
            )

        stats_box.markdown(
            stat_row("Steps",    str(i)) +
            stat_row("Anomalies", str(anomaly_count)) +
            stat_row("Rate",     f"{rate:.1f}%") +
            stat_row("Trained",  str(trained_count)),
            unsafe_allow_html=True
        )

        time.sleep(0.2)

except KeyboardInterrupt:
    engine.close()
