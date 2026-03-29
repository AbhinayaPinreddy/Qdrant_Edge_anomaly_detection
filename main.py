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
st.set_page_config(layout="wide")
st.title(" Real-Time Sensor Anomaly Detection")

engine = QdrantEdgeEngine(fresh=True)
detector = AnomalyDetector(engine)

# UI
col1, col2, col3, col4 = st.columns(4)
temp_box = col1.empty()
hum_box = col2.empty()
vib_box = col3.empty()
sim_box = col4.empty()

chart = st.empty()
alert_box = st.empty()

# Data
scores = []
anomaly_x = []
anomaly_y = []

window = deque(maxlen=10)

t = 0
i = 0

try:
    while True:
        i += 1
        t += 0.1

        # =========================
        # RANDOM SENSOR DATA
        # =========================
        temperature = 25 + 3 * math.sin(t) + np.random.normal(0, 0.7)
        humidity = 60 + 8 * math.sin(t / 2) + np.random.normal(0, 1.5)
        vibration = 0.02 + 0.02 * math.sin(t) + np.random.normal(0, 0.02)

        # =========================
        # STRONG RANDOM ANOMALIES
        # =========================
        if np.random.rand() < 0.05:
            temperature += np.random.uniform(40, 80)
            vibration += np.random.uniform(3, 8)

        if np.random.rand() < 0.05:
            humidity -= np.random.uniform(30, 60)

        # =========================
        # FEATURE EXTRACTION
        # =========================
        window.append([temperature, humidity, vibration])

        if len(window) < 10:
            continue

        window_np = np.array(window)

        features = []
        for j in range(window_np.shape[1]):
            col = window_np[:, j]

            features.extend([
                col.mean(),
                col.std(),
                col.min(),
                col.max(),
                col[-1] - col[0]
            ])

        vector = np.array(features)

        # =========================
        result = detector.process(vector)

        scores.append(result.similarity)

        # =========================
        # METRICS
        # =========================
        temp_box.metric("🌡 Temp", f"{temperature:.2f}")
        hum_box.metric("💧 Hum", f"{humidity:.2f}")
        vib_box.metric("⚙ Vib", f"{vibration:.3f}")
        sim_box.metric("📊 Similarity", f"{result.similarity:.3f}")

        # =========================
        # ANOMALY POINTS
        # =========================
        if result.is_anomaly:
            anomaly_x.append(i)
            anomaly_y.append(result.similarity)

            alert_box.error(
                f" Anomaly | Similarity={result.similarity:.3f}"
            )
        else:
            alert_box.empty()

        # =========================
        # GRAPH
        # =========================
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=scores,
            mode='lines',
            name='Similarity',
            line=dict(color='#00BFFF')
        ))

        fig.add_trace(go.Scatter(
            x=anomaly_x,
            y=anomaly_y,
            mode='markers',
            marker=dict(color='red', size=8),
            name='Anomaly'
        ))

        # Threshold line
        fig.add_hline(
            y=config.ANOMALY_THRESHOLD,
            line_dash="dash",
            line_color="yellow"
        )

        fig.update_layout(
            template="plotly_dark",
            height=400
        )

        chart.plotly_chart(fig, width="stretch")

        time.sleep(0.2)

except KeyboardInterrupt:
    engine.close()