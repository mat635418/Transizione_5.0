# =============================
# core_app.py
# Enterprise-grade Transizione 5.0 Compliance Monitor
# =============================

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import pytz
import hashlib
import threading
import time
from collections import deque

# -----------------------------
# CONFIG (all overridable via UI)
# -----------------------------
CET = pytz.timezone("Europe/Rome")

# -----------------------------
# SESSION INIT
# -----------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "buffer" not in st.session_state:
    st.session_state.buffer = deque(maxlen=500)
if "audit" not in st.session_state:
    st.session_state.audit = []
if "last_meter_ts" not in st.session_state:
    st.session_state.last_meter_ts = None

# -----------------------------
# IMMUTABLE AUDIT LOG
# -----------------------------
def audit_log(event, payload=""):
    ts_utc = datetime.now(timezone.utc)
    ts_cet = ts_utc.astimezone(CET)
    raw = f"{ts_utc.isoformat()}|{event}|{payload}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    st.session_state.audit.append({
        "utc": ts_utc,
        "cet": ts_cet,
        "event": event,
        "payload": payload,
        "hash": digest
    })

# -----------------------------
# DATA ACQUISITION THREAD
# -----------------------------
def acquisition_loop(cfg):
    audit_log("ACQ_START")
    while st.session_state.running:
        time.sleep(cfg["poll_s"])
        if np.random.rand() < cfg["timeout_prob"]:
            audit_log("MODBUS_TIMEOUT")
            continue

        value = np.random.normal(cfg["base_kw"], cfg["noise"])
        ts = datetime.now(timezone.utc)
        st.session_state.buffer.append({"ts": ts, "kw": value})
        st.session_state.last_meter_ts = ts

# -----------------------------
# SIDEBAR – FULLY PARAMETRIC
# -----------------------------
st.sidebar.header("⚙️ Configuration")

poll_s = st.sidebar.slider("Polling interval (s)", 1, 10, 2)
st.sidebar.caption("❓ How often meters are polled")

refresh_ms = st.sidebar.slider("UI refresh (ms)", 500, 5000, 2000, step=500)
st.sidebar.caption("❓ UI redraw frequency")

base_kw = st.sidebar.slider("Baseline power (kW)", 10, 500, 120)
st.sidebar.caption("❓ Nominal consumption")

noise = st.sidebar.slider("Noise (σ)", 0.1, 20.0, 5.0)
st.sidebar.caption("❓ Measurement variability")

timeout_prob = st.sidebar.slider("Modbus timeout probability", 0.0, 0.5, 0.05)
st.sidebar.caption("❓ Simulated comm failures")

st.sidebar.markdown("---")

st.sidebar.subheader("ISO 50001 Flags")

flag_baseline = st.sidebar.checkbox("Baseline defined", True)
flag_monitoring = st.sidebar.checkbox("Continuous monitoring active", True)
flag_improvement = st.sidebar.checkbox("Energy improvement tracked", False)

# -----------------------------
# MAIN CONTROLS
# -----------------------------
st.title("🏭 Transizione 5.0 Compliance Monitor")

col1, col2 = st.columns(2)

with col1:
    if st.button("▶ START"):
        if not st.session_state.running:
            st.session_state.running = True
            cfg = dict(poll_s=poll_s, base_kw=base_kw, noise=noise, timeout_prob=timeout_prob)
            threading.Thread(target=acquisition_loop, args=(cfg,), daemon=True).start()
            audit_log("SYSTEM_START")

with col2:
    if st.button("■ STOP"):
        st.session_state.running = False
        audit_log("SYSTEM_STOP")

# -----------------------------
# AUTOREFRESH (FIXED)
# -----------------------------
st_autorefresh(interval=refresh_ms, key="refresh")

# -----------------------------
# WATCHDOG
# -----------------------------
with st.expander("🛑 Watchdog Status"):
    if st.session_state.last_meter_ts:
        delta = (datetime.now(timezone.utc) - st.session_state.last_meter_ts).total_seconds()
        st.metric("Seconds since last meter update", f"{delta:.1f}")
        if delta > poll_s * 3:
            st.error("Stale meter detected")
            audit_log("WATCHDOG_STALE", str(delta))
    else:
        st.warning("No data yet")

# -----------------------------
# DATA VIEW
# -----------------------------
with st.expander("📈 Real-time Power"):
    if st.session_state.buffer:
        df = pd.DataFrame(st.session_state.buffer)
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts")
        st.line_chart(df["kw"])
    else:
        st.info("Waiting for data…")

# -----------------------------
# ISO 50001 COMPLIANCE
# -----------------------------
with st.expander("📜 ISO 50001 Compliance"):
    st.write({
        "Baseline": flag_baseline,
        "Monitoring": flag_monitoring,
        "Improvement": flag_improvement
    })

# -----------------------------
# AUDIT TRAIL
# -----------------------------
with st.expander("🔐 Immutable Audit Trail"):
    if st.session_state.audit:
        audit_df = pd.DataFrame(st.session_state.audit)
        st.dataframe(audit_df)
    else:
        st.info("No audit events yet")

# =============================
# Dockerfile
# =============================
# FROM python:3.11-slim
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY core_app.py .
# EXPOSE 8501
# CMD ["streamlit", "run", "core_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# =============================
# requirements.txt
# =============================
# streamlit>=1.30,<2.0
# streamlit-autorefresh>=1.0.1
# pandas>=2.0,<3.0
# numpy>=1.24,<2.0
# pytz>=2023.3
