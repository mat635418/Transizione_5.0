import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime, timezone
import pytz
import random

# =========================
# STREAMLIT SETUP
# =========================

st.set_page_config(
    page_title="Transizione 5.0 Compliance Monitor",
    layout="wide",
    page_icon="🏭"
)

# =========================
# SIDEBAR — CONFIGURATION
# =========================

st.sidebar.title("⚙️ System Configuration")

with st.sidebar.expander("📡 Acquisition & Refresh", expanded=True):
    refresh_sec = st.slider(
        "Refresh interval (seconds)",
        min_value=1,
        max_value=10,
        value=1,
        help="Defines how often the system acquires new data from the energy meter."
    )

    buffer_limit = st.slider(
        "Data buffer size (rows)",
        min_value=100,
        max_value=5000,
        step=100,
        value=500,
        help="Maximum number of samples retained in memory for analysis and charts."
    )

with st.sidebar.expander("🔌 Communication Reliability", expanded=True):
    timeout_probability = st.slider(
        "Modbus timeout probability (%)",
        min_value=0,
        max_value=20,
        value=3,
        help="Simulated probability of a Modbus communication timeout during acquisition."
    )

    stale_threshold = st.slider(
        "Stale data threshold (seconds)",
        min_value=2,
        max_value=30,
        value=5,
        help="Maximum allowed time without new data before triggering a stale-data alarm."
    )

with st.sidebar.expander("📊 Efficiency & GSE Logic", expanded=True):
    baseline_kwh_unit = st.number_input(
        "Baseline efficiency (kWh / unit)",
        value=1.20,
        step=0.01,
        help="Ex-ante efficiency of the replaced machine, used as GSE comparison baseline."
    )

    rolling_window = st.slider(
        "Rolling window (samples)",
        min_value=10,
        max_value=200,
        value=50,
        help="Number of recent RUNNING samples used to compute rolling EnPI."
    )

    min_units = st.slider(
        "Minimum units for compliance",
        min_value=1,
        max_value=100,
        value=10,
        help="Minimum produced units required before declaring compliance."
    )

    savings_threshold = st.slider(
        "Required savings (%)",
        min_value=1.0,
        max_value=20.0,
        value=3.0,
        step=0.5,
        help="Minimum energy savings percentage required by GSE for tax credit eligibility."
    )

st.sidebar.info("All parameters are configurable to match real GSE audit assumptions.")

# =========================
# TIMEZONE
# =========================

CET = pytz.timezone("Europe/Rome")

# =========================
# VIRTUAL MODBUS METER
# =========================

class VirtualEnergyMeter:
    def __init__(self):
        self.energy_kwh = 12500.0
        self.units = 5000
        self.status = "IDLE"
        self.last_read_ts = time.time()

    def read(self, timeout_prob):
        if random.random() < timeout_prob:
            raise TimeoutError("Modbus timeout")

        now = time.time()
        elapsed_h = (now - self.last_read_ts) / 3600
        self.last_read_ts = now

        if self.status == "IDLE":
            power = np.random.normal(2.5, 0.2)
            if random.random() > 0.96:
                self.status = "RUNNING"
        else:
            power = np.random.normal(45, 3)
            if random.random() > 0.85:
                self.units += 1
            if random.random() < 0.04:
                self.status = "IDLE"

        self.energy_kwh += max(power, 0) * elapsed_h

        ts_utc = datetime.now(timezone.utc)
        ts_cet = ts_utc.astimezone(CET)

        return {
            "ts_utc": ts_utc,
            "ts_cet": ts_cet,
            "power_kw": round(power, 2),
            "energy_kwh": self.energy_kwh,
            "units": self.units,
            "status": self.status,
            "comm_status": "OK"
        }

# =========================
# KPI LOGIC
# =========================

def compute_enpi(df, baseline, window):
    active = df[df.status == "RUNNING"].tail(window)

    if len(active) < 2:
        return None, None, 0

    d_energy = active.energy_kwh.iloc[-1] - active.energy_kwh.iloc[0]
    d_units = active.units.iloc[-1] - active.units.iloc[0]

    if d_units <= 0:
        return None, None, d_units

    enpi = d_energy / d_units
    savings = (baseline - enpi) / baseline * 100

    return enpi, savings, d_units

# =========================
# SESSION STATE
# =========================

if "meter" not in st.session_state:
    st.session_state.meter = VirtualEnergyMeter()

if "buffer" not in st.session_state:
    st.session_state.buffer = pd.DataFrame(
        columns=["ts_utc", "ts_cet", "power_kw",
                 "energy_kwh", "units", "status", "comm_status"]
    )

if "running" not in st.session_state:
    st.session_state.running = False

if "last_acq_time" not in st.session_state:
    st.session_state.last_acq_time = None

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

# =========================
# HEADER
# =========================

st.title("🏭 Transizione 5.0 Compliance Monitor")
st.caption("Configurable, audit-ready energy efficiency validation for GSE tax credit eligibility")

# =========================
# CONTROLS
# =========================

c1, c2 = st.columns(2)

with c1:
    if st.button("▶ START", type="primary"):
        st.session_state.running = True
        st.session_state.buffer = st.session_state.buffer.iloc[0:0]
        st.session_state.last_acq_time = None
        st.session_state.last_refresh = time.time()

with c2:
    if st.button("⏹ STOP"):
        st.session_state.running = False

# =========================
# MAIN CONTENT
# =========================

with st.expander("📡 Live Data Acquisition", expanded=True):
    st.write("This section shows real-time power data acquired from the energy meter.")

    if st.session_state.running:
        try:
            reading = st.session_state.meter.read(timeout_probability / 100)
            st.session_state.last_acq_time = time.time()
        except TimeoutError:
            reading = {
                "ts_utc": datetime.now(timezone.utc),
                "ts_cet": datetime.now(timezone.utc).astimezone(CET),
                "power_kw": np.nan,
                "energy_kwh": np.nan,
                "units": np.nan,
                "status": "FAULT",
                "comm_status": "TIMEOUT"
            }

        st.session_state.buffer = pd.concat(
            [st.session_state.buffer, pd.DataFrame([reading])],
            ignore_index=True
        ).tail(buffer_limit)

with st.expander("📊 Efficiency KPIs & GSE Compliance", expanded=True):
    st.write("""
    Energy Performance Indicators (EnPI) are computed on a rolling basis,
    considering only productive machine states, in accordance with GSE methodology.
    """)

    enpi, savings, produced = compute_enpi(
        st.session_state.buffer,
        baseline_kwh_unit,
        rolling_window
    )

    latest = st.session_state.buffer.iloc[-1] if len(st.session_state.buffer) else None
    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Active Power",
              f"{latest.power_kw:.1f} kW" if latest is not None and pd.notna(latest.power_kw) else "–")

    m2.metric("Units (session)", int(produced) if produced else 0)
    m3.metric("Current EnPI", f"{enpi:.3f} kWh/u" if enpi else "–")

    compliant = (
        savings is not None and
        savings > savings_threshold and
        produced >= min_units
    )

    m4.metric(
        "Savings vs baseline",
        f"{savings:.2f}%" if savings else "–",
        delta="COMPLIANT" if compliant else "NON-COMPLIANT",
        delta_color="normal" if compliant else "inverse"
    )

with st.expander("📈 Power Consumption Trend", expanded=True):
    st.write("Historical power profile used to support efficiency analysis.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=st.session_state.buffer.ts_cet,
        y=st.session_state.buffer.power_kw,
        fill="tozeroy",
        mode="lines",
        name="Power (kW)"
    ))
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

with st.expander("🚨 System Status & Alerts", expanded=True):
    stale = False
    if st.session_state.last_acq_time:
        stale = (time.time() - st.session_state.last_acq_time) > stale_threshold

    if latest is not None:
        if stale:
            st.error("🚨 Data is stale — no recent meter updates")
        elif latest.comm_status != "OK":
            st.warning("⚠ Modbus communication timeout detected")
        elif compliant:
            st.success("✅ GSE tax credit conditions satisfied")
        else:
            st.warning("⚠ Efficiency below required threshold")

with st.expander("🗂 Audit & Data Export", expanded=False):
    st.write("Download the full audit trail used for certification and traceability.")

    if not st.session_state.buffer.empty:
        st.download_button(
            "⬇️ Export audit log (CSV)",
            st.session_state.buffer.to_csv(index=False),
            file_name="transizione_5_0_audit_log.csv"
        )

# =========================
# HEARTBEAT REFRESH
# =========================

if st.session_state.running:
    now = time.time()
    if now - st.session_state.last_refresh >= refresh_sec:
        st.session_state.last_refresh = now
        st.experimental_set_query_params(t=int(now))
