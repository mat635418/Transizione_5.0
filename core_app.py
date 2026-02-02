import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime, timezone
import pytz
import random

# =========================
# CONFIGURATION
# =========================

REFRESH_MS = 1000
BUFFER_LIMIT = 500
ROLLING_WINDOW = 50
STALE_THRESHOLD_SEC = 5
MIN_UNITS_FOR_COMPLIANCE = 10

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

    def read(self):
        # Simulate Modbus timeout
        if random.random() < 0.03:
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
# KPI CALCULATION
# =========================

def compute_enpi(df, baseline):
    active = df[df.status == "RUNNING"].tail(ROLLING_WINDOW)

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
# STREAMLIT SETUP
# =========================

st.set_page_config(
    page_title="Transizione 5.0 Compliance Monitor",
    layout="wide",
    page_icon="🏭"
)

st.markdown("""
<style>
.stMetric {
    background: #f5f7fa;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("⚙️ System Configuration")
baseline = st.sidebar.number_input(
    "Baseline efficiency (kWh / unit)",
    value=1.20,
    step=0.01
)
st.sidebar.info("GSE requirement: **> 3% savings**")

# =========================
# SESSION STATE
# =========================

if "meter" not in st.session_state:
    st.session_state.meter = VirtualEnergyMeter()

if "buffer" not in st.session_state:
    st.session_state.buffer = pd.DataFrame(
        columns=[
            "ts_utc", "ts_cet",
            "power_kw", "energy_kwh",
            "units", "status", "comm_status"
        ]
    )

if "running" not in st.session_state:
    st.session_state.running = False

if "last_acq_time" not in st.session_state:
    st.session_state.last_acq_time = None

# =========================
# HEADER
# =========================

st.title("🏭 Transizione 5.0 Compliance Monitor")
st.caption("Real-time energy efficiency validation for tax credit eligibility (GSE)")

# =========================
# CONTROLS
# =========================

c1, c2 = st.columns([1, 1])

with c1:
    if st.button("▶ START", type="primary"):
        st.session_state.running = True
        st.session_state.buffer = st.session_state.buffer.iloc[0:0]
        st.session_state.last_acq_time = None

with c2:
    if st.button("⏹ STOP"):
        st.session_state.running = False

# =========================
# DATA ACQUISITION (BUFFERED)
# =========================

if st.session_state.running:
    try:
        reading = st.session_state.meter.read()
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
    ).tail(BUFFER_LIMIT)

# =========================
# WATCHDOG
# =========================

stale = False
if st.session_state.last_acq_time:
    stale = (time.time() - st.session_state.last_acq_time) > STALE_THRESHOLD_SEC

# =========================
# KPI COMPUTATION
# =========================

enpi, savings, produced = compute_enpi(st.session_state.buffer, baseline)

# =========================
# METRICS
# =========================

m1, m2, m3, m4 = st.columns(4)

latest = st.session_state.buffer.iloc[-1] if len(st.session_state.buffer) else None

m1.metric(
    "Active Power",
    f"{latest.power_kw:.1f} kW" if latest is not None and pd.notna(latest.power_kw) else "–"
)

m2.metric("Units (session)", int(produced) if produced else 0)

m3.metric("Current EnPI", f"{enpi:.3f} kWh/u" if enpi else "–")

compliant = (
    savings is not None and
    savings > 3 and
    produced >= MIN_UNITS_FOR_COMPLIANCE
)

m4.metric(
    "Savings vs baseline",
    f"{savings:.2f}%" if savings else "–",
    delta="COMPLIANT" if compliant else "NON-COMPLIANT",
    delta_color="normal" if compliant else "inverse"
)

# =========================
# POWER CHART
# =========================

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=st.session_state.buffer.ts_cet,
    y=st.session_state.buffer.power_kw,
    mode="lines",
    fill="tozeroy",
    name="Power (kW)"
))

fig.update_layout(
    height=350,
    margin=dict(l=0, r=0, t=30, b=0),
    title="Power Consumption (CET)"
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# STATUS & ALERTS
# =========================

if latest is not None:
    if stale:
        st.error("🚨 DATA STALE – No recent meter updates")
    elif latest.comm_status != "OK":
        st.warning("⚠ Modbus communication timeout detected")
    elif compliant:
        st.success("✅ Tax credit conditions satisfied")
    else:
        st.warning("⚠ Efficiency below GSE threshold")

# =========================
# AUDIT EXPORT
# =========================

if not st.session_state.buffer.empty:
    st.download_button(
        "⬇️ Export audit log (CSV)",
        st.session_state.buffer.to_csv(index=False),
        file_name="transizione_5_0_audit_log.csv"
    )

# =========================
# AUTO REFRESH
# =========================

if st.session_state.running:
    st.autorefresh(interval=REFRESH_MS, key="refresh")
