import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

# =========================
# CONFIG
# =========================

REFRESH_MS = 1000
ROLLING_WINDOW = 50
MIN_UNITS_FOR_COMPLIANCE = 10

# =========================
# VIRTUAL METER
# =========================

class VirtualEnergyMeter:
    def __init__(self):
        self.energy_kwh = 12500.0
        self.units = 5000
        self.status = "IDLE"
        self.last_ts = time.time()

    def read(self):
        now = time.time()
        elapsed_h = (now - self.last_ts) / 3600
        self.last_ts = now

        if self.status == "IDLE":
            power = np.random.normal(2.5, 0.2)
            if np.random.rand() > 0.96:
                self.status = "RUNNING"
        else:
            power = np.random.normal(45, 3)
            if np.random.rand() > 0.85:
                self.units += 1
            if np.random.rand() < 0.04:
                self.status = "IDLE"

        self.energy_kwh += power * elapsed_h

        return {
            "timestamp": datetime.now(),
            "power_kw": round(power, 2),
            "energy_kwh": self.energy_kwh,
            "units": self.units,
            "status": self.status
        }

# =========================
# KPI LOGIC
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
    page_title="Transizione 5.0 – Energy Monitor",
    layout="wide",
    page_icon="⚡"
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

st.sidebar.title("⚙️ Configuration")
baseline = st.sidebar.number_input(
    "Baseline efficiency (kWh/unit)",
    value=1.20,
    step=0.01
)

st.sidebar.info("GSE Target: **> 3% savings**")

# =========================
# SESSION STATE
# =========================

if "meter" not in st.session_state:
    st.session_state.meter = VirtualEnergyMeter()

if "log" not in st.session_state:
    st.session_state.log = pd.DataFrame(
        columns=["timestamp", "power_kw", "energy_kwh", "units", "status"]
    )

if "running" not in st.session_state:
    st.session_state.running = False

# =========================
# HEADER
# =========================

st.title("🏭 Transizione 5.0 Compliance Monitor")
st.caption("Real-time energy efficiency validation for tax credit eligibility")

# =========================
# CONTROLS
# =========================

c1, c2 = st.columns([1, 1])
with c1:
    if st.button("▶ START", type="primary"):
        st.session_state.running = True
        st.session_state.log = st.session_state.log.iloc[0:0]

with c2:
    if st.button("⏹ STOP"):
        st.session_state.running = False

# =========================
# DATA ACQUISITION
# =========================

if st.session_state.running:
    reading = st.session_state.meter.read()
    st.session_state.log = pd.concat(
        [st.session_state.log, pd.DataFrame([reading])],
        ignore_index=True
    ).tail(500)

    st.experimental_rerun()

# =========================
# KPI COMPUTATION
# =========================

enpi, savings, produced = compute_enpi(st.session_state.log, baseline)

# =========================
# METRICS
# =========================

m1, m2, m3, m4 = st.columns(4)

m1.metric("Power", 
          f"{st.session_state.log.power_kw.iloc[-1]:.1f} kW" if len(st.session_state.log) else "–")

m2.metric("Units (session)",
          produced if produced else 0)

m3.metric("EnPI",
          f"{enpi:.3f} kWh/u" if enpi else "–")

compliant = savings and savings > 3 and produced >= MIN_UNITS_FOR_COMPLIANCE

m4.metric(
    "Savings vs baseline",
    f"{savings:.2f}%" if savings else "–",
    delta="COMPLIANT" if compliant else "NON-COMPLIANT",
    delta_color="normal" if compliant else "inverse"
)

# =========================
# CHART
# =========================

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=st.session_state.log.timestamp,
    y=st.session_state.log.power_kw,
    fill="tozeroy",
    mode="lines",
    name="Power (kW)"
))

fig.update_layout(
    height=350,
    margin=dict(l=0, r=0, t=30, b=0),
    title="Power Consumption"
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# STATUS & COMPLIANCE
# =========================

if len(st.session_state.log):
    status = st.session_state.log.status.iloc[-1]
    color = "green" if status == "RUNNING" else "orange"

    st.markdown(f"""
    <div style="padding:20px;border-radius:10px;background:{color};color:white;text-align:center">
        <h3>Machine status</h3>
        <h1>{status}</h1>
    </div>
    """, unsafe_allow_html=True)

    if compliant:
        st.success("✅ Tax credit conditions met")
    else:
        st.warning("⚠ Efficiency not yet compliant")

# =========================
# EXPORT
# =========================

if not st.session_state.log.empty:
    st.download_button(
        "⬇️ Export audit CSV",
        st.session_state.log.to_csv(index=False),
        file_name="transizione_5_0_audit_log.csv"
    )

# =========================
# AUTO REFRESH
# =========================

if st.session_state.running:
    st.experimental_set_query_params(t=int(time.time()))
    st.autorefresh(interval=REFRESH_MS, key="refresh")
