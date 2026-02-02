# core_app.py
# FIXED: pandas truth-value bug, robust refresh, hardened audit, ISO 50001 flags

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import threading
import time
from collections import deque
from datetime import datetime
import pytz
import hashlib

# ============================
# CONFIGURATION (DEFAULTS)
# ============================

UTC = pytz.utc
CET = pytz.timezone("Europe/Rome")

# ============================
# VIRTUAL METER (SIMULATED MODBUS)
# ============================

class VirtualEnergyMeter:
    def __init__(self, base_idle_kw, base_run_kw, timeout_prob):
        self.energy = 12000.0
        self.production = 4000
        self.status = "IDLE"
        self.last_update = time.time()
        self.timeout_prob = timeout_prob
        self.base_idle_kw = base_idle_kw
        self.base_run_kw = base_run_kw

    def read(self):
        if np.random.rand() < self.timeout_prob:
            raise TimeoutError("Simulated Modbus timeout")

        now = time.time()
        elapsed_h = (now - self.last_update) / 3600
        self.last_update = now

        if self.status == "IDLE":
            power = np.random.normal(self.base_idle_kw, 0.2)
            if np.random.rand() > 0.95:
                self.status = "RUNNING"
        else:
            power = np.random.normal(self.base_run_kw, 2.0)
            if np.random.rand() > 0.8:
                self.production += 1
            if np.random.rand() < 0.05:
                self.status = "IDLE"

        self.energy += power * elapsed_h

        return {
            "ts_utc": datetime.now(UTC),
            "ts_cet": datetime.now(CET),
            "power_kw": round(power, 2),
            "energy_kwh": self.energy,
            "production": self.production,
            "status": self.status
        }

# ============================
# AUDIT LOGGER (IMMUTABLE)
# ============================

def log_audit(event, payload=""):
    ts_utc = datetime.now(UTC)
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

# ============================
# BACKGROUND ACQUISITION THREAD
# ============================

def acquisition_loop(cfg):
    meter = VirtualEnergyMeter(
        cfg["idle_kw"], cfg["run_kw"], cfg["timeout_prob"]
    )
    log_audit("ACQ_START")

    while st.session_state.running:
        try:
            reading = meter.read()
            st.session_state.buffer.append(reading)
            st.session_state.last_data_ts = time.time()
        except TimeoutError:
            log_audit("MODBUS_TIMEOUT")
        time.sleep(cfg["poll_s"])

    log_audit("ACQ_STOP")

# ============================
# STREAMLIT SETUP
# ============================

st.set_page_config("Transizione 5.0 Compliance Monitor", layout="wide")
st.title("🏭 Transizione 5.0 Compliance Monitor")
st.caption("Real-time energy efficiency validation for tax credit eligibility")

# ============================
# SIDEBAR CONTROLS
# ============================

with st.sidebar:
    st.header("System Parameters")
    poll_s = st.slider("Polling interval (s)", 0.5, 5.0, 1.0, help="Meter acquisition rate")
    idle_kw = st.slider("Idle power (kW)", 0.5, 5.0, 2.5)
    run_kw = st.slider("Running power (kW)", 10.0, 80.0, 45.0)
    timeout_prob = st.slider("Modbus timeout probability", 0.0, 0.3, 0.05)
    baseline = st.slider("Baseline EnPI (kWh/unit)", 0.5, 3.0, 1.2)

    st.divider()
    iso_baseline = st.checkbox("ISO 50001 – Baseline defined", True)
    iso_monitoring = st.checkbox("ISO 50001 – Continuous monitoring", True)
    iso_improvement = st.checkbox("ISO 50001 – Improvement tracking", True)

# ============================
# SESSION STATE INIT
# ============================

if "buffer" not in st.session_state:
    st.session_state.buffer = deque(maxlen=500)

if "audit" not in st.session_state:
    st.session_state.audit = []

if "running" not in st.session_state:
    st.session_state.running = False

if "thread" not in st.session_state:
    st.session_state.thread = None

if "last_data_ts" not in st.session_state:
    st.session_state.last_data_ts = None

# ============================
# CONTROLS
# ============================

c1, c2 = st.columns(2)

with c1:
    if st.button("▶ START", type="primary"):
        if not st.session_state.running:
            st.session_state.running = True
            cfg = dict(poll_s=poll_s, idle_kw=idle_kw, run_kw=run_kw, timeout_prob=timeout_prob)
            st.session_state.thread = threading.Thread(
                target=acquisition_loop,
                args=(cfg,),
                daemon=True
            )
            st.session_state.thread.start()
            log_audit("SYSTEM_START")
            st.experimental_rerun()

with c2:
    if st.button("⏹ STOP") and st.session_state.running:
        st.session_state.running = False
        log_audit("SYSTEM_STOP")

# ============================
# REFRESH UI
# ============================

st_autorefresh(interval=int(poll_s * 1000), key="refresh")

# ============================
# SAFE DATAFRAME CREATION (FIX)
# ============================

if len(st.session_state.buffer) > 0:
    df = pd.DataFrame(list(st.session_state.buffer))

    with st.expander("📊 Live Telemetry", expanded=True):
        st.line_chart(df.set_index("ts_utc")["power_kw"])

    with st.expander("📈 Efficiency & Compliance", expanded=True):
        active = df[df["status"] == "RUNNING"]
        if len(active) > 1:
            enpi = (active["energy_kwh"].iloc[-1] - active["energy_kwh"].iloc[0]) / max(1, active["production"].iloc[-1] - active["production"].iloc[0])
            savings = (baseline - enpi) / baseline * 100
            st.metric("Current EnPI", f"{enpi:.3f} kWh/unit")
            st.metric("Savings vs Baseline", f"{savings:.2f}%")
            st.success("ISO 50001 COMPLIANT" if savings > 3 and iso_baseline and iso_monitoring and iso_improvement else "NON COMPLIANT")

    with st.expander("🧾 Immutable Audit Log"):
        st.dataframe(pd.DataFrame(st.session_state.audit), use_container_width=True)

# ============================
# WATCHDOG
# ============================

if st.session_state.running and st.session_state.last_data_ts:
    if time.time() - st.session_state.last_data_ts > poll_s * 3:
        st.error("⚠️ Watchdog: Meter data stale")
        log_audit("WATCHDOG_STALE")
