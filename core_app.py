import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURATION & CLASSES ---

class VirtualEnergyMeter:
    """
    Simulates a hardware meter (like a Siemens PAC3200) connected via Modbus.
    In a real app, this would use the 'pymodbus' library to query IP 192.168.x.x
    """
    def __init__(self):
        # We initialize random starting values to simulate a running factory
        self.energy_accumulator = 12500.0  # Total kWh consumed lifetime
        self.production_accumulator = 5000 # Total units produced lifetime
        self.current_status = "IDLE"
        self.last_update = time.time()

    def read(self):
        """Generates realistic industrial data patterns."""
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now

        # Stochastic state change: Machine switches between Idle and Running
        if self.current_status == "IDLE":
            power_kw = np.random.normal(2.5, 0.1) # Low base load
            if np.random.random() > 0.95: self.current_status = "RUNNING"
        else:
            power_kw = np.random.normal(45.0, 2.0) # High working load
            # Simulate producing units (approx 1 unit every few seconds)
            if np.random.random() > 0.8: 
                self.production_accumulator += 1
            if np.random.random() < 0.05: self.current_status = "IDLE"

        # Integrate Power to Energy (kWh)
        # Power (kW) * Time (hours)
        energy_increment = power_kw * (elapsed / 3600.0)
        self.energy_accumulator += energy_increment

        return {
            "timestamp": datetime.now(),
            "active_power_kw": round(power_kw, 2),
            "total_energy_kwh": self.energy_accumulator,
            "production_count": self.production_accumulator,
            "status": self.current_status
        }

def calculate_transizione_50_metrics(df, baseline_kpi):
    """
    Calculates the Energy Performance Indicator (EnPI) required by GSE.
    Formula: Energy / Units Produced (Normalized)
    """
    if len(df) < 5:
        return 0.0, 0.0

    # Filter: We only evaluate efficiency when the machine is actually working
    # (GSE protocols often allow excluding idle time from specific efficiency metrics)
    active_data = df[df['status'] == "RUNNING"]
    
    if active_data.empty:
        return 0.0, 0.0

    # Calculate Delta (End - Start) for the current session
    energy_delta = active_data['total_energy_kwh'].iloc[-1] - active_data['total_energy_kwh'].iloc[0]
    production_delta = active_data['production_count'].iloc[-1] - active_data['production_count'].iloc[0]

    if production_delta <= 0:
        return 0.0, 0.0

    current_enpi = energy_delta / production_delta # kWh per Unit
    
    # Calculate Savings % vs Old Machine (Baseline)
    savings_pct = ((baseline_kpi - current_enpi) / baseline_kpi) * 100
    
    return current_enpi, savings_pct

# --- STREAMLIT UI SETUP ---

st.set_page_config(
    page_title="Transizione 5.0 | Energy Monitor",
    layout="wide",
    page_icon="⚡"
)

# Custom CSS to look like a Milanese Enterprise App
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stAlert { padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.title("⚙️ System Config")
st.sidebar.subheader("Asset: CNC_Milling_01")
baseline_input = st.sidebar.number_input(
    "Ex-Ante Baseline (kWh/Unit)", 
    value=1.20, 
    help="The efficiency of the old machine being replaced."
)

st.sidebar.divider()
st.sidebar.info("Transizione 5.0 Target: **> 3% Savings**")

# Session State Initialization
if 'data_log' not in st.session_state:
    st.session_state.data_log = pd.DataFrame(
        columns=["timestamp", "active_power_kw", "total_energy_kwh", "production_count", "status"]
    )
if 'meter_connection' not in st.session_state:
    st.session_state.meter_connection = VirtualEnergyMeter()
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False

# --- MAIN DASHBOARD LAYOUT ---

st.title("🏭 Transizione 5.0 Compliance Monitor")
st.markdown("Real-time telemetry for **Tax Credit Certification (GSE)**")

# Top Metrics Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
metric_placeholder_1 = kpi1.empty()
metric_placeholder_2 = kpi2.empty()
metric_placeholder_3 = kpi3.empty()
metric_placeholder_4 = kpi4.empty()

# Charts Area
col_chart_left, col_chart_right = st.columns([2, 1])
chart_power_placeholder = col_chart_left.empty()
status_placeholder = col_chart_right.empty()

# Control Buttons
c1, c2 = st.columns([1, 10])
with c1:
    start_btn = st.button("▶ START", type="primary")
with c2:
    stop_btn = st.button("⏹ STOP")

if start_btn:
    st.session_state.monitoring = True
if stop_btn:
    st.session_state.monitoring = False

# --- MAIN LOOP ---

if st.session_state.monitoring:
    # We use a placeholder container to loop without blocking the UI entirely
    while st.session_state.monitoring:
        # 1. READ DATA
        reading = st.session_state.meter_connection.read()
        
        # 2. UPDATE DATAFRAME
        new_row = pd.DataFrame([reading])
        st.session_state.data_log = pd.concat(
            [st.session_state.data_log, new_row], ignore_index=True
        )
        
        # Keep only last 500 points to preserve memory
        if len(st.session_state.data_log) > 500:
            st.session_state.data_log = st.session_state.data_log.iloc[-500:]

        # 3. CALCULATE COMPLIANCE
        current_enpi, savings_pct = calculate_transizione_50_metrics(
            st.session_state.data_log, baseline_input
        )

        # 4. VISUALIZE METRICS
        with metric_placeholder_1:
            st.metric("Real-Time Power", f"{reading['active_power_kw']} kW", delta=reading['status'])
        
        with metric_placeholder_2:
            st.metric("Units Produced (Session)", 
                      f"{reading['production_count'] - st.session_state.data_log['production_count'].iloc[0]}")

        with metric_placeholder_3:
            st.metric("Current Efficiency (EnPI)", f"{current_enpi:.3f} kWh/u")

        with metric_placeholder_4:
            # Color logic: Green if passing GSE requirements (>3%), Red if failing
            is_compliant = savings_pct > 3.0
            st.metric(
                "GSE Savings vs Baseline", 
                f"{savings_pct:.2f}%", 
                delta="COMPLIANT" if is_compliant else "NON-COMPLIANT",
                delta_color="normal" if is_compliant else "inverse"
            )

        # 5. VISUALIZE CHARTS (Using Plotly for better look)
        with chart_power_placeholder:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.data_log['timestamp'],
                y=st.session_state.data_log['active_power_kw'],
                mode='lines',
                name='Power (kW)',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy'
            ))
            fig.update_layout(
                title="Power Consumption Profile",
                xaxis_title="Time",
                yaxis_title="kW",
                height=350,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

        with status_placeholder:
            # Simple Status Indicator
            status_color = "green" if reading['status'] == "RUNNING" else "orange"
            st.markdown(f"""
                <div style="text-align: center; padding: 20px; background-color: {status_color}; color: white; border-radius: 10px;">
                    <h3>MACHINE STATUS</h3>
                    <h1>{reading['status']}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            # Compliance Alert
            if savings_pct > 3.0:
                st.success("✅ TAX CREDIT ELIGIBLE")
            else:
                st.error("❌ EFFICIENCY TOO LOW")

        # 6. SLEEP (Control refresh rate)
        time.sleep(1)
        # Check if stop was pressed (Streamlit reruns script on button click, 
        # so this loop breaks naturally on rerun if state changes, 
        # but adding an explicit break helps logic flow)
        if not st.session_state.monitoring:
            break
