import streamlit as st
import numpy as np
import pickle
import json

# ── PAGE CONFIGURATION ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SolarCheck Africa",
    page_icon="☀️",
    layout="centered"
)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align: center; color: #f39c12;'>☀️ SolarCheck Africa</h1>
    <h4 style='text-align: center; color: #7f8c8d;'>AI-Powered Solar Panel Quality Checker</h4>
    <p style='text-align: center; color: #95a5a6; font-size: 13px;'>
        Built for Cameroonian markets | Yaoundé, Cameroon 2026
    </p>
    <hr>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('solarcheck_model_v2.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('solarcheck_scaler_v2.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_model()
    st.success("✅ AI Model loaded successfully — SolarCheck is ready.")
except:
    st.error("⚠️ Model files not found. Make sure solarcheck_model_v2.pkl and solarcheck_scaler_v2.pkl are in the same folder as app.py")
    st.stop()

# ── SIDEBAR INFO ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. Enter the readings from your panel test
    2. Click **Check This Panel**
    3. Get instant AI verdict

    ---
    ### 🔌 Compatible With
    - Any 12V / 24V solar panel
    - Panels sold in Yaoundé, Douala, and rural Cameroon

    ---
    ### ⚡ Fill Factor Guide
    - **Above 0.75** → Healthy
    - **0.55 - 0.74** → Degraded
    - **Below 0.55** → Counterfeit

    ---
    *SolarCheck Africa*
    *Université de Yaoundé I*
    *Founder: Gamsi*
    """)

# ── INPUT FORM ────────────────────────────────────────────────────────────────
st.markdown("### 📋 Enter Panel Readings")
st.markdown("Type in the values measured from your solar panel:")

col1, col2 = st.columns(2)

with col1:
    voltage = st.number_input(
        "Voltage — Vmp (V)",
        min_value=0.0, max_value=60.0,
        value=22.8, step=0.1,
        help="Optimum operating voltage from your panel"
    )
    current = st.number_input(
        "Current — Imp (A)",
        min_value=0.0, max_value=20.0,
        value=4.39, step=0.01,
        help="Optimum operating current from your panel"
    )
    irradiance = st.number_input(
        "Irradiance (W/m²)",
        min_value=0.0, max_value=1200.0,
        value=980.0, step=10.0,
        help="Sunlight intensity at time of test. Typical Yaoundé value: 950-1000"
    )

with col2:
    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0, max_value=50.0,
        value=29.0, step=0.5,
        help="Ambient temperature at time of test"
    )
    efficiency = st.number_input(
        "Panel Efficiency (%)",
        min_value=0.0, max_value=30.0,
        value=19.4, step=0.1,
        help="Module efficiency from panel label or datasheet"
    )
    voc = st.number_input(
        "Open Circuit Voltage — Voc (V)",
        min_value=0.0, max_value=80.0,
        value=27.36, step=0.1,
        help="Voltage when no load is connected to the panel"
    )

isc = st.number_input(
    "Short Circuit Current — Isc (A)",
    min_value=0.0, max_value=20.0,
    value=4.87, step=0.01,
    help="Current when panel terminals are shorted"
)

# ── CALCULATED VALUES ─────────────────────────────────────────────────────────
power_output = voltage * current
fill_factor  = (voltage * current) / (voc * isc) if (voc * isc) > 0 else 0
temp_corrected_efficiency = efficiency * (1 - (0.35 / 100) * (temperature - 25))

st.markdown("---")
st.markdown("### ⚡ Calculated Values")
calc1, calc2, calc3 = st.columns(3)
calc1.metric("Power Output (W)",        f"{power_output:.1f} W")
calc2.metric("Fill Factor",             f"{fill_factor:.3f}")
calc3.metric("Temp-Corrected Eff (%)",  f"{temp_corrected_efficiency:.2f} %")

# ── FILL FACTOR VISUAL INDICATOR ──────────────────────────────────────────────
if fill_factor >= 0.75:
    ff_color  = "🟢"
    ff_status = "Excellent — consistent with genuine panel"
elif fill_factor >= 0.55:
    ff_color  = "🟡"
    ff_status = "Moderate — panel may be degraded"
else:
    ff_color  = "🔴"
    ff_status = "Low — possible counterfeit panel"

st.markdown(f"**Fill Factor Status:** {ff_color} {ff_status}")

# ── PREDICT BUTTON ────────────────────────────────────────────────────────────
st.markdown("---")
check_button = st.button("🔍 CHECK THIS PANEL", use_container_width=True)

if check_button:
    # Prepare input in same order as training features
    features = np.array([[
        voltage,
        current,
        irradiance,
        temperature,
        efficiency,
        power_output,
        fill_factor,
        temp_corrected_efficiency
    ]])

    features_scaled = scaler.transform(features)
    prediction      = model.predict(features_scaled)[0]
    probabilities   = model.predict_proba(features_scaled)[0]
    confidence      = probabilities[prediction] * 100

    # ── VERDICT DISPLAY ───────────────────────────────────────────────────────
    label_names = {
        0: "✅ HEALTHY PREMIUM",
        1: "✅ HEALTHY BASIC",
        2: "⚠️ DEGRADED",
        3: "❌ COUNTERFEIT"
    }
    actions = {
        0: "Safe to buy. Genuine premium panel performing as rated.",
        1: "Safe to buy. Honest basic panel, fairly priced for its specs.",
        2: "Avoid or negotiate heavily. Real panel but significantly worn out.",
        3: "DO NOT BUY. Fake panel detected. Alert the vendor and report to SolarCheck registry."
    }
    colors = {
        0: "#27ae60",
        1: "#2ecc71",
        2: "#f39c12",
        3: "#e74c3c"
    }

    st.markdown("---")
    st.markdown("### 🏆 AI VERDICT")

    verdict_html = f"""
    <div style='
        background-color: {colors[prediction]};
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    '>
        <h2 style='color: white; margin: 0;'>{label_names[prediction]}</h2>
        <p style='color: white; font-size: 18px; margin: 5px 0;'>
            Confidence: {confidence:.1f}%
        </p>
    </div>
    """
    st.markdown(verdict_html, unsafe_allow_html=True)

    st.markdown(f"**Action:** {actions[prediction]}")

    # ── PROBABILITY BREAKDOWN ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Probability Breakdown")
    category_names = ["Healthy Premium", "Healthy Basic", "Degraded", "Counterfeit"]
    for i, (name, prob) in enumerate(zip(category_names, probabilities)):
        st.progress(float(prob), text=f"{name}: {prob*100:.1f}%")

    # ── TEST SUMMARY ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Full Test Summary")
    summary_data = {
        "Parameter":  ["Voltage (V)", "Current (A)", "Power (W)", "Efficiency (%)",
                        "Fill Factor", "Irradiance (W/m²)", "Temperature (°C)",
                        "Temp-Corrected Efficiency (%)"],
        "Value":      [f"{voltage}",  f"{current}",  f"{power_output:.1f}",
                        f"{efficiency}", f"{fill_factor:.3f}", f"{irradiance}",
                        f"{temperature}", f"{temp_corrected_efficiency:.2f}"],
        "Status":     [
            "✅" if voltage > 15 else "⚠️",
            "✅" if current > 3  else "⚠️",
            "✅" if power_output > 50 else "⚠️",
            "✅" if efficiency > 12 else "⚠️",
            "✅" if fill_factor > 0.75 else ("⚠️" if fill_factor > 0.55 else "❌"),
            "✅",
            "✅",
            "✅" if temp_corrected_efficiency > 10 else "⚠️"
        ]
    }

    import pandas as pd
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #95a5a6; font-size: 12px;'>
    SolarCheck Africa | Université de Yaoundé I | Founder: Gamsi | 2026<br>
    <em>Changing how the world sees Africa — one panel at a time.</em>
</p>
""", unsafe_allow_html=True)
