import streamlit as st
import numpy as np
import pickle
import pandas as pd
import base64
import json
import re
from PIL import Image
import io

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
    st.success("✅ AI Model loaded — SolarCheck is ready.")
except:
    st.error("⚠️ Model files not found. Make sure solarcheck_model_v2.pkl and solarcheck_scaler_v2.pkl are uploaded to GitHub.")
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📖 How to Use")
    st.markdown("""
    **Option 1 — Photo Scan:**
    Take a photo of the spec sticker on the back of the panel. SolarCheck reads it automatically.

    **Option 2 — Manual Entry:**
    Type in the values yourself from your measurements.

    ---
    ### ⚡ Fill Factor Guide
    - **Above 0.75** → Healthy
    - **0.55 - 0.74** → Degraded
    - **Below 0.55** → Counterfeit

    ---
    ### 📍 Common Panels in Cameroon
    - Bluesun 100W: Voc=27.36V, Isc=4.87A
    - Canadian Solar 450W: Voc=48.9V, Isc=11.54A
    - Jinko 550W: Voc=49.62V, Isc=14.03A

    ---
    *SolarCheck Africa*
    *Université de Yaoundé I*
    *Founder: Gamsi*
    """)

# ── INPUT MODE SELECTOR ───────────────────────────────────────────────────────
st.markdown("### 🔍 Choose How to Check Your Panel")
input_mode = st.radio(
    "Select input method:",
    ["📸 Scan Specification Label (Photo)", "✏️ Enter Values Manually"],
    horizontal=True
)

# ── SHARED STATE FOR VALUES ───────────────────────────────────────────────────
# These defaults match the Bluesun BSM100M-36 100W panel
# The most common size panel in Cameroonian rural markets
default_values = {
    "voltage":     22.8,
    "current":     4.39,
    "irradiance":  980.0,
    "temperature": 29.0,
    "efficiency":  19.4,
    "voc":         27.36,
    "isc":         4.87
}

extracted_values = {}

# ════════════════════════════════════════════════════════════════════════════
# OPTION 1 — PHOTO SCAN MODE
# Vendor photographs the spec sticker on the back of the panel
# Gemini AI reads the image and extracts all electrical parameters
# ════════════════════════════════════════════════════════════════════════════
if "📸 Scan Specification Label (Photo)" in input_mode:

    st.markdown("---")
    st.markdown("### 📸 Scan Panel Specification Label")
    st.markdown("""
    Take a clear photo of the **specification sticker on the back of the solar panel**.
    It usually looks like a white or silver label with electrical values printed on it.

    **For best results:**
    - Hold the phone steady and close to the label
    - Make sure lighting is good — no shadow over the text
    - The whole label should be visible in the frame
    """)

    uploaded_image = st.file_uploader(
        "Upload photo of panel specification label",
        type=["jpg", "jpeg", "png", "webp"],
        help="Take a photo of the spec sticker on the back of your solar panel"
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        # Convert to RGB — fixes crash when image is PNG with transparency (RGBA)
        if image.mode in ("RGBA", "P", "LA"):
            image = image.convert("RGB")

        st.image(image, caption="Uploaded specification label", use_column_width=True)

        # Convert image to bytes to send to Gemini AI
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="JPEG")
        img_bytes = img_buffer.getvalue()

        with st.spinner("🤖 SolarCheck AI is reading your panel specifications..."):
            try:
                from google import genai
                from google.genai import types

                gemini_key = st.secrets.get("GEMINI_API_KEY", "")
                if not gemini_key:
                    st.error("Gemini API key not configured. Add GEMINI_API_KEY to Streamlit Secrets.")
                    st.stop()

                client = genai.Client(api_key=gemini_key)

                prompt = """You are reading a solar panel specification label for SolarCheck Africa, a panel quality verification system used in Cameroonian markets.

STEP 1 — Extract these electrical values from the label text:
- Pmax (Maximum Power in Watts)
- Vmp (Optimum Operating Voltage in V)
- Imp (Optimum Operating Current in A)
- Voc (Open Circuit Voltage in V)
- Isc (Short Circuit Current in A)
- Efficiency (Module Efficiency in %)
- Brand (manufacturer name printed on the label)

STEP 2 — Visually inspect the LABEL ITSELF (the sticker, not the panel) for signs of counterfeiting. Score each item honestly:
- print_quality: 0-10, how sharp and clean the printed text looks (10 = crisp professional print, 0 = blurry, smudged, or pixelated)
- font_consistency: 0-10, whether all text uses the same font, size, and alignment throughout (10 = fully consistent, 0 = mixed fonts/sizes that suggest tampering or relabeling)
- certification_marks_visible: true if you can see CE, IEC, TUV, or ISO marks anywhere on the label, false if none are visible
- red_flags: a short list of any specific suspicious details (example: "spelling error in brand name", "logo color looks slightly off", "no serial number printed")

Return ONLY a JSON object with these exact keys. Use null for any electrical value not found.
{"pmax": 100, "vmp": 22.8, "imp": 4.39, "voc": 27.36, "isc": 4.87, "efficiency": 19.4, "brand": "Bluesun",
 "label_check": {"print_quality": 8, "font_consistency": 9, "certification_marks_visible": true, "red_flags": []}}

If this is not a solar panel specification label, return: {"error": "Not a solar panel specification label"}

Return ONLY the JSON. No explanation. No other text."""

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        types.Part.from_bytes(
                            data=img_bytes,
                            mime_type="image/jpeg"
                        ),
                        prompt
                    ]
                )

                raw_text = response.text.strip()
                clean_text = re.sub(r"```json|```", "", raw_text).strip()
                specs = json.loads(clean_text)

                if "error" in specs:
                    st.error(f"⚠️ {specs['error']}. Please upload a clear photo of the panel's specification label.")
                else:
                    st.success("✅ Specification label read successfully!")

                    # Display what was extracted
                    st.markdown("#### 📋 Values Extracted From Your Label:")
                    extracted_cols = st.columns(3)
                    fields = [
                        ("Pmax (W)",       specs.get("pmax")),
                        ("Vmp (V)",        specs.get("vmp")),
                        ("Imp (A)",        specs.get("imp")),
                        ("Voc (V)",        specs.get("voc")),
                        ("Isc (A)",        specs.get("isc")),
                        ("Efficiency (%)", specs.get("efficiency")),
                    ]
                    for idx, (label, value) in enumerate(fields):
                        col = extracted_cols[idx % 3]
                        if value is not None:
                            col.metric(label, f"{value}")
                        else:
                            col.metric(label, "Not found — estimated")

                    # ── LABEL AUTHENTICITY CHECK ──────────────────────────────
                    # Second, independent signal — checks the STICKER itself for
                    # signs of counterfeiting (print quality, fonts, cert marks).
                    # This runs alongside the electrical fill-factor test below,
                    # not instead of it — two independent checks catch more fakes
                    # than either one alone.
                    st.markdown("---")
                    st.markdown("#### 🔎 Label Authenticity Check")

                    label_check      = specs.get("label_check", {})
                    print_quality    = label_check.get("print_quality", 5)
                    font_consistency = label_check.get("font_consistency", 5)
                    cert_visible     = label_check.get("certification_marks_visible", False)
                    red_flags        = label_check.get("red_flags", [])

                    la1, la2 = st.columns(2)
                    la1.metric("Print Quality",      f"{print_quality}/10")
                    la2.metric("Font Consistency",   f"{font_consistency}/10")

                    label_score = (print_quality + font_consistency) / 2
                    if not cert_visible:
                        label_score -= 1  # missing marks isn't proof of fake, but it's a flag

                    if cert_visible:
                        st.markdown("✅ Certification marks (CE/IEC/TUV) visible on label")
                    else:
                        st.markdown("⚠️ No certification marks visible — not proof of a fake on its own, but worth a closer look")

                    if red_flags:
                        st.markdown("**🚩 Red flags noticed on the label:**")
                        for flag in red_flags:
                            st.markdown(f"- {flag}")

                    if label_score < 5:
                        st.warning("⚠️ This label shows signs that may indicate a counterfeit sticker. Weigh this together with the electrical test result below — don't decide on the label alone.")
                    elif label_score < 7:
                        st.info("ℹ️ Label looks mostly fine, with a few minor inconsistencies. Cross-check with the electrical test below.")
                    else:
                        st.success("✅ Label print quality and consistency look genuine.")

                    # Store extracted values — None means not found on label
                    extracted_values = {
                        "voltage":     specs.get("vmp"),
                        "current":     specs.get("imp"),
                        "irradiance":  default_values["irradiance"],
                        "temperature": default_values["temperature"],
                        "efficiency":  specs.get("efficiency"),
                        "voc":         specs.get("voc"),
                        "isc":         specs.get("isc")
                    }
                    st.session_state["extracted"] = extracted_values
                    st.session_state["pmax_rated"] = specs.get("pmax", None)

                    # Allow user to adjust temperature and irradiance
                    st.markdown("#### 🌤️ Adjust for Current Conditions in Yaoundé")
                    adj1, adj2 = st.columns(2)
                    with adj1:
                        live_temp = st.number_input(
                            "Current Temperature (°C)",
                            min_value=15.0, max_value=50.0,
                            value=29.0, step=0.5,
                            help="Current ambient temperature where you are testing"
                        )
                    with adj2:
                        live_irr = st.number_input(
                            "Current Irradiance (W/m²)",
                            min_value=200.0, max_value=1200.0,
                            value=980.0, step=10.0,
                            help="Sunlight intensity right now. Typical Yaoundé: 950-1000"
                        )
                    st.session_state["extracted"]["temperature"] = live_temp
                    st.session_state["extracted"]["irradiance"]  = live_irr

            except json.JSONDecodeError:
                st.warning("⚠️ Could not read label clearly. Try a clearer photo or use Manual Entry.")
            except Exception as e:
                st.error(f"🔴 REAL ERROR: {type(e).__name__}: {str(e)}")

# ════════════════════════════════════════════════════════════════════════════
# OPTION 2 — MANUAL ENTRY MODE
# ════════════════════════════════════════════════════════════════════════════
if "✏️ Enter Values Manually" in input_mode:

    st.markdown("---")
    st.markdown("### ✏️ Enter Panel Readings")
    st.markdown("Type in the values from your measurements or from the panel label:")

    col1, col2 = st.columns(2)
    with col1:
        voltage     = st.number_input("Voltage — Vmp (V)",              min_value=0.0, max_value=60.0,   value=22.8,  step=0.1)
        current     = st.number_input("Current — Imp (A)",              min_value=0.0, max_value=20.0,   value=4.39,  step=0.01)
        irradiance  = st.number_input("Irradiance (W/m²)",              min_value=0.0, max_value=1200.0, value=980.0, step=10.0)
    with col2:
        temperature = st.number_input("Temperature (°C)",               min_value=0.0, max_value=50.0,   value=29.0,  step=0.5)
        efficiency  = st.number_input("Panel Efficiency (%)",           min_value=0.0, max_value=30.0,   value=19.4,  step=0.1)
        voc         = st.number_input("Open Circuit Voltage — Voc (V)", min_value=0.0, max_value=80.0,   value=27.36, step=0.1)
    isc = st.number_input("Short Circuit Current — Isc (A)",            min_value=0.0, max_value=20.0,   value=4.87,  step=0.01)

    st.session_state["extracted"] = {
        "voltage": voltage, "current": current, "irradiance": irradiance,
        "temperature": temperature, "efficiency": efficiency,
        "voc": voc, "isc": isc
    }
    st.session_state["pmax_rated"] = None

# ── SMART CALCULATED VALUES ───────────────────────────────────────────────────
# Works even when some values are missing from the panel label
# Every African panel is different — we handle all of them
# ─────────────────────────────────────────────────────────────────────────────
if "extracted" in st.session_state and st.session_state["extracted"]:
    vals = st.session_state["extracted"]

    # Step 1 — Get voltage. Use Vmp if found. If not, estimate from Voc (80% rule)
    voltage = vals["voltage"] if vals.get("voltage") else (vals.get("voc") or default_values["voc"]) * 0.8

    # Step 2 — Get current. Use Imp if found. If not, estimate from Isc (90% rule)
    current = vals["current"] if vals.get("current") else (vals.get("isc") or default_values["isc"]) * 0.9

    # Step 3 — Get Voc. Use from label if found. If not, estimate from voltage
    voc = vals["voc"] if vals.get("voc") else voltage / 0.8

    # Step 4 — Get Isc. Use from label if found. If not, estimate from current
    isc = vals["isc"] if vals.get("isc") else current / 0.9

    # Step 5 — Get efficiency. Use from label if found. If not, use standard default
    efficiency = vals["efficiency"] if vals.get("efficiency") else default_values["efficiency"]

    # Step 6 — Get temperature and irradiance — always available
    temperature = vals.get("temperature", default_values["temperature"])
    irradiance  = vals.get("irradiance",  default_values["irradiance"])

    # Step 7 — Calculate power, fill factor, and temperature-corrected efficiency
    power_output              = voltage * current
    denom                     = voc * isc
    fill_factor               = power_output / denom if denom > 0 else 0
    temp_corrected_efficiency = efficiency * (1 - (0.35 / 100) * (temperature - 25))

    st.markdown("---")
    st.markdown("### ⚡ Calculated Values")
    c1, c2, c3 = st.columns(3)
    c1.metric("Power Output (W)",       f"{power_output:.1f} W")
    c2.metric("Fill Factor",            f"{fill_factor:.3f}")
    c3.metric("Temp-Corrected Eff (%)", f"{temp_corrected_efficiency:.2f} %")

    # Fill Factor indicator
    if fill_factor >= 0.75:
        st.markdown("**Fill Factor Status:** 🟢 Excellent — consistent with genuine panel")
    elif fill_factor >= 0.55:
        st.markdown("**Fill Factor Status:** 🟡 Moderate — panel may be degraded")
    else:
        st.markdown("**Fill Factor Status:** 🔴 Low — possible counterfeit panel")

    # Rated power comparison if available from photo scan
    pmax_rated = st.session_state.get("pmax_rated")
    if pmax_rated:
        power_ratio = (power_output / pmax_rated) * 100
        st.markdown(f"**Rated Power:** {pmax_rated}W | **Measured:** {power_output:.1f}W | **Delivering:** {power_ratio:.1f}% of rated power")
        if power_ratio < 60:
            st.warning(f"⚠️ This panel is only delivering {power_ratio:.1f}% of its claimed {pmax_rated}W. Serious underperformance detected.")

    # ── PREDICT BUTTON ────────────────────────────────────────────────────────
    st.markdown("---")
    check = st.button("🔍 CHECK THIS PANEL", use_container_width=True)

    if check:
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
        bg_colors = {0: "#27ae60", 1: "#2ecc71", 2: "#f39c12", 3: "#e74c3c"}

        st.markdown("---")
        st.markdown("### 🏆 AI VERDICT")
        st.markdown(f"""
        <div style='background-color:{bg_colors[prediction]};padding:20px;
                    border-radius:10px;text-align:center;margin:10px 0;'>
            <h2 style='color:white;margin:0;'>{label_names[prediction]}</h2>
            <p style='color:white;font-size:18px;margin:5px 0;'>Confidence: {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**Recommended Action:** {actions[prediction]}")

        # Probability breakdown
        st.markdown("---")
        st.markdown("### 📊 Probability Breakdown")
        for name, prob in zip(
            ["Healthy Premium", "Healthy Basic", "Degraded", "Counterfeit"],
            probabilities
        ):
            st.progress(float(prob), text=f"{name}: {prob*100:.1f}%")

        # Full test summary
        st.markdown("---")
        st.markdown("### 📋 Full Test Summary")
        summary = {
            "Parameter": ["Voltage (V)", "Current (A)", "Power (W)", "Efficiency (%)",
                          "Fill Factor", "Irradiance (W/m²)", "Temperature (°C)",
                          "Temp-Corrected Efficiency (%)"],
            "Value":     [f"{voltage:.2f}", f"{current:.2f}",
                          f"{power_output:.1f}", f"{efficiency:.1f}",
                          f"{fill_factor:.3f}", f"{irradiance:.0f}",
                          f"{temperature:.1f}", f"{temp_corrected_efficiency:.2f}"],
            "Status":    [
                "✅" if voltage     > 15   else "⚠️",
                "✅" if current     > 3    else "⚠️",
                "✅" if power_output > 50  else "⚠️",
                "✅" if efficiency  > 12   else "⚠️",
                "✅" if fill_factor > 0.75 else ("⚠️" if fill_factor > 0.55 else "❌"),
                "✅", "✅",
                "✅" if temp_corrected_efficiency > 10 else "⚠️"
            ]
        }
        st.dataframe(pd.DataFrame(summary), use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='text-align:center;color:#95a5a6;font-size:12px;'>
    SolarCheck Africa | Université de Yaoundé I | Founder: Gamsi | 2026<br>
    <em>Changing how the world sees Africa — one panel at a time.</em>
</p>
""", unsafe_allow_html=True)
