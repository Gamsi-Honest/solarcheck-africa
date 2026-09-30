"""
SolarCheck Africa — GSI (General Superintelligence) Mode.

Version 3 replaces the single-shot classifier UI with a cognitive
architecture, exposed so a market vendor can actually use it:

    law layer  →  council of specialists  →  fusion  →  metacognition
               →  decision, money-at-risk and a safe deployment plan.

Two ideas drive the interface:

  1. Every number is traceable. Each verdict can be opened up to the agent,
     law or formula that produced it, including the points where the system
     disagrees with itself.
  2. "I don't know yet, measure this" is a valid, first-class answer. The
     engine would rather tell a vendor which measurement settles the question
     than hand over a confident guess.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ── PAGE CONFIGURATION ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="SolarCheck Africa — GSI Mode",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

VERSION = "3.0.0-gsi"
HERE = os.path.dirname(os.path.abspath(__file__))


def _full_width_kwarg():
    """Streamlit renamed `use_container_width` to `width="stretch"`.

    Support both so the same file runs on a pinned older Streamlit (as in the
    devcontainer) and on a current release, without deprecation noise.
    """
    try:
        from streamlit import __version__ as _version

        parts = tuple(int(part) for part in _version.split(".")[:2] if part.isdigit())
        if parts >= (1, 49):
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


FULL_WIDTH = _full_width_kwarg()

CLASS_TONE = {
    0: ("#1e8449", "✅"),
    1: ("#27ae60", "✅"),
    2: ("#d68910", "⚠️"),
    3: ("#c0392b", "❌"),
}
# States that must not borrow a verdict colour, because no verdict was reached.
NEUTRAL_TONE = ("#566573", "🧪")
STATE_LABEL = {
    "VERDICT": "⚖️ Verdict",
    "PROBABLE": "🎯 Probable",
    "INCONCLUSIVE": "🤔 Inconclusive",
    "MEASUREMENT_ERROR": "🧪 Measurement invalid",
}


# ── MODEL / ENGINE LOADING ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    import pickle

    with open(os.path.join(HERE, "solarcheck_model_v2.pkl"), "rb") as fh:
        model = pickle.load(fh)
    with open(os.path.join(HERE, "solarcheck_scaler_v2.pkl"), "rb") as fh:
        scaler = pickle.load(fh)
    return model, scaler


@st.cache_resource(show_spinner=False)
def load_engine():
    """Build the GSI engine once per session, including probability calibration."""
    from gsi import GSIEngine
    from gsi import agents as gsi_agents

    model, scaler = load_model()
    priors = gsi_agents.load_priors(os.path.join(HERE, "gsi", "data", "feature_priors.json"))
    engine = GSIEngine(model, scaler, priors=priors)
    note = engine.calibrate_from_csv(os.path.join(HERE, "solarcheck_training_data_v2.csv"))
    return engine, note


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 GSI Mode")
    st.caption(
        "A council of specialist agents + physics law enforcement + metacognition. "
        "Not one model guessing — seven channels arguing, with an audit trail."
    )

    from gsi.regions import labels as region_labels

    region_keys = list(region_labels().keys())
    region_choice = st.selectbox(
        "📍 Market / location",
        region_keys,
        index=region_keys.index("yaounde") if "yaounde" in region_keys else 0,
        format_func=lambda k: region_labels()[k],
        help="Solar yield, grid tariff, price bands and counterfeit prevalence differ by market.",
    )

    st.markdown("---")
    st.markdown("### 📖 How to get a good verdict")
    st.markdown(
        """
1. **Photograph the spec sticker** on the back of the panel (or type the values).
2. Add **photos of the panel surface** if you can — the forensics agent reads
   busbar periodicity, glare, colour cast and duplicate listings.
3. Enter the **asking price**. Half the value of this tool is the money maths.
4. Read the **counter-case** before you decide. It is the system arguing
   against itself.
"""
    )

    st.markdown("---")
    with st.expander("⚙️ Engine status", expanded=False):
        try:
            engine_probe, note = load_engine()
            st.markdown(f"**Version:** `{VERSION}`")
            st.markdown(f"**Calibration:** {note}")
            priors_ok = engine_probe.priors is not None
            st.markdown(f"**Statistical priors:** {'loaded ✅' if priors_ok else 'missing ⚠️'}")
            from gsi import agents as gsi_agent_names

            st.markdown(
                "**Agents online:** "
                + ", ".join(
                    [
                        gsi_agent_names.A_ML,
                        gsi_agent_names.A_PHYSICS,
                        gsi_agent_names.A_PERCEPTION,
                        gsi_agent_names.A_LABEL,
                        gsi_agent_names.A_ECON,
                        gsi_agent_names.A_DEPLOY,
                        gsi_agent_names.A_RED,
                    ]
                )
            )
        except Exception as exc:  # pragma: no cover
            st.error(f"Engine could not start: {exc}")

    with st.expander("🎯 Physical limits the engine enforces", expanded=False):
        st.markdown(
            """
- `Imp ≤ Isc`, `Vmp < Voc` — non-negotiable
- Fill factor ceiling **0.88** (single-junction silicon)
- Efficiency ceiling **33.7%** (Shockley-Queisser detailed balance)
- Temperature coefficient **−0.29 … −0.45 %/°C**
- Voc drift **−0.30 %/°C** (cold-morning string sizing uses this)
- Isc scales ~linearly with irradiance
"""
        )

    st.markdown("---")
    st.caption(
        "**SolarCheck Africa** · Université de Yaoundé I\n\nFounder: **Gamsi** · 2026\n\n"
        "*Changing how the world sees Africa — one panel at a time.*"
    )


# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <h1 style='text-align:center;color:#f39c12;margin-bottom:0;'>☀️ SolarCheck Africa</h1>
    <p style='text-align:center;color:#7f8c8d;font-size:18px;margin-top:4px;'>
        <b>GSI Mode</b> · a council of specialists, physics law enforcement and metacognition
    </p>
    """,
    unsafe_allow_html=True,
)

# ── INPUT SECTION ────────────────────────────────────────────────────────────
DEFAULTS = {
    "voltage": 22.8,
    "current": 4.39,
    "irradiance": 980.0,
    "temperature": 29.0,
    "efficiency": 19.4,
    "voc": 27.36,
    "isc": 4.87,
}

if "label_specs" not in st.session_state:
    st.session_state["label_specs"] = None
if "label_images" not in st.session_state:
    st.session_state["label_images"] = []
if "surface_images" not in st.session_state:
    st.session_state["surface_images"] = []

tab_label, tab_manual, tab_photos = st.tabs(
    ["📸 Scan spec sticker", "✏️ Type the values", "🔬 Bench photos (forensics)"]
)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — GEMINI OCR OF THE SPEC STICKER
# ════════════════════════════════════════════════════════════════════════════
OCR_PROMPT = """You are reading a solar panel specification label for SolarCheck Africa, a panel quality verification system used in African markets.

STEP 1 — Extract these electrical values printed on the label:
- Pmax (Maximum Power in Watts)
- Vmp (Optimum Operating Voltage in V)
- Imp (Optimum Operating Current in A)
- Voc (Open Circuit Voltage in V)
- Isc (Short Circuit Current in A)
- Efficiency (Module Efficiency in %)
- Brand (manufacturer name printed on the label)
- model (the model code, e.g. BSM100M-36)
- all_text (every piece of legible text on the sticker, verbatim, as one string)

STEP 2 — Visually inspect the LABEL ITSELF (the sticker, not the panel) for signs of counterfeiting. Score each item honestly. If the photo is blurred or glared, say so in red_flags rather than guessing.
- print_quality: 0-10, how sharp and clean the printed text looks (10 = crisp professional print, 0 = blurry, smudged, or pixelated)
- font_consistency: 0-10, whether all text uses the same font, size, and alignment throughout (10 = fully consistent, 0 = mixed fonts/sizes suggesting tampering or relabelling)
- certification_marks_visible: true if you can see CE, IEC, TUV, or ISO marks anywhere on the label, false if none are visible
- red_flags: a short list of specific suspicious details (example: "spelling error in brand name", "logo colour looks slightly off", "no serial number printed", "photo too blurred to judge print quality")

Return ONLY a JSON object with these exact keys. Use null for any electrical value not found.
{"pmax": 100, "vmp": 22.8, "imp": 4.39, "voc": 27.36, "isc": 4.87, "efficiency": 19.4, "brand": "Bluesun", "model": "BSM100M-36", "all_text": "BLUESUN ...",
 "label_check": {"print_quality": 8, "font_consistency": 9, "certification_marks_visible": true, "red_flags": []}}

If this is not a solar panel specification label, return: {"error": "Not a solar panel specification label"}

Return ONLY the JSON. No explanation. No other text."""


def get_gemini_client():
    from google import genai

    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""
    key = key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return None, (
            "Gemini API key not configured. Add `GEMINI_API_KEY` to Streamlit Secrets "
            "(or the environment) to enable photo scanning — or type the values in the "
            "second tab, which runs the exact same council."
        )
    return genai.Client(api_key=key), None


def judge_image(image: Image.Image, cache_key: str):
    """Run Gemini once per unique image; cache the extraction in session state."""
    cache = st.session_state.setdefault("ocr_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    client, err = get_gemini_client()
    if err:
        return {"error": err}

    from google.genai import types

    if image.mode in ("RGBA", "P", "LA"):
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg"),
                OCR_PROMPT,
            ],
        )
        raw = (response.text or "").strip()
        cleaned = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        data = {"error": "The model returned text I could not parse as specifications. Try a clearer photo."}
    except Exception as exc:
        data = {"error": f"{type(exc).__name__}: {exc}"}

    cache[cache_key] = data
    return data


with tab_label:
    st.markdown(
        """
Take a clear photo of the **specification sticker on the back of the panel** — the white or
silver label with the electrical values on it. The vision model reads the numbers *and*
inspects the sticker itself for printing quality, font consistency and certification marks.
"""
    )
    uploaded = st.file_uploader(
        "Photo of the panel specification label",
        type=["jpg", "jpeg", "png", "webp"],
        key="label_uploader",
    )

    if uploaded is not None:
        digest = hashlib.sha256(uploaded.getvalue()).hexdigest()[:16]
        image = Image.open(uploaded)
        if image.mode in ("RGBA", "P", "LA"):
            image = image.convert("RGB")
        st.session_state["label_images"] = [image]

        left, right = st.columns([1, 1])
        with left:
            st.image(image, caption="Uploaded specification label")
        with right:
            with st.spinner("🧠 Reading the sticker and inspecting it for forgery…"):
                specs = judge_image(image, digest)

            if specs.get("error"):
                st.error(f"⚠️ {specs['error']}")
            else:
                st.success("✅ Sticker read.")
                st.session_state["label_specs"] = specs
                fields = [
                    ("Pmax (W)", specs.get("pmax")),
                    ("Vmp (V)", specs.get("vmp")),
                    ("Imp (A)", specs.get("imp")),
                    ("Voc (V)", specs.get("voc")),
                    ("Isc (A)", specs.get("isc")),
                    ("Efficiency (%)", specs.get("efficiency")),
                ]
                cols = st.columns(3)
                for idx, (name, value) in enumerate(fields):
                    cols[idx % 3].metric(name, "—" if value is None else f"{value}")
                if specs.get("brand"):
                    st.markdown(f"**Brand read from label:** {specs['brand']} {specs.get('model') or ''}")

    if st.session_state["label_specs"] and not st.session_state["label_specs"].get("error"):
        specs = st.session_state["label_specs"]
        st.markdown("---")
        st.markdown("#### 🔎 What the sticker itself says")
        check = specs.get("label_check") or {}
        c1, c2, c3 = st.columns(3)
        c1.metric("Print quality", f"{check.get('print_quality', '—')}/10")
        c2.metric("Font consistency", f"{check.get('font_consistency', '—')}/10")
        c3.metric("Certification marks", "present" if check.get("certification_marks_visible") else "not visible")
        for flag in check.get("red_flags") or []:
            st.warning(f"🚩 {flag}")
        st.caption(
            "These sticker readings are *evidence for the council*, not a verdict. The LabelForge "
            "agent also checks the brand string against known manufacturers for lookalike spellings."
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — MANUAL ENTRY
# ════════════════════════════════════════════════════════════════════════════
with tab_manual:
    st.markdown(
        "Type in what you measured. **Leave a field at 0 if you do not have it** — the engine will "
        "estimate it, say so out loud, and tell you which missing measurement matters most."
    )
    m1, m2, m3 = st.columns(3)
    with m1:
        voltage = st.number_input("Vmp — Voltage at max power (V)", 0.0, 90.0, DEFAULTS["voltage"], 0.1)
        current = st.number_input("Imp — Current at max power (A)", 0.0, 20.0, DEFAULTS["current"], 0.01)
        irradiance = st.number_input("Irradiance (W/m²)", 0.0, 1400.0, DEFAULTS["irradiance"], 10.0)
    with m2:
        temperature = st.number_input("Cell/ambient temperature (°C)", 0.0, 80.0, DEFAULTS["temperature"], 0.5)
        efficiency = st.number_input("Module efficiency (%)", 0.0, 40.0, DEFAULTS["efficiency"], 0.1)
        pmax = st.number_input("Nameplate power on the label (W)", 0.0, 800.0, 100.0, 5.0)
    with m3:
        voc = st.number_input("Voc — Open circuit voltage (V)", 0.0, 100.0, DEFAULTS["voc"], 0.1)
        isc = st.number_input("Isc — Short circuit current (A)", 0.0, 25.0, DEFAULTS["isc"], 0.01)
        price = st.number_input("Asking price (local currency, 0 = unknown)", 0.0, 5_000_000.0, 35000.0, 500.0)

    entered = {
        "voltage": voltage,
        "current": current,
        "irradiance": irradiance,
        "temperature": temperature,
        "efficiency": efficiency,
        "voc": voc,
        "isc": isc,
    }
    # A field left sitting at the seeded default is an *assumption*, not a
    # measurement. Claiming otherwise would hide the estimate from the engine's
    # honesty accounting, so only values the operator actually changed count.
    st.session_state["manual_values"] = {
        **entered,
        "pmax": pmax,
        "price": price,
        "provided": tuple(
            name
            for name, value in entered.items()
            if value and value > 0 and abs(float(value) - float(DEFAULTS[name])) > 1e-9
        ),
    }


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — BENCH PHOTOS FOR IMAGE FORENSICS
# ════════════════════════════════════════════════════════════════════════════
with tab_photos:
    st.markdown(
        """
Add **front-of-panel photos** (ideally 2–3 from different angles, in even daylight). The
perception agent measures focus, glare, exposure, colour cast and the busbar grid periodicity,
and compares the photos against each other for near-duplicates.

It also refuses to score a sticker on a blurred photo, and will tell you to retake it.
"""
    )
    surface_files = st.file_uploader(
        "Panel surface photos",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="surface_uploader",
    )
    if surface_files:
        images = []
        cols = st.columns(min(4, len(surface_files)))
        for idx, file in enumerate(surface_files):
            img = Image.open(file)
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            images.append(img)
            cols[idx % len(cols)].image(img, caption=f"Photo {idx+1}")
        st.session_state["surface_images"] = images
        st.caption(f"{len(images)} photo(s) queued for the forensics agent.")


# ════════════════════════════════════════════════════════════════════════════
# RUN THE ENGINE
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")


def build_inputs():
    """Combine whichever input path the operator used into one PanelInputs."""
    from gsi import PanelInputs

    specs = st.session_state.get("label_specs") or {}
    manual = {
        k: v
        for k, v in (st.session_state.get("manual_values") or {}).items()
        if k not in ("provided", "pmax", "price")
    }

    source = "manual"
    values = dict(manual)
    provided = set(st.session_state.get("manual_values", {}).get("provided", ()))
    pmax = st.session_state.get("manual_values", {}).get("pmax", 0.0) or None
    price = st.session_state.get("manual_values", {}).get("price", 0.0) or None

    if specs and not specs.get("error"):
        # Sticker values win where the manual form was left at its default.
        label_provided = set()
        for key, spec_key in (("voltage", "vmp"), ("current", "imp"), ("voc", "voc"), ("isc", "isc"), ("efficiency", "efficiency")):
            spec_value = specs.get(spec_key)
            if spec_value:
                values[key] = float(spec_value)
                label_provided.add(key)
            else:
                values.setdefault(key, None)
        if specs.get("pmax") and not pmax:
            pmax = float(specs["pmax"])
        provided = provided | label_provided
        source = "mixed" if manual or provided != label_provided else "label"

    # Irradiance and temperature are live bench conditions, not sticker values:
    # they count as measured when the operator moved them off the seed value.
    live = {
        name
        for name in ("irradiance", "temperature")
        if st.session_state.get("manual_values", {}).get(name) is not None
        and abs(
            float(st.session_state["manual_values"][name]) - float(DEFAULTS[name])
        )
        > 1e-9
    }
    provided = provided | live

    images = list(st.session_state.get("label_images") or []) + list(st.session_state.get("surface_images") or [])

    return PanelInputs(
        voltage=values.get("voltage"),
        current=values.get("current"),
        irradiance=values.get("irradiance"),
        temperature=values.get("temperature"),
        efficiency=values.get("efficiency"),
        voc=values.get("voc"),
        isc=values.get("isc"),
        pmax_rated=pmax,
        brand=(specs.get("brand") if specs else None),
        model_text=(specs.get("all_text") if specs else None),
        label_check=(specs.get("label_check") if specs else None),
        images=images,
        price=price,
        region=region_choice,
        source=source,
        provided=tuple(sorted(provided)),
    )


run_col, note_col = st.columns([1, 2])
with run_col:
    run_clicked = st.button("🧠 RUN GSI ANALYSIS", type="primary", **FULL_WIDTH)
with note_col:
    st.caption(
        "The council runs on everything you provided. Nothing is uploaded anywhere except the "
        "optional Gemini sticker read."
    )

if run_clicked:
    try:
        engine, note = load_engine()
        inputs = build_inputs()
        with st.spinner("🧠 Council deliberating — law checks, fusion, metacognition, counter-case…"):
            report = engine.reason(inputs)
        st.session_state["gsi_report"] = report
        st.session_state["gsi_inputs"] = inputs
        st.session_state["gsi_ran_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception as exc:
        st.error(f"🔴 Engine error: {type(exc).__name__}: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())


# ════════════════════════════════════════════════════════════════════════════
# REPORT RENDERING
# ════════════════════════════════════════════════════════════════════════════
report = st.session_state.get("gsi_report")

if report is None:
    st.info(
        "### 🧠 GSI mode is ready\n"
        "Feed it a sticker photo, typed measurements, or both. You will get a verdict **with the "
        "argument for and against it**, the money at risk, a safe way to wire the panel, and — when "
        "the evidence is thin — an honest *inconclusive* plus the single measurement that would "
        "settle the question."
    )
else:
    if report.state in ("MEASUREMENT_ERROR", "INCONCLUSIVE"):
        tone, icon = NEUTRAL_TONE
    else:
        tone, icon = CLASS_TONE[report.verdict_class]
    st.markdown("---")

    # ── Verdict banner ─────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style='background:{tone};padding:22px;border-radius:14px;text-align:center;'>
            <div style='color:#fff;opacity:.85;font-size:14px;letter-spacing:2px;'>
                {STATE_LABEL.get(report.state, report.state).upper()}
            </div>
            <h2 style='color:white;margin:6px 0;'>{icon} {report.display_label}</h2>
            <p style='color:white;font-size:17px;margin:0;'>
                Effective confidence: <b>{report.confidence*100:.0f}%</b>
                &nbsp;·&nbsp; council of {len([o for o in report.opinions if o.verdict is not None])} voting agents
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Run at {st.session_state.get('gsi_ran_at', '')} · engine {report.engine_version}")

    if report.veto_reason:
        st.error(f"⛔ **{report.veto_reason}**")

    # ── Decision ───────────────────────────────────────────────────────────
    st.markdown("### 🧭 Decision")
    st.markdown(f"**{report.decision}**")

    # ── Narrative ──────────────────────────────────────────────────────────
    st.markdown("### 🗣️ The argument")
    st.markdown(report.narrative)

    # ── Probabilities + uncertainty side by side ───────────────────────────
    pc1, pc2 = st.columns([1, 1])
    with pc1:
        st.markdown("#### 📊 Belief across the four hypotheses")
        for name, prob in report.probabilities.items():
            st.progress(float(prob), text=f"{name.replace('_', ' ')}: {prob*100:.1f}%")
        st.caption(
            "Weighted logarithmic pooling of every agent's independent evidence channel, "
            "sharpened by regional base rates for the counterfeit class."
        )
    with pc2:
        st.markdown("#### 🎲 Uncertainty decomposition")
        unc = report.uncertainty
        u1, u2, u3 = st.columns(3)
        u1.metric("Total", f"{unc.total_entropy_bits:.2f} bits")
        u2.metric("Inherent", f"{unc.aleatoric_bits:.2f} bits")
        u3.metric("Removable", f"{unc.epistemic_bits:.2f} bits")
        st.progress(float(unc.agreement), text=f"Council agreement: {unc.agreement*100:.0f}%")
        st.markdown(unc.narrative)
        st.caption(
            "Total entropy is split into irreducible ambiguity in the reading and the part that "
            "more evidence would remove. High agreement with high entropy means the council is "
            "confidently unsure — which is different from disagreeing."
        )

    # ── Next best measurements (the planner) ───────────────────────────────
    if report.next_tests:
        st.markdown("### 🎯 The measurements that would change this answer")
        st.caption(
            "Expected information gain: how much of the remaining uncertainty each measurement "
            "would remove, computed by re-running the council with the class-typical outcome of "
            "each test, weighted by the current belief."
        )
        for test in report.next_tests:
            with st.container(border=True):
                t1, t2 = st.columns([1, 3])
                t1.metric(test.feature, f"{test.expected_information_gain_bits:.2f} bits")
                t2.markdown(f"**How:** {test.how}\n\n{test.why}")

    # ── Council table ──────────────────────────────────────────────────────
    st.markdown("### 🏛️ The council")
    rows = []
    for op in report.opinions:
        rows.append(
            {
                "Agent": op.agent,
                "Verdict": "abstained" if op.verdict is None else op.label.replace("_", " "),
                "Score": round(op.score, 2),
                "Confidence": f"{op.confidence*100:.0f}%",
                "Vote weight": round(op.weight, 2),
                "Their reasoning": op.rationale[:150],
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, **FULL_WIDTH)

    for op in report.opinions:
        title = f"{op.agent} — {'abstained' if op.verdict is None else op.label.replace('_', ' ')} ({op.confidence*100:.0f}% confident)"
        with st.expander(title, expanded=False):
            st.markdown(f"*{op.rationale}*")
            if not op.findings:
                st.caption("No findings recorded.")
            for finding in op.findings:
                icons = {"good": "✅", "info": "ℹ️", "warn": "⚠️", "critical": "⛔"}
                weight = f" _(evidence weight {finding.weight:+.2f})_" if abs(finding.weight) > 0.001 else ""
                st.markdown(f"{icons.get(finding.severity, '•')} **{finding.code}** — {finding.message}{weight}")

    # ── Metacognition ──────────────────────────────────────────────────────
    st.markdown("### 🪞 Metacognition — the system auditing itself")
    if not report.metacognition:
        st.caption("No self-audit concerns raised on this reading.")
    for finding in report.metacognition:
        icons = {"good": "✅", "info": "ℹ️", "warn": "⚠️", "critical": "⛔"}
        st.markdown(f"{icons.get(finding.severity, '•')} **{finding.code}** — {finding.message}")

    # ── Red team ───────────────────────────────────────────────────────────
    if report.counter_case and report.counter_case.arguments:
        st.markdown("### 😈 The case against this verdict")
        st.markdown(
            f"*{report.counter_case.summary}* "
            f"(challenger: **{['HEALTHY_PREMIUM','HEALTHY_BASIC','DEGRADED','COUNTERFEIT'][report.counter_case.target_class]}**)"
        )
        for idx, arg in enumerate(report.counter_case.arguments, 1):
            with st.container(border=True):
                st.markdown(f"**{idx}. {arg['claim']}**")
                st.markdown(
                    f"- **Test that settles it:** {arg['test']}\n"
                    f"- **Cost / time:** {arg['cost']} · {arg['duration']}\n"
                    f"- **If it fails, the verdict flips to:** `{arg['flips_to']}`"
                )

    # ── Physics ────────────────────────────────────────────────────────────
    st.markdown("### ⚡ What physics found")
    phys = report.physics
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Fill factor", f"{phys.fill_factor:.3f}")
    p2.metric("Measured power", f"{phys.power_output:.1f} W")
    p3.metric("Power at STC", f"{phys.temp_corrected_power:.1f} W")
    p4.metric("Checks run", f"{phys.checks_run}")
    if phys.dataclass_match and phys.dataclass_match.get("confidence", 0) > 0.4:
        match = phys.dataclass_match
        st.info(
            f"🔍 Electrical fingerprint match: **{match['brand']} {match['model']}** "
            f"({match['pmax']:g} W, {match['efficiency']}% efficiency, {match['technology']}) — "
            f"{match['confidence']*100:.0f}% confidence."
        )
    for finding in phys.findings:
        icons = {"good": "✅", "info": "ℹ️", "warn": "⚠️", "critical": "⛔"}
        st.markdown(f"{icons.get(finding.severity, '•')} {finding.message}")

    # ── Economics ──────────────────────────────────────────────────────────
    econ = report.economics
    if econ:
        st.markdown(f"### 💰 Money — {report.region.name}, {report.region.country}")
        e1, e2, e3, e4 = st.columns(4)
        sym = report.region.currency_symbol
        e1.metric("Price per watt", "—" if econ.price_per_watt is None else f"{econ.price_per_watt:.0f} {sym}")
        e2.metric("Honest band", "—" if not econ.band else f"{econ.band[0]}–{econ.band[1]} {sym}/W")
        e3.metric("Money at risk", f"{econ.money_at_risk:,.0f} {sym}")
        e4.metric("Payback", "—" if econ.payback_years is None else f"{econ.payback_years:.1f} yrs")
        st.markdown(
            f"- **Yield estimate:** {econ.daily_kwh:.2f} kWh/day · "
            f"{econ.daily_kwh*365:.0f} kWh/year at {report.region.peak_sun_hours:.1f} peak-sun-hours\n"
            f"- **Grid tariff used:** {report.region.tariff_per_kwh:g} {sym}/kWh "
            f"(grid reliability {report.region.grid_reliability*100:.0f}%)\n"
            + (
                f"- **Levelised comparison:** solar {econ.solar_cost_per_kwh:,.0f} {sym}/kWh vs "
                f"petrol generator {econ.generator_cost_per_kwh:,.0f} {sym}/kWh\n"
                if econ.solar_cost_per_kwh and econ.generator_cost_per_kwh
                else ""
            )
            + f"- **Market context:** counterfeit prevalence reported at "
              f"{report.region.counterfeit_prevalence*100:.0f}% of panels in circulation."
        )
        st.caption(report.region.notes or "")

    # ── Deployment ─────────────────────────────────────────────────────────
    plan = report.deployment
    if plan:
        st.markdown("### 🛠️ Safe way to build with this panel")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Max in series", f"{plan.max_series}")
        d2.metric("Cable (10 m run)", f"{plan.cable_mm2:.1f} mm²")
        d3.metric("Fuse per string", f"{plan.fuse_a:.0f} A")
        d4.metric("Controller", f"{plan.controller_a:.0f} A")
        st.markdown(
            f"- **Voc at 15 °C (cold dawn):** {plan.voc_cold:.1f} V per panel\n"
            f"- **Array estimate:** {plan.array_kwp:.3f} kWp → {plan.daily_kwh:.2f} kWh/day "
            f"({plan.annual_kwh:.0f} kWh/year)\n"
            f"- **One day of battery storage:** ~{plan.battery_ah_12v:.0f} Ah at 12 V (50% depth of discharge)"
        )
        for finding in plan.findings:
            st.markdown(f"ℹ️ {finding.message}")

    # ── Audit trail + export ───────────────────────────────────────────────
    st.markdown("### 🧾 Audit trail")
    st.caption(
        "Every stage of the reasoning, including fusion weights and the metacognition checks that "
        "fired. This is what makes the verdict reviewable by someone else."
    )
    with st.expander("Show the full audit trail", expanded=False):
        st.json(report.audit)

    payload = report.to_dict()
    payload["inputs"] = {
        "region": report.region.key,
        "price": st.session_state.get("gsi_inputs").price if st.session_state.get("gsi_inputs") else None,
        "source": st.session_state.get("gsi_inputs").source if st.session_state.get("gsi_inputs") else None,
    }
    st.download_button(
        "⬇️ Download this verdict as JSON (for the SolarCheck registry)",
        data=json.dumps(payload, indent=2, default=str),
        file_name=f"solarcheck_gsi_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
    )

    st.markdown("---")
    st.markdown(
        """
#### ⚠️ What this verdict is and is not

This is a **probabilistic engineering judgement built from the measurements you supplied**, not a
laboratory certification. The council's confidence is calibrated against what the model has
actually seen, and it is deliberately reduced when the input is unusual, when parameters were
estimated rather than measured, or when the agents disagree.

A verdict of *genuine* at 90% still means one in ten panels like this would surprise the system.
That is why the counter-case is printed above the decision — and why the *inconclusive* answer
exists at all.
"""
    )

st.markdown("---")
st.markdown(
    """
<p style='text-align:center;color:#95a5a6;font-size:12px;'>
    SolarCheck Africa · GSI Mode 3.0 · Université de Yaoundé I · Founder: Gamsi · 2026<br>
    <em>Changing how the world sees Africa — one panel at a time.</em>
</p>
""",
    unsafe_allow_html=True,
)
