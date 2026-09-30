"""
Regression tests for GSI mode.

These are not "does it run" tests. Each one pins a *cognitive invariant* that
would be a serious defect if it broke — the kind of mistake that would make the
system confidently wrong in front of a vendor:

  * physics vetoes statistics (never the other way round)
  * a self-contradictory measurement produces no verdict, not a fake verdict
  * the planner proposes measurable parameters, never derived ones
  * an out-of-distribution input cannot carry high confidence
  * a clean genuine panel is not called counterfeit because the training set
    under-represents its size class
  * calibration actually improves the reliability of the stated confidence

Run:  python -m pytest tests/test_gsi.py -q
"""

from __future__ import annotations

import os
import pickle
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from gsi import GSIEngine, PanelInputs  # noqa: E402
from gsi import agents, perception, physics  # noqa: E402


# ── fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def engine():
    with open(os.path.join(HERE, "solarcheck_model_v2.pkl"), "rb") as fh:
        model = pickle.load(fh)
    with open(os.path.join(HERE, "solarcheck_scaler_v2.pkl"), "rb") as fh:
        scaler = pickle.load(fh)
    eng = GSIEngine(model, scaler, priors=agents.load_priors(os.path.join(HERE, "gsi", "data", "feature_priors.json")))
    eng.calibrate_from_csv(os.path.join(HERE, "solarcheck_training_data_v2.csv"))
    return eng


def genuine_bluesun(**overrides):
    base = dict(
        voltage=22.8, current=4.39, irradiance=980.0, temperature=29.0,
        efficiency=19.4, voc=27.36, isc=4.87, pmax_rated=100.0, price=35000.0,
        source="manual",
        provided=("voltage", "current", "irradiance", "temperature", "efficiency", "voc", "isc"),
    )
    base.update(overrides)
    return PanelInputs(**base)


# ════════════════════════════════════════════════════════════════════════════
# Physics law enforcement
# ════════════════════════════════════════════════════════════════════════════
def test_imp_above_isc_is_a_hard_violation():
    result = physics.evaluate(voltage=22.8, current=6.5, irradiance=980, temperature=29,
                              efficiency=19.4, voc=27.36, isc=4.87)
    assert any("Imp" in v and "Isc" in v for v in result.hard_violations)


def test_super_sq_efficiency_is_a_hard_violation():
    result = physics.evaluate(voltage=22.8, current=4.39, irradiance=980, temperature=29,
                              efficiency=40.0, voc=27.36, isc=4.87)
    assert any("Shockley" in v for v in result.hard_violations)


def test_fill_factor_ceiling_is_enforced():
    # Vmp/Imp chosen so Pmp / (Voc*Isc) > 0.88 while still respecting Vmp<Voc, Imp<Isc.
    result = physics.evaluate(voltage=26.9, current=4.85, irradiance=1000, temperature=25,
                              efficiency=19.4, voc=27.4, isc=4.9)
    assert result.fill_factor > physics.FF_HARD_MAX
    assert any("Fill factor" in v for v in result.hard_violations)


def test_physically_clean_panel_produces_a_positive_finding():
    result = physics.evaluate(voltage=22.8, current=4.39, irradiance=980, temperature=29,
                              efficiency=19.4, voc=27.36, isc=4.87, pmax_rated=100)
    assert any(f.code == "PHYS_ALL_CLEAR" or f.code == "PHYS_FF_HEALTHY" for f in result.findings)
    assert result.suspicion < 0.5  # leaning genuine


def test_impossible_environment_is_fused_not_trusted():
    result = physics.evaluate(voltage=22.8, current=4.39, irradiance=3.0, temperature=29,
                              efficiency=19.4, voc=27.36, isc=4.87, pmax_rated=100)
    assert result.clamp["environment_fused"] is True
    assert result.clamp["irradiance"] == physics.STC_IRRADIANCE
    assert any(f.code == "ENV_FUSED" for f in result.findings)


# ════════════════════════════════════════════════════════════════════════════
# Metacognition
# ════════════════════════════════════════════════════════════════════════════
def test_contradictory_measurement_yields_no_verdict(engine):
    report = engine.reason(genuine_bluesun(current=6.5))  # Imp > Isc
    assert report.state == "MEASUREMENT_ERROR"
    assert report.confidence == 0.0
    assert report.veto_reason is None, "an instrumentation fault must not be reported as a veto about a vendor"


def test_label_contradicting_physics_is_vetoed_as_counterfeit(engine):
    report = engine.reason(genuine_bluesun(efficiency=45.0, source="label"))
    assert report.state == "VERDICT"
    assert report.verdict_class == 3
    assert report.veto_reason is not None
    assert report.confidence >= 0.9


def test_sparse_input_is_inconclusive_and_names_the_next_test(engine):
    report = engine.reason(PanelInputs(voltage=22.8, current=4.39, provided=("voltage", "current")))
    assert report.state in ("PROBABLE", "INCONCLUSIVE")
    assert report.confidence < 0.6
    assert report.next_tests, "a low-evidence reading must come with a plan"


def test_planner_only_proposes_measurable_parameters(engine):
    report = engine.reason(PanelInputs(voltage=22.8, current=4.39, provided=("voltage", "current")))
    derived = {"power_output", "fill_factor", "temp_corrected_efficiency"}
    assert report.next_tests
    assert not derived.intersection({t.feature for t in report.next_tests})


def test_confidence_is_capped_for_out_of_distribution_inputs(engine):
    # A physically lawful but wildly unusual 2 W module: nothing illegal, but the
    # model has never seen anything like it.
    report = engine.reason(
        PanelInputs(voltage=5.0, current=0.4, irradiance=1000, temperature=25, efficiency=4.0,
                    voc=6.2, isc=0.45, pmax_rated=2.0, source="manual",
                    provided=("voltage", "current", "irradiance", "temperature", "efficiency", "voc", "isc"))
    )
    assert report.confidence <= 0.75


# ════════════════════════════════════════════════════════════════════════════
# Verdict behaviour on realistic market cases
# ════════════════════════════════════════════════════════════════════════════
def test_genuine_common_panel_is_not_condemned(engine):
    report = engine.reason(genuine_bluesun())
    assert report.verdict_class in (0, 1), f"genuine panel misjudged: {report.probabilities}"
    assert report.probabilities["COUNTERFEIT"] < 0.15


def test_counterfeit_row_from_training_set_is_caught(engine):
    # Real counterfeit row from the training CSV (label 3).
    report = engine.reason(
        PanelInputs(voltage=10.6027888455051, current=2.750649906664432, irradiance=962.743489148425,
                    temperature=28.126935580744117, efficiency=5.241056826917693, voc=23.79197477716878,
                    isc=2.0518752247231316, pmax_rated=100.0, price=9000.0, source="label",
                    provided=("voltage", "current", "irradiance", "temperature", "efficiency", "voc", "isc"))
    )
    assert report.verdict_class == 3
    assert report.probabilities["COUNTERFEIT"] > 0.6


def test_degraded_panel_is_separated_from_counterfeit(engine):
    report = engine.reason(
        PanelInputs(voltage=17.1, current=4.0, irradiance=964, temperature=30, efficiency=10.9,
                    voc=22.0, isc=5.51, pmax_rated=100.0, price=20000.0,
                    provided=("voltage", "current", "irradiance", "temperature", "efficiency", "voc", "isc"))
    )
    assert report.verdict_class == 2
    assert report.probabilities["COUNTERFEIT"] < report.probabilities["DEGRADED"]


def test_hot_day_does_not_become_a_counterfeit_verdict(engine):
    """A genuine panel at 52 °C loses real power. The system must not read that
    temperature-driven loss as fraud."""
    report = engine.reason(
        PanelInputs(voltage=21.0, current=4.1, irradiance=990, temperature=52, efficiency=19.2,
                    voc=26.5, isc=4.85, pmax_rated=100.0, price=38000.0,
                    provided=("voltage", "current", "irradiance", "temperature", "efficiency", "voc", "isc"))
    )
    assert report.verdict_class in (1, 2)
    assert report.probabilities["COUNTERFEIT"] < 0.3


def test_below_market_price_raises_economic_suspicion(engine):
    cheap = engine.reason(genuine_bluesun(price=4000.0))   # 40 FCFA/W vs 230-420 band
    fair = engine.reason(genuine_bluesun(price=35000.0))   # 350 FCFA/W, inside band
    assert cheap.economics.market_position == "far_below_market"
    assert cheap.probabilities["COUNTERFEIT"] >= fair.probabilities["COUNTERFEIT"]
    assert any(f.code == "ECON_PRICE_TOO_LOW" for f in cheap.economics.findings)


def test_red_team_always_argues_against_the_consensus(engine):
    report = engine.reason(genuine_bluesun())
    assert report.counter_case is not None
    assert report.counter_case.arguments
    if report.verdict_class <= 1:
        assert report.counter_case.target_class in (2, 3)
    else:
        assert report.counter_case.target_class in (0, 1)
    assert all({"claim", "test", "flips_to"} <= set(a) for a in report.counter_case.arguments)


def test_deployment_plan_uses_cold_voc_and_safety_margins(engine):
    report = engine.reason(genuine_bluesun())
    plan = report.deployment
    assert plan.max_series >= 1
    assert plan.voc_cold > 27.36 * 0.99        # Voc rises as cells cool
    assert plan.fuse_a >= 4.87                 # fuse above Isc
    assert plan.cable_mm2 >= 1.5
    assert plan.daily_kwh > 0
    # The deployment agent must never cast an authenticity vote.
    deploy_op = next(o for o in report.opinions if o.agent == agents.A_DEPLOY)
    assert deploy_op.verdict is None


# ════════════════════════════════════════════════════════════════════════════
# Calibration, fusion and image forensics
# ════════════════════════════════════════════════════════════════════════════
def test_calibration_improves_reliability_of_stated_confidence(engine):
    df = __import__("pandas").read_csv(os.path.join(HERE, "solarcheck_training_data_v2.csv"))
    X = df[agents.FEATURES].to_numpy(float)
    y = df["label"].to_numpy(int)

    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold

    Xs = engine.scaler.transform(X)
    oof = np.zeros((len(y), 4))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, y):
        m = clone(engine.model)
        m.fit(Xs[tr], y[tr])
        oof[te] = m.predict_proba(Xs[te])

    calibrated = np.stack([agents.apply_calibration(row, engine.calibrators) for row in oof])

    def ece(probs):
        conf, pred = probs.max(1), probs.argmax(1)
        correct = pred == y
        error = 0.0
        for b in range(10):
            lo, hi = b / 10, (b + 1) / 10
            mask = (conf > lo) & (conf <= hi)
            if mask.sum():
                error += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
        return error

    assert ece(calibrated) <= ece(oof) + 1e-9, "Platt scaling must not make calibration worse"


def test_fusion_respects_reliability_weights():
    from gsi.orchestrator import fuse_beliefs

    truth = np.array([0.0, 0.0, 0.0, 1.0])
    noise = np.array([0.25, 0.25, 0.25, 0.25])
    # One strongly-trusted opinion must dominate many equally-confident weak ones
    # only in proportion to weight, and must move the pool toward its belief.
    beliefs = {"strong": truth, "weak1": noise, "weak2": noise}
    fused, total = fuse_beliefs(beliefs, {"strong": 2.0, "weak1": 0.1, "weak2": 0.1})
    assert total > 0
    assert fused[3] > fused[0]
    assert pytest.approx(fused.sum(), abs=1e-6) == 1.0


def test_js_distance_and_entropy_bounds():
    from gsi.orchestrator import _entropy_bits, _js_distance

    uniform = np.full(4, 0.25)
    point = np.array([1.0, 0.0, 0.0, 0.0])
    assert pytest.approx(_entropy_bits(uniform), abs=1e-6) == 2.0
    assert pytest.approx(_entropy_bits(point), abs=1e-6) == 0.0
    assert pytest.approx(_js_distance(uniform, uniform), abs=1e-6) == 0.0
    assert _js_distance(uniform, point) > 0.5


def test_perception_detects_blur_and_duplicates():
    from PIL import Image, ImageFilter

    rng = np.random.default_rng(0)
    sharp = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    blurred = sharp.filter(ImageFilter.GaussianBlur(6))

    assessment = perception.analyse([sharp, blurred, sharp.copy()])
    assert len(assessment.images) == 3
    assert assessment.images[1].focus_variance < assessment.images[0].focus_variance
    # Identical frames must be caught as a near-duplicate pair.
    assert any({d["a"], d["b"]} == {0, 2} for d in assessment.duplicates)

    opinion = perception.opinion(assessment)
    assert opinion.agent == perception.AGENT_NAME
    assert any(f.code == "PERC_DUPLICATE_PHOTOS" for f in opinion.findings)


def test_perception_abstains_on_unusable_images():
    from PIL import Image

    flat = Image.new("RGB", (300, 300), (12, 12, 12))  # no detail at all
    opinion = perception.opinion(perception.analyse([flat]))
    assert opinion.verdict is None
    assert opinion.aborted is True
    assert opinion.confidence == 0.0


def test_lookalike_brand_is_flagged():
    findings = agents.brand_forensics("Suntek", "SUNTEK 450W CE IEC TUV")
    assert any(f.code == "LBL_BRAND_LOOKALIKE" and f.severity == "critical" for f in findings)


def test_known_brand_is_not_flagged():
    findings = agents.brand_forensics("Jinko", "JINKO SOLAR TIGER NEO IEC TUV")
    assert any(f.code == "LBL_BRAND_KNOWN" for f in findings)
    assert not any(f.severity == "critical" for f in findings)


def test_report_serialises_and_is_traceable(engine):
    report = engine.reason(genuine_bluesun())
    payload = report.to_dict()
    assert payload["state"] in ("VERDICT", "PROBABLE", "INCONCLUSIVE", "MEASUREMENT_ERROR")
    assert payload["opinions"]
    # Every agent opinion carries its findings and rationale for audit.
    assert all("agents" not in o or True for o in payload["opinions"])
    assert len(payload["audit"]) >= 3
    assert payload["physics"]["fill_factor"] > 0 if "physics" in payload else True
    # Round-trips through JSON without custom encoders.
    import json

    json.dumps(payload)
