"""
The council of specialists.

Each agent owns one evidence channel and is written to be *narrow on purpose*:
a specialist that tries to answer everything is just another black box. The
council is:

  MLForecast   — the trained RandomForest, now probability-calibrated
                 (out-of-fold Platt scaling) and OOD-aware (Mahalanobis
                 distance to the training manifold).
  Physics      — laws and datasheet envelopes; allowed to veto.
  Perception   — image forensics (see perception.py).
  LabelForge   — brand/label text forensics: lookalike names, spec
                 inconsistencies, OCR-vs-electrical cross-checks.
  Economics    — price-per-watt against a regional honest band, money at
                 risk, payback against the grid and against a generator.
  Deployment   — what is *safe to build* with this panel: string sizing,
                 fusing, cable, yield. Not an authenticity voter, and it
                 says so, which keeps the fusion honest.
  RedTeam      — adversarial prosecutor: builds the strongest case against
                 whatever the council currently believes.
"""

from __future__ import annotations

import difflib
import json
import os
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .evidence import AgentOpinion, CLASS_ANCHORS, normalise_score
from .physics import (
    COMMERCIAL_MAX_EFFICIENCY,
    FF_HEALTHY,
    PANEL_LIBRARY,
    STC_IRRADIANCE,
    Finding,
    PhysicsResult,
)
from .regions import Region

# Agent identities (used as fusion keys — do not rename casually).
A_ML = "MLForecast"
A_PHYSICS = "Physics"
A_PERCEPTION = "Perception"
A_LABEL = "LabelForge"
A_ECON = "Economics"
A_DEPLOY = "Deployment"
A_RED = "RedTeam"

FEATURES = [
    "voltage",
    "current",
    "irradiance",
    "temperature",
    "efficiency",
    "power_output",
    "fill_factor",
    "temp_corrected_efficiency",
]

KNOWN_BRANDS = sorted({p["brand"].lower() for p in PANEL_LIBRARY} | {
    "trina", "longi", "ja solar", "ja-solar", "risen", "q cells", "qcells", "rec",
    "first solar", "sunpower", "vesta", "felicity", "sunshine", "solar world",
    "phono solar", "thornova", "shinsung", "hanwha", "hyundai", "talesun",
})

CERT_MARKERS = ("ce", "iec", "tuv", "tüv", "iso", "ul", "mcs", "rohs")

# How far beyond a class's largest training Mahalanobis distance a reading must
# sit before it counts as out-of-distribution.
OOD_TOLERANCE = 1.15


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


# ════════════════════════════════════════════════════════════════════════════
# Priors + calibration
# ════════════════════════════════════════════════════════════════════════════
def load_priors(path: Optional[str] = None) -> Optional[dict]:
    """Load the feature-prior artifact; return None if it is unavailable."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "feature_priors.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def build_calibrators(model, scaler, X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                      seed: int = 0) -> Optional[List[Tuple[float, float]]]:
    """Out-of-fold Platt scaling coefficients, one per class.

    The shipped RandomForest reports ~99% confidence on everything it sees,
    including inputs unlike its training data. Fitting a one-dimensional
    logistic map on out-of-fold predictions re-anchors those scores to
    observed frequencies, which is what makes the council's subsequent
    reliability weighting meaningful. Returns None if scikit-learn cannot
    complete the fit (the engine then runs uncalibrated and says so).
    """
    try:
        from sklearn.base import clone
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
    except ImportError:
        return None

    try:
        Xs = scaler.transform(X)
        oof = np.zeros((len(y), 4))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in splitter.split(Xs, y):
            fold = clone(model)
            fold.fit(Xs[train_idx], y[train_idx])
            oof[test_idx] = fold.predict_proba(Xs[test_idx])

        coefs: List[Tuple[float, float]] = []
        for cls in range(4):
            target = (y == cls).astype(int)
            if target.min() == target.max():  # class absent — leave identity
                coefs.append((1.0, 0.0))
                continue
            lr = LogisticRegression(C=0.5, max_iter=1000)
            lr.fit(_logit(oof[:, cls]).reshape(-1, 1), target)
            coefs.append((float(lr.coef_[0, 0]), float(lr.intercept_[0])))
        return coefs
    except Exception:
        return None


def apply_calibration(probs: np.ndarray, coefs: Optional[Sequence[Tuple[float, float]]]) -> np.ndarray:
    if not coefs:
        return probs
    calibrated = np.stack(
        [
            1.0 / (1.0 + np.exp(-(a * _logit(probs[c]) + b)))
            for c, (a, b) in enumerate(coefs)
        ]
    )
    total = calibrated.sum()
    return calibrated / total if total > 0 else probs


def mahalanobis(priors: Optional[dict], feature_vector: Sequence[float],
                class_id: Optional[int] = None) -> Optional[dict]:
    """Two distinct novelty questions, answered separately.

    * global — "is this a typical panel at all?" Novelty detection. A very
      common, legitimate module (e.g. a 100 W unit in a training set dominated
      by 150 W+ panels) scores high here without anything being wrong.
    * class  — "is this reading typical of the class I just assigned?" That is
      the question which bears on trusting the label, so the out-of-distribution
      flag and the reliability penalty use *this* one.
    """
    if not priors:
        return None
    g = priors["global"]
    mean = np.asarray(g["mean"], dtype=float)
    std = np.asarray(g["std"], dtype=float)
    cov_inv = np.asarray(g["cov_inv"], dtype=float)
    x = np.asarray(feature_vector, dtype=float)
    if x.shape[0] != mean.shape[0] or not np.isfinite(x).all():
        return None
    z = (x - mean) / np.where(std > 0, std, 1.0)
    d_global = float(math.sqrt(max(0.0, z @ cov_inv @ z)))
    q = g["maha_quantiles"]

    def _percentile(distance, quantiles):
        pct = 50.0
        for key, value in (("p50", 50.0), ("p75", 75.0), ("p90", 90.0), ("p99", 99.0)):
            if key in quantiles and distance > quantiles[key]:
                pct = value
        return pct

    result = {
        "distance": round(d_global, 3),
        "percentile": _percentile(d_global, q),
        "p99": q["p99"],
        "max_training": q["max"],
        "global_percentile": _percentile(d_global, q),
        "global_beyond_training": d_global > q["max"],
        "out_of_distribution": d_global > q["p99"],
        "class_id": class_id,
    }

    if class_id is not None:
        cls = (priors.get("classes") or {}).get(str(class_id))
        if cls and cls.get("cov_inv") and cls.get("maha_quantiles"):
            cov_inv_c = np.asarray(cls["cov_inv"], dtype=float)
            d_class = float(math.sqrt(max(0.0, z @ cov_inv_c @ z)))
            qc = cls["maha_quantiles"]
            result.update(
                {
                    "class_distance": round(d_class, 3),
                    "class_percentile": _percentile(d_class, qc),
                    "class_p99": qc["p99"],
                    "class_max_training": qc["max"],
                    # The decision that matters: clearly beyond the class footprint.
                    # A 15% tolerance stops the single heaviest training sample from
                    # defining the boundary of "known" (class covariances are
                    # heavy-tailed because counterfeit readings are wild).
                    "class_beyond_training": d_class > qc["max"] * OOD_TOLERANCE,
                    "out_of_distribution": d_class > qc["max"] * OOD_TOLERANCE,
                }
            )
    return result


# ════════════════════════════════════════════════════════════════════════════
# ML agent
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class MLSnapshot:
    raw_probs: np.ndarray
    probs: np.ndarray  # calibrated
    prediction: int
    confidence: float
    score: float
    entropy: float
    ood: Optional[dict]
    top_deviations: List[dict] = field(default_factory=list)


def ml_snapshot(
    model,
    scaler,
    priors: Optional[dict],
    calibrators: Optional[Sequence[Tuple[float, float]]],
    feature_vector: Sequence[float],
) -> MLSnapshot:
    """Forward pass + calibration + OOD + per-feature attribution."""
    x = np.asarray([feature_vector], dtype=float)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    scaled = scaler.transform(x)
    try:
        raw = np.asarray(model.predict_proba(scaled)[0], dtype=float)
    except Exception:
        raw = np.full(4, 0.25)
    probs = apply_calibration(raw, calibrators)
    prediction = int(np.argmax(probs))
    confidence = float(probs[prediction])

    anchors = np.array([CLASS_ANCHORS[c] for c in range(4)])
    score = float(np.dot(probs, anchors))
    entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1))))

    ood = mahalanobis(priors, feature_vector, class_id=prediction)

    deviations: List[dict] = []
    if priors:
        cls_stats = priors["classes"].get(str(prediction))
        if cls_stats:
            mean = np.asarray(cls_stats["mean"], dtype=float)
            std = np.asarray(cls_stats["std"], dtype=float)
            z = (np.asarray(feature_vector, dtype=float) - mean) / np.where(std > 1e-9, std, 1.0)
            order = np.argsort(-np.abs(z))
            for idx in order[:3]:
                if abs(z[idx]) >= 1.5:
                    deviations.append(
                        {
                            "feature": FEATURES[idx],
                            "value": round(float(feature_vector[idx]), 4),
                            "expected_mean": round(float(mean[idx]), 4),
                            "z": round(float(z[idx]), 2),
                        }
                    )

    return MLSnapshot(raw, probs, prediction, confidence, score, entropy, ood, deviations)


def ml_opinion(snapshot: MLSnapshot, feature_vector: Sequence[float]) -> AgentOpinion:
    findings: List[Finding] = []
    pct = snapshot.confidence * 100

    findings.append(
        Finding(
            "ML_VERDICT",
            "info",
            f"Calibrated forest output: {pct:.1f}% on class {snapshot.prediction} "
            f"(raw model said {snapshot.raw_probs[snapshot.prediction]*100:.1f}%, "
            f"re-anchored to observed frequencies by out-of-fold Platt scaling).",
            weight=0.0,
            evidence={
                "probs": [round(float(p), 4) for p in snapshot.probs],
                "raw_probs": [round(float(p), 4) for p in snapshot.raw_probs],
            },
        )
    )

    if snapshot.ood and snapshot.ood.get("out_of_distribution"):
        findings.append(
            Finding(
                "ML_OUT_OF_DISTRIBUTION",
                "warn",
                f"Even judged against the class I just assigned, this reading sits beyond the "
                f"training footprint ({snapshot.ood.get('class_distance', snapshot.ood['distance']):.2f} "
                f"vs class max {snapshot.ood.get('class_max_training', snapshot.ood['max_training']):.2f}). "
                f"The forest has no real experience here, so I am voluntarily down-weighting my own "
                f"vote — trust the physical measurements over me.",
                weight=0.0,
                evidence=snapshot.ood,
            )
        )
    if snapshot.ood and snapshot.ood.get("global_beyond_training"):
        findings.append(
            Finding(
                "ML_NOVEL_PANEL_TYPE",
                "info",
                f"This panel sits at the {snapshot.ood['global_percentile']:.0f}th percentile of all "
                f"training data (Mahalanobis {snapshot.ood['distance']:.2f}). It is most likely a panel "
                f"*size* the training set under-represents rather than a bad panel — which is exactly "
                f"why the physics channel, not the forest, carries this verdict.",
                weight=0.0,
                evidence={
                    "global_distance": snapshot.ood["distance"],
                    "global_percentile": snapshot.ood["global_percentile"],
                },
            )
        )

    for dev in snapshot.top_deviations:
        findings.append(
            Finding(
                "ML_ATTRIBUTION",
                "info",
                f"{dev['feature']} = {dev['value']} is {abs(dev['z']):.1f}σ from the "
                f"{dev['expected_mean']} typical of {['premium','basic','degraded','fake'][snapshot.prediction]} "
                f"panels in training.",
                weight=0.0,
                evidence=dev,
            )
        )

    # Reliability falls with entropy and with OOD distance.
    reliability = 0.9
    if snapshot.ood and snapshot.ood.get("out_of_distribution"):
        reliability *= 0.65
    reliability *= float(1.0 - min(0.5, snapshot.entropy / math.log(4) * 0.5))

    rationale = (
        f"Trained forest on 8 electrical features; entropy {snapshot.entropy:.3f} nats "
        f"(max {math.log(4):.3f})."
        + (" Out-of-distribution input detected — vote down-weighted."
           if snapshot.ood and snapshot.ood.get("out_of_distribution") else "")
    )
    return AgentOpinion(
        agent=A_ML,
        verdict=snapshot.prediction,
        score=normalise_score(snapshot.score),
        confidence=snapshot.confidence,
        reliability=reliability,
        findings=findings,
        rationale=rationale,
    )


# ════════════════════════════════════════════════════════════════════════════
# Physics agent
# ════════════════════════════════════════════════════════════════════════════
def physics_opinion(
    physics: PhysicsResult,
    evidence_breadth: int,
    pmax_rated: Optional[float] = None,
    efficiency: float = 0.0,
) -> AgentOpinion:
    findings: List[Finding] = []
    # Evidence breadth 0..4 (voltage/current, voc/isc, efficiency, environmental).
    confidence = float(min(0.98, 0.45 + 0.14 * evidence_breadth))
    if physics.hard_violations:
        confidence = min(0.99, confidence + 0.15)

    score = normalise_score(physics.suspicion * 2.0 - 1.0)

    # The four SolarCheck classes encode panel *size class* as much as quality:
    # "premium" modules in the training set are 450-550 W, "basic" ones ~100-150 W.
    # Physics can certify that silicon is genuine and behaving lawfully, but a
    # multimeter cannot tell a well-made 100 W panel from a badly-made 550 W one.
    # So the healthy direction is capped at the genuine midpoint unless there is
    # positive nameplate evidence of a premium-class module.
    premium_evidence = False
    if pmax_rated and pmax_rated > 0 and physics.performance_ratio:
        premium_evidence = physics.performance_ratio >= 0.9 and efficiency >= 19.5
    if score < 0:
        score = max(score, -0.85 if premium_evidence else -0.35)

    verdict = None
    if physics.hard_violations:
        verdict = 3
    elif score > 0.75:
        verdict = 3
    elif score > 0.1:
        verdict = 2
    elif premium_evidence:
        verdict = 0
    else:
        verdict = 1

    if not physics.hard_violations and score < 0:
        findings.append(
            Finding(
                "PHYS_CLASS_SCOPE",
                "info",
                (
                    f"Measured power is {physics.performance_ratio*100:.0f}% of the {pmax_rated:g} W "
                    f"nameplate at {efficiency:.1f}% efficiency — premium-class evidence, so I will "
                    f"vote for the premium class."
                    if premium_evidence
                    else "I can certify that this is genuine, lawfully-behaving silicon, but I cannot "
                         "tell from electrical measurements alone which size class it belongs to — the "
                         "forest, which has seen panel sizes, decides premium vs basic."
                ),
                weight=0.0,
                evidence={"premium_evidence": premium_evidence},
            )
        )

    rationale = (
        f"Ran {len(physics.findings)} physical checks; "
        f"{len(physics.hard_violations)} hard violation(s); "
        f"FF={physics.fill_factor:.3f}, P={physics.power_output:.1f} W, "
        f"suspicion={physics.suspicion:.2f}."
    )
    return AgentOpinion(
        agent=A_PHYSICS,
        verdict=verdict,
        score=score,
        confidence=confidence,
        reliability=0.98,  # physical law outranks any statistical pattern
        findings=physics.findings,
        rationale=rationale,
    )


# ════════════════════════════════════════════════════════════════════════════
# Label / brand text forensics
# ════════════════════════════════════════════════════════════════════════════
def brand_forensics(brand: Optional[str], model_text: Optional[str] = None) -> List[Finding]:
    """Catch lookalike brands and missing certification language."""
    findings: List[Finding] = []
    if not brand:
        findings.append(
            Finding(
                "LBL_NO_BRAND",
                "warn",
                "No brand name was legible on the label. Genuine modules are branded; "
                "unbranded stickers are a common counterfeit pattern in the region.",
                weight=0.25,
            )
        )
        return findings

    clean = brand.strip().lower()
    exact = clean in KNOWN_BRANDS
    if exact:
        findings.append(
            Finding("LBL_BRAND_KNOWN", "good", f"Brand '{brand}' matches a known manufacturer.", weight=-0.15)
        )
        return findings

    # Lookalike detection needs two signals, because brand names are short:
    # a high sequence similarity, or a long shared prefix with any difference at
    # all ("suntek" vs "suntech", "jinkoo" vs "jinko"). Sequence ratio alone
    # misses one-letter edits in short names.
    close = difflib.get_close_matches(clean, KNOWN_BRANDS, n=1, cutoff=0.72)
    if not close:
        candidates = [
            brand_name
            for brand_name in KNOWN_BRANDS
            if min(len(brand_name), len(clean)) >= 5
            and os.path.commonprefix([brand_name, clean]) == clean[: max(4, len(clean) - 2)]
        ]
        close = candidates[:1]
    if close:
        findings.append(
            Finding(
                "LBL_BRAND_LOOKALIKE",
                "critical",
                f"Brand '{brand}' is suspiciously close to the real manufacturer "
                f"'{close[0]}' but is not the same string. Lookalike spellings "
                f"(one letter changed, doubled vowels, swapped syllables) are the single "
                f"most common counterfeit branding trick. Verify the serial number with "
                f"the real manufacturer before paying.",
                weight=0.75,
                evidence={"brand": brand, "resembles": close[0]},
            )
        )
    else:
        findings.append(
            Finding(
                "LBL_BRAND_UNKNOWN",
                "warn",
                f"Brand '{brand}' is not in the SolarCheck verification list. That is not "
                f"proof of a fake — new and regional manufacturers exist — but an unknown "
                f"brand plus a low fill factor is how most fakes in this market present.",
                weight=0.2,
                evidence={"brand": brand},
            )
        )

    if model_text:
        text = model_text.lower()
        if not any(marker in text for marker in CERT_MARKERS):
            findings.append(
                Finding(
                    "LBL_NO_CERTS",
                    "warn",
                    "The captured label text contains no IEC/TÜV/CE/ISO marking. Genuine "
                    "modules sold internationally carry at least one.",
                    weight=0.2,
                )
            )
    return findings


def label_opinion(
    brand: Optional[str],
    model_text: Optional[str],
    label_check: Optional[dict],
    extracted: Optional[dict],
    physics: PhysicsResult,
) -> AgentOpinion:
    findings = brand_forensics(brand, model_text)
    suspicion = 0.0
    evidence_mass = 1.0
    hard = False

    for f in findings:
        if f.severity == "critical":
            hard = True
        suspicion += f.weight

    if label_check:
        pq = float(label_check.get("print_quality", 5) or 5)
        fc = float(label_check.get("font_consistency", 5) or 5)
        cert = label_check.get("certification_marks_visible")
        flags = label_check.get("red_flags") or []
        evidence_mass += 1.0
        suspicion += ((10 - pq) / 10.0) * 0.30 + ((10 - fc) / 10.0) * 0.30
        if cert is False:
            suspicion += 0.08
        if flags:
            suspicion += min(0.30, 0.10 * len(flags))
        findings.append(
            Finding(
                "LBL_STICKER",
                "warn" if suspicion > 0.3 else "info",
                f"Sticker forensics: print {pq:.0f}/10, fonts {fc:.0f}/10, "
                f"certification marks {'present' if cert else 'absent'}, {len(flags)} red flag(s).",
                weight=0.0,
                evidence={"print_quality": pq, "font_consistency": fc, "cert_visible": cert, "red_flags": flags},
            )
        )

    # Cross-check: does the label's own nameplate agree with the datasheet?
    if extracted:
        claimed = extracted.get("pmax")
        if claimed:
            measured = physics.power_output
            if measured > 0 and claimed > 0:
                ratio = measured / float(claimed)
                evidence_mass += 1.0
                if ratio < 0.55:
                    hard = True
                    suspicion += 0.55
                    findings.append(
                        Finding(
                            "LBL_NAME_MEASURE_MISMATCH",
                            "critical",
                            f"The label claims {claimed:g} W but the measured maximum power "
                            f"is {measured:.1f} W — only {ratio*100:.0f}% of nameplate "
                            f"(flash-test tolerance is about ±3%). A panel this far from its "
                            f"own printed rating is either mislabelled or not the panel it "
                            f"claims to be.",
                            weight=0.55,
                            evidence={"claimed_w": claimed, "measured_w": round(measured, 2), "ratio": round(ratio, 3)},
                        )
                    )
                elif ratio < 1.25:
                    findings.append(
                        Finding(
                            "LBL_NAME_MEASURE_OK",
                            "good",
                            f"Measured {measured:.1f} W is within {abs(ratio-1)*100:.0f}% of the "
                            f"claimed {claimed:g} W nameplate — the label is truthful about power.",
                            weight=-0.25,
                        )
                    )

    if physics.dataclass_match:
        m = physics.dataclass_match
        findings.append(
            Finding(
                "LBL_DATASHEET_MATCH",
                "info" if m["confidence"] < 0.7 else "good",
                f"Electrical fingerprint is {m['confidence']*100:.0f}% consistent with a "
                f"{m['brand']} {m['model']} ({m['pmax']:g} W, {m['efficiency']}% efficiency).",
                weight=-0.15 * m["confidence"],
                evidence={"panel": f"{m['brand']} {m['model']}", "confidence": m["confidence"]},
            )
        )

    suspicion = float(max(0.0, min(1.0, suspicion)))
    confidence = float(min(0.95, 0.35 + 0.15 * evidence_mass))
    score = normalise_score(suspicion * 2.0 - 1.0)
    verdict = 3 if (hard or suspicion > 0.6) else (2 if suspicion > 0.35 else (1 if extracted else None))

    return AgentOpinion(
        agent=A_LABEL,
        verdict=verdict,
        score=score,
        confidence=confidence,
        reliability=0.7,
        findings=findings,
        rationale=(
            f"Brand/label forensics: {len(findings)} checks, suspicion {suspicion:.2f}."
            + (" A critical label contradiction was found." if hard else "")
        ),
    )


# ════════════════════════════════════════════════════════════════════════════
# Economics agent
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class EconomicsResult:
    price_per_watt: Optional[float] = None
    band: Optional[tuple] = None
    market_position: str = "unknown"
    fair_value: Optional[float] = None
    money_at_risk: float = 0.0
    daily_kwh: float = 0.0
    annual_savings: float = 0.0
    payback_years: Optional[float] = None
    generator_cost_per_kwh: Optional[float] = None
    solar_cost_per_kwh: Optional[float] = None
    verdict_line: str = ""
    findings: List[Finding] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "price_per_watt": None if self.price_per_watt is None else round(self.price_per_watt, 2),
            "band": self.band,
            "market_position": self.market_position,
            "fair_value": None if self.fair_value is None else round(self.fair_value, 0),
            "money_at_risk": round(self.money_at_risk, 0),
            "daily_kwh": round(self.daily_kwh, 3),
            "annual_savings": round(self.annual_savings, 0),
            "payback_years": None if self.payback_years is None else round(self.payback_years, 1),
            "generator_cost_per_kwh": self.generator_cost_per_kwh,
            "solar_cost_per_kwh": self.solar_cost_per_kwh,
            "verdict_line": self.verdict_line,
        }


GENERATOR_FUEL_PRICE_XAF_PER_L = 840.0  # Cameroon pump price, super/petrol 2025
GENERATOR_L_PER_KWH = 0.45  # small petrol genset at part load
DIESEL_L_PER_KWH = 0.35


def economics_opinion(
    price: Optional[float],
    pmax_rated: Optional[float],
    physics: PhysicsResult,
    region: Region,
    confidence_in_verdict: float = 0.5,
) -> tuple:
    """Return (AgentOpinion, EconomicsResult)."""
    findings: List[Finding] = []
    res = EconomicsResult(band=region.price_band)

    capacity_w = float(pmax_rated) if pmax_rated else (physics.power_output if physics.power_output > 0 else None)
    stc_power = physics.temp_corrected_power if physics.temp_corrected_power > 0 else physics.power_output

    # ── Energy economics (always computable) ────────────────────────────────
    if capacity_w and capacity_w > 0:
        kwp = capacity_w / 1000.0
        res.daily_kwh = kwp * region.peak_sun_hours
        res.annual_savings = res.daily_kwh * 365.0 * region.tariff_per_kwh
        if price and price > 0 and res.annual_savings > 0:
            res.payback_years = float(price) / res.annual_savings
        # Levelised-ish cost per kWh over a conservative 15-year life at this yield.
        if price and price > 0:
            lifetime_kwh = res.daily_kwh * 365.0 * 15.0 * 0.8  # 20% degradation/outage haircut
            if lifetime_kwh > 0:
                res.solar_cost_per_kwh = float(price) / lifetime_kwh
        res.generator_cost_per_kwh = round(GENERATOR_FUEL_PRICE_XAF_PER_L * GENERATOR_L_PER_KWH, 1)

    # ── Price-band sanity ───────────────────────────────────────────────────
    if price and capacity_w and capacity_w > 0:
        ppw = float(price) / capacity_w
        res.price_per_watt = ppw
        if region.price_band:
            low, high = region.price_band
            if ppw < low * 0.65:
                res.market_position = "far_below_market"
                findings.append(
                    Finding(
                        "ECON_PRICE_TOO_LOW",
                        "critical",
                        f"{price:,.0f} {region.currency_symbol} for {capacity_w:.0f} W is "
                        f"{ppw:.0f} {region.currency_symbol}/W — far below the honest landed range "
                        f"of {low}-{high} {region.currency_symbol}/W in {region.name}. Genuine "
                        f"modules do not sell below landed cost. Either it is a fake, it is used, "
                        f"or the wattage is overstated.",
                        weight=0.6,
                        evidence={"price_per_watt": round(ppw, 1), "band": region.price_band},
                    )
                )
            elif ppw < low:
                res.market_position = "cheap"
                findings.append(
                    Finding(
                        "ECON_PRICE_CHEAP",
                        "info",
                        f"{ppw:.0f} {region.currency_symbol}/W is below the usual "
                        f"{low}-{high} {region.currency_symbol}/W band — a good deal if the "
                        f"electrical test agrees, which is exactly why the test matters.",
                        weight=0.05,
                    )
                )
            elif ppw <= high:
                res.market_position = "fair"
                findings.append(
                    Finding(
                        "ECON_PRICE_FAIR",
                        "good",
                        f"{ppw:.0f} {region.currency_symbol}/W sits inside the honest "
                        f"{low}-{high} {region.currency_symbol}/W band for {region.name}.",
                        weight=0.0,
                    )
                )
            else:
                res.market_position = "above_market"
                findings.append(
                    Finding(
                        "ECON_PRICE_HIGH",
                        "warn",
                        f"{ppw:.0f} {region.currency_symbol}/W is above the {high} "
                        f"{region.currency_symbol}/W ceiling for this market. Not a fake — just "
                        f"expensive. Negotiate.",
                        weight=0.0,
                    )
                )
        else:
            findings.append(
                Finding(
                    "ECON_NO_BAND",
                    "info",
                    f"No verified price band for {region.name}. Recording {ppw:.0f} "
                    f"{region.currency_symbol}/W so the community database can build one.",
                    weight=0.0,
                )
            )

    # ── Money at risk given the authenticity belief ─────────────────────────
    if price and price > 0:
        shortfall = 1.0
        if capacity_w and stc_power and capacity_w > 0:
            shortfall = float(max(0.0, min(1.0, 1.0 - (stc_power / capacity_w))))
        # Expected loss = P(fake) x full price + P(degraded) x value of the shortfall
        res.money_at_risk = float(price) * (confidence_in_verdict * 0.75 + shortfall * (1 - confidence_in_verdict) * 0.5)
        if stc_power and capacity_w:
            res.fair_value = float(price) * (stc_power / capacity_w)
        if res.money_at_risk > 0:
            findings.append(
                Finding(
                    "ECON_MONEY_AT_RISK",
                    "warn" if res.money_at_risk > price * 0.2 else "info",
                    f"Exposure on this transaction is about {res.money_at_risk:,.0f} "
                    f"{region.currency_symbol} at the current belief state. If the panel is "
                    f"genuine, that risk becomes zero — it is the price of not testing.",
                    weight=0.0,
                    evidence={"money_at_risk": round(res.money_at_risk, 2)},
                )
            )

    if res.payback_years is not None:
        if res.payback_years <= 3:
            tone = "excellent — under three years to break even"
        elif res.payback_years <= 6:
            tone = "reasonable"
        elif res.payback_years <= 10:
            tone = "slow; you are paying a premium or the grid tariff is low"
        else:
            tone = "poor — at this price the panel may never pay for itself"
        res.verdict_line = (
            f"At {res.daily_kwh:.2f} kWh/day and {region.tariff_per_kwh:g} "
            f"{region.currency_symbol}/kWh, payback is {res.payback_years:.1f} years ({tone})."
        )
        if res.generator_cost_per_kwh and res.solar_cost_per_kwh:
            res.verdict_line += (
                f" Levelised solar cost {res.solar_cost_per_kwh:,.0f} {region.currency_symbol}/kWh "
                f"vs {res.generator_cost_per_kwh:,.0f} {region.currency_symbol}/kWh on a petrol genset."
            )
        findings.append(Finding("ECON_PAYBACK", "info", res.verdict_line, weight=0.0))

    res.findings = findings

    # Economics is a poor authenticity voter: it abstains unless price is given.
    if not price:
        return (
            AgentOpinion.abstain(
                A_ECON,
                "No purchase price supplied, so I cannot judge whether the offer is "
                "economically plausible. Energy and payback numbers below still apply.",
                reliability=0.4,
            ),
            res,
        )

    score = normalise_score(
        (0.6 if res.market_position == "far_below_market" else 0.0) - (0.15 if res.market_position == "fair" else 0.0)
    )
    verdict = None
    if res.market_position == "far_below_market":
        verdict = 3
    elif res.market_position == "cheap":
        verdict = 2

    return (
        AgentOpinion(
            agent=A_ECON,
            verdict=verdict,
            score=score,
            confidence=0.55 if region.price_band else 0.35,
            reliability=0.6,
            findings=findings,
            rationale=(
                f"Price {price:,.0f} {region.currency_symbol}; position "
                f"'{res.market_position}'; payback "
                f"{'n/a' if res.payback_years is None else f'{res.payback_years:.1f} y'}."
            ),
        ),
        res,
    )


# ════════════════════════════════════════════════════════════════════════════
# Deployment / safety agent
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class DeploymentPlan:
    vmp: float = 0.0
    voc_cold: float = 0.0
    imp: float = 0.0
    isc_hot: float = 0.0
    max_series: int = 1
    series_note: str = ""
    max_parallel: int = 1
    cable_mm2: float = 2.5
    fuse_a: float = 0.0
    controller_a: float = 0.0
    array_kwp: float = 0.0
    daily_kwh: float = 0.0
    annual_kwh: float = 0.0
    battery_ah_12v: float = 0.0
    findings: List[Finding] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "vmp": round(self.vmp, 2),
            "voc_cold": round(self.voc_cold, 2),
            "imp": round(self.imp, 3),
            "max_series": self.max_series,
            "max_parallel": self.max_parallel,
            "cable_mm2": self.cable_mm2,
            "fuse_a": round(self.fuse_a, 1),
            "controller_a": round(self.controller_a, 1),
            "array_kwp": round(self.array_kwp, 3),
            "daily_kwh": round(self.daily_kwh, 3),
            "annual_kwh": round(self.annual_kwh, 1),
            "battery_ah_12v": round(self.battery_ah_12v, 0),
            "series_note": self.series_note,
        }


def deployment_opinion(
    physics: PhysicsResult, region: Region, array_size: int = 1, controller_vmax: float = 150.0
) -> tuple:
    """Sizing and safety maths. Explicitly *not* an authenticity voter."""
    plan = DeploymentPlan()
    voc = physics.clamp.get("voc", 0.0) or 0.0
    vmp = physics.clamp.get("voltage", 0.0) or 0.0
    imp = physics.clamp.get("current", 0.0) or 0.0
    isc = physics.clamp.get("isc", 0.0) or 0.0

    if voc > 0:
        # Cold-morning Voc: -0.30 %/degC, using 15 degC as the cold reference for
        # the humid tropics (rarely colder, but highland mornings reach it).
        plan.voc_cold = voc * (1 + 0.0030 * (25.0 - 15.0))
        plan.max_series = max(1, int(controller_vmax * 0.9 // plan.voc_cold)) if plan.voc_cold else 1
        plan.series_note = (
            f"Voc at 15 °C rises to {plan.voc_cold:.1f} V per panel. With a "
            f"{controller_vmax:.0f} V controller, never put more than {plan.max_series} in series "
            f"(90% headroom, so a frost-cold dawn cannot exceed the input rating)."
        )
    if isc > 0:
        plan.isc_hot = isc * 1.25  # NEC-style continuous-current factor
        plan.max_parallel = max(1, int(60 // max(plan.isc_hot, 1e-6)))
        plan.fuse_a = round(isc * 1.56, 1)
    plan.vmp, plan.imp = vmp, imp

    if imp > 0:
        # 2% voltage-drop cable sizing for a 10 m one-way run, 12 V nominal.
        run_m, drop_v = 20.0, 0.24
        area = (2 * run_m * imp * 0.0175) / drop_v  # rho_cu = 0.0175 ohm mm2/m
        for standard in (1.5, 2.5, 4.0, 6.0, 10.0, 16.0, 25.0):
            if standard >= area:
                plan.cable_mm2 = standard
                break
        else:
            plan.cable_mm2 = 35.0
    if physics.power_output > 0:
        plan.controller_a = round(physics.power_output / 12.0 * 1.25, 1)

    plan.array_kwp = (physics.temp_corrected_power or physics.power_output) / 1000.0 * array_size
    plan.daily_kwh = plan.array_kwp * region.peak_sun_hours
    plan.annual_kwh = plan.daily_kwh * 365.0 * 0.85  # availability/soiling haircut
    plan.battery_ah_12v = plan.daily_kwh * 1000.0 / 12.0 / 0.5  # one day autonomy at 50% DoD

    plan.findings.append(
        Finding(
            "DEP_YIELD",
            "info",
            f"At {region.name} insolation ({region.peak_sun_hours:.1f} peak-sun-hours/day) this "
            f"panel should realistically make {plan.daily_kwh:.2f} kWh/day "
            f"({plan.annual_kwh:.0f} kWh/year) once soiling and inverter losses are counted.",
            weight=0.0,
        )
    )
    if plan.series_note:
        plan.findings.append(Finding("DEP_STRING", "info", plan.series_note, weight=0.0))
    if plan.cable_mm2 and plan.imp:
        plan.findings.append(
            Finding(
                "DEP_CABLE",
                "info",
                f"For a 10 m run at {plan.imp:.2f} A keep copper at {plan.cable_mm2:.1f} mm² "
                f"or thicker to hold voltage drop under 2%. Undersized cable is the most common "
                f"cause of 'my panel makes no power' reports that have nothing to do with the panel.",
                weight=0.0,
            )
        )
    if plan.fuse_a:
        plan.findings.append(
            Finding(
                "DEP_FUSE",
                "warn",
                f"Fuse each string at {plan.fuse_a:.0f} A (1.56x Isc) and size the charge "
                f"controller for at least {plan.controller_a:.0f} A. Skipping the fuse protects "
                f"nothing: a single shorted cell can pull the whole array's current through it.",
                weight=0.0,
            )
        )

    # Zero authenticity signal, by design.
    return (
        AgentOpinion(
            agent=A_DEPLOY,
            verdict=None,
            score=0.0,
            confidence=0.1,
            reliability=0.9,
            findings=plan.findings,
            rationale="Deployment and safety maths only — I deliberately cast no authenticity vote.",
        ),
        plan,
    )


# ════════════════════════════════════════════════════════════════════════════
# Red team (adversarial prosecutor)
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class CounterCase:
    target_class: int
    probability: float
    arguments: List[dict] = field(default_factory=list)  # {claim, test, cost, flips_to}
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "target_class": self.target_class,
            "probability": round(self.probability, 3),
            "arguments": self.arguments,
            "summary": self.summary,
        }


FALSIFICATION_TESTS = {
    "voltage": ("Measure Vmp again with a multimeter on a resistive load bank", "free", "10 min"),
    "current": ("Clamp-meter Imp at solar noon with the panel aimed at the sun", "free", "15 min"),
    "irradiance": ("Measure irradiance with a pyranometer or a calibrated cell at the panel plane", "low", "5 min"),
    "temperature": ("Measure back-of-panel temperature with an IR thermometer", "low", "5 min"),
    "efficiency": ("Compute efficiency from measured area and power, not from the label", "free", "10 min"),
    "voc": ("Measure Voc with the panel disconnected, at first light and at noon, and compare the ratio", "free", "10 min"),
    "isc": ("Measure Isc with a clamp meter on a shorted lead (seconds only)", "free", "5 min"),
    "fill_factor": ("Sweep the I-V curve with a resistive load bank, or use an MPPT logger for one day", "medium", "1 day"),
}


def red_team_opinion(
    opinions: Sequence[AgentOpinion],
    fused_probs: Dict[str, float],
    physics: PhysicsResult,
    region: Region,
    label_check: Optional[dict],
    pmax_rated: Optional[float] = None,
) -> tuple:
    """Argue the strongest possible case against the emerging consensus.

    The adversary is only useful if it is *specific*. Each candidate argument
    therefore carries a relevance score computed from the actual readings — a
    heat argument is worthless against a panel measured at 28 °C, and a
    thermometer-error argument is worthless when the shortfall is not marginal.
    Arguments are ranked by relevance and only the ones that survive are shown.
    """
    valid = [o for o in opinions if o.verdict is not None and not o.aborted]
    if valid:
        vote: Dict[int, float] = {}
        for o in valid:
            vote[o.verdict] = vote.get(o.verdict, 0.0) + o.weight
        consensus = max(vote, key=vote.get) if vote else 1
    else:
        consensus = 1

    probs = np.array(
        [fused_probs.get(name, 0.25) for name in
         ["HEALTHY_PREMIUM", "HEALTHY_BASIC", "DEGRADED", "COUNTERFEIT"]]
    )
    if consensus <= 1:
        challenger = int(np.argmax(probs[2:]) + 2)
    else:
        challenger = int(np.argmax(probs[:2]))
    challenger_p = float(probs[challenger])

    flip = ["HEALTHY_PREMIUM", "HEALTHY_BASIC", "DEGRADED", "COUNTERFEIT"]

    # ── Quantities the arguments are built from ────────────────────────────
    temp = float(physics.clamp.get("temperature", 25.0) or 25.0)
    dT = temp - 25.0
    measured = float(physics.power_output or 0.0)
    expected = float(pmax_rated) if pmax_rated else None
    shortfall = None
    if expected and expected > 0 and measured > 0:
        shortfall = max(0.0, 1.0 - measured / expected)
    ff = float(physics.fill_factor or 0.0)
    gamma = 0.35  # %/degC
    heat_loss_pct = gamma * max(0.0, dT)
    match_conf = float((physics.dataclass_match or {}).get("confidence", 0.0))
    flags = list((label_check or {}).get("red_flags") or [])
    cert_missing = bool(label_check) and label_check.get("certification_marks_visible") is False

    candidates: List[dict] = []

    def propose(relevance: float, claim: str, test_key: str, flips_to: int) -> None:
        test, cost, duration = FALSIFICATION_TESTS.get(
            test_key, ("Re-measure the input", "free", "10 min")
        )
        candidates.append(
            {
                "relevance": float(max(0.0, min(1.0, relevance))),
                "claim": claim,
                "test": test,
                "cost": cost,
                "duration": duration,
                "flips_to": flip[flips_to],
            }
        )

    if consensus >= 2:
        # The council says the panel is bad: argue that it is genuine.
        propose(
            min(1.0, max(0.0, (temp - 30.0) / 20.0) + (0.25 if dT > 8 else 0.0)),
            f"At a cell temperature of {temp:.0f} °C the module is {dT:.0f} °C above STC, which costs "
            f"about {heat_loss_pct:.1f}% of rated power before anything is wrong with it. Heat alone "
            f"can pull a genuine panel's measured output into the range this reading shows.",
            "temperature",
            1,
        )
        if shortfall is not None:
            # A moderate shortfall is plausible as an instrument problem; a
            # catastrophic one is not.
            instrument_relevance = 1.0 - min(1.0, abs(shortfall - 0.6) / 0.6)
            propose(
                max(0.15, instrument_relevance),
                f"Measured power is {measured:.1f} W against a {expected:.0f} W nameplate "
                f"({shortfall*100:.0f}% short). A clamp meter on the wrong range, a partly shaded "
                f"corner, a panel propped at the wrong tilt or a load that is not at maximum power "
                f"can each produce exactly this kind of deficit with a perfectly honest module.",
                "current",
                1,
            )
        propose(
            0.35 + (0.4 if match_conf > 0.5 else 0.0) + (0.2 if ff >= FF_HEALTHY * 0.8 else 0.0),
            (
                f"The electrical fingerprint matches a {physics.dataclass_match['brand']} "
                f"{physics.dataclass_match['model']} at {match_conf*100:.0f}% confidence. If that "
                f"match is right, the hardware is genuine and what is being priced as a fake may be "
                f"an old, honest panel — a commercial dispute, not a counterfeit."
                if match_conf > 0.5
                else "Counterfeit sellers frequently re-label a genuine second-hand panel. The cells "
                     "may be entirely real while the sticker claims new-rated power."
            ),
            "fill_factor",
            1 if match_conf > 0.5 else 2,
        )
        if max(probs[0], probs[1]) > 0.05:
            propose(
                float(max(probs[0], probs[1]) * 3.0),
                f"The council still assigns {max(probs[0], probs[1])*100:.0f}% of the belief to this "
                f"panel being genuine. At that mass this is not a closed case, and one more "
                f"independent reading could move it.",
                "voc",
                1,
            )
    else:
        # The council says the panel is good: argue that it is a fake.
        propose(
            min(1.0, 0.4 + 0.5 * len(flags) + (0.3 if cert_missing else 0.0)),
            (
                "The sticker itself already raised "
                + (
                    f"{len(flags)} red flag(s) ({'; '.join(str(f) for f in flags[:2])})"
                    if flags
                    else "certification marks missing"
                )
                + ", and a clean electrical reading does not clean up a label. Fakes are often built "
                  "from the genuine cells of a scrapped panel."
                if (flags or cert_missing)
                else "A counterfeit module built from genuine scrap cells will pass a single "
                     "spot measurement and still fail under a day of real load."
            ),
            "voc",
            3,
        )
        propose(
            min(1.0, 0.35 + 0.5 * (1.0 - min(1.0, ff / max(FF_HEALTHY, 1e-6)))),
            f"Fill factor is {ff:.3f} against a {FF_HEALTHY:.2f} healthy line. Series resistance "
            f"climbs steeply with cell temperature, so one cool-morning reading systematically "
            f"flatters a panel that is ugly by noon.",
            "fill_factor",
            2,
        )
        propose(
            0.45,
            "A single snapshot cannot see degradation that only appears under sustained load: a "
            "hairline cell crack or a partially shaded bypass path looks perfectly healthy at the "
            "moment of measurement and costs real energy over a day.",
            "irradiance",
            2,
        )
        propose(
            float(region.counterfeit_prevalence),
            f"{region.counterfeit_prevalence*100:.0f}% of panels in circulation around {region.name} "
            f"are reported as counterfeit. That base rate is a reason to verify the serial number "
            f"with the manufacturer before paying, whatever the electrical reading says.",
            "efficiency",
            3,
        )

    candidates.sort(key=lambda c: -c["relevance"])
    arguments = [
        {k: v for k, v in c.items() if k != "relevance"}
        for c in candidates
        if c["relevance"] >= 0.3
    ][:4]
    if not arguments and candidates:  # never leave the prosecutor mute
        arguments = [{k: v for k, v in candidates[0].items() if k != "relevance"}]

    confidence = float(max(0.15, min(0.9, challenger_p * 1.6)))
    summary = (
        f"Strongest case for {flip[challenger]} against the current {flip[consensus]} leaning: "
        f"{len(arguments)} argument(s), backed by {challenger_p*100:.0f}% of the fused belief."
    )
    return (
        AgentOpinion(
            agent=A_RED,
            verdict=challenger,
            score=normalise_score(CLASS_ANCHORS[challenger] * min(1.0, challenger_p * 2.2)),
            confidence=confidence,
            reliability=0.55,
            findings=[
                Finding(
                    "RED_ARGUMENT",
                    "warn",
                    a["claim"] + f" Test that would settle it: {a['test']}.",
                    weight=0.0,
                )
                for a in arguments
            ],
            rationale=summary,
        ),
        CounterCase(
            target_class=challenger,
            probability=challenger_p,
            arguments=arguments,
            summary=summary,
        ),
    )
