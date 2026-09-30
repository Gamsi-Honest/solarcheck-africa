"""
GSI orchestrator — where the agents become a single, auditable judgement.

Pipeline::

    inputs ─► law layer (physics.evaluate)
           └► council  ── MLForecast ─┐
                        ── Physics   ─┤
                        ── Perception┤─► fusion (weighted log-opinion pool
                        ── LabelForge┤   + regional base-rate prior)
                        ── Economics ─┤
                        ── RedTeam   ─┘
           ─► metacognition: consistency audit, hard vetoes, calibration,
              uncertainty decomposition, self-critique
           ─► planner: expected information gain for the next measurement
           ─► decision: verdict state, money at risk, safe deployment

Design commitments (these are the difference between "more features" and GSI):

  1. Physics outranks statistics. A hard physical violation vetoes the
     classifier outright, and the report says which law was broken.
  2. The system may answer "I don't know yet, and here is what to measure."
     INCONCLUSIVE is a first-class outcome, not a failure.
  3. Confidence is never the raw softmax. It is fused agreement, damped by
     out-of-distribution distance, evidence breadth and measurement quality.
  4. Every number in the report is traceable to an agent, a law or a formula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import agents, perception, physics
from .agents import (
    A_DEPLOY,
    A_ECON,
    A_LABEL,
    A_ML,
    A_PERCEPTION,
    A_PHYSICS,
    A_RED,
    FEATURES,
    DeploymentPlan,
    EconomicsResult,
)
from .evidence import AgentOpinion, CLASS_ANCHORS, LABELS
from .physics import Finding, PhysicsResult
from .regions import Region, get_region

CLASS_ORDER = ["HEALTHY_PREMIUM", "HEALTHY_BASIC", "DEGRADED", "COUNTERFEIT"]
TRAINING_PREVALENCE = 0.25  # each class is 25% of the training set

# Parameters a technician can actually put an instrument on, as opposed to
# quantities the engine derives from them.
MEASURABLE_FEATURES = ("voltage", "current", "irradiance", "temperature", "efficiency", "voc", "isc")

# Position of each measurable parameter inside the model's feature vector
# (voc and isc are inputs to the physics layer and datasheet reasoning, but the
# forest consumes derived features instead).
MODEL_FEATURE_INDEX = {name: i for i, name in enumerate(FEATURES)}

DEFAULT_VALUES = {
    "voltage": 22.8,
    "current": 4.39,
    "irradiance": 980.0,
    "temperature": 29.0,
    "efficiency": 19.4,
    "voc": 27.36,
    "isc": 4.87,
}

AGENT_RELIABILITY = {
    A_ML: 0.9,
    A_PHYSICS: 0.98,
    A_PERCEPTION: 0.75,
    A_LABEL: 0.7,
    A_ECON: 0.6,
    A_DEPLOY: 0.0,  # deliberately casts no authenticity vote
    # The red team is a *critic*, not an evidence channel: its arguments are a
    # reinterpretation of evidence already counted, so letting it vote would
    # double-count that evidence and could even make its "case against" argue
    # against a verdict it had just produced. It informs the human instead.
    A_RED: 0.0,
}


# ════════════════════════════════════════════════════════════════════════════
# Inputs
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class PanelInputs:
    voltage: Optional[float] = None  # Vmp
    current: Optional[float] = None  # Imp
    irradiance: Optional[float] = None
    temperature: Optional[float] = None
    efficiency: Optional[float] = None
    voc: Optional[float] = None
    isc: Optional[float] = None
    pmax_rated: Optional[float] = None
    brand: Optional[str] = None
    model_text: Optional[str] = None
    label_check: Optional[dict] = None
    images: Sequence = ()
    price: Optional[float] = None
    region: str = "yaounde"
    source: str = "manual"  # manual | label | mixed
    array_size: int = 1
    controller_vmax: float = 150.0
    provided: Tuple[str, ...] = ()

    def is_measured(self, name: str) -> bool:
        return name in self.provided


@dataclass
class UncertaintyReport:
    total_entropy_bits: float = 0.0
    aleatoric_bits: float = 0.0  # disagreement inherent in the inputs
    epistemic_bits: float = 0.0  # reducible by measuring more
    agreement: float = 0.0  # 1 - Jensen-Shannon distance between agents
    ood_percentile: float = 0.0
    evidence_breadth: float = 0.0
    evidence_gap_bits: float = 0.0  # allowance for never-measured parameters
    channel_sharpening_bits: float = 0.0  # agreement collapsed by pooling (diagnostic)
    narrative: str = ""

    def to_dict(self) -> dict:
        return {
            "total_entropy_bits": round(self.total_entropy_bits, 3),
            "aleatoric_bits": round(self.aleatoric_bits, 3),
            "epistemic_bits": round(self.epistemic_bits, 3),
            "agreement": round(self.agreement, 3),
            "ood_percentile": self.ood_percentile,
            "evidence_breadth": round(self.evidence_breadth, 3),
            "evidence_gap_bits": round(self.evidence_gap_bits, 3),
            "channel_sharpening_bits": round(self.channel_sharpening_bits, 3),
            "narrative": self.narrative,
        }


@dataclass
class NextTest:
    feature: str
    expected_information_gain_bits: float
    expected_entropy_after: float
    how: str
    why: str

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "expected_information_gain_bits": round(self.expected_information_gain_bits, 3),
            "expected_entropy_after": round(self.expected_entropy_after, 3),
            "how": self.how,
            "why": self.why,
        }


@dataclass
class GSIReport:
    state: str  # VERDICT | PROBABLE | INCONCLUSIVE | MEASUREMENT_ERROR
    verdict_class: int
    headline: str
    confidence: float  # 0..1 calibrated
    probabilities: Dict[str, float]
    narrative: str
    decision: str
    uncertainty: UncertaintyReport
    opinions: List[AgentOpinion]
    physics: PhysicsResult
    economics: Optional[EconomicsResult] = None
    deployment: Optional[DeploymentPlan] = None
    counter_case: Optional[agents.CounterCase] = None
    next_tests: List[NextTest] = field(default_factory=list)
    metacognition: List[Finding] = field(default_factory=list)
    veto_reason: Optional[str] = None
    region: Optional[Region] = None
    features: Dict[str, float] = field(default_factory=dict)
    audit: List[dict] = field(default_factory=list)
    engine_version: str = "3.0.0-gsi"

    # ── convenience ────────────────────────────────────────────────────────
    @property
    def label(self) -> str:
        """The raw class name, for programmatic use."""
        return LABELS[self.verdict_class]

    @property
    def display_label(self) -> str:
        """What a human should be shown.

        When the state is not a verdict, naming a class would be misleading —
        the whole point of MEASUREMENT_ERROR is that no class has been earned.
        """
        if self.state == "MEASUREMENT_ERROR":
            return "NO VERDICT — INVALID MEASUREMENT"
        if self.state == "INCONCLUSIVE":
            return "INCONCLUSIVE — MORE EVIDENCE NEEDED"
        return LABELS[self.verdict_class].replace("_", " ")

    def to_dict(self) -> dict:
        return {
            "engine_version": self.engine_version,
            "state": self.state,
            "verdict": self.label,
            "display_label": self.display_label,
            "verdict_class": self.verdict_class,
            "headline": self.headline,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "narrative": self.narrative,
            "decision": self.decision,
            "veto_reason": self.veto_reason,
            "uncertainty": self.uncertainty.to_dict(),
            "next_tests": [t.to_dict() for t in self.next_tests],
            "metacognition": [f.to_dict() for f in self.metacognition],
            "counter_case": None if self.counter_case is None else self.counter_case.to_dict(),
            "economics": None if self.economics is None else self.economics.to_dict(),
            "deployment": None if self.deployment is None else self.deployment.to_dict(),
            "region": None if self.region is None else self.region.key,
            "features": {k: round(float(v), 4) for k, v in self.features.items()},
            "opinions": [o.to_dict() for o in self.opinions],
            "audit": self.audit,
        }


# ════════════════════════════════════════════════════════════════════════════
# Fusion primitives
# ════════════════════════════════════════════════════════════════════════════
def _score_to_belief(score: float, confidence: float, sharpness: float = 6.0) -> np.ndarray:
    """Convert an agent's scalar opinion into a distribution over the 4 classes.

    The agent's point estimate is matched to class anchors with a Gaussian
    compatibility kernel; the result is then tempered by the agent's
    confidence (a low-confidence agent is mixed back toward uniform, so it
    cannot dominate the pool by accident).
    """
    score = float(max(-1.0, min(1.0, score)))
    conf = float(max(0.0, min(1.0, confidence)))
    anchors = np.array([CLASS_ANCHORS[c] for c in range(4)])
    logits = -sharpness * (score - anchors) ** 2
    logits -= logits.max()
    belief = np.exp(logits)
    belief /= belief.sum()
    uniform = np.full(4, 0.25)
    return conf * belief + (1.0 - conf) * uniform


def fuse_beliefs(
    beliefs: Dict[str, np.ndarray],
    weights: Dict[str, float],
    prior: Optional[np.ndarray] = None,
    prior_strength: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """Weighted logarithmic opinion pool + tempered regional base rate.

    Returns (fused_probabilities, normalised_weights_sum).
    """
    total = sum(max(0.0, weights.get(name, 0.0)) for name in beliefs)
    if total <= 0:
        return np.full(4, 0.25), 0.0

    log_post = np.zeros(4)
    for name, belief in beliefs.items():
        w = max(0.0, weights.get(name, 0.0)) / total
        if w <= 0:
            continue
        log_post += w * np.log(np.clip(belief, 1e-9, 1))

    if prior is not None:
        log_post += prior_strength * np.log(np.clip(prior, 1e-6, 1))

    log_post -= log_post.max()
    post = np.exp(log_post)
    post /= post.sum()
    return post, total


def _js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon distance in [0,1] between two distributions."""
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.clip(b[mask], 1e-12, 1))))
    return float(math.sqrt(max(0.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m))))


def _entropy_bits(p: np.ndarray) -> float:
    return float(-np.sum(p * np.log2(np.clip(p, 1e-12, 1))))


# ════════════════════════════════════════════════════════════════════════════
# Engine
# ════════════════════════════════════════════════════════════════════════════
class GSIEngine:
    def __init__(
        self,
        model,
        scaler,
        priors: Optional[dict] = None,
        calibrators: Optional[Sequence[Tuple[float, float]]] = None,
        calibration_note: str = "",
    ):
        self.model = model
        self.scaler = scaler
        self.priors = priors if priors is not None else agents.load_priors()
        self.calibrators = calibrators
        self.calibration_note = calibration_note or (
            "out-of-fold Platt scaling" if calibrators else "raw model probabilities (uncalibrated)"
        )

    # ── calibration ────────────────────────────────────────────────────────
    def calibrate_from_csv(self, csv_path: str) -> str:
        """Fit calibration coefficients from the training CSV, if present."""
        import pandas as pd  # imported lazily so the engine works without pandas

        try:
            df = pd.read_csv(csv_path)
            X = df[FEATURES].to_numpy(dtype=float)
            y = df["label"].to_numpy(dtype=int)
            coefs = agents.build_calibrators(self.model, self.scaler, X, y)
            if coefs:
                self.calibrators = coefs
                self.calibration_note = "out-of-fold Platt scaling"
                return self.calibration_note
        except Exception as exc:  # pragma: no cover - depends on deployment files
            return f"calibration unavailable ({type(exc).__name__}); using raw probabilities"
        return "calibration unavailable; using raw probabilities"

    # ── main entry point ───────────────────────────────────────────────────
    def reason(self, inputs: PanelInputs) -> GSIReport:
        region = get_region(inputs.region)
        audit: List[dict] = []

        # 1 ── Resolve inputs (fill gaps with estimates, remembering which).
        resolved, estimated = self._resolve(inputs)
        audit.append({"step": "resolve_inputs", "estimated": sorted(estimated), "resolved": {
            k: round(float(v), 4) for k, v in resolved.items() if isinstance(v, (int, float))
        }})

        # 2 ── Physics law layer.
        phys = physics.evaluate(
            voltage=resolved["voltage"],
            current=resolved["current"],
            irradiance=resolved["irradiance"],
            temperature=resolved["temperature"],
            efficiency=resolved["efficiency"],
            voc=resolved["voc"],
            isc=resolved["isc"],
            pmax_rated=inputs.pmax_rated,
        )
        clamp = phys.clamp
        feature_vector = self._feature_vector(clamp)

        # 3 ── Continuous channels.
        snapshot = agents.ml_snapshot(self.model, self.scaler, self.priors, self.calibrators, feature_vector)
        breadth = self._evidence_breadth(inputs, phys)
        physics_op = agents.physics_opinion(
            phys, breadth, pmax_rated=inputs.pmax_rated, efficiency=clamp["efficiency"]
        )
        ml_op = agents.ml_opinion(snapshot, feature_vector)

        # 4 ── Discrete channels.
        perc_assessment = None
        if inputs.images:
            perc_assessment = perception.analyse(inputs.images)
        perception_op = perception.opinion(perc_assessment or perception.PerceptionAssessment(), inputs.label_check)
        label_op = agents.label_opinion(inputs.brand, inputs.model_text, inputs.label_check, self._label_payload(inputs), phys)

        opinions: List[AgentOpinion] = [ml_op, physics_op, perception_op, label_op]

        # 5 ── First-pass belief (needed to inform economics and red team).
        beliefs, weights = self._beliefs_from(opinions)
        prior = self._regional_prior(region)
        pre_probs, _ = fuse_beliefs(beliefs, weights, prior=prior)
        econ_op, econ = agents.economics_opinion(
            inputs.price, inputs.pmax_rated, phys, region, confidence_in_verdict=float(pre_probs[3] + pre_probs[2])
        )
        deploy_op, deploy = agents.deployment_opinion(phys, region, inputs.array_size, inputs.controller_vmax)
        opinions.extend([econ_op, deploy_op])

        # 6 ── Final belief from every voting channel.
        beliefs, weights = self._beliefs_from(opinions)
        probs, _ = fuse_beliefs(beliefs, weights, prior=prior)
        probs_map = {CLASS_ORDER[i]: float(probs[i]) for i in range(4)}

        # 7 ── The prosecutor attacks the verdict that will actually be reported.
        red_op, counter_case = agents.red_team_opinion(
            opinions, probs_map, phys, region, inputs.label_check, pmax_rated=inputs.pmax_rated
        )
        opinions.append(red_op)  # reported, but weightless in the fusion above

        # 8 ── Uncertainty decomposition.
        uncertainty = self._uncertainty(opinions, beliefs, weights, probs, snapshot, breadth, estimated)

        # 9 ── Metacognition: audit, vetoes, calibration, self-critique.
        verdict_class, state, veto_reason, meta_findings = self._metacognition(
            inputs, phys, snapshot, probs, opinions, uncertainty, estimated, region
        )

        # 9b ── If physics vetoed, the reported belief must say so too.
        # Otherwise the headline would read "COUNTERFEIT at 95%" while the
        # probability bars still showed the statistical posterior — the exact
        # kind of internal contradiction this architecture exists to prevent.
        if veto_reason and state != "MEASUREMENT_ERROR":
            state = "VERDICT"
            floor = float(max(0.90, probs[verdict_class]))
            remaining = 1.0 - floor
            others = probs.copy()
            others[verdict_class] = 0.0
            total_other = float(others.sum())
            if total_other > 0:
                others = others / total_other
            else:
                others = np.full(4, 0.25)
            probs = others * remaining
            probs[verdict_class] = floor
            probs_map = {CLASS_ORDER[i]: float(probs[i]) for i in range(4)}
            audit.append(
                {
                    "step": "veto_override",
                    "law_broken": veto_reason,
                    "statistical_posterior_before_override": {
                        k: round(float(v), 4)
                        for k, v in zip(CLASS_ORDER, others * remaining + 0)
                    },
                    "forced_class": CLASS_ORDER[verdict_class],
                    "reason": "Physical law outranks statistical association.",
                }
            )
        confidence = self._calibrated_confidence(probs, verdict_class, state, uncertainty, snapshot, veto_reason)

        # 10 ── Planner: what single measurement would most reduce the doubt?
        next_tests = self._plan_next_tests(inputs, resolved, phys, probs, prior)

        # Fold the planner's achievable gain into the epistemic figure: the
        # decomposition above only sees the channels that already spoke, while
        # the planner sees the evidence that is still missing.
        if next_tests:
            best_gain = next_tests[0].expected_information_gain_bits
            uncertainty.epistemic_bits = max(uncertainty.epistemic_bits, best_gain)
            uncertainty.narrative = self._uncertainty_narrative(uncertainty, estimated, snapshot)
            uncertainty.narrative += (
                f" The single most valuable next measurement is {next_tests[0].feature} "
                f"({best_gain:.2f} bits recoverable), then "
                + ", ".join(t.feature for t in next_tests[1:]) + "."
                if len(next_tests) > 1
                else f" The single most valuable next measurement is {next_tests[0].feature} "
                     f"({best_gain:.2f} bits recoverable)."
            )

        # 11 ── Decision + narrative.
        decision = self._decision(state, verdict_class, econ, confidence)
        narrative = self._narrative(
            state, verdict_class, confidence, probs, phys, uncertainty, econ, counter_case,
            meta_findings, region, estimated, inputs,
        )
        headline = self._headline(state, verdict_class, confidence, veto_reason)

        audit.append(
            {
                "step": "fusion",
                "weights": {k: round(float(v), 3) for k, v in weights.items()},
                "posterior": {k: round(v, 4) for k, v in probs_map.items()},
                "calibration": self.calibration_note,
            }
        )
        audit.append(
            {
                "step": "metacognition",
                "state": state,
                "checks": [f.code for f in meta_findings],
                "confidence": round(confidence, 4),
            }
        )
        audit.append(
            {
                "step": "planner",
                "top_test": next_tests[0].feature if next_tests else None,
                "eig_bits": round(next_tests[0].expected_information_gain_bits, 3) if next_tests else 0.0,
            }
        )

        return GSIReport(
            state=state,
            verdict_class=verdict_class,
            headline=headline,
            confidence=confidence,
            probabilities=probs_map,
            narrative=narrative,
            decision=decision,
            uncertainty=uncertainty,
            opinions=opinions,
            physics=phys,
            economics=econ,
            deployment=deploy,
            counter_case=counter_case,
            next_tests=next_tests,
            metacognition=meta_findings,
            veto_reason=veto_reason,
            region=region,
            features={name: float(value) for name, value in zip(FEATURES, feature_vector)},
            audit=audit,
        )

    # ── helpers ────────────────────────────────────────────────────────────
    def _resolve(self, inputs: PanelInputs) -> Tuple[dict, set]:
        """Fill missing measurements using datasheet relations, and record
        which values are estimates rather than measurements."""
        estimated = set()

        def _pick(name, raw, derived_fn=None):
            if raw is not None and raw != "" and float(raw) != 0.0:
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    pass
            estimated.add(name)
            return derived_fn() if derived_fn else DEFAULT_VALUES[name]

        voc = _pick("voc", inputs.voc, lambda: (float(inputs.voltage) / 0.8 if inputs.voltage else DEFAULT_VALUES["voc"]))
        isc = _pick("isc", inputs.isc, lambda: (float(inputs.current) / 0.9 if inputs.current else DEFAULT_VALUES["isc"]))
        voltage = _pick("voltage", inputs.voltage, lambda: voc * 0.8)
        current = _pick("current", inputs.current, lambda: isc * 0.9)
        efficiency = _pick("efficiency", inputs.efficiency)
        irradiance = _pick("irradiance", inputs.irradiance)
        temperature = _pick("temperature", inputs.temperature)

        if inputs.provided:
            estimated -= {f for f in inputs.provided}

        return (
            {
                "voltage": voltage,
                "current": current,
                "irradiance": irradiance,
                "temperature": temperature,
                "efficiency": efficiency,
                "voc": voc,
                "isc": isc,
            },
            estimated,
        )

    def _feature_vector(self, clamp: dict) -> np.ndarray:
        return np.array(
            [
                clamp["voltage"],
                clamp["current"],
                clamp["irradiance"],
                clamp["temperature"],
                clamp["efficiency"],
                clamp["power_output"],
                clamp["fill_factor"],
                clamp["temp_corrected_efficiency"],
            ],
            dtype=float,
        )

    def _evidence_breadth(self, inputs: PanelInputs, phys: PhysicsResult) -> int:
        breadth = 0
        if inputs.voltage or inputs.current:
            breadth += 1
        if inputs.voc or inputs.isc:
            breadth += 1
        if inputs.efficiency:
            breadth += 1
        if inputs.images:
            breadth += 1
        if inputs.label_check or inputs.brand:
            breadth += 1
        if inputs.price:
            breadth += 1
        if inputs.pmax_rated:
            breadth += 1
        return min(breadth, 5)

    def _label_payload(self, inputs: PanelInputs) -> Optional[dict]:
        if inputs.pmax_rated is None and not inputs.label_check:
            return None
        return {"pmax": inputs.pmax_rated or 0}

    def _beliefs_from(self, opinions: Sequence[AgentOpinion]) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        beliefs: Dict[str, np.ndarray] = {}
        weights: Dict[str, float] = {}
        for op in opinions:
            if op.verdict is None or op.aborted or op.confidence <= 0:
                continue
            if AGENT_RELIABILITY.get(op.agent) == 0.0:
                continue  # non-voting channel (critic / deployment maths)
            base = AGENT_RELIABILITY.get(op.agent, 0.5)
            # The agent's own reliability estimate modulates the prior trust.
            trust = float(max(0.05, min(1.0, 0.5 * base + 0.5 * op.reliability)))
            weight = trust * op.confidence
            if weight <= 0:
                continue
            # The ML agent publishes a full distribution; use it directly.
            if op.agent == A_ML and op.findings:
                raw = next((f.evidence.get("probs") for f in op.findings if f.code == "ML_VERDICT"), None)
                if raw:
                    dist = np.asarray(raw, dtype=float)
                    if dist.size == 4 and dist.sum() > 0:
                        dist = dist / dist.sum()
                        uniform = np.full(4, 0.25)
                        beliefs[op.agent] = op.confidence * dist + (1 - op.confidence) * uniform
                        weights[op.agent] = weight
                        continue
            beliefs[op.agent] = _score_to_belief(op.score, op.confidence)
            weights[op.agent] = weight
        return beliefs, weights

    def _regional_prior(self, region: Region) -> np.ndarray:
        """Tempered regional base rate for the COUNTERFEIT class.

        Training data is 25% counterfeit by construction, which is not the
        same as the local market. We shift the prior by sqrt(ratio) so the
        regional statistic nudges belief without steamrolling the evidence.
        """
        ratio = max(0.05, min(0.9, region.counterfeit_prevalence)) / TRAINING_PREVALENCE
        factor = math.sqrt(ratio)
        prior = np.array([1.0, 1.0, 1.0, factor], dtype=float)
        return prior / prior.sum()

    def _uncertainty(
        self,
        opinions: Sequence[AgentOpinion],
        beliefs: Dict[str, np.ndarray],
        weights: Dict[str, float],
        probs: np.ndarray,
        snapshot: agents.MLSnapshot,
        breadth: int,
        estimated: set,
    ) -> UncertaintyReport:
        report = UncertaintyReport()
        report.total_entropy_bits = _entropy_bits(probs)
        report.evidence_breadth = breadth / 5.0

        total_w = sum(weights.values()) or 1.0
        # Standard decomposition: total predictive entropy H(P) splits into
        #   aleatoric = E_i[H(P_i)]  (what each channel cannot resolve alone)
        #   epistemic = H(P) - E_i[H(P_i)]  (mutual information between channels,
        #              i.e. what more *independent* evidence would remove)
        # plus an explicit, separate allowance for parameters that were never
        # measured at all.
        mean_channel_entropy = sum(weights[name] * _entropy_bits(beliefs[name]) for name in beliefs) / total_w
        # A logarithmic opinion pool is sharper than the mean of its inputs, so
        # the naive "mean channel entropy" can exceed the pooled entropy. Cap it
        # so the reported split is always coherent (the raw figure is kept for
        # diagnostics), then treat the leftover as reducible uncertainty.
        report.aleatoric_bits = min(mean_channel_entropy, report.total_entropy_bits)
        report.epistemic_bits = max(0.0, report.total_entropy_bits - report.aleatoric_bits)
        report.channel_sharpening_bits = max(0.0, mean_channel_entropy - report.total_entropy_bits)
        report.evidence_gap_bits = 0.9 * len(estimated) / len(MEASURABLE_FEATURES)

        # Agreement: mean Jensen-Shannon distance of each channel to the pool.
        distances = [_js_distance(beliefs[name], probs) * weights[name] for name in beliefs]
        report.agreement = float(max(0.0, 1.0 - sum(distances) / total_w * 1.6))

        if snapshot.ood:
            report.ood_percentile = float(
                snapshot.ood.get("class_percentile", snapshot.ood["percentile"])
            )

        # "Removable" uncertainty is whichever is larger: the information the
        # channels disagree about, or the plain hole left by unmeasured inputs.
        report.epistemic_bits = max(report.epistemic_bits, report.evidence_gap_bits)
        report.narrative = self._uncertainty_narrative(report, estimated, snapshot)
        return report

    def _uncertainty_narrative(
        self, report: "UncertaintyReport", estimated: set, snapshot: agents.MLSnapshot
    ) -> str:
        parts = [
            f"{report.total_entropy_bits:.2f} bits of total uncertainty: "
            f"{report.aleatoric_bits:.2f} bits is ambiguity inherent to this evidence, "
            f"{report.epistemic_bits:.2f} bits is removable by measuring more"
        ]
        if report.epistemic_bits > 0.05:
            parts[-1] += (
                ". The removable part is the difference between 'this looks wrong' and "
                "'this is wrong', and the planner below attacks exactly that part."
            )
        else:
            parts[-1] += " — this reading has already been squeezed dry, more of the same data will not help."
        if report.agreement < 0.55 and report.total_entropy_bits > 0.3:
            parts.append(
                "The council disagrees with itself — the agents' independent channels point in "
                "different directions, so treat the headline as provisional."
            )
        else:
            parts.append(f"Council agreement is {report.agreement*100:.0f}%.")
        if snapshot.ood and snapshot.ood.get("out_of_distribution"):
            parts.append(
                "The reading falls outside the region the model was trained on, so the statistical "
                "channel is voluntarily discounted and the physical channel carries more of the weight."
            )
        if estimated:
            parts.append(
                f"{len(estimated)} parameter(s) were estimated rather than measured: "
                f"{', '.join(sorted(estimated))}."
            )
        return " ".join(parts)

    def _metacognition(
        self,
        inputs: PanelInputs,
        phys: PhysicsResult,
        snapshot: agents.MLSnapshot,
        probs: np.ndarray,
        opinions: Sequence[AgentOpinion],
        uncertainty: UncertaintyReport,
        estimated: set,
        region: Region,
    ) -> Tuple[int, str, Optional[str], List[Finding]]:
        findings: List[Finding] = []
        ordered = np.argsort(-probs)
        verdict_class = int(ordered[0])
        top = float(probs[ordered[0]])
        second = float(probs[ordered[1]])
        veto_reason = None

        # ── Check 1: hard physical violations ──────────────────────────────
        if phys.hard_violations:
            detail = "; ".join(phys.hard_violations[:2])
            from_label = inputs.source in ("label", "mixed") or (inputs.label_check is not None)
            if from_label:
                verdict_class = 3
                veto_reason = (
                    f"VETO: physical law broken by the printed specification ({detail}). "
                    f"No measurement error can explain a label that contradicts physics — "
                    f"the panel is misrepresented."
                )
                findings.append(Finding("META_VETO_LABEL", "critical", veto_reason, weight=0.0))
            else:
                findings.append(
                    Finding(
                        "META_VETO_MEASUREMENT",
                        "critical",
                        f"Physical law broken by the *measurement* ({detail}). This is almost "
                        f"always an instrument or transcription problem, not a counterfeit. "
                        f"Re-measure before drawing any conclusion — I will not certify or "
                        f"condemn a panel on self-contradictory data.",
                        weight=0.0,
                    )
                )
                # A self-contradictory *measurement* is an instrumentation problem,
                # not a verdict about a vendor: no veto, no confidence, no class.
                return (verdict_class, "MEASUREMENT_ERROR", None, findings)

        # ── Check 2: statistics vs physics consistency ─────────────────────
        # Map the model's class probabilities onto the same 0..1 suspicion
        # axis the physics layer uses, so the two channels are comparable.
        ml_suspicion = float(np.dot(snapshot.probs, [0.0, 0.25, 0.7, 1.0]))
        coherence = physics.coherence_score(phys, ml_suspicion)
        if coherence < 0.45:
            findings.append(
                Finding(
                    "META_CHANNEL_CONFLICT",
                    "warn",
                    f"Physical forensics and the statistical model tell different stories "
                    f"(coherence {coherence:.2f}). When that happens the honest answer is usually "
                    f"'not enough evidence', not 'average the two numbers'.",
                    weight=0.0,
                    evidence={"coherence": round(coherence, 3), "physics_suspicion": round(phys.suspicion, 3),
                              "ml_suspicion": round(ml_suspicion, 3)},
                )
            )

        # ── Check 3: out-of-distribution honesty ───────────────────────────
        if snapshot.ood and snapshot.ood.get("out_of_distribution"):
            findings.append(
                Finding(
                    "META_OOD",
                    "warn",
                    "This panel sits outside the manifold the model was trained on. Every "
                    "statistical number in this report is weaker than usual; the physical checks "
                    "carry the verdict.",
                    weight=0.0,
                )
            )

        # ── Check 4: prior shift honesty ───────────────────────────────────
        findings.append(
            Finding(
                "META_PRIOR",
                "info",
                f"Regional base rate for {region.name} ({region.counterfeit_prevalence*100:.0f}% of "
                f"panels in circulation reported counterfeit) has been folded into the posterior "
                f"at half strength. That is why an ambiguous reading leans slightly toward caution.",
                weight=0.0,
            )
        )

        # ── Check 5: is the margin decisive? ───────────────────────────────
        state = "VERDICT"
        if top - second < 0.15 or top < 0.5:
            state = "INCONCLUSIVE"
            findings.append(
                Finding(
                    "META_MARGIN",
                    "warn",
                    f"Top two hypotheses are within {abs(top-second)*100:.0f} points "
                    f"({CLASS_ORDER[ordered[0]]} {top*100:.0f}% vs {CLASS_ORDER[ordered[1]]} "
                    f"{second*100:.0f}%). That is not a verdict, that is a coin landing on its "
                    f"edge — measure more, then re-run.",
                    weight=0.0,
                )
            )
        elif uncertainty.agreement < 0.5:
            state = "PROBABLE"
            findings.append(
                Finding(
                    "META_AGREEMENT_LOW",
                    "warn",
                    f"Council agreement is only {uncertainty.agreement*100:.0f}%. The verdict is "
                    f"reported as probable, never as certain.",
                    weight=0.0,
                )
            )
        elif top < 0.7:
            state = "PROBABLE"

        # ── Check 6: does the verdict survive its own counter-argument? ────
        if verdict_class in (0, 1) and phys.fill_factor and phys.fill_factor < physics.FF_HEALTHY:
            state = "PROBABLE" if state == "VERDICT" else state
            findings.append(
                Finding(
                    "META_SELF_DOUBT",
                    "warn",
                    f"The council called this genuine while the fill factor ({phys.fill_factor:.3f}) "
                    f"sits below the {physics.FF_HEALTHY} healthy line. I have downgraded the "
                    f"verdict to probable: a single clean afternoon reading does not outweigh a "
                    f"weak I-V curve.",
                    weight=0.0,
                )
            )

        # ── Check 7: how much of this is assumption, not evidence? ─────────
        if len(estimated) >= 4:
            if state == "VERDICT":
                state = "PROBABLE"
            findings.append(
                Finding(
                    "META_ASSUMPTIONS",
                    "warn",
                    f"{len(estimated)} of 8 electrical parameters were estimated from datasheet "
                    f"relations rather than measured ({', '.join(sorted(estimated))}). Estimated "
                    f"inputs propagate into the verdict, so its confidence is capped.",
                    weight=0.0,
                )
            )

        if state == "INCONCLUSIVE" and top >= 0.4:
            verdict_class = int(ordered[0])

        return verdict_class, state, veto_reason, findings

    def _calibrated_confidence(
        self,
        probs: np.ndarray,
        verdict_class: int,
        state: str,
        uncertainty: UncertaintyReport,
        snapshot: agents.MLSnapshot,
        veto_reason: Optional[str],
    ) -> float:
        raw = float(probs[verdict_class])
        conf = raw
        conf *= 0.55 + 0.45 * float(max(0.0, min(1.0, uncertainty.agreement)))
        conf *= 0.7 + 0.3 * uncertainty.evidence_breadth
        if snapshot.ood and snapshot.ood.get("out_of_distribution"):
            conf *= 0.8
        if state == "MEASUREMENT_ERROR":
            return 0.0
        if state == "INCONCLUSIVE":
            conf = min(conf, 0.5) * 0.75
        elif state == "PROBABLE":
            conf = min(conf, 0.8)
        if veto_reason:
            conf = max(conf, 0.95)
        return float(max(0.0, min(0.99, conf)))

    # ── planner ────────────────────────────────────────────────────────────
    def _plan_next_tests(
        self,
        inputs: PanelInputs,
        resolved: dict,
        phys: PhysicsResult,
        probs: np.ndarray,
        prior: np.ndarray,
        max_tests: int = 3,
    ) -> List[NextTest]:
        """Expected information gain of measuring each missing parameter.

        For every candidate parameter we ask: if I measured this and it came
        back at the class-conditional mean of each hypothesis (weighted by the
        current posterior), how much entropy would the council lose? That is a
        one-step-lookahead approximation to expected information gain — cheap,
        and far better than guessing.
        """
        # Only raw, physically measurable parameters can be "tested next".
        # power_output / fill_factor / temp_corrected_efficiency are *derived*
        # from the others, so proposing them would be a category error.
        candidates = [f for f in MEASURABLE_FEATURES if not inputs.is_measured(f)]
        if not candidates:
            return []

        current_entropy = _entropy_bits(probs)
        results: List[NextTest] = []

        class_stats = self.priors.get("classes") if self.priors else None

        measurable_stats = (self.priors or {}).get("measurable", {})

        for feature in candidates:
            model_idx = MODEL_FEATURE_INDEX.get(feature)
            expected_entropy = 0.0

            for cls in range(4):
                p_cls = float(probs[cls])
                if p_cls < 0.02:
                    continue
                candidate_value = None
                # Preferred: class-conditional mean of this parameter.
                stats_for_class = measurable_stats.get(str(cls))
                if stats_for_class and feature in stats_for_class:
                    candidate_value = float(stats_for_class[feature]["mean"])
                elif model_idx is not None and class_stats and str(cls) in class_stats:
                    candidate_value = float(class_stats[str(cls)]["mean"][model_idx])
                if candidate_value is None:
                    candidate_value = float(resolved.get(feature, DEFAULT_VALUES.get(feature, 0.0)))

                variant = dict(resolved)
                variant[feature] = candidate_value
                score, entropy = self._score_variant(variant, inputs, prior)
                expected_entropy += p_cls * entropy

            eig = current_entropy - expected_entropy
            if eig <= 0.01:
                continue
            results.append(
                NextTest(
                    feature=feature,
                    expected_information_gain_bits=eig,
                    expected_entropy_after=expected_entropy,
                    how=agents.FALSIFICATION_TESTS.get(
                        feature, ("Re-measure this parameter", "free", "10 min")
                    )[0],
                    why=self._why_this_test(feature, probs),
                )
            )

        results.sort(key=lambda t: -t.expected_information_gain_bits)
        return results[:max_tests]

    def _score_variant(self, variant: dict, inputs: PanelInputs, prior: np.ndarray) -> Tuple[float, float]:
        """Cheap two-channel re-evaluation used by the planner."""
        try:
            variant_phys = physics.evaluate(
                voltage=variant["voltage"],
                current=variant["current"],
                irradiance=variant["irradiance"],
                temperature=variant["temperature"],
                efficiency=variant["efficiency"],
                voc=variant["voc"],
                isc=variant["isc"],
                pmax_rated=inputs.pmax_rated,
            )
            vec = self._feature_vector(variant_phys.clamp)
            snap = agents.ml_snapshot(self.model, self.scaler, self.priors, self.calibrators, vec)
            phys_op = agents.physics_opinion(
                variant_phys, 4, pmax_rated=inputs.pmax_rated, efficiency=variant_phys.clamp["efficiency"]
            )
            ml_op = agents.ml_opinion(snap, vec)
            beliefs, weights = self._beliefs_from([ml_op, phys_op])
            post, total = fuse_beliefs(beliefs, weights, prior=prior)
            if total <= 0:
                return 0.0, _entropy_bits(np.full(4, 0.25))
            return float(np.dot(post, [CLASS_ANCHORS[c] for c in range(4)])), _entropy_bits(post)
        except Exception:
            return 0.0, _entropy_bits(np.full(4, 0.25))

    def _why_this_test(self, feature: str, probs: np.ndarray) -> str:
        reasons = {
            "voltage": "Vmp anchors the working point of the I-V curve; without it the fill factor is mostly assumed.",
            "current": "Imp drives measured power directly, so it separates 'degraded' from 'genuine but hot'.",
            "irradiance": "Every power number scales with irradiance; a wrong irradiance reading corrupts everything downstream.",
            "temperature": "Cell temperature sets how much of the shortfall is physics rather than fraud.",
            "efficiency": "Efficiency is the hardest number to fake on a label and the easiest to check against area.",
            "voc": "Voc at first light against Voc at noon exposes shunted or partly disconnected cells.",
            "isc": "Isc is nearly linear in irradiance — it is the cleanest cross-check on the light level.",
            "fill_factor": "The I-V sweep is the single measurement that separates a fake from a worn genuine panel.",
        }
        base = reasons.get(feature, "This parameter is currently assumed rather than measured.")
        leaning = "counterfeit" if probs[3] > 0.4 else ("degraded" if probs[2] > 0.4 else "genuine")
        return base + f" Right now the council leans '{leaning}', and this is the measurement most likely to change it."

    # ── decision & narrative ───────────────────────────────────────────────
    def _decision(self, state: str, verdict_class: int, econ: Optional[EconomicsResult], confidence: float) -> str:
        if state == "MEASUREMENT_ERROR":
            return (
                "No verdict. Do not act on this report. Re-measure with a second instrument — the current "
                "numbers contradict the laws of physics, which means the data is wrong, not the panel."
            )
        if state == "INCONCLUSIVE":
            return (
                "Do not pay yet. Run the recommended test below; one measurement is usually enough "
                "to move this from 'coin on its edge' to a real verdict."
            )

        money = ""
        if econ and econ.money_at_risk:
            sym = "FCFA"
            money = f" Exposure on this transaction is about {econ.money_at_risk:,.0f} {sym}."

        if verdict_class == 3:
            return (
                "DO NOT BUY. Treat this panel as misrepresented."
                + money
                + " Report the vendor and the label photo to the SolarCheck registry so the same "
                "sticker does not reach the next buyer."
            )
        if verdict_class == 2:
            return (
                "Buy only at a heavy discount, and only if the price reflects the measured power."
                + money
                + " Fair value is the printed wattage times the fraction it actually delivers."
            )
        if verdict_class == 1:
            return (
                "Safe to buy at a fair basic-panel price. Verify the serial number and keep the receipt."
                + money
            )
        return "Safe to buy. Genuine module performing as rated." + money

    def _headline(self, state: str, verdict_class: int, confidence: float, veto_reason: Optional[str]) -> str:
        if state == "MEASUREMENT_ERROR":
            return "⚠️ MEASUREMENT INVALID — no verdict given, the numbers contradict physics"
        if state == "INCONCLUSIVE":
            return "🤔 INCONCLUSIVE — one more measurement needed"
        names = {
            0: "✅ GENUINE — PREMIUM CLASS",
            1: "✅ GENUINE — BASIC CLASS",
            2: "⚠️ DEGRADED — real panel, real losses",
            3: "❌ COUNTERFEIT / MISREPRESENTED",
        }
        prefix = "PROBABLE: " if state == "PROBABLE" else ""
        return f"{prefix}{names[verdict_class]} ({confidence*100:.0f}% confidence)"

    def _narrative(
        self,
        state: str,
        verdict_class: int,
        confidence: float,
        probs: np.ndarray,
        phys: PhysicsResult,
        uncertainty: UncertaintyReport,
        econ: Optional[EconomicsResult],
        counter_case: Optional[agents.CounterCase],
        meta_findings: List[Finding],
        region: Region,
        estimated: set,
        inputs: PanelInputs,
    ) -> str:
        ordered = np.argsort(-probs)
        sentences: List[str] = []

        if state == "MEASUREMENT_ERROR":
            sentences.append(
                "I am not going to give you a verdict on this reading, because the reading "
                "contradicts physics — and a verdict built on impossible numbers would be a lie "
                "with a confident voice."
            )
            sentences.append(meta_findings[0].message if meta_findings else "")
            return " ".join(s for s in sentences if s)

        sentences.append(
            f"Working through {phys.checks_run} physical law checks, a calibrated forest, "
            f"label forensics and economics, the balance of evidence lands on "
            f"{CLASS_ORDER[verdict_class].replace('_', ' ').lower()} at {confidence*100:.0f}% "
            f"effective confidence."
        )
        if 0.05 <= probs[3] <= 0.95:
            sentences.append(
                f"The counterfeit hypothesis still holds {probs[3]*100:.0f}% of the belief mass. "
                f"That is a judgement about probability, not certainty — no single bench reading "
                f"can be certainty."
            )

        ff = phys.fill_factor
        if ff:
            tone = (
                "healthy — consistent with genuine silicon"
                if ff >= physics.FF_HEALTHY
                else ("marginal, the signature of age or heat" if ff >= physics.FF_SUSPICIOUS_LOW
                      else "below the counterfeiting threshold, which is the strongest electrical tell")
            )
            sentences.append(
                f"Fill factor came out at {ff:.3f}, {tone}; measured power {phys.power_output:.1f} W, "
                f"which at {phys.clamp['temperature']:.0f} °C corresponds to "
                f"{phys.temp_corrected_power:.1f} W at standard test conditions."
            )

        if phys.dataclass_match and phys.dataclass_match["confidence"] > 0.6:
            m = phys.dataclass_match
            sentences.append(
                f"The electrical fingerprint matches a {m['brand']} {m['model']} "
                f"({m['confidence']*100:.0f}% confidence), so if this is a fake, it is a fake wearing "
                f"a real panel's numbers."
            )

        sentences.append(uncertainty.narrative)

        if counter_case and counter_case.arguments:
            strength = counter_case.probability
            if strength < 0.05:
                sentences.append(
                    f"I tried to argue against this verdict and could not build a case worth more "
                    f"than {'a fraction of a percent of' if strength < 0.005 else f'{strength*100:.1f}% of'} "
                    f"the belief. Strongest surviving objection: "
                    f"{counter_case.arguments[0]['claim']}"
                )
            else:
                sentences.append(
                    f"I ran the case against myself. {counter_case.summary} "
                    f"The strongest objection: {counter_case.arguments[0]['claim']}"
                )

        if econ and econ.verdict_line:
            if verdict_class == 3:
                # Quoting a payback period for a panel we believe is fake would be
                # an invitation to buy it.
                sentences.append(
                    "For completeness on the economics: " + econ.verdict_line.split("(")[0].strip()
                    + " — but that assumes the panel is what the label claims, which is exactly "
                      "what this report disputes."
                )
            else:
                sentences.append(econ.verdict_line)

        if estimated:
            sentences.append(
                f"Be aware that {len(estimated)} parameter(s) were estimated rather than measured "
                f"({', '.join(sorted(estimated))}); measured values always outrank my estimates."
            )
        sentences.append(
            f"Local context: {region.name} gets about {region.peak_sun_hours:.1f} usable peak-sun-hours "
            f"per day and reports roughly {region.counterfeit_prevalence*100:.0f}% counterfeit prevalence."
        )
        return " ".join(sentences)
