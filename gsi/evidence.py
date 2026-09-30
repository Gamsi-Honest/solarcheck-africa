"""
Evidence objects shared by every GSI agent.

An AgentOpinion is the only currency the council accepts. It is deliberately
richer than a label:

  * a signed verdict score in [-1, +1] (health -> counterfeit),
  * a self-reported confidence in [0, 1],
  * a reliability weight the orchestrator may adapt,
  * the audit trail of findings that produced it.

Because every agent speaks the same language, the fusion step can be a
principled operation (weighted log-opinion pooling) instead of ad-hoc
`if` trees, and the critic can explain *which* agent moved the verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .physics import Finding  # re-exported for convenience

__all__ = ["Finding", "AgentOpinion", "normalise_score", "score_to_label"]

LABELS = {
    0: "HEALTHY_PREMIUM",
    1: "HEALTHY_BASIC",
    2: "DEGRADED",
    3: "COUNTERFEIT",
}
LABEL_TO_ID = {v: k for k, v in LABELS.items()}

# Canonical position of each class on the health -> counterfeit axis.
CLASS_ANCHORS = {0: -0.85, 1: -0.35, 2: 0.45, 3: 1.0}


@dataclass
class AgentOpinion:
    agent: str
    verdict: Optional[int]  # class id, or None if the agent abstains
    score: float  # -1 (certainly genuine) .. +1 (certainly counterfeit)
    confidence: float  # 0..1 self-assessed
    reliability: float  # 0..1 orchestrator-assigned trust
    findings: List[Finding] = field(default_factory=list)
    rationale: str = ""
    aborted: bool = False  # True when the agent refused to opine (guardrail)

    @property
    def label(self) -> str:
        return LABELS.get(self.verdict, "UNKNOWN") if self.verdict is not None else "ABSTAIN"

    @property
    def weight(self) -> float:
        """Effective voting weight: trust x confidence, with an abstention floor."""
        if self.verdict is None or self.confidence <= 0:
            return 0.0
        return max(0.0, min(1.0, self.reliability)) * max(0.0, min(1.0, self.confidence))

    @property
    def criticals(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "critical"]

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "verdict": self.verdict,
            "label": self.label,
            "score": round(float(self.score), 4),
            "confidence": round(float(self.confidence), 4),
            "reliability": round(float(self.reliability), 4),
            "weight": round(float(self.weight), 4),
            "rationale": self.rationale,
            "aborted": self.aborted,
            "findings": [f.to_dict() for f in self.findings],
        }

    @staticmethod
    def abstain(agent: str, reason: str, reliability: float = 0.3) -> "AgentOpinion":
        return AgentOpinion(
            agent=agent,
            verdict=None,
            score=0.0,
            confidence=0.0,
            reliability=reliability,
            rationale=reason,
            aborted=True,
        )


def normalise_score(score: float) -> float:
    """Clip a raw score into the canonical opinion interval."""
    try:
        return float(max(-1.0, min(1.0, score)))
    except (TypeError, ValueError):
        return 0.0


def score_to_class(score: float) -> int:
    """Map a fused score back onto the four SolarCheck classes."""
    if score <= -0.60:
        return 0
    if score <= 0.00:
        return 1
    if score <= 0.72:
        return 2
    return 3


def score_to_label(score: float) -> str:
    return LABELS[score_to_class(score)]


def logs_to_probs(scores: Dict[str, float], weights: Dict[str, float], beta: float = 1.7) -> Dict[str, float]:
    """Weighted log-opinion pooling over the four classes.

    Each agent contributes a preference vector over the classes, sharpened by
    `beta * weight`, and we sum in log space (so a strongly held independent
    opinion dominates, but ten agreeing weak opinions can still win). This is
    the standard logarithmic opinion pool, not a heuristic average.
    """
    total_w = sum(max(0.0, w) for w in weights.values())
    if total_w <= 0:
        return {label: 0.25 for label in LABELS.values()}

    logits = {cid: 0.0 for cid in LABELS}
    for agent, score in scores.items():
        w = max(0.0, weights.get(agent, 0.0)) / total_w
        s = normalise_score(score)
        for cid, anchor in CLASS_ANCHORS.items():
            # Gaussian compatibility between the agent's score and the class
            # anchor, evaluated in log space.
            distance = (s - anchor) ** 2
            logits[cid] += w * beta * (-distance / 0.5)

    peak = max(logits.values())
    exps = {cid: math.exp(v - peak) for cid, v in logits.items()}
    z = sum(exps.values())
    return {LABELS[cid]: exps[cid] / z for cid in LABELS}
