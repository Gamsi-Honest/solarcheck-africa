"""
GSI — General Superintelligence Mode for SolarCheck Africa.

GSI mode replaces the single forward-pass classifier with a *cognitive
architecture*: a council of specialist agents (physics, machine perception,
forensics, economics, safety, adversarial red-team) whose independent
opinions are fused with reliability weighting, then audited by a
metacognitive critic that is allowed to downgrade, veto or reject the
consensus.

What "GSI" means here, precisely — no mysticism:

  1. LAW ENFORCEMENT   Physics is treated as ground truth. Physical
                       impossibilities (e.g. Imp > Isc, FF outside the
                       Shockley-Queisser envelope) veto any statistical
                       verdict, however confident the classifier is.
  2. COUNCIL           Several narrow specialists, each with an independent
                       evidence channel, instead of one monolithic model.
  3. FUSION            Log-opinion pooling with per-agent reliability
                       weights, plus explicit conflict (Jensen-Shannon)
                       measurement and aleatoric/epistemic uncertainty
                       decomposition.
  4. METACOGNITION     The system reasons about its own reasoning: audit
                       trail, self-critique, calibrated confidence, and an
                       explicit account of what would change its mind.
  5. ADVERSARIALISM    A red-team agent argues the strongest case *against*
                       the verdict and ranks the next test that would
                       resolve the disagreement best (expected information
                       gain).
  6. ACTION            The output is a decision with a consequence model
                       (money, safety, energy yield), not a bare label.

Public API::

    from gsi import GSIEngine, PanelInputs

    engine = GSIEngine(model, scaler)
    report = engine.reason(PanelInputs(voltage=22.8, current=4.39, ...))
    print(report.verdict.headline)
"""

from .orchestrator import GSIEngine, PanelInputs, GSIReport  # noqa: F401
from .evidence import AgentOpinion, Finding  # noqa: F401

__all__ = [
    "GSIEngine",
    "PanelInputs",
    "GSIReport",
    "AgentOpinion",
    "Finding",
]

__version__ = "3.0.0-gsi"
