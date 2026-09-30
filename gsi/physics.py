"""
Physics law-enforcement layer for GSI mode.

This module is the *ground truth* channel. Everything here is either
(a) a hard physical constraint from semiconductor/electrical theory, or
(b) an empirically established engineering envelope with the source named.

Rules are deliberately two-tier:

  HARD   — cannot be true of a working silicon PV module under any sunlight.
           These are allowed to veto a statistical verdict.
  SOFT   — unusual but physically possible (e.g. a panel being tested at an
           odd irradiance). These only shift belief.

Vocabulary used across the module:

  Vmp, Imp  voltage / current at maximum power point
  Voc, Isc  open-circuit voltage / short-circuit current
  FF        fill factor = Pmp / (Voc * Isc)
  NOCT      nominal operating cell temperature (~42-45 degC for modern modules)
  gamma     temperature coefficient of Pmax, per degC (typically -0.29..-0.45)

Key references encoded as constants below:
  * Shockley-Queisser detailed-balance limit for a single-junction silicon
    cell: ~33.7% theoretical, ~26-27% reached by commercial modules in 2025.
  * Fill factor: crystalline silicon modules span ~0.70-0.85; anything above
    ~0.88 is not physically attainable with a real single-junction module.
  * Temperature coefficient of Pmax for silicon: -0.29 %/degC to -0.45 %/degC.
  * IEC 61215 / IEC 61730 nameplate tolerances; typical flash-test spread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# ── Constants ────────────────────────────────────────────────────────────────
STC_IRRADIANCE = 1000.0  # W/m^2
STC_TEMPERATURE = 25.0  # degC
NOCT_TYPICAL = 45.0  # degC

SQ_LIMIT_EFFICIENCY = 33.7  # % single-junction detailed-balance limit
COMMERCIAL_MAX_EFFICIENCY = 26.5  # % best shipping silicon modules, 2025
COMMERCIAL_MIN_EFFICIENCY = 4.0  # % below this it is not a modern PV module

FF_HARD_MAX = 0.88  # not attainable by single-junction silicon
FF_HARD_MIN = 0.10  # below this there is no meaningful power extraction
FF_SUSPICIOUS_LOW = 0.55  # SolarCheck counterfeiting threshold (Cameroon field data)
FF_HEALTHY = 0.75

PMAX_TEMP_COEFF_RANGE = (0.29, 0.45)  # %/degC magnitude, silicon
VOC_TEMP_COEFF_RANGE = (0.20, 0.40)
ISC_TEMP_COEFF_RANGE = (-0.02, 0.10)

# Datasheet-referenced panel archetypes seen in the Cameroonian market.
PANEL_LIBRARY = [
    {
        "brand": "Bluesun",
        "model": "BSM100M-36",
        "pmax": 100.0,
        "vmp": 22.8,
        "imp": 4.39,
        "voc": 27.36,
        "isc": 4.87,
        "efficiency": 19.4,
        "cells": 36,
        "technology": "poly PERC",
        "typical_price_xaf": 35000.0,  # honest retail band, ~350 XAF/W
    },
    {
        "brand": "Suntech",
        "model": "STP280-24/Vd",
        "pmax": 280.0,
        "vmp": 31.6,
        "imp": 8.87,
        "voc": 38.8,
        "isc": 9.35,
        "efficiency": 14.4,
        "cells": 60,
        "technology": "poly",
        "typical_price_xaf": 84000.0,  # ~300 XAF/W
    },
    {
        "brand": "Canadian Solar",
        "model": "CS3W-450MS",
        "pmax": 450.0,
        "vmp": 41.1,
        "imp": 10.96,
        "voc": 49.1,
        "isc": 11.60,
        "efficiency": 20.4,
        "cells": 144,
        "technology": "mono PERC half-cut",
        "typical_price_xaf": 110000.0,  # ~245 XAF/W, imported premium
    },
    {
        "brand": "CSI Solar",
        "model": "CS6L-455MS",
        "pmax": 455.0,
        "vmp": 41.5,
        "imp": 10.97,
        "voc": 49.7,
        "isc": 11.63,
        "efficiency": 20.6,
        "cells": 144,
        "technology": "mono PERC half-cut",
        "typical_price_xaf": 115000.0,
    },
    {
        "brand": "Jinko",
        "model": "JKM550M-72HL4-V",
        "pmax": 550.0,
        "vmp": 41.4,
        "imp": 13.29,
        "voc": 49.62,
        "isc": 14.03,
        "efficiency": 21.3,
        "cells": 144,
        "technology": "mono PERC half-cut",
        "typical_price_xaf": 132000.0,  # ~240 XAF/W
    },
]

# Cell count -> Voc sanity band at STC. Each silicon cell contributes roughly
# 0.62-0.72 V of open-circuit voltage at 25 degC.
VOC_PER_CELL_RANGE = (0.55, 0.75)


@dataclass
class Finding:
    """One auditable statement produced by a specialist agent.

    Every finding carries its own weight so the metacognitive layer can
    explain *why* a verdict moved, and the UI can show the reasoning chain.
    """

    code: str
    severity: str  # info | good | warn | critical
    message: str
    weight: float = 0.0  # evidence strength toward suspicion (+) or health (-)
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "weight": round(float(self.weight), 4),
            "evidence": self.evidence,
        }


@dataclass
class PhysicsResult:
    findings: List[Finding] = field(default_factory=list)
    hard_violations: List[str] = field(default_factory=list)
    fill_factor: float = 0.0
    power_output: float = 0.0
    temp_corrected_power: float = 0.0
    temp_corrected_efficiency: float = 0.0
    irradiance_corrected_power: float = 0.0
    performance_ratio: Optional[float] = None
    dataclass_match: Optional[dict] = None
    irradiance_plausibility: float = 1.0  # 0..1 confidence that irradiance is real sun
    clamp: dict = field(default_factory=dict)  # sanitised feature values
    suspicion: float = 0.0  # 0..1 aggregated physics suspicion
    checks_run: int = 0  # how many law/envelope tests actually executed


def ffill(value, default=0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    return v if v == v and abs(v) != float("inf") else float(default)


def estimate_normalised_conditions(irradiance: float, temperature: float) -> tuple:
    """Return (G_ratio, dT) relative to STC, with safe flooring."""
    g_ratio = max(ffill(irradiance, STC_IRRADIANCE), 1.0) / STC_IRRADIANCE
    dT = ffill(temperature, STC_TEMPERATURE) - STC_TEMPERATURE
    return g_ratio, dT


def fuse_pressures(irradiance: float, temperature: float) -> float:
    """First-order channel-intoxication score for the two *measured
    environment* variables.

    Irradiance and temperature in this system come from a hardware probe the
    operator may simply have entered wrongly; they are not properties of the
    panel. When they look pathological we do not silently clip them, we
    *fuse* them from the cluster of plausible channels (their training-data
    median) and tell the operator we did. This keeps one bad sensor reading
    from producing a counterfeit verdict against an honest vendor.
    """
    pressure = 0.0
    g = ffill(irradiance, STC_IRRADIANCE)
    t = ffill(temperature, STC_TEMPERATURE)
    if g <= 50.0 or g > 1400.0:
        pressure += 0.6
    if t < -10.0 or t > 75.0:
        pressure += 0.6
    return min(pressure, 1.0)


def _match_datasheet(power_output: float, vmp: float, imp: float, voc: float, isc: float):
    """Nearest-nameplate identification by electrical fingerprint."""
    best, best_err = None, None
    for panel in PANEL_LIBRARY:
        err = 0.0
        weights = 0.0
        for obs, key, tol in (
            (vmp, "vmp", 0.25),
            (imp, "imp", 0.25),
            (voc, "voc", 0.15),
            (isc, "isc", 0.15),
            (power_output, "pmax", 0.30),
        ):
            if obs and obs > 0:
                err += (abs(obs - panel[key]) / panel[key]) * tol
                weights += tol
        if weights == 0:
            continue
        err /= weights
        # Panel power classes are far apart, so require a close match to claim one.
        if err <= 0.35 and (best_err is None or err < best_err):
            best, best_err = dict(panel), err

    if best is None:
        return None
    best["match_error"] = round(float(best_err), 4)
    best["confidence"] = round(float(max(0.0, 1.0 - best_err / 0.35)), 3)
    return best


def evaluate(
    *,
    voltage,
    current,
    irradiance,
    temperature,
    efficiency,
    voc,
    isc,
    pmax_rated=None,
) -> PhysicsResult:
    """Run every physical law and engineering envelope check we know.

    Returns a PhysicsResult whose `clamp` dict holds the values the rest of
    the pipeline should use after fusion.
    """
    res = PhysicsResult()

    vmp = ffill(voltage)
    imp = ffill(current)
    g = ffill(irradiance, STC_IRRADIANCE)
    t = ffill(temperature, STC_TEMPERATURE)
    eff = ffill(efficiency)
    voc = ffill(voc)
    isc = ffill(isc)

    # ── 0. Environment fusion ───────────────────────────────────────────────
    pressure = fuse_pressures(g, t)
    if pressure > 0.3:
        res.findings.append(
            Finding(
                "ENV_FUSED",
                "warn",
                f"Environment reading is outside anything physically usable "
                f"(irradiance {g:.0f} W/m², temperature {t:.1f} °C). I am fusing it "
                f"to standard test conditions ({STC_IRRADIANCE:.0f} W/m² / "
                f"{STC_TEMPERATURE:.0f} °C) instead of trusting it, and the verdict "
                f"below is adjusted accordingly.",
                weight=0.0,
                evidence={"irradiance": g, "temperature": t, "fusion_pressure": pressure},
            )
        )
        if g <= 50.0 or g > 1400.0:
            g = STC_IRRADIANCE
        if t < -10.0 or t > 75.0:
            t = STC_TEMPERATURE

    res.checks_run += 1

    # ── 1. Sign / magnitude sanity ──────────────────────────────────────────
    res.checks_run += 1
    for name, value in (("Vmp", vmp), ("Imp", imp), ("Voc", voc), ("Isc", isc)):
        if value < 0:
            res.hard_violations.append(f"{name} is negative ({value:.3f})")
            res.findings.append(
                Finding(
                    "PHYS_NEGATIVE",
                    "critical",
                    f"{name} = {value:.3f} is negative. A PV module cannot source "
                    f"negative power. The measurement rig or the reading is wrong.",
                    weight=0.9,
                    evidence={"parameter": name, "value": value},
                )
            )

    # ── 2. Voc / Isc / Vmp ordering ─────────────────────────────────────────
    res.checks_run += 1
    tol = 0.02
    if voc > 0 and vmp > voc * (1 + tol):
        res.hard_violations.append(f"Vmp ({vmp:.2f} V) exceeds Voc ({voc:.2f} V)")
        res.findings.append(
            Finding(
                "PHYS_VMP_GT_VOC",
                "critical",
                f"Vmp ({vmp:.2f} V) is above Voc ({voc:.2f} V). Maximum-power "
                f"voltage is always strictly below open-circuit voltage for a "
                f"real module — this reading is internally impossible.",
                weight=0.85,
                evidence={"vmp": vmp, "voc": voc},
            )
        )
    elif voc > 0:
        ratio = vmp / voc
        if not (0.65 <= ratio <= 0.95):
            res.findings.append(
                Finding(
                    "PHYS_VMP_VOC_RATIO",
                    "warn",
                    f"Vmp/Voc = {ratio:.2f} is outside the 0.65-0.95 band seen on "
                    f"every crystalline-silicon datasheet. Either the label/measurement "
                    f"is off or the cells are badly mismatched.",
                    weight=0.25,
                    evidence={"ratio": round(ratio, 4)},
                )
            )

    if isc > 0 and imp > isc * (1 + tol):
        res.hard_violations.append(f"Imp ({imp:.2f} A) exceeds Isc ({isc:.2f} A)")
        res.findings.append(
            Finding(
                "PHYS_IMP_GT_ISC",
                "critical",
                f"Imp ({imp:.2f} A) is above Isc ({isc:.2f} A). Current at the "
                f"maximum power point cannot exceed short-circuit current — a "
                f"signature of a fabricated or mis-transcribed label.",
                weight=0.85,
                evidence={"imp": imp, "isc": isc},
            )
        )
    elif isc > 0:
        ratio = imp / isc
        if not (0.70 <= ratio <= 0.99):
            res.findings.append(
                Finding(
                    "PHYS_IMP_ISC_RATIO",
                    "warn",
                    f"Imp/Isc = {ratio:.2f} is unusual (datasheets sit at 0.85-0.97).",
                    weight=0.2,
                    evidence={"ratio": round(ratio, 4)},
                )
            )

    # ── 3. Power consistency ────────────────────────────────────────────────
    res.checks_run += 1
    power_output = vmp * imp
    res.power_output = power_output

    if voc > 0 and isc > 0:
        ff = power_output / (voc * isc)
    else:
        ff = 0.0
    res.fill_factor = ff

    if abs(ff) > FF_HARD_MAX:
        # Only a hard violation when all inputs are positive (a negative
        # product means the sign check above already caught it).
        if vmp > 0 and imp > 0 and voc > 0 and isc > 0:
            res.hard_violations.append(
                f"Fill factor {ff:.3f} exceeds the physical ceiling {FF_HARD_MAX}"
            )
        res.findings.append(
            Finding(
                "PHYS_FF_IMPOSSIBLE",
                "critical",
                f"Fill factor {ff:.3f} is beyond the {FF_HARD_MAX} ceiling for "
                f"single-junction silicon (best commercial modules reach ~0.84). "
                f"A label printing this is fabricated, or the measurement is "
                f"self-inconsistent.",
                weight=0.8,
                evidence={"fill_factor": round(ff, 4)},
            )
        )
    elif ff < FF_HARD_MIN:
        res.findings.append(
            Finding(
                "PHYS_FF_ABSURD",
                "critical",
                f"Fill factor {ff:.3f} means almost no power is being extracted. "
                f"This is a dead or non-functional module, not a working panel.",
                weight=0.7,
                evidence={"fill_factor": round(ff, 4)},
            )
        )
    elif ff < FF_SUSPICIOUS_LOW:
        res.findings.append(
            Finding(
                "PHYS_FF_LOW",
                "warn",
                f"Fill factor {ff:.3f} is below the {FF_SUSPICIOUS_LOW} "
                f"counterfeiting threshold used by SolarCheck field data. Low FF is "
                f"the single strongest electrical signal of a fake or badly degraded "
                f"module, because it means series resistance is high or the cells are "
                f"not what the label claims.",
                weight=0.55,
                evidence={"fill_factor": round(ff, 4)},
            )
        )
    elif ff >= FF_HEALTHY:
        res.findings.append(
            Finding(
                "PHYS_FF_HEALTHY",
                "good",
                f"Fill factor {ff:.3f} is in the healthy range for a genuine module.",
                weight=-0.45,
                evidence={"fill_factor": round(ff, 4)},
            )
        )
    else:
        res.findings.append(
            Finding(
                "PHYS_FF_MARGINAL",
                "warn",
                f"Fill factor {ff:.3f} is marginal — between degraded and healthy. "
                f"Consistent with age-related series-resistance rise.",
                weight=0.15,
                evidence={"fill_factor": round(ff, 4)},
            )
        )

    # ── 4. Temperature coefficients ─────────────────────────────────────────
    res.checks_run += 1
    dT = t - STC_TEMPERATURE
    gamma = 0.35  # %/degC, datasheet-typical for the panels in PANEL_LIBRARY
    temp_corrected_power = power_output / (1 + (-gamma / 100.0) * dT)
    res.temp_corrected_power = temp_corrected_power
    res.temp_corrected_efficiency = eff / (1 + (-gamma / 100.0) * dT) if eff else 0.0

    if dT > 30 and eff > 0:
        res.findings.append(
            Finding(
                "PHYS_HOT_DAY",
                "info",
                f"Cells are {dT:.0f} °C above STC. Real modules lose "
                f"{PMAX_TEMP_COEFF_RANGE[0]:.2f}-{PMAX_TEMP_COEFF_RANGE[1]:.2f} %/°C, so "
                f"a {power_output:.0f} W reading today corresponds to roughly "
                f"{temp_corrected_power:.0f} W at 25 °C — do not judge the label on the "
                f"hot reading alone.",
                weight=-0.1,
                evidence={"dT": round(dT, 2), "corrected_power": round(temp_corrected_power, 2)},
            )
        )

    # ── 5. Efficiency envelope ──────────────────────────────────────────────
    res.checks_run += 1
    if eff > SQ_LIMIT_EFFICIENCY:
        res.hard_violations.append(
            f"Efficiency {eff:.1f}% exceeds the Shockley-Queisser limit"
        )
        res.findings.append(
            Finding(
                "PHYS_EFF_IMPOSSIBLE",
                "critical",
                f"Claimed efficiency {eff:.1f}% is above the {SQ_LIMIT_EFFICIENCY}% "
                f"detailed-balance limit for any single-junction silicon cell. No "
                f"laboratory on Earth has built this. The number is fabricated.",
                weight=0.95,
                evidence={"efficiency": eff},
            )
        )
    elif eff > COMMERCIAL_MAX_EFFICIENCY:
        res.findings.append(
            Finding(
                "PHYS_EFF_TOO_HIGH",
                "warn",
                f"Efficiency {eff:.1f}% exceeds the {COMMERCIAL_MAX_EFFICIENCY}% best "
                f"shipping commercial modules. Possible if this is a laboratory cell, "
                f"otherwise the number is inflated.",
                weight=0.45,
                evidence={"efficiency": eff},
            )
        )
    elif 0 < eff < COMMERCIAL_MIN_EFFICIENCY:
        res.findings.append(
            Finding(
                "PHYS_EFF_TOO_LOW",
                "warn",
                f"Efficiency {eff:.1f}% is below {COMMERCIAL_MIN_EFFICIENCY}%, lower "
                f"than any modern PV module. Consistent with a non-PV surface being "
                f"sold as a panel.",
                weight=0.4,
                evidence={"efficiency": eff},
            )
        )

    # ── 6. Irradiance / current coupling ────────────────────────────────────
    res.checks_run += 1
    g_ratio, _ = estimate_normalised_conditions(g, t)
    if isc > 0 and 0.05 <= g_ratio <= 1.5:
        expected_isc = isc * g_ratio
        res.clamp["expected_isc"] = round(expected_isc, 4)

    # Performance ratio against the nameplate, normalised for BOTH irradiance
    # and cell temperature. This is the fairest possible test of "does this panel
    # deliver what its sticker claims", and therefore the strongest single piece
    # of evidence either way — so it is checked explicitly rather than left
    # implicit in the fill factor.
    if pmax_rated and power_output > 0 and g_ratio > 0.05:
        pr = (temp_corrected_power / g_ratio) / float(pmax_rated)
        res.performance_ratio = pr
        if pr > 1.25:
            res.findings.append(
                Finding(
                    "PHYS_PR_OVER_ONE",
                    "warn",
                    f"Corrected to standard test conditions the module delivers {pr*100:.0f}% of "
                    f"its {pmax_rated:g} W nameplate — above 100% by more than flash-test "
                    f"tolerance. Either the nameplate is understated (rare) or the "
                    f"irradiance/current reading is too high.",
                    weight=0.3,
                    evidence={"performance_ratio": round(pr, 3)},
                )
            )
        elif pr >= 0.92:
            res.findings.append(
                Finding(
                    "PHYS_PR_MATCHES_NAMEPLATE",
                    "good",
                    f"Temperature- and irradiance-corrected output is {pr*100:.0f}% of the "
                    f"{pmax_rated:g} W nameplate. A module that hits its own printed rating once "
                    f"heat and light are accounted for is behaving exactly like an honest panel.",
                    weight=-0.5,
                    evidence={"performance_ratio": round(pr, 3)},
                )
            )
        elif pr >= 0.80:
            res.findings.append(
                Finding(
                    "PHYS_PR_SLIGHT_SHORTFALL",
                    "info",
                    f"Corrected output is {pr*100:.0f}% of the {pmax_rated:g} W nameplate — a mild "
                    f"shortfall, consistent with a few years of normal degradation (0.5-0.8%/year).",
                    weight=0.1,
                    evidence={"performance_ratio": round(pr, 3)},
                )
            )
        else:
            res.findings.append(
                Finding(
                    "PHYS_PR_SHORTFALL",
                    "warn",
                    f"Corrected output is only {pr*100:.0f}% of the {pmax_rated:g} W nameplate, "
                    f"after allowing for both heat and light. That is far beyond normal ageing and "
                    f"means the panel does not do what its own sticker claims.",
                    weight=0.4,
                    evidence={"performance_ratio": round(pr, 3)},
                )
            )

    # ── 7. Datasheet identification ─────────────────────────────────────────
    res.checks_run += 1
    res.dataclass_match = _match_datasheet(power_output, vmp, imp, voc, isc)

    # ── 7b. Clean bill of health ────────────────────────────────────────────
    # Absence of findings is evidence too. Without this, a panel that passes
    # every check would be scored as though nothing were known about it.
    severe = [f for f in res.findings if f.severity in ("warn", "critical")]
    if not severe and res.checks_run >= 5:
        res.findings.append(
            Finding(
                "PHYS_ALL_CLEAR",
                "good",
                f"No physical or envelope violation across {res.checks_run} checks: voltage "
                f"ordering, current ordering, fill factor, temperature response and efficiency "
                f"all sit inside the envelope of genuine silicon.",
                weight=-0.55,
            )
        )

    # ── 8. Aggregate physics suspicion ──────────────────────────────────────
    # Signed weights: positive raises suspicion, negative lowers it.
    raw = sum(f.weight for f in res.findings)
    res.suspicion = max(0.0, min(1.0, 0.5 + raw * 0.35))

    res.clamp.update(
        {
            "voltage": vmp,
            "current": imp,
            "irradiance": g,
            "temperature": t,
            "efficiency": eff,
            "voc": voc,
            "isc": isc,
            "power_output": power_output,
            "fill_factor": ff,
            "temp_corrected_efficiency": res.temp_corrected_efficiency,
            "temp_corrected_power": temp_corrected_power,
            "environment_fused": pressure > 0.3,
        }
    )
    return res


def coherence_score(physics: PhysicsResult, ml_suspicion: float) -> float:
    """How well does the physics story agree with the statistical story?

    Returns 0..1 where 1 means "physics and ML tell the same story".
    A low value is the most interesting meta-signal in the system: it means
    the situation is novel and neither channel should be trusted blindly.
    """
    return float(max(0.0, 1.0 - abs(physics.suspicion - ml_suspicion)))
