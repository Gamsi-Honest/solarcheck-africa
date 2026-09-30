"""
Machine-perception agent: test-bench photo forensics.

This is real image analysis, not a description of image analysis. For every
photo the operator submits, the module measures:

  * focus        — variance of the Laplacian (classic blur detector).
                   A blurred photo of a spec label cannot be used to judge
                   print quality, and the system must say so instead of
                   inventing a print-quality score.
  * exposure     — mean luma plus clipped-shadow / blown-highlight fractions.
  * glare        — fraction of near-white low-saturation pixels, which is what
                   a torch or the sun does to a glossy laminate.
  * colour cast  — channel means vs grey-world expectation. Repainted or
                   reprinted laminate often carries a visible cast that a
                   grey-world estimator picks up.
  * cell grid    — dominant periodicity of the busbar/grid pattern, measured
                   with an FFT of the row/column luminance profile. Genuine
                   modules show a sharp periodicity at a physical pitch;
                   printed vinyl facsimiles usually do not.
  * duplicates   — pairwise dHash (64-bit) Hamming distance across submitted
                   photos. Near-identical photos presented as "different
                   panels" is a known resale trick.

Every measurement degrades gracefully: if an image is unusable the agent
abstains with a reason rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from .evidence import AgentOpinion, normalise_score
from .physics import Finding

AGENT_NAME = "Perception (image forensics)"

# Thresholds chosen for phone photos of spec labels taken in the field.
BLUR_ACCEPTABLE = 90.0  # variance of Laplacian
BLUR_GOOD = 220.0
GLARE_ACCEPTABLE = 0.02  # fraction of near-white pixels
GLARE_BAD = 0.10
SHADOW_ACCEPTABLE = 0.06
CAST_ACCEPTABLE = 0.055  # normalised channel divergence


@dataclass
class ImageMetrics:
    index: int
    width: int
    height: int
    mean_luma: float
    shadow_fraction: float
    highlight_fraction: float
    glare_fraction: float
    focus_variance: float
    colour_cast: float
    cast_direction: str
    grid_period_px: Optional[float]
    grid_strength: float
    usable: bool
    issues: List[str] = field(default_factory=list)

    @property
    def quality(self) -> float:
        """0..1 usability score combining focus, glare and exposure."""
        q = 1.0
        if self.focus_variance < BLUR_GOOD:
            q *= max(0.0, (self.focus_variance - BLUR_ACCEPTABLE * 0.4) / BLUR_GOOD)
        q *= max(0.0, 1.0 - min(1.0, self.glare_fraction / GLARE_BAD))
        q *= max(0.0, 1.0 - min(1.0, self.shadow_fraction / (SHADOW_ACCEPTABLE * 3)))
        return float(max(0.0, min(1.0, q)))

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "size": f"{self.width}x{self.height}",
            "mean_luma": round(self.mean_luma, 1),
            "shadow_fraction": round(self.shadow_fraction, 4),
            "glare_fraction": round(self.glare_fraction, 4),
            "focus_variance": round(self.focus_variance, 1),
            "colour_cast": round(self.colour_cast, 4),
            "cast_direction": self.cast_direction,
            "grid_period_px": None if self.grid_period_px is None else round(self.grid_period_px, 1),
            "grid_strength": round(self.grid_strength, 3),
            "quality": round(self.quality, 3),
            "usable": self.usable,
            "issues": self.issues,
        }


@dataclass
class PerceptionAssessment:
    images: List[ImageMetrics] = field(default_factory=list)
    duplicates: List[dict] = field(default_factory=list)
    median_quality: float = 0.0
    usable_count: int = 0

    def to_dict(self) -> dict:
        return {
            "images": [i.to_dict() for i in self.images],
            "duplicates": self.duplicates,
            "median_quality": round(self.median_quality, 3),
            "usable_count": self.usable_count,
        }


# ── Primitive measurements ───────────────────────────────────────────────────
def _to_gray(img: Image.Image) -> np.ndarray:
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr @ np.array([0.299, 0.587, 0.114])
    return arr


def _focus_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian — the standard blur metric."""
    if min(gray.shape) < 3:
        return 0.0
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    padded = np.pad(gray, 1, mode="edge")
    lap = np.zeros_like(gray)
    for i in range(3):
        for j in range(3):
            if kernel[i, j]:
                lap += kernel[i, j] * padded[i : i + gray.shape[0], j : j + gray.shape[1]]
    return float(lap.var())


def _glare_fraction(rgb: np.ndarray) -> float:
    luma = rgb @ np.array([0.299, 0.587, 0.114])
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    near_white = (luma > 245) & (chroma < 18)
    return float(near_white.mean())


def _colour_cast(rgb: np.ndarray) -> tuple:
    """Grey-world divergence and the direction of the cast."""
    means = rgb.reshape(-1, 3).mean(axis=0)
    if means.sum() <= 0:
        return 0.0, "neutral"
    norm = means / means.mean()
    divergence = float(np.abs(norm - 1.0).max())
    channels = ["red", "green", "blue"]
    direction = channels[int(np.argmax(norm))]
    if divergence < CAST_ACCEPTABLE:
        direction = "neutral"
    return divergence, direction


def _grid_periodicity(gray: np.ndarray) -> tuple:
    """Dominant spatial period of the cell grid, in pixels.

    The busbars of a real module produce a strong quasi-periodic luminance
    pattern. We take the row-mean profile (which cancels most noise), remove
    the DC term, window it, and look for the strongest spectral peak outside
    the very low frequencies.
    """
    profile = gray.mean(axis=1)
    profile = profile - profile.mean()
    n = profile.size
    if n < 32 or not np.any(profile):
        return None, 0.0
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(profile * window))
    freqs = np.fft.rfftfreq(n, d=1.0)
    mask = freqs > 1.0 / 64.0  # ignore trends slower than 64 px
    if not np.any(mask):
        return None, 0.0
    spectrum_masked = np.where(mask, spectrum, 0.0)
    peak = int(np.argmax(spectrum_masked))
    strength = float(spectrum_masked[peak] / (spectrum.sum() + 1e-9))
    if strength < 0.05 or freqs[peak] <= 0:
        return None, strength
    return float(1.0 / freqs[peak]), strength


def _dhash(img: Image.Image, size: int = 8) -> int:
    """64-bit difference hash — cheap, robust near-duplicate detector."""
    small = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bit)
    return bits


def _hamming(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))


# ── Agent entry point ────────────────────────────────────────────────────────
def analyse(images: Sequence[Image.Image], max_side: int = 720) -> PerceptionAssessment:
    assessment = PerceptionAssessment()
    hashes: List[int] = []

    for idx, original in enumerate(images):
        img = original
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        if max(img.size) > max_side:
            scale = max_side / float(max(img.size))
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)

        rgb = np.asarray(img.convert("RGB"), dtype=np.float64)
        gray = _to_gray(img)
        luma = rgb @ np.array([0.299, 0.587, 0.114])

        focus = _focus_variance(gray)
        glare = _glare_fraction(rgb)
        cast, direction = _colour_cast(rgb)
        period, strength = _grid_periodicity(gray)

        metrics = ImageMetrics(
            index=idx,
            width=img.width,
            height=img.height,
            mean_luma=float(luma.mean()),
            shadow_fraction=float((luma < 35).mean()),
            highlight_fraction=float((luma > 245).mean()),
            glare_fraction=glare,
            focus_variance=focus,
            colour_cast=cast,
            cast_direction=direction,
            grid_period_px=period,
            grid_strength=strength,
            usable=True,
        )

        if focus < BLUR_ACCEPTABLE:
            metrics.issues.append(
                f"Photo {idx+1} is blurred (focus score {focus:.0f}, need {BLUR_ACCEPTABLE:.0f}+). "
                f"Print-quality forensics on this image would be guesswork."
            )
        if glare > GLARE_ACCEPTABLE:
            metrics.issues.append(
                f"Photo {idx+1} has {glare*100:.0f}% blown-out pixels — glare or direct flash is "
                f"washing out part of the surface."
            )
        if metrics.shadow_fraction > SHADOW_ACCEPTABLE:
            metrics.issues.append(
                f"Photo {idx+1}: {metrics.shadow_fraction*100:.0f}% of the frame is crushed black; "
                f"part of the panel is lost in shadow."
            )
        if cast > CAST_ACCEPTABLE and direction != "neutral":
            metrics.issues.append(
                f"Photo {idx+1} has a {direction} colour cast ({cast:.3f}). Could be the light source, "
                f"or could be an edited/repainted listing photo — worth a second look in person."
            )
        if metrics.quality < 0.35:
            metrics.usable = False

        assessment.images.append(metrics)
        hashes.append(_dhash(img))

    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            distance = _hamming(hashes[i], hashes[j])
            if distance <= 6:
                assessment.duplicates.append(
                    {
                        "a": i,
                        "b": j,
                        "hamming": distance,
                        "note": (
                            "These two photos are near-identical. If they are meant to be different "
                            "panels (or different sellers), one is being reused as a template — a "
                            "classic counterfeit-listing pattern."
                        ),
                    }
                )

    qualities = [m.quality for m in assessment.images]
    if qualities:
        assessment.median_quality = float(np.median(qualities))
    assessment.usable_count = sum(1 for m in assessment.images if m.usable)
    return assessment


def opinion(assessment: PerceptionAssessment, label_check: Optional[dict] = None) -> AgentOpinion:
    """Turn image measurements into a council opinion.

    Key epistemic stance: photo forensics can *strengthen* suspicion (glare
    hiding a defect, a duplicated listing, a colour cast on a "new" label) but
    a clean photo set cannot by itself certify authenticity. So the positive
    (healthy) direction is capped, and the agent abstains when the images are
    too poor to support any claim.
    """
    findings: List[Finding] = []

    if not assessment.images:
        return AgentOpinion.abstain(
            AGENT_NAME, "No photos submitted — image forensics has nothing to analyse."
        )

    if assessment.usable_count == 0:
        for m in assessment.images:
            for issue in m.issues:
                findings.append(Finding("PERC_UNUSABLE", "warn", issue, weight=0.0))
        return AgentOpinion(
            agent=AGENT_NAME,
            verdict=None,
            score=0.0,
            confidence=0.0,
            reliability=0.2,
            findings=findings,
            rationale=(
                "Every submitted photo is too blurred, glared or underexposed to judge. "
                "I refuse to score a label I cannot see clearly — retake in even daylight, "
                "filling the frame with the sticker."
            ),
            aborted=True,
        )

    suspicion = 0.0
    evidence_mass = 0.0
    notes: List[str] = []

    median_quality = assessment.median_quality
    evidence_mass += 1.0
    if median_quality >= 0.7:
        findings.append(
            Finding(
                "PERC_QUALITY_GOOD",
                "good",
                f"Photo set is usable (median quality {median_quality:.2f}); measurements "
                f"below are trustworthy enough to reason from.",
                weight=-0.05,
            )
        )
    elif median_quality < 0.45:
        suspicion += 0.15
        findings.append(
            Finding(
                "PERC_QUALITY_POOR",
                "warn",
                f"Median image quality is {median_quality:.2f}. Forensics on these images "
                f"are weak evidence either way — treat them as a hint, not proof.",
                weight=0.15,
            )
        )

    if assessment.duplicates:
        d = assessment.duplicates[0]
        suspicion += 0.28
        evidence_mass += 1.0
        findings.append(
            Finding(
                "PERC_DUPLICATE_PHOTOS",
                "warn",
                f"Photos {d['a']+1} and {d['b']+1} are near-duplicates (Hamming {d['hamming']}/64). "
                f"Identical images presented as different panels is a documented resale trick.",
                weight=0.28,
                evidence={"pairs": assessment.duplicates},
            )
        )

    well_framed = [m for m in assessment.images if m.usable]
    grid_images = [m for m in well_framed if m.grid_period_px is not None and m.grid_strength >= 0.08]
    if grid_images:
        periods = [m.grid_period_px for m in grid_images]
        spread = (max(periods) - min(periods)) / max(1.0, float(np.mean(periods)))
        evidence_mass += 1.0
        if spread > 0.35 and len(periods) >= 2:
            suspicion += 0.18
            findings.append(
                Finding(
                    "PERC_GRID_INCONSISTENT",
                    "warn",
                    f"The cell-grid pitch differs by {spread*100:.0f}% across your photos "
                    f"({', '.join(f'{p:.0f}px' for p in periods)}). Same panel photographed "
                    f"from different distances should keep a consistent *relative* structure; "
                    f"large disagreement suggests more than one panel is in the set.",
                    weight=0.18,
                    evidence={"periods_px": [round(p, 1) for p in periods]},
                )
            )
        else:
            findings.append(
                Finding(
                    "PERC_GRID_DETECTED",
                    "good",
                    f"Regular busbar/grid periodicity detected ({np.mean(periods):.0f}px, "
                    f"strength {np.mean([m.grid_strength for m in grid_images]):.2f}) on "
                    f"{len(grid_images)} photo(s) — the surface is a real woven cell grid, "
                    f"not a flat printed sheet.",
                    weight=-0.25,
                )
            )
            suspicion -= 0.10
    else:
        evidence_mass += 0.5
        suspicion += 0.10
        findings.append(
            Finding(
                "PERC_NO_GRID",
                "warn",
                "No clear cell-grid periodicity found. Common causes: the photo shows only the "
                "back sticker (fine), heavy blur, or a smooth printed facsimile. Not proof of "
                "anything on its own — but a genuine front-of-panel photo usually shows the grid.",
                weight=0.10,
            )
        )

    strong_casts = [m for m in well_framed if m.colour_cast > CAST_ACCEPTABLE * 1.6 and m.cast_direction != "neutral"]
    if strong_casts:
        suspicion += 0.10
        evidence_mass += 0.5
        findings.append(
            Finding(
                "PERC_COLOUR_CAST",
                "warn",
                f"{len(strong_casts)} photo(s) carry a strong {'/'.join(sorted({m.cast_direction for m in strong_casts}))} "
                f"cast. Repainted glass or an edited listing photo both look like this. Confirm the "
                f"panel in person before paying.",
                weight=0.10,
            )
        )

    glare_images = [m for m in well_framed if m.glare_fraction > GLARE_ACCEPTABLE]
    if glare_images:
        findings.append(
            Finding(
                "PERC_GLARE_LIMITS",
                "info",
                f"{len(glare_images)} photo(s) have glare covering part of the surface. Whatever "
                f"defect sits under that glare was invisible to me — do not read the clean verdict "
                f"as covering the whole panel.",
                weight=0.05,
            )
        )

    if label_check:
        pq = float(label_check.get("print_quality", 5) or 5)
        fc = float(label_check.get("font_consistency", 5) or 5)
        red_flags = label_check.get("red_flags") or []
        cert = label_check.get("certification_marks_visible")
        # Only trust these subjective scores when the photo was actually usable.
        trust = 1.0 if median_quality >= 0.6 else 0.5
        evidence_mass += 1.0
        label_suspicion = ((10 - pq) / 10.0 * 0.5 + (10 - fc) / 10.0 * 0.5) * trust
        if cert is False:
            label_suspicion += 0.10 * trust
        if red_flags:
            label_suspicion += min(0.35, 0.12 * len(red_flags)) * trust
        suspicion += label_suspicion * 0.9
        findings.append(
            Finding(
                "PERC_LABEL_SCORES",
                "warn" if label_suspicion > 0.35 else "info",
                f"Sticker forensics: print quality {pq:.0f}/10, font consistency {fc:.0f}/10, "
                f"{'certification marks present' if cert else 'no certification marks visible'}"
                + (f", {len(red_flags)} red flag(s): {', '.join(map(str, red_flags[:3]))}" if red_flags else "")
                + f". Weighted by image quality ({trust:.1f}).",
                weight=label_suspicion,
            )
        )

    suspicion = float(max(0.0, min(1.0, suspicion)))
    confidence = float(max(0.15, min(0.85, evidence_mass / 4.0)))
    # Positive direction is capped: a clean photo is never proof of authenticity.
    score = normalise_score(suspicion * 2.0 - 1.0 if suspicion > 0.5 else -0.25 * (1 - suspicion * 2))

    verdict = None
    if suspicion > 0.55:
        verdict = 3
    elif suspicion > 0.30:
        verdict = 2
    elif median_quality >= 0.7 and not assessment.duplicates and grid_images:
        verdict = 2 if suspicion > 0.18 else 1

    notes.append(
        f"Analysed {len(assessment.images)} photo(s): {assessment.usable_count} usable, "
        f"median quality {median_quality:.2f}, {len(assessment.duplicates)} near-duplicate pair(s), "
        f"grid detected on {len(grid_images)}."
    )
    return AgentOpinion(
        agent=AGENT_NAME,
        verdict=verdict,
        score=score,
        confidence=confidence,
        reliability=0.75,
        findings=findings,
        rationale=" ".join(notes),
    )
