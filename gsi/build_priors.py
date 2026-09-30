"""
Build the feature prior file used by GSI mode.

GSI mode needs three things that a bare classifier does not carry:

  1. Class-conditional feature statistics — so the council can ask
     "how typical is this reading for a genuine panel?" instead of only
     "which label wins?".
  2. A Mahalanobis geometry of the training manifold — so the engine can
     say "this reading is unlike anything I was trained on" (out of
     distribution detection) instead of silently hallucinating confidence.
  3. Marginal distributions — so the planner can rank *which missing
     measurement* would most reduce uncertainty (expected information
     gain) before it is taken.

Run from the repository root::

    python -m gsi.build_priors

Writes gsi/data/feature_priors.json. The file is a build artifact, but it
is committed on purpose: the deployed app must not depend on the raw CSV.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

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

# Parameters a technician can put an instrument on. voc/isc are not model
# features but the information-gain planner needs class-conditional statistics
# for them, because "measure Voc" is one of the most decisive field tests.
MEASURABLE = [
    "voltage",
    "current",
    "irradiance",
    "temperature",
    "efficiency",
    "voc",
    "isc",
]

CLASS_NAMES = {
    0: "HEALTHY_PREMIUM",
    1: "HEALTHY_BASIC",
    2: "DEGRADED",
    3: "COUNTERFEIT",
}

CSV_NAME = "solarcheck_training_data_v2.csv"
OUT_PATH = os.path.join(os.path.dirname(__file__), "data", "feature_priors.json")


def _round(value, ndigits=6):
    if isinstance(value, (list, tuple)):
        return [_round(v, ndigits) for v in value]
    if isinstance(value, np.ndarray):
        return _round(value.tolist(), ndigits)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if not np.isfinite(f):
            return None
        return round(f, ndigits)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def build(csv_path: str = CSV_NAME) -> dict:
    df = pd.read_csv(csv_path)

    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Training CSV is missing required columns: {missing}")

    X = df[FEATURES].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    # Drop physically degenerate rows that were used for training only.
    finite = np.isfinite(X).all(axis=1)
    X, y = X[finite], y[finite]

    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std_safe = np.where(std > 1e-9, std, 1.0)
    Z = (X - mean) / std_safe

    cov = np.cov(Z, rowvar=False)
    # Ridge stabilisation: 8 features, 2400 rows, covariance is well posed,
    # but counterfeits can be near-collinear, so keep the inverse sane.
    cov_reg = cov + np.eye(cov.shape[0]) * 1e-6
    cov_inv = np.linalg.pinv(cov_reg)

    maha = np.sqrt(np.einsum("ij,jk,ik->i", Z, cov_inv, Z))

    # Class-conditional statistics for the measurable parameters, used by the
    # planner to ask "if Voc came back typical of a fake, how surprised would I be?"
    measurable = {}
    for cid in sorted(np.unique(y).tolist()):
        mask = y == cid
        measurable[str(cid)] = {
            feat: {
                "mean": _round(df.loc[mask, feat].mean()),
                "std": _round(df.loc[mask, feat].std(ddof=0)),
                "p05": _round(df.loc[mask, feat].quantile(0.05)),
                "p95": _round(df.loc[mask, feat].quantile(0.95)),
            }
            for feat in MEASURABLE
            if feat in df.columns
        }

    classes = {}
    for cid in sorted(np.unique(y).tolist()):
        mask = y == cid
        sub = X[mask]
        # Class-conditional geometry: "is this reading typical *of the class I
        # just assigned*?" is a far better trust question than "is it typical of
        # all panels". A legitimate 100 W module is far from the global mean
        # (most training data is 150 W+), yet perfectly normal among 100 W units.
        sub_z = (sub - mean) / std_safe
        cov_c = np.cov(sub_z, rowvar=False) + np.eye(sub_z.shape[1]) * 1e-5
        cov_inv_c = np.linalg.pinv(cov_c)
        maha_c = np.sqrt(np.einsum("ij,jk,ik->i", sub_z, cov_inv_c, sub_z))
        classes[str(cid)] = {
            "cov_inv": _round(cov_inv_c),
            "maha_quantiles": {
                "p50": _round(np.percentile(maha_c, 50)),
                "p90": _round(np.percentile(maha_c, 90)),
                "p99": _round(np.percentile(maha_c, 99)),
                "max": _round(maha_c.max()),
            },
            "name": CLASS_NAMES.get(cid, f"CLASS_{cid}"),
            "prior": _round(mask.mean(), 6),
            "n": int(mask.sum()),
            "mean": _round(sub.mean(axis=0)),
            "std": _round(sub.std(axis=0, ddof=0)),
            "p05": _round(np.percentile(sub, 5, axis=0)),
            "p50": _round(np.percentile(sub, 50, axis=0)),
            "p95": _round(np.percentile(sub, 95, axis=0)),
            "min": _round(sub.min(axis=0)),
            "max": _round(sub.max(axis=0)),
        }

    payload = {
        "schema": "solarcheck-gsi-priors/1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_csv": os.path.basename(csv_path),
        "n_samples": int(len(X)),
        "features": FEATURES,
        "class_names": CLASS_NAMES,
        "global": {
            "mean": _round(mean),
            "std": _round(std_safe),
            "cov_inv": _round(cov_inv),
            "maha_quantiles": {
                "p50": _round(np.percentile(maha, 50)),
                "p75": _round(np.percentile(maha, 75)),
                "p90": _round(np.percentile(maha, 90)),
                "p99": _round(np.percentile(maha, 99)),
                "max": _round(maha.max()),
            },
        },
        "classes": classes,
        "measurable": measurable,
    }
    return payload


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(here, CSV_NAME)
    if not os.path.exists(csv_path):
        csv_path = CSV_NAME
    payload = build(csv_path)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(
        f"Wrote {OUT_PATH} — {payload['n_samples']} rows, "
        f"maha p99={payload['global']['maha_quantiles']['p99']:.2f}"
    )


if __name__ == "__main__":
    main()
