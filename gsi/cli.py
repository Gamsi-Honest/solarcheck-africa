"""
Command-line interface for the GSI engine.

The council does not need a browser. This is how the same reasoning gets used
to build a registry — running thousands of panels through the engine offline —
and how the engine's behaviour is inspected and regression-tested.

Usage::

    # Built-in scenarios, human readable
    python -m gsi.cli --demo

    # Same scenarios, rendered as a Markdown document
    python -m gsi.cli --demo --markdown docs/GSI_EXAMPLE_VERDICTS.md

    # One panel from a JSON file
    python -m gsi.cli --json panel.json --pretty

    # A batch of panels from CSV -> verdict CSV (async registry ingestion)
    python -m gsi.cli --csv panels.csv --out verdicts.csv

CSV columns understood (all optional except one of voltage/current):
    voltage,current,irradiance,temperature,efficiency,voc,isc,pmax_rated,
    price,brand,region
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import warnings
from typing import List, Optional

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from . import agents  # noqa: E402
from .orchestrator import CLASS_ORDER, GSIEngine, PanelInputs  # noqa: E402


def load_engine(calibrate: bool = True) -> GSIEngine:
    with open(os.path.join(REPO_ROOT, "solarcheck_model_v2.pkl"), "rb") as fh:
        model = pickle.load(fh)
    with open(os.path.join(REPO_ROOT, "solarcheck_scaler_v2.pkl"), "rb") as fh:
        scaler = pickle.load(fh)
    engine = GSIEngine(
        model,
        scaler,
        priors=agents.load_priors(os.path.join(REPO_ROOT, "gsi", "data", "feature_priors.json")),
    )
    if calibrate:
        engine.calibrate_from_csv(os.path.join(REPO_ROOT, "solarcheck_training_data_v2.csv"))
    return engine


# ── Built-in scenarios: the cases this system must get right ────────────────
SCENARIOS = [
    (
        "Genuine Bluesun 100 W (the most common rural panel), fairly priced",
        dict(voltage=22.8, current=4.39, irradiance=980, temperature=29, efficiency=19.4,
             voc=27.36, isc=4.87, pmax_rated=100, price=35000, region="yaounde", source="manual",
             provided=("voltage", "current", "irradiance", "temperature", "efficiency", "voc", "isc")),
    ),
    (
        "Counterfeit 100 W offered at a third of market price",
        dict(voltage=10.6, current=2.75, irradiance=963, temperature=28, efficiency=5.2,
             voc=23.8, isc=2.05, pmax_rated=100, price=9000, region="douala", source="label",
             provided=("voltage", "current", "irradiance", "temperature", "efficiency", "voc", "isc")),
    ),
    (
        "Genuine 550 W Jinko, regional premium price",
        dict(voltage=41.4, current=13.29, irradiance=980, temperature=33, efficiency=21.3,
             voc=49.62, isc=14.03, pmax_rated=550, price=90000, region="garoua", source="label",
             provided=("voltage", "current", "irradiance", "temperature", "efficiency", "voc", "isc")),
    ),
    (
        "Worn genuine panel: real hardware, real losses",
        dict(voltage=17.1, current=4.0, irradiance=964, temperature=30, efficiency=10.9,
             voc=22.0, isc=5.51, pmax_rated=100, price=20000, region="bamenda", source="manual",
             provided=("voltage", "current", "irradiance", "temperature", "efficiency", "voc", "isc")),
    ),
    (
        "Genuine panel measured on a 52 °C roof — heat, not fraud",
        dict(voltage=21.0, current=4.1, irradiance=990, temperature=52, efficiency=19.2,
             voc=26.5, isc=4.85, pmax_rated=100, price=38000, region="maroua", source="manual",
             provided=("voltage", "current", "irradiance", "temperature", "efficiency", "voc", "isc")),
    ),
    (
        "Sticker printed with 45% efficiency (above the physical limit)",
        dict(voltage=22.8, current=4.39, irradiance=980, temperature=29, efficiency=45.0,
             voc=27.36, isc=4.87, pmax_rated=100, price=45000, region="yaounde", source="label",
             provided=("voltage", "current", "irradiance", "temperature", "efficiency", "voc", "isc")),
    ),
    (
        "Only voltage and current measured — thin evidence",
        dict(voltage=22.8, current=4.39, provided=("voltage", "current")),
    ),
    (
        "Meter misread: Imp reported above Isc",
        dict(voltage=22.8, current=6.5, irradiance=980, temperature=29, efficiency=19.4,
             voc=27.36, isc=4.87, source="manual",
             provided=("voltage", "current", "irradiance", "temperature", "efficiency", "voc", "isc")),
    ),
]


def run_demo(engine: GSIEngine, markdown_path: Optional[str] = None) -> None:
    rows: List[dict] = []
    for title, kwargs in SCENARIOS:
        report = engine.reason(PanelInputs(**kwargs))
        rows.append(
            {
                "scenario": title,
                "state": report.state,
                "verdict": report.display_label,
                "confidence": f"{report.confidence*100:.0f}%",
                "counterfeit_p": f"{report.probabilities['COUNTERFEIT']*100:.0f}%",
                "next_test": report.next_tests[0].feature if report.next_tests else "—",
                "report": report,
            }
        )

    width = max(len(r["scenario"]) for r in rows)
    print(f"\n{'SCENARIO'.ljust(width)}  {'STATE':<18} {'VERDICT':<20} {'CONF':>5}  {'P(fake)':>7}  NEXT TEST")
    print("-" * (width + 72))
    for r in rows:
        print(
            f"{r['scenario'].ljust(width)}  {r['state']:<18} {r['verdict']:<20} "
            f"{r['confidence']:>5}  {r['counterfeit_p']:>7}  {r['next_test']}"
        )

    if markdown_path:
        render_markdown(rows, markdown_path)
        print(f"\nMarkdown report written to {markdown_path}")


def render_markdown(rows: List[dict], path: str) -> None:
    lines = [
        "# SolarCheck Africa — GSI mode, worked examples",
        "",
        "Generated by `python -m gsi.cli --demo --markdown docs/GSI_EXAMPLE_VERDICTS.md`.",
        "Every figure below is produced by the engine in this repository; nothing is hand-edited.",
        "",
        "| Scenario | State | Verdict | Confidence | P(counterfeit) | Next test |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['state']} | {row['verdict']} | "
            f"{row['confidence']} | {row['counterfeit_p']} | {row['next_test']} |"
        )

    for row in rows:
        report = row["report"]
        lines += [
            "",
            "---",
            "",
            f"## {row['scenario']}",
            "",
            f"**{report.headline}** — state `{report.state}`, effective confidence "
            f"{report.confidence*100:.0f}%.",
            "",
            f"**Decision:** {report.decision}",
            "",
            f"**Reasoning:** {report.narrative}",
            "",
            "### Belief distribution",
            "",
        ]
        for name, prob in report.probabilities.items():
            lines.append(f"- {name.replace('_', ' ')}: **{prob*100:.1f}%**")

        lines += ["", "### Council", "", "| Agent | Verdict | Score | Confidence | Weight |", "|---|---|---|---|---|"]
        for op in report.opinions:
            verdict = "abstained" if op.verdict is None else op.label.replace("_", " ")
            lines.append(
                f"| {op.agent} | {verdict} | {op.score:+.2f} | {op.confidence*100:.0f}% | {op.weight:.2f} |"
            )

        if report.metacognition:
            lines += ["", "### Metacognition (self-audit)", ""]
            for finding in report.metacognition:
                lines.append(f"- **{finding.code}** ({finding.severity}): {finding.message}")

        if report.counter_case and report.counter_case.arguments:
            lines += ["", "### The case against this verdict", ""]
            for idx, arg in enumerate(report.counter_case.arguments, 1):
                lines.append(
                    f"{idx}. {arg['claim']}  \n"
                    f"   *Test that settles it:* {arg['test']} (cost {arg['cost']}, {arg['duration']}) — "
                    f"flips to `{arg['flips_to']}` if it fails."
                )

        if report.next_tests:
            lines += ["", "### Measurements that would change the answer", ""]
            for test in report.next_tests:
                lines.append(
                    f"- **{test.feature}** — {test.expected_information_gain_bits:.2f} bits recoverable. "
                    f"How: {test.how}. {test.why}"
                )

        phys = report.physics
        lines += [
            "",
            "### Physics",
            "",
            f"- Fill factor: {phys.fill_factor:.3f}",
            f"- Measured power: {phys.power_output:.1f} W → {phys.temp_corrected_power:.1f} W at STC",
            f"- Checks executed: {phys.checks_run}",
            f"- Hard violations: {len(phys.hard_violations)}",
        ]
        for finding in phys.findings:
            lines.append(f"- `{finding.code}` ({finding.severity}): {finding.message}")

        if report.veto_reason:
            lines += ["", f"> **VETO — {report.veto_reason}**"]

        if report.economics:
            econ = report.economics
            lines += [
                "",
                "### Money",
                "",
                f"- Price per watt: {econ.price_per_watt if econ.price_per_watt is None else round(econ.price_per_watt)}",
                f"- Honest regional band: {econ.band}",
                f"- Money at risk: {econ.money_at_risk:,.0f}",
                f"- Payback: {'—' if econ.payback_years is None else f'{econ.payback_years:.1f} years'}",
                f"- {econ.verdict_line}" if econ.verdict_line else "",
            ]

        if report.deployment:
            plan = report.deployment
            lines += [
                "",
                "### Deployment",
                "",
                f"- Max panels in series: {plan.max_series} (Voc at 15 °C = {plan.voc_cold:.1f} V)",
                f"- Cable: {plan.cable_mm2:.1f} mm² for a 10 m run · fuse {plan.fuse_a:.0f} A · "
                f"controller {plan.controller_a:.0f} A",
                f"- Estimated yield: {plan.daily_kwh:.2f} kWh/day, {plan.annual_kwh:.0f} kWh/year",
            ]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(line for line in lines if line is not None) + "\n")


def run_json(engine: GSIEngine, path: str, pretty: bool) -> None:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if "provided" in payload:
        payload["provided"] = tuple(payload["provided"])
    payload = {k: v for k, v in payload.items() if k in PanelInputs.__dataclass_fields__}
    report = engine.reason(PanelInputs(**payload))
    print(json.dumps(report.to_dict(), indent=2 if pretty else None, default=str))


def run_csv(engine: GSIEngine, path: str, out_path: str) -> None:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    fields = [
        "row", "state", "verdict", "confidence", "p_counterfeit", "p_degraded",
        "fill_factor", "power_output", "money_at_risk", "next_test", "veto_reason",
        "n_hard_violations", "top_counter_argument",
    ]
    out_rows = []
    for idx, row in enumerate(rows, 1):
        kwargs = {}
        for key in ("voltage", "current", "irradiance", "temperature", "efficiency", "voc", "isc",
                    "pmax_rated", "price"):
            if row.get(key) not in (None, ""):
                try:
                    kwargs[key] = float(row[key])
                except ValueError:
                    pass
        for key in ("brand", "region", "source"):
            if row.get(key):
                kwargs[key] = row[key]
        kwargs.setdefault("provided", ("voltage", "current", "irradiance", "temperature",
                                       "efficiency", "voc", "isc", "price"))
        report = engine.reason(PanelInputs(**kwargs))
        top_arg = ""
        if report.counter_case and report.counter_case.arguments:
            top_arg = report.counter_case.arguments[0]["claim"][:160]
        out_rows.append(
            {
                "row": idx,
                "state": report.state,
                "verdict": report.display_label,
                "confidence": round(report.confidence, 4),
                "p_counterfeit": round(report.probabilities["COUNTERFEIT"], 4),
                "p_degraded": round(report.probabilities["DEGRADED"], 4),
                "fill_factor": round(report.physics.fill_factor, 4),
                "power_output": round(report.physics.power_output, 2),
                "money_at_risk": round(report.economics.money_at_risk, 0) if report.economics else "",
                "next_test": report.next_tests[0].feature if report.next_tests else "",
                "veto_reason": report.veto_reason or "",
                "n_hard_violations": len(report.physics.hard_violations),
                "top_counter_argument": top_arg,
            }
        )

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"{len(out_rows)} panel(s) assessed → {out_path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SolarCheck Africa GSI engine — command line")
    parser.add_argument("--demo", action="store_true", help="run the built-in scenario suite")
    parser.add_argument("--json", metavar="FILE", help="assess one panel described in JSON")
    parser.add_argument("--csv", metavar="FILE", help="assess a CSV of panels")
    parser.add_argument("--out", metavar="FILE", help="output CSV for --csv (default verdicts.csv)")
    parser.add_argument("--markdown", metavar="FILE", help="write the demo run as a Markdown report")
    parser.add_argument("--pretty", action="store_true", help="indent JSON output")
    parser.add_argument("--no-calibrate", action="store_true", help="skip Platt calibration (faster)")
    args = parser.parse_args(argv)

    if not any((args.demo, args.json, args.csv)):
        parser.print_help()
        return 1

    engine = load_engine(calibrate=not args.no_calibrate)

    if args.demo:
        run_demo(engine, args.markdown)
    if args.json:
        run_json(engine, args.json, args.pretty)
    if args.csv:
        run_csv(engine, args.csv, args.out or "verdicts.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
