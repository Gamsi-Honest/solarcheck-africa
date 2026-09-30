# ☀️ SolarCheck Africa — GSI Mode

**AI-powered solar panel quality verification for African markets.** Detects healthy, degraded and
counterfeit panels from bench measurements and a photo of the spec sticker. Built from Cameroonian
market conditions at Université de Yaoundé I.

**Founder: Gamsi** · 2026

---

## What "GSI mode" means here

GSI (General Superintelligence) mode is **not** a bigger model and **not** a marketing label. It is a
change in architecture: the single forward pass is replaced by a system that can **reason, doubt,
disagree with itself, veto its own statistics, and say "I don't know — measure this next."**

Six commitments define it. Every one is enforced in code and pinned by a test:

| # | Commitment | Where it lives |
|---|---|---|
| 1 | **Physics outranks statistics.** A hard physical violation vetoes the classifier, however confident it is. | `gsi/physics.py`, `gsi/orchestrator.py` (`META_VETO_*`) |
| 2 | **A council, not an oracle.** Independent specialists with separate evidence channels, fused with reliability weights. | `gsi/agents.py`, `gsi/orchestrator.py` (`fuse_beliefs`) |
| 3 | **"Inconclusive" is a real answer.** When the top two hypotheses are within 15 points, the system refuses to name a verdict. | `gsi/orchestrator.py` (`_metacognition`) |
| 4 | **The system audits itself.** Every verdict carries the counter-argument, the assumptions it rests on, and its own confidence penalties. | `_metacognition`, `red_team_opinion` |
| 5 | **The next test is part of the answer.** Expected information gain ranks the measurement that would most reduce the doubt. | `gsi/orchestrator.py` (`_plan_next_tests`) |
| 6 | **Confidence is calibrated, not asserted.** Out-of-fold Platt scaling replaces raw softmax scores; out-of-distribution inputs **reduce** confidence. | `gsi/agents.py` (`build_calibrators`, `mahalanobis`) |

---

## Architecture

```
                        ┌──────────────────────────────┐
   measurements ───────►│  LAW LAYER  (physics.py)     │  signs, Vmp<Voc, Imp≤Isc,
   sticker photo ──────►│  hard constraints + envelopes│  FF ceiling, S-Q limit,
   bench photos  ──────►│  environment fusion          │  temperature coefficients
                        └───────────────┬──────────────┘
                                        │ clamp + findings
                        ┌───────────────▼──────────────┐
                        │  COUNCIL                     │
                        │   MLForecast   calibrated RF + Mahalanobis OOD
                        │   Physics      laws & envelopes   (weight .98)
                        │   Perception   image forensics    (weight .75)
                        │   LabelForge   brand/sticker text (weight .70)
                        │   Economics    price band, payback(weight .60)
                        │   Deployment   string/cable/fuse  (weight 0 — no vote)
                        └───────────────┬──────────────┘
                                        │ weighted logarithmic opinion pool
                                        │ + tempered regional base rate
                        ┌───────────────▼──────────────┐
                        │  METACOGNITION               │  channel conflict, OOD,
                        │  audit · veto · calibrate    │  evidence breadth, margin,
                        └───────────────┬──────────────┘  self-doubt, assumptions
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
  RED TEAM (critic)            PLANNER (EIG)                  DECISION
  strongest counter-case       next best measurement          verdict state, money
  + falsification tests        + how to take it               at risk, deployment
```

### The agents

| Agent | Evidence channel | Can veto? | Votes? |
|---|---|---|---|
| **MLForecast** | Calibrated RandomForest (8 electrical features), OOD-aware | no | yes |
| **Physics** | 8 law/envelope checks + performance-ratio test | **yes** | yes |
| **Perception** | Blur, glare, exposure, colour cast, busbar periodicity, duplicate detection | no | yes |
| **LabelForge** | Brand lookalikes, sticker forensics, label-vs-measurement contradiction | yes (contradictions) | yes |
| **Economics** | Price per watt vs regional band, money at risk, payback vs grid and generator | no | yes |
| **Deployment** | Voc cold-morning string sizing, fusing, cable, yield | no | **no, by design** |
| **RedTeam** | Argues the strongest case *against* the verdict, with the test that would settle it | no | **no, by design** |

The red team does not vote because its arguments are a reinterpretation of evidence already counted —
letting it vote would double-count that evidence. The deployment agent does not vote because string
sizing says nothing about authenticity, and pretending otherwise would corrupt the fusion.

### Physics that is enforced

- `Imp ≤ Isc` and `Vmp < Voc` — non-negotiable for a working module
- Fill factor ceiling **0.88** (single-junction silicon is not physically able to exceed this)
- Efficiency ceiling **33.7%** (Shockley–Queisser detailed balance) and a 26.5% commercial realism band
- Temperature coefficient of power **−0.29 … −0.45 %/°C**
- Voc drift **−0.30 %/°C** (drives cold-morning string sizing at 15 °C)
- Isc scaling with irradiance; environment readings outside 50–1400 W/m² or −10 … 75 °C are **fused**
  to STC rather than trusted
- **Performance ratio** against the nameplate, normalised for *both* heat and light — the fairest
  single test of "does this panel do what its sticker claims"

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py            # UI  (binds 0.0.0.0:8501 via .streamlit/config.toml)
```

Optional but recommended for photo scanning: add a Gemini API key.

```toml
# .streamlit/secrets.toml   (git-ignored)
GEMINI_API_KEY = "your-key"
```

The app runs fine without it — manual entry exercises exactly the same council.

### Command line (registry / batch use)

```bash
python -m gsi.cli --demo                          # 8 built-in scenarios, incl. edge cases
python -m gsi.cli --demo --markdown out.md        # rendered worked examples
python -m gsi.cli --json panel.json --pretty      # one panel, full report as JSON
python -m gsi.cli --csv panels.csv --out verdicts.csv   # batch verdicts for a registry
```

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/test_gsi.py -q
```

The 25 tests pin behavioural invariants, not just execution — e.g. *a physics veto must beat a
confident classifier*, *a self-contradictory measurement must produce no verdict*, *a genuine panel
measured on a 52 °C roof must not be called counterfeit*, and *calibration must not worsen
reliability*.

---

## Worked examples

`docs/GSI_EXAMPLE_VERDICTS.md` is generated by the CLI and shows the full report for eight
scenarios — including the ones where the honest answer is "inconclusive" or "your meter is wrong".
Regenerate any time:

```bash
python -m gsi.cli --demo --markdown docs/GSI_EXAMPLE_VERDICTS.md
```

| Scenario | Result |
|---|---|
| Genuine Bluesun 100 W, fairly priced | `HEALTHY_BASIC`, 65% |
| Counterfeit 100 W at a third of market price | `COUNTERFEIT`, 95% (label veto) |
| Genuine 550 W Jinko | `HEALTHY_PREMIUM`, 79% |
| Worn genuine panel | `DEGRADED`, 79% |
| Genuine panel on a 52 °C roof | `HEALTHY_BASIC`, 51% — heat, not fraud |
| Sticker claiming 45% efficiency | `COUNTERFEIT`, 95% (Shockley–Queisser veto) |
| Only voltage and current measured | `INCONCLUSIVE` + the measurement to take next |
| Meter misread (Imp > Isc) | `MEASUREMENT_ERROR`, 0% — no verdict given |

---

## Repository layout

```
app.py                     Streamlit UI (GSI mode)
gsi/
  __init__.py              public API
  orchestrator.py          fusion, metacognition, planner, decision, narrative
  agents.py                the council + calibration + OOD detection
  physics.py               law layer, findings, datasheet library
  perception.py            image forensics (numpy/PIL only)
  regions.py               African market intelligence (yield, tariffs, price bands)
  build_priors.py          derives gsi/data/feature_priors.json from the training CSV
  cli.py                   command line: demo, single JSON, batch CSV
  data/feature_priors.json class statistics, covariances, Mahalanobis geometry
tests/test_gsi.py          invariant tests
docs/GSI_EXAMPLE_VERDICTS.md  generated worked examples
solarcheck_model_v2.pkl    trained RandomForest (accuracy 0.994 on held-out data)
solarcheck_scaler_v2.pkl   StandardScaler
solarcheck_training_data_v2.csv  2,400 samples, 4 classes
```

Regenerate the statistical priors after retraining:

```bash
python -m gsi.build_priors
```

### A note on the training data

The GSI layer surfaced a real limitation worth fixing at the data level: the training set's
"healthy basic" class is dominated by ~150 W modules, so a *very* common 100 W rural panel sits
outside the model's experience. Rather than hide this, the engine detects it (class-conditional
Mahalanobis distance) and reports it:

> *"This panel sits at the 99th percentile of all training data… it is most likely a panel size the
> training set under-represents rather than a bad panel — which is exactly why the physics channel,
> not the forest, carries this verdict."*

Collecting 100 W and thin-film samples is the highest-value next data task.

---

## Regional coverage

19 markets across Africa (`gsi/regions.py`), each with specific yield, blended residential tariff,
grid reliability, an honest price band per watt and a counterfeit-prevalence estimate — Cameroon
(Yaoundé, Douala, Bafoussam, Bamenda, Garoua, Maroua, Kribi, Ngaoundéré), Senegal, Côte d'Ivoire,
Nigeria, Ghana, Kenya, Rwanda, DR Congo, Tanzania, Uganda, Zambia, Ethiopia and Niger.

Price bands are published only where they are defensible (XAF/XOF markets); elsewhere the economics
agent records the price for the community database instead of guessing, and says so.

---

## Deliberate limits

- The verdict is a **probabilistic engineering judgement**, not a laboratory certification.
- Sticker forensics from a phone photo is **supporting evidence, never proof** — the perception agent
  caps its positive direction and refuses to score a label it cannot see clearly.
- A "genuine at 90%" verdict still means one panel in ten like this would surprise the system. That
  is why the counter-case is printed *above* the decision, and why `INCONCLUSIVE` exists.
- The model was trained on synthetic samples anchored to real datasheets; field measurements from
  actual Cameroonian market panels are the next step to real calibration.

---

*SolarCheck Africa · GSI Mode 3.0 · Université de Yaoundé I · Founder: Gamsi · 2026*
*Changing how the world sees Africa — one panel at a time.*
