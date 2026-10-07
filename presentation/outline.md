# Oral presentation — 8 to 10 minutes

Target: approximately equal contribution, about 4–5 minutes each.

## Slide 1 — ConstructionGPT
- project title
- authors
- one-sentence research question

## Slide 2 — Why this problem matters
- early construction estimates affect feasibility/procurement
- manual interpretation of technical plans is slow
- cost also contains hidden market/project effects

## Slide 3 — Existing approaches and gap
- structured ML
- construction document AI
- case-based reasoning
- VLMs
- gap: raw BC plans + retrieved historical projects for trade cost

## Slide 4 — Dataset
- 11 real projects
- foundation labels: 8
- framing labels: 9
- roofing labels: 9
- strict missing-data and leakage policy

## Slide 5 — Proposed method
- plans -> preprocessing -> VLM representation
- training-project retrieval -> top-k similar projects
- trade-cost prediction

## Slide 6 — Experimental design
- cost/m²
- XGBoost
- MLP
- VLM-only
- VLM + retrieval
- project-level cross-validation
- MAE/RMSE/MAPE

## Slide 7 — Results
- final comparison table
- one clear chart
- emphasise VLM-only versus VLM+retrieval

## Slide 8 — Retrieval example / short system view
- show one held-out plan
- retrieved similar projects
- predicted versus actual
- optional short Vercel system demonstration

## Slide 9 — Conclusions and limitations
- answer research question
- small dataset / noisy historical costs
- future work

## Suggested speaking split

### Mike
Slides 1–4: problem, importance, existing approaches, dataset context.

### Lakshay
Slides 5–9: method, experiments, retrieval, results, system view, conclusions.
