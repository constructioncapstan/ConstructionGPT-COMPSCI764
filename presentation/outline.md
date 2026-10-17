# Oral presentation — final 8–10 minute plan

Target recording: approximately **8:55**.

- **Mike:** Slides 1–5, approximately 4:50
- **Lakshay:** Slides 6–9, approximately 4:05

This keeps both contributions within the required 4–5 minute range to normal speaking variation.

## Slide 1 — ConstructionGPT
- research question
- 11 real projects
- 95 selected pages
- foundation / framing / roofing scope

## Slide 2 — Why the problem matters
- early estimates affect feasibility and procurement
- plans contain useful geometry/specification evidence
- historical cost also contains supplier, site, market and scope effects

## Slide 3 — Existing approaches and gap
- structured ML
- case-based reasoning
- document AI / VLMs
- gap: raw BC-plan representation + historical-project retrieval for trade cost
- H1: retrieval should reduce error relative to the same visual predictor

## Slide 4 — Dataset and audit
- 11 real projects only; no synthetic properties
- 8 foundation labels, 9 framing, 9 roofing
- cancelled/superseded/misfiled evidence removed
- missing remains missing
- project is the atomic split unit

## Slide 5 — Proposed method
- frozen selected plan pages
- 32×32 grayscale retrieval views for the lightweight scored experiment
- fold-specific CLIP-inspired contrastive image-text encoder
- project embedding + PCA/log-area/ridge visual cost head
- top-3 training-project cosine retrieval
- retrieval/visual blend selected by inner LOPO

## Slide 6 — Evaluation
- 11-fold leave-one-project-out evaluation
- cost/m², XGBoost, MLP, visual predictor, retrieval, visual + retrieval
- MAE, RMSE, MAPE
- strict no-leakage rule for representation training and retrieval

## Slide 7 — Final results

| Trade | Best MAPE | CVL visual | CVL + retrieval |
|---|---:|---:|---:|
| Foundation | XGBoost 36.88% | 39.81% | 49.46% |
| Framing | MLP 23.37% | 40.83% | 37.52% |
| Roofing | Cost/m² 42.65% | 46.51% | 66.91% |

Overall MAE: XGBoost ≈ $9.43k; CVL visual ≈ $9.57k; CVL + retrieval ≈ $12.67k.

**Main conclusion:** H1 is not supported.

## Slide 8 — Why retrieval failed
- case bank too small
- drawing style can dominate cost-relevant geometry
- historical costs include hidden commercial/site effects
- top-k forces a precedent even when no close case exists
- example poor match: P02 (70 m², 1 unit) → P06 (505 m², 3 units)
- future retriever should combine visual similarity with structured eligibility and allow “no suitable precedent”

## Slide 9 — Conclusion and future work
- retrieval did not consistently reduce error
- visual representations still contained some predictive signal
- strongest method depended on trade
- increase real case base and standardise target scope
- use high-resolution view-specific crops
- evaluate frozen/pretrained Qwen2.5-VL or Florence-2
- combine visual and structured retrieval constraints
- report uncertainty and route low-confidence cases to estimator review

## Optional system view

A live software demonstration is not required by the rubric. If a separate ConstructionGPT academic deployment is stable, use at most 20–30 seconds on one preloaded/cached example and present it only as an implementation aid, not as evidence of production accuracy.
