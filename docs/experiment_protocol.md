# ConstructionGPT Experimental Protocol

## 1. Research question

Can vision-language representations extracted directly from residential Building Consent plans predict visually grounded construction trade costs, and does retrieval of similar historical projects improve prediction accuracy?

## 2. Main hypothesis

**H1:** Adding similar-project retrieval to a fixed VLM-based prediction pipeline reduces trade-level prediction error compared with the same VLM pipeline without retrieval.

## 3. Dataset

The official experiment uses **11 real residential projects only**.

Observed label availability after document and PO audit:

- foundation / concrete: 8 projects
- framing manufacturing / frame-truss package: 9 projects
- roofing supply/install: 9 projects

Missing labels remain missing. They are never silently converted to zero or replaced with generated labels for the final experiment.

## 4. Target definitions

### Foundation
Foundation-system cost excluding separately identified earthworks, driveway and unrelated site works where these can be separated.

### Framing
A supplier manufacturing/package target centred on pre-nailed frames, trusses and balance-of-roof/manufacturing scope. This is intentionally narrower than a complete commercial framing cost head so that projects can be compared more consistently.

### Roofing
Main roof supply-and-install package. Fascia, gutter and downpipe values are excluded when separately quoted. Cancelled and superseded roof quotes are not used as final labels.

All targets are ex GST.

## 5. Leakage prevention

The project identifier is the atomic split unit.

Before a held-out project is evaluated, it must be excluded from:

- structured-model fitting
- scaling/encoding fitting
- page/image training examples
- VLM adaptation
- embedding index construction
- retrieval candidates
- target-normalisation statistics

This rule also applies when multiple lots belong to one project: they stay in the same project split.

## 6. Baselines

### B0 — Training-fold cost per area
For each held-out project, calculate the average trade $/m² from labelled training projects only, then multiply by the held-out floor area.

### B1 — XGBoost
Use a compact structured feature set: floor area, units, storeys, bedrooms, foundation type, roof form, cladding and site complexity.

### B2 — MLP
Use the same structured inputs with one-hot encoding, scaling and a small regularised feed-forward network.

Given the small sample, model complexity must remain deliberately limited.

## 7. Plan-image preprocessing

The Step-1 page manifest is frozen before model evaluation.

It contains **95 selected pages** across the 11 projects. Selected page types include site, floor, roof, elevation, section, foundation, framing and bracing views where available.

Each selected page produces:

- a 150-DPI full-page JPEG for VLM experiments;
- a cropped retrieval JPEG for CLIP similarity.

The retrieval crop removes a small fixed amount from page edges to reduce the chance that similarity is dominated by title blocks, addresses, logos or approval stamps.

Raw PDFs and rendered images remain private.

## 8. Vision-language branch

The first VLM experiment uses the frozen full-page image set.

The preferred implementation is a frozen or parameter-efficient representation rather than full-model training from scratch.

No historical price is included in the visual prompt/input.

## 9. Retrieval branch

Project/page embeddings are generated from the frozen retrieval-crop images using a fixed vision-language retrieval model.

For every held-out project:

1. build the retrieval index from training projects only;
2. compute cosine similarity;
3. aggregate page similarities into project-level similarity;
4. retrieve top-k similar training projects;
5. provide compact retrieved project/cost context to the prediction stage.

## 10. Primary ablation

Use the same VLM representation and evaluation projects for:

- VLM without retrieval
- VLM + retrieval

Any change in error can then be attributed more cleanly to retrieval.

## 11. Evaluation

For each trade, report:

- labelled project count
- MAE
- RMSE
- MAPE
- per-project absolute error

Also report:

- mean/median retrieval similarity
- qualitative neighbour audit
- notable failure cases
- computational cost
- limitations caused by small n and noisy historical targets

## 12. Reproducibility

Every experiment should record:

- dataset version
- page-manifest version
- project IDs used
- target definition
- random seed
- cross-validation protocol
- model parameters
- retrieval k
- software versions
- generated result file
