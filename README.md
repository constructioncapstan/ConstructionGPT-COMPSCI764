# ConstructionGPT — COMPSCI 764

Research project by **Lakshay Arora** and **Mike Ashton** for COMPSCI 764 at the University of Auckland.

## Research question

Can vision-language representations extracted directly from residential Building Consent plans predict visually grounded construction trade costs, and does retrieval of similar historical projects improve prediction accuracy?

## Dataset

The current research dataset contains **11 real residential projects**. No synthetic properties are used in the official experiment.

After document and PO auditing, the current observed-target coverage is:

- Foundation / concrete: **8 / 11**
- Framing manufacturing / frame-truss package: **9 / 11**
- Roofing supply and installation: **9 / 11**

Missing labels remain missing. They are not replaced with zero or synthetic values for the final evaluation.

Raw drawings, addresses, POs, invoices and supplier records remain private and are not committed to this public repository.

## Step 1 — plan preprocessing

The audited source drawings have now been reduced to a frozen **95-page primary image manifest** across the 11 projects.

The selected pages cover, where available:

- site plans
- ground/upper floor plans
- roof and roof-framing plans
- elevations and sections
- structural foundation plans/details
- structural framing plans
- structural bracing plans

Each selected page is rendered in two forms:

1. **full page** for VLM experiments;
2. **retrieval crop** for CLIP-style similarity retrieval, with page edges cropped to reduce title-block/address/logo bias.

The rendered private image dataset is not committed. The public repository contains only the anonymised page manifest and the reproducible rendering script.

## Experimental comparison

The planned evaluation compares:

1. Cost-per-area baseline
2. XGBoost on structured plan/project features
3. Small MLP on the same structured features
4. Vision-language representation model
5. Same VLM + similar-project retrieval

The primary ablation is **VLM-only vs VLM + retrieval**.

## Evaluation

Primary metrics:

- MAE
- RMSE
- MAPE

Because the dataset is small, evaluation is performed at the **project level** using leave-one-project-out or equivalent project-level cross-validation. Metrics for each trade are computed only on projects with an observed target for that trade.

No validation or test project may appear in the training data or retrieval index.

## Repository structure

```
data/                 schemas and non-sensitive metadata only
src/
  preprocessing/      PDF/page and structured-feature processing
  baselines/          cost/m², XGBoost and MLP baselines
  vlm/                vision-language representation experiments
  retrieval/          CLIP-style embedding retrieval
  evaluation/         metrics and evaluation utilities
scripts/               reproducible experiment entry points
experiments/           experiment configuration
results/               generated aggregate results only
docs/                  methodology and dataset protocol
presentation/          presentation planning
```

## Private data layout

The rendering script expects private audited PDFs outside version control:

```
data/private/raw_plans/
  P01/
    arch.pdf
    struct.pdf
  P02/
    arch.pdf
    struct.pdf
  ...
```

Some projects have an additional source role such as `framing.pdf`.

Run preprocessing with:

```bash
python -m src.preprocessing.render_selected_pages \
  --manifest data/plan_manifest_public.csv \
  --raw-root data/private/raw_plans \
  --output-root data/private/rendered_plans
```

## Current status

- Repository scaffold: complete
- 11-project document/PO audit: complete
- Research dataset V1: complete
- Baseline/evaluation code: implemented
- **Step 1 plan preprocessing: complete — 95 primary pages frozen**
- CLIP retrieval experiment: next
- VLM experiment: follows retrieval baseline
