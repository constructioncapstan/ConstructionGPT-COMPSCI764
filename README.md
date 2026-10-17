# ConstructionGPT — COMPSCI 764

Research project by **Lakshay Arora** and **Mike Ashton** for COMPSCI 764 at the University of Auckland.

## Research question

Can vision-language representations extracted directly from residential Building Consent plans predict visually grounded construction trade costs, and does retrieval of similar historical projects improve prediction accuracy?

## Dataset

The official research dataset contains **11 real residential projects**. No synthetic properties are used in the final experiment.

Observed-target coverage after the document and PO audit:

- Foundation / concrete: **8 / 11**
- Framing manufacturing / frame-truss package: **9 / 11**
- Roofing supply and installation: **9 / 11**

Missing labels remain missing. They are not replaced with zero or synthetic values for final evaluation.

Raw drawings, addresses, POs, invoices and supplier records remain private and are not committed to this public repository.

## Plan preprocessing

The audited source drawings were reduced to a frozen **95-page primary image manifest** across the 11 projects. Selected pages include site plans, floor plans, roof and roof-framing plans, elevations/sections, and structural foundation/framing/bracing views where available.

Each selected page is rendered in two forms:

1. a full page retained for higher-resolution multimodal work;
2. a retrieval crop with page edges removed to reduce title-block/address/logo bias.

For the reported lightweight experiment, retrieval views are converted to grayscale and reduced to 32 x 32 pixels. The rendered private image dataset is not committed.

## Final experimental comparison

The leakage-controlled leave-one-project-out experiment compares:

1. training-fold cost-per-area baseline;
2. XGBoost on structured project features;
3. a small structured MLP;
4. a fold-specific CLIP-inspired contrastive image-text representation + ridge cost head;
5. visual similarity retrieval;
6. the same visual predictor augmented by retrieval.

The encoder is retrained within every outer fold. The held-out project never appears in representation fitting, structured-model fitting, normalisation, or the retrieval index.

## Final result

The main retrieval hypothesis was **not supported**.

| Trade | Best MAPE | CVL visual MAPE | CVL + retrieval MAPE |
|---|---:|---:|---:|
| Foundation | XGBoost — 36.88% | 39.81% | 49.46% |
| Framing | MLP — 23.37% | 40.83% | 37.52% |
| Roofing | Cost/m² — 42.65% | 46.51% | 66.91% |

Across the 26 scored trade-project observations, XGBoost has the lowest overall MAE (~NZ$9.43k), with the contrastive visual predictor close behind (~NZ$9.57k). Retrieval augmentation increases overall MAE to ~NZ$12.67k.

This negative result is part of the research contribution: visually similar drawings are not necessarily similar in cost when the case base is small and historical targets contain commercial, site and scope effects that are not visible in the consent plans.

See `results/final_results.md` and `results/final_metrics.csv`.

## Repository structure

```text
data/                 schemas and non-sensitive metadata only
src/
  preprocessing/      PDF/page and structured-feature processing
  baselines/          cost/m², XGBoost and MLP baselines
  vlm/                contrastive image-text representation
  retrieval/          embedding and nearest-neighbour retrieval
  evaluation/         metrics and evaluation utilities
scripts/               reproducible experiment entry points
experiments/           experiment configuration
results/               aggregate public results only
docs/                  methodology and dataset protocol
presentation/          presentation planning
```

## Private data layout

The public code expects private audited data outside version control, for example:

```text
data/private/
  research_dataset.csv
  plan32.npy
  plan32_meta.json
  raw_plans/
    P01/
      arch.pdf
      struct.pdf
    ...
```

Preprocessing can be reproduced with:

```bash
python -m src.preprocessing.render_selected_pages \
  --manifest data/plan_manifest_public.csv \
  --raw-root data/private/raw_plans \
  --output-root data/private/rendered_plans
```

The full LOPO experiment can be reproduced after preparing the private image array and metadata:

```bash
python scripts/run_full_experiment.py \
  --dataset data/private/research_dataset.csv \
  --images data/private/plan32.npy \
  --image-meta data/private/plan32_meta.json
```

## Status

- 11-project document and PO audit: complete
- Research dataset: complete
- 95-page preprocessing manifest: complete
- Structured baselines: complete
- Contrastive visual experiment: complete
- Similar-project retrieval experiment: complete
- Retrieval ablation and LOPO evaluation: complete
- Aggregate results: committed
- Final report and oral presentation: prepared separately for course submission
