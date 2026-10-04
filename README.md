# ConstructionGPT — COMPSCI 764

Research project by **Lakshay Arora** and **Mike Ashton** for COMPSCI 764 at the University of Auckland.

## Research question

Can vision-language representations extracted directly from residential Building Consent plans predict visually grounded construction trade costs, and does retrieval of similar historical projects improve prediction accuracy?

## Initial scope

The study focuses on three trade groups with relatively direct visual/structural evidence in consent drawings:

- Foundation / concrete
- Timber framing
- Roofing

## Experimental comparison

The planned evaluation compares:

1. Cost-per-area baseline
2. XGBoost on structured project features
3. Small multilayer perceptron (MLP) on structured project features
4. Vision-language model prediction
5. Vision-language model + similar-project retrieval

Retrieval will use CLIP-style embeddings and cosine similarity over training projects only.

## Evaluation

Primary metrics:

- MAE
- RMSE
- MAPE

All train/validation/test splits must occur at the **project level** before page extraction or retrieval indexing to prevent leakage.

## Repository structure

```
data/                 dataset schemas and non-sensitive metadata
src/
  preprocessing/      plan preparation and feature processing
  baselines/          cost/m², XGBoost and MLP models
  vlm/                vision-language model experiments
  retrieval/          embedding and nearest-neighbour retrieval
  evaluation/         metrics and evaluation utilities
experiments/          experiment configurations and run notes
results/              result tables, figures and summaries
docs/                 research protocol and methodology notes
presentation/         presentation assets and planning
```

## Data policy

Raw client drawings, purchase orders, invoices, addresses, supplier information, and other commercially sensitive records must **not** be committed to this public repository. Only anonymised or derived research data should be stored here.

## Status

Repository initialized. Dataset construction and audit are the next stage.
