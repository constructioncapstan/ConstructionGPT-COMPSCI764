# ConstructionGPT Experimental Protocol

## 1. Research question

Can vision-language representations extracted directly from residential Building Consent plans predict visually grounded construction trade costs, and does retrieval of similar historical projects improve prediction accuracy?

## 2. Main hypothesis

**H1:** Adding similar-project retrieval to a fixed VLM-based prediction pipeline reduces trade-level prediction error compared with the same VLM pipeline without retrieval.

## 3. Secondary comparison

Structured-feature models provide a strong conventional baseline for determining whether information learned directly from plan imagery contributes useful predictive signal.

## 4. Target trades

The initial experiment is restricted to:

1. foundation / concrete
2. timber framing
3. roofing

## 5. Dataset construction

Each property is treated as one independent research project. Historical trade targets must be reconstructed from available project cost evidence.

Target cleaning should document:

- included POs/invoices
- excluded variations
- cancelled or duplicated orders
- credits
- GST treatment
- target date
- confidence/quality status
- unresolved ambiguity

Missing information must not be silently interpreted as zero.

## 6. Leakage prevention

The dataset must be split at project level before:

- rendering pages into training examples
- extracting embeddings
- fitting feature normalisation
- building a retrieval index

For test-project inference, retrieval candidates must come only from the permitted training/history pool.

## 7. Baselines

### B0 — Cost per area
Simple trade cost-per-m² estimate.

### B1 — XGBoost
Structured project attributes to trade-level target cost.

### B2 — MLP
Small feed-forward neural network on the same structured input features.

## 8. Vision-language branch

Plan sheets will be rendered at sufficient resolution and relevant drawing views/pages selected. VLM representations will be used for trade-cost prediction.

The exact VLM implementation will be frozen before final evaluation.

## 9. Retrieval branch

Plan/project representations will be embedded using a fixed retrieval model. Cosine similarity will select top-k historical training projects.

The primary ablation compares:

- VLM without retrieval
- same VLM + retrieval

## 10. Evaluation

Report trade-level and aggregate:

- MAE
- RMSE
- MAPE

Also record:

- retrieval quality examples
- failure cases
- computational considerations
- limitations caused by noisy historical targets

## 11. Reproducibility

Final experiments should record:

- random seed
- dataset version
- split definition
- model/configuration
- retrieval k
- target-normalisation method
- software dependencies
- metric implementation
