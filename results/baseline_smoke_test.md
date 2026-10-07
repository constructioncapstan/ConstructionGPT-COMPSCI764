# Preliminary baseline smoke test

**Status:** preliminary engineering checkpoint, not final reported experiment.

The current 11-project dataset was evaluated using leave-one-project-out prediction on observed labels only.

## Current metrics

| Trade | Model | n | MAE (NZD) | RMSE (NZD) | MAPE |
| --- | --- | ---: | ---: | ---: | ---: |
| Foundation | Cost/m² | 8 | 17,455.43 | 25,772.87 | 39.55% |
| Foundation | XGBoost | 8 | 12,930.09 | 18,015.37 | 37.54% |
| Foundation | MLP | 8 | 24,887.03 | 32,441.48 | 52.25% |
| Framing package | Cost/m² | 9 | 4,539.42 | 5,327.70 | 26.65% |
| Framing package | XGBoost | 9 | 7,778.97 | 10,105.66 | 39.60% |
| Framing package | MLP | 9 | 10,519.58 | 13,980.03 | 46.19% |
| Roofing | Cost/m² | 9 | 7,266.97 | 11,918.64 | 45.73% |
| Roofing | XGBoost | 9 | 6,049.30 | 6,701.63 | 51.77% |
| Roofing | MLP | 9 | 6,629.89 | 8,038.45 | 44.96% |

## Interpretation

These numbers are useful as a pipeline check, not as a final scientific conclusion.

The sample is extremely small and several projects are structurally very different (single dwellings versus multi-lot developments). The MLP is particularly vulnerable to overfitting. The cost-per-area baseline remains competitive, especially for the framing package.

Before the final report, these results should be rerun after:

1. final confirmation of all structured features;
2. VLM feature extraction is frozen;
3. retrieval configuration is frozen;
4. any remaining PO scope discrepancies are documented;
5. the exact evaluation protocol is locked.

The final paper should report the same project-level split rule for every method.
