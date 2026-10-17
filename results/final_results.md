# Final experimental results

The final experiment uses **11 real residential projects only** and leave-one-project-out (LOPO) evaluation. Missing labels are not scored. Historical trade targets are approximately normalised to a common 2026-Q2 price basis before evaluation.

## Main result

The retrieval hypothesis was **not supported**. Adding visually similar historical projects did not consistently improve the fold-specific contrastive visual predictor.

| Trade | Best baseline / model | Best MAPE | CVL visual MAPE | CVL + retrieval MAPE |
|---|---|---:|---:|---:|
| Foundation | XGBoost | 36.88% | 39.81% | 49.46% |
| Framing | MLP | 23.37% | 40.83% | 37.52% |
| Roofing | Cost/m² | 42.65% | 46.51% | 66.91% |

Across all 26 scored trade-project observations, XGBoost achieved the lowest overall MAE at approximately **NZ$9.43k**, while the contrastive visual model was close at **NZ$9.57k**. Retrieval augmentation increased overall MAE to approximately **NZ$12.67k**.

## Interpretation

Retrieval slightly improved framing MAPE but worsened foundation and roofing. Paired error analysis found no statistically reliable retrieval benefit at the available sample size. The qualitative neighbour audit also showed that visual sheet similarity can select projects that differ substantially in scale, dwelling count, storeys, structural system, or supplier scope.

The final conclusion is therefore deliberately negative: **cost-relevant similarity is harder than visual similarity on this small case base**. The result motivates future retrieval that combines visual embeddings with structured eligibility constraints and uncertainty thresholds.

## Reproducibility and privacy

Only aggregate results are committed publicly. Raw consent drawings, project addresses, supplier quotations, invoices, and project-linked historical price records remain private.
