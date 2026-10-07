# Final report outline

Target length: 10–12 pages excluding references, IEEE format.

## Abstract — 150–200 words
- problem and motivation
- 11-project dataset
- baseline + VLM + retrieval methodology
- main quantitative finding
- main limitation

## 1. Introduction — about 1–1.5 pages
- why early residential cost estimation matters
- why Building Consent plans are difficult multimodal inputs
- research question
- H1: retrieval improves VLM trade-cost prediction
- contributions

## 2. Related Work — maximum 1 page
Condense the submitted literature review around:
- structured construction-cost prediction
- case-based / similar-project reasoning
- drawing/document understanding
- VLMs and multimodal retrieval
- resulting research gap

## 3. Methodology — largest section
### 3.1 Dataset and target construction
- 11 real projects
- project audit
- foundation/framing/roofing target definitions
- missing-label policy
- revisions/cancelled/duplicate handling

### 3.2 Plan preprocessing
- architectural and structural page selection
- rendering resolution
- project-level split before crops/embeddings

### 3.3 Structured baselines
- cost/m²
- XGBoost
- MLP

### 3.4 Vision-language representation
- frozen/pre-trained VLM setup
- page/project aggregation
- regression/prediction head

### 3.5 Similar-project retrieval
- CLIP-style embeddings
- cosine similarity
- top-k
- training-only retrieval index

### 3.6 Experimental protocol
- leave-one-project-out/project-level CV
- MAE, RMSE, MAPE
- retrieval ablation
- qualitative neighbour audit

## 4. Implementation and Results
- implementation details and compute
- baseline table
- VLM-only results
- VLM + retrieval results
- per-trade comparison
- per-project errors
- retrieval examples
- failure analysis

## 5. Discussion
- did H1 hold?
- where retrieval helped/hurt
- target noise
- drawing similarity vs title-block/layout similarity
- implications of small n

## 6. Conclusion and Future Work — about 1 page
- answer research question
- strongest finding
- limitations
- larger dataset
- cost normalisation
- better retrieval representations
- parameter-efficient VLM adaptation

## References
Minimum 10; use the strongest sources from the submitted literature review.

## Contributions
Brief, factual breakdown of Lakshay and Mike's work.
