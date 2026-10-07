# Step 1 — Plan Preprocessing Protocol

## Goal

Convert the audited Building Consent drawing sets into a reproducible image dataset for the VLM and CLIP-retrieval experiments.

## Dataset unit

The **project** is the atomic unit. Pages are never split independently across training and evaluation.

The official dataset contains 11 real projects only.

## Source freeze

For each project, the preprocessing manifest identifies the audited architectural and structural source roles and selected 1-based PDF page numbers.

The public manifest is anonymised and contains only:

- project ID
- source role
- page number
- page type
- rendered dimensions

Raw file names, project addresses, Drive identifiers and drawings stay private.

## Selected page types

The primary image set covers cost-relevant drawing views where available:

- site plan
- ground-floor plan
- upper-floor plan for multi-storey projects
- roof / roof-framing plan
- elevations
- building section
- structural foundation plan/details
- structural floor/framing plan
- structural bracing plan

Multi-unit projects retain multiple floor/roof/foundation pages under one project ID.

## Rendering

Selected pages are rendered at **150 DPI** as JPEG.

Two images are produced from each selected page:

1. **Full page** — used for VLM experiments because annotations, schedules and drawing context may contain useful information.
2. **Retrieval crop** — used for CLIP-style similarity retrieval. A small amount is cropped from the page edges to reduce retrieval based on title blocks, addresses, logos, approval stamps and other page furniture.

Current crop fractions:

- left: 4%
- top: 4%
- right boundary: 90%
- bottom boundary: 90%

These values are fixed before retrieval evaluation.

## Current image count

The frozen Step-1 manifest contains **95 selected pages** across 11 projects.

Rendering therefore produces:

- 95 full-page images
- 95 retrieval-crop images
- 190 images in total

The rendered image dataset remains private.

## Version alignment

The selected drawings are intended to represent the audited consent/project state used for the historical cost labels.

For version-sensitive projects, especially multi-unit work with later drawing revisions, revision chronology is retained as a limitation rather than silently mixing late drawing revisions with earlier cost evidence.

## Leakage control

For every cross-validation fold:

- all images from the held-out project are excluded from model fitting;
- all images from the held-out project are excluded from the CLIP retrieval index;
- project-level embeddings are aggregated only from pages belonging to that project;
- no target cost appears in image filenames, prompts or retrieval inputs.

## Quality assurance

Selected pages are visually reviewed after rendering.

Checks include:

- correct project
- correct page role
- readable dimensions and annotations
- no obvious blank/wrong page
- multi-unit pages remain grouped under a single project ID
- structural pages correspond to the same audited project

## Reproduction

Place the private audited PDFs in:

```
data/private/raw_plans/P01/arch.pdf
data/private/raw_plans/P01/struct.pdf
...
```

Some projects have an additional role, for example `framing.pdf`.

Then run:

```bash
python -m src.preprocessing.render_selected_pages \
  --manifest data/plan_manifest_public.csv \
  --raw-root data/private/raw_plans \
  --output-root data/private/rendered_plans
```

No raw drawing or rendered private image should be committed to the public repository.
