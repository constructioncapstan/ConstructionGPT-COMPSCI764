# Dataset V1 — 11 real projects

The official ConstructionGPT experiment uses eleven audited residential projects.

## Target coverage

| Target | Labelled projects | Missing |
| --- | ---: | ---: |
| Foundation / concrete | 8 | 3 |
| Framing manufacturing / frame-truss package | 9 | 2 |
| Roofing supply/install | 9 | 2 |

## Missing values

Missing targets remain missing. They are not set to zero and are not replaced with synthetic labels for final evaluation.

Each trade experiment therefore uses its own labelled subset.

## Privacy

The public repository contains no addresses, raw drawings, supplier POs, invoices or commercially sensitive project-level cost records.

Private data should be stored under `data/private/` or outside the repository.

## Target scope

The three labels are intentionally scoped to make historical records more comparable:

- **Foundation:** foundation-system package, excluding separately identifiable earthworks/driveways.
- **Framing:** supplier manufacturing/package amount for pre-nailed frames, trusses and related manufacturing scope.
- **Roofing:** main roof supply-and-install package, excluding separately quoted fascia/gutters/downpipes when separable.

## Data quality

The audit identified several issues that are retained in the private provenance table:

- misfiled documents
- superseded quotes
- cancelled POs
- duplicate records
- drawing revisions after original pricing
- partial trade scopes

These are treated as modelling limitations rather than silently corrected.
