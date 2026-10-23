# CrossScore — controlled research-code evidence

This repository contains the frozen B2 teacher/student training and OOD evaluation code, the mechanism-matched RFKD comparator, five-seed result summaries, and their reproducibility checks.

## Scope
- `frozen/b2/`: CE, vanilla KD, CrossScore legacy-as-code and corrected teacher-temperature T=6 controlled experiment implementations.
- `frozen/rfkd/`: RFKD Energy-order T=6 mechanism-matched reimplementation. **Not authenticated official RFKD code.**
- `frozen/teacher/`: frozen teacher training source.
- `frozen/audit/`: evaluation and scientific-audit scripts.
- `evidence/`: summaries, source checksums and independent audits. OOD domains are not independent training seeds.
- `scripts/`: checks and portable workspace preparation.

**Important provenance limit:** The original submitted-paper historical checkpoints are not source-certified by this release. The controlled B2/P1-8 runs are separate from the original submitted result tables. The unverified historical source-candidate directory is deliberately excluded from the public-release review copy; it is archived privately for further research verification. This release is not evidence that every original-paper experiment has been reproduced.

## Minimal verification
Run with Python 3:

```bash
python scripts/verify_hashes.py
python scripts/recompute_stats.py
```

## Data and checkpoints
Full GPU training and OOD evaluation require external third-party benchmark images, original evaluation pool variants and teacher/student checkpoints. They are **not included** in this GitHub package. Existing scripts may refer to the original project directory; reproduce paths in a separate workspace and verify the environment rather than claiming a one-command rerun. See DATA_ACCESS.md.

## Interpretation
The five-seed matched comparison did **not** establish CrossScore corrected T=6 AUROC superiority over the matched RFKD T=6 energy-order baseline. We retain negative and positive results without post-hoc selection.

## Before making this repo public
1. Check rights/co-author permissions for every first-party source file and external dependency.
2. Adopt an actual LICENSE file (the current repository has no asserted license), and verify author information in CITATION.cff.
3. Check code for secrets or internal paths; select which checkpoints are legally releasable.
4. Validate installation and smoke execution on an independent environment.
5. Once approved, publish GitHub release v1.0.0, archive it using Zenodo, and insert the **real** DOI into the journal manuscript.

This is a **review-stage research-code release**, not a final journal claim of comprehensive reproduction.
