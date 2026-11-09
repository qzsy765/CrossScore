# CrossScore round two source and numerical evidence

This v1.2.0 release extends the v1.1.0 source for Scientific Reports revision f111c943-cb85-4a1f-859e-d1dc17946c07. It preserves the earlier release and adds the completed matched-temperature component study and norm-matched follow-up. The corresponding fixed-version Zenodo DOI is assigned through the repository’s enabled GitHub release integration; the concept record is https://doi.org/10.5281/zenodo.23585106.

## Scope

The new evidence is the complete 8-cell, 5-seed C/E/R factorial (12 archived anchors and 28 new 60-epoch trainings), the six-model norm-matched follow-up with nine archived controls, their frozen protocols, evaluation records, and separately implemented numerical checks. Teacher temperature is 6 and student probability temperature is 1 in the factorial. Adverse outcomes, AMP skipped steps and failed scheduling amendments are retained. Previously examined OOD domains and the original loss scales limit inference.

## CPU reproduction from archived numerical records

Use Python 3.10 or later with the two packages in analysis/requirements_cpu.txt. From this folder:

```bash
python analysis/recompute_factorial.py --evidence-dir evidence/factorial --output-dir reproduced/factorial
python analysis/recompute_normmatch.py --evidence-dir evidence/normmatch --output-dir reproduced/normmatch
```

These commands recompute every reported cell summary and factorial contrast from the included domain-level/model-level records and compare with the frozen results. They do not train models or re-evaluate images. The original raw-image-level independent metric checks were performed on the research server: 3,600 checks for the factorial and 1,350 for the norm-matched program. Their immutable reports and original checking sources are included. Raw per-image score arrays remain outside this archive; the CPU entry points cannot independently repeat that raw-array audit.

To regenerate the numerical figure, also install analysis/requirements_figures.txt and run `python analysis/make_factorial_figure.py --evidence-dir evidence/factorial --output-dir reproduced/figure`. The resulting CSV lists all plotted seed values.

## Exact scientific source

frozen_project_sources preserves the project-relative tree of the 14 Python dependencies in the factorial protocol, including its trainer, data loaders, model definitions and OOD evaluator. Each original Python dependency hash is verified in SOURCE_DEPENDENCY_AUDIT.json. The pretrained teacher checkpoint is an external dependency recorded by hash, not a bundled file. The norm-matched trainer and its ID-only protocol/amendment records are preserved alongside it. execution_amendments retains dispatch changes separately from scientific training code. previous_release_v1.1.0 is an unchanged extraction of the official GitHub v1.1.0 tag at commit 9f3eab4, retaining its original notices and portability limitations.

The sources use the original absolute project root /root/private_data/projects/OOD/cross_score_v2. To execute unmodified frozen trainers, recreate this tree in an isolated workspace/container and supply the external datasets, OpenOOD lists, teacher checkpoint and protocol records at the recorded locations. Data-loader environment overrides are documented in the source. Check the trainer CLI and frozen protocol before launching. New paths or any modified source must be treated as a distinct adaptation with new source hashes; never overwrite the archived protocol to imply an identical historical run. The old release contains its own path-adaptation helper. A clean-environment end-to-end GPU training reproduction of this combined archive has not been performed. Archived historical student checkpoint binaries are not bundled, so bitwise replay of the reused anchors is not claimed.

## Data and dependencies

ImageNet-200 and the five OOD domains follow the public OpenOOD benchmark lists at https://github.com/Jingkang50/OpenOOD. Acquire images under each original dataset's terms. Supply the torchvision ResNet-50 ImageNet-1K checkpoint resnet50-0676ba61.pth with SHA256 0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a. Do not redistribute images or weights using this archive's code license. Existing software dependencies (PyTorch, torchvision, NumPy, SciPy, scikit-learn and Pillow) retain their own licenses; their binary distributions are not bundled. Recorded scientific environments are included in the audit reports and must be distinguished from the CPU-summary environment recorded in VALIDATION.json.

## Version and license

The project software retains the MIT license and copyright notice from the existing repository. The previous fixed-version DOI is https://doi.org/10.5281/zenodo.23686448; it identifies the earlier v1.1.0 software, rather than the new v1.2.0 additions. Release metadata is in ZENODO_METADATA.json. Its creator follows the prior software record; all manuscript authors remain listed in the manuscript. Use the fixed-version DOI of the v1.2.0 record after its publication; do not cite the v1.1.0 DOI as covering the new additions. The article and its reply require separate final author approval before journal submission.
