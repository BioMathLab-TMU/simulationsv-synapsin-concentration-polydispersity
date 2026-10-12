# Synapsin concentration and vesicle-size polydispersity

This repository contains the frozen simulation inputs, article-specific
trajectories, source code, numerical audit tables, and reproducible analysis
for a 2 × 2 study of synapsin concentration (10/60 µM) and synaptic-vesicle
core-radius dispersion (0/2.5 nm). There are three independent, shared seed
blocks per condition: 12 completed trajectories in total.

This is a **structural-snapshot study**. A connected-component change in this
finite, confined simulation is not by itself proof of thermodynamic liquid–
liquid phase separation, biological kinetics, equilibrium, or a drug effect.
All force terms (electrostatic, hydrophobic, and van der Waals) are held on in
every included run; this dataset does not isolate their effects.

## What is here

| Path | Contents |
|---|---|
| [`configs/`](configs/) | The 12 exact study settings, with publication-safe run labels |
| [`data/manifest.json`](data/manifest.json) | Design, public file paths, SHA-256 digests and per-file numerical-export audits |
| [`data/*.h5`](data/) | Lossless article-specific snapshots and runtime/QC arrays |
| [`analysis/`](analysis/) | Standalone spatial, density/RDF, paired-statistics and exploratory ML code |
| [`reference/`](reference/) | Reviewed numerical tables underlying the figures and reported estimates |
| [`figures/`](figures/) | Figure PDFs, PNGs and compressed 320-dpi TIFFs for this study only |
| [`METHODS.md`](METHODS.md) | Model/analysis definitions and inference limits |
| [`EXPERIMENTAL_CONTEXT.md`](EXPERIMENTAL_CONTEXT.md) | Primary experimental comparisons and non-comparability limits |
| [`STATISTICAL_DECISIONS.md`](STATISTICAL_DECISIONS.md) | Audit of additional small-sample statistical suggestions |
| [`THIRD_PARTY.md`](THIRD_PARTY.md) | Notices for the vendored C++ headers |

The private parent repository, manuscript drafts, personal paths, and other
study matrices are **not included**. The public HDF5 files omit private Git/
workstation provenance and article-irrelevant per-particle force-decomposition
and pairwise-diagnostic arrays. The complete originals remain in the private
project. Every **exported** dataset was compared to its original for name,
data type, shape, and exact bytes; the manifest records the original raw-file
hash and the published-file hash. Scientific arrays were never rescaled,
rounded, re-sampled, or reordered during export.

## Verify the analysis

Use Python 3.12. Create a virtual environment, install
[`requirements.txt`](requirements.txt), then from the repository root run:

```bash
python -m analysis.run --manifest-only
python -m analysis.run
python -m analysis.inference_checks
python -m analysis.paper_figures
python -m analysis.plots
```

The first command checks the complete 2 × 2 × 3 manifest, the 12 settings and
the 12 data-file SHA-256 values. The second independently rebuilds 200 graph
snapshots per run, 0/2/4/6-nm shell sensitivities, the 60-frame late summaries,
interior-probe local density, finite-geometry-null RDF, seed-paired contrasts,
blocked HC3 factorial estimates, Holm-adjusted planned contrasts, and
exploratory ML. It compares the recomputed metrics against
`reference/`; all generated tables and an audit JSON go to ignored `outputs/`.
The third command enumerates all eight concentration-label remappings that
keep each seed block's two radius arms together, and checks the first saved
window. The fourth regenerates all nine publication figures in three formats in
`figures/` using the archived `Single_Execution` palette; this changes
visualization only, not the HDF5 files or published run values. The final
command creates four optional compact diagnostic plots in `outputs/plots/`.
On the verified package, the pipeline ends with:

```text
ARTICLE_TWO_REANALYSIS_COMPLETE runs=12 status=all_checks_passed
```

The seed, not the individual saved frame or particle, is the unit of
independent replication. ML uses leave-one-seed-block-out validation and is
descriptive, not a phase classifier or prospective biological predictor.

## Build the simulation code

The included C++23 code is a publication subset of the archived simulation
source. The active numerical/force logic is unchanged; disabled historical
scenario examples and private project layout were removed. On Ubuntu, Mint,
or WSL, install a C++23 compiler, CMake ≥3.20, HDF5 C/C++ development files,
and Boost system/filesystem development files. OpenMP is required. Example:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
python3 tests/smoke_simulator.py
```

The smoke test runs four steps in an isolated temporary directory and checks
that a nonempty HDF5 output was produced. It is a **software check only**, not
a reproduction or validation of the 28-ms scientific result. To rerun a full
trajectory intentionally, use a matching `configs/*.json` file, for example:

```bash
./build/SimulationSV configs/C10_mono_seed9210001.json
```

Full reruns require substantially more time/resources and may not be bitwise
identical across compiler, CPU, HDF5, OpenMP, and library versions. The frozen
published trajectories and their hashes, rather than an assertion of
cross-platform bitwise deterministic reruns, are the numerical record.

## Attribution and usage

Please cite the accompanying article when it becomes available. Until then,
[`CITATION.cff`](CITATION.cff) identifies this dataset and its authors.
Third-party header licenses are listed in [`THIRD_PARTY.md`](THIRD_PARTY.md).
No license for the study-specific code/data has been assigned yet; public
visibility alone does not grant reuse rights. The repository owner should
choose and add the intended license before relying on it as an openly
reusable software/data release.
