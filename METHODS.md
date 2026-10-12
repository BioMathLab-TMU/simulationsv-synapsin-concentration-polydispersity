# Reproducibility and interpretation

## Study design and physical setting

The 12 runs form a balanced 2 × 2 design with three fixed seed blocks
(9210001–9210003). Each run has 200 vesicles. At 10 µM the input has 602
synapsin agents; at 60 µM it has 3613. The vesicle core radius is either fixed
at 20.45 nm or drawn from a 20.45 ± 2.5-nm truncated distribution with
allowed core radii 15–30 nm. A 4.8-nm corona is added to the core for
effective excluded-volume/interaction geometry. The domain is a closed,
reflecting sphere of 0.1 µm³. These are **recorded model inputs**, not a
claim that every presynaptic terminal has this size or composition.

The actual configurations are authoritative. The included simulator uses
Euler–Maruyama with Δt = 0.0007 ms for 40,000 moves (28 ms); 200 states were
saved at 200-step intervals (0.14 ms), starting at step 200. The force
neighborhood is a fixed two-cell linked-cell neighborhood without an extra
radial force cutoff. Steric admissibility is imposed by bounded-overlap
projection; the synaptic-vesicle corona may interpenetrate. The closed-domain
geometry uses direct Euclidean distances in analysis, never periodic images.

All three force terms remain on in all 12 runs. Radius dispersion also changes
the realized third moment of the core-radius distribution and effective
hydrodynamic radii; the comparison is **not** an isolated variance-only
experiment at identical packing fraction and diffusion coefficient.

## Structural graph and summary measures

For pair *i,j*, let **x** be the saved center, *R* the effective outer
radius, and *g* the outer-surface gap:

`g_ij = ||x_i - x_j|| - R_i - R_j`.

An edge exists when `g_ij ≤ τ`. The preregistered primary shell is τ = 2 nm;
τ = 0, 4, 6 nm are sensitivity definitions. This shell is an **analysis
threshold**, not a force cutoff or a count of actual microscopic collisions.
Components are connected components of the undirected graph at each saved
state. The largest component is selected by its **total agent count**, and the
reported vesicle fraction is the number of vesicles in that component divided
by 200. Species recruitment is the fraction of each species in any
non-singleton component. A run summary averages each frame measure over the
last 60 saved states (19.74–28 ms). These correlated frames are never counted
as 60 independent replicates.

The supplementary measures include component gyration-tensor anisotropy,
radial synapsin-minus-vesicle core–shell index (scaled by gyration radius),
and type assortativity within mixed components. The code contains exact
definitions and handles missing/undefined component statistics explicitly.

## Density, RDF, and uncertainty

Interior-probe local number density is computed on six selected late frames
with 4096 uniformly sampled candidate probe centers, using probe radii
35/50/70 nm. Probes touching the boundary are excluded. This is an
interior-conditional spatial diagnostic, **not** a fitted dense-phase
concentration. RDF-like pair counts are divided by a radius-accessible
ideal-gas geometric null in the same closed domain, with four null replicas,
4-nm bins up to 120 nm, and a minimum expected count of five per bin. That
null corrects finite geometry but not excluded volume, interactions, or
equilibration.

Each plotted inferential unit is one seed-level trajectory. Seed-paired
differences compare 10 to 60 µM and monodisperse to polydisperse at fixed
other-factor level. The seed-level factor interaction is a difference in
differences. Descriptive 95% paired-*t* intervals use 2 degrees of freedom;
exact two-sided sign-flip tests enumerate all `2³ = 8` sign patterns. Thus
their smallest possible two-sided *p* is 0.25; they cannot support a
conventional `p < .05` claim with three blocks. The exploratory nine-feature
logistic classifier uses fold-local imputation/scaling and holds out an entire
seed block at a time. Its targets are the known experimental settings, not
independently measured biological states.

For concentration-label sensitivity, the public script enumerates all eight
joint within-seed flips, applying each flip to both radius arms before
rerunning the complete leave-one-seed-out classifier. Two remappings attain
the observed perfect accuracy. The 2/8 tail fraction describes this finite
label exercise; the imposed design was not randomized, so it is not a
population-level significance test. The three shared seed labels do not
become six independent blocks because the two radius arms consume random
numbers differently. Further small-sample decisions are documented in
[`STATISTICAL_DECISIONS.md`](STATISTICAL_DECISIONS.md).

The supplementary factorial model has an intercept, two seed-block dummy
variables, the two treatment indicators, and their product. It has 12 rows,
rank 6 and six residual degrees of freedom. Coefficient intervals use the
HC3 heteroscedasticity-robust sandwich covariance and a *t* critical value
with six residual degrees of freedom. Holm correction is applied across the
three terms per observable. Separately, the four prespecified within-stratum
paired contrasts receive Holm correction in two families per observable.
The public pipeline rebuilds and cross-checks these tables too; their
small-sample estimates should not be mistaken for population-level effects.

## What this release does and does not establish

The HDF5 arrays and code reproduce finite-time structural organization and
the specified numerical quality-control checks. These 12 runs do not, alone,
establish thermodynamic LLPS, equilibrium, a physical phase boundary,
first-passage/contact kinetics, or drug response. In particular, the large
component fraction approaches a ceiling in the 60-µM conditions; interpret
its contrast together with recruitment, component structure, density, and
the shell sensitivity rather than as a sole phase marker.

The public HDF5 is an article-specific, lossless **subset**. Provenance
containing private machine paths, private Git state and unrelated per-particle
force-decomposition arrays was not published. The original raw HDF5 SHA-256,
public HDF5 SHA-256, exact numerical identity check and dataset count for
each run are in [`data/manifest.json`](data/manifest.json). The parent project
retains the full originals without modification.
