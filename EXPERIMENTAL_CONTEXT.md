# External biological context for this article

The included model outputs are finite-time proximity-graph and spatial
descriptors. None of the cited experiments is an exact realization of the
model's two concentration levels, radius law, closed sphere, and graph-edge
definition. The comparisons below therefore have different strengths.

## A quantitative geometry-scale comparison

[Fernández-Busnadiego et al., *Journal of Cell Biology* (2010), DOI
10.1083/jcb.200908082](https://doi.org/10.1083/jcb.200908082) measured
synaptic-vesicle membrane diameter in untreated synaptosomes by cryo-electron
tomography: **41.6 ± 5.6 nm (mean ± across-vesicle SD)**. The model's
single-radius membrane core is 40.9 nm in diameter. Across the three unique
heterogeneous radius samples at 10 µM, the realized core diameter is about
**41.2 ± 4.8 nm**, where the second number is the mean within-run population
SD, not an uncertainty interval for the group mean. The 60-µM arm reuses the
same three radius samples and is not counted again. See
[`supp_experimental_context.pdf`](figures/supp_experimental_context.pdf) and
[`reference/external_experimental_benchmarks.csv`](reference/external_experimental_benchmarks.csv).

This comparison supports the **scale** of the model's membrane-core diameters;
it does not validate the assumed truncated-normal distribution, the effective
4.8-nm corona, synapsin concentration, or cluster topology. The experimental
preparation is a synaptosome, not this model's finite reflecting sphere.

## Qualitative organization context, not a fitted endpoint

The same tomography study reports that more than 80% of vesicles in untreated
synaptosomes had at least one *morphologically resolved connector*, and that
large connector-defined clusters occurred. Our graph's `Q_SV` is based on a
2-nm effective outer-surface separation between agents. A geometric edge
cannot be identified with a physical connector. We therefore cite this as a
motivation for reporting local and global organization separately, **not**
as a numerical benchmark for `Q_SV` or `L_SV`.

[Alfken et al., *European Physical Journal E* (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11233366/)
observed concentration- and protein/lipid-ratio-dependent changes in
reconstituted synapsin–vesicle condensate morphology. Both concentration and
vesicle/lipid abundance change across their experimental preparations; their
20–30-minute images, membrane deformation, and protein/lipid ratios are not
matched to our 28-ms, 200-vesicle simulation. The paper supplies biological
motivation for a concentration contrast, not a direct numerical target for
our connected-component fractions.

[Hoffmann et al., *Nature Communications* (2023)](https://doi.org/10.1038/s41467-023-42372-6)
provide an official [source-data workbook](https://doi.org/10.6084/m9.figshare.24179445.v1)
already audited in the private research project. Its measured synapsin
transport and condensate-diameter outcomes are not the model's graph
connectivity or vesicle-radius distribution. We do not redistribute that
third-party workbook here or portray it as direct validation of this article.

The publication figures are regenerated from the included simulation data by
`python -m analysis.paper_figures`; external numeric values enter only the
clearly labeled supplementary geometry-context panel.
