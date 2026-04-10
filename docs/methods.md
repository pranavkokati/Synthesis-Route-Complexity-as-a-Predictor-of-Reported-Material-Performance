# Methods

## 1. Data Acquisition

Crystal structures are retrieved from the Materials Project (MP) database
via the `mp-api` Python client (v≥0.40). The MP database contains DFT-relaxed
structures with thermodynamic properties computed using the Vienna Ab initio
Simulation Package (VASP) with PBE-GGA exchange-correlation functional,
projector-augmented wave (PAW) pseudopotentials, and a plane-wave energy
cutoff of 520 eV.

**Query filters:**
- 2–6 distinct element types (`num_elements`)
- 4–80 atoms per unit cell (`num_sites`)
- All crystal systems included

**Synthesizability labels** are defined from the energy above the convex hull
(`energy_above_hull`, eV/atom):

| Class | Label | Criterion |
|-------|-------|-----------|
| 0 | Stable | E_hull = 0 |
| 1 | Metastable | 0 < E_hull ≤ 0.1 eV/atom |
| 2 | Unstable | E_hull > 0.1 eV/atom |

The 0.1 eV/atom threshold follows the literature convention for kinetic
accessibility of metastable phases (Sun et al., *Science Advances*, 2016).

**Binary label** for logistic regression: `synthesizable = 1` if
`theoretical = False` (structure has an ICSD reference), else 0.

**Stratified sampling:** Equal numbers of structures are drawn from each class
using a reproducible random seed (default: 42). This ensures balanced class
representation and simplifies statistical comparison.

---

## 2. Coordination Environment Classification

Coordination environments (CEs) are classified using the `ChemEnv` module in
`pymatgen` (Waroquiers et al., *Chemistry of Materials*, 2017).

**Algorithm:**
1. `LocalGeometryFinder` computes the local geometry at each atomic site by
   identifying neighbouring atoms within a distance factor of 1.41× the
   minimum intersite distance.
2. Each site's geometry is matched to one of ~30 reference polyhedra
   (octahedron O:6, tetrahedron T:4, cuboctahedron C:12, square plane S:4,
   linear L:2, etc.) using the continuous symmetry measure (CSM).
3. `SimplestChemenvStrategy` assigns each site to the best-matching CE symbol
   (distance cutoff 1.4, angle cutoff 0.3).

**Parallelisation:** Sites are computed in parallel using
`concurrent.futures.ProcessPoolExecutor`. A per-structure SIGALRM timeout
(default 120 s) prevents runaway jobs on pathological structures. Completed
results are saved incrementally to disk.

---

## 3. Descriptor Computation

For each structure, CE symbols are tallied across all atomic sites to produce
a frequency distribution {cₖ : k = 1…K} where cₖ is the count of sites with
CE symbol k and K is the number of distinct CE types.

Let p_k = c_k / Σc_k be the normalised probability.

**Shannon entropy:**
```
H = −Σₖ pₖ log(pₖ)   (nats; log = natural logarithm)
```
H = 0 for a single CE type (all sites identical); H = log(K) for the uniform
distribution (maximum disorder).

**Distinct environments:**
```
K = number of distinct CE symbols
```

**Dominance:**
```
d = max(pₖ)   (fraction of sites in the most common environment)
```

**Gini coefficient:**
```
G = (2/K) Σᵢ i·pᵢ⁽ˢᵒʳᵗᵉᵈ⁾ − (K+1)/K
```
where p is sorted in ascending order and i is 1-indexed.
G = 0 for uniform distribution; approaches 1 for maximum inequality.

---

## 4. Statistical Analysis

### 4.1 One-Way ANOVA

A one-way ANOVA tests whether mean CE entropy differs significantly across
the three synthesizability classes. Effect size is quantified by η² (eta-squared):

```
η² = SS_between / SS_total
```

Conventions: η² < 0.01 negligible; 0.01–0.06 small; 0.06–0.14 medium; > 0.14 large.

Post-hoc pairwise comparisons use Tukey's Honestly Significant Difference (HSD)
test with the Studentized range distribution. Cohen's d is reported for each pair.

Multiple-comparisons correction across the four descriptors (H, K, d, G) uses the
Benjamini–Hochberg procedure (false discovery rate α = 0.05).

### 4.2 Logistic Regression

Binary synthesizability (ICSD-observed vs. theoretical) is modelled with
L2-penalised logistic regression (sklearn `LogisticRegression`, `lbfgs` solver,
class-weight balanced). Features are standardised (zero mean, unit variance).

Three nested models are compared:

| Model | Features |
|-------|----------|
| Naive | nsites, nelements |
| Entropy | H, K, d, G |
| Full | H, K, d, G, nsites, nelements |

Model discrimination is quantified by ROC-AUC. Bootstrap 95% confidence
intervals on AUC are computed from 1,000 stratified resamples.

Goodness-of-fit uses McFadden's pseudo-R²:
```
R²_McF = 1 − LL(model) / LL(null)
```
where LL(null) uses the marginal class probability.

### 4.3 Spearman Correlation

Spearman's rank correlation ρ between CE entropy and the continuous
energy\_above\_hull is computed on all structures, including zeros (stable class).
The rank-based measure is preferred over Pearson's r because energy\_above\_hull
is right-skewed.

### 4.4 Composition Control

To rule out the confound that "simple elements form simple structures", a
composition-corrected entropy is computed:

```
H_corr(s) = H(s) − mean_{e ∈ elements(s)} [mean_{t: e∈elements(t)} H(t)]
```

The element-specific baseline is the mean CE entropy across all structures
containing that element. ANOVA is repeated on H_corr.

### 4.5 Subgroup Analysis

**Crystal-system subgroups:** Structures are grouped by crystal system
(cubic, tetragonal, orthorhombic, hexagonal, trigonal, monoclinic, triclinic).
ANOVA is run within each group (minimum 10 structures per class).

**Pettifor chemical-scale subgroups:** The Pettifor span of a structure is
defined as:
```
Δχ = max(mendeleev_no(e)) − min(mendeleev_no(e))   for e ∈ elements(s)
```
Structures are binned into quartiles of Δχ. ANOVA is run within each quartile.
This tests whether the entropy–synthesizability relationship is driven by
chemical diversity or is intrinsic to the geometric descriptor.

---

## 5. Reproducibility

All random operations use `numpy.random.default_rng(seed)` with fixed seed 42.
The Materials Project access date is stored in the cache file
(`data/mp_structures_cache.json`) for database-version traceability.

Software versions are pinned in `requirements.txt`.
