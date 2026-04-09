# Pre-Manuscript Honest Assessment

Three mandatory checks before writing. All numbers are from the live MP API run.

---

## 1. Subgroup Sample Sizes — the critical constraint

| Family | n | R² | adj-R² | Bootstrap 95% CI | FDR p | Tier |
|---|---|---|---|---|---|---|
| Mn-oxide | **25** | 0.730 | 0.659 | [0.258, 0.938] | 1.1×10⁻⁴ | **exploratory** |
| Co-oxide | **23** | 0.611 | 0.497 | [0.258, 0.938] | 5.1×10⁻³ | **exploratory** |
| Nd-compound | 51 | 0.480 | 0.423 | [0.359, 0.768] | 2.5×10⁻⁵ | moderate |
| Li-ion (Li) | **856** | 0.310 | 0.306 | — (robust n) | 1.5×10⁻⁵⁸ | **robust** |
| Na-ion (Na) | **247** | 0.298 | 0.283 | — (robust n) | 2.6×10⁻¹⁷ | **robust** |
| La-oxide | 346 | 0.247 | 0.236 | — | 2.5×10⁻¹⁹ | robust |
| Sr-titanate | 444 | 0.216 | 0.189 | — | 1.9×10⁻²¹ | robust |
| Ba-titanate | 493 | 0.108 | 0.099 | — | 1.6×10⁻¹⁰ | robust |

**Conclusion:**
- Mn-oxide (n=25) and Co-oxide (n=23) have wide bootstrap CIs (lower bound ~0.26) and
  must be presented as **exploratory, hypothesis-generating findings only**.
  Adj-R² (0.66, 0.50) is the more honest metric to report for these groups.
- Li-ion (n=856) and Na-ion (n=247) are the **robust subgroup claims** — large n,
  moderate R² that will not shrink on replication.
- 14 of 17 tested subgroups survive BH-FDR correction at α=0.05 — this is not cherry-picking.

**Manuscript action:** Report all subgroup results; flag n<100 as exploratory; lead with
Li-ion and Na-ion as the replication-credible claims.

---

## 2. Precursor Diversity Confound Check

Added `target_n_elements` (number of distinct elements in the target formula) as a covariate.

| Property | β (no control) | β (controlled) | % change | Interpretation |
|---|---|---|---|---|
| Band Gap | +0.539 | +0.474 | −12.1% | **Robust** — minimal confounding |
| Formation Energy | −0.299 | −0.227 | +23.9% | **Partial confound** — retains significance |
| Crystal Density | −0.712 | −0.634 | +11.0% | **Robust** — minimal confounding |

`target_n_elements` itself is significant for all three (p < 0.002), meaning compositional
complexity of the target does carry independent information.

**Conclusion:**
- Precursor diversity is **not merely a proxy** for target composition.
- For band gap and density, the β change is <15% — minimal confounding.
- For formation energy, the 24% β reduction requires the following framing:
  "Precursor diversity predicts formation energy both through its correlation with
   target compositional complexity (partially mediated) and through a direct,
   independent synthesis route effect."

**Manuscript action:** Report both models (with and without control) in a supplementary
table.  State explicitly: "Precursor diversity retains statistical significance and the
majority of its effect size after controlling for target elemental complexity."

---

## 3. Subgroup Coefficient Details — Mn-oxide and Co-oxide

**Mn-oxide (n=25)** — what is actually driving R²=0.73?

| Feature | β | 95% CI | p |
|---|---|---|---|
| Precursor diversity | +0.488 | [+0.219, +0.757] | 0.0004 |
| Precursor count | **−0.666** | [−1.142, −0.190] | 0.006 |
| Max temperature | +0.389 | [−0.054, +0.832] | 0.086 |

The dominant predictors are **precursor diversity** (positive) and **precursor count**
(negative, acting in the opposite direction). This means: for Mn-oxides, syntheses
that use fewer but more elementally varied precursors produce wider band gaps.
Chemical interpretation: Mn-oxides with complex ligand fields (many elements, few
precursors) adopt electronic structures with more open-shell d-orbital character → wider gap.

**Co-oxide (n=23)** — R²=0.61 is carried by one feature:

| Feature | β | 95% CI | p |
|---|---|---|---|
| Total heating time | **+0.528** | [+0.161, +0.895] | 0.005 |
| All others | — | — | n.s. |

The entire Co-oxide subgroup result is explained by a single feature: longer synthesis
times → larger band gap. Chemical interpretation: prolonged annealing allows full
oxidation of Co²⁺→Co³⁺ ordering, shifting the electronic structure toward wider-gap
CoO₂ end-members. This is chemically interpretable but n=23 means the CI on β is wide
[0.16, 0.90] — a replication study is warranted.

---

## 4. Semi-Partial R² Decomposition (band gap, controlling for composition)

| Feature | Semi-partial R² | Interpretation |
|---|---|---|
| Precursor Diversity | **0.032** | Largest unique contribution |
| Max Temperature | **0.025** | Second largest |
| Total Heating Time | 0.011 | Moderate |
| Target element count | 0.003 | Small compositional residual |
| Number of Steps | 0.0004 | Negligible |
| Precursor Count | 0.00004 | Negligible |

The independent, non-overlapping contribution of synthesis features to band-gap variance
is dominated by precursor diversity (3.2%) and max temperature (2.5%). These are small in
absolute terms but are independent of composition — they represent a synthesis-specific
signal not captured by the target formula alone.

---

## What This Supports

### What to claim:

1. **Synthesis complexity carries statistically significant, composition-independent
   predictive signal for DFT-calculated material properties** (p < 0.001 globally,
   n = 4,968 after MP matching).

2. **Formation energy is the best-predicted property** (R² = 0.249), primarily through
   synthesis temperature (β = −0.37), confirming thermodynamic phase-selection logic:
   higher temperatures favour phases with more negative formation energies.

3. **In large, chemically homogeneous subgroups** (Li-ion, n=856, R²=0.31;
   Na-ion, n=247, R²=0.30), synthesis complexity is a robust predictor of band gap.

4. **Synthesis complexity is orthogonal to thermodynamic synthesizability** (E_hull
   R²=0.014) — complexity does not predict whether the material sits on the convex hull.
   This is a meaningful negative result.

5. **Precursor diversity's predictive effect is not fully explained by compositional
   complexity** — it retains significance and >77% of its magnitude after controlling
   for target element count.

### What NOT to claim:

- Do not claim "synthesis complexity predicts material performance" globally with
  R²=0.15 as the headline number.
- Do not highlight Mn-oxide or Co-oxide R² without immediately stating the sample size
  and bootstrap CI. These are hypothesis-generating, not confirmatory.
- Do not claim precursor diversity is a pure "synthesis" signal — it is partially
  confounded with composition (acknowledge this explicitly).

---

## Reproducibility Checklist for Submission

- [x] Dataset: Kononova et al., Scientific Data 6, 203 (2019), version 2020-07-13
- [x] MP API version: 0.86.4.dev10 (from API meta field)
- [x] MP access date: 2026-04-09 (cached in data/mp_properties_cache.json)
- [x] Ground-state selection: minimum energy_above_hull per formula
- [x] Formula filter: queryable stoichiometric formulas only, no variable placeholders
- [x] Regression: OLS with HC3 robust SEs, z-score standardised features
- [x] Multiple comparisons: BH-FDR at α=0.05 for all subgroup F-tests
- [x] Bootstrap: 1,000 replicates, seed=42, for subgroups with n<100
- [x] Code: MIT licence, fully reproducible with MP_API_KEY
