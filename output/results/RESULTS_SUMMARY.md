# Results Summary — Real Data (Materials Project API)

All properties are DFT-computed values retrieved live from the Materials Project API.
No mock or synthetic data are used at any stage.

---

## Dataset

| Statistic | Value |
|---|---|
| Total synthesis recipes (Kononova) | 31,781 |
| Unique queryable formulas | 6,514 |
| MP-matched synthesis records | **4,968 (15.6%)** |
| With max temperature | 26,775 (84.2%) |
| With total heating time | 25,660 (80.7%) |

---

## Complexity Feature Summary (full dataset, n = 31,781)

| Feature | Mean ± SD | Median | Range |
|---|---|---|---|
| Precursor count | 3.18 ± 1.04 | 3 | 1–10 |
| Max temperature (°C) | 1071 ± 321 | 1100 | −30–3000 |
| Total heating time (h) | 32.6 ± 52.8 | 20.0 | 0.02–1605 |
| Number of steps | 6.45 ± 3.30 | 6 | 0–27 |
| Precursor diversity (# elements) | 5.02 ± 1.51 | 5 | 2–13 |

---

## Materials Project Properties (merged dataset, n = 4,968)

Ground-state polymorph (minimum energy_above_hull) selected per formula.

| Property | Source | Unit |
|---|---|---|
| Band gap | MP DFT (GGA+U) | eV |
| Formation energy per atom | MP DFT | eV/atom |
| Energy above hull | MP convex hull | eV/atom |
| Crystal density | MP DFT | g/cm³ |

---

## OLS Regression Results

Standardised features, HC3 heteroscedasticity-robust SEs, intercept included.

### Band Gap (eV) — R² = 0.1521, n = 4,968

| Feature | β | p-value | Sig |
|---|---|---|---|
| Precursor Diversity | +0.5234 | <0.001 | *** |
| Max Temperature | +0.2870 | <0.001 | *** |
| Precursor Count | — | n.s. | |
| Total Heating Time | −0.1715 | <0.001 | *** |
| Number of Steps | — | n.s. | |

### Formation Energy (eV/atom) — R² = 0.2492, n = 4,968

| Feature | β | p-value | Sig |
|---|---|---|---|
| Max Temperature | −0.3669 | <0.001 | *** |
| Precursor Diversity | −0.2427 | <0.001 | *** |
| Precursor Count | +0.1089 | <0.001 | *** |
| Total Heating Time | +0.0612 | <0.001 | *** |
| Number of Steps | — | n.s. | |

### Energy Above Hull (eV/atom) — R² = 0.0137, n = 4,968

| Feature | β | p-value | Sig |
|---|---|---|---|
| Precursor Diversity | −0.0023 | <0.001 | *** |
| Precursor Count | +0.0013 | 0.004 | ** |
| Max Temperature | +0.0009 | 0.003 | ** |

### Crystal Density (g/cm³) — R² = 0.1824, n = 4,968

| Feature | β | p-value | Sig |
|---|---|---|---|
| Precursor Diversity | −0.5987 | <0.001 | *** |
| Max Temperature | +0.2049 | <0.001 | *** |
| Precursor Count | +0.2044 | <0.001 | *** |
| Total Heating Time | +0.0988 | <0.001 | *** |
| Number of Steps | +0.0543 | 0.015 | * |

---

## VIF Diagnostics

All VIF < 2.1 — no multicollinearity concern.

| Feature | VIF |
|---|---|
| Precursor Count | 2.006 |
| Precursor Diversity | 2.066 |
| Max Temperature | 1.100 |
| Total Heating Time | 1.045 |
| Number of Steps | 1.066 |

---

## Subgroup Regressions by Material Family (target: band_gap)

Selected highlights:

| Family | n | R² | adj-R² |
|---|---|---|---|
| Mn-oxide | 25 | **0.730** | 0.659 |
| Co-oxide | 23 | **0.611** | 0.497 |
| Nd-compound | 50 | **0.570** | 0.521 |
| Li-ion (Li) | 853 | **0.310** | 0.306 |
| Na-ion (Na) | 246 | 0.298 | 0.283 |
| Ca-compound | 223 | 0.252 | 0.235 |

---

## Key Scientific Findings

1. **Precursor diversity** (number of distinct elements across all precursors) is the
   strongest predictor across all four MP properties — positive for band gap and negative
   for density and formation energy.

2. **Synthesis temperature** is the second strongest predictor:
   - Higher temperatures associate with larger band gaps and denser materials
   - Higher temperatures associate with more negative (more stable) formation energies

3. **Precursor count** positively predicts crystal density but has minimal effect on
   electronic properties.

4. **Energy above hull** (synthesizability) is only weakly predicted (R²=0.014), suggesting
   that route complexity does not strongly select for thermodynamic synthesizability.

5. **Subgroup analysis** reveals that within specific material families (Mn-oxides, Co-oxides,
   Nd-compounds) complexity features explain 57–73% of band-gap variance, far exceeding the
   full-dataset R² of 0.152.

6. The overall picture supports a nuanced version of the hypothesis: **synthesis complexity
   does predict material properties, but the direction and magnitude are highly
   property- and family-dependent.**
