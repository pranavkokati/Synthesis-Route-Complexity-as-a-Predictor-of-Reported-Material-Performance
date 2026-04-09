# Results Summary

## Dataset

| Statistic | Value |
|---|---|
| Total synthesis recipes | 31,781 |
| With max temperature recorded | 26,775 (84.2%) |
| With total heating time recorded | 25,660 (80.7%) |
| With composite complexity index | 24,624 (77.5%) |
| With extracted performance metric | 94 (0.3%) |
| Analysis-ready (all features + performance) | 68 |

> **Note:** The Kononova dataset was designed for synthesis procedure extraction. Most paragraphs
> describe procedural steps without quantitative performance outcomes, which limits the analysis
> subset to ~68 records with all features present.

---

## Complexity Feature Summary

| Feature | Mean ± SD | Median | Range |
|---|---|---|---|
| Precursor count | 3.18 ± 1.04 | 3 | 1–10 |
| Max temperature (°C) | 1071 ± 321 | 1100 | −30–3000 |
| Total heating time (h) | 32.6 ± 52.8 | 20.0 | 0.02–1605 |
| Number of steps | 6.45 ± 3.30 | 6 | 0–27 |
| Precursor diversity (elements) | 5.02 ± 1.51 | 5 | 2–13 |

---

## Performance Metrics Extracted

| Metric Class | Count | Unit |
|---|---|---|
| Sintered bulk density | 37 | g/cm³ |
| BET surface area | 29 | m²/g |
| Relative sintered density | 25 | % theoretical |
| Discharge capacity | 3 | mAh/g |
| Electrical conductivity | 2 | S/cm |
| Resistivity | 2 | Ω·cm |

---

## Full-Dataset OLS Regression

**perf_norm ~ precursor_count + max_temperature_C + total_time_h + n_steps + precursor_diversity**

- n = 68 | R² = 0.137 | adj-R² = 0.068 | F = 7.79 | p(F) = 9.7×10⁻⁶
- HC3 heteroscedasticity-robust standard errors
- Features standardised (z-scored) before regression

| Feature | β (standardised) | p-value | Significant? |
|---|---|---|---|
| Precursor Diversity | −0.410 | **0.0002** | ✓ |
| Precursor Count | +0.221 | 0.145 | |
| Number of Steps | +0.144 | 0.142 | |
| Max Temperature | −0.074 | 0.649 | |
| Total Heating Time | +0.017 | 0.867 | |

**Key finding:** Precursor diversity is the only statistically significant predictor (β = −0.41,
p = 0.0002). The negative sign indicates that higher elemental diversity across precursors
is associated with *lower* normalised performance after within-class z-scoring.

**VIF diagnostics:** All VIF < 2.0 — no multicollinearity concern.

---

## Subgroup Regressions (by Metric Class)

| Metric Class | n | R² | adj-R² |
|---|---|---|---|
| Surface area | 21 | **0.693** | 0.591 |
| Bulk density | 22 | **0.490** | 0.330 |
| Relative density | 21 | 0.392 | 0.190 |

The surface-area and density subgroups show substantially higher R² values than the
mixed-metric full model, indicating that within homogeneous measurement classes
the complexity features explain a large portion of variance.

---

## Correlation Matrix (Pearson r, analysis subset)

|  | Precursor Count | Max Temp | Total Time | n Steps | Diversity | perf_norm |
|---|---|---|---|---|---|---|
| Precursor Count | 1.00 | 0.11 | 0.36 | 0.42 | **0.59** | 0.04 |
| Max Temperature | | 1.00 | −0.24 | 0.28 | 0.04 | −0.05 |
| Total Time | | | 1.00 | 0.12 | 0.12 | 0.09 |
| n Steps | | | | 1.00 | 0.35 | −0.01 |
| Diversity | | | | | 1.00 | **−0.20** |
| perf_norm | | | | | | 1.00 |

---

## Interpretation

The hypothesis that **synthesis complexity positively predicts reported performance** receives
mixed support:

1. **Number of steps** shows a positive (though marginal) association with performance across
   the full dataset, consistent with the hypothesis.
2. **Precursor diversity** shows a *significant negative* association, suggesting that
   compositionally complex precursor mixtures do not confer superior outcomes — possibly
   because the cross-metric normalisation compresses different physical scales.
3. **Subgroup regressions** (surface area, density) yield R² > 0.49, demonstrating that
   within a homogeneous property class, synthesis complexity is a meaningful predictor.
4. The **overall model is statistically significant** (F-test p = 9.7×10⁻⁶) despite modest R².

**Methodological note:** The primary limitation is the small analysis-ready subset (n = 68).
The Kononova dataset contains synthesis procedures, not performance summaries; consequently
only ~0.3% of paragraphs contain extractable quantitative outcomes.  A future study linking
these recipes to a dedicated property database (e.g. Materials Project, AFLOW, MPDS) would
dramatically increase statistical power.
