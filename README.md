# Synthesis Route Complexity as a Predictor of Reported Material Performance

A data-driven study testing whether synthesis route complexity — operationally defined from the Kononova et al. (2019) solid-state synthesis dataset — is a statistically significant positive predictor of reported material performance.

---

## Scientific Hypothesis

> Synthesis route complexity (number of precursors, number of steps, temperature range, synthesis duration, and elemental diversity) is a statistically significant positive predictor of the performance metrics reported for the resulting material, across solid-state inorganic synthesis.

---

## Dataset

**Kononova et al., *Scientific Data* 6, 203 (2019)**
- DOI: [10.1038/s41597-019-0224-1](https://doi.org/10.1038/s41597-019-0224-1)
- Source: [CederGroupHub/text-mined-synthesis_public](https://github.com/CederGroupHub/text-mined-synthesis_public)
- ~31,000 solid-state synthesis recipes extracted from the literature by NLP
- Each recipe contains: precursor list, synthesis operations with temperature/time conditions, target formula, and the original paragraph text

---

## Complexity Features

| Feature | Symbol | Definition |
|---|---|---|
| Precursor count | *n*_prec | Number of distinct precursor compounds |
| Max temperature | *T*_max | Maximum sintering/calcination temperature (°C) |
| Total heating time | *t*_total | Cumulative heating time across all operations (hours) |
| Number of steps | *n*_steps | Total number of described processing operations |
| Precursor diversity | *n*_elem | Number of distinct elements across all precursors |

A **composite complexity index** is computed as the mean of the per-feature z-scores.

---

## Performance Extraction

Quantitative performance metrics are extracted from synthesis paragraph text using regex pattern matching. Extracted metric classes (in priority order):

1. **Discharge/specific capacity** — mAh g⁻¹
2. **Ionic/electrical conductivity** — S cm⁻¹
3. **BET surface area** — m² g⁻¹
4. **Band gap** — eV
5. **Energy density** — Wh kg⁻¹
6. **Power density** — W kg⁻¹

Values are log₁₀-transformed and z-score normalised within each metric class to enable cross-class comparison.

---

## Statistical Methods

- **Multivariate OLS** with HC3 heteroscedasticity-robust standard errors
- **Pearson correlation matrix** with significance annotation
- **Variance Inflation Factors** (VIF) for multicollinearity diagnostics
- **Subgroup regressions** by metric class and material family

---

## Project Structure

```
.
├── main.py                      # Pipeline entry point
├── requirements.txt
├── src/
│   ├── data_loader.py           # Dataset loading and parsing
│   ├── feature_extractor.py     # Complexity feature computation
│   ├── performance_extractor.py # Regex-based performance metric extraction
│   ├── materials_project.py     # Optional MP API integration
│   ├── analysis.py              # OLS, correlation, VIF, subgroup tests
│   └── visualization.py        # Publication-quality figures
├── data/
│   └── solid_state_synthesis.json
└── output/
    ├── figures/                 # PNG figures (Fig 1–5, S1)
    └── results/                 # CSV result tables
```

---

## Output Figures

| Figure | Description |
|---|---|
| `fig1_feature_distributions.png` | KDE histograms of all five complexity features |
| `fig2_correlation_heatmap.png` | Pearson correlation matrix with significance stars |
| `fig3_performance_scatter.png` | Performance vs. top-2 predictive features with regression lines |
| `fig4_regression_coefficients.png` | OLS coefficients with 95% CI (red = p<0.05) |
| `fig5_subgroup_r2.png` | R² by material family / metric subgroup |
| `figS1_performance_by_class.png` | Box plots of performance value distributions |

---

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py

# Custom paths
python main.py --data /path/to/dataset.json --output /path/to/output
```

### Optional: Materials Project Integration

Set `MP_API_KEY` to enable E_hull retrieval:

```bash
export MP_API_KEY="your_key_here"
python main.py
```

---

## Key Results

See `output/results/` for:
- `feature_summary.csv` — descriptive statistics for each complexity feature
- `ols_full_results.csv` — regression coefficients, standard errors, p-values
- `correlation_matrix.csv` — Pearson correlation matrix
- `feature_importance.csv` — features ranked by |coefficient|
- `subgroup_results.csv` — R² for each subgroup regression

---

## Citation

If you use this analysis, please cite:

> Kononova, O., Huo, H., He, T., Rong, Z., Tshitoyan, V., Manor, A., & Ceder, G. (2019).
> Text-mined dataset of inorganic materials synthesis recipes.
> *Scientific Data*, 6, 203. https://doi.org/10.1038/s41597-019-0224-1
