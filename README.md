# Synthesis Route Complexity as a Predictor of Reported Material Performance

A data-driven study testing whether synthesis route complexity — operationally defined
from the Kononova et al. (2019) solid-state synthesis dataset — is a statistically
significant predictor of DFT-computed material properties from the Materials Project.

**No mock or synthetic data are used at any stage.**

---

## Scientific Hypothesis

> Synthesis route complexity (number of precursors, number of steps, temperature range,
> synthesis duration, and elemental diversity) is a statistically significant predictor
> of material properties, across solid-state inorganic synthesis.

---

## Data Sources

| Source | Description |
|---|---|
| **Kononova et al., *Scientific Data* 6, 203 (2019)** | 31,782 solid-state synthesis recipes extracted from the literature by NLP |
| **Materials Project REST API v2** | DFT-computed band gap, formation energy, E-hull, and crystal density for each matched target formula |

- Dataset DOI: [10.1038/s41597-019-0224-1](https://doi.org/10.1038/s41597-019-0224-1)
- Dataset source: [CederGroupHub/text-mined-synthesis_public](https://github.com/CederGroupHub/text-mined-synthesis_public)
- MP API: [next-gen.materialsproject.org/api](https://next-gen.materialsproject.org/api)

---

## Synthesis Complexity Features

| Feature | Symbol | Definition |
|---|---|---|
| Precursor count | *n*_prec | Number of distinct precursor compounds |
| Max temperature | *T*_max | Maximum sintering/calcination temperature (°C) |
| Total heating time | *t*_total | Cumulative heating time across all operations (hours) |
| Number of steps | *n*_steps | Total number of described processing operations |
| Precursor diversity | *n*_elem | Number of distinct elements across all precursors |

A **composite complexity index** is also computed as the mean of per-feature z-scores.

---

## Materials Project Properties (outcome variables)

| Property | Unit | Description |
|---|---|---|
| `band_gap` | eV | Electronic band gap (GGA+U) |
| `formation_energy_per_atom` | eV/atom | Thermodynamic formation energy |
| `energy_above_hull` | eV/atom | Distance from the convex hull (synthesizability) |
| `density` | g/cm³ | Crystal density of the ground-state polymorph |

For each target formula the ground-state polymorph (minimum `energy_above_hull`) is selected.
MP properties are retrieved and cached locally; no re-querying is needed on subsequent runs.

---

## Statistical Methods

- **Multivariate OLS** with z-score standardised features and HC3 heteroscedasticity-robust SEs
- **Pearson correlation matrix** with significance annotation
- **Variance Inflation Factors** (VIF) for multicollinearity diagnostics
- **Subgroup OLS** by dominant-cation material family (band gap as primary target)

---

## Key Results (n = 4,968 matched records)

| Property | R² | Top predictor |
|---|---|---|
| Band Gap (eV) | **0.152** | Precursor Diversity (β = +0.52, p<0.001) |
| Formation Energy (eV/atom) | **0.249** | Max Temperature (β = −0.37, p<0.001) |
| Energy Above Hull (eV/atom) | 0.014 | Precursor Diversity (β = −0.002, p<0.001) |
| Crystal Density (g/cm³) | **0.182** | Precursor Diversity (β = −0.60, p<0.001) |

Subgroup R² for band gap reaches **0.73** (Mn-oxides) and **0.61** (Co-oxides).

---

## Project Structure

```
.
├── main.py                      # Pipeline entry point
├── requirements.txt
├── src/
│   ├── data_loader.py           # Dataset loading and parsing
│   ├── feature_extractor.py     # Complexity feature computation
│   ├── materials_project.py     # MP API batch queries + local cache
│   ├── analysis.py              # OLS, correlation, VIF, subgroup regressions
│   └── visualization.py        # Publication-quality figures
├── data/
│   ├── solid_state_synthesis.json      # Kononova dataset (download separately)
│   └── mp_properties_cache.json        # Local MP property cache (auto-generated)
└── output/
    ├── figures/                 # PNG figures (Fig 1–5, S1)
    └── results/                 # CSV tables + RESULTS_SUMMARY.md
```

---

## Output Figures

| Figure | Description |
|---|---|
| `fig1_feature_distributions.png` | KDE histograms of all five complexity features |
| `fig2_correlation_heatmap.png` | Pearson correlation: features + all four MP properties |
| `fig3_property_scatter.png` | Each MP property vs. most predictive complexity feature |
| `fig4_regression_coefficients.png` | OLS coefficients + 95% CI, one panel per property |
| `fig5_subgroup_r2.png` | R² by material family (band gap model) |
| `figS1_mp_property_distributions.png` | MP property distributions in the merged dataset |

---

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset
# Place solid-state_dataset_20200713.json.xz from:
# https://github.com/CederGroupHub/text-mined-synthesis_public
# into data/ and decompress with: xz -d data/solid-state_dataset_20200713.json.xz

# Run the full pipeline
MP_API_KEY="your_key_here" python main.py

# Or pass the key explicitly
python main.py --mp-key your_key_here
```

MP properties are cached to `data/mp_properties_cache.json` after the first run,
so subsequent runs are instant for the query step.

---

## Citation

If you use this analysis, please cite:

> Kononova, O., Huo, H., He, T., Rong, Z., Tshitoyan, V., Manor, A., & Ceder, G. (2019).
> Text-mined dataset of inorganic materials synthesis recipes.
> *Scientific Data*, 6, 203. https://doi.org/10.1038/s41597-019-0224-1

> Jain, A. et al. (2013). Commentary: The Materials Project: A materials genome approach
> to accelerating materials innovation. *APL Materials*, 1(1), 011002.
> https://doi.org/10.1063/1.4812323
