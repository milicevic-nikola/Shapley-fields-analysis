# Shapley Fields in the Mouse Olfactory Bulb

Code accompanying

> N. Milićević, S. D. Burton, M. Wachowiak, V. Itskov. **Shapley Fields Reveal Chemotopic Organization in the Mouse Olfactory Bulb Across Diverse Chemical Feature Sets.** *PLOS Computational Biology* (accepted). Preprint: [bioRxiv 10.1101/2025.02.26.640432](https://doi.org/10.1101/2025.02.26.640432)

The olfactory bulb has no natural coordinate system for chemical stimuli. This repository implements **Shapley fields**: for each glomerulus, we train a random-forest classifier that predicts its binary responses to odorants from chemical descriptors, and compute the SHAP value of every descriptor. Laying those importances out over the glomerular map gives, for each descriptor, a spatial "field" over the dorsal bulb — an olfactory analogue of a retinotopic receptive field.

The code uses these fields to ask:

- **Chemotopy vs. chemotuning.** Does knowing *where* a glomerulus is help predict its responses, beyond its own descriptor tuning?
- **Spatial structure.** Are Shapley fields spatially organized (Moran's I, Geary's C, clustering on the bulb) compared with shuffled controls?
- **Feature-set robustness.** Do different descriptor sets (PubChem/CACTVS, a GNN perceptual embedding, Dragon, SMARTS, MACCS) give the same organization? Compared via principal angles / Grassmannian distances between Shapley-field subspaces.
- **Chemical classes.** Do principal components of the Shapley fields align with standard chemical classes (acids, esters, pyrazines, …)?
- **Concentration.** How do Shapley fields change between high and low odorant concentrations?

---

## Data

Widefield epifluorescence recordings of the dorsal olfactory bulb (OMP-driven indicator), 59 odorants plus 2 controls. The raw `.mat` files are **not** included in this repository; contact the authors for access.

Each response file stores the glomerulus × odorant matrix in `ORdata.RespMatrix`; each map file stores glomerular ROI masks (256 × 256) in `MapsData.roi.mask`. Odorant names are in `master_odorlist_omp83-85subset.txt`.

### With amines (all 59 odorants) — `resp_data/all_odors/`

| Bulb | Response matrix | ROI masks |
|---|---|---|
| omp83L | `omp83Lhigh_ORdata_dFregistered_checked_aminswitch112622.mat` | `omp83L_high_MapsData_mwsel_aminswitch.mat` |
| omp83R | `omp83Rhigh_ORdata_dFregistered_checkedaminswitch112622.mat` | `omp83R_high_MapsData_mwsel_aminswitch.mat` |

### Without amines (52 odorants), high and low concentration — `resp_data/no_amines/`

| Bulb | Response matrix | ROI masks |
|---|---|---|
| omp84 high | `omp84high_ORdata_dFregistered_checked_noamines112622.mat` | `omp84highMapsData_mwsels_1_2sec_100222.mat` |
| omp85 high | `omp85high_ORdata_mwsel_dFregistered_checked_noamines112622.mat` | `omp85highMapsData1_2sec_mwsel_163_112022.mat` |
| omp84 low | `omp84low_ORdata_dFregistered_checked_noamines112622.mat` | `omp84lowMapsData_mwsels_1_2sec_100222.mat` |
| omp85 low | `omp85low_ORdata_mwsel_dFregistered_checked_noamines112622.mat` | `omp85lowMapsData1_2sec_mwsel_163_112022.mat` |

Amines are odorants 26–32 in the 59-odorant ordering.

### Chemical descriptors

| Name in code | Description |
|---|---|
| `cact` | PubChem / CACTVS descriptors (`cactvs_our_odors.csv`) |
| `pca_cact` | PCA-reduced CACTVS |
| `pom21` | Graph-neural-network embedding trained on human odor perception (`learned_descrs_21odor_labels.csv`) |
| `haddad32` | 32 Dragon descriptors (Haddad et al.) |
| `edragon1377` | Full eDragon set |
| `smarts42` | 42 SMARTS substructure counts |
| `maccs` | MACCS keys |
| `decorr09` | Chemsense + CACTVS after removing descriptors with correlation > 0.9 |

---

## Pipeline

```
raw .mat files
   │  Load_the_data_from_the_files*.jl          (read responses + ROI masks)
   ▼
produce_data_for_response_prediction_and_clustering*.jl
   │  → binary responses, glomerulus positions/areas, masks, descriptors (.jld)
   ▼
model training + SHAP (server scripts, Distributed.jl, 40 workers)
   │  compute_shapley_fields.jl                 → Shapley fields per glomerulus
   │  new_chemotuning_server.jl, shap_ordered_metrics*_server.jl
   │                                            → prediction curves (PR-AUC / ROC-AUC)
   │  shapley_rank_comparisons_server.jl        → SHAP ranking stability (Kendall τ)
   │  find_statistically_significant_shap_values_on_bulb_stat_test.jl
   │                                            → permutation p-values
   │  Grassmanian_distance_analysis.jl          → shuffled-null subspace distances
   ▼
analysis + figures (run interactively)
      Fig2_plots.jl, Fig3_plots.jl, shap_values_clustering_high_and_low_conc.jl,
      explaining_glom_odorant_pair_predictions.jl, shapley_chem_fields.m
```

### Models

Both model families are scikit-learn `RandomForestClassifier`s (called from Julia via PyCall) with `class_weight="balanced_subsample"`.

- **Chemotopy model** (`*_10_fold`). One forest over all (glomerulus, odorant) pairs, with features = odorant descriptors + glomerulus position (x, y) and area. Evaluated by 10-fold cross-validation over pairs.
- **Chemotuning model** (`*_leave_one_out_per_glom`). One forest per glomerulus, descriptors only. Evaluated by leaving out one odorant at a time.

Descriptors are ranked by mean |SHAP|, and models are retrained on the top 1, 2, …, *n* descriptors to produce performance-vs-number-of-features curves. Metric tuples store PR-AUC at index 2 and ROC-AUC at index 4.

`new_chemotuning_server.jl` separates *spatial* information from glomerulus *identity*: it adds a random per-glomerulus ID column and, as a control, permutes positions consistently across glomeruli, so position still identifies a glomerulus but no longer encodes its neighbours.

### Shapley fields

`compute_shapley_fields.jl` fits a per-glomerulus forest and stores its SHAP vector, split into positive and negative parts (so 2 × *d* features for *d* descriptors). Stacking glomeruli gives the matrix `shap_per_glom_matrix` (glomeruli × 2*d*). `shaps_on_bulb_matrices` paints each column onto the ROI masks to produce the field images.

### Statistics

- **Permutation test** for Shapley values: each glomerulus's response vector is shuffled across odorants (1000×); one-sided p-values; glomerulus × descriptor significance relations at high/low concentration (consistent, increasing, decreasing).
- **Spatial organization**: Ward clustering of glomeruli on Shapley vectors (cosine distance), with Moran's I and Geary's C against 500 shuffles.
- **Feature-set comparison**: PCA of the Shapley fields for two descriptor sets; geodesic, chordal, Binet–Cauchy and Fubini–Study distances between the PC subspaces, compared against a shuffled-odorant null.

---

## Repository layout

| File | Purpose |
|---|---|
| `Initialize_analyses.jl` | Packages, data loaders (`load_data_with_amines`, `load_data_no_amines_new`), descriptor loaders, normalization, mask/window utilities, clustering and plotting helpers |
| `Useful_functions.jl` | Small helpers: top-*k* `findmax`/`findmin`, Grassmannian distances via `scipy.linalg.subspace_angles` |
| `More_functions.jl` | Gaussian-smoothed chemical fields (`compute_chem_fields`) |
| `python_shap_functions_new.jl` | Random forest + SHAP wrappers (training, orderings, cross-validated metrics) |
| `Load_the_data_from_the_files.jl` | Loader for omp83L / omp83R (all odorants) |
| `Load_the_data_from_the_files_no_amines.jl` | Loader for omp84 / omp85, high concentration |
| `Load_the_data_from_the_files_high_and_low_conc_no_amines.jl` | Loader for omp84 / omp85, high and low concentration |
| `produce_data_for_response_prediction_and_clustering*.jl` | Builds the processed `.jld` datasets |
| `compute_shapley_fields.jl` | Shapley fields per glomerulus |
| `new_chemotuning_server.jl` | Chemotopy vs. identity experiment |
| `shap_ordered_metrics_server.jl`, `shap_ordered_metrics_descrs_comparison_server.jl` | Prediction curves and descriptor-set baselines |
| `shapley_rank_comparisons_server.jl`, `shap_values_rankings.jl` | SHAP ranking stability |
| `find_statistically_significant_shap_values_on_bulb_stat_test.jl` | Permutation test |
| `Grassmanian_distance_analysis.jl` | Null distribution of subspace distances |
| `shap_values_clustering_high_and_low_conc.jl` | Concentration shifts, clustering, Moran's I / Geary's C |
| `explaining_glom_odorant_pair_predictions.jl` | Local SHAP waterfall for a single glomerulus–odorant prediction |
| `Fig2_plots.jl`, `Fig3_plots.jl` | Manuscript figures |
| `shapley_chem_fields.m` | MATLAB rendering of Shapley fields exported to `shap_chem_fields.mat` |

---

## Requirements

**Julia** (1.6+): `JLD`, `MAT`, `Distributed`, `IterTools`, `MultivariateStats`, `Clustering`, `Distances`, `DataFrames`, `StatsBase`, `Statistics`, `Distributions`, `Plots`, `StatsPlots`, `PyPlot`, `ColorSchemes`, `Colors`, `TSne`, `Measures`, `AngleBetweenVectors`, `PyCall`.

**Python** (used through PyCall): `scikit-learn`, `shap`, `scipy`, `numpy`.

**MATLAB** (optional): only for `shapley_chem_fields.m`.

```julia
using Pkg
Pkg.add(["JLD","MAT","IterTools","MultivariateStats","Clustering","Distances",
         "DataFrames","StatsBase","Distributions","Plots","StatsPlots","PyPlot",
         "ColorSchemes","Colors","TSne","Measures","AngleBetweenVectors","PyCall"])
ENV["PYTHON"] = "/path/to/python"   # environment with scikit-learn, shap, scipy
Pkg.build("PyCall")
```

## Running

1. Place the raw data under `resp_data/all_odors/` and `resp_data/no_amines/`, and update the hard-coded paths at the top of each script (they currently point to `~/Dropbox/Nikola-Vladimir/...` or `~/Documents/Nikola-Vladimir/...`).
2. Build the processed datasets with the `produce_data_*` scripts.
3. Run the server scripts, e.g.
   ```bash
   julia compute_shapley_fields.jl
   ```
   Each starts 40 worker processes (`addprocs(40)`); reduce this on smaller machines. Choose the descriptor set and dataset with `descr_set_name` and `dataset_name` near the top of each script.
4. Make figures with `Fig2_plots.jl` and `Fig3_plots.jl`. These were written to be run block by block in an interactive session; several blocks expect you to reload results for a different descriptor set between steps.

## Notes

- The figure and analysis scripts are research code, run interactively; they are not meant to execute top to bottom.
- `Grassmanian_distance_analysis.jl` defines its own distance functions using angles between corresponding PC columns, which override the principal-angle versions in `Useful_functions.jl`. Use one definition consistently for both the real and null values.
- The loaders select odorants by column position; check that the amine columns (26–32) are where expected when adding new data files.

## Citation

```bibtex
@article{milicevic_shapley_fields,
  author  = {Mili{\'c}evi{\'c}, Nikola and Burton, Shawn D. and Wachowiak, Matt and Itskov, Vladimir},
  title   = {Shapley Fields Reveal Chemotopic Organization in the Mouse Olfactory Bulb Across Diverse Chemical Feature Sets},
  journal = {PLOS Computational Biology},
  note    = {Accepted. Preprint: bioRxiv 10.1101/2025.02.26.640432},
  doi     = {10.1101/2025.02.26.640432}
}
```

## License

<!-- e.g. MIT -->

## Contact

Nikola Milićević — University of Texas at El Paso
Source code for producing the figures in the manuscript
