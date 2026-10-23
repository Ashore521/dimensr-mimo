# Candidate 1: regularized linear heads

Developed after inspecting v1 training/validation only. Baseline global mean
validation NRMSE was 0.7382105219. Concrete showed seed-dependent selection;
Air Quality had a substantial train/validation gap; Parkinson remained weak
under subject separation. No sealed test labels or metrics were inspected.

Add `head_ridge=0.001` on the max-absolute-value-scaled design matrix, with
penalty strength `n_train * head_ridge`. Intercepts are not penalized. The default
0 preserves original OLS behavior. Search budgets, data, normalization and seeds
are identical to v1. This is candidate 1 of the protocol's two permitted revisions.
