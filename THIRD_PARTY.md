# Third-party dependencies

Optional dependencies are disabled in the default build.

| Dependency | Version | License | Source | SHA256 |
|---|---:|---|---|---|
| Eigen | 3.4.0 | MPL-2.0 (individual files may carry compatible notices) | `https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz` | `8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72` |

Eigen is fetched only when `QBT_ENABLE_PORTFOLIO_MATH=ON`. The archive hash is checked by CMake before extraction.

## Research references

The local QuEST implementation in `portfolio_math` was written independently from
the mathematical definitions and numerical procedure in:

- Olivier Ledoit and Michael Wolf, "Spectrum Estimation: A Unified Framework for
  Covariance Matrix Estimation and PCA in Large Dimensions", *Journal of
  Multivariate Analysis* 139 (2015), DOI: `10.1016/j.jmva.2015.04.006`.
- Olivier Ledoit and Michael Wolf, "Numerical Implementation of the QuEST
  Function", *Computational Statistics & Data Analysis* 115 (2017), DOI:
  `10.1016/j.csda.2017.06.004`.
- Olivier Ledoit and Michael Wolf, "Shrinkage Estimation of Large Covariance
  Matrices: Keep It Simple, Statistician?", *Journal of Multivariate Analysis*
  186 (2021), DOI: `10.1016/j.jmva.2021.104796` (CC BY 4.0). The
  minimum-variance angle-weight and singular null-space formulas used by
  `LW-NLS-MV-QUEST` are implemented from the equations in the article.

For centered return panels, the V1 implementation freezes the independent
convention `n_eff = row_count - 1` and `S = Y'Y / n_eff`. Both the raw row
count and effective QuEST observation count are recorded in nonlinear
shrinkage diagnostics.

The separately distributed `QuEST_v027` MATLAB package is not a build or runtime
dependency. Its source is marked "All rights reserved" and was not copied or
translated into this project.
