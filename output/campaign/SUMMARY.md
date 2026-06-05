# Calibration campaign — results summary

Interim screening resolution (NOT thesis-final). Columns: D = validated true loss; KS/V/ACF1/ACF2/Hill = Franke-standardised component deltas (sampling-SDs off); kurtosis = diagnostic only. key theta = stage-2 optimum. R2 = held-out D-surrogate accuracy (>~0.5 trustworthy). Each experiment's behaviour is gated behind an env flag, default OFF, so E0 is the recoverable baseline.

| EID | regime | D | KS | V | ACF1 | ACF2 | Hill | kurtosis | key theta | notes |
|-----|--------|---|----|---|------|------|------|----------|-----------|-------|
| E0 | calm | 44.02 | 16.803 | 7.123 | 4.600 | 14.365 | 1.126 | 6.8 | `ft_sigma_c=0.905, zi_alpha=0.448, zi_delta=0.483` | R2=0.81; depth~244; baseline control (no changes) |
| E0 | stressed | 13.84 | 6.723 | 0.251 | 0.597 | 3.714 | 2.556 | 35.0 | `ft_sigma_c=0.746, zi_alpha=0.196, zi_delta=0.16, p_zi=0.162` | R2=0.77; depth~274; baseline control (no changes) |
| E2 | calm | 41.70 | 16.985 | 8.034 | 2.869 | 12.648 | 1.161 | 7.3 | `ft_sigma_c=0.812, zi_alpha=0.435, zi_delta=0.401, mt_lambda=0.0179` | R2=0.78; depth~256; mt_lambda in the loop (0.004-0.20) |
| E2 | stressed | 11.93 | 5.515 | 0.754 | 1.533 | 3.221 | 0.908 | 39.6 | `ft_sigma_c=0.524, zi_alpha=0.193, zi_delta=0.28, p_zi=0.161, mt_lambda=0.0411` | R2=0.74; depth~255; mt_lambda in the loop (0.004-0.20) |
| E3 | calm | 41.23 | 8.560 | 3.251 | 7.390 | 13.948 | 8.079 | 11.8 | `ft_sigma_c=1.54, zi_alpha=0.314, zi_delta=0.126` | R2=0.69; depth~329; mt_lambda pinned ~3h half-life |
| E3 | stressed | 14.38 | 5.294 | 1.832 | 2.683 | 2.634 | 1.932 | 55.4 | `ft_sigma_c=0.727, zi_alpha=0.154, zi_delta=0.17, p_zi=0.152` | R2=0.74; depth~253; mt_lambda pinned ~3h half-life |
| E4a | calm | 43.11 | 15.909 | 7.233 | 3.468 | 13.963 | 2.541 | 10.1 | `ft_sigma_c=0.653, zi_alpha=0.32, zi_delta=0.343` | R2=0.58; depth~260; 30 short-lambda MT, long lambda |
| E4a | stressed | 16.68 | 8.192 | 0.850 | 0.637 | 4.198 | 2.798 | 31.0 | `ft_sigma_c=0.572, zi_alpha=0.166, zi_delta=0.188, p_zi=0.181` | R2=0.83; depth~272; 30 short-lambda MT, long lambda |
| E4b | calm | 44.46 | 17.914 | 10.927 | 1.483 | 12.600 | 1.534 | 8.9 | `ft_sigma_c=0.662, zi_alpha=0.232, zi_delta=0.242` | R2=0.62; depth~262; two-cohort MT 15 long + 15 short |
| E4b | stressed | 16.80 | 7.789 | 0.924 | 0.732 | 4.456 | 2.894 | 30.2 | `ft_sigma_c=0.572, zi_alpha=0.166, zi_delta=0.188, p_zi=0.181` | R2=0.84; depth~280; two-cohort MT 15 long + 15 short |
| E6 | stressed | — |  |  |  |  |  |  | `gaps vs shock c-sweep` | see E6_gaps_vs_shock.csv. gapped: cl_def~9.1 cm_def~1.2 maxWF~1.4; shock: cl_def~29.1 cm_def~4.1 maxWF~4.0 |
| E5 | calm | 38.41 | 18.094 | 9.137 | 1.989 | 9.166 | 0.019 | 5.9 | `ft_sigma_c=0.607, zi_alpha=0.439, zi_delta=0.128, ft_alpha=0.765, mt_alpha=0.272, ft_delta=0.327, mt_delta=0.327` | R2=0.76; depth~373; FT/MT Bernoulli gate + cancellation (high-dim) |
| E5 | stressed | 14.70 | 6.625 | 2.027 | 2.897 | 2.805 | 0.347 | 68.2 | `ft_sigma_c=0.63, zi_alpha=0.176, zi_delta=0.373, p_zi=0.154, ft_alpha=0.816, mt_alpha=0.258, ft_delta=0.167, mt_delta=0.176` | R2=0.83; depth~197; FT/MT Bernoulli gate + cancellation (high-dim) |
| E1 | calm | 44.45 | 18.064 | 7.419 | 4.958 | 10.273 | 3.739 | 13.7 | `ft_sigma_c=0.516, zi_alpha=0.243, zi_delta=0.126` | R2=0.58; depth~285; clearing/CCP tier active in the loop |
| E1 | stressed | 27.98 | 14.277 | 0.604 | 1.721 | 4.604 | 6.774 | 93.9 | `ft_sigma_c=0.636, zi_alpha=0.0887, zi_delta=0.102, p_zi=0.153` | R2=0.84; depth~224; clearing/CCP tier active in the loop |

_Campaign finished in 2.56h at 2026-06-04 23:55._
