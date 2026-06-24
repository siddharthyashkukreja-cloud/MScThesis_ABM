# Robustness experiments — severity, close-out, mechanism

Generated 2026-06-23 13:11

## Severity sweep (reactive margin)
 severity  drawdown  client_def  member_def  reach_fund  IM_peak_B
     1.00    -35.65         7.6         1.5         0.8      39.15
     1.25    -42.34        19.9         3.2         1.0      34.81
     1.50    -48.36        22.3         3.5         1.0      30.22
     1.75    -53.73        24.6         4.4         1.0      27.56
     2.00    -58.57        29.0         4.5         1.0      25.12

## Close-out recovery sweep
 closeout_recovery margin  member_def  reach_fund  client_def
              0.70 flat08         0.4         0.3         7.9
              0.70 flat12         0.7         0.7         5.1
              0.80 flat08         0.4         0.2         7.9
              0.80 flat12         0.8         0.7         4.9
              0.90 flat08         0.4         0.0         8.0
              0.90 flat12         0.7         0.0         4.8
              0.95 flat08         0.4         0.0         8.0
              0.95 flat12         0.7         0.0         4.8

## Tail-concentration mechanism (|position| at client default)
margin  n_defaults  median    p90    p99    max  tail_gt1800
flat04         126   802.0 1338.0 3232.0 7273.0        0.024
flat08          89   813.0 1393.0 5589.0 5590.0        0.079
flat12          58   842.0 3746.0 4666.0 4745.0        0.207
