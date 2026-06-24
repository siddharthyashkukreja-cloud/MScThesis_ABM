# Overnight JOINT agent+FV calibration

Generated 2026-06-24 11:58

## Stage A - calibration
- joint optimum: mt_gamma=0.721, ft_sigma_c=0.735, f=0.881, w=0.507, ratio=0.320, sigma_d=0.001, zi_alpha=0.372, zi_mu=0.079, zi_delta=0.318
- loss D: joint=6.73 vs baseline=20.857  (1698 evals, 612.0 min)
- drift=-2.133e-05/min (~30% over 42d); jumps dropped (log-vol carries the tail)

## Stage B - validation
- joint coverage 9/10; mean |gap| joint=0.64 vs base=2.64SD; ensemble D joint=18.31 vs base=24.5

## Stage C - clearing
- 80 paths; member defaults 72; deepest waterfall 5; client defaults mean 8.4; IM peak $44.82B
