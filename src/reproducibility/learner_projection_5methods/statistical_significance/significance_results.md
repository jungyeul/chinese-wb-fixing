# Paired Significance Test for Learner Benchmark F1

Input files:

- `../csv/LTP_projection.csv`
- `../csv/step1_projection.csv`
- `../csv/step2_projection.csv`

Method:

- Reconstructed per-sentence token counts from official projection CSVs.
- Verified raw source strings match between gold and prediction for all 1,079 sentences.
- Verified corpus token F1 matches the paper table before testing.
- Used paired sentence bootstrap for 95% confidence intervals over F1 differences.
- Used paired approximate randomization over sentence-level outcomes for two-sided p-values.
- Iterations: 50,000.
- Random seed: 20260522.

## Corpus F1 Check

| Method | Correct | Predicted | Gold | Token F1 |
|---|---:|---:|---:|---:|
| D | 32,253 | 32,681 | 32,536 | 0.989097934588 |
| P1 | 32,229 | 32,577 | 32,536 | 0.989940564864 |
| P2 | 32,268 | 32,547 | 32,536 | 0.991595347479 |

## Paired Tests

| Comparison | Delta F1 | Delta percentage points | 95% bootstrap CI for delta | Approx. randomization p-value |
|---|---:|---:|---:|---:|
| D -> P1 | 0.000842630277 | 0.084263 | [-0.000635759710, 0.002362534967] | 0.26903462 |
| P1 -> P2 | 0.001654782615 | 0.165478 | [0.000854323697, 0.002473131316] | 0.00006000 |
| D -> P2 | 0.002497412892 | 0.249741 | [0.000884235658, 0.004126551784] | 0.00225995 |

## Interpretation

The improvement from `P1` to `P2` is statistically reliable under both tests.
The improvement from `D` to `P2` is also statistically reliable.
The smaller improvement from `D` to `P1` is not significant under this paired test.

Possible paper sentence:

> Paired sentence-level tests confirm that the full projection model significantly improves over both direct segmentation and identical-character projection: compared with `P1`, `P2` improves token F1 by 0.00165 (95% bootstrap CI [0.00085, 0.00247], approximate randomization `p < 0.001`), and compared with `D`, it improves by 0.00250 (95% CI [0.00088, 0.00413], `p = 0.0023`). The smaller gain from `D` to `P1` is not significant (`p = 0.269`).
