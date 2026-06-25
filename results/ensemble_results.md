# Ensemble + Model Soup Evaluation Results

Top-1 test accuracy (%) for all eight evaluation modes across 8 datasets, 5 shot settings, and 3 context lengths (M=4, 8, 16).

Methods grouped:
- **Logit-space ensembles**: CoOp (best/mean), Snapshot, Multi-Seed, Full Ensemble.
- **Weight-space soups**: Uniform Soup, Greedy Soup, Full Uniform Soup.

## Oxford Pets

### M = 4

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 90.22 | 89.67 | 89.97 | 89.86 | 92.61 |
| CoOp (mean ± std) | 89.91 | 88.93 | 89.81 | 89.74 | 91.67 |
| Snapshot Ensemble | 90.54 | 90.99 | 91.09 | 91.52 | 93.00 |
| Multi-Seed Ensemble | 91.58 | 91.71 | 92.34 | 92.31 | 93.76 |
| Full Ensemble (seeds × snapshots) | 91.63 | 92.42 | 92.78 | 93.30 | 94.09 |
| Uniform Soup (final seeds) | 91.99 | 90.81 | 91.17 | 90.62 | 86.89 |
| Greedy Soup (final seeds) | 90.22 | 90.92 | 91.17 | 89.86 | 92.61 |
| Full Uniform Soup (seeds × snapshots) | 91.82 | 91.58 | 91.58 | 91.31 | 92.59 |

### M = 8

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 90.71 | 90.62 | 90.98 | 90.11 | 91.61 |
| CoOp (mean ± std) | 90.17 | 89.81 | 90.51 | 89.94 | 91.49 |
| Snapshot Ensemble | 90.68 | 91.76 | 91.80 | 91.66 | 93.08 |
| Multi-Seed Ensemble | 92.31 | 92.80 | 93.46 | 92.67 | 93.95 |
| Full Ensemble (seeds × snapshots) | 92.15 | 93.10 | 93.92 | 93.38 | 94.39 |
| Uniform Soup (final seeds) | 91.11 | 92.45 | 92.12 | 86.13 | 86.67 |
| Greedy Soup (final seeds) | 91.80 | 92.45 | 90.98 | 90.05 | 91.61 |
| Full Uniform Soup (seeds × snapshots) | 90.57 | 92.59 | 91.80 | 91.22 | 89.04 |

### M = 16

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 89.75 | 91.33 | 90.71 | 89.48 | 92.04 |
| CoOp (mean ± std) | 88.77 | 90.29 | 90.35 | 89.32 | 91.20 |
| Snapshot Ensemble | 88.80 | 91.35 | 91.57 | 91.23 | 92.73 |
| Multi-Seed Ensemble | 91.20 | 92.83 | 93.19 | 92.42 | 93.40 |
| Full Ensemble (seeds × snapshots) | 91.31 | 92.97 | 93.59 | 93.35 | 93.95 |
| Uniform Soup (final seeds) | 87.46 | 91.31 | 91.88 | 78.58 | 78.06 |
| Greedy Soup (final seeds) | 90.68 | 91.52 | 91.25 | 89.48 | 92.04 |
| Full Uniform Soup (seeds × snapshots) | 86.94 | 91.47 | 92.67 | 89.13 | 88.47 |

## DTD

### M = 4

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 47.93 | 54.61 | 58.45 | 63.36 | 69.39 |
| CoOp (mean ± std) | 46.91 | 53.15 | 57.70 | 63.08 | 67.59 |
| Snapshot Ensemble | 48.23 | 54.63 | 59.22 | 65.03 | 69.41 |
| Multi-Seed Ensemble | 54.31 | 60.58 | 65.96 | 69.09 | 71.81 |
| Full Ensemble (seeds × snapshots) | 53.66 | 60.34 | 65.01 | 69.92 | 72.58 |
| Uniform Soup (final seeds) | 44.27 | 50.47 | 48.17 | 46.93 | 50.65 |
| Greedy Soup (final seeds) | 47.93 | 54.61 | 57.80 | 63.36 | 69.39 |
| Full Uniform Soup (seeds × snapshots) | 47.40 | 50.77 | 51.65 | 53.55 | 54.96 |

### M = 8

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 50.24 | 54.61 | 58.10 | 63.42 | 69.56 |
| CoOp (mean ± std) | 47.70 | 53.07 | 57.66 | 62.65 | 68.42 |
| Snapshot Ensemble | 49.86 | 55.28 | 59.26 | 64.76 | 70.04 |
| Multi-Seed Ensemble | 53.96 | 61.35 | 65.66 | 69.80 | 73.58 |
| Full Ensemble (seeds × snapshots) | 54.08 | 61.17 | 65.54 | 70.27 | 72.93 |
| Uniform Soup (final seeds) | 49.70 | 44.33 | 50.18 | 43.91 | 39.95 |
| Greedy Soup (final seeds) | 52.30 | 54.61 | 57.27 | 62.94 | 68.20 |
| Full Uniform Soup (seeds × snapshots) | 51.54 | 47.81 | 53.31 | 53.43 | 52.66 |

### M = 16

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 49.94 | 53.72 | 59.28 | 64.13 | 69.39 |
| CoOp (mean ± std) | 47.66 | 52.70 | 57.82 | 63.06 | 68.44 |
| Snapshot Ensemble | 49.19 | 55.28 | 59.16 | 64.50 | 69.70 |
| Multi-Seed Ensemble | 55.02 | 59.81 | 65.19 | 69.50 | 72.93 |
| Full Ensemble (seeds × snapshots) | 54.91 | 60.93 | 65.07 | 70.15 | 73.05 |
| Uniform Soup (final seeds) | 48.76 | 41.19 | 38.83 | 37.94 | 39.18 |
| Greedy Soup (final seeds) | 49.94 | 53.72 | 59.28 | 62.53 | 69.39 |
| Full Uniform Soup (seeds × snapshots) | 49.35 | 44.80 | 48.17 | 44.62 | 47.52 |

## EuroSAT

### M = 4

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 63.91 | 69.11 | 72.84 | 83.53 | 86.38 |
| CoOp (mean ± std) | 58.43 | 62.97 | 72.37 | 82.34 | 85.20 |
| Snapshot Ensemble | 60.71 | 64.78 | 72.14 | 82.50 | 84.79 |
| Multi-Seed Ensemble | 69.57 | 73.56 | 81.51 | 86.65 | 88.41 |
| Full Ensemble (seeds × snapshots) | 70.04 | 72.65 | 80.43 | 85.89 | 87.52 |
| Uniform Soup (final seeds) | 59.81 | 59.53 | 61.84 | 52.54 | 66.10 |
| Greedy Soup (final seeds) | 63.91 | 69.11 | 72.84 | 83.33 | 86.38 |
| Full Uniform Soup (seeds × snapshots) | 62.20 | 62.59 | 65.43 | 74.77 | 69.11 |

### M = 8

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 61.21 | 68.14 | 74.90 | 85.70 | 87.04 |
| CoOp (mean ± std) | 55.52 | 61.16 | 73.03 | 82.40 | 86.16 |
| Snapshot Ensemble | 56.39 | 63.04 | 71.84 | 82.70 | 85.96 |
| Multi-Seed Ensemble | 63.91 | 72.86 | 80.90 | 86.62 | 89.11 |
| Full Ensemble (seeds × snapshots) | 64.94 | 73.30 | 78.79 | 86.17 | 88.44 |
| Uniform Soup (final seeds) | 57.54 | 47.21 | 46.21 | 56.02 | 68.62 |
| Greedy Soup (final seeds) | 61.79 | 68.14 | 74.90 | 85.70 | 87.04 |
| Full Uniform Soup (seeds × snapshots) | 60.31 | 60.05 | 53.54 | 65.46 | 71.06 |

### M = 16

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 59.14 | 69.01 | 72.30 | 84.85 | 86.68 |
| CoOp (mean ± std) | 52.63 | 63.32 | 71.48 | 83.24 | 85.79 |
| Snapshot Ensemble | 54.68 | 64.82 | 71.80 | 83.02 | 85.55 |
| Multi-Seed Ensemble | 64.44 | 73.79 | 81.59 | 87.41 | 88.70 |
| Full Ensemble (seeds × snapshots) | 65.63 | 73.19 | 80.77 | 86.35 | 87.64 |
| Uniform Soup (final seeds) | 56.31 | 43.53 | 53.90 | 50.16 | 48.53 |
| Greedy Soup (final seeds) | 59.14 | 69.01 | 71.86 | 84.85 | 86.68 |
| Full Uniform Soup (seeds × snapshots) | 58.02 | 51.96 | 63.57 | 58.53 | 65.98 |

## Oxford Flowers

### M = 4

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 83.07 | 88.75 | 91.72 | 94.56 | 96.47 |
| CoOp (mean ± std) | 80.54 | 87.75 | 91.12 | 94.05 | 96.03 |
| Snapshot Ensemble | 80.04 | 87.75 | 91.16 | 94.07 | 95.99 |
| Multi-Seed Ensemble | 86.44 | 93.63 | 95.17 | 96.14 | 97.2 |
| Full Ensemble (seeds × snapshots) | 84.33 | 91.92 | 94.23 | 95.74 | 97 |
| Uniform Soup (final seeds) | 76.7 | 70.2 | 74.71 | 72.47 | 74.62 |
| Greedy Soup (final seeds) | 83.07 | 88.75 | 91.72 | 94.56 | 95.62 |
| Full Uniform Soup (seeds × snapshots) | 75.52 | 76.33 | 80.43 | 77.71 | 78.89 |

### M = 8

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 80.11 | 87.58 | 91.8 | 95.17 | 96.51 |
| CoOp (mean ± std) | 79.35 | 87.01 | 91 | 94.75 | 96.35 |
| Snapshot Ensemble | 79.27 | 86.59 | 90.65 | 94.9 | 96.64 |
| Multi-Seed Ensemble | 85.02 | 92.33 | 95.33 | 96.63 | 97.65 |
| Full Ensemble (seeds × snapshots) | 83.92 | 90.66 | 94.28 | 96.43 | 97.36 |
| Uniform Soup (final seeds) | 76.74 | 66.67 | 69.39 | 60.86 | 63.01 |
| Greedy Soup (final seeds) | 80.11 | 87.41 | 91.76 | 94.88 | 96.06 |
| Full Uniform Soup (seeds × snapshots) | 76.05 | 70.24 | 75.76 | 68.37 | 69.02 |

### M = 16

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 80.35 | 88.39 | 92.12 | 95.09 | 96.83 |
| CoOp (mean ± std) | 79.65 | 87.75 | 91.53 | 94.97 | 96.7 |
| Snapshot Ensemble | 80.13 | 87.24 | 91.8 | 95.14 | 96.51 |
| Multi-Seed Ensemble | 86.07 | 93.46 | 95.21 | 96.63 | 97.6 |
| Full Ensemble (seeds × snapshots) | 84.49 | 91.43 | 94.72 | 96.59 | 97.65 |
| Uniform Soup (final seeds) | 71.5 | 69.06 | 58.55 | 63.3 | 54.28 |
| Greedy Soup (final seeds) | 80.35 | 88.27 | 92.12 | 95.09 | 96.75 |
| Full Uniform Soup (seeds × snapshots) | 71.25 | 72.88 | 68.86 | 71.01 | 63.62 |

## Caltech-101

### M = 4

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 94.20 | 93.83 | 94.08 | 94.73 | 95.78 |
| CoOp (mean ± std) | 93.52 | 93.41 | 93.54 | 94.66 | 95.21 |
| Snapshot Ensemble | 93.75 | 93.97 | 94.04 | 94.94 | 95.52 |
| Multi-Seed Ensemble | 94.44 | 94.56 | 94.97 | 95.42 | 96.15 |
| Full Ensemble (seeds × snapshots) | 94.56 | 94.52 | 94.97 | 95.42 | 95.90 |
| Uniform Soup (final seeds) | 93.96 | 94.16 | 93.23 | 92.78 | 89.57 |
| Greedy Soup (final seeds) | 93.55 | 94.16 | 94.08 | 94.69 | 94.97 |
| Full Uniform Soup (seeds × snapshots) | 93.91 | 94.36 | 94.00 | 94.08 | 94.12 |

### M = 8

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 93.59 | 93.67 | 94.89 | 95.29 | 95.70 |
| CoOp (mean ± std) | 93.13 | 93.17 | 94.19 | 94.74 | 95.50 |
| Snapshot Ensemble | 93.39 | 93.68 | 94.39 | 95.23 | 95.75 |
| Multi-Seed Ensemble | 94.48 | 94.93 | 95.05 | 96.06 | 96.39 |
| Full Ensemble (seeds × snapshots) | 94.40 | 95.21 | 95.09 | 95.90 | 96.27 |
| Uniform Soup (final seeds) | 93.67 | 93.83 | 93.87 | 93.43 | 81.18 |
| Greedy Soup (final seeds) | 94.52 | 94.08 | 94.89 | 95.29 | 95.17 |
| Full Uniform Soup (seeds × snapshots) | 93.71 | 94.16 | 94.77 | 94.52 | 93.55 |

### M = 16

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 93.51 | 94.20 | 93.87 | 95.05 | 95.78 |
| CoOp (mean ± std) | 93.08 | 93.90 | 93.50 | 94.86 | 95.63 |
| Snapshot Ensemble | 93.55 | 94.24 | 94.21 | 95.16 | 95.71 |
| Multi-Seed Ensemble | 94.24 | 95.13 | 95.13 | 95.82 | 96.27 |
| Full Ensemble (seeds × snapshots) | 94.40 | 95.09 | 95.46 | 95.62 | 96.31 |
| Uniform Soup (final seeds) | 90.99 | 93.27 | 89.49 | 87.06 | 82.64 |
| Greedy Soup (final seeds) | 93.43 | 92.82 | 93.23 | 94.60 | 95.62 |
| Full Uniform Soup (seeds × snapshots) | 91.03 | 93.14 | 91.32 | 92.70 | 91.97 |

## Food-101

### M = 4

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 84.30 | 83.47 | 83.13 | 82.94 | 83.61 |
| CoOp (mean ± std) | 82.91 | 82.09 | 82.10 | 82.52 | 83.54 |
| Snapshot Ensemble | 84.11 | 83.94 | 83.93 | 84.60 | 85.78 |
| Multi-Seed Ensemble | 85.71 | 85.89 | 86.15 | 86.65 | 87.26 |
| Full Ensemble (seeds × snapshots) | 86.06 | 86.46 | 86.62 | 87.20 | 87.85 |
| Uniform Soup (final seeds) | 85.14 | 83.32 | 82.37 | 78.30 | 72.80 |
| Greedy Soup (final seeds) | 85.35 | 83.49 | 83.13 | 82.94 | 83.57 |
| Full Uniform Soup (seeds × snapshots) | 85.29 | 84.74 | 84.52 | 82.46 | 83.36 |

### M = 8

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 83.54 | 83.21 | 82.92 | 83.18 | 83.80 |
| CoOp (mean ± std) | 82.38 | 81.84 | 82.19 | 82.65 | 83.55 |
| Snapshot Ensemble | 83.98 | 83.33 | 84.12 | 84.50 | 85.71 |
| Multi-Seed Ensemble | 85.48 | 85.69 | 86.20 | 86.75 | 87.27 |
| Full Ensemble (seeds × snapshots) | 85.87 | 86.18 | 86.64 | 87.21 | 87.86 |
| Uniform Soup (final seeds) | 85.81 | 83.14 | 82.09 | 82.72 | 63.59 |
| Greedy Soup (final seeds) | 85.45 | 84.40 | 83.53 | 83.18 | 83.49 |
| Full Uniform Soup (seeds × snapshots) | 85.72 | 85.37 | 85.14 | 82.49 | 82.76 |
2
### M = 16

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 84.03 | 83.26 | 83.08 | 83.02 | 83.42 |
| CoOp (mean ± std) | 82.62 | 81.98 | 81.89 | 82.37 | 83.37 |
| Snapshot Ensemble | 83.77 | 83.49 | 83.43 | 84.41 | 85.56 |
| Multi-Seed Ensemble | 85.46 | 85.73 | 86.94 | 86.68 | 87.30 |
| Full Ensemble (seeds × snapshots) | 85.87 | 86.11 | 86.55 | 87.20 | 87.69 |
| Uniform Soup (final seeds) | 85.30 | 80.57 | 76.33 | 72.69 | 65.45 |
| Greedy Soup (final seeds) | 85.30 | 83.26 | 83.08 | 82.82 | 83.38 |
| Full Uniform Soup (seeds × snapshots) | 85.41 | 83.78 | 81.55 | 81.39 | 75.69 |

## UCF-101

### M = 4

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 70.34 | 73.17 | 75.92 | 79.91 | 81.60 |
| CoOp (mean ± std) | 69.28 | 72.62 | 75.31 | 79.42 | 80.71 |
| Snapshot Ensemble | 71.04 | 74.85 | 77.08 | 80.76 | 82.19 |
| Multi-Seed Ensemble | 74.02 | 77.53 | 80.86 | 83.77 | 84.09 |
| Full Ensemble (seeds × snapshots) | 74.28 | 77.93 | 80.89 | 84.27 | 84.30 |
| Uniform Soup (final seeds) | 69.36 | 67.67 | 68.09 | 64.13 | 56.15 |
| Greedy Soup (final seeds) | 70.34 | 71.61 | 75.92 | 79.91 | 81.60 |
| Full Uniform Soup (seeds × snapshots) | 69.97 | 71.40 | 72.93 | 71.42 | 64.92 |

### M = 8

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 69.81 | 73.67 | 77.00 | 79.62 | 82.05 |
| CoOp (mean ± std) | 69.08 | 73.12 | 75.18 | 79.21 | 81.49 |
| Snapshot Ensemble | 70.30 | 74.15 | 77.26 | 80.74 | 82.54 |
| Multi-Seed Ensemble | 73.06 | 78.67 | 80.25 | 83.58 | 84.25 |
| Full Ensemble (seeds × snapshots) | 72.35 | 77.85 | 80.81 | 84.03 | 84.40 |
| Uniform Soup (final seeds) | 69.52 | 62.57 | 60.14 | 61.04 | 49.48 |
| Greedy Soup (final seeds) | 69.07 | 73.67 | 75.05 | 78.77 | 81.05 |
| Full Uniform Soup (seeds × snapshots) | 70.39 | 68.17 | 70.47 | 67.94 | 60.48 |

### M = 16

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 69.79 | 72.85 | 76.45 | 79.78 | 81.89 |
| CoOp (mean ± std) | 69.39 | 71.65 | 75.38 | 78.68 | 81.60 |
| Snapshot Ensemble | 70.39 | 73.01 | 77.06 | 80.21 | 83.27 |
| Multi-Seed Ensemble | 73.25 | 76.55 | 80.54 | 82.77 | 84.40 |
| Full Ensemble (seeds × snapshots) | 73.46 | 76.45 | 80.36 | 83.35 | 85.09 |
| Uniform Soup (final seeds) | 68.36 | 54.69 | 58.08 | 51.23 | 49.11 |
| Greedy Soup (final seeds) | 69.79 | 72.85 | 76.45 | 79.78 | 81.15 |
| Full Uniform Soup (seeds × snapshots) | 68.38 | 60.67 | 65.05 | 60.80 | 58.52 |

## FGVC-Aircraft

### M = 4

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 26.88 | 28.98 | 33.72 | 37.23 | 41.16 |
| CoOp (mean ± std) | 26.58 | 28.76 | 32.55 | 36.44 | 40.77 |
| Snapshot Ensemble | 27.84 | 31.86 | 34.34 | 37.87 | 40.79 |
| Multi-Seed Ensemble | 30 | 34.05 | 37.71 | 41.16 | 44.28 |
| Full Ensemble (seeds × snapshots) | 30.84 | 34.41 | 36.93 | 40.47 | 42.87 |
| Uniform Soup (final seeds) | 26.61 | 24 | 25.02 | 14.7 | 24.69 |
| Greedy Soup (final seeds) | 26.88 | 28.98 | 33.72 | 37.23 | 41.07 |
| Full Uniform Soup (seeds × snapshots) | 25.62 | 28.53 | 27.69 | 27.18 | 29.64 |

### M = 8

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 27.72 | 30.57 | 33.48 | 37.68 | 42.84 |
| CoOp (mean ± std) | 26.86 | 29.1 | 31.72 | 36.89 | 42.36 |
| Snapshot Ensemble | 28.01 | 31 | 33.53 | 38.47 | 42.55 |
| Multi-Seed Ensemble | 29.55 | 33.6 | 36.24 | 41.49 | 47.19 |
| Full Ensemble (seeds × snapshots) | 29.76 | 34.05 | 36.78 | 41.91 | 45.18 |
| Uniform Soup (final seeds) | 24.72 | 24.18 | 24.75 | 24.09 | 24.15 |
| Greedy Soup (final seeds) | 27.72 | 28.35 | 33.48 | 36.72 | 42.84 |
| Full Uniform Soup (seeds × snapshots) | 26.49 | 26.61 | 26.31 | 26.94 | 27.42 |

### M = 16

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|---|---|---|---|---|---|
| CoOp (best seed) | 28.14 | 29.76 | 33.33 | 37.77 | 42.42 |
| CoOp (mean ± std) | 27.34 | 29.03 | 32.22 | 36.72 | 42.35 |
| Snapshot Ensemble | 27.33 | 31.32 | 34.08 | 38.43 | 43.24 |
| Multi-Seed Ensemble | 30 | 34.68 | 37.44 | 43.44 | 46.98 |
| Full Ensemble (seeds × snapshots) | 29.4 | 34.83 | 36.99 | 44.31 | 46.59 |
| Uniform Soup (final seeds) | 24.96 | 22.08 | 18.27 | 5.4 | 15.18 |
| Greedy Soup (final seeds) | 27.78 | 29.76 | 33.33 | 37.77 | 42.42 |
| Full Uniform Soup (seeds × snapshots) | 24.48 | 25.47 | 24 | 9.75 | 21.87 |

