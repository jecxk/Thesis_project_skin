# Metadata-fusion evaluation summary

**Generated:** 2026-06-07T20:27:41

## Per-model results (metadata-fusion)

| Model | Accuracy | Macro F1 | Macro AUC | Cohen Kappa | MCC |
|---|---:|---:|---:|---:|---:|
| eb0 | 83.30 % | 0.7679 | — | 0.6961 | 0.6991 |
| rn50 | 87.69 % | 0.8095 | — | 0.7601 | 0.7608 |
| dn121 | 85.63 % | 0.7929 | — | 0.7273 | 0.7275 |
| swin | 89.82 % | 0.8709 | — | 0.8047 | 0.8050 |

## Ensemble

| Metric | Value |
|---|---|
| Accuracy | **90.49 %** |
| Macro F1 | **0.8728** |
| Macro AUC | **0.9838** |
| Cohen Kappa | 0.8174 |
| MCC | 0.8177 |

## Comparison vs.\ image-only baseline (seed 42 ensemble)

| Metric | Image-only baseline | + Metadata fusion | Δ |
|---|---:|---:|---:|
| Accuracy | 88.89 % | 90.49 % | +1.60 pp |
| Macro F1 | 0.8341 | 0.8728 | +0.0387 |
| Macro AUC | 0.9812 | 0.9838 | +0.0026 |