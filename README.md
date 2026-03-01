# Multi-Scale Masking & SIGReg for Brain-JEPA

Exploring multi-scale temporal masking and spectral regularization for fMRI self-supervised learning.

Based on [Brain-JEPA](https://github.com/Eric-LRL/Brain-JEPA) (NeurIPS 2024).

## What's different

- **Multi-scale masking**: predicting targets at short (~10s), medium (~48s), and long (~144s) temporal scales instead of single-scale
- **SIGReg loss**: spectral regularization to prevent representation collapse
- **UCLA dataset**: tested on UCLA CNP (261 subjects, gender classification)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision pyyaml pandas scipy scikit-learn tqdm matplotlib
```

## Training

```bash
cd Brain-JEPA
PYTHONPATH=. torchrun --nproc_per_node=1 src/trainv2.py --config configs/ucla.yaml
```

Configs:
- `ucla_baseline.yaml` - original Brain-JEPA (single-scale, smooth L1 only)
- `ucla_sigreg.yaml` - + SIGReg loss
- `ucla_multiscale.yaml` - + multi-scale masking
- `ucla.yaml` - both

## Evaluation

```bash
python linear_probe.py --checkpoint logs/ucla_*/ucla_train-latest.pth.tar
```

5-fold CV, gender classification (F vs M), reports AUC-ROC.

## Results

| Config | AUC |
|--------|-----|
| Baseline | 0.54 |
| +Multiscale | 0.54 |
| +SIGReg | 0.56 |
| S+M | 0.57 |

Differences not statistically significant (~3% variance between runs).

## Notes

- Batch size 8 on RTX A5000 (24GB), 50 epochs ~30min
- UCLA parcellized with Schaefer 400 atlas, TR=2s
- Data not included (download from OpenNeuro ds000030)
