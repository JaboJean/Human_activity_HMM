# Hidden Markov Models - [ HMM Group 43 ]

> **Course Project — Human Activity Recognition with Smartphone Inertial Sensors**
> Jean & Thierry | March 2026

---

## Overview

This project implements a **Hidden Markov Model (HMM)**-based pipeline for recognising four human activities — *Standing*, *Walking*, *Jumping*, and *Still* — from raw 6-axis inertial sensor data collected using iPhone X smartphones. The system combines time-domain and frequency-domain feature engineering with per-class Gaussian HMMs trained via the Baum–Welch algorithm and decoded using the Viterbi algorithm.

**Final results:** 99.17% train accuracy · **100% test accuracy** · Perfect sensitivity & specificity across all four classes.

---

## Repository Structure

```
├── data/
│   ├── jumping/
│   │   ├── jean_jumping_1/          # Accelerometer.csv, Gyroscope.csv, Metadata.csv
│   │   ├── jean_jumping_2/
│   │   ├── ...
│   │   ├── thierry_jumping_1/
│   │   └── ...
│   ├── standing/
│   │   ├── jean_standing_1/
│   │   └── ...
│   ├── walking/
│   │   ├── jean_walking_1/
│   │   └── ...
│   └── still/
│       ├── jean_still_1/
│       └── ...
├── HMM_Human_Activity_Recognition G43.ipynb   # Main notebook
├── Hidden Markov Models - [ HMM Group 43 ]Report.docx                    # Full project report (4–5 pages)
└── README.md                              # This file
```

Each session folder contains:
| File | Description |
|------|-------------|
| `Accelerometer.csv` | 3-axis accelerometer data (x, y, z) at 100 Hz |
| `Gyroscope.csv` | 3-axis gyroscope data (x, y, z) at 100 Hz |
| `Metadata.csv` | Device info, sampling rate, recording timestamp |

---

## Dataset Summary

| Activity | Sessions | Samples | Duration | Train Windows | Test Windows |
|----------|----------|---------|----------|---------------|--------------|
| Standing | 12 | 9,936 | ~99.4 s | 152 | 30 |
| Walking  | 12 | 10,145 | ~101.5 s | 154 | 30 |
| Jumping  | 12 | 7,501 | ~75.0 s | 109 | 23 |
| Still    | 14 | 11,852 | ~118.5 s | 186 | 32 |
| **Total** | **50** | **39,434** | **~394 s** | **601** | **115** |

**Data collection details:**
- **Both participants:** iPhone X — Sensor Logger v1.47.1 (iOS)
- **Sampling rate:** 100 Hz (10 ms interval) — identical across both devices, no resampling needed
- **Sensors:** Accelerometer (x, y, z) · Gyroscope (x, y, z)

---

## Methodology

### Preprocessing
1. Merge Accelerometer + Gyroscope CSVs on nearest timestamp (`merge_asof`)
2. Linear interpolation for missing samples; forward/back-fill for edge gaps
3. 4th-order Butterworth low-pass filter at 20 Hz (zero-phase `filtfilt`) to remove high-frequency noise
4. Sliding window segmentation: **1-second windows (100 samples) with 50% overlap (50-sample step)**

> **Window size rationale:** 1 second captures ≥1 complete walking gait cycle (~0.8–1.2 s) and multiple jump cycles (~0.5 s). 50% overlap doubles training data and smooths activity boundaries.

### Feature Extraction (47 features total)

**Time-domain (both accelerometer and gyroscope):**
- Mean, Variance, RMS, Standard Deviation per axis
- Signal Magnitude Area (SMA)
- Inter-axis correlations (xy, xz, yz)

**Frequency-domain (accelerometer only):**
- Dominant frequency per axis
- Spectral energy in the 0.5–5 Hz activity band
- Top-3 FFT magnitudes per axis

All features normalised using **Z-score standardisation** (fit on train, applied to test).

### HMM Architecture

One `GaussianHMM` (diagonal covariance) per activity class, trained independently:

| Component | Value |
|-----------|-------|
| Hidden states per HMM | 2 |
| Covariance type | Diagonal |
| Training algorithm | Baum–Welch (EM) |
| Convergence criterion | \|ΔlogL\| < 1×10⁻⁴ |
| Max iterations | 200 |
| Min covariance floor | 1×10⁻³ (numerical stability) |
| Decoding | Viterbi algorithm |
| Classification | Maximum log-likelihood across all class HMMs |

### Numerical Stability Notes
The *still* activity produces near-zero feature variance (phone flat on table), which can cause Gaussian covariance collapse during Baum–Welch. Three fixes were applied:
- `min_covar = 1e-3` prevents eigenvalue collapse
- `repair_transmat()` replaces zero-sum rows with uniform distributions
- Retry loop with fresh random seeds (up to 5 attempts)

---

## Results

### Training Results

| Activity | Log-Likelihood | Converged | Iterations |
|----------|---------------|-----------|------------|
| Standing | 12,307.42 | ✓ | 6 / 200 |
| Walking  | 2,642.73  | ✓ | 19 / 200 |
| Jumping  | −3,311.78 | ✓ | 4 / 200 |
| Still    | 24,526.52 | ✓ | 7 / 200 |

### Test Evaluation (Unseen Data)

| Activity | N Samples | Sensitivity | Specificity | Accuracy | F1 |
|----------|-----------|-------------|-------------|----------|----|
| Standing | 30 | 1.0000 | 1.0000 | 1.0000 | 1.00 |
| Walking  | 30 | 1.0000 | 1.0000 | 1.0000 | 1.00 |
| Jumping  | 23 | 1.0000 | 1.0000 | 1.0000 | 1.00 |
| Still    | 32 | 1.0000 | 1.0000 | 1.0000 | 1.00 |
| **Overall** | **115** | **1.0000** | **1.0000** | **1.0000** | **1.00** |

> **Train accuracy: 99.17% · Test accuracy: 100.00%**

---

## Getting Started

### Prerequisites

```bash
pip install hmmlearn scikit-learn seaborn matplotlib numpy pandas scipy
```

### Running in Google Colab (Recommended)

1. Upload `data.zip` to your Colab environment
2. Upload `HMM_Human_Activity_Recognition.ipynb`
3. Set runtime to **GPU (T4)** under *Runtime → Change runtime type*
4. Run all cells — the notebook will automatically mount Google Drive and save outputs to `MyDrive/HMM_Activity_Recognition/`

### Running Locally

1. Clone or download this repository
2. Ensure the `data/` folder is in the same directory as the notebook
3. Update `DATA_ROOT` in Cell 6 if needed
4. Run all cells in order

### Output Files (saved automatically)

| File | Description |
|------|-------------|
| `01_raw_signals.png` | Raw accelerometer & gyroscope signals per activity |
| `02_feature_distributions.png` | Top-8 discriminative feature histograms |
| `03_baumwelch_convergence.png` | Log-likelihood convergence curves |
| `04_transition_matrices.png` | Learned intra-class transition matrices |
| `05_emission_means.png` | Emission mean scatter plot |
| `06_viterbi_decoded.png` | Viterbi decoded hidden state sequence |
| `07_confusion_matrix.png` | Test set confusion matrix |
| `08_sensitivity_specificity.png` | Per-class sensitivity & specificity bar chart |
| `09_inter_activity_transitions.png` | Inter-activity transition heatmaps |
| `10_pca_feature_space.png` | PCA projection of 47-D feature space |
| `11_psd_comparison.png` | Power spectral density per activity |
| `hmm_models.pkl` | Serialised trained HMM models |
| `scaler.pkl` | Fitted StandardScaler |
| `test_predictions.csv` | Window-level predictions on test set |
| `summary.json` | Accuracy, metrics, and hyperparameter summary |

---

## Task Allocation

| Task | Jean | Thierry |
|------|------|---------|
| Data Collection — Jumping & Walking | ✓ | ✓ |
| Data Collection — Standing & Still | ✓ | ✓ |
| Preprocessing & Filtering | ✓ | |
| Feature Extraction — Time Domain | | ✓ |
| Feature Extraction — Frequency Domain | ✓ | |
| HMM Architecture Design | ✓ | ✓ |
| Baum–Welch Training Implementation | | ✓ |
| Viterbi Decoding Implementation | ✓ | |
| Evaluation & Metrics | | ✓ |
| Visualisations | ✓ | ✓ |
| Report Writing | ✓ | ✓ |
| GitHub Repository Management | ✓ | ✓ |

---

## References

- Bao, L., & Intille, S. S. (2004). Activity recognition from user-annotated acceleration data. *Pervasive Computing*, 1–17.
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257–286.
- Lara, O. D., & Labrador, M. A. (2013). A survey on human activity recognition using wearable sensors. *IEEE Communications Surveys & Tutorials*, 15(3), 1192–1209.
- Ronao, C. A., & Cho, S. B. (2016). Human activity recognition with smartphone sensors using deep learning neural networks. *Expert Systems with Applications*, 59, 235–244.
