# DaT Parkinson's Challenge: Deep Learning for Parkinsonian Syndrome Detection

[![DrivenData](https://img.shields.io/badge/DrivenData-DaT_Parkinson's_Challenge-blue.svg)](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/)
[![Final Rank](https://img.shields.io/badge/Leaderboard_Rank-%23194_%2F_1%2C009-success.svg)](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/leaderboard/)
[![Final Score](https://img.shields.io/badge/Final_Log_Loss-0.3619-brightgreen.svg)](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/)
[![Public Score](https://img.shields.io/badge/Public_Log_Loss-0.3205-blue.svg)](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch_%7C_CUDA_12.9-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end deep learning framework developed for the [DrivenData DaT Parkinson's Challenge](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/) (€25,000 prize pool, completed September 2026). The challenge focused on the automated classification of Dopamine Transporter (DaT) SPECT neuroimaging scans as **normal** or **abnormal** (indicative of parkinsonian syndromes) to support early, objective, and accurate clinical diagnosis.

**DrivenData Competitor Profile:** [mdabubokkor](https://www.drivendata.org/users/mdabubokkor/)

---

## 🏆 Final Competition Results

Out of **1,009 joined competitors**, our solution achieved a top-tier standing on the final test evaluation:

| Benchmark Metric | Final Performance | Competition Details |
|:---|:---:|:---|
| **Final Leaderboard Rank** | **#194** | Top ~19% among 1,009 competitors |
| **Final Evaluation (Private) Log Loss** | **`0.3619`** | Primary evaluation metric (lower is better) |
| **Public Benchmark Log Loss** | **`0.3205`** | Scored on hidden test subset |
| **Best Submission ID** | **`id-319540`** | Fully evaluated containerized submission |
| **Secondary Metric (AUROC)** | **`> 0.925`** | Area Under ROC Curve (higher is better) |

### Iterative Submission Progression

| Submission ID | Public Log Loss | Final Test Log Loss | Architecture Highlights | Status |
|:---|:---:|:---:|:---|:---:|
| **`id-319540`** | **`0.3205`** | **`0.3619`** | **Multi-Perspective Deep Ensemble (Compact 3D + Multi-View 2.5D + Signature Regularization)** | **Completed (Best Result)** |
| `id-318669` | `0.3394` | `0.3795` | Tournament Ensemble: 2.5D CNN (65.9%) + 3D DenseNet-121 (28.6%) + 3D ResNet (5.5%) | Completed |
| `id-318718` | `0.3413` | `0.3836` | Domain-Balanced Robust Ensemble with Striatal ROI Priors | Completed |
| `id-319019` | `0.3599` | `0.4088` | Multi-Seed Extended Augmentation Baseline | Completed |

---

## 🧠 Technical Methodology & Architecture

Classification of DaT-SPECT imaging requires capturing subtle bilateral asymmetries and specific uptake deficits in the putamen and caudate nucleus while remaining invariant to inter-scanner and reconstruction variations.

```
                           ┌──────────────────────────────────────────────┐
                           │      Input 3D DaT-SPECT Scan (NIfTI)         │
                           └──────────────────────┬───────────────────────┘
                                                  │
                      ┌───────────────────────────┴───────────────────────────┐
                      ▼                                                       ▼
       ┌─────────────────────────────┐                         ┌─────────────────────────────┐
       │   2.5D Multi-View Branch    │                         │    Compact 3D CNN Branch    │
       │  (Striatal Projection Slices│                         │  (Full Volumetric Context   │
       │   & Hemisphere Asymmetry)   │                         │     Spatial Convolutions)   │
       └──────────────┬──────────────┘                         └──────────────┬──────────────┘
                      │                                                       │
                      ▼                                                       ▼
       ┌─────────────────────────────┐                         ┌─────────────────────────────┐
       │   5-Fold Hemisphere Models  │                         │    5-Fold 3D Deep Models    │
       │     (Probability Weight)    │                         │     (Probability Weight)    │
       │            0.75             │                         │            0.25             │
       └──────────────┬──────────────┘                         └──────────────┬──────────────┘
                      │                                                       │
                      └───────────────────────────┬───────────────────────────┘
                                                  │
                                                  ▼
                               ┌─────────────────────────────────────┐
                               │   Ensemble Fusion & Bounded Head    │
                               │        p ∈ [1e-5, 1 - 1e-5]         │
                               └──────────────────┬──────────────────┘
                                                  │
                                                  ▼
                               ┌─────────────────────────────────────┐
                               │ Final Calibrated Probability Score  │
                               └─────────────────────────────────────┘
```

### 1. Hybrid 2.5D & 3D Multi-Perspective Fusion
- **Hemisphere-Aware 2.5D Projections:** Deep convolutional backbones trained on high-density striatal slices. Extracts quantitative bilateral uptake asymmetry—a hallmark clinical biomarker of parkinsonian neurodegeneration.
- **Compact 3D Volumetric CNNs:** Directly models global volumetric spatial dependencies across the whole brain without slice-selection bias.
- **Balanced Ensemble:** Fuses complementary 2D slice features (weight: `0.75`) and 3D volumetric context (weight: `0.25`).

### 2. Scanner Acquisition-Signature Validation Protocol
- Scans originate from diverse clinical sites with differing gamma camera collimators, reconstruction filters, and voxel resolutions.
- Designed an acquisition-signature-grouped cross-validation scheme to prevent scanner leakage across validation folds, ensuring that model predictions generalize cleanly to unseen medical centers.

### 3. Metric-Aligned Probability Calibration
- Log Loss heavily penalizes confident incorrect predictions:
  $$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
- Predictions are bounded to $[10^{-5}, 1 - 10^{-5}]$, guarding against log-loss divergence on edge cases.

---

## 📁 Repository Structure

```text
├── LICENSE
├── README.md
└── compact3d_v2_submission/
    ├── submission_compact3d_v2_ready_20260814.zip  # Scored containerized submission archive
    ├── main.py                                      # Container entrypoint for batch inference
    ├── manifest.json                                # Model ensemble configuration
    ├── readiness_manifest.json                      # Validation metrics and reproducibility logs
    ├── SHA256SUMS                                   # Cryptographic integrity verification
    ├── compact3d_domain_balanced_v2_protocol.md     # Cross-validation protocol & ablation study
    └── public_results.md                            # Benchmark comparisons & error analysis
```

---

## ⚙️ Inference & Execution Pipeline

The solution complies strictly with DrivenData's offline Docker execution environment:

- **Isolated Runtime:** Zero external network access during inference; all model weights are contained within the package assets.
- **Standardized I/O Contract:**
  - Input: `/code_execution/data/submission_format.csv` and `/code_execution/data/niftis/<scan_id>.nii.gz`
  - Output: `/code_execution/submission.csv`
- **Execution Efficiency:** Processes complete test volumes with GPU-accelerated inference in minutes.

---

## 📜 Disclaimer & Ethical Use

This repository is maintained for scientific research, competition benchmarking, and algorithmic reproducibility. It is not an FDA-cleared or CE-marked medical device and is not intended for primary clinical diagnosis or medical decision-making in patient care.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
