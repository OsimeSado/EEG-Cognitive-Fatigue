# EEG Cognitive Fatigue Classification: Mental Arithmetic Task

## 1. Introduction

This project focuses on the classification of cognitive states—specifically the transition from a **baseline (pre-task)** state to a **mental fatigue (post-task)** state induced by sustained mental arithmetic.

By leveraging advanced EEG feature engineering and machine learning techniques, the pipeline extracts **spectral, nonlinear, temporal, and network-level features** to identify neural markers associated with cognitive workload and fatigue.

Each EEG channel is treated as an **independent sample**, enabling fine-grained analysis of fatigue-related neural dynamics across subjects and conditions.

---

## 2. Dataset Information

This project utilizes the **“EEG During Mental Arithmetic Tasks”** dataset by **Zyma et al. (2019)**, publicly available via **PhysioNet**.

### Participants
- **36 healthy subjects**
- Subjects categorized into:
  - **Group G**: Good task performance
  - **Group B**: Poor task performance

### Experimental Conditions
- **Pre-task EEG (_1)**: Background / baseline recording  
- **Post-task EEG (_2)**: EEG recorded during intensive mental arithmetic (fatigue condition)

### EEG Hardware
- Neurocom EEG system  
- **23-channel configuration**
- International **10–20 electrode placement system**

The dataset was used **exactly as provided** by the original authors, which already included:
- High-pass filtering (≤ 30 Hz)
- 50 Hz notch filtering
- Independent Component Analysis (ICA) for:
  - Eye artifacts
  - Muscle artifacts
  - Cardiac artifacts
- Selection of artifact-free **60-second EEG segments**

---

## 3. Pipeline Overview

The project is organized into a **three-stage computational pipeline**, designed for clarity, modularity, and reproducibility.

### 1️⃣ Data Loading Stage (`data_loader.py`)
- Handles all EEG data ingestion
- Loads:
  - Subject-level metadata (`subject-info.csv`)
  - EEG recordings for:
    - Pre-task (baseline)
    - Post-task (mental arithmetic / fatigue)
- Automatically:
  - Identifies available subjects
  - Separates baseline and task recordings
  - Extracts channel names, sampling rate, and recording duration
- Optional utilities for:
  - Raw EEG visualization
  - Subject-level summary reporting

**Note:**  
No signal preprocessing (filtering, ICA, artifact rejection) is performed here, as the dataset is already preprocessed.

---

### 2️⃣ Feature Extraction Stage (`feature_extraction.py`)
- Consumes EEG data provided by `data_loader.py`
- Computes a comprehensive set of:
  - Spectral features
  - Complexity measures
  - Temporal (sliding-window) dynamics
  - Functional connectivity
  - Graph-theoretic network metrics
- Saves engineered features as structured **CSV files** for downstream analysis

---

### 3️⃣ Classification Stage (`ml_classification.py`)
- Loads all extracted feature sets
- Merges features across domains
- Performs classification using:
  - Baseline machine learning models
  - Dimensionality reduction
  - Advanced ensemble learning techniques
- Generates quantitative performance metrics and publication-ready visualizations

---

## 4. Features Extracted (`feature_extraction.py`)

For each EEG channel, the following feature categories are computed to capture complementary aspects of brain activity.

### 4.1 Spectral Features (Frequency Domain)
Derived from the Power Spectral Density (PSD).

**Absolute Band Powers**
- delta_abs (1–4 Hz)
- theta_abs (4–8 Hz)
- alpha_abs (8–13 Hz)
- beta_abs (13–30 Hz)
- gamma_abs (30–80 Hz)

**Relative Band Powers**
- delta_rel, theta_rel, alpha_rel, beta_rel, gamma_rel

**Band Ratios**
- θ/β ratio → cognitive load and fatigue
- α/θ ratio → attentional regulation and mental effort

---

### 4.2 Complexity Features (Nonlinear Dynamics)
Quantify signal irregularity and dynamical complexity.

**Entropy Measures**
- sample_entropy
- spectral_entropy

**Hjorth Parameters**
- hjorth_activity → signal variance
- hjorth_mobility → mean frequency
- hjorth_complexity → frequency variation over time

**Fractal Dimension**
- fractal_dimension (Higuchi method)

---

### 4.3 Sliding Window Features (Temporal Dynamics)
Computed using **10-second sliding windows**:
- delta_power
- theta_power
- alpha_power
- beta_power

These features capture temporal trends and variability during sustained cognitive effort.

---

### 4.4 Connectivity and Graph Features (Network Analysis)

**Connectivity**
- Average alpha-band (8–13 Hz) coherence between all channel pairs

**Graph-Theoretic Metrics**
- Global Efficiency
- Local Efficiency
- Clustering Coefficient
- Characteristic Path Length

These metrics quantify how mental fatigue alters information integration and segregation in brain networks.

---

## 5. Features Used in `ml_classification.py`

The `load_all_features_enhanced()` method aggregates engineered features from CSV files:
- `*_band_powers.csv`
- `*_complexity.csv`
- `*_sliding_window.csv`
- `*_graph_metrics.csv`

All feature groups are merged into a unified dataset for classification.

---

## 6. Classification Results

Experiments were conducted using a **70/30 train–test split** on:

**1,512 samples = 36 subjects × 2 conditions × 21 channels**

| Method | Accuracy | Improvement |
| ------ | -------- | ----------- |
| Stacking Ensemble | **1.0000** | +11.82% |
| Voting Ensemble | 0.9890 | +10.59% |
| Baseline SVM | 0.8943 | +0.00% |
| SVM + PCA | 0.8722 | −2.46% |
| SVM + Interactions | 0.6542 | −26.85% |

🏆 **Best Model:** Stacking Ensemble (100% accuracy)

**Top 5 Predictive Features**
- graph_global_efficiency
- graph_clustering_coefficient
- graph_local_efficiency
- theta_trend
- alpha_variability

---

## 7. Generated Outputs

All outputs are saved in `ml_advanced_results/`:
- confusion_matrices.png
- roc_curves.png
- feature_importance.png, feature_importance.csv
- methods_comparison.png
- advanced_methods_summary.csv

---

## 8. How to Run the Project

### Step 1: Prepare Data
``` bash
python data_loader.py
```
### Step 2: Feature Extraction
```bash
python feature_extraction.py
```
### Step 3: Classification
```bash
python ml_classification.py
```
### 9. Conclusion
This project presents a clear and modular EEG analysis pipeline designed for reliability and reproducibility.
Since the data are already preprocessed, the workflow emphasizes robust data loading and well-defined feature extraction, avoiding redundant processing steps. 
The extracted features capture spectral, complexity, and network-level characteristics of brain activity, making the pipeline suitable for cognitive EEG research and downstream modeling tasks.

### 10. Citation
Original Dataset
Zyma, I., et al. (2019). Electroencephalograms during Mental Arithmetic Task Performance. Data, 4(1), 14.
https://doi.org/10.3390/data4010014

PhysioNet
Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.
