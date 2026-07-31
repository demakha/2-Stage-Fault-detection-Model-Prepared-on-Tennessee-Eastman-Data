# Tennessee Eastman Process — Fault Detection & Diagnosis

A two-stage industrial fault detection and diagnosis system built on the Tennessee Eastman Chemical Process dataset, implementing Statistical Process Control (SPC) and Machine Learning.

---

## Problem Statement

Given continuous sensor readings from 52 process variables in a chemical plant, detect whether the plant is operating normally or experiencing a fault — and if faulty, identify which of 20 specific fault types is occurring.

This is a safety-critical problem where:
- **False Negatives** (missed faults) → equipment damage, safety risk
- **False Positives** (false alarms) → unnecessary plant shutdowns

---

## Architecture

```
New Sensor Data (52 sensors, every 3 minutes)
            │
            ▼
    ┌─────────────────┐
    │  Preprocessing  │  StandardScaler + PCA (fitted on normal data only)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────────────────────┐
    │  Stage 1: Anomaly Detection     │  Hotelling T² + SPE statistics
    │  (PCA-based SPC)                │  OR logic: flag if T² OR SPE > threshold
    └────────┬────────────────────────┘
             │
      ┌──────┴──────┐
      │             │
   NORMAL        ANOMALY
      │             │
      ▼             ▼
  ✅ Safe    ┌─────────────────────────┐
             │  Stage 2: Fault Diagnosis│  XGBoost multi-class classifier
             │  (XGBoost Classifier)    │  Identifies fault type (1-20)
             └─────────────────────────┘
```

---

## Key Findings from EDA

### 1. PCA Structure Breakdown (Strongest Finding)
Fault introduction **reduces PCA components** needed for 95% variance from **36 to just 10** — a 72% reduction. This quantitatively proves fault cascade effects through sensor correlation breakdown.

```
Normal data:  36 components for 95% variance
Faulty data:  10 components for 95% variance
              ──► 26 NEW fault-specific variance directions emerge!
              ──► Normal PCA reconstruction fails → High SPE (454x higher!)
```

### 2. Fault Classification Framework (2×2 Matrix)

| | High Signal Strength | Low Signal Strength |
|---|---|---|
| **Low Inconsistency** | Quadrant A: "Deterministic Spikes" — Easy to detect | Quadrant C: "Quiet Baselines" — Hard to detect |
| **High Inconsistency** | Quadrant B: "Stochastic Shocks" — Easy to detect, hard to diagnose | Quadrant D: "Industrial Nightmares" — Nearly impossible |

### 3. T² vs SPE Complementarity
- **T²** detects faults that **shift the process mean** in PC space
- **SPE** detects faults that **break normal correlation structure**
- Key example: Fault 4 — T²=35%, SPE=99.94% → OR logic is essential!

### 4. Confusion Cluster Discovery
Faults 3, 9, 15 form a **confusion cluster** — all dominated by background noise sensor (xmeas_37), indistinguishable in static PCA space, suggesting temporal drift detectable only by LSTM/GRU.

---

## Results

### Stage 1 — Anomaly Detection

| Metric | Value |
|---|---|
| False Positive Rate (OR logic) | ~2.08% |
| False Positive Rate (AND logic) | ~0.01% |
| Detection Rate (strong faults: 1,2,6,7,8,12,13,14) | >97% |
| Detection Rate (weak faults: 3,9,15) | <3% |

**Detection rates by fault type:**

| Fault Category | Examples | T² Rate | SPE Rate | Combined (OR) |
|---|---|---|---|---|
| High T² + High SPE | 1, 2, 6, 7, 8 | >97% | >80% | >97% |
| Low T² + High SPE | 4, 11, 16, 17 | <50% | >40% | >43% |
| Low T² + Low SPE | 3, 9, 15 | <2% | <2% | <3% |

### Stage 2 — Fault Diagnosis

| Fault | F1 Score | Remark |
|---|---|---|
| Fault 7 | 0.9999 | Excellent |
| Fault 1 | 0.9903 | Excellent |
| Fault 6 | 0.9907 | Excellent |
| Fault 3 | 0.2288 | Poor — temporal drift |
| Fault 9 | 0.1706 | Poor — temporal drift |
| Fault 15 | 0.1000 | Poor — temporal drift |

**Overall Macro F1 Score: ~0.65**

---

## Why These Technical Decisions?

### Why PCA over manual feature removal?
Normal and faulty data show **different correlation structures** — new correlation pairs emerge during faults. Removing features based on normal data correlation would discard sensors that become critical fault indicators. PCA preserves all information mathematically.

### Why fit Scaler and PCA on Normal data only?
Fitting on combined data causes **data leakage** — fault patterns contaminate the baseline statistics, making them no longer representative of normal operation.

### Why OR logic for Stage 1?
T² and SPE are **independent** for faulty data — different faults affect each statistic differently. AND logic would miss Fault 4 entirely (T²=35% but SPE=99.94%).

### Why XGBoost for Stage 2?
- Handles 20 classes natively
- Memory efficient for large datasets (4.8M rows)
- Captures complex non-linear fault patterns
- Provides feature importance for interpretability

### Why stratified sampling at simulation run level?
Individual row sampling breaks temporal integrity within simulation runs. Sampling complete simulation runs preserves temporal patterns while respecting independence between runs.

---

## Known Limitations

### Gradual Faults (3, 9, 15) — Undetectable by Current Approach
These faults show <3% Stage 1 detection and <0.25 F1 in Stage 2. Root cause: they create variance in directions **neither captured by T² nor reconstructable by SPE** in static snapshots.

**Why:** These are gradual temporal drift faults. A single 3-minute snapshot looks nearly identical to normal operation. Only a **sequence** of readings reveals the developing pattern.

**Proposed solution:** LSTM/GRU temporal model to capture sequential fault development — current XGBoost baseline establishes performance benchmark for comparison.

---

## Project Structure

```
├── notebooks/
│   ├── 1_EDA.ipynb                    ──► Domain-driven data exploration
│   ├── 2_Preprocessing.ipynb          ──► Scaler, PCA fitting, data transformation
│   ├── 3_Stage1_AnomalyDetection.ipynb──► T²/SPE statistics, threshold calculation
│   └── 4_Stage2_FaultDiagnosis.ipynb  ──► XGBoost training, confusion matrix
├── models/
│   ├── scaler.pkl                     ──► Fitted StandardScaler
│   ├── pca_normal_baseline.pkl        ──► Fitted PCA (normal data only)
│   ├── stage1_detector.pkl            ──► T²/SPE thresholds
│   └── stage2_xgboost.pkl             ──► Trained XGBoost classifier
├── pipeline.py                        ──► Production pipeline script
└── README.md
```

---

## How To Run

### Train complete pipeline:
```bash
python pipeline.py
```

### Load and predict on new data:
```python
from pipeline import load_pipeline, predict

# Load trained pipeline
scaler, pca, detector, stage2_model, process_cols = load_pipeline()

# Predict on new sensor readings
results = predict(
    raw_sensor_data,
    scaler, pca, detector, stage2_model, process_cols
)

print(results[['Stage1_Result', 'Fault_Type', 'T2_Statistic', 'SPE_Statistic']])
```

### Output format:
```
Stage1_Result    Fault_Type          T2_Statistic    SPE_Statistic
NORMAL           Normal Operation    34.21           1.87
ANOMALY          Fault 1             892.43          456.21
ANOMALY          Fault 6             1243.87         2341.56
```

---

## Requirements

```
numpy
pandas
scikit-learn
xgboost
scipy
pyreadr
joblib
matplotlib
seaborn
```

Install:
```bash
pip install numpy pandas scikit-learn xgboost scipy pyreadr joblib matplotlib seaborn
```

---

## Dataset

Tennessee Eastman Process dataset from Kaggle:
- 4 files: Fault Free Training/Testing, Faulty Training/Testing
- 52 process variables (41 measurements + 11 manipulated variables)
- 500 simulation runs per dataset
- Sampled every 3 minutes

[Dataset Link](https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset)

---

## Future Improvements

1. **LSTM/GRU Model** — Capture temporal fault development for gradual faults (3, 9, 15)
2. **Online Learning** — Update model thresholds as plant conditions change over time
3. **CI/CD Pipeline** — Automated retraining when model performance degrades
4. **Explainability** — SHAP values to explain which sensors triggered each fault diagnosis
