"""
Tennessee Eastman Process — Fault Detection & Diagnosis Pipeline
================================================================
Two-Stage Architecture:
    Stage 1: Anomaly Detection  (PCA-based T² and SPE statistics)
    Stage 2: Fault Diagnosis    (XGBoost multi-class classifier)

"""

import numpy as np
import pandas as pd
import joblib
from scipy import stats


# ============================================================
# CONFIGURATION
# ============================================================

# Process sensor columns (excludes metadata)
PROCESS_COLS = (
    [f'xmeas_{i}' for i in range(1, 42)] +
    [f'xmv_{i}'   for i in range(1, 12)]
)

# Fault introduction times (hours) and sampling interval (minutes)
FAULT_INTRO_TRAINING = 1    # 1 hour for training data
FAULT_INTRO_TESTING  = 8    # 8 hours for testing data
SAMPLING_INTERVAL    = 3    # 3 minutes per sample

# PCA components (captures 95% variance of normal operation)
N_COMPONENTS = 36

# Detection threshold confidence level
ALPHA = 0.01  # 1% false positive rate

# Fault labels mapping
FAULT_LABELS = {0: 'Normal Operation', **{i: f'Fault {i}' for i in range(1, 21)}}


# ============================================================
# SECTION 1: PREPROCESSING FUNCTIONS
# ============================================================

def remove_prefault_samples(df, fault_intro_hours, sampling_interval_mins=3):
    """
    Remove pre-fault normal period from faulty dataset.

    In the TEP dataset, faults are introduced after a period of normal
    operation. These pre-fault rows are mislabeled as faulty and must
    be removed to avoid label noise.

    Parameters:
    -----------
    df                   : Faulty dataframe with 'sample' column
    fault_intro_hours    : Hours before fault is introduced
                           (Training=1hr, Testing=8hr)
    sampling_interval_mins: Sampling frequency in minutes (default=3)

    Returns:
    --------
    Cleaned dataframe with only actual fault samples
    """
    prefault_samples = int((fault_intro_hours * 60) / sampling_interval_mins)

    print(f"Fault introduced at  : {fault_intro_hours} hours")
    print(f"Sampling interval    : {sampling_interval_mins} minutes")
    print(f"Pre-fault samples    : {prefault_samples} samples per simulation run")
    print(f"Rows before removal  : {df.shape[0]:,}")

    df_clean = df[df['sample'] > prefault_samples].copy()

    print(f"Rows after removal   : {df_clean.shape[0]:,}")
    print(f"Rows removed         : {df.shape[0] - df_clean.shape[0]:,}")

    return df_clean


def handle_missing_values(df, process_cols):
    """
    Handle missing values using forward fill then backward fill.

    Justified by continuous plant operation:
    If a sensor reading is missing, the true value is very close
    to the previous or next reading (steady-state assumption).

    Parameters:
    -----------
    df           : Dataframe with sensor readings
    process_cols : List of sensor column names

    Returns:
    --------
    Dataframe with no missing values
    """
    df = df.copy()
    df[process_cols] = df[process_cols].ffill().bfill()

    missing = df[process_cols].isnull().sum().sum()
    print(f"Missing values after imputation: {missing}")

    return df


def fit_scaler(normal_train_df, process_cols):
    """
    Fit StandardScaler on Normal (Fault Free) Training data ONLY.

    Critical: Fitting only on normal data establishes TRUE baseline
    statistics. Fitting on combined data causes DATA LEAKAGE where
    fault patterns contaminate the baseline reference.

    Parameters:
    -----------
    normal_train_df : Fault Free Training dataframe
    process_cols    : List of sensor column names

    Returns:
    --------
    Fitted StandardScaler object
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(normal_train_df[process_cols])

    print(f"Scaler fitted on Normal Training data: {normal_train_df.shape}")
    print(f"Features scaled: {len(process_cols)}")

    return scaler


def fit_pca(normal_train_scaled, n_components=36):
    """
    Fit PCA on scaled Normal (Fault Free) Training data ONLY.

    Critical: PCA fitted on normal data learns the BASELINE correlation
    structure. When faulty data is projected into this normal PCA space,
    deviations reveal fault patterns.

    Key finding from EDA:
    - Normal data needs 36 PCs for 95% variance
    - Faulty data only needs 10 PCs (72% reduction!)
    - This proves faults create massive correlation cascade effects

    Parameters:
    -----------
    normal_train_scaled : Scaled normal training data (numpy array)
    n_components        : Number of PCA components (default=36)

    Returns:
    --------
    Fitted PCA object
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    pca.fit(normal_train_scaled)

    variance_explained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"PCA fitted on Normal Training data")
    print(f"Components: {n_components}")
    print(f"Variance explained: {variance_explained:.2f}%")

    return pca


def preprocess_dataset(df, scaler, pca, process_cols,
                        fault_intro_hours=None, sampling_interval=3,
                        is_faulty=False):
    """
    Complete preprocessing pipeline for any dataset.

    Applies in correct order:
    1. Remove pre-fault rows (faulty data only)
    2. Handle missing values
    3. Scale using fitted scaler (transform only, never refit!)
    4. PCA transform using fitted PCA (transform only, never refit!)

    Parameters:
    -----------
    df                 : Raw dataframe
    scaler             : Fitted StandardScaler (from normal training)
    pca                : Fitted PCA (from normal training)
    process_cols       : Sensor column names
    fault_intro_hours  : Hours before fault intro (None for fault free)
    sampling_interval  : Minutes between samples (default=3)
    is_faulty          : Whether this is faulty dataset

    Returns:
    --------
    pca_data    : PCA transformed features (n_samples, n_components)
    scaled_data : Scaled features (n_samples, n_features)
    labels      : Fault labels if faulty, None otherwise
    """
    df = df.copy()

    # Step 1: Remove pre-fault rows (faulty data only)
    if is_faulty and fault_intro_hours is not None:
        df = remove_prefault_samples(df, fault_intro_hours, sampling_interval)

    # Step 2: Handle missing values
    df = handle_missing_values(df, process_cols)

    # Step 3: Scale using fitted scaler (NEVER refit!)
    scaled_data = scaler.transform(df[process_cols])

    # Step 4: PCA transform using fitted PCA (NEVER refit!)
    pca_data = pca.transform(scaled_data)

    # Extract labels if faulty
    labels = df['faultNumber'].values if is_faulty else None

    print(f"Preprocessing complete:")
    print(f"  Scaled shape: {scaled_data.shape}")
    print(f"  PCA shape:    {pca_data.shape}")

    return pca_data, scaled_data, labels



# SECTION 2: STAGE 1 — ANOMALY DETECTION FUNCTIONS


def compute_t2_spe(data_pca, data_scaled, pca_model):
    """
    Compute Hotelling T^2 and SPE (Q) statistics for each sample.

    T2 Statistic (Hotelling, 1947):
    --------------------------------
    Measures deviation WITHIN the PCA model.
    "How far is this sample from the center of normal operation
     in principal component space?"

    Formula: T^2 = summation of (score_i^2 / eigenvalue_i)
    Distribution: Follows F-distribution when parameters estimated from data
    Detects: Faults that shift process mean in PC space

    SPE Statistic / Q Statistic (Jackson-Mudholkar, 1979):
    -------------------------------------------------------
    Measures deviation OUTSIDE the PCA model.
    "How well does the normal PCA reconstruct this sample?"

    Formula: SPE = ||x_scaled - x_reconstructed||^2
    Distribution: Approximated by weighted Chi-squared distribution
    Detects: Faults that break normal correlation structure

    Key insight: Normal data needs 36 PCs (95% variance) while faulty data
    needs only 10 PCs - proving faults create 26 NEW variance directions
    outside normal PCA space, causing massive SPE increase (454x higher!)

    Parameters:
    -----------
    data_pca    : PCA transformed data (n_samples, n_components)
    data_scaled : Original scaled data  (n_samples, n_features)
    pca_model   : Fitted PCA object (from normal data ONLY)

    Returns:
    --------
    t2  : T^2 statistic per sample (n_samples,)
    spe : SPE statistic per sample (n_samples,)
    """
    # T^2 Statistic
    eigenvalues = pca_model.explained_variance_
    t2 = np.sum((data_pca ** 2) / eigenvalues, axis=1)

    # SPE Statistic
    data_reconstructed = pca_model.inverse_transform(data_pca)
    residuals = data_scaled - data_reconstructed
    spe = np.sum(residuals ** 2, axis=1)

    return t2, spe


def compute_theoretical_thresholds(t2_normal, spe_normal, n_samples, n_components, alpha=0.01):
    """
    Compute theoretically justified control limits.

    T^2 Threshold (Hotelling F-distribution):
    -----------------------------------------
    When PCA parameters are estimated from data, T^2 follows a scaled
    F-distribution (Hotelling, 1947):
        T^2 x ((n-p)/(p(n-1))) ~ F(p, n-p)

    SPE Threshold (Jackson-Mudholkar approximation):
    -------------------------------------------------
    SPE is approximated by a weighted Chi-squared distribution using
    method of moments matching (Jackson & Mudholkar, 1979):
        SPE = g x Chi-squared(h)
    where g and h are estimated from observed SPE mean and variance.

    Parameters:
    -----------
    t2_normal   : T^2 values from normal training data
    spe_normal  : SPE values from normal training data
    n_samples   : Number of normal training samples
    n_components: Number of PCA components
    alpha       : Significance level (default=0.01 for 99% confidence)

    Returns:
    --------
    t2_threshold  : Theoretical T^2 control limit
    spe_threshold : Theoretical SPE control limit
    """
    n = n_samples
    p = n_components

    # T^2 Threshold - F-distribution (exact)
    f_critical = stats.f.ppf(1 - alpha, p, n - p)
    t2_threshold = ((p * (n - 1)) / (n - p)) * f_critical

    # SPE Threshold - Jackson-Mudholkar approximation (weighted Chi-squared)
    mean_spe = spe_normal.mean()
    var_spe  = np.var(spe_normal, ddof=1)

    g = var_spe / (2 * mean_spe)           # Scaling factor
    h = (2 * mean_spe ** 2) / var_spe      # Degrees of freedom

    chi2_critical = stats.chi2.ppf(1 - alpha, h)
    spe_threshold = g * chi2_critical

    print(f"Theoretical T^2  threshold (F-dist):          {t2_threshold:.4f}")
    print(f"Theoretical SPE threshold (Jackson-Mudholkar): {spe_threshold:.4f}")

    return t2_threshold, spe_threshold


def fit_stage1_detector(normal_train_pca, normal_train_scaled, pca_model, alpha=0.01):
    """
    Fit Stage 1 Anomaly Detector using PCA T^2 and SPE statistics.

    Uses theoretically justified thresholds:
    - T^2:  Hotelling F-distribution (exact)
    - SPE: Jackson-Mudholkar weighted Chi-squared (approximation)

    Detection logic: OR rule
    "Flag as ANOMALY if T^2 > threshold OR SPE > threshold"

    Justified by:
    - T^2 and SPE are INDEPENDENT for faulty data
    - Different faults affect each statistic differently
    - Fault 4: T^2=35%, SPE=99.94% - AND would miss it completely!
    - OR maximises detection while keeping FPR acceptable

    Parameters:
    -----------
    normal_train_pca    : PCA data from normal training (n_samples, n_components)
    normal_train_scaled : Scaled data from normal training (n_samples, n_features)
    pca_model           : Fitted PCA object
    alpha               : Significance level (default=0.01)

    Returns:
    --------
    detector : Dictionary containing thresholds and statistics
    """
    print("Fitting Stage 1 Anomaly Detector...")

    # Compute T^2 and SPE on normal training data
    t2_normal, spe_normal = compute_t2_spe(
        normal_train_pca,
        normal_train_scaled,
        pca_model
    )

    n = normal_train_pca.shape[0]
    p = normal_train_pca.shape[1]

    # Compute theoretical thresholds
    t2_threshold, spe_threshold = compute_theoretical_thresholds(
        t2_normal, spe_normal, n, p, alpha
    )

    # Also compute empirical thresholds for comparison
    t2_empirical  = np.percentile(t2_normal, (1 - alpha) * 100)
    spe_empirical = np.percentile(spe_normal, (1 - alpha) * 100)

    print(f"\nEmpirical T^2  threshold:  {t2_empirical:.4f}")
    print(f"Empirical SPE threshold:  {spe_empirical:.4f}")
    print(f"T^2  % error: {((t2_empirical - t2_threshold) / t2_threshold) * 100:.4f}%")
    print(f"SPE % error: {((spe_empirical - spe_threshold) / spe_threshold) * 100:.4f}%")

    detector = {
        't2_threshold'  : t2_threshold,
        'spe_threshold' : spe_threshold,
        't2_empirical'  : t2_empirical,
        'spe_empirical' : spe_empirical,
        'alpha'         : alpha,
        'n_samples'     : n,
        'n_components'  : p,
        't2_normal_mean': t2_normal.mean(),
        'spe_normal_mean': spe_normal.mean()
    }

    print("\nStage 1 Detector fitted successfully!")
    return detector


def predict_stage1(data_pca, data_scaled, pca_model, detector):
    """
    Stage 1 prediction: Normal or Anomaly.

    Uses OR logic: flag as ANOMALY if T^2 OR SPE exceeds threshold.

    Parameters:
    -----------
    data_pca    : PCA transformed data (n_samples, n_components)
    data_scaled : Scaled data (n_samples, n_features)
    pca_model   : Fitted PCA object
    detector    : Fitted detector dictionary (from fit_stage1_detector)

    Returns:
    --------
    predictions : Array of 0 (Normal) or 1 (Anomaly) per sample
    t2_values   : T^2 statistic per sample
    spe_values  : SPE statistic per sample
    """
    t2_values, spe_values = compute_t2_spe(data_pca, data_scaled, pca_model)

    t2_alarm  = t2_values  > detector['t2_threshold']
    spe_alarm = spe_values > detector['spe_threshold']

    # OR logic - flag if EITHER exceeds threshold
    predictions = (t2_alarm | spe_alarm).astype(int)

    return predictions, t2_values, spe_values


def evaluate_stage1(predictions, true_labels, dataset_name="Dataset"):
    """
    Evaluate Stage 1 anomaly detector performance.

    For Normal data   -> measures False Positive Rate
    For Faulty data   -> measures True Positive Rate per fault type

    Parameters:
    -----------
    predictions  : Stage 1 predictions (0=Normal, 1=Anomaly)
    true_labels  : True labels (0=Normal, 1-20=Fault type)
    dataset_name : Name for reporting

    Returns:
    --------
    results : Dictionary of evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Stage 1 Evaluation: {dataset_name}")
    print(f"{'='*60}")

    is_normal = true_labels == 0

    if is_normal.all():
        # Normal data - measure False Positive Rate
        fpr = predictions.mean() * 100
        print(f"False Positive Rate: {fpr:.2f}%")
        print(f"(Target: ~{1.0:.1f}% at 99th percentile threshold)")
        return {'fpr': fpr}

    else:
        # Faulty data - measure detection rate per fault
        results = {}
        print(f"{'Fault':<8} {'Detection Rate':>15}")
        print(f"{'-'*25}")

        for fault in sorted(np.unique(true_labels)):
            mask = true_labels == fault
            rate = predictions[mask].mean() * 100
            results[f'fault_{fault}'] = rate
            print(f"Fault {fault:<4} {rate:>13.2f}%")

        avg_rate = np.mean(list(results.values()))
        print(f"\nAverage Detection Rate: {avg_rate:.2f}%")
        results['average'] = avg_rate
        return results



# SECTION 3: STAGE 2 — FAULT DIAGNOSIS FUNCTIONS


def fit_stage2_classifier(X_train, y_train, X_val, y_val):
    """
    Fit Stage 2 XGBoost Fault Classifier.

    Trained on Faulty Training data ONLY (labels 1-20).
    Labels converted to 0-19 for XGBoost compatibility.

    Sampling strategy:
    - Data split at simulation run level (not row level)
    - Preserves temporal integrity within each simulation
    - 200 of 400 training runs sampled (stratified by run)
    - All 20 fault types equally represented

    Parameters:
    -----------
    X_train : Training features (n_samples, n_components)
    y_train : Training labels (1-20)
    X_val   : Validation features
    y_val   : Validation labels (1-20)

    Returns:
    --------
    model : Fitted XGBoost classifier
    """
    from xgboost import XGBClassifier

    # Convert labels to 0-indexed (XGBoost requirement)
    y_train_xgb = y_train - 1
    y_val_xgb   = y_val   - 1

    model = XGBClassifier(
        n_estimators          = 100,
        max_depth             = 6,
        learning_rate         = 0.1,
        n_jobs                = -1,
        random_state          = 42,
        eval_metric           = 'mlogloss',
        early_stopping_rounds = 10
    )

    print("Training Stage 2 XGBoost Classifier...")
    model.fit(
        X_train, y_train_xgb,
        eval_set = [(X_val, y_val_xgb)],
        verbose  = 10
    )

    print("Stage 2 Classifier trained successfully!")
    return model


def predict_stage2(data_pca, model):
    """
    Stage 2 prediction: Which fault type?

    Only called when Stage 1 flags an ANOMALY.
    Predicts which of the 20 fault types is occurring.

    Parameters:
    -----------
    data_pca : PCA transformed data (n_samples, n_components)
    model    : Fitted XGBoost classifier

    Returns:
    --------
    predictions : Fault type predictions (1-20)
    """
    # Predict (returns 0-19)
    raw_predictions = model.predict(data_pca)

    # Convert back to original labels (1-20)
    predictions = raw_predictions + 1

    return predictions


def evaluate_stage2(predictions, true_labels):
    """
    Evaluate Stage 2 fault diagnosis performance.

    Metrics:
    - F1 Score per fault type (Precision x Recall balance)
    - Macro F1 Score (average across all fault types)
    - Confusion matrix analysis

    Parameters:
    -----------
    predictions  : Stage 2 predictions (1-20)
    true_labels  : True fault labels (1-20)

    Returns:
    --------
    results : DataFrame with per-fault metrics
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(true_labels, predictions)
    results = []

    for i in range(1, 21):
        TP = cm[i-1][i-1]
        FP = cm[:, i-1].sum() - TP
        FN = cm[i-1, :].sum() - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0)

        ho = cm[i-1, :].sum()
        vi = cm[:, i-1].sum()
        remark = ("Under-predicted" if ho > vi else
                  "Over-predicted"  if ho < vi else
                  "Accurate")

        results.append({
            'Fault'    : f'Fault_{i}',
            'Precision': round(precision, 4),
            'Recall'   : round(recall, 4),
            'F1_Score' : round(f1, 4),
            'Remark'   : remark
        })

    df_results = pd.DataFrame(results).sort_values('F1_Score', ascending=True)
    macro_f1   = df_results['F1_Score'].mean()

    print(f"\nStage 2 Evaluation Results:")
    print(f"{'='*70}")
    print(df_results.to_string(index=False))
    print(f"{'='*70}")
    print(f"Macro F1 Score: {macro_f1:.4f}")

    return df_results, macro_f1


# SECTION 4: END-TO-END PREDICTION PIPELINE


def predict(raw_sensor_data, scaler, pca, detector, stage2_model, process_cols):
    """
    End-to-end prediction pipeline for new sensor readings.

    Two-Stage Architecture:
    Stage 1: Is plant operating normally or is there a fault?
    Stage 2: If fault detected, which fault type is it?

    Parameters:
    -----------
    raw_sensor_data : DataFrame with raw sensor readings (52 sensors)
    scaler          : Fitted StandardScaler
    pca             : Fitted PCA (from normal data)
    detector        : Fitted Stage 1 detector
    stage2_model    : Fitted Stage 2 XGBoost classifier
    process_cols    : Sensor column names

    Returns:
    --------
    results : DataFrame with predictions and confidence metrics
    """
    # Step 1: Handle missing values
    data = raw_sensor_data.copy()
    data[process_cols] = data[process_cols].ffill().bfill()

    # Step 2: Scale using fitted scaler
    scaled_data = scaler.transform(data[process_cols])

    # Step 3: PCA transform
    pca_data = pca.transform(scaled_data)

    # Step 4: Stage 1 - Anomaly Detection
    stage1_pred, t2_values, spe_values = predict_stage1(
        pca_data, scaled_data, pca, detector
    )

    # Step 5: Stage 2 - Fault Diagnosis (only for anomalies)
    fault_predictions = np.zeros(len(stage1_pred), dtype=int)
    anomaly_mask = stage1_pred == 1

    if anomaly_mask.any():
        fault_predictions[anomaly_mask] = predict_stage2(
            pca_data[anomaly_mask],
            stage2_model
        )

    # Step 6: Compile results
    results = pd.DataFrame({
        'Stage1_Result'  : ['ANOMALY' if p == 1 else 'NORMAL' for p in stage1_pred],
        'Fault_Type'     : [FAULT_LABELS.get(f, f'Fault_{f}') for f in fault_predictions],
        'T2_Statistic'   : t2_values.round(4),
        'SPE_Statistic'  : spe_values.round(4),
        'T2_Threshold'   : detector['t2_threshold'],
        'SPE_Threshold'  : detector['spe_threshold'],
        'T2_Alarm'       : t2_values  > detector['t2_threshold'],
        'SPE_Alarm'      : spe_values > detector['spe_threshold']
    })

    return results


# SECTION 5: SAVE AND LOAD PIPELINE


def save_pipeline(scaler, pca, detector, stage2_model, save_dir='models'):
    """
    Save all pipeline components to disk.

    Parameters:
    -----------
    scaler       : Fitted StandardScaler
    pca          : Fitted PCA
    detector     : Stage 1 detector dictionary
    stage2_model : Fitted XGBoost model
    save_dir     : Directory to save models
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(scaler,       f'{save_dir}/scaler.pkl')
    joblib.dump(pca,          f'{save_dir}/pca_normal_baseline.pkl')
    joblib.dump(detector,     f'{save_dir}/stage1_detector.pkl')
    joblib.dump(stage2_model, f'{save_dir}/stage2_xgboost.pkl')
    joblib.dump(PROCESS_COLS, f'{save_dir}/process_cols.pkl')

    print(f"Pipeline saved to '{save_dir}/':")
    print(f"  scaler.pkl")
    print(f"  pca_normal_baseline.pkl")
    print(f"  stage1_detector.pkl")
    print(f"  stage2_xgboost.pkl")
    print(f"  process_cols.pkl")


def load_pipeline(save_dir='models'):
    """
    Load complete pipeline from disk.

    Parameters:
    -----------
    save_dir : Directory containing saved models

    Returns:
    --------
    scaler, pca, detector, stage2_model, process_cols
    """
    scaler       = joblib.load(f'{save_dir}/scaler.pkl')
    pca          = joblib.load(f'{save_dir}/pca_normal_baseline.pkl')
    detector     = joblib.load(f'{save_dir}/stage1_detector.pkl')
    stage2_model = joblib.load(f'{save_dir}/stage2_xgboost.pkl')
    process_cols = joblib.load(f'{save_dir}/process_cols.pkl')

    print("Pipeline loaded successfully!")
    print(f"  Scaler:        {type(scaler).__name__}")
    print(f"  PCA:           {type(pca).__name__} ({pca.n_components_} components)")
    print(f"  Detector:      T^2={detector['t2_threshold']:.2f}, SPE={detector['spe_threshold']:.2f}")
    print(f"  Stage 2 Model: {type(stage2_model).__name__}")

    return scaler, pca, detector, stage2_model, process_cols


# SECTION 6: MAIN - FULL TRAINING RUN


if __name__ == "__main__":

    import pyreadr
    import gc

    print("="*60)
    print("Tennessee Eastman Process")
    print("Fault Detection & Diagnosis Pipeline")
    print("="*60)

    # Load Data
    print("\nLoading datasets...")
    fftr = pyreadr.read_r("TEP_FaultFree_Training.RData")
    fftr_data = fftr['fault_free_training']
    del fftr; gc.collect()

    ftr = pyreadr.read_r("TEP_Faulty_Training.RData")
    ftr_data = ftr['faulty_training']
    del ftr; gc.collect()

    fft = pyreadr.read_r("TEP_FaultFree_Testing.RData")
    fft_data = fft['fault_free_testing']
    del fft; gc.collect()

    ft = pyreadr.read_r("TEP_Faulty_Testing.RData")
    ft_data = ft['faulty_testing']
    del ft; gc.collect()

    print("All datasets loaded!")

    # Fit Scaler
    print("\nFitting StandardScaler on Normal Training data...")
    fftr_clean = handle_missing_values(fftr_data, PROCESS_COLS)
    scaler = fit_scaler(fftr_clean, PROCESS_COLS)

    # Fit PCA
    print("\nFitting PCA on Normal Training data...")
    fftr_scaled = scaler.transform(fftr_clean[PROCESS_COLS])
    pca = fit_pca(fftr_scaled, n_components=N_COMPONENTS)

    # Preprocess All Datasets
    print("\nPreprocessing all datasets...")

    fftr_pca, fftr_scaled, _ = preprocess_dataset(
        fftr_data, scaler, pca, PROCESS_COLS,
        is_faulty=False
    )
    del fftr_data; gc.collect()

    ftr_pca, ftr_scaled, ftr_labels = preprocess_dataset(
        ftr_data, scaler, pca, PROCESS_COLS,
        fault_intro_hours=FAULT_INTRO_TRAINING,
        sampling_interval=SAMPLING_INTERVAL,
        is_faulty=True
    )
    del ftr_data; gc.collect()

    fft_pca, fft_scaled, _ = preprocess_dataset(
        fft_data, scaler, pca, PROCESS_COLS,
        is_faulty=False
    )
    del fft_data; gc.collect()

    ft_pca, ft_scaled, ft_labels = preprocess_dataset(
        ft_data, scaler, pca, PROCESS_COLS,
        fault_intro_hours=FAULT_INTRO_TESTING,
        sampling_interval=SAMPLING_INTERVAL,
        is_faulty=True
    )
    del ft_data; gc.collect()

    # Stage 1: Fit Anomaly Detector
    print("\nFitting Stage 1 Anomaly Detector...")
    detector = fit_stage1_detector(fftr_pca, fftr_scaled, pca, alpha=ALPHA)

    # Stage 1: Evaluate
    print("\nEvaluating Stage 1...")

    # False Positive Rate on Normal Testing
    normal_test_labels = np.zeros(len(fft_pca))
    normal_pred, _, _  = predict_stage1(fft_pca, fft_scaled, pca, detector)
    evaluate_stage1(normal_pred, normal_test_labels, "Normal Testing (FPR)")

    # Detection Rate on Faulty Testing
    faulty_pred, _, _ = predict_stage1(ft_pca, ft_scaled, pca, detector)
    evaluate_stage1(faulty_pred, ft_labels, "Faulty Testing (Detection Rate)")

    # Stage 2: Train/Val Split
    print("\nPreparing Stage 2 training data...")

    train_size = 400 * 480 * 20
    X_train = ftr_pca[:train_size]
    y_train = ftr_labels[:train_size]
    X_val   = ftr_pca[train_size:]
    y_val   = ftr_labels[train_size:]

    # Stratified sampling at simulation run level
    sim_runs = np.repeat(np.arange(1, 501), 480 * 20)
    sim_runs_train = sim_runs[:train_size]

    np.random.seed(42)
    selected_runs = np.random.choice(np.arange(1, 401), size=200, replace=False)
    sample_mask   = np.isin(sim_runs_train, selected_runs)

    X_sample = X_train[sample_mask]
    y_sample = y_train[sample_mask]

    print(f"Training sample: {X_sample.shape}")
    print(f"Validation set:  {X_val.shape}")

    # Stage 2: Fit Classifier
    print("\nFitting Stage 2 XGBoost Classifier...")
    stage2_model = fit_stage2_classifier(X_sample, y_sample, X_val, y_val)

    # Stage 2: Evaluate
    print("\nEvaluating Stage 2...")
    val_pred = predict_stage2(X_val, stage2_model)
    df_results, macro_f1 = evaluate_stage2(val_pred, y_val)

    # Save Pipeline
    print("\nSaving pipeline...")
    save_pipeline(scaler, pca, detector, stage2_model)

    print("\n" + "="*60)
    print("Pipeline Training Complete!")
    print(f"Stage 2 Macro F1 Score: {macro_f1:.4f}")
    print("="*60)
