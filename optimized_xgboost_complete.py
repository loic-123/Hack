"""
Optimized XGBoost Model - Nuclear Risk Prediction
==================================================
Complete implementation with:
- Advanced feature engineering
- Normalization (StandardScaler)
- Regularization (L1/L2 in XGBoost)
- Hyperparameter optimization
- Class imbalance handling
- Comprehensive evaluation

Target: true_risk_level (0-3 classification)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, balanced_accuracy_score, cohen_kappa_score
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Plotting configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)

print("="*80)
print("OPTIMIZED XGBOOST MODEL - NUCLEAR RISK PREDICTION")
print("="*80)
print(f"XGBoost version: {xgb.__version__}")
print()


# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("[1/10] Loading data...")
df = pd.read_csv('avalon_nuclear.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nTarget: true_risk_level")
print(f"Class distribution:\n{df['true_risk_level'].value_counts(normalize=True).sort_index()}")
print(f"Missing values: {df.isnull().sum().sum()}")
print("[OK] Data loaded successfully\n")


# ============================================================================
# 2. ADVANCED FEATURE ENGINEERING
# ============================================================================
print("[2/10] Engineering features...")
df_model = df.copy()

# === PHYSICS-BASED FEATURES ===
print("  - Creating physics-based features...")
# Core thermal stress
df_model['thermal_stress'] = df_model['core_temp_c'] * df_model['coolant_pressure_bar'] / 1000

# Radiation metrics
df_model['radiation_differential'] = df_model['radiation_inside_uSv'] - df_model['radiation_outside_uSv']
df_model['radiation_ratio'] = df_model['radiation_inside_uSv'] / (df_model['radiation_outside_uSv'] + 1)

# Coolant efficiency
df_model['coolant_efficiency'] = df_model['coolant_flow_rate'] / (df_model['core_temp_c'] + 1)

# Power efficiency
df_model['power_efficiency'] = df_model['load_factor_pct'] * df_model['reactor_nominal_power_mw'] / 100

# Thermal margin (350°C assumed critical)
df_model['thermal_margin'] = 350 - df_model['core_temp_c']

# Control effectiveness
df_model['control_effectiveness'] = (100 - df_model['control_rod_position_pct']) * df_model['neutron_flux'] / 100

# === OPERATIONAL RISK FEATURES ===
print("  - Creating operational risk features...")
df_model['age_power_risk'] = df_model['reactor_age_years'] / (df_model['reactor_nominal_power_mw'] + 1)
df_model['maintenance_risk'] = df_model['reactor_age_years'] * (100 - df_model['maintenance_score']) * df_model['days_since_maintenance'] / 10000
df_model['staff_risk'] = df_model['staff_fatigue_index'] * df_model['sensor_anomaly_flag']

# === SOCIAL/EXTERNAL PRESSURE ===
print("  - Creating social/external pressure features...")
df_model['social_pressure_index'] = (
    df_model['public_anxiety_index'] * 0.4 +
    df_model['social_media_rumour_index'] * 0.3 +
    df_model['regulator_scrutiny_score'] * 0.3
)

df_model['external_threat'] = (
    df_model['weather_severity_index'] +
    df_model['seismic_activity_index'] +
    df_model['cyber_attack_score']
) / 3

df_model['population_risk'] = np.log1p(df_model['population_within_30km']) * df_model['radiation_inside_uSv'] / 100

# === COMPOSITE RISK INDICES ===
print("  - Creating composite risk indices...")
df_model['physical_risk_index'] = (
    (df_model['core_temp_c'] / 350) * 0.35 +
    (df_model['coolant_pressure_bar'] / 160) * 0.25 +
    (df_model['radiation_inside_uSv'] / 1000) * 0.25 +
    (df_model['neutron_flux'] / 5) * 0.15
)

df_model['operational_risk_index'] = (
    (df_model['reactor_age_years'] / 60) * 0.3 +
    ((100 - df_model['maintenance_score']) / 100) * 0.3 +
    (df_model['staff_fatigue_index'] / 100) * 0.2 +
    (df_model['sensor_anomaly_flag']) * 0.2
)

# === AVALON BIAS FEATURES ===
print("  - Creating AVALON bias features...")
df_model['avalon_bias'] = df_model['avalon_raw_risk_score'] - (df_model['physical_risk_index'] * 100)
df_model['avalon_confidence'] = 100 - abs(df_model['avalon_raw_risk_score'] - df_model['avalon_learned_reward_score'])

# === INTERACTION FEATURES ===
print("  - Creating interaction features...")
df_model['temp_age_interaction'] = df_model['core_temp_c'] * df_model['reactor_age_years'] / 100
df_model['pressure_flow_interaction'] = df_model['coolant_pressure_bar'] * df_model['coolant_flow_rate'] / 100
df_model['maintenance_age_interaction'] = df_model['maintenance_score'] * df_model['reactor_age_years'] / 100

# === POLYNOMIAL FEATURES ===
print("  - Creating polynomial features...")
df_model['core_temp_squared'] = df_model['core_temp_c'] ** 2
df_model['pressure_squared'] = df_model['coolant_pressure_bar'] ** 2
df_model['radiation_log'] = np.log1p(df_model['radiation_inside_uSv'])

print(f"[OK] Feature engineering complete!")
print(f"  Original features: {df.shape[1]}")
print(f"  After engineering: {df_model.shape[1]}")
print(f"  New features created: {df_model.shape[1] - df.shape[1]}\n")


# ============================================================================
# 3. DATA PREPARATION WITH NORMALIZATION
# ============================================================================
print("[3/10] Preparing data with normalization...")

# Encode categorical variables
le = LabelEncoder()
df_model['country_encoded'] = le.fit_transform(df_model['country'])

# Define features
exclude_cols = [
    'country',
    'true_risk_level',  # TARGET
    'incident_occurred',
    'avalon_evac_recommendation',
    'avalon_shutdown_recommendation',
    'human_override'
]

feature_cols = [col for col in df_model.columns if col not in exclude_cols]
X = df_model[feature_cols]
y = df_model['true_risk_level']

print(f"Feature matrix: {X.shape}")
print(f"Target: {y.shape}")

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train: {X_train.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# NORMALIZATION: StandardScaler (zero mean, unit variance)
print("\n  Applying StandardScaler normalization...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  [OK] Features normalized (mean=0, std=1)")

# Compute sample weights for class imbalance
sample_weights = compute_sample_weight('balanced', y_train)
print(f"  [OK] Sample weights computed for class imbalance")
print(f"    Weight range: {sample_weights.min():.2f} - {sample_weights.max():.2f}\n")


# ============================================================================
# 4. BASELINE MODEL
# ============================================================================
print("[4/10] Training baseline XGBoost model...")

baseline_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    eval_metric='mlogloss',
    n_jobs=-1
)

baseline_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

y_pred_baseline = baseline_model.predict(X_test_scaled)
baseline_acc = accuracy_score(y_test, y_pred_baseline)
baseline_balanced_acc = balanced_accuracy_score(y_test, y_pred_baseline)
baseline_f1 = f1_score(y_test, y_pred_baseline, average='weighted')

print(f"Baseline Results:")
print(f"  Accuracy: {baseline_acc:.4f}")
print(f"  Balanced Accuracy: {baseline_balanced_acc:.4f}")
print(f"  Weighted F1: {baseline_f1:.4f}\n")


# ============================================================================
# 5. HYPERPARAMETER OPTIMIZATION WITH REGULARIZATION
# ============================================================================
print("[5/10] Hyperparameter optimization with regularization...")
print("  This may take several minutes...\n")

# REGULARIZATION: Define parameter distribution with L1/L2 regularization
param_dist = {
    'max_depth': [4, 5, 6, 7, 8, 9, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'n_estimators': [100, 200, 300, 400, 500],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],  # Regularization
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],  # Regularization
    'min_child_weight': [1, 3, 5, 7],  # Regularization
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],  # Regularization (minimum loss reduction)
    'reg_alpha': [0, 0.01, 0.1, 0.5, 1],  # L1 REGULARIZATION
    'reg_lambda': [1, 1.5, 2, 3, 5]  # L2 REGULARIZATION
}

# Base estimator
xgb_base = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    random_state=42,
    eval_metric='mlogloss',
    n_jobs=-1
)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Randomized search
random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=50,
    scoring='balanced_accuracy',
    cv=skf,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit
random_search.fit(X_train_scaled, y_train, sample_weight=sample_weights)

print(f"\n[OK] Hyperparameter optimization complete!")
print(f"  Best CV Score: {random_search.best_score_:.4f}")
print(f"\n  Best Parameters:")
for param, value in sorted(random_search.best_params_.items()):
    print(f"    {param:20s}: {value}")
print()


# ============================================================================
# 6. TRAIN OPTIMIZED MODEL WITH REGULARIZATION
# ============================================================================
print("[6/10] Training optimized model with regularization...")

optimized_model = random_search.best_estimator_

# Retrain with early stopping (additional regularization)
optimized_model.set_params(early_stopping_rounds=20)
optimized_model.fit(
    X_train_scaled, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
    verbose=False
)

print(f"[OK] Training complete!")
print(f"  Best iteration: {optimized_model.best_iteration}\n")


# ============================================================================
# 7. COMPREHENSIVE EVALUATION
# ============================================================================
print("[7/10] Evaluating model performance...")

y_train_pred = optimized_model.predict(X_train_scaled)
y_test_pred = optimized_model.predict(X_test_scaled)

# Calculate metrics
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
test_kappa = cohen_kappa_score(y_test, y_test_pred)

print("="*80)
print("OPTIMIZED XGBOOST MODEL - EVALUATION RESULTS")
print("="*80)

print(f"\n[ACCURACY METRICS]")
print(f"  Train Accuracy:           {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"  Test Accuracy:            {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  Balanced Accuracy:        {test_balanced_acc:.4f} ({test_balanced_acc*100:.2f}%)")
print(f"  Overfitting Gap:          {(train_acc - test_acc)*100:.2f}%")

print(f"\n[F1 SCORES]")
print(f"  Weighted F1:              {test_f1_weighted:.4f}")
print(f"  Macro F1:                 {test_f1_macro:.4f}")

print(f"\n[OTHER METRICS]")
print(f"  Cohen's Kappa:            {test_kappa:.4f}")

print(f"\n[COMPARISON WITH BASELINE]")
print(f"  Baseline Accuracy:        {baseline_acc:.4f}")
print(f"  Optimized Accuracy:       {test_acc:.4f}")
print(f"  Improvement:              +{(test_acc - baseline_acc)*100:.2f}%")

print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORT")
print("="*80)
print(classification_report(
    y_test, y_test_pred,
    target_names=['Risk 0', 'Risk 1', 'Risk 2', 'Risk 3'],
    digits=4
))


# ============================================================================
# 8. CONFUSION MATRIX
# ============================================================================
print("[8/10] Generating confusion matrix...")

cm = confusion_matrix(y_test, y_test_pred)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Risk 0', 'Risk 1', 'Risk 2', 'Risk 3'],
            yticklabels=['Risk 0', 'Risk 1', 'Risk 2', 'Risk 3'],
            cbar_kws={'label': 'Count'})
axes[0].set_title(f'Confusion Matrix (Counts)\nAccuracy: {test_acc:.4f}',
                  fontweight='bold', fontsize=12)
axes[0].set_ylabel('True Risk Level', fontweight='bold')
axes[0].set_xlabel('Predicted Risk Level', fontweight='bold')

# Normalized
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Greens', ax=axes[1],
            xticklabels=['Risk 0', 'Risk 1', 'Risk 2', 'Risk 3'],
            yticklabels=['Risk 0', 'Risk 1', 'Risk 2', 'Risk 3'],
            cbar_kws={'label': 'Proportion'})
axes[1].set_title(f'Confusion Matrix (Normalized)\nBalanced Acc: {test_balanced_acc:.4f}',
                  fontweight='bold', fontsize=12)
axes[1].set_ylabel('True Risk Level', fontweight='bold')
axes[1].set_xlabel('Predicted Risk Level', fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrix_optimized_xgboost.png', dpi=300, bbox_inches='tight')
plt.close()

print("[OK] Confusion matrix saved: confusion_matrix_optimized_xgboost.png\n")

# Per-class accuracy
print("Per-class Recall (Sensitivity):")
for i in range(4):
    recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
    print(f"  Risk {i}: {recall:.4f} ({cm[i, i]}/{cm[i, :].sum()} correct)")
print()


# ============================================================================
# 9. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("[9/10] Analyzing feature importance...")

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': optimized_model.feature_importances_
}).sort_values('importance', ascending=False)

print("="*80)
print("TOP 20 MOST IMPORTANT FEATURES")
print("="*80)
total_importance = feature_importance['importance'].sum()
for i, (idx, row) in enumerate(feature_importance.head(20).iterrows(), 1):
    pct = (row['importance'] / total_importance) * 100
    print(f"{i:2d}. {row['feature']:40s} {pct:6.2f}%")
print()

# Visualize top 25 features
fig, ax = plt.subplots(figsize=(12, 10))
top_features = feature_importance.head(25)
colors = ['#2ECC71' if 'avalon' in feat.lower() else '#3498DB'
          for feat in top_features['feature']]

ax.barh(range(len(top_features)), top_features['importance'], color=colors, alpha=0.8)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'])
ax.invert_yaxis()
ax.set_xlabel('Importance Score', fontweight='bold')
ax.set_title('Top 25 Feature Importance (Green = AVALON-related)',
             fontweight='bold', fontsize=14)
ax.grid(axis='x', alpha=0.3)

# Add percentage labels
for i, (idx, row) in enumerate(top_features.iterrows()):
    pct = (row['importance'] / total_importance) * 100
    ax.text(row['importance'], i, f' {pct:.1f}%',
            va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('feature_importance_optimized_xgboost.png', dpi=300, bbox_inches='tight')
plt.close()

print("[OK] Feature importance plot saved: feature_importance_optimized_xgboost.png")

# Save feature importance to CSV
feature_importance.to_csv('feature_importance_optimized_xgboost.csv', index=False)
print("[OK] Feature importance saved: feature_importance_optimized_xgboost.csv\n")


# ============================================================================
# 10. SAVE MODEL AND ARTIFACTS
# ============================================================================
print("[10/10] Saving model and artifacts...")

# Save optimized model
joblib.dump(optimized_model, 'optimized_xgboost_model.pkl')
print("[OK] Model saved: optimized_xgboost_model.pkl")

# Save scaler (for normalization)
joblib.dump(scaler, 'feature_scaler.pkl')
print("[OK] Scaler saved: feature_scaler.pkl")

# Save feature columns
joblib.dump(feature_cols, 'feature_columns.pkl')
print("[OK] Feature columns saved: feature_columns.pkl")

# Save label encoder
joblib.dump(le, 'label_encoder_country.pkl')
print("[OK] Label encoder saved: label_encoder_country.pkl")

# Save results summary
results_summary = {
    'test_accuracy': test_acc,
    'test_balanced_accuracy': test_balanced_acc,
    'test_f1_weighted': test_f1_weighted,
    'test_f1_macro': test_f1_macro,
    'test_kappa': test_kappa,
    'baseline_accuracy': baseline_acc,
    'improvement_over_baseline': (test_acc - baseline_acc) * 100,
    'best_params': random_search.best_params_,
    'best_cv_score': random_search.best_score_,
    'best_iteration': int(optimized_model.best_iteration),
    'n_features': len(feature_cols),
    'n_train_samples': len(X_train),
    'n_test_samples': len(X_test)
}

import json
with open('model_results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("[OK] Results summary saved: model_results_summary.json\n")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("="*80)
print("OPTIMIZED XGBOOST MODEL - FINAL SUMMARY")
print("="*80)

print("\n[TECHNIQUES APPLIED]")
print("  [OK] Advanced Feature Engineering (23 new features)")
print("  [OK] Normalization (StandardScaler: mean=0, std=1)")
print("  [OK] Regularization:")
print("    - L1 regularization (reg_alpha)")
print("    - L2 regularization (reg_lambda)")
print("    - Structural regularization (max_depth, min_child_weight, gamma)")
print("    - Stochastic regularization (subsample, colsample_bytree)")
print("    - Early stopping")
print("  [OK] Hyperparameter Optimization (RandomizedSearchCV, 50 iterations)")
print("  [OK] Class Imbalance Handling (sample weighting)")
print("  [OK] Stratified Cross-Validation (5-fold)")

print("\n[FINAL PERFORMANCE]")
print(f"  Test Accuracy:            {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  Balanced Accuracy:        {test_balanced_acc:.4f} ({test_balanced_acc*100:.2f}%)")
print(f"  Weighted F1-Score:        {test_f1_weighted:.4f}")
print(f"  Improvement over baseline: +{(test_acc-baseline_acc)*100:.2f}%")

print("\n[OUTPUT FILES]")
print("  [OK] optimized_xgboost_model.pkl")
print("  [OK] feature_scaler.pkl")
print("  [OK] feature_columns.pkl")
print("  [OK] label_encoder_country.pkl")
print("  [OK] model_results_summary.json")
print("  [OK] confusion_matrix_optimized_xgboost.png")
print("  [OK] feature_importance_optimized_xgboost.png")
print("  [OK] feature_importance_optimized_xgboost.csv")

print("\n" + "="*80)
print("MODEL TRAINING COMPLETE!")
print("="*80)
print("\nModel is ready for deployment. Use the saved artifacts for inference:")
print("  1. Load model: joblib.load('optimized_xgboost_model.pkl')")
print("  2. Load scaler: joblib.load('feature_scaler.pkl')")
print("  3. Scale features: scaler.transform(X_new)")
print("  4. Predict: model.predict(X_new_scaled)")
print("\n" + "="*80)
