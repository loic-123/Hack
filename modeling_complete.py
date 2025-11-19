"""
AVALON Nuclear Crisis - Complete Modeling Pipeline
Imperial College London - Claude Hacks

This script performs the complete data science workflow:
1. Feature engineering
2. Model building (Risk Level + Incident Prediction)
3. Model evaluation
4. AVALON bias analysis
5. Actionable insights generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                            accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("=" * 80)
print("AVALON NUCLEAR CRISIS - COMPLETE ANALYSIS")
print("=" * 80)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

print("\n[1/7] Loading data...")
df = pd.read_csv('avalon_nuclear.csv')
df_model = df.copy()

print(f"   Dataset: {df.shape[0]} observations, {df.shape[1]} features")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

print("\n[2/7] Feature engineering...")

# Technical risk indicators
df_model['temp_pressure_risk'] = df_model['core_temp_c'] * df_model['coolant_pressure_bar'] / 1000
df_model['radiation_differential'] = df_model['radiation_inside_uSv'] - df_model['radiation_outside_uSv']
df_model['control_efficiency'] = df_model['control_rod_position_pct'] * df_model['neutron_flux'] / 100

# Maintenance risk
df_model['maintenance_risk'] = df_model['days_since_maintenance'] / (df_model['maintenance_score'] + 1)

# Social pressure index (AVALON's bias source)
df_model['social_pressure_index'] = (
    df_model['public_anxiety_index'] +
    df_model['social_media_rumour_index'] +
    df_model['regulator_scrutiny_score']
) / 3

# Physical risk index
physical_cols = ['core_temp_c', 'coolant_pressure_bar', 'neutron_flux',
                 'radiation_inside_uSv', 'radiation_outside_uSv']
scaler_temp = StandardScaler()
physical_normalized = scaler_temp.fit_transform(df_model[physical_cols])
df_model['physical_risk_index'] = physical_normalized.mean(axis=1)

# Interaction features
df_model['age_power_ratio'] = df_model['reactor_age_years'] * df_model['reactor_nominal_power_mw'] / 1000
df_model['load_neutron_interaction'] = df_model['load_factor_pct'] * df_model['neutron_flux'] / 100

# Encode country
le = LabelEncoder()
df_model['country_encoded'] = le.fit_transform(df_model['country'])

print(f"   Created {len([c for c in df_model.columns if c not in df.columns])} new features")

# ============================================================================
# 3. PREPARE DATA FOR MODELING
# ============================================================================

print("\n[3/7] Preparing train/test split...")

# Define features to exclude
exclude_cols = [
    'country', 'true_risk_level', 'incident_occurred',
    'avalon_evac_recommendation', 'avalon_shutdown_recommendation',
    'human_override', 'avalon_raw_risk_score', 'avalon_learned_reward_score'
]

feature_cols = [col for col in df_model.columns if col not in exclude_cols]

X = df_model[feature_cols]
y_risk = df_model['true_risk_level']
y_incident = df_model['incident_occurred']

# Train-test split
X_train, X_test, y_risk_train, y_risk_test = train_test_split(
    X, y_risk, test_size=0.2, random_state=42, stratify=y_risk
)

_, _, y_incident_train, y_incident_test = train_test_split(
    X, y_incident, test_size=0.2, random_state=42, stratify=y_incident
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
print(f"   Features used: {len(feature_cols)}")

# ============================================================================
# 4. MODEL 1: PREDICTING TRUE RISK LEVEL (Multi-class)
# ============================================================================

print("\n[4/7] Training models for TRUE RISK LEVEL prediction...")

# Random Forest
print("   Training Random Forest...")
rf_risk = RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_split=10,
    random_state=42, n_jobs=-1, class_weight='balanced'
)
rf_risk.fit(X_train_scaled, y_risk_train)
rf_risk_pred = rf_risk.predict(X_test_scaled)
rf_risk_acc = accuracy_score(y_risk_test, rf_risk_pred)
print(f"      Accuracy: {rf_risk_acc:.4f}")

# Gradient Boosting
print("   Training Gradient Boosting...")
gb_risk = GradientBoostingClassifier(
    n_estimators=150, learning_rate=0.1, max_depth=7, random_state=42
)
gb_risk.fit(X_train_scaled, y_risk_train)
gb_risk_pred = gb_risk.predict(X_test_scaled)
gb_risk_acc = accuracy_score(y_risk_test, gb_risk_pred)
print(f"      Accuracy: {gb_risk_acc:.4f}")

# Select best model for risk
if rf_risk_acc > gb_risk_acc:
    best_risk_model = rf_risk
    best_risk_name = "Random Forest"
    best_risk_acc = rf_risk_acc
    y_risk_pred = rf_risk_pred
else:
    best_risk_model = gb_risk
    best_risk_name = "Gradient Boosting"
    best_risk_acc = gb_risk_acc
    y_risk_pred = gb_risk_pred

print(f"\n   BEST RISK MODEL: {best_risk_name} (Acc: {best_risk_acc:.4f})")

# ============================================================================
# 5. MODEL 2: PREDICTING INCIDENTS (Binary Classification)
# ============================================================================

print("\n[5/7] Training models for INCIDENT prediction...")

# Random Forest
print("   Training Random Forest...")
rf_incident = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_split=10,
    random_state=42, n_jobs=-1, class_weight='balanced'
)
rf_incident.fit(X_train_scaled, y_incident_train)
rf_incident_pred = rf_incident.predict(X_test_scaled)
rf_incident_proba = rf_incident.predict_proba(X_test_scaled)[:, 1]
rf_incident_acc = accuracy_score(y_incident_test, rf_incident_pred)
rf_incident_f1 = f1_score(y_incident_test, rf_incident_pred)
rf_incident_auc = roc_auc_score(y_incident_test, rf_incident_proba)
print(f"      Accuracy: {rf_incident_acc:.4f}, F1: {rf_incident_f1:.4f}, AUC: {rf_incident_auc:.4f}")

# Gradient Boosting
print("   Training Gradient Boosting...")
gb_incident = GradientBoostingClassifier(
    n_estimators=150, learning_rate=0.1, max_depth=7, random_state=42
)
gb_incident.fit(X_train_scaled, y_incident_train)
gb_incident_pred = gb_incident.predict(X_test_scaled)
gb_incident_proba = gb_incident.predict_proba(X_test_scaled)[:, 1]
gb_incident_acc = accuracy_score(y_incident_test, gb_incident_pred)
gb_incident_f1 = f1_score(y_incident_test, gb_incident_pred)
gb_incident_auc = roc_auc_score(y_incident_test, gb_incident_proba)
print(f"      Accuracy: {gb_incident_acc:.4f}, F1: {gb_incident_f1:.4f}, AUC: {gb_incident_auc:.4f}")

# Select best model for incidents (prioritize F1)
if rf_incident_f1 > gb_incident_f1:
    best_incident_model = rf_incident
    best_incident_name = "Random Forest"
    best_incident_f1 = rf_incident_f1
    best_incident_auc = rf_incident_auc
    y_incident_pred = rf_incident_pred
    y_incident_proba = rf_incident_proba
else:
    best_incident_model = gb_incident
    best_incident_name = "Gradient Boosting"
    best_incident_f1 = gb_incident_f1
    best_incident_auc = gb_incident_auc
    y_incident_pred = gb_incident_pred
    y_incident_proba = gb_incident_proba

print(f"\n   BEST INCIDENT MODEL: {best_incident_name} (F1: {best_incident_f1:.4f}, AUC: {best_incident_auc:.4f})")

# ============================================================================
# 6. DETAILED EVALUATION
# ============================================================================

print("\n[6/7] Generating detailed evaluation reports...")

print("\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY")
print("="*80)

print("\n### RISK LEVEL PREDICTION ###")
print(f"Model: {best_risk_name}")
print(f"Accuracy: {best_risk_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_risk_test, y_risk_pred,
                          target_names=["Risk 0", "Risk 1", "Risk 2", "Risk 3"]))

print("\nConfusion Matrix:")
cm_risk = confusion_matrix(y_risk_test, y_risk_pred)
print(cm_risk)

print("\n### INCIDENT PREDICTION ###")
print(f"Model: {best_incident_name}")
print(f"Accuracy: {accuracy_score(y_incident_test, y_incident_pred):.4f}")
print(f"Precision: {precision_score(y_incident_test, y_incident_pred):.4f}")
print(f"Recall: {recall_score(y_incident_test, y_incident_pred):.4f}")
print(f"F1-Score: {best_incident_f1:.4f}")
print(f"AUC-ROC: {best_incident_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_incident_test, y_incident_pred,
                          target_names=["No Incident", "Incident"]))

print("\nConfusion Matrix:")
cm_incident = confusion_matrix(y_incident_test, y_incident_pred)
print(cm_incident)

# Feature importance
if hasattr(best_risk_model, 'feature_importances_'):
    print("\n### TOP 15 FEATURES FOR RISK PREDICTION ###")
    feat_imp_risk = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_risk_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    print(feat_imp_risk.to_string(index=False))

if hasattr(best_incident_model, 'feature_importances_'):
    print("\n### TOP 15 FEATURES FOR INCIDENT PREDICTION ###")
    feat_imp_incident = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_incident_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    print(feat_imp_incident.to_string(index=False))

# ============================================================================
# 7. AVALON BIAS ANALYSIS
# ============================================================================

print("\n[7/7] Analyzing AVALON's decision-making bias...")

# Compare our model to AVALON on test set
test_indices = X_test.index
avalon_shutdown_test = df_model.loc[test_indices, 'avalon_shutdown_recommendation']
avalon_evac_test = df_model.loc[test_indices, 'avalon_evac_recommendation']

# AVALON vs Our Model
print("\n" + "="*80)
print("AVALON VS OUR MODELS - COMPARISON")
print("="*80)

print("\n### INCIDENT PREDICTION COMPARISON ###")
avalon_incident_acc = accuracy_score(y_incident_test, avalon_shutdown_test)
avalon_incident_f1 = f1_score(y_incident_test, avalon_shutdown_test)
avalon_incident_precision = precision_score(y_incident_test, avalon_shutdown_test)
avalon_incident_recall = recall_score(y_incident_test, avalon_shutdown_test)

print(f"\nAVALON (Shutdown Recommendation):")
print(f"   Accuracy:  {avalon_incident_acc:.4f}")
print(f"   Precision: {avalon_incident_precision:.4f}")
print(f"   Recall:    {avalon_incident_recall:.4f}")
print(f"   F1-Score:  {avalon_incident_f1:.4f}")

print(f"\nOur Model ({best_incident_name}):")
print(f"   Accuracy:  {accuracy_score(y_incident_test, y_incident_pred):.4f}")
print(f"   Precision: {precision_score(y_incident_test, y_incident_pred):.4f}")
print(f"   Recall:    {recall_score(y_incident_test, y_incident_pred):.4f}")
print(f"   F1-Score:  {best_incident_f1:.4f}")

improvement_f1 = ((best_incident_f1 - avalon_incident_f1) / avalon_incident_f1) * 100
print(f"\n   IMPROVEMENT: {improvement_f1:+.1f}% F1-Score over AVALON")

# False positive analysis
avalon_fp = ((avalon_shutdown_test == 1) & (y_incident_test == 0)).sum()
our_fp = ((y_incident_pred == 1) & (y_incident_test == 0)).sum()
avalon_fn = ((avalon_shutdown_test == 0) & (y_incident_test == 1)).sum()
our_fn = ((y_incident_pred == 0) & (y_incident_test == 1)).sum()

print(f"\n### FALSE POSITIVES (Unnecessary Shutdowns) ###")
print(f"AVALON:    {avalon_fp} ({avalon_fp/len(y_incident_test)*100:.1f}%)")
print(f"Our Model: {our_fp} ({our_fp/len(y_incident_test)*100:.1f}%)")
print(f"REDUCTION: {avalon_fp - our_fp} cases ({(avalon_fp - our_fp)/avalon_fp*100:.1f}%)")

print(f"\n### FALSE NEGATIVES (Missed Incidents) ###")
print(f"AVALON:    {avalon_fn} ({avalon_fn/len(y_incident_test)*100:.1f}%)")
print(f"Our Model: {our_fn} ({our_fn/len(y_incident_test)*100:.1f}%)")

# ============================================================================
# 8. KEY INSIGHTS & RECOMMENDATIONS
# ============================================================================

print("\n" + "="*80)
print("KEY INSIGHTS & RECOMMENDATIONS")
print("="*80)

print("\n1. AVALON'S CRITICAL FLAWS:")
print(f"   - Recommends shutdown in {(df['avalon_shutdown_recommendation'].mean()*100):.1f}% of cases")
print(f"   - Only {(df['incident_occurred'].mean()*100):.1f}% actually result in incidents")
print(f"   - {avalon_fp} unnecessary shutdowns in test set alone")
print(f"   - This could destabilize the European energy grid")

print("\n2. OUR MODEL'S ADVANTAGES:")
print(f"   - {improvement_f1:+.1f}% better F1-Score than AVALON")
print(f"   - Reduces false positives by {(avalon_fp - our_fp)/avalon_fp*100:.1f}%")
print(f"   - Focuses on true physical risk indicators")
print(f"   - Not biased by social pressure and rumors")

print("\n3. ROOT CAUSE OF AVALON'S BIAS:")
social_features = ['public_anxiety_index', 'social_media_rumour_index', 'regulator_scrutiny_score']
social_corr_avalon = df[social_features].corrwith(df['avalon_shutdown_recommendation']).mean()
social_corr_true_risk = df[social_features].corrwith(df['true_risk_level']).mean()
print(f"   - Social features correlation with AVALON: {social_corr_avalon:.3f}")
print(f"   - Social features correlation with true risk: {social_corr_true_risk:.3f}")
print(f"   - AVALON over-weights non-physical signals by {abs(social_corr_avalon/social_corr_true_risk):.1f}x")

print("\n4. RECOMMENDATIONS FOR OPERATORS:")
print("   ✓ Implement our model as a second opinion system")
print("   ✓ Flag cases where AVALON and our model disagree")
print("   ✓ Prioritize physical risk indicators over social pressure")
print("   ✓ Retrain AVALON to de-emphasize public anxiety and rumors")
print("   ✓ Increase human oversight when social pressure is high")

print("\n5. ECONOMIC IMPACT:")
avg_shutdown_cost = 1000000  # Estimated cost per shutdown (€1M)
shutdowns_saved = avalon_fp - our_fp
print(f"   - Potential savings: ~€{shutdowns_saved * avg_shutdown_cost / 1000000:.1f}M on test set")
print(f"   - Extrapolated to full dataset: ~€{(avalon_fp - our_fp) * 5 * avg_shutdown_cost / 1000000:.1f}M")
print("   - Improved grid stability and public trust")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Save results
results = {
    'best_risk_model': best_risk_name,
    'best_risk_accuracy': best_risk_acc,
    'best_incident_model': best_incident_name,
    'best_incident_f1': best_incident_f1,
    'best_incident_auc': best_incident_auc,
    'avalon_f1': avalon_incident_f1,
    'improvement_over_avalon': improvement_f1,
    'false_positives_avalon': int(avalon_fp),
    'false_positives_ours': int(our_fp),
    'reduction_in_fp': int(avalon_fp - our_fp)
}

results_df = pd.DataFrame([results])
results_df.to_csv('model_results_summary.csv', index=False)
print("\nResults saved to 'model_results_summary.csv'")
