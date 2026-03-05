# SPH6004 Assignment - Feature Selection
# MIMIC-IV v3.1: predicting ICU discharge (survival)

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, mutual_info_classif, RFE
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# load data
df = pd.read_csv('/home/ubuntu/upload/Assignment1_mimicdataset(in).csv')
print(f"dataset shape: {df.shape}")

# target: discharge = 1 (survived), 0 (died in ICU)
df['discharge'] = 1 - df['icu_death_flag']
print(df['discharge'].value_counts())
print(f"discharge rate: {df['discharge'].mean()*100:.1f}%")

# drop ID columns, time columns, and leakage variables
drop_cols = [
    'subject_id', 'hadm_id', 'stay_id',
    'intime', 'outtime', 'deathtime',
    'los',
    'hospital_expire_flag',
    'icu_death_flag',
]
df_clean = df.drop(columns=drop_cols)
print(f"after dropping id/time/leakage cols: {df_clean.shape}")

# encode categorical vars
cat_cols = ['first_careunit', 'last_careunit', 'insurance', 'language',
            'race', 'marital_status', 'gender']

race_map = {
    'WHITE': 'WHITE', 'WHITE - OTHER EUROPEAN': 'WHITE', 'WHITE - RUSSIAN': 'WHITE',
    'WHITE - BRAZILIAN': 'WHITE', 'WHITE - EASTERN EUROPEAN': 'WHITE',
    'BLACK/AFRICAN AMERICAN': 'BLACK', 'BLACK/AFRICAN': 'BLACK',
    'BLACK/CAPE VERDEAN': 'BLACK', 'BLACK/CARIBBEAN ISLAND': 'BLACK',
    'ASIAN': 'ASIAN', 'ASIAN - CHINESE': 'ASIAN', 'ASIAN - SOUTH EAST ASIAN': 'ASIAN',
    'ASIAN - ASIAN INDIAN': 'ASIAN', 'ASIAN - KOREAN': 'ASIAN',
    'HISPANIC/LATINO - PUERTO RICAN': 'HISPANIC', 'HISPANIC/LATINO - DOMINICAN': 'HISPANIC',
    'HISPANIC/LATINO - GUATEMALAN': 'HISPANIC', 'HISPANIC/LATINO - CUBAN': 'HISPANIC',
    'HISPANIC/LATINO - SALVADORAN': 'HISPANIC', 'HISPANIC/LATINO - CENTRAL AMERICAN': 'HISPANIC',
    'HISPANIC/LATINO - MEXICAN': 'HISPANIC', 'HISPANIC/LATINO - COLOMBIAN': 'HISPANIC',
    'HISPANIC/LATINO - HONDURAN': 'HISPANIC', 'HISPANIC OR LATINO': 'HISPANIC',
}
df_clean['race'] = df_clean['race'].map(lambda x: race_map.get(x, 'OTHER'))
df_clean['language'] = df_clean['language'].apply(
    lambda x: x if x == 'English' else 'Non-English' if pd.notna(x) else x
)

for col in cat_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna('UNKNOWN')
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))

# drop features with >50% missing
missing_pct = df_clean.drop(columns=['discharge']).isnull().mean()
high_missing = missing_pct[missing_pct > 0.50].index.tolist()
print(f"dropping {len(high_missing)} high-missing features")
df_clean = df_clean.drop(columns=high_missing)

# impute remaining NaNs with median
feature_cols = [c for c in df_clean.columns if c != 'discharge']
imputer = SimpleImputer(strategy='median')
df_clean[feature_cols] = imputer.fit_transform(df_clean[feature_cols])

X = df_clean.drop(columns=['discharge'])
y = df_clean['discharge']
all_features = X.columns.tolist()
print(f"features before selection: {len(all_features)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)


# method 1: variance threshold
vt = VarianceThreshold(threshold=0.01)
vt.fit(X_train_scaled)
vt_mask = vt.get_support()
vt_features = X_train.columns[vt_mask].tolist()
removed_vt = X_train.columns[~vt_mask].tolist()
print(f"variance threshold: kept {len(vt_features)}, removed {removed_vt}")

# method 2: correlation filter
corr_matrix = X_train_scaled.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop_corr = set()
threshold_corr = 0.90

for col in upper.columns:
    correlated = upper.index[upper[col] > threshold_corr].tolist()
    for c in correlated:
        corr_with_target_c = abs(X_train_scaled[c].corr(y_train))
        corr_with_target_col = abs(X_train_scaled[col].corr(y_train))
        if corr_with_target_c < corr_with_target_col:
            to_drop_corr.add(c)
        else:
            to_drop_corr.add(col)

corr_features = [f for f in X_train.columns if f not in to_drop_corr]
print(f"after correlation filter: {len(corr_features)} features kept")

# method 3: ANOVA F-test
selector_anova = SelectKBest(f_classif, k='all')
selector_anova.fit(X_train_scaled, y_train)
anova_scores = pd.Series(selector_anova.scores_, index=X_train.columns).sort_values(ascending=False)
anova_pvalues = pd.Series(selector_anova.pvalues_, index=X_train.columns)
anova_significant = anova_pvalues[anova_pvalues < 0.05].index.tolist()
print(f"ANOVA significant features (p<0.05): {len(anova_significant)}")

# method 4: L1 regularization
lasso = LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=5000, random_state=42)
lasso.fit(X_train_scaled, y_train)
lasso_coefs = pd.Series(np.abs(lasso.coef_[0]), index=X_train.columns).sort_values(ascending=False)
lasso_features = lasso_coefs[lasso_coefs > 0].index.tolist()
print(f"L1 non-zero features: {len(lasso_features)}")

# method 5: random forest importance
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_importance = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
rf_threshold = rf_importance.mean()
rf_features = rf_importance[rf_importance > rf_threshold].index.tolist()
print(f"RF features above mean importance: {len(rf_features)}")

# method 6: RFE
rfe_estimator = LogisticRegression(penalty='l2', C=1.0, max_iter=5000, random_state=42, solver='lbfgs')
rfe = RFE(estimator=rfe_estimator, n_features_to_select=20, step=5)
rfe.fit(X_train_scaled, y_train)
rfe_features = X_train.columns[rfe.support_].tolist()
print(f"RFE selected: {len(rfe_features)} features")


# vote across methods
feature_votes = pd.DataFrame(index=all_features)
feature_votes['variance']     = feature_votes.index.isin(vt_features).astype(int)
feature_votes['correlation']  = feature_votes.index.isin(corr_features).astype(int)
feature_votes['anova']        = feature_votes.index.isin(anova_significant).astype(int)
feature_votes['lasso']        = feature_votes.index.isin(lasso_features).astype(int)
feature_votes['rf_importance']= feature_votes.index.isin(rf_features).astype(int)
feature_votes['rfe']          = feature_votes.index.isin(rfe_features).astype(int)
feature_votes['total_votes']  = feature_votes.sum(axis=1)
feature_votes = feature_votes.sort_values('total_votes', ascending=False)

# keep features selected by at least 4/6 methods
min_votes = 4
final_features = feature_votes[feature_votes['total_votes'] >= min_votes].index.tolist()
print(f"final features (>={min_votes}/6 votes): {len(final_features)}")
print(f"reduction: {len(all_features)} -> {len(final_features)}")


# validate with cross-validation
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

lr_all = LogisticRegression(max_iter=5000, random_state=42)
cv_scores_all = cross_val_score(lr_all, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"LR all features CV AUC: {cv_scores_all.mean():.4f} +/- {cv_scores_all.std():.4f}")

X_train_selected = X_train_scaled[final_features]
lr_sel = LogisticRegression(max_iter=5000, random_state=42)
cv_scores_sel = cross_val_score(lr_sel, X_train_selected, y_train, cv=5, scoring='roc_auc')
print(f"LR selected features CV AUC: {cv_scores_sel.mean():.4f} +/- {cv_scores_sel.std():.4f}")

rf_all = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
cv_rf_all = cross_val_score(rf_all, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"RF all features CV AUC: {cv_rf_all.mean():.4f} +/- {cv_rf_all.std():.4f}")

rf_sel = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
cv_rf_sel = cross_val_score(rf_sel, X_train_selected, y_train, cv=5, scoring='roc_auc')
print(f"RF selected features CV AUC: {cv_rf_sel.mean():.4f} +/- {cv_rf_sel.std():.4f}")


# save outputs
pd.Series(final_features).to_csv('/home/ubuntu/project/selected_features.csv', index=False, header=['feature'])
feature_votes.to_csv('/home/ubuntu/project/feature_votes.csv')
anova_scores.to_csv('/home/ubuntu/project/anova_scores.csv', header=['f_score'])
rf_importance.to_csv('/home/ubuntu/project/rf_importance.csv', header=['importance'])
lasso_coefs.to_csv('/home/ubuntu/project/lasso_coefs.csv', header=['coefficient'])

X_train_final = X_train[final_features]
X_test_final  = X_test[final_features]
X_train_final.to_csv('/home/ubuntu/project/X_train.csv', index=False)
X_test_final.to_csv('/home/ubuntu/project/X_test.csv', index=False)
y_train.to_csv('/home/ubuntu/project/y_train.csv', index=False, header=['discharge'])
y_test.to_csv('/home/ubuntu/project/y_test.csv', index=False, header=['discharge'])

scaler_sel = StandardScaler()
X_train_final_scaled = pd.DataFrame(
    scaler_sel.fit_transform(X_train_final),
    columns=final_features,
    index=X_train_final.index
)
X_test_final_scaled = pd.DataFrame(
    scaler_sel.transform(X_test_final),
    columns=final_features,
    index=X_test_final.index
)
X_train_final_scaled.to_csv('/home/ubuntu/project/X_train_scaled.csv', index=False)
X_test_final_scaled.to_csv('/home/ubuntu/project/X_test_scaled.csv', index=False)

print("done.")
