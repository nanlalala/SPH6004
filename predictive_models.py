# SPH6004 Assignment - Predictive Models
# Models: LR, LDA, QDA, KNN, Naive Bayes, Decision Tree, SVM

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# load pre-processed data
X_train = pd.read_csv('/home/ubuntu/project/X_train.csv')
X_test  = pd.read_csv('/home/ubuntu/project/X_test.csv')
y_train = pd.read_csv('/home/ubuntu/project/y_train.csv')['discharge']
y_test  = pd.read_csv('/home/ubuntu/project/y_test.csv')['discharge']

print(f"train: {X_train.shape}, test: {X_test.shape}")
print(y_train.value_counts())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
feature_names  = X_train.columns.tolist()

# LinearSVC doesn't output probabilities natively, wrap with calibration
linear_svc = LinearSVC(C=1.0, max_iter=5000, random_state=42, dual=True)
svm_model  = CalibratedClassifierCV(linear_svc, cv=3)

models = {
    'Logistic Regression': LogisticRegression(
        C=1.0, penalty='l2', solver='lbfgs', max_iter=5000, random_state=42
    ),
    'LDA':          LinearDiscriminantAnalysis(),
    'QDA':          QuadraticDiscriminantAnalysis(reg_param=0.1),
    'KNN':          KNeighborsClassifier(n_neighbors=7, metric='minkowski', n_jobs=-1),
    'Naive Bayes':  GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=10, min_samples_split=20, min_samples_leaf=10, random_state=42
    ),
    'SVM (Linear)': svm_model,
}

# models that require scaled input
scaled_models = {'Logistic Regression', 'LDA', 'QDA', 'KNN', 'SVM (Linear)'}

results    = {}
cv_results = {}
roc_data   = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\ntraining {name}...")
    t0 = time.time()

    X_tr = X_train_scaled if name in scaled_models else X_train.values
    X_te = X_test_scaled  if name in scaled_models else X_test.values

    cv_auc = cross_val_score(model, X_tr, y_train, cv=cv, scoring='roc_auc',  n_jobs=-1)
    cv_acc = cross_val_score(model, X_tr, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_f1  = cross_val_score(model, X_tr, y_train, cv=cv, scoring='f1',       n_jobs=-1)

    cv_results[name] = {
        'cv_auc_mean': cv_auc.mean(), 'cv_auc_std': cv_auc.std(),
        'cv_acc_mean': cv_acc.mean(), 'cv_acc_std': cv_acc.std(),
        'cv_f1_mean':  cv_f1.mean(),  'cv_f1_std':  cv_f1.std(),
    }
    print(f"  5-fold CV AUC: {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    elapsed = time.time() - t0

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    ap   = average_precision_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    results[name] = {
        'Accuracy': acc, 'Sensitivity': sensitivity, 'Specificity': specificity,
        'Precision': prec, 'Recall': rec, 'F1-Score': f1,
        'AUC-ROC': auc, 'AP': ap,
        'Training Time (s)': elapsed,
        'Confusion Matrix': cm,
    }

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr, auc)

    print(f"  acc={acc:.4f}  sens={sensitivity:.4f}  spec={specificity:.4f}  "
          f"f1={f1:.4f}  auc={auc:.4f}  ({elapsed:.1f}s)")
    print(f"  TN={tn} FP={fp} FN={fn} TP={tp}")


# summary table
comparison_data = []
for name in results:
    row = {k: v for k, v in results[name].items() if k != 'Confusion Matrix'}
    row['Model']            = name
    row['CV AUC (mean)']    = cv_results[name]['cv_auc_mean']
    row['CV AUC (std)']     = cv_results[name]['cv_auc_std']
    row['CV Accuracy (mean)'] = cv_results[name]['cv_acc_mean']
    row['CV F1 (mean)']     = cv_results[name]['cv_f1_mean']
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data).set_index('Model')
comparison_df = comparison_df.sort_values('AUC-ROC', ascending=False)
print("\n", comparison_df.to_string())
comparison_df.to_csv('/home/ubuntu/project/model_comparison.csv')


# plots
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# ROC curves
plt.figure(figsize=(10, 8))
for i, (name, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
    plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
             label=f'{name} (AUC={auc_val:.4f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/ubuntu/project/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# bar chart comparison
model_names = comparison_df.index.tolist()
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, metric in enumerate(['Accuracy', 'F1-Score', 'AUC-ROC']):
    vals = comparison_df[metric].values
    bars = axes[idx].barh(model_names, vals, color=colors[:len(model_names)])
    axes[idx].set_xlabel(metric, fontsize=12)
    axes[idx].set_title(metric, fontsize=13)
    axes[idx].set_xlim([min(vals) - 0.05, 1.0])
    for bar, v in zip(bars, vals):
        axes[idx].text(v + 0.003, bar.get_y() + bar.get_height()/2,
                       f'{v:.4f}', va='center', fontsize=9)
    axes[idx].grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('/home/ubuntu/project/model_comparison_bars.png', dpi=150, bbox_inches='tight')
plt.close()

# confusion matrices
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes_flat = axes.flatten()
for i, (name, metrics) in enumerate(results.items()):
    cm = metrics['Confusion Matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_flat[i],
                xticklabels=['Died', 'Discharged'],
                yticklabels=['Died', 'Discharged'])
    axes_flat[i].set_title(name, fontsize=11)
    axes_flat[i].set_ylabel('Actual')
    axes_flat[i].set_xlabel('Predicted')
if len(results) < 8:
    axes_flat[-1].axis('off')
plt.suptitle('Confusion Matrices', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('/home/ubuntu/project/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

# CV AUC with error bars
fig, ax = plt.subplots(figsize=(10, 6))
cv_means = [cv_results[n]['cv_auc_mean'] for n in model_names]
cv_stds  = [cv_results[n]['cv_auc_std']  for n in model_names]
bars = ax.barh(model_names, cv_means, xerr=cv_stds,
               color=colors[:len(model_names)], capsize=5, alpha=0.8)
ax.set_xlabel('AUC-ROC', fontsize=12)
ax.set_title('5-Fold CV AUC-ROC', fontsize=14)
ax.set_xlim([min(cv_means) - 0.05, 1.0])
for bar, v, s in zip(bars, cv_means, cv_stds):
    ax.text(v + s + 0.005, bar.get_y() + bar.get_height()/2,
            f'{v:.4f}', va='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('/home/ubuntu/project/cv_auc_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nbest AUC: {comparison_df['AUC-ROC'].idxmax()} ({comparison_df['AUC-ROC'].max():.4f})")
print(f"best F1:  {comparison_df['F1-Score'].idxmax()} ({comparison_df['F1-Score'].max():.4f})")
