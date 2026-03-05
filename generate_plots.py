# generate plots for the report

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


df            = pd.read_csv('/home/ubuntu/upload/Assignment1_mimicdataset(in).csv')
feature_votes = pd.read_csv('/home/ubuntu/project/feature_votes.csv', index_col=0)
rf_importance = pd.read_csv('/home/ubuntu/project/rf_importance.csv', index_col=0)
anova_scores  = pd.read_csv('/home/ubuntu/project/anova_scores.csv', index_col=0)
lasso_coefs   = pd.read_csv('/home/ubuntu/project/lasso_coefs.csv', index_col=0)


# target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

counts = df['icu_death_flag'].value_counts()
labels = [
    'Discharged Alive\n(n={:,})'.format(counts[0]),
    'Died in ICU\n(n={:,})'.format(counts[1])
]
axes[0].pie([counts[0], counts[1]], labels=labels,
            colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11})
axes[0].set_title('ICU Outcome Distribution', fontsize=13, fontweight='bold')

df_alive = df[df['icu_death_flag'] == 0]['age']
df_dead  = df[df['icu_death_flag'] == 1]['age']
axes[1].hist(df_alive, bins=30, alpha=0.6, label='Discharged',  color='#2ecc71', density=True)
axes[1].hist(df_dead,  bins=30, alpha=0.6, label='Died in ICU', color='#e74c3c', density=True)
axes[1].set_xlabel('Age', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title('Age Distribution by ICU Outcome', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig('/home/ubuntu/project/target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved: target_distribution.png")


# feature selection voting heatmap
votes_sorted = feature_votes.sort_values('total_votes', ascending=False)
vote_cols   = ['variance', 'correlation', 'anova', 'lasso', 'rf_importance', 'rfe']
vote_labels = ['Variance\nThreshold', 'Correlation\nFilter', 'ANOVA\nF-test',
               'L1 Lasso', 'RF\nImportance', 'RFE']

fig, ax = plt.subplots(figsize=(10, 14))
sns.heatmap(votes_sorted[vote_cols].values,
            yticklabels=votes_sorted.index,
            xticklabels=vote_labels,
            cmap='YlOrRd',
            cbar_kws={'label': 'Selected (1) / Not Selected (0)'},
            linewidths=0.5, linecolor='white', ax=ax)
ax.set_title('Feature Selection Voting Matrix', fontsize=14, fontweight='bold')
ax.set_ylabel('Features', fontsize=12)

for i, feat in enumerate(votes_sorted.index):
    total = votes_sorted.loc[feat, 'total_votes']
    ax.text(len(vote_cols) + 0.3, i + 0.5, f'{int(total)}/6', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('/home/ubuntu/project/feature_voting_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved: feature_voting_heatmap.png")


# feature importance comparison (RF vs ANOVA)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

top_rf = rf_importance.head(20)
axes[0].barh(range(len(top_rf)), top_rf['importance'].values, color='#3498db')
axes[0].set_yticks(range(len(top_rf)))
axes[0].set_yticklabels(top_rf.index, fontsize=10)
axes[0].invert_yaxis()
axes[0].set_xlabel('Importance', fontsize=12)
axes[0].set_title('Top 20 - Random Forest Importance', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')

top_anova = anova_scores.head(20)
axes[1].barh(range(len(top_anova)), top_anova['f_score'].values, color='#e67e22')
axes[1].set_yticks(range(len(top_anova)))
axes[1].set_yticklabels(top_anova.index, fontsize=10)
axes[1].invert_yaxis()
axes[1].set_xlabel('F-Score', fontsize=12)
axes[1].set_title('Top 20 - ANOVA F-Score', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/home/ubuntu/project/feature_importance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved: feature_importance_comparison.png")


# correlation heatmap of selected features
selected_features = pd.read_csv('/home/ubuntu/project/selected_features.csv')['feature'].tolist()
X_train = pd.read_csv('/home/ubuntu/project/X_train.csv')
corr = X_train[selected_features].corr()

fig, ax = plt.subplots(figsize=(16, 14))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0,
            square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
            ax=ax, vmin=-1, vmax=1,
            xticklabels=True, yticklabels=True)
ax.set_title('Correlation Heatmap of Selected Features', fontsize=14, fontweight='bold')
plt.xticks(fontsize=8, rotation=45, ha='right')
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig('/home/ubuntu/project/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved: correlation_heatmap.png")


# missing data pattern
missing_pct = df.drop(
    columns=['subject_id', 'hadm_id', 'stay_id', 'intime', 'outtime', 'deathtime']
).isnull().mean() * 100
missing_pct = missing_pct.sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]

fig, ax = plt.subplots(figsize=(14, 8))
bar_colors = ['#e74c3c' if x > 50 else '#f39c12' if x > 10 else '#2ecc71'
              for x in missing_pct.values]
ax.barh(range(len(missing_pct)), missing_pct.values, color=bar_colors)
ax.set_yticks(range(len(missing_pct)))
ax.set_yticklabels(missing_pct.index, fontsize=6)
ax.invert_yaxis()
ax.set_xlabel('Missing (%)', fontsize=12)
ax.set_title('Missing Data Pattern - MIMIC-IV', fontsize=14, fontweight='bold')
ax.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='50% cutoff')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('/home/ubuntu/project/missing_data_pattern.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved: missing_data_pattern.png")


# dimensionality reduction pipeline summary
fig, ax = plt.subplots(figsize=(10, 5))
stages = [
    'Original\nFeatures',
    'After Removing\nID/Time/Leakage',
    'After Removing\n>50% Missing',
    'After Feature\nSelection'
]
counts = [140, 132, 43, 36]
bars = ax.bar(stages, counts,
              color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
              width=0.6, edgecolor='white', linewidth=2)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(count), ha='center', va='bottom', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Features', fontsize=12)
ax.set_title('Feature Dimensionality Reduction Pipeline', fontsize=14, fontweight='bold')
ax.set_ylim(0, 160)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('/home/ubuntu/project/dimensionality_reduction.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved: dimensionality_reduction.png")

print("\nall plots done.")
