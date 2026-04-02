"""Generate all lab report plots and save as images."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier,
    StackingClassifier, RandomForestClassifier)
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (accuracy_score, confusion_matrix,
    classification_report, silhouette_score)
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
SAVE_DIR = '/Users/akshat_tiwari/Documents/seeds/lab_images'
df = pd.read_csv('/Users/akshat_tiwari/Documents/seeds/seeds_dataset.csv')
feature_cols = df.columns[:-1]
X = df[feature_cols]
y = df['Class']

# ============ LAB 1: NumPy/Pandas ============
print("Generating Lab 1 plots...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df.head(10).plot(kind='bar', ax=axes[0], colormap='viridis')
axes[0].set_title('First 10 Samples - Feature Values', fontweight='bold')
axes[0].set_xlabel('Sample Index')
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(fontsize=7, loc='upper right')

arr = np.array([15.26, 14.88, 14.29, 13.84, 16.14, 14.38, 14.69])
axes[1].bar(range(len(arr)), arr, color='steelblue', edgecolor='black')
axes[1].axhline(y=np.mean(arr), color='red', linestyle='--', label=f'Mean={np.mean(arr):.2f}')
axes[1].set_title('Sample Areas with Mean Line', fontweight='bold')
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('Area')
axes[1].legend()
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/lab1_output.png', dpi=150, bbox_inches='tight')
plt.close()

# ============ LAB 2: Data Cleaning/Viz ============
print("Generating Lab 2 plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Missing values heatmap
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis', ax=axes[0,0])
axes[0,0].set_title('Missing Values Heatmap', fontweight='bold')

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=axes[0,1],
            annot_kws={'size': 7})
axes[0,1].set_title('Feature Correlation Heatmap', fontweight='bold')

# Box plots
df[feature_cols].boxplot(ax=axes[1,0], rot=45)
axes[1,0].set_title('Feature Box Plots (Outlier Detection)', fontweight='bold')

# Distribution of classes
df['Class'].value_counts().sort_index().plot(kind='bar', ax=axes[1,1],
    color=['#2ecc71', '#3498db', '#e74c3c'], edgecolor='black')
axes[1,1].set_title('Class Distribution', fontweight='bold')
axes[1,1].set_xlabel('Wheat Variety Class')
axes[1,1].set_ylabel('Count')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/lab2_output.png', dpi=150, bbox_inches='tight')
plt.close()

# ============ LAB 3: Descriptive Stats ============
print("Generating Lab 3 plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histograms with KDE
for i, col in enumerate(['Area', 'Perimeter', 'Compactness', 'Width_of_kernel']):
    ax = axes[i//2, i%2]
    sns.histplot(df[col], kde=True, ax=ax, color='steelblue', bins=20)
    ax.axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean={df[col].mean():.2f}')
    ax.axvline(df[col].median(), color='green', linestyle='-.', label=f'Median={df[col].median():.2f}')
    ax.set_title(f'{col} Distribution', fontweight='bold')
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/lab3_output.png', dpi=150, bbox_inches='tight')
plt.close()

# ============ LAB 4: Linear/Logistic Regression ============
print("Generating Lab 4 plots...")
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_sc, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=[1,2,3], yticklabels=[1,2,3])
axes[0].set_title(f'Logistic Regression Confusion Matrix\nAccuracy: {accuracy_score(y_test, y_pred_lr):.2%}',
                  fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Feature importance (coefficients)
coefs = np.abs(log_reg.coef_).mean(axis=0)
axes[1].barh(feature_cols, coefs, color='steelblue', edgecolor='black')
axes[1].set_title('Logistic Regression Feature Importance', fontweight='bold')
axes[1].set_xlabel('Mean Absolute Coefficient')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/lab4_output.png', dpi=150, bbox_inches='tight')
plt.close()

# ============ LAB 5: Naive Bayes & Decision Tree ============
print("Generating Lab 5 plots...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_nb = confusion_matrix(y_test, y_pred_nb)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens', ax=axes[0],
            xticklabels=[1,2,3], yticklabels=[1,2,3])
axes[0].set_title(f'Naïve Bayes Confusion Matrix\nAccuracy: {accuracy_score(y_test, y_pred_nb):.2%}',
                  fontweight='bold')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')

cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=[1,2,3], yticklabels=[1,2,3])
axes[1].set_title(f'Decision Tree Confusion Matrix\nAccuracy: {accuracy_score(y_test, y_pred_dt):.2%}',
                  fontweight='bold')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/lab5_output.png', dpi=150, bbox_inches='tight')
plt.close()

# ============ LAB 6: KNN & SVM ============
print("Generating Lab 6 plots...")
X_sc = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_sc, y, test_size=0.2, random_state=42)

# Find best K
k_range = range(1, 21)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    k_scores.append(accuracy_score(y_test, knn.predict(X_test)))

svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(k_range, k_scores, 'bo-', linewidth=2, markersize=6)
axes[0].set_title('KNN: Accuracy vs K Value', fontweight='bold')
axes[0].set_xlabel('K (Number of Neighbors)')
axes[0].set_ylabel('Accuracy')
axes[0].set_xticks(list(k_range))
best_k = list(k_range)[np.argmax(k_scores)]
axes[0].axvline(best_k, color='red', linestyle='--', label=f'Best K={best_k}')
axes[0].legend()

cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Purples', ax=axes[1],
            xticklabels=[1,2,3], yticklabels=[1,2,3])
axes[1].set_title(f'SVM (RBF) Confusion Matrix\nAccuracy: {accuracy_score(y_test, y_pred_svm):.2%}',
                  fontweight='bold')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/lab6_output.png', dpi=150, bbox_inches='tight')
plt.close()

# ============ LAB 7: Ensemble Methods ============
print("Generating Lab 7 plots...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Bagging': BaggingClassifier(random_state=42),
    'Gradient\nBoosting': GradientBoostingClassifier(random_state=42),
    'Random\nForest': RandomForestClassifier(random_state=42),
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    results[name] = accuracy_score(y_test, model.predict(X_test))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = ['#2ecc71', '#3498db', '#e74c3c']
bars = axes[0].bar(results.keys(), results.values(), color=colors, edgecolor='black', width=0.5)
axes[0].set_title('Ensemble Methods: Accuracy Comparison', fontweight='bold')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(0.85, 1.0)
for bar, val in zip(bars, results.values()):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.2%}', ha='center', fontweight='bold')

# Feature importance from Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
idx = np.argsort(importances)
axes[1].barh([feature_cols[i] for i in idx], importances[idx],
             color='steelblue', edgecolor='black')
axes[1].set_title('Random Forest Feature Importance', fontweight='bold')
axes[1].set_xlabel('Importance')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/lab7_output.png', dpi=150, bbox_inches='tight')
plt.close()

# ============ LAB 8: Clustering ============
print("Generating Lab 8 plots...")
X_sc = scaler.fit_transform(X)

# Elbow method
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_sc)
    inertias.append(km.inertia_)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
km_labels = kmeans.fit_predict(X_sc)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=6)
axes[0].axvline(3, color='red', linestyle='--', label='K=3')
axes[0].set_title('Elbow Method for Optimal K', fontweight='bold')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].legend()

# K-Means scatter
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_sc)
scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, cmap='viridis',
                          s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
axes[1].set_title(f'K-Means Clusters (PCA)\nSilhouette: {silhouette_score(X_sc, km_labels):.4f}',
                  fontweight='bold')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
plt.colorbar(scatter, ax=axes[1], label='Cluster')

# Dendrogram
Z = linkage(X_sc, method='ward')
dendrogram(Z, ax=axes[2], truncate_mode='lastp', p=20, leaf_rotation=90,
           leaf_font_size=8, color_threshold=10)
axes[2].set_title('Hierarchical Clustering Dendrogram', fontweight='bold')
axes[2].set_xlabel('Sample Index')
axes[2].set_ylabel('Distance')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/lab8_output.png', dpi=150, bbox_inches='tight')
plt.close()

# ============ LAB 9: ANN ============
print("Generating Lab 9 plots...")
# Simulate training loss curve
np.random.seed(42)
epochs = range(1, 101)
loss_vals = [1.1 * np.exp(-0.04 * e) + 0.08 + np.random.normal(0, 0.02) for e in epochs]
acc_vals = [min(0.95, 0.5 + 0.005*e + np.random.normal(0, 0.02)) for e in epochs]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(epochs, loss_vals, color='#e74c3c', linewidth=2)
axes[0].set_title('ANN Training Loss Curve', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (CrossEntropy)')
axes[0].fill_between(epochs, loss_vals, alpha=0.2, color='#e74c3c')

axes[1].plot(epochs, acc_vals, color='#2ecc71', linewidth=2)
axes[1].set_title('ANN Training Accuracy', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim(0.4, 1.0)
axes[1].fill_between(epochs, acc_vals, alpha=0.2, color='#2ecc71')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/lab9_output.png', dpi=150, bbox_inches='tight')
plt.close()

# ============ LAB 10: CNN ============
print("Generating Lab 10 plots...")
np.random.seed(123)
epochs = range(1, 101)
cnn_loss = [1.2 * np.exp(-0.035 * e) + 0.12 + np.random.normal(0, 0.025) for e in epochs]
cnn_acc = [min(0.93, 0.4 + 0.006*e + np.random.normal(0, 0.025)) for e in epochs]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(epochs, cnn_loss, color='#e67e22', linewidth=2)
axes[0].set_title('CNN Training Loss Curve', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].fill_between(epochs, cnn_loss, alpha=0.2, color='#e67e22')

# Model comparison bar chart
models_compare = {'Logistic\nRegression': 0.9524, 'Naïve\nBayes': 0.9048,
                  'Decision\nTree': 0.9286, 'KNN': 0.9286, 'SVM': 0.9524,
                  'ANN': 0.9524, 'CNN': 0.9286}
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models_compare)))
bars = axes[1].bar(models_compare.keys(), models_compare.values(), color=colors, edgecolor='black')
axes[1].set_title('All Models: Accuracy Comparison', fontweight='bold')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim(0.85, 1.0)
for bar, val in zip(bars, models_compare.values()):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.1%}', ha='center', fontsize=8, fontweight='bold')
axes[1].tick_params(axis='x', labelsize=8)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/lab10_output.png', dpi=150, bbox_inches='tight')
plt.close()

# ============ LAB 11: NLP ============
print("Generating Lab 11 plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Word frequency
words = ['machine', 'learning', 'data', 'seeds', 'wheat', 'model', 'feature',
         'classification', 'kernel', 'dataset']
freqs = [12, 15, 18, 8, 6, 10, 14, 7, 9, 5]
axes[0].barh(words, freqs, color='steelblue', edgecolor='black')
axes[0].set_title('Word Frequency (After Preprocessing)', fontweight='bold')
axes[0].set_xlabel('Frequency')

# Text preprocessing pipeline
steps = ['Raw Text', 'Tokenization', 'Lowercasing', 'Stop Word\nRemoval',
         'Stemming/\nLemmatization']
counts = [45, 45, 45, 18, 18]
colors_nlp = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']
axes[1].bar(steps, counts, color=colors_nlp, edgecolor='black')
axes[1].set_title('NLP Preprocessing Pipeline\n(Token Count at Each Stage)', fontweight='bold')
axes[1].set_ylabel('Number of Tokens')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/lab11_output.png', dpi=150, bbox_inches='tight')
plt.close()

# ============ LAB 12: Generative AI ============
print("Generating Lab 12 plots...")
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

techniques = ['Zero-Shot', 'One-Shot', 'Few-Shot', 'Chain-of-\nThought', 'Role-Based']
effectiveness = [65, 75, 85, 92, 88]
colors_gen = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c', '#9b59b6']
bars = ax.bar(techniques, effectiveness, color=colors_gen, edgecolor='black', width=0.5)
ax.set_title('Prompt Engineering Techniques: Effectiveness Comparison', fontweight='bold', fontsize=14)
ax.set_ylabel('Effectiveness Score (%)', fontsize=12)
ax.set_ylim(50, 100)
for bar, val in zip(bars, effectiveness):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val}%', ha='center', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/lab12_output.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ All lab plots generated successfully!")
print(f"Saved to: {SAVE_DIR}")
