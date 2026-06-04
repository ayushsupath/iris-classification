# ============================================================
#   PROJECT 1: Iris Flower Classification
#   Dataset  : sklearn built-in (Fisher's Iris)
#   Algorithm: Random Forest Classifier
#   Author   : ML Assignment
# ============================================================



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

# ── 1. Load Dataset ──────────────────────────────────────────
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("=" * 55)
print("        IRIS FLOWER CLASSIFICATION PROJECT")
print("=" * 55)
print(f"\n📊 Dataset Shape  : {df.shape}")
print(f"🌸 Classes        : {list(iris.target_names)}")
print(f"\n🔍 First 5 rows:")
print(df.head().to_string(index=False))
print(f"\n📈 Basic Statistics:")
print(df.describe().round(2))

# ── 2. Exploratory Data Analysis ─────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Iris Dataset – Exploratory Data Analysis", fontsize=15, fontweight="bold")

features = iris.feature_names
colors   = ["#e74c3c", "#2ecc71", "#3498db"]

for i, ax in enumerate(axes.flatten()):
    for j, species in enumerate(iris.target_names):
        subset = df[df["species"] == species]
        ax.hist(subset[features[i]], alpha=0.7, label=species,
                color=colors[j], bins=15, edgecolor="white")
    ax.set_title(features[i].title(), fontweight="bold")
    ax.set_xlabel("Value (cm)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("iris_eda.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ EDA chart saved → iris_eda.png")

# ── 3. Pairplot (feature relationships) ──────────────────────
pairplot_df = df.copy()
pairplot_df.columns = ["Sepal Len", "Sepal Wid", "Petal Len", "Petal Wid", "Species"]
pp = sns.pairplot(pairplot_df, hue="Species",
                  palette={"setosa": "#e74c3c", "versicolor": "#2ecc71",
                            "virginica": "#3498db"},
                  plot_kws={"alpha": 0.7})
pp.fig.suptitle("Feature Pair Relationships", y=1.02, fontsize=14, fontweight="bold")
plt.savefig("iris_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Pairplot saved  → iris_pairplot.png")

# ── 4. Preprocessing ─────────────────────────────────────────
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\n🔀 Train size : {X_train.shape[0]} samples")
print(f"🔀 Test size  : {X_test.shape[0]} samples")

# ── 5. Model Training ────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
model.fit(X_train, y_train)

# ── 6. Evaluation ────────────────────────────────────────────
y_pred    = model.predict(X_test)
accuracy  = accuracy_score(y_test, y_pred)

print("\n" + "=" * 55)
print("              MODEL EVALUATION")
print("=" * 55)
print(f"\n🎯 Accuracy : {accuracy * 100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            linewidths=1, linecolor="white")
plt.title("Confusion Matrix – Iris Classification", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Predicted Label", fontsize=11)
plt.ylabel("True Label", fontsize=11)
plt.tight_layout()
plt.savefig("iris_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Confusion matrix → iris_confusion_matrix.png")

# Feature Importance
importances = model.feature_importances_
feat_df = pd.DataFrame({
    "Feature"   : iris.feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(8, 4))
sns.barplot(data=feat_df, x="Importance", y="Feature",
            palette="Blues_r", edgecolor="white")
plt.title("Feature Importance – Random Forest", fontsize=13, fontweight="bold")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("iris_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Feature importance → iris_feature_importance.png")

# ── 7. Predict a new sample ──────────────────────────────────
print("\n" + "=" * 55)
print("         PREDICTING A NEW FLOWER SAMPLE")
print("=" * 55)
sample = np.array([[5.1, 3.5, 1.4, 0.2]])   # typical setosa
sample_scaled = scaler.transform(sample)
prediction    = model.predict(sample_scaled)
probability   = model.predict_proba(sample_scaled)[0]

print(f"\n  Input features : Sepal=5.1cm x 3.5cm | Petal=1.4cm x 0.2cm")
print(f"  Predicted class: 🌸 {iris.target_names[prediction[0]].upper()}")
print(f"  Confidence     : {max(probability) * 100:.1f}%")
print(f"  All probs      : {dict(zip(iris.target_names, probability.round(3)))}")

print("\n✅ Project 1 complete!\n")
