# 🌸 Iris Flower Classification

A machine learning project to classify Iris flowers into three species — **Setosa**, **Versicolor**, and **Virginica** — using the classic Fisher's Iris dataset.

---

## 📌 Objective

To build a classification model that can predict the species of an Iris flower based on its physical measurements (sepal and petal dimensions).

---

## 📂 Dataset

| Property | Details |
|---|---|
| **Name** | Fisher's Iris Dataset |
| **Source** | `sklearn.datasets.load_iris()` (built-in) |
| **Total Samples** | 150 |
| **Classes** | Setosa, Versicolor, Virginica |
| **Features** | 4 (Sepal Length, Sepal Width, Petal Length, Petal Width) |
| **Class Balance** | 50 samples per class (balanced) |

---

## 🧠 Algorithm Used

**Random Forest Classifier**

Random Forest is an ensemble learning method that builds multiple decision trees and combines their outputs (majority voting) to improve accuracy and reduce overfitting.

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 100 trees |
| `max_depth` | 5 |
| `random_state` | 42 |

---

## ⚙️ Tech Stack

| Library | Purpose |
|---|---|
| `scikit-learn` | Model training, preprocessing, evaluation |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting charts |
| `seaborn` | Pairplot and heatmap visualization |

---

## 🔄 Project Workflow

```
Load Dataset → EDA → Feature Scaling → Train/Test Split → Model Training → Evaluation → Prediction
```

1. **Load Dataset** — Load iris data from sklearn, convert to DataFrame
2. **EDA** — Histogram plots for each feature by species, pairplot for feature relationships
3. **Preprocessing** — StandardScaler for feature normalization
4. **Train/Test Split** — 80% training, 20% testing (stratified)
5. **Model Training** — Random Forest with 100 estimators
6. **Evaluation** — Accuracy, Precision, Recall, F1-Score, Confusion Matrix
7. **Prediction** — Predict species for a new flower sample

---

## 📊 Results

| Metric | Value |
|---|---|
| **Accuracy** | 93.33% |
| **Precision (avg)** | 0.93 |
| **Recall (avg)** | 0.93 |
| **F1-Score (avg)** | 0.93 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Setosa | 1.00 | 1.00 | 1.00 |
| Versicolor | 0.90 | 0.90 | 0.90 |
| Virginica | 0.90 | 0.90 | 0.90 |

---

## 📁 Output Files Generated

| File | Description |
|---|---|
| `iris_eda.png` | Histogram of each feature grouped by species |
| `iris_pairplot.png` | Pairwise feature relationship plot |
| `iris_confusion_matrix.png` | Confusion matrix heatmap |
| `iris_feature_importance.png` | Feature importance bar chart |

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

### 2. Run the Script
```bash
python iris_classification.py
```

No dataset download required — data loads automatically from sklearn.

---

## 🔍 Key Findings

- **Petal Length** and **Petal Width** are the most important features for classification
- **Setosa** is perfectly separable from the other two species
- **Versicolor** and **Virginica** have some overlap, making them harder to distinguish
- Random Forest achieves **93.33% accuracy** on the test set

---

## 📚 References

- Fisher, R.A. (1936). *The use of multiple measurements in taxonomic problems*
- [Scikit-learn Iris Dataset Documentation](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
- [Random Forest Classifier — sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
