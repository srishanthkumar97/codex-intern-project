# Iris Flower Classification Project
# Requirements:
# pip install matplotlib seaborn scikit-learn pandas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
# You can also try:
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# 1. Load the Dataset
# -----------------------------
iris = load_iris()

# Features and target
X = iris.data                      # shape: (150, 4)
y = iris.target                    # shape: (150,)

# Convert to DataFrame for easy analysis
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nClass distribution:")
print(df['species'].value_counts())

# -----------------------------
# 2. Exploratory Data Analysis
# -----------------------------

# Histograms for each feature
plt.figure(figsize=(10, 6))
df.iloc[:, :-1].hist(bins=15, figsize=(10, 6))
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.show()

# Pairplot to see separation between classes
sns.pairplot(df, hue="species", diag_kind="hist")
plt.suptitle("Pairwise Feature Relationships", y=1.02, fontsize=16)
plt.show()

# Correlation heatmap
plt.figure(figsize=(6, 4))
corr = df.iloc[:, :-1].corr()
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X = iris.data   # features
y = iris.target # labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# -----------------------------
# 4. Preprocessing (Scaling)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. Model Training
# -----------------------------
# Using K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=5)  # you can change k
knn.fit(X_train_scaled, y_train)

# If you want to try Logistic Regression or Decision Tree:
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(max_iter=200)
# model.fit(X_train_scaled, y_train)

# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier(random_state=42)
# model.fit(X_train, y_train)  # trees don't need scaling

# -----------------------------
# 6. Evaluation
# -----------------------------
y_pred = knn.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy of KNN model: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Plot confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - KNN")
plt.show()
