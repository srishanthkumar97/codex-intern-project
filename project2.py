# spam_visual.py
# SMS Spam/Ham classifier with visualizations

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. LOAD DATASET
# Make sure spam.csv is in the same folder as this script
print("Loading dataset...")

df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only the first 2 columns (label + message)
# Many Kaggle/other versions have extra unnamed columns
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

print("\nFirst 5 rows:")
print(df.head())

# Remove missing values if any
df = df.dropna()

# 2. SIMPLE EXPLORATION PLOT: CLASS BALANCE
label_counts = df["label"].value_counts()
plt.figure()
label_counts.plot(kind="bar")
plt.title("Number of Ham vs Spam Messages")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 3. TRAIN / TEST SPLIT
X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))

# 4. TEXT TO NUMERIC FEATURES (TF-IDF)
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=3000
)

print("\nFitting TF-IDF vectorizer...")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. TRAIN MODEL (Naive Bayes)
print("\nTraining Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 6. EVALUATION
print("\nEvaluating model...")
y_pred = model.predict(X_test_tfidf)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label="spam")
rec = recall_score(y_test, y_pred, pos_label="spam")
f1 = f1_score(y_test, y_pred, pos_label="spam")

print(f"\nAccuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")

# 7. VISUAL 1: CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])

plt.figure()
disp.plot(values_format="d")
plt.title("Confusion Matrix - Spam Classifier")
plt.tight_layout()
plt.show()

# 8. VISUAL 2: BAR CHART OF METRICS
metrics_names = ["Accuracy", "Precision", "Recall", "F1-score"]
metrics_values = [acc, prec, rec, f1]

plt.figure()
plt.bar(metrics_names, metrics_values)
plt.ylim(0, 1)
for i, v in enumerate(metrics_values):
    plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
plt.title("Model Performance Metrics")
plt.ylabel("Score")
plt.tight_layout()
plt.show()

# 9. FUNCTION TO TEST NEW MESSAGES
def predict_message(text: str) -> str:
    """
    Predict if a given message is spam or ham.
    """
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    return prediction

# 10. EXAMPLE PREDICTIONS
print("\n--- Example predictions ---")
examples = [
    "Congratulations! You have won a $1000 Walmart gift card. Click here to claim now.",
    "Hi bro, are you coming to class tomorrow?",
    "URGENT! Your account has been suspended. Send your password to reactivate.",
    "Dad I reached home safely."
]

for msg in examples:
    print("\nMessage:", msg)
    print("Prediction:", predict_message(msg))