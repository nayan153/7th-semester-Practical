import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load dataset (TSV format)
data = pd.read_csv(
    "spam.csv",
    sep="\t",
    names=["label", "message"]
)

# Convert labels into numerical values
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Input and target
X = data["message"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text into numerical features
vectorizer = CountVectorizer()

X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

# Create Naive Bayes model
model = MultinomialNB()

# Train model
model.fit(X_train_vector, y_train)

# Prediction
y_pred = model.predict(X_test_vector)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Naive Bayes Accuracy:", round(accuracy * 100, 2), "%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Visualize Confusion Matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)

disp.plot()

plt.title("Naive Bayes Spam Classification")
plt.savefig("naive_bayes_confusion_matrix.png")

print("\nVisualization saved as naive_bayes_confusion_matrix.png")