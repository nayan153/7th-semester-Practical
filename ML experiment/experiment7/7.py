import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load the Wine dataset
data = pd.read_csv(
    "9408623-b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv"
)

# Display dataset information
print("Dataset Shape:", data.shape)
print("\nColumns:")
print(data.columns)

# Input features and target
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nDecision Tree Accuracy:", round(accuracy * 100, 2), "%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Visualize Decision Tree
plt.figure(figsize=(16, 8))

plot_tree(
    model,
    feature_names=X.columns.astype(str),
    filled=True
)

plt.title("Decision Tree for Wine Classification")
plt.tight_layout()
plt.savefig("decision_tree_wine.png")

print("\nDecision Tree saved as decision_tree_wine.png")

# Visualize Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Wine Classification Confusion Matrix")
plt.savefig("wine_confusion_matrix.png")

print("Confusion Matrix saved as wine_confusion_matrix.png")