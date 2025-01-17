from transformers import pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Load the dataset
df = pd.read_csv("all-data.csv", encoding="ISO-8859-1", header=None, names=["Sentiment", "Text"])

# Map dataset labels to match model output
label_mapping = {"positive": "LABEL_2", "neutral": "LABEL_1", "negative": "LABEL_0"}
df["Sentiment"] = df["Sentiment"].map(label_mapping)

# Drop any unmapped labels
df = df.dropna(subset=["Sentiment"])

# Split the dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Generate predictions for the test set
test_df["Predicted"] = test_df["Text"].apply(lambda text: sentiment_pipeline(text)[0]["label"])

# Calculate accuracy
accuracy = accuracy_score(test_df["Sentiment"], test_df["Predicted"])
print(f"Model Accuracy: {accuracy:.4f}")

# Save results to a CSV file
test_df.to_csv("predictions.csv", index=False)

# Display the first few rows of the test set with predictions
print(test_df.head())

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Compute accuracy
accuracy = accuracy_score(test_df["Sentiment"], test_df["Predicted"])
print(f"Model Accuracy: {accuracy:.2f}")

# Display a classification report
print(classification_report(test_df["Sentiment"], test_df["Predicted"]))

# Generate confusion matrix
cm = confusion_matrix(test_df["Sentiment"], test_df["Predicted"])

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_mapping.values(), yticklabels=label_mapping.values())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Compute confidence scores
test_df["Confidence"] = [sentiment_pipeline(text)[0]["score"] for text in test_df["Text"]]

# Identify low-confidence misclassifications
low_confidence_errors = test_df[(test_df["Sentiment"] != test_df["Predicted"]) & (test_df["Confidence"] < 0.6)]
print("Low-confidence misclassifications:")
print(low_confidence_errors)

