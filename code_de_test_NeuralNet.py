import pandas as pd
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Step 1: Load the model and vectorizer
vectorizer = load("vectorizer1")
clf = load("clfNeuralNet")

# Step 2: Prepare test data
test_data = pd.read_csv('test-devinesqui.csv', sep=';', names=['id', 'title', 'abstract', 'classe'],skiprows=1)

# Step 3: Transform the test data
X_test=load("testdata-deviensqui")

# Step 4: Make predictions
predictions = clf.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(test_data['classe'], predictions)
report = classification_report(test_data['classe'], predictions)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)
