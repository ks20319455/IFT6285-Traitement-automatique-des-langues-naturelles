import pandas as pd
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Function to display top n features for each class
def display_top_features_for_classes(coef, feature_names, n=10):
    for i in range(coef.shape[0]):
        temp=np.argsort(coef[i])
        top_indices = temp[-n:]  # Get indices of the top n features
        top_features = feature_names[top_indices]
        top_coefficients = coef[i][top_indices]
        print(f"Class {i}:")
        for feature, coeficient in zip(top_features, top_coefficients):
            print(f"  {feature}: {coeficient:.4f}")
        print()

# Step 1: Load the model and vectorizer
vectorizer = load("vectorizer1")
clf = load("clfLogisticReg")

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



# Get feature names and coefficients
feature_names = vectorizer.get_feature_names_out()
coefficients = clf.coef_

# Display top features for each class
display_top_features_for_classes(coefficients, feature_names, n=10)
