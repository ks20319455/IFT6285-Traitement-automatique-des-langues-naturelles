from sklearn.linear_model import LogisticRegression
from joblib import dump, load
from sklearn.neural_network import MLPClassifier
import pandas as pd

vectorizer = load("vectorizer1")
df = pd.read_csv('train-devinesqui.csv',sep=';', names=['id', 'title', 'abstract', 'classe'],skiprows=1)
X_train=load("traindata-deviensqui")
# Initialize the MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(100,),  # You can adjust the number of neurons and layers
                    activation='relu',          # Activation function
                    solver='adam',             # Solver for weight optimization
                    alpha=0.0001,              # Regularization parameter
                    batch_size='auto',         # Batch size for mini-batch learning
                    learning_rate='constant',  # Learning rate schedule
                    learning_rate_init=0.001,  # Initial learning rate
                    max_iter=200,              # Maximum number of iterations
                    random_state=40,           # For reproducibility
                    verbose=1)                 # Verbosity mode
clf.fit(X_train, df['classe'])
dump(clf, "clfNeuralNet")