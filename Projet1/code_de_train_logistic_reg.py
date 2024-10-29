from sklearn.linear_model import LogisticRegression
from joblib import dump, load
import pandas as pd

vectorizer = load("vectorizer1")
df = pd.read_csv('train-devinesqui.csv',sep=';', names=['id', 'title', 'abstract', 'classe'],skiprows=1)
X_train=load("traindata-deviensqui")
clf = LogisticRegression(C=9,class_weight='balanced', solver='newton-cg', multi_class='multinomial', n_jobs=-1, random_state=40, verbose=1)
clf.fit(X_train, df['classe'])
dump(clf, "clfLogisticReg")