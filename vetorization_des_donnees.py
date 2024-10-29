import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import  TfidfVectorizer

vectorizer = TfidfVectorizer()
df_train = pd.read_csv('train-devinesqui.csv',sep=';', names=['id', 'title', 'abstract', 'classe'],skiprows=1)
df_test = pd.read_csv('test-devinesqui.csv', sep=';', names=['id', 'title', 'abstract', 'classe'],skiprows=1)
X_train=vectorizer.fit_transform(df_train['title'] + " " + df_train['abstract'])
X_test=vectorizer.transform(df_test['title'] + " " + df_test['abstract'])
dump(X_train,"traindata-deviensqui")
dump(X_test,"testdata-deviensqui")
dump(vectorizer, "vectorizer1")