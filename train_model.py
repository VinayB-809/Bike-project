import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

print("START")

data = pd.read_csv("data.csv")

X = data[['Speed','Weather','Road','Helmet','Alcohol']]
y = data['Accident']

model = LogisticRegression()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully")