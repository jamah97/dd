import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
  # Load Our Dataset
df = pd.read_csv("diabetes.csv")
	  # feature selection
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
		# feature scaling
#sc = StandardScaler()
#X = sc.fit_transform(x)
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.15,random_state = 5)


#model Logistic Regression
reg = LogisticRegression()
reg.fit(X_train, y_train)
y_pred=reg.predict(X_test)
print('Logistic Regression accuracy score:', accuracy_score(y_test,y_pred)*100)


import pickle

file = open('DD_model.pkl', 'wb')

pickle.dump(reg, file)
