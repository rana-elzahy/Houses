
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error



dataset = pd.read_csv('houses.csv')
dataset.head(20)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(dataset)
dataset = imp.transform(dataset)


X = dataset[:, :-1]
y = dataset[:, -1]


sc = StandardScaler()
X = sc.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

 
sgd = SGDRegressor( penalty = 'l2' , max_iter=1000, tol=1e-3 , loss = 'squared_loss')

sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test) 
y_pred

print("train score",sgd.score(X_train,y_train)
print("test score",sgd.score(X_test,y_test)

mean_absolute_error(y_test, y_pred)
mean_squared_error(y_test, y_pred)


