import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression,LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score,LeaveOneOut
from sklearn.datasets import fetch_openml,load_digits
import seaborn as sns
import openai as op
from sklearn.neural_network import MLPClassifier
data={
    'hour':[1,2,3,4,5,6,7],
    'practicetest':[0,1,1,2,3,3,4],
    'score':[34,39,44,51,58,67,76]
}
df=pd.DataFrame(data)
x=df.drop('score',axis=1)
y=df['score']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
s=pd.DataFrame({
    'hour':[3],
    'practicetest':[1]
})
prediction=model.predict(s)
print("prediction:",prediction)
print("mse",mean_squared_error(y_test,y_pred))
print("r2:",r2_score(y_test,y_pred))