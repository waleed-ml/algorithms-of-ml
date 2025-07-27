import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression,LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score,LeaveOneOut
from sklearn.datasets import fetch_openml,load_digits,load_breast_cancer
import seaborn as sns
import openai as op
import pandas as pd
from sklearn.neural_network import MLPClassifier
data=load_breast_cancer()

df=pd.DataFrame(data.data,columns=data.feature_names)
df["target"]=data.target
x=df.drop('target',axis=1)
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print("prediction:",y_pred)
print("accuracy:",accuracy_score(y_test,y_pred))
print("classification  report:",classification_report(y_test,y_pred))


cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)
plt.show()