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
from sklearn.model_selection import StratifiedKFold, cross_val_score,LeaveOneOut,KFold
from sklearn.datasets import fetch_openml,load_digits,load_breast_cancer
import seaborn as sns
import openai as op
import pandas as pd
from sklearn.neural_network import MLPClassifier
data={
    #feature==f
    'f1':np.random.randint(10,100,20),
      'f2':np.random.randint(50,150,20),
      'f3':np.random.randint(1,10,20),
  'f4':np.random.randint(5,15,20),
  'target':np.random.choice([0,1],20)
}
df=pd.DataFrame(data)
x=df.drop('target',axis=1)
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=DecisionTreeClassifier()
kf=KFold(n_splits=5)
score=cross_val_score(model,x,y,cv=kf)
print("each fold accuracy:",score)
print("avg accuracy:",score.mean())