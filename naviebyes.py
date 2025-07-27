import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder,MinMaxScaler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression,LinearRegression,Ridge,Lasso,Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score,LeaveOneOut
from sklearn.datasets import fetch_openml,load_digits,load_iris,load_breast_cancer
import seaborn as sns
import openai as op
from sklearn.neural_network import MLPClassifier
data={
  'Age':        [25, 32, 47, 51, 62, 22, 34, 45, 52, 23, 38, 41, 26, 57, 48, 30, 33, 29, 44, 37],
    'Salary':     [25000, 40000, 60000, 80000, 100000, 22000, 39000, 56000, 76000, 24000, 45000, 50000,
                   27000, 88000, 62000, 34000, 41000, 30000, 53000, 47000],
    'Experience': [1, 5, 10, 12, 20, 0, 6, 9, 14, 1, 7, 8, 2, 18, 11, 4, 5, 3, 10, 7],
    'Score':      [60, 65, 80, 88, 95, 55, 66, 78, 85, 59, 70, 72, 61, 92, 82, 63, 67, 62, 79, 71],
    'Purchased':  [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1]
}
df=pd.DataFrame(data)
x=df.drop('Purchased', axis=1)
y=df['Purchased']

# Step 4: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Step 5: Feature scaling (standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train_scaled, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test_scaled)
print("Predicted:", y_pred[:10])
print("Actual:", y_test[:10])
# Step 8: Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Visualize confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('GaussianNB Confusion Matrix')
plt.show()