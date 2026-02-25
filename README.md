# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary libraries.
2.Load the dataset.

3.Preprocess the data (handle missing values, encode categorical variables).

4.Split the data into features (X) and target (y).

5.Create polynomial features.

6.Set up pipelines for Ridge, Lasso, and ElasticNet models.

7.Fit the models on the training data.

8.Evaluate model performance using R² score and Mean Squared Error.

9.Compare the results.

## Program:
```
/*
Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
Developed by: A PRAVEEN KISHORE
RegisterNumber:  212225220074

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("encoded_car_data (1).csv")
print(data.head())

data = pd.get_dummies(data, drop_first=True)

X = data.drop('price', axis=1)
y = data['price']

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5)
}

results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    results[name] = {'MSE': mse, 'R2 Score': r2}

results_df = pd.DataFrame(results).T.reset_index()
results_df.rename(columns={'index': 'Model'}, inplace=True)

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='MSE', data=results_df, palette='viridis')
plt.title('Mean Squared Error (MSE)')

plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='R2 Score', data=results_df, palette='viridis')
plt.title('R2 Score')
plt.tight_layout()
plt.show()

*/
```

## Output:
<img width="1307" height="512" alt="1" src="https://github.com/user-attachments/assets/7cb1a7d2-e3b5-4c82-bb4b-958820ef4c49" />



## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
