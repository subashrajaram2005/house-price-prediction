import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# Load dataset
datasets = pd.read_csv('boston_house_prices.csv')

# Rename target column
datasets.rename(columns={'MEDV': 'PRICE'}, inplace=True)

# Basic EDA
print(datasets.head())
print(datasets.shape)
print(datasets.isnull().sum())
print(datasets.describe())

# Correlation heatmap
plt.figure(figsize=(10,10))
sns.heatmap(datasets.corr(), annot=True, fmt='.1f', cmap='Blues')
plt.show()

# Split features and target
X = datasets.drop(columns=['PRICE'])
Y = datasets['PRICE']

# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Initialize and train model
model = XGBRegressor()
model.fit(X_train, Y_train)

# Predictions on train data
trainpredict = model.predict(X_train)
print('Train R2 Score:', metrics.r2_score(Y_train, trainpredict))
print('Train MAE:', metrics.mean_absolute_error(Y_train, trainpredict))

# Plot train predictions
plt.scatter(Y_train, trainpredict)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Train Data: Actual vs Predicted Prices')
plt.show()

# Predictions on test data
testpredict = model.predict(X_test)
print('Test R2 Score:', metrics.r2_score(Y_test, testpredict))
print('Test MAE:', metrics.mean_absolute_error(Y_test, testpredict))

# Plot test predictions
plt.scatter(Y_test, testpredict)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Test Data: Actual vs Predicted Prices')
plt.show()

# Test sample prediction
test_sample_np = np.array([[0.38735,0,25.65,0,0.581,5.613,95.6,1.7572,2,188,19.1,359.29,27.26]])
predicted_price = model.predict(test_sample_np)
print(f"Predicted house price: {predicted_price[0]:.2f} (in $1000s)")
