# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/AnomaData.csv")

# Exploratory Data Analysis (EDA)
# Show data summary
print(data.info())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Check for outliers using box plots or other visualization techniques
plt.figure(figsize=(10, 6))
sns.boxplot(data=data.drop(columns='y'))
plt.title("Boxplot of Features")
plt.xticks(rotation=90)
plt.show()

# Data Cleaning
# Handle missing values
# For example, fill missing values with mean, median, or mode
data.fillna(data.mean(), inplace=True)

# Feature Engineering and Selection
# Create new features if necessary
# Perform feature selection techniques if needed

# Train/Test Split
X = data.drop(columns='y')
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection
model = IsolationForest()

data['time'] = pd.to_datetime(data['time'])

# Model Training
model.fit(X_train)

# Model Evaluation
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Convert predictions to binary labels (1: anomaly, -1: normal)
y_pred_train = np.where(y_pred_train == -1, 1, 0)
y_pred_test = np.where(y_pred_test == -1, 1, 0)

# Evaluate model
print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
print("Classification Report (Test Set):")
print(classification_report(y_test, y_pred_test))
print("Confusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_pred_test))

# Hyperparameter Tuning
# Example: GridSearchCV for Isolation Forest
params = {'n_estimators': [100, 200, 300],
          'max_samples': [100, 200, 300],
          'contamination': [0.05, 0.1, 0.2]}

grid_search = GridSearchCV(estimator=IsolationForest(), param_grid=params, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Model Deployment Plan
# Develop a plan for deploying the model in a production environment

# Documentation
# Write a detailed report explaining design choices, performance evaluation, and future work
# Include instructions for installation and execution of the pipeline
<!---
harvinder42/harvinder42 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
