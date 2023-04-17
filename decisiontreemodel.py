import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
import graphviz
import os


os.environ["PATH"] += os.pathsep + '/usr/local/opt/graphviz/bin'

# Load datasets into Pandas dataframes
heart_df = pd.read_csv("Heart_disease_statlog.csv")

# Split the data into features (X) and target (y)
X = heart_df.drop('target', axis=1)
y = heart_df['target']

# Convert categorical features to one-hot encoding
X = pd.get_dummies(X, columns=['sex', 'cp', 'restecg', 'slope', 'thal'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter values to search
param_grid = {'max_depth': [None, 5, 10, 15],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'criterion': ['gini', 'entropy']}

# Create a GridSearchCV object
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameter values
best_params = grid_search.best_params_

model = DecisionTreeClassifier(**best_params, random_state=42)
model.fit(X_train, y_train)

# make predictions on the test set
y_pred = model.predict(X_test)

heart_df['target'] = heart_df['target'].astype(str)

# Generate DOT data from the decision tree
dot_data = export_graphviz(model, out_file=None, 
                           feature_names=X_train.columns,
                           class_names=heart_df['target'].unique(),  # Update this line
                           filled=True, rounded=True, special_characters=True)

# Render the decision tree visualization using Graphviz
graph = graphviz.Source(dot_data)
graph.render("decision_tree_visualization", view=True)

# print some predictions
print("Actual target values:")
print(y_test.values[:10])

print("\nPredicted target values:")
print(y_pred[:10])

input_data = {'age': [55], 'sex': [1], 'cp': [2], 'trestbps': [135], 'chol': [210], 'fbs': [0],
              'restecg': [1], 'thalach': [160], 'exang': [0], 'oldpeak': [2.5], 'slope': [1],
              'ca': [1], 'thal': [2]}
input_df = pd.DataFrame(data=input_data)

input_df = pd.get_dummies(input_df, columns=['sex', 'cp', 'restecg', 'slope', 'thal'])

input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

predicted_target = model.predict(input_df)
print(predicted_target)