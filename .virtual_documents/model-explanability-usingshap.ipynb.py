import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import xgboost as xgb
from xgboost import XGBClassifier

import shap


df_train = pd.read_csv('../../data/train.csv')
df_test = pd.read_csv('../../data/test.csv')


age_median = 28.0
df_train['Age'] = df_train['Age'].fillna(age_median)
df_test['Age'] = df_test['Age'].fillna(age_median)

f = lambda x: 1 if x == 'female' else 0
df_train['Sex'] = df_train['Sex'].apply(f)
df_test['Sex'] = df_test['Sex'].apply(f)


df_train.columns


df_train


feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
target_name = ['Survived']

X = df_train[feature_names]
y = df_train[target_name]

X_test = df_test[feature_names]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)


model = XGBClassifier(
    n_estimators=50, 
    objective='binary:logistic', 
    max_depth=5, 
    learning_rate=0.2, 
    verbosity=3, 
    n_jobs=-1, 
    random_state=0, 
)


model.fit(X_train.values, y_train.values)


shap.initjs()


explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_valid)


# Sample of 'Survived = 0'
shap.force_plot(
    explainer.expected_value, 
    shap_values[0, :], 
    X_valid.iloc[0, :]
)


# Sample of 'Survived = 0'
shap.force_plot(
    explainer.expected_value, 
    shap_values[7, :], 
    X_valid.iloc[7, :]
)


shap.force_plot(
    explainer.expected_value, 
    shap_values, 
    X_valid
)


# summarize the effects of all the features
shap.summary_plot(shap_values, X_valid)


shap.summary_plot(shap_values, X_valid, plot_type="bar")



