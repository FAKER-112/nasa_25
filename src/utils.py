import os 
import sys

import pandas as pd
import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

class RareCategoryImputer(BaseEstimator, TransformerMixin):
    def __init__(self, min_count=5):
        self.min_count = min_count
        self.most_frequent_ = {}
        self.rare_values_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            counts = X[col].value_counts()
            self.rare_values_[col] = counts[counts < self.min_count].index.tolist()
            self.most_frequent_[col] = X[col].mode()[0]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].replace(self.rare_values_[col], self.most_frequent_[col])
            X[col] = X[col].astype(str)  # <-- ensure all values are strings
        return X



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise e
    

def evaluate_classifiers(X_train, y_train, X_test, y_test, models, params, scoring='accuracy'):
    """
    Evaluate multiple classification models using GridSearchCV.

    Args:
        X_train, y_train: training features and labels
        X_test, y_test: testing features and labels
        models: dict of model name -> sklearn estimator
        params: dict of model name -> hyperparameter grid
        scoring: metric for GridSearchCV ('accuracy' or 'f1_macro', etc.)

    Returns:
        report: dict of model name -> test score
        best_models: dict of model name -> trained model with best params
    """
    report = {}
    best_models = {}
    
    for name, model in models.items():
        print(f"Training and tuning {name}...")
        param_grid = params.get(name, {})
        
        gs = GridSearchCV(model, param_grid, cv=3, scoring=scoring, n_jobs=-1)
        gs.fit(X_train, y_train)
        
        # Set model to best parameters and retrain
        model.set_params(**gs.best_params_)
        model.fit(X_train, y_train)
        best_models[name] = model
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Evaluate
        train_score = accuracy_score(y_train, y_train_pred)
        test_score = accuracy_score(y_test, y_test_pred)
        
        print(f"{name} - Train {scoring}: {train_score:.4f}, Test {scoring}: {test_score:.4f}")
        
        report[name] = test_score
    
    return report

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise e