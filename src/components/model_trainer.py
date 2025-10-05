import os
import sys
import logging
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

from src.utils import save_object,evaluate_classifiers

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            # models = {
            #     "RandomForest": RandomForestClassifier(random_state=42),
            #     "GradientBoosting": GradientBoostingClassifier(random_state=42),
            #     "AdaBoost": AdaBoostClassifier(random_state=42),
            #     "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            #     "RidgeClassifier": RidgeClassifier(random_state=42),
            #     "SVC": SVC(probability=True, random_state=42),
            #     "LinearSVC": LinearSVC(max_iter=5000, random_state=42),
            #     "DecisionTree": DecisionTreeClassifier(random_state=42),
            #     "KNeighbors": KNeighborsClassifier(),
            #     "GaussianNB": GaussianNB()
            # }
            # params = {
            #     "RandomForest": {
            #         "n_estimators": [50, 100, 200],
            #         "max_depth": [None, 10, 20],
            #         "min_samples_split": [2, 5],
            #         "min_samples_leaf": [1, 2]
            #     },
            #     "GradientBoosting": {
            #         "n_estimators": [50, 100],
            #         "learning_rate": [0.01, 0.1],
            #         "max_depth": [3, 5]
            #     },
            #     "AdaBoost": {
            #         "n_estimators": [50, 100],
            #         "learning_rate": [0.5, 1.0]
            #     },
            #     "LogisticRegression": {
            #         "C": [0.1, 1, 10],
            #         "solver": ["liblinear", "lbfgs"]
            #     },
            #     "RidgeClassifier": {
            #         "alpha": [0.1, 1.0, 10.0]
            #     },
            #     "SVC": {
            #         "C": [0.1, 1, 10],
            #         "kernel": ["linear", "rbf"],
            #         "gamma": ["scale", "auto"]
            #     },
            #     "LinearSVC": {
            #         "C": [0.1, 1, 10]
            #     },
            #     "DecisionTree": {
            #         "max_depth": [None, 10, 20],
            #         "min_samples_split": [2, 5],
            #         "min_samples_leaf": [1, 2]
            #     },
            #     "KNeighbors": {
            #         "n_neighbors": [3, 5, 7],
            #         "weights": ["uniform", "distance"]
            #     },
            #     "GaussianNB": {},
            #     "MultinomialNB": {
            #         "alpha": [0.1, 1.0, 2.0]
            #     },
                
            # }
            models = {
                "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
                "DecisionTree": DecisionTreeClassifier(random_state=42),
                "GaussianNB": GaussianNB()
            }
            params = {
                "RandomForest": {"n_estimators": [50, 100], "max_depth": [None, 10]},
                "LogisticRegression": {"C": [1.0], "solver": ["liblinear"]},
                "DecisionTree": {"max_depth": [None, 10]},
                "GaussianNB": {}
            }


            model_report:dict=evaluate_classifiers(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,params=params)
            ## To get best model score from dict        
            best_model_score = max(sorted(model_report.values()))   
            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score<0.6:
                raise Exception("No best model found")  
            logging.info(f"Best model found on both training and testing dataset")
            logging.info(f"Best Model Name: {best_model_name}, Accuracy Score: {best_model_score}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
                        
            # Encode labels
            le = LabelBinarizer()
            y_train_enc = le.fit_transform(y_train)
            y_test_enc = le.transform(y_test)

            # Fit model on encoded labels
            best_model.fit(X_train, y_train_enc)

            # Predictions
            predicted_enc = best_model.predict(X_test)
            y_score = best_model.predict_proba(X_test)

            # Decode predictions if you want string labels
            predicted = le.inverse_transform(predicted_enc)

            # Metrics
            accuracy = accuracy_score(y_test, predicted)  # original string labels
            f1 = f1_score(y_test_enc, predicted_enc, average='weighted')  # encoded labels
            try:
                roc_auc = roc_auc_score(y_test_enc, y_score, multi_class='ovr', average='weighted')
            except ValueError:
                roc_auc = 0 # Handle case where ROC-AUC cannot be computed
            return accuracy,f1, roc_auc
        


        except Exception as e:  
            raise e