#Import required libraries
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from dataclasses import dataclass
from Source.logger import logging
from Source.exception import customexception
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import  r2_score
from Source.utils.utils import evaluate_model
from Source.utils.utils import save_object

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train_array,test_array):

        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'LinearRegression': LinearRegression(),
            'DecisionTreeRegressor': DecisionTreeRegressor(),
            'RandomForestRegressor': RandomForestRegressor(),
            'RandomizedSearchCV' : RandomizedSearchCV()
            }

            # Hyperparameter grid for RandomForest
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
            # Number of features to consider at every split
            max_features = ["auto", "sqrt"]
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10, 15, 100]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 5, 10]
            param_grid = {
            'n_estimators': n_estimators,
            'max_features' : max_features, 
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf' :min_samples_leaf
            }
            #Hyperparameter tuning using RandomSearchCV

            Random_search = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions= param_grid, scoring="neg_mean_squared_error",cv=3, n_jobs=1, verbose=2,random_state=42)
            
            print("Best parameters found by RandomizedSearchCV :")
            print(Random_search.best_params_)
            Random_search.fit(X_train, y_train)
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_names = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
           

            best_model_name = models[best_model_names]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            
            # Save the best model
            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model_name
            )
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise customexception(e,sys)

       