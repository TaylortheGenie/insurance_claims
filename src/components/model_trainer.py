import os 
import sys
from dataclasses import dataclass

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

#selected features
features=['population_density','steering_type','policy_tenure','age_of_car','age_of_policyholder']

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:  
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test sets")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Support Vector Machines": SVC(random_state=45),
                "Random Forests": RandomForestClassifier(random_state=45),
                "Gradient Boosting": GradientBoostingClassifier(random_state=45),
                "AdaBoost": AdaBoostClassifier(random_state=45),
                "XGBoost": XGBClassifier(random_state=45),
                "CatBoost": CatBoostClassifier(random_state=45)
            }

            params={
                "Support Vector Machines":{
                    'C':[0.1, 0.5, 1, 2, 5, 10],
                    'gamma':['scale', 'auto', 0.5, 1, 1.5, 2],
                    'shrinking':[True, False],
                    'tol':[1e-3, 1e-2, 1e-1]
                },
                "Random Forests":{
                    'criterion' : ['gini', 'entropy', 'log_loss'],
                    'n_estimators': [8,16,32,64,128,256],
                    'max_depth' : [None, 5, 10, 15, 20],
                    'min_samples_split' : [2, 3, 4, 5, 6]
                },
                "Gradient Boosting":{
                    'learning_rate':[0.1,0.5,1,2],
                    'max_depth':[3,6,9,12],
                    'n_estimators':[8,16,32,64,128,256],
                    'min_samples_split' : [2, 3, 4, 5, 6],
                    'subsample': [0,0.2,0.4,0.6,0.8,1]
                },
                "AdaBoost":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "XGBoost":{
                    'max_depth' : [None, 5, 10, 15, 20],
                    'subsample': [0,0.2,0.4,0.6,0.8,1]
                },
                "CatBoost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                }
            }

            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)
        
            #obtain the best model score
            best_model_score = max(sorted(model_report.values()))

            #to get the best model's name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            logging.info("Best model obtained: %s", best_model)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)

            f1 = f1_score(y_test, predicted)
            return f1

        except Exception as e:
            raise CustomException(e, sys)