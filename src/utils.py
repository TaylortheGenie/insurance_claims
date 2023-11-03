import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(x_train,y_train,x_test,y_test,models,params):
    try:

        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs=GridSearchCV(model,param,cv=3)
            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)     #train the model

            logging.info("The best parameters are %s", gs.best_params_)
            

            y_train_preds = model.predict(x_train)

            y_test_preds = model.predict(x_test)

            train_score = f1_score(y_train, y_train_preds)

            test_score = f1_score(y_test, y_test_preds)

            report[list(models.keys())[i]] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)
    
def scale(df,col,max,min):
    try:
        df[col] = (df[col]-min)/(max-min)
        #return df

    except Exception as e:
        raise CustomException(e,sys)