import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from sklearn.metrics import f1_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:

        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(x_train, y_train) #train the model 

            y_train_preds = model.predict(x_train)

            y_test_preds = model.predict(x_test)

            train_score = f1_score(y_train, y_train_preds)

            test_score = f1_score(y_test, y_test_preds)

            report[list(models.keys())[i]] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
