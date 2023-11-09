import sys 
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            preprocessor=load_object(file_path=preprocessor_path)
            model=load_object(file_path=model_path)
            data_processed=preprocessor.transform(features)
            preds=model.predict_proba(data_processed)[:,1]
            preds=pd.Series(preds)
            preds=np.where(preds<0.533,0,1)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:      #mapping all inputs from html to backend
    def __init__(self,
        policy_tenure: int,
        age_of_car: int,
        age_of_policyholder: int,
        population_density: int,
        steering_type: str):
        
        self.policy_tenure = policy_tenure

        self.age_of_car = age_of_car
        
        self.age_of_policyholder = age_of_policyholder

        self.population_density = population_density

        self.steering_type = steering_type

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "policy_tenure": [self.policy_tenure],
                "age_of_car": [self.age_of_car],
                "age_of_policyholder": [self.age_of_policyholder],
                "population_density": [self.population_density],
                "steering_type": [self.steering_type]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)