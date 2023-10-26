import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from imblearn.under_sampling import RandomUnderSampler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        
        try:
            features=['policy_tenure', 'age_of_car', 'age_of_policyholder', 'population_density', 'cylinder', 'transmission_type','rear_brakes_type', 'is_parking_camera', 'is_tpms', 
                      'is_parking_sensors','is_rear_window_wiper', 'is_rear_window_defogger']
            target=['is_claim']
            scale_features=['population_density', 'cylinder']
            categorical_features=['transmission_type','rear_brakes_type', 'is_parking_camera', 'is_tpms', 'is_parking_sensors','is_rear_window_wiper', 'is_rear_window_defogger']

            num_preprocessor = Pipeline(
                steps=[
                ('scaler', MinMaxScaler(feature_range=(0,1)))
                ]
            )

            cat_preprocessor = Pipeline(
                steps=[
                ('encoder', OrdinalEncoder()),
                ('scale', MinMaxScaler(feature_range=(0,1)))
                ]
            ) 
                  
            #code for undersampling the data set and redefining features
            
            
            logging.info("Numerical columns scaling completed")
            logging.info("Categorical columns encoding completed.")
            
            preprocessor  = ColumnTransformer(
                [
                ('num', num_preprocessor, scale_features),
                ('cat', cat_preprocessor, categorical_features),
                ], remainder='passthrough'
            )

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)

            test_df=pd.read_csv(test_path)

            logging.info("Reading train and test data completed")

            logging.info("Obtaining preprocessing object.")

            preprocessing_obj=self.get_data_transformer_object()

            features=['population_density', 'cylinder', 'transmission_type','rear_brakes_type', 'is_parking_camera', 'is_tpms', 'is_parking_sensors','is_rear_window_wiper', 'is_rear_window_defogger',
                      'policy_tenure', 'age_of_car', 'age_of_policyholder', 'is_claim']
            target=['is_claim']
            numerical_features=['population_density', 'cylinder', 'policy_tenure', 'age_of_car', 'age_of_policyholder', ]
            categorical_features=['transmission_type','rear_brakes_type', 'is_parking_camera', 'is_tpms', 'is_parking_sensors','is_rear_window_wiper', 'is_rear_window_defogger']

            train_df,target_train_df=RandomUnderSampler(random_state=45).fit_resample(train_df[features[0:12]],train_df[target])
            input_train_df=train_df[features[0:12]]
            #target_train_df=train_df[target]

            input_test_df=test_df[features[0:12]]#.drop(columns=[target],axis=1)
            target_test_df=test_df[target]

            logging.info(
                f"Applying preprocessing object on train and test sets."
            )

            input_train_arr=preprocessing_obj.fit_transform(input_train_df)
            input_test_arr=preprocessing_obj.transform(input_test_df)

            train_arr=np.c_[
                input_train_arr, np.array(target_train_df)
            ]
            test_arr=np.c_[
                input_test_arr, np.array(target_test_df)
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)
        

