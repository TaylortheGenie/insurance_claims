from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.utils import scale

application=Flask(__name__)


#route for homepage

@application.route('/')
def home():
    return render_template('home.html') #paragraph and tings from home

@application.route('/predict',methods=['POST'])
def predict():
        data=CustomData(
            policy_tenure=float(request.form.get('policy_tenure')),
            age_of_car=float(request.form.get('age_of_car')),
            age_of_policyholder=float(request.form.get('age_of_policyholder')),
            population_density=float(request.form.get('population_density')),
            steering_type=request.form.get('steering_type')
        )

        preds_df=data.get_data_as_dataframe()
        #perform scaling
        scale(preds_df,'policy_tenure',5,0.5)     #for the policy tenure
        scale(preds_df,'age_of_car',15,0)     #for the vehicle age
        scale(preds_df,'age_of_policyholder',52,18)     #for age of the policyholder
        
        print(preds_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(preds_df)
        return render_template('predicted.html', results=results[0])
    
@application.route("/warning.png")
def warning():
    return render_template("warning.png")

@application.route("/okay.jpg")
def okay():
    return render_template("okay.jpg")

if __name__=="__main__":
    application.run(host="0.0.0.0")
