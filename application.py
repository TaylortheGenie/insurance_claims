from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.utils import scale

application=Flask(__name__)
app = application


#route for homepage

@app.route('/')
def home_page():
    return render_template('index.html') 

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    if request.method=='GET':
        return render_template('index.html')
    
    else:
        data=CustomData(
            policy_tenure=float(request.form.get('policy_tenure')),
            age_of_car=float(request.form.get('age_of_car')),
            age_of_policyholder=float(request.form.get('age_of_policyholder')),
            population_density=float(request.form.get('population_density')),
            steering_type=request.form.get('steering_type')
        )

        preds_df=data.get_data_as_dataframe()
        #perform scaling
        scale(preds_df,'policy_tenure',5,0.5)     
        scale(preds_df,'age_of_car',15,0)     
        scale(preds_df,'age_of_policyholder',52,18)
        
        print(preds_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(preds_df)
        if results[0] == 0:
            results = "low"
        else:
            results = "high"

        return render_template('index.html', prediction="The possibility of filing an insurance within the next six months is {}.".format(results))

if __name__=="__main__":
    app.run(host="0.0.0.0")
