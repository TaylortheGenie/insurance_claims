# Vehicle Insurance Claim Prediction

**Author**: Eugene Taylor Sampson <br />
**Email**: esampson692@gmail.com <br />
**LinkedIn**: https://www.linkedin.com/in/eugene-taylor-sampson-67869621b/ <br />

:exclamation: If you find this repository, do not hesitate to reach out. Thanks! :exclamation:

## Introduction

This is an end to end machine learning project using flask for deployment.

The goal of this project is to predict the *possibility* that an insurance policyholder files a claim within the next six months. This is a classification analysis project.

**Predictor variables**:

The raw dataset comprises of 43 predictor variables, and had to be filtered down to just 5 variables-a suitable balance between user input and model performance had to be taken into consideration.

These 5 features are:

`policy_tenure`: The duration of the insurance policy.

`age_of_car`: The age of the policyholder's vehicle.

`age_of_policyholder`: The age of the policyholder.

`population_density`: The number people per unit of area.

`steering_type`: The type of steering used in the policyholder's vehicle. This was the only categorical variable used for generating predictions.

**Target variable**:

`is_claim`: A boolean indicator for whether or not the claim was filed. This variable is numerical in nature and extremely imbalanced. 

Special thanks to Analytics Vidhya for providing the [dataset](https://www.kaggle.com/datasets/avikumart/analytics-vidhya-nov22-insurance-claims-dataset).

## Directories

- `artifacts`: This folder contains the datasets and pickled files.

- `catboost_info`: Contains information on the Catboost models used for training and testing.

- `notebook`: This folder contains the notebooks involved in the data exploration and machine learing prediction phases of the project. This is useful for those who may want to view the predictions in a more traditional way.

- `src`: This is the source folder, which contains python scripts for data ingestion, transformation, preprocessing and model training. Python scripts for custom error handling, information logging and functions for the project can also be found in this folder. 

- `static`: This folder contains images for the various webpages. It is essential to the flask application.

- `templates`: This folder contains HTML scripts for the webpages designs.

- `app.py`: Python script for flask application.

- `README.md`: Marked down file containing information about the project. 

- `requirements.txt`: Text file containing libraries relevant to the project.

- `setup.py`: Python script for installing the libraries present in 'requirements.txt'.

## Project Approach

### Backend Development

1. Problem identification:
    - I wanted to determine the factors involved in filing insurance claims in the short term.
    - Research on possible datasets led me to find and download the dataset from Kaggle.

2. Virtual environment:
    - A virtual environment was created.

3. Project set-up:
    - Relevant libraries were installed, with logging, error handling and a compilation of functions to be used in the project.

4. Data ingestion:
    - The Kaggle dataset is read and stored as a csv file.
    - The data is then split into training and testing sets, which are also stored.

5. Data analysis:
    - Exploratory and explanatory analysis was conducted on the dataset.
    - Interesting insights were obtained from the analysis.
    - The target variable was extremely imbalanced, and sampling techniques had to be implemented.
    - Categorical and numerical variables were correlated to each other and were considered redundant.
    - There were no missing values, nor duplicated variables in the dataset
    - Most features had low cardinality.
    - Three numerical features had already been scaled by the dataset provider, and had to be taken into consideration for the predictive models.

6. Data transformation:
    - The best set of features were obtained and selected in this project phase.
    - The RandomUnderSampler yielded the best results for the model after experimentation, and was implemented on the train and test splits.
    - A column transformer pipeline was created to handle numerical and categorical features.
    - The preprocessing object was saved as a pickled file.

7. Model training:
    - Grid search cross validation was implemented on the train data, alongside hyperparemeter tuning.
    - The best model and its corresponding hyperparamters were returned.
    - The metric used for the basis of the selection is the f1_score.
    - The best model is the GradientBoostingClassifier.

8. Prediction pipeline:
    - This pipeline was necessary for the flask application. 
    - The decision threshold was implemented in this pipeline.
    - It converts all the user input into a dataframe and the generates predictions using the pickled files.

9. Flask application creation:
    - The flask application is created to use user input to generate predictions inside a web application.
