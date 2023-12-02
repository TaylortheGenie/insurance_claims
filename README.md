# Vehicle Insurance Claim Prediction

**Author**: Eugene Taylor Sampson <br />
**Email**: esampson692@gmail.com <br />
**LinkedIn**: https://www.linkedin.com/in/eugene-taylor-sampson-67869621b/ <br />

:exclamation: If you find this repository, do not hesitate to reach out. Thanks! :exclamation:

## Introduction

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

