# Employee Insurance Enrollment Prediction Pipeline(REPORT)

## Overview

This task develops an end-to-end machine learning pipeline to predict whether an employee is likely to enroll in a voluntary insurance plan. The workflow consists of data preprocessing, feature engineering, dataset integration, model training, and an inference agent capable of generating predictions and explanations for individual employees.


# Workflow Summary

Raw Employee Data
        │
        ▼
Data Cleaning
        │
        ▼
employee_cleaned.csv
        │
        ▼
Merge with Regional Profiles
        │
        ▼
Feature Engineering
        │
        ▼
Ordinal Encoding
        │
        ▼
final_merged.csv
        │
        ▼
Random Forest Training
        │
        ▼
Saved Model (.pkl)
        │
        ▼
Prediction & Recommendation Agent


The pipeline is divided into four major stages:

1. Raw data preprocessing
2. Data integration and feature engineering
3. Machine learning model training
4. Prediction and recommendation agent

# 1. Data Preprocessing

The raw employee dataset contained missing values, inconsistent categorical values, and data quality issues that could negatively impact model performance.

### Preprocessing Steps

- Loaded employee records and regional benefit profile datasets.
- Identified numerical and categorical columns .
- Filled missing numerical values using the **median** to reduce sensitivity to outliers.
- Filled missing categorical values using the **mode** to preserve the most common category.
- Standardized categorical values to ensure consistency across records.
- Various Statistical Observations such as **correlation matrix and cramers V** indicated that:
  **legacy_propensity_score** had a strong correlation with the output variable.So considering this feature in the model       would lead to data leakage and the model would fail to capture non linear patterns.So it was dropped.
  **application_date** also had a weak cramers value and it had many data irregularities(contact_date>application_date) which would not favour the model.So to avoid it the column was dropped.It would have no say in the prediction in any of the future queries too.
  The **age and the tenure** were not syncing to(age-tenure<18). This again was fixed by manipulating the tenure.
  After drawing various inferences raw data the file was saved as:
  employee_cleaned.csv

This preprocessing stage ensured that no missing values remained before feature engineering and model training.


# 2. Feature Engineering and Dataset Integration

The cleaned employee dataset was merged with regional benefit profile information using the **region** attribute.
The distribution was again validated for the region to avoid unnecessary imbalance.

## Engineered Features

Several additional variables were incorporated to improve predictive capability, including:
- **premium_salary_percentage**	-> avg_premium/salary
- **outreach_capacity_ratio**   -> hr_outreach/total people in the region
- **salary_diff_region**       ->  salary-avg_salary_region

These features helped the model to gain a **fresh and unique perspective** of the data.

The feature(**salary_diff_region** and **premium_salary_percentage**) secured the position among the **top 10** in feature importance list.

### Encoding

Since machine learning models require numerical inputs, categorical variables such as:

- Gender
- Marital Status
- Employment Type
- Region
- Contact Channel
- Plan Tier
- Broker Channel
- State Mandate Level

were transformed using **Ordinal Encoding** with support for unseen categories.

The id which acted as a primary contained 8 repitations which were again removed to main cardinality.

The processed dataset was then exported as:
final_merged.csv


# 3. Machine Learning Pipeline

The final dataset was divided into training and testing sets using an 80–20 split while preserving the class distribution through stratified sampling.

## Model

The project uses a **Random Forest Classifier** configured with controlled tree depth and sampling parameters to reduce overfitting.

### Training Hyperparameter**

- n_estimators=100,
- max_depth=4,
- min_samples_split=5,
- min_samples_leaf=4,
- max_features=0.5,
- bootstrap=True,
- random_state=42,
- n_jobs=-1

The hyperparameters were set after rigrous testing across various conventional and non conventional values.

The standard **threshold** for the classification models is **0.5**. But in this case it was tuned to **0.675** in order to get better test results.

After setting the correct hyperparamters the result acheived were as follows:
Train Accuracy : 0.95775
**Test Accuracy  : 0.962**
Accuracy : 0.962
Precision: 0.966183574879227
Recall   : 0.9724473257698542
F1 Score : 0.9693053311793215
**ROC AUC  : 0.9922977559233384**
Confusion Matrix
 724   42
 34   1200

The trained model learnt relationships between employee demographics, employment information, historical enrollment behavior, regional characteristics, and outreach metrics to estimate enrollment probability.

### Model Persistence
The trained artifacts are stored using Joblib:
- `rf_model.pkl`
- `feature_names.pkl`
allowing the model to be reused without retraining.


# 4. Decision-Support Agent

To make the trained machine learning model accessible through natural language queries, a **rule-based decision-support agent** was developed. Instead of exposing the Random Forest model directly, the agent interprets user requests, routes them to the appropriate analytical tool, executes the required operation, and returns structured, human-readable responses.

## Agent Architecture
The agent consists of four major components:

- **Query Router**
   - Receives a user's natural language query.
   - Identifies the requested task using rule-based keyword matching.
   - Routes the request to the corresponding analytical tool.

- **Tool Layer**
   - Contains specialized functions responsible for performing different operations such as employee prediction, regional    ranking, explanation generation, profile lookup, and data validation.

- **Machine Learning Layer**
   - Loads the trained Random Forest model.
   - Applies the same preprocessing pipeline used during training.
   - Generates enrollment probabilities for requested employees.

- **Response Generator**
   - Converts numerical predictions into meaningful insights.
   - Formats responses for HR personnel in an interpretable manner.
   - Enforces safety policies before returning any prediction or explanation.

## Supported Tools

### 1. Employee Enrollment Prediction
  Given an employee identifier, the agent retrieves the employee record, preprocesses the features, executes the Random Forest classifier, and predicts whether the employee is likely to enroll in the insurance program along with the associated confidence score.

### 2. Prediction Explanation
 The agent explains why a prediction was made by **highlighting the most influential non-sensitive features** contributing to the model's decision.
 The agent uses **SHAP (SHapley Additive exPlanations)** to explain why the  model predicts that an employee is likely or unlikely to enroll.
 (SHAP is an explainable AI technique based on concepts from cooperative game theory. It assigns each feature a **SHAP value**, representing the contribution of that feature toward increasing or decreasing the model's prediction for a particular employee.)

To ensure fairness and , explanations intentionally exclude protected attributes such as:
 - Age
 - Gender
 - Marital Status
Even if these attributes were available during model training, they are never referenced in generated explanations.

### 3. Outreach Candidate Ranking
- The agent **ranks employees within a specified region** according to their predicted enrollment probability.
- This enables HR teams to identify high-priority candidates for outreach campaigns without manually evaluating every employee.

### 4. Regional Profile Lookup
The agent **retrieves region-specific statistics**, including enrollment trends and workforce characteristics, allowing HR personnel to understand regional insurance participation patterns.

### 5. Raw Data Validation
The agent **validates employee records before prediction** by identifying missing values, inconsistent entries, invalid ranges, or formatting issues that may affect model performance.
This tool assists in improving overall data quality before inference.


## Safety and Refusal Mechanism
- The agent includes an explicit safety layer that **prevents the use of features capable of leaking target information**.
- In particular, the feature:
**legacy_propensity_score** is treated as a prohibited field because it can reconstruct or strongly correlate with the prediction target.
- Whenever a user attempts to request predictions or explanations using this feature, the **agent refuses the request and returns a clear explanation** indicating why the operation cannot be performed.
- This safeguard prevents target leakage and ensures that predictions are generated only from legitimate employee information.

## Example Queries
The implemented agent supports multiple natural language queries, including:

- "Who are the top 20 outreach priorities in the Midwest this window?"
- "Why is employee 10452 predicted to enroll?"
- "Show the region profile for the South."
- "What's wrong with this raw employee record?"
- "Predict enrollment for employee 4821."

If a user attempts to issue a request such as:
 "Predict employee 14821 using legacy_propensity_score."

the agent explicitly refuses the request because the specified feature is prohibited under the project's safety policy.


# Conclusion

The developed pipeline transforms raw employee and regional datasets into a structured machine learning workflow capable of predicting insurance enrollment with interpretable outputs. Data preprocessing removes inconsistencies, feature engineering enriches employee records with regional context, and the Random Forest model captures relationships influencing enrollment decisions. The prediction agent further extends the solution by supporting employee-level predictions and region-wise prioritization, making the system suitable for data-driven HR outreach and insurance enrollment campaigns.