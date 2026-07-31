# Employee Insurance Enrollment Prediction & Decision Support Agent




## Overview
 **1.The Problem:**
 Many organizations offer voluntary insurance and employee benefit programs to improve workforce well-being. However, identifying employees who are most likely to enroll remains a significant challenge. HR teams often rely on broad outreach campaigns or manual analysis, leading to inefficient resource allocation, lower engagement rates, and increased operational costs. Furthermore, employee data is typically fragmented across multiple sources, requiring extensive preprocessing before meaningful insights can be generated.

  **2.Proposed Solution:**
 To address this challenge, this project presents an end-to-end machine learning solution that predicts employee insurance enrollment using demographic, employment, historical, and region-specific information. The pipeline performs data cleaning, missing value imputation, feature engineering, categorical encoding, and dataset integration before training a Random Forest classifier to estimate enrollment probability. On top of the predictive model, a rule-based decision-support agent enables users to interact with the system through natural language queries, supporting employee-level predictions, SHAP-based explanations, outreach candidate ranking, regional profile lookup, and raw data validation while enforcing safeguards against target leakage and unsafe feature usage.

  **3.Uniqueness and Innovation:**
 The uniqueness of this project lies in combining predictive machine learning with an explainable and safety-aware decision-support agent rather than exposing only a standalone classifier. The system integrates **Python, Pandas, NumPy, Scikit-learn, SHAP, Joblib, and Jupyter Notebook** to build the complete analytics pipeline. It incorporates SHAP (SHapley Additive Explanations) for transparent prediction explanations, implements a custom query-routing agent for multi-tool interaction, and includes explicit refusal mechanisms to prevent the use of prohibited features such as `legacy_propensity_score`. These innovations make the solution not only accurate but also interpretable, responsible, and practical for real-world HR decision support.


This project develops an end-to-end machine learning pipeline for predicting whether an employee is likely to enroll in a voluntary insurance program. It includes:

- Data preprocessing
- Feature engineering
- Random Forest model training
- SHAP-based prediction explanations
- Rule-based decision-support agent with multiple analytical tools

The agent supports natural language queries for prediction, explanation, employee ranking, region lookup, and raw data validation while enforcing safety policies to prevent target leakage.

## Features
- Missing value imputing
- Feature engineering
- Ordinal Encoding
- Random Forest Classification
- SHAP Explainability
- Employee prediction
- Outreach candidate ranking
- Region profile lookup
- Raw row validation
- Explicit refusal for prohibited features

## Instructions to run
1. commit the repository
2. install the requirements
3. run  `agent.ipynb`

The notebook loads the preprocessed datasets and the trained Random Forest model directly from disk.
No preprocessing or retraining is required.

### Reproducing the Complete Pipeline:

If you wish to reproduce the project from the raw datasets, execute the notebooks in the following order:

1. `preprocess_1.ipynb`
2. `preprocess_2.ipynb`
3. `final_pipeline.ipynb`
4. `agent.ipynb`

These notebooks perform data cleaning, feature engineering, model training, and finally launch the decision-support agent.

Note: The raw dataset provided for the assignment has been excluded from the repository using the `.gitignore` file.

## AI Usage
Primary AI Assistant:
- **OpenAI ChatGPT (GPT-5.5)**

AI assistance was used as a development aid. All preprocessing, feature engineering, model training, testing, and final verification were performed and validated by the task author.


