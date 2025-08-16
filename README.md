# Heart Disease Dataset UCI

This project analyzes the **UCI Heart Disease** dataset to identify the key factors that influence the likelihood of having heart disease. The analysis includes various statistical tests, a logistic regression model, and model evaluation techniques like ROC curves and cross-validation.

## Project Overview

- **Objective**: 
  - Explore key clinical factors related to heart disease using hypothesis testing and regression modeling.
  - Build and evaluate predictive models to predict heart disease based on clinical data.
  
- **Methods**: 
  - **Hypothesis Testing**: Conduct t-tests and chi-square tests to examine the relationship between clinical variables and heart disease.
  - **Logistic Regression**: Use GLM (Generalized Linear Model) with a binomial family to build a predictive model.
  - **Model Evaluation**: Evaluate model performance using cross-validation, ROC curve, and Hosmer-Lemeshow test.
  - **Feature Selection**: Apply LASSO (Least Absolute Shrinkage and Selection Operator) regularization to identify significant features.

## Data

The dataset used in this project is the **Heart Disease UCI dataset**, which contains information on 1025 patient records with 14 features. The data can be found in the file `HeartDiseaseTrain-Test.csv` and includes features such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol Levels
- Max Heart Rate
- Exercise-Induced Angina
- and more...

## File Structure
Heart Disease Dataset UCI/
│
├─ HeartDiseaseTrain-Test.csv # Dataset
├─ Heart Disease Dataset UCI.R # Main R analysis script
├─ Heart Disease Dataset UCI.Rmd # R Markdown file for report
├─ Heart-Disease-Dataset-UCI.pdf # Output report (PDF)
└─ README.md # Project description


### Instructions

1. **Clone the repository**:

   You can clone this repository to your local machine using the following command:

   ```bash
   git clone https://github.com/YourUsername/Heart-Disease-Dataset-UCI.git

2. **Set Up R Environment**:
install.packages(c("tidyverse", "caTools", "glmnet", "pROC", "caret", "MASS"))

3. **Run the Analysis**:

source("Heart Disease Dataset UCI.R")

4. **Generate the Report**:
To generate a detailed report in HTML or PDF format, open the R Markdown file Heart Disease Dataset UCI.Rmd in RStudio and click the Knit button to create a report.

5. **Evaluation**:

Once the analysis is complete, you can view the output model performance, including the ROC curve, cross-validation results, and the list of significant predictors.




**Results**

The logistic regression model achieved high accuracy, and the ROC curve showed an AUC of 0.940.

Key predictors of heart disease include sex, chest pain type, resting blood pressure, and exercise-induced angina.