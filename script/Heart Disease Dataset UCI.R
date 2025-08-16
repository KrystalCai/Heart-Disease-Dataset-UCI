
# ===============================================================
# Project: Heart Disease Analysis (UCI Dataset)
# Author: Krystal Cai
# Date: 2025-08-16
# Language: R
# ===============================================================


# -------------------- Introduction ----------------------------
# This project analyzes the UCI Heart Disease dataset to explore 
# potential clinical predictors of heart disease. The dataset 
# contains 1025 patient records with 14 features, including 
# demographic, physiological, and medical examination data.
#
# Objectives:
# 1. Conduct basic hypothesis testing (t-tests, chi-square tests) 
#    to identify significant differences in key variables between 
#    patients with and without heart disease.
# 2. Build logistic regression models to predict heart disease 
#    occurrence based on clinical features.
# 3. Evaluate model performance with goodness-of-fit tests, 
#    cross-validation, ROC curves, and LASSO regularization.
#
# Methods:
# - Data cleaning and exploration using 'tidyverse'
# - Hypothesis testing for continuous (t-test) and categorical 
#   (chi-square) variables
# - Logistic regression (GLM with binomial family)
# - Model diagnostics: multicollinearity (VIF), interaction terms,
#   non-linear terms, Hosmer-Lemeshow test
# - Model evaluation: ROC curve, cross-validation, OR estimation
#
# This analysis provides insights into the most important 
# clinical features contributing to heart disease and develops 
# a predictive model with high accuracy and interpretability.
# ===============================================================

library(tidyverse)

h <- read.csv("~/Desktop/Heart Disease Dataset UCI/HeartDiseaseTrain-Test.csv")
str(h)
head(h)

# Convert categorical variables to factors
h$sex <- factor(h$sex, levels = c("Male", "Female"))
h$chest_pain_type <- factor(h$chest_pain_type)
h$fasting_blood_sugar <- factor(h$fasting_blood_sugar)
h$rest_ecg <- factor(h$rest_ecg)
h$exercise_induced_angina <- factor(h$exercise_induced_angina)
h$slope <- factor(h$slope)
h$vessels_colored_by_flourosopy <- factor(h$vessels_colored_by_flourosopy)

# Logistic regression model to predict heart disease (target)
model <- glm(target ~ age + sex + chest_pain_type + resting_blood_pressure + cholestoral + 
               fasting_blood_sugar + rest_ecg + Max_heart_rate + exercise_induced_angina + 
               oldpeak + slope + vessels_colored_by_flourosopy + thalassemia, 
             family = binomial, data = h)

# Output the model results
summary(model)

# The model indicates several key predictors of heart disease, with gender, chest pain type, resting blood pressure, and heart rate being among the most significant.

# The overall fit of the model seems good, with an AIC value of 652.82, suggesting that the model does a reasonable job at predicting heart disease.

# t-test: Check if there is a significant difference in age based on heart disease presence
t_test_age <- t.test(age ~ target, data = h)
print(t_test_age)

# t-test: Check if there is a significant difference in max heart rate based on heart disease presence
t_test_max_hr <- t.test(Max_heart_rate ~ target, data = h)
print(t_test_max_hr)


# This study performed independent two-sample t-tests to examine differences in age and maximum heart rate (Max_heart_rate) between individuals with and without heart disease (target=0/1). Results indicated:

# For age, the mean age in the no-heart-disease group (target=0) was 56.57 years, compared to 52.41 years in the heart-disease group (target=1). The difference was statistically significant (p < 0.001), suggesting that in this sample, individuals with heart disease were younger on average than those without.

# For maximum heart rate, the mean value was 139.13 bpm in the no-heart-disease group and 158.59 bpm in the heart-disease group, with the difference being highly significant (p < 0.001). This indicates that heart disease patients had a significantly higher maximum heart rate than non-patients in this dataset.

# Consistent with the logistic regression analysis, maximum heart rate remained a significant positive predictor in the multivariate model, while age was only marginally significant. This suggests that maximum heart rate may be a more stable predictor of heart disease risk, whereas the effect of age may be influenced by other covariates.

# Load required packages
library(tidyverse)
library(car)               # For VIF
library(ResourceSelection) # For Hosmer-Lemeshow test
library(pROC)              # For ROC and AUC

#---------------------------
# 1. Chi-square tests for categorical variables
#---------------------------
# Convert to factor if needed
categorical_vars <- c("sex", "chest_pain_type", "fasting_blood_sugar",
                      "rest_ecg", "exercise_induced_angina",
                      "slope", "vessels_colored_by_flourosopy", "thalassemia")

h[categorical_vars] <- lapply(h[categorical_vars], as.factor)

# Run Chi-square tests
chi_results <- lapply(categorical_vars, function(var) {
  tbl <- table(h[[var]], h$target)
  test <- chisq.test(tbl)
  list(variable = var, p_value = test$p.value)
})

chi_results
# Except for fasting_blood_sugar, all other categorical variables show significant association with heart disease status.



#---------------------------
# 2. VIF check for multicollinearity
#---------------------------
# Refit the logistic model
model <- glm(target ~ age + sex + chest_pain_type + resting_blood_pressure + cholestoral + 
               fasting_blood_sugar + rest_ecg + Max_heart_rate + exercise_induced_angina + 
               oldpeak + slope + vessels_colored_by_flourosopy + thalassemia, 
             family = binomial, data = h)

# Calculate VIF
vif_values <- vif(model)
vif_values

# No significant multicollinearity detected among predictors (all VIF values are well below 5).

#---------------------------
# 3. Hosmer-Lemeshow goodness-of-fit test
#---------------------------
# Group into 10 groups for the test
hl_test <- hoslem.test(h$target, fitted(model), g=10)
hl_test

# The Hosmer–Lemeshow test shows p < 0.001, indicating poor calibration — the model’s predicted probabilities deviate from actual outcomes.

#---------------------------
# 4. ROC and AUC
#---------------------------
roc_obj <- roc(h$target, fitted(model))
auc_value <- auc(roc_obj)

# Plot ROC curve
plot(roc_obj, main=paste("ROC Curve - AUC =", round(auc_value, 3)))

# The ROC curve lies above the diagonal, with an AUC around 0.9+, indicating strong discriminative ability.

# Conclusion from 1 to 4:Most predictors are significantly associated with heart disease. The model has strong discriminative power (high AUC) but poor calibration per Hosmer–Lemeshow, suggesting room for improvement in model fit.







# Add interaction and non-linear terms
model2 <- glm(
  target ~ age + sex + chest_pain_type + resting_blood_pressure + cholestoral +
    fasting_blood_sugar + rest_ecg + Max_heart_rate + exercise_induced_angina +
    oldpeak + I(oldpeak^2) + slope + vessels_colored_by_flourosopy + thalassemia +
    age:sex + Max_heart_rate:chest_pain_type,
  family = binomial, data = h
)

summary(model2)



library(caret)
set.seed(123)
cv_control <- trainControl(method = "cv", number = 10, classProbs = TRUE,
                           summaryFunction = twoClassSummary)

# Convert target to factor for caret
h$target <- factor(h$target, levels = c(0, 1), labels = c("No", "Yes"))

cv_model <- train(
  target ~ age + sex + chest_pain_type + resting_blood_pressure + cholestoral +
    fasting_blood_sugar + rest_ecg + Max_heart_rate + exercise_induced_angina +
    oldpeak + I(oldpeak^2) + slope + vessels_colored_by_flourosopy + thalassemia +
    age:sex + Max_heart_rate:chest_pain_type,
  data = h, method = "glm", family = "binomial",
  trControl = cv_control, metric = "ROC"
)

cv_model


#result explain
exp(cbind(OR = coef(model2), confint(model2)))


#LASSO
library(glmnet)
x <- model.matrix(target ~ age + sex + chest_pain_type + resting_blood_pressure + cholestoral +
                    fasting_blood_sugar + rest_ecg + Max_heart_rate + exercise_induced_angina +
                    oldpeak + slope + vessels_colored_by_flourosopy + thalassemia, h)[,-1]
y <- as.numeric(h$target) - 1

set.seed(123)
lasso_cv <- cv.glmnet(x, y, alpha = 1, family = "binomial")
coef(lasso_cv, s = "lambda.min")

# Using the UCI Heart Disease dataset, this study employed hypothesis testing, logistic regression, and model diagnostics to identify significant clinical predictors of heart disease, including sex, chest pain type, maximum heart rate, exercise-induced angina, oldpeak, number of coronary vessels, and thalassemia type. The enhanced model showed excellent predictive performance (cross-validated ROC = 0.940) and confirmed model stability. Some variables, such as fasting blood sugar and rest ECG, were not significant, suggesting the need for further validation in future studies.
















