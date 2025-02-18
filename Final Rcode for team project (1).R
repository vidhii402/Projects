##packages and remove it
installed_packages <- rownames(installed.packages())
installed_packages
sapply(installed_packages, remove.packages)

# Install and load required packages
required_packages <- c("caret", "xgboost", "randomForest", "pROC", "smotefamily",
                       "dplyr", "fastDummies", "nnet", "PRROC", "tidyverse","e1071")

install.packages("caret")

install.packages("xgboost")

install.packages("randomForest")

install.packages("pROC")

install.packages("smotefamily")

install.packages("dplyr")

install.packages("fastDummies")

install.packages("nnet")

install.packages("PRROC")

install.packages("tidyverse")

install.packages("e1071")
#####################################################################

#Cleaning process for the raw data
data<-read.csv(file.choose())
print(data)
#install.packages('dplyr')
library(dplyr)
#install.packages('fastDummies')
library(fastDummies)

data <- data %>% select(-id)
data1 <- data %>%
  filter(gender != "Other")
data$gender <- ifelse(data$gender == "Male", 1, 0) #others was removed. male as 1, female as 0

data <- dummy_cols(data, select_columns = "work_type", remove_first_dummy = FALSE)

data$avg_glucose_level <- cut(data$avg_glucose_level, 
                              breaks = c(0, 70, 100, 125, 300), 
                              labels = c("Low", "Normal", "High", "Extremely High"), 
                              include.lowest = TRUE)

data$bmi <- as.numeric(gsub("[^0-9.]", NA, data$bmi))

avg_s <- data %>%
  group_by(avg_glucose_level) %>%
  summarise(avg_s = mean(bmi, na.rm = TRUE))
print(avg_s)
#1 Low                28.2
#2 Normal             28.5
#3 High               28.0
#4 Extremely High     31.4
data <- data %>%
  left_join(avg_s, by = "avg_glucose_level") %>%
  mutate(bmi = ifelse(is.na(bmi), avg_s, bmi)) %>%
  select(-avg_s)  

# Replace n/a bmi by group average according to average glucose level 
data$bmi <- cut(data$bmi, 
                breaks = c(0, 18.5, 25, 30, 100), 
                labels = c("Underweight", "Normal", "Overweight", "Obese"), 
                include.lowest = TRUE)

#bmi categorized
data <- dummy_cols(data, select_columns = "smoking_status", remove_first_dummy = FALSE)
##age
data<- subset(data, age >= 18)
#Remove all people below 18 yrs old as they have a different calculation formula for bmi
data$ever_married <- ifelse(data$ever_married == "Yes", 1, 0)

data <- dummy_cols(data, select_columns = "Residence_type", remove_first_dummy = FALSE)

data <- dummy_cols(data, select_columns = "avg_glucose_level", remove_first_dummy = FALSE)

data <- dummy_cols(data, select_columns = "bmi", remove_first_dummy = FALSE)
data <- data %>% select(-work_type)
data <- data %>% select(-Residence_type)
data <- data %>% select(-avg_glucose_level)
data <- data %>% select(-bmi)
data <- data %>% select(-smoking_status)
write.csv(data,"cleaned_data01.csv",row.names = FALSE)

#install.packages('nnet')
library(nnet)
#install.packages('caret')
library(caret)

# Separate known and unknown smoking status
known_smoking <- data[data$smoking_status_Unknown == 0, ]
unknown_smoking <- data[data$smoking_status_Unknown == 1, ]

#for known smoking status
features <- c('gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type_children',
              'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed',
              'Residence_type_Rural', 'Residence_type_Urban', 'avg_glucose_level_Low', 'avg_glucose_level_Normal',
              'avg_glucose_level_High', 'avg_glucose_level_Extremely High', 'bmi_Underweight', 'bmi_Normal',
              'bmi_Overweight', 'bmi_Obese')

X <- known_smoking[, features]
y <- as.factor(apply(known_smoking[, c('smoking_status_formerly smoked', 
                                       'smoking_status_never smoked', 
                                       'smoking_status_smokes')], 1, which.max))

# Train the model
model <- multinom(y ~ ., data = X)

# Predict smoking status for unknown case
X_unknown <- unknown_smoking[, features]
unknown_predictions <- predict(model, newdata = X_unknown)

# Update the dataset with predictions
for (i in 1:nrow(unknown_smoking)) {
  if (unknown_predictions[i] == 1) {
    unknown_smoking[i, "smoking_status_formerly smoked"] <- 1
  } else if (unknown_predictions[i] == 2) {
    unknown_smoking[i, "smoking_status_never smoked"] <- 1
  } else {
    unknown_smoking[i, "smoking_status_smokes"] <- 1
  }
}

# Set the unknown status to zero
unknown_smoking$smoking_status_Unknown <- 0

# Combine known and updated unknown smoking status data
updated_data <- rbind(known_smoking, unknown_smoking)

# Remove the 'smoking_status_Unknown' column
updated_data <- updated_data %>% select(-smoking_status_Unknown)

#Save the updated dataset
write.csv(updated_data, 'updated_smoking_status_data.csv', row.names = FALSE)
#Changes
cat("Original distribution of smoking status:\n")
print(colSums(data[, c('smoking_status_formerly smoked', 
                       'smoking_status_never smoked', 
                       'smoking_status_smokes', 
                       'smoking_status_Unknown')]))
cat("\nUpdated distribution of smoking status:\n")
print(colSums(updated_data[, c('smoking_status_formerly smoked', 
                               'smoking_status_never smoked', 
                               'smoking_status_smokes')]))
cat("\nNumber of records with unknown smoking status reassigned:", nrow(unknown_smoking))

write.csv(updated_data,"PreAnalysis.csv",row.names = FALSE)

#libraries
library(tidyverse)
library(caret)
library(randomForest)
library(smotefamily) 
library(pROC) 

# Load the data
data <- read.csv("PreAnalysis.csv")
# Convert stroke variable to factor (binary classification)
data$stroke <- as.factor(data$stroke)
# Check the class distribution before applying SMOTE
table(data$stroke)
# Apply SMOTE to handle class imbalance in stroke data
set.seed(123)
smote_data <- SMOTE(X = data[, -which(names(data) == "stroke")], target = data$stroke, K = 5, dup_size = 1)
# Combine the generated data into a dataframe and ensure proper formatting
balanced_data <- smote_data$data
# Rename 'class' to 'stroke'
balanced_data <- balanced_data %>% rename(stroke = class)

# Convert stroke variable in balanced_data to a factor with correct levels
balanced_data$stroke <- as.factor(balanced_data$stroke)

# Split the data into training and testing sets (80% training, 20% testing)
set.seed(123)
trainIndex <- createDataPartition(balanced_data$stroke, p = .8, list = FALSE, times = 1)
train_data <- balanced_data[trainIndex, ]
test_data <- balanced_data[-trainIndex, ]

# Random Forest Classification for causal relationship with cross-validation
set.seed(123)
rf_model <- train(as.factor(stroke) ~ ., 
                  data = train_data, 
                  method = "rf", 
                  trControl = trainControl(method = "cv", number = 5),
                  importance = TRUE)

# Print Random Forest Model Summary
print(rf_model)

# Evaluate the model on the test data
predictions <- predict(rf_model, newdata = test_data)

# Ensure the factor levels for predictions and test_data match
test_data$stroke <- factor(test_data$stroke, levels = c("0", "1"))
predictions <- factor(predictions, levels = c("0", "1"))

# Compute the confusion matrix
conf_matrix <- confusionMatrix(predictions, test_data$stroke)

# Print the confusion matrix to see accuracy, sensitivity, and specificity
print(conf_matrix)

# Feature Importance
varImpPlot(rf_model$finalModel)

# ROC Curve and AUC for the test set
roc_curve <- roc(as.numeric(test_data$stroke), as.numeric(predictions))
auc_value <- auc(roc_curve)
print(paste("AUC: ", auc_value))
plot(roc_curve, main = "ROC Curve for Stroke Prediction")

#Libraries
library(tidyverse)
library(caret)
library(randomForest)
library(smotefamily) 
library(pROC) 

# Load the data we have
data <- read.csv("PreAnalysis.csv")

# Convert stroke variable to factor (binary classification)
#data$stroke <- as.factor(data$stroke)

# Check the class distribution before applying SMOTE
table(data$stroke)

# Apply SMOTE to handle class imbalance
set.seed(123)
smote_data <- SMOTE(X = data[, -which(names(data) == "stroke")], target = data$stroke, K = 5, dup_size = 1)

# Combine the generated data into a dataframe and ensure proper formatting
balanced_data <- smote_data$data
#balanced_data$class <- as.factor(smote_data$class)

# Rename 'class' to 'stroke' to maintain consistency
balanced_data <- balanced_data %>% rename(stroke = class)

# Convert stroke variable in balanced_data to a factor with correct levels
balanced_data$stroke <- as.factor(balanced_data$stroke)

# Split the data into training and testing sets (80% training, 20% testing)
set.seed(123)
trainIndex <- createDataPartition(balanced_data$stroke, p = .8, list = FALSE, times = 1)
train_data <- balanced_data[trainIndex, ]
test_data <- balanced_data[-trainIndex, ]

# Check for missing columns in train_data compared to the original data
missing_columns <- setdiff(names(data), names(train_data))
if (length(missing_columns) > 0) {
  cat("Missing columns in train_data:", missing_columns, "\n")
  
  # Add the missing columns back to the training data (fill with zeros for non-existent levels)
  for (col in missing_columns) {
    train_data[[col]] <- 0
  }
} else {
  cat("No missing columns in train_data.\n")
}

# Check if 'work_type_Self-employed' exists in train_data, if not, add it with zeros
# if (!"work_type_Self-employed" %in% names(train_data)) {
#  train_data$work_type_Self-employed <- 0
#  cat("'work_type_Self-employed' added to train_data with all values set to 0.\n")
# }

# 2. Ensure factor levels consistency between train_data and test_data
factor_columns <- sapply(train_data, is.factor)
levels_info <- lapply(train_data[, factor_columns], levels)
cat("Factor levels in training data:\n")
#print(levels_info)

# Add missing factor levels from test_data to train_data and vice versa
train_data <- train_data %>%
  mutate(across(where(is.factor), ~ factor(., levels = unique(c(levels(.), levels(test_data[[cur_column()]]))))))

test_data <- test_data %>%
  mutate(across(where(is.factor), ~ factor(., levels = unique(c(levels(.), levels(train_data[[cur_column()]]))))))

###Train Weighted Random Forest Model
# Assign higher weight to the minority class (stroke) to balance the model.
set.seed(123)
rf_weighted <- randomForest(as.factor(stroke) ~ ., 
                            data = train_data, 
                            mtry = 12,   # Using the best 'mtry' 
                            ntree = 500, 
                            importance = TRUE, 
                            classwt = c(0.5, 2))  # Giving more weight to the stroke class

# Print the summary of the weighted Random Forest model
print(rf_weighted)

# Evaluate the weighted model on the test data
predictions_weighted <- predict(rf_weighted, newdata = test_data)

# Ensure the factor levels for predictions and test_data match
test_data$stroke <- factor(test_data$stroke, levels = c("0", "1"))
predictions_weighted <- factor(predictions_weighted, levels = c("0", "1"))

# Confusion Matrix for the Weighted Random Forest model
conf_matrix_weighted <- confusionMatrix(predictions_weighted, test_data$stroke)

# Print the confusion matrix to see accuracy, sensitivity, specificity, etc.
print(conf_matrix_weighted)

# Feature Importance for the Weighted Model
varImpPlot(rf_weighted)

### ROC Curve and AUC for the Weighted Model ###
roc_curve_weighted <- roc(as.numeric(test_data$stroke), as.numeric(predictions_weighted))
auc_value_weighted <- auc(roc_curve_weighted)
print(paste("AUC for Weighted Random Forest: ", auc_value_weighted))
plot(roc_curve_weighted, main = "ROC Curve for Weighted Stroke Prediction")

###XGboost (First try)
#install.packages("xgboost")
library(xgboost)

# Convert the training data to DMatrix format
train_matrix <- xgb.DMatrix(data = as.matrix(train_data[, -which(names(train_data) == "stroke")]),
                            label = as.numeric(train_data$stroke) - 1)

# Train the XGBoost model (First try)
set.seed(123)
xgb_model <- xgboost(data = train_matrix, 
                     max_depth = 6, 
                     nrounds = 100, 
                     objective = "binary:logistic")

# Make predictions on the test data
test_matrix <- xgb.DMatrix(data = as.matrix(test_data[, -which(names(test_data) == "stroke")]))
xgb_predictions <- predict(xgb_model, test_matrix)

# Convert predictions to binary values
xgb_predictions <- ifelse(xgb_predictions > 0.5, 1, 0)

# Evaluate the performance
conf_matrix_xgb <- confusionMatrix(factor(xgb_predictions, levels = c(0, 1)), test_data$stroke)
print(conf_matrix_xgb)


############################################################################################################
library(caret)
library(pROC)  
library(PRROC)  
library(e1071)  
#install.packages("PRROC")
##Hyperparameter Tuning for Random Forest using caret
set.seed(123)

# Define tuning grid for Random Forest
rf_grid <- expand.grid(mtry = c(2, 5, 7, 12))

# Control parameters for cross-validation (5-fold cross-validation)
control <- trainControl(method = "cv", number = 5, verboseIter = TRUE, 
                        classProbs = TRUE, summaryFunction = twoClassSummary)

# Ensure 'stroke' is a factor with valid R variable names as levels
train_data$stroke <- factor(train_data$stroke, levels = c(0, 1), labels = c("no_stroke", "stroke"))
test_data$stroke <- factor(test_data$stroke, levels = c(0, 1), labels = c("no_stroke", "stroke"))

#run the Random Forest tuning
rf_tuned <- train(as.factor(stroke) ~ ., 
                  data = train_data, 
                  method = "rf",
                  metric = "ROC",   # Optimize based on ROC
                  tuneGrid = rf_grid,
                  trControl = control)

# Print the best Random Forest model after tuning
print(rf_tuned)
print(rf_tuned$bestTune)

##Hyperparameter Tuning for XGBoost using caret
set.seed(123)

#Define tuning grid for XGBoost
xgb_grid <- expand.grid(nrounds = 100, 
                        max_depth = c(4, 6, 8), 
                        eta = c(0.01, 0.1, 0.3), 
                        gamma = 0, 
                        colsample_bytree = 0.8, 
                        min_child_weight = 1, 
                        subsample = 0.8)

#Train the XGBoost model with cross-validation and tune hyperparameters
xgb_tuned <- train(as.factor(stroke) ~ ., 
                   data = train_data, 
                   method = "xgbTree", 
                   metric = "ROC",  # Optimize based on ROC
                   tuneGrid = xgb_grid,
                   trControl = control)

# Print the best XGBoost model after tuning
print(xgb_tuned)
print(xgb_tuned$bestTune)

#Evaluate the Tuned Models on Test Data
# Random Forest predictions on the test set
rf_tuned_predictions <- predict(rf_tuned, test_data)
rf_tuned_probs <- predict(rf_tuned, test_data, type = "prob")

# XGBoost predictions on the test set
xgb_tuned_predictions <- predict(xgb_tuned, test_data)
xgb_tuned_probs <- predict(xgb_tuned, test_data, type = "prob")

# Confusion matrix for both models
rf_conf_matrix <- confusionMatrix(rf_tuned_predictions, test_data$stroke)
xgb_conf_matrix <- confusionMatrix(xgb_tuned_predictions, test_data$stroke)

#confusion matrix and accuracy for Random Forest
print("Random Forest Confusion Matrix:")
print(rf_conf_matrix)

#confusion matrix and accuracy for XGBoost
print("XGBoost Confusion Matrix:")
print(xgb_conf_matrix)

# AUC-ROC for Random Forest
rf_roc <- roc(as.numeric(test_data$stroke), as.numeric(rf_tuned_probs[, 2]))
print(paste("Random Forest AUC-ROC:", auc(rf_roc)))

# AUC-ROC for XGBoost
xgb_roc <- roc(as.numeric(test_data$stroke), as.numeric(xgb_tuned_probs[, 2]))
print(paste("XGBoost AUC-ROC:", auc(xgb_roc)))

# Precision-Recall AUC for Random Forest
rf_pr <- pr.curve(scores.class0 = rf_tuned_probs[, 2], 
                  weights.class0 = as.numeric(test_data$stroke) - 1, 
                  curve = TRUE)
plot(rf_pr, main = "Precision-Recall AUC for Random Forest")
print(paste("Random Forest Precision-Recall AUC:", rf_pr$auc.integral))

# Precision-Recall AUC for XGBoost
xgb_pr <- pr.curve(scores.class0 = xgb_tuned_probs[, 2], 
                   weights.class0 = as.numeric(test_data$stroke) - 1, 
                   curve = TRUE)
plot(xgb_pr, main = "Precision-Recall AUC for XGBoost")
print(paste("XGBoost Precision-Recall AUC:", xgb_pr$auc.integral))

######
library(caret)

# Calculate Precision and Recall for Random Forest
precision_rf <- posPredValue(rf_tuned_predictions, test_data$stroke, positive = "stroke")
recall_rf <- sensitivity(rf_tuned_predictions, test_data$stroke, positive = "stroke")

# Print Precision and Recall for Random Forest
print(paste("Random Forest Precision:", precision_rf))
print(paste("Random Forest Recall:", recall_rf))

# Calculate Precision and Recall for XGBoost
precision_xgb <- posPredValue(xgb_tuned_predictions, test_data$stroke, positive = "stroke")
recall_xgb <- sensitivity(xgb_tuned_predictions, test_data$stroke, positive = "stroke")

# Print Precision and Recall for XGBoost
print(paste("XGBoost Precision:", precision_xgb))
print(paste("XGBoost Recall:", recall_xgb))

# Calculate F1-Score for Random Forest
rf_f1 <- 2 * ((precision_rf * recall_rf) / (precision_rf + recall_rf))
print(paste("Random Forest F1-Score:", rf_f1))

# Calculate F1-Score for XGBoost
xgb_f1 <- 2 * ((precision_xgb * recall_xgb) / (precision_xgb + recall_xgb))
print(paste("XGBoost F1-Score:", xgb_f1))

#libraries
library(caret)
library(randomForest)
library(xgboost)
library(smotefamily)
# Feature importance for Random Forest
varImpPlot(rf_model$finalModel)
# Feature importance for XGBoost
xgb_importance <- xgb.importance(model = xgb_model)
xgb.plot.importance(xgb_importance)

# Try to train the logistic regression model
logistic_model <- glm(stroke ~ avg_glucose_level_High + avg_glucose_level_Extremely.High + age + hypertension + heart_disease, 
                      data = balanced_data, 
                      family = binomial)

#summary of the logistic regression model to interpret the odds ratios
summary(logistic_model)
#interpretation of model coefficients
exp(coef(logistic_model)) 

