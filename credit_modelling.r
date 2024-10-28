# Load necessary libraries
library(caret)          # For data splitting, training, and evaluation
library(rpart)          # For decision tree model
library(rpart.plot)     # For plotting decision trees
library(pROC)           # For evaluating the model's performance
library(dplyr)          # For data manipulation

# Step 1: Load the dataset

credit_data <- read.csv('credit_data.csv')

# Step 2: Explore the dataset
str(credit_data)
summary(credit_data)

# Step 3: Preprocess the data
# Handling missing values (if any)
credit_data <- na.omit(credit_data)

# Converting categorical variables to factors (if necessary)
credit_data$Default <- factor(credit_data$Default, levels = c(0, 1))

# Step 4: Split the data into training and test sets
set.seed(123) # For reproducibility
train_index <- createDataPartition(credit_data$Default, p = 0.7, list = FALSE)
train_data <- credit_data[train_index, ]
test_data <- credit_data[-train_index, ]

# Step 5: Logistic Regression Model
logistic_model <- glm(Default ~ ., data = train_data, family = binomial)

# Summarize the logistic regression model
summary(logistic_model)

# Predict on the test set
logistic_predictions <- predict(logistic_model, test_data, type = "response")

# Convert probabilities to class labels (threshold = 0.5)
logistic_pred_class <- ifelse(logistic_predictions > 0.5, 1, 0)

# Evaluate the logistic regression model
confusionMatrix(factor(logistic_pred_class), test_data$Default)
roc_curve_logistic <- roc(test_data$Default, logistic_predictions)
plot(roc_curve_logistic, main = "ROC Curve - Logistic Regression", col = "blue")

# Step 6: Decision Tree Model
decision_tree_model <- rpart(Default ~ ., data = train_data, method = "class")

# Plot the decision tree
rpart.plot(decision_tree_model, main = "Decision Tree for Credit Risk Assessment")

# Predict on the test set
tree_predictions <- predict(decision_tree_model, test_data, type = "class")

# Evaluate the decision tree model
confusionMatrix(tree_predictions, test_data$Default)

# Display results for Logistic Regression
cat("\nLogistic Regression Evaluation:\n")
print(confusionMatrix(factor(logistic_pred_class), test_data$Default))
cat("\nAUC for Logistic Regression:", auc(roc_curve_logistic), "\n")

# Display results for Decision Tree
cat("\nDecision Tree Evaluation:\n")
print(confusionMatrix(tree_predictions, test_data$Default))

# Save the models
saveRDS(logistic_model, file = "logistic_model.rds")
saveRDS(decision_tree_model, file = "decision_tree_model.rds")
