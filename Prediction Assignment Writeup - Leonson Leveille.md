library(plyr)
library(dplyr)
library(caret)

rm(list = ls())
gc()

## Let's import the data and take a look at it.

data <- read.csv("pml-training.csv", stringsAsFactors = F, na.strings = c("NA", ""))
head(data)

## Seems that we do not need the first two variables and the cvtd_timestamp variable

data <- subset(data, select = -c(X, user_name, cvtd_timestamp))

## How many columns and rows do we have?

dim(data)

## We have 19622 rows and 157 variables. With that many variables, it may be a good choice to use PCA.

## Let's see if we have any NAs in this dataset.

sapply(data, function(x) sum(is.na(x)))

## Some of the variables have a lot of Nas. We are just going to delete these variables.

data <- data[!(colSums(is.na(data)) > 0)]

## We now have only 57 variables left. 

## What about the class distribution?

cbind(freq = table(data$class), percentage = prop.table(table(data$class))*100)

## The classe variable is not imbalanced.

## Let's transform the variable with the character class into integer

unique(data$new_window)

unique(data$classe)

data[data$new_window == "no", "new_window"] <- 1
data[data$new_window == "yes", "new_window"] <- 2
data$new_window <- as.integer(data$new_window)

data[data$classe == "A", "classe"] <- 1
data[data$classe == "B", "classe"] <- 2
data[data$classe == "C", "classe"] <- 3
data[data$classe == "D", "classe"] <- 4
data[data$classe == "E", "classe"] <- 5
data$classe <- as.integer(data$classe)

## Let's transform the above variables into factors.

data$new_window <- as.factor(data$new_window)
data$classe <- as.factor(data$classe)

## Let's create and compare some models

## First, i'm going to split my data. 80% for training and 20% for testing.
## To compare my model, I'm going to use a 3-fold Cross Validation.
## Let's use the metric Accuracy.

set.seed(21)
trainIndex <- createDataPartition(data$classe, p = 0.8, list = F)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]

trainControl <- trainControl(method = "cv", number = 3)
metric = "Accuracy"

svm_model <- train(classe ~ ., data = train, trControl = trainControl, metric = metric, method = "svmRadial")
gbm_model <- train(classe ~ ., data = train, trControl = trainControl, metric = metric, method = "gbm", verbose = F)
knn_model <- train(classe ~ ., data = train, trControl = trainControl, metric = metric, method = "knn")
results <- resamples(list(SVM = svm_model, GBM = gbm_model, KNN = knn_model))
summary(results)
dotplot(results)

## KNN is really not good. We are going to use GBM to predict. 

predictions <- predict(gbm_model, test)
confusionMatrix(predictions, test$classe)

## Now let's predict on pml-testing

pml_testing <- read.csv("pml-testing.csv", stringsAsFactors = F, na.strings = c("NA", ""))
pml_testing <- subset(pml_testing, select = -c(X, user_name, cvtd_timestamp))
pml_testing <- pml_testing[!(colSums(is.na(pml_testing)) > 0)]
pml_testing[pml_testing$new_window == "no", "new_window"] <- 1
pml_testing[pml_testing$new_window == "yes", "new_window"] <- 2
pml_testing$new_window <- as.integer(pml_testing$new_window)
pml_testing$new_window <- as.factor(pml_testing$new_window)
predictions <- predict(gbm_model, pml_testing)

predictions

## Here's my predictions: 
## B A B A A E D B A A B C B A E E A B B B 