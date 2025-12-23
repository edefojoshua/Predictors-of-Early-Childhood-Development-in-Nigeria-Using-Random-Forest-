
 usethis::use_gpl3_license("Joshua W. Edefo")

  # ------------------------------------------------------------
# License: GPL-3.0
# Copyright (C) 2025 Joshua W. Edefo
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# ------------------------------------------------------------

Introduction

#This study applies machine-learning methods to identify predictors
#of early childhood development outcomes in Nigeria using nationally representative survey data.

library(haven)        # read_dta()
library(data.table)  # fast data manipulation, :=, .SD
library(fst)          # write_fst(), read_fst()
library(caret)        # train(), createDataPartition(), confusionMatrix()
library(ranger)       # random forest engine
library(ggplot2)      # plots
library(pROC)         # ROC / AUC
library(rcompanion)  # cramerV()
library(DescTools)   # cramerV()
library(usethis)    # project-setup package


set.seed(123)


# Import Stata file
library(haven)

data_path <- "C:/Users/edefoj/Documents/Per/MICS/Nigeria_2021"

data <- read_dta(file.path(data_path, "children_pro.dta"))

setDT(data) 

# Inspect data
head(data)
class(data)

# Select specified columns
data <- data[, .(
  agegrp1_cat, HL4, agegrp, HH6, zone, windex5MICS,
  cdisability, melevel_comb, EC1, ethnicity,
  UCD2G, UCD2K, UCF18, CA1
)]

# Save and reload data


if (!file.exists(data_path)) {
  write_fst(data, data_path, compress = 50)
}

data <- read_fst(data_path, as.data.table = TRUE)


write_fst(data, "./data.fst", compress = 50)
data <- read_fst("./data.fst", as.data.table = TRUE)

# Inspect structure
str(data)
nrow(data)
names(data)
head(data)
class(data)

# Modeling libraries
library(caret)
library(ranger)

# Target distribution
unique(data$agegrp1_cat)
prop.table(table(data$agegrp1_cat))

# Missing values
any(is.na(data))

############################################
# Data cleaning for ML
############################################

ordinal_vars <- c("agegrp", "windex5MICS", "melevel_comb", "EC1")
nominal_vars <- setdiff(names(data), ordinal_vars)

# Convert ordinal predictors
data[, (ordinal_vars) := lapply(.SD, function(x) factor(x, ordered = TRUE)),
     .SDcols = ordinal_vars]

# Convert remaining numeric predictors to factors
data[, names(data) := lapply(.SD, function(x)
  if (is.numeric(x) && !is.ordered(x)) as.factor(x) else x)]

############################################
# Class balancing
############################################

tbl <- table(data$agegrp1_cat)
w <- 1 / tbl
w <- w / min(w)

############################################
# Feature exploration
############################################

library(ggplot2)
ggplot(data, aes(x = ethnicity)) + geom_bar()
ggplot(data, aes(x = agegrp)) + geom_bar()

############################################
# Association checks
############################################

library(rcompanion)

cat_cols <- names(Filter(is.factor, data))

cramer_matrix <- matrix(
  NA, length(cat_cols), length(cat_cols),
  dimnames = list(cat_cols, cat_cols)
)

for (i in cat_cols) {
  for (j in cat_cols) {
    cramer_matrix[i, j] <- cramerV(
      table(data[[i]], data[[j]]),
      bias.correct = TRUE
    )
  }
}

############################################
# Leakage check
############################################

target_var <- "agegrp1_cat"
predictors_list <- setdiff(names(data), target_var)

library(DescTools)
cramerV_target <- sapply(
  data[, ..predictors_list],
  function(x) cramerV(x, data[[target_var]])
)

sort(cramerV_target, decreasing = TRUE)

############################################
# Train/test split
############################################

set.seed(123)

test_index <- createDataPartition(data$agegrp1_cat, p = 0.2, list = FALSE)
test_data  <- data[test_index]
train_data <- data[-test_index]

train_weights <- w[train_data$agegrp1_cat]

############################################
# Fix factor levels
############################################

train_data$agegrp1_cat <- factor(
  train_data$agegrp1_cat,
  levels = c("0", "1"),
  labels = c("class0", "class1")
)

test_data$agegrp1_cat <- factor(
  test_data$agegrp1_cat,
  levels = c("0", "1"),
  labels = c("class0", "class1")
)

############################################
# Cross-validation
############################################

tc <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = "final",
  classProbs = TRUE,
  allowParallel = TRUE
)

############################################
# Default ranger model
############################################

default_model <- train(
  agegrp1_cat ~ .,
  data = train_data,
  method = "ranger",
  trControl = tc,
  tuneGrid = data.frame(
    mtry = floor(sqrt(ncol(train_data) - 1)),
    splitrule = "gini",
    min.node.size = 1
  ),
  weights = train_weights,
  importance = "impurity"
)

############################################
# Tuned ranger model
############################################

tuned_grid <- expand.grid(
  mtry = c(3, 5, 7),
  splitrule = c("gini", "extratrees"),
  min.node.size = c(1, 5, 10)
)

tuned_model <- train(
  agegrp1_cat ~ .,
  data = train_data,
  method = "ranger",
  trControl = tc,
  tuneGrid = tuned_grid,
  weights = train_weights,
  importance = "impurity"
)

############################################
# Model selection
############################################

best_model <- if (max(tuned_model$results$Accuracy) >
                  max(default_model$results$Accuracy)) {
  tuned_model
} else {
  default_model
}

############################################
# Test performance
############################################

test_pred <- predict(best_model, test_data)
mean(test_pred == test_data$agegrp1_cat)

confusionMatrix(
  test_pred,
  test_data$agegrp1_cat,
  positive = "class1"
)

############################################
# ROC / AUC
############################################

library(pROC)
test_prob <- predict(best_model, test_data, type = "prob")[, "class1"]
roc_obj <- roc(test_data$agegrp1_cat, test_prob)
auc(roc_obj)
plot(roc_obj)

############################################
# Feature importance
############################################

imp <- best_model$finalModel$variable.importance
imp_df <- data.frame(
  Feature = names(imp),
  Importance = imp
)

imp_df <- imp_df[order(imp_df$Importance, decreasing = TRUE), ]

library(dplyr)
imp_df2 <- aggregate(
  Importance ~ gsub("\\..*|\\^.*", "", Feature),
  imp_df,
  sum
)

ggplot(imp_df2, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col() +
  coord_flip()

############################################
# Partial Dependence Plots
############################################

library(pdp)
pdp_age <- partial(best_model, pred.var = "agegrp", train = train_data)
plot(pdp_age)


setwd("C:/Users/edefoj/Documents/Per/MICS/Nigeria _2021"
)

#install.packages("haven")

# Load package
library(haven)
library(data.table)
library(fst)


# Import Stata file
#data <- read_dta("children_pro.dta")

# View the data
head(data)
class(data)
#setDT(data)


# select the specified columns
data <- data[, .(agegrp1_cat, HL4, agegrp, HH6, zone, windex5MICS,
                 cdisability, melevel_comb, EC1, ethnicity,
                 UCD2G, UCD2K, UCF18, CA1)]

write_fst(data, "./data.fst", compress = 50)



data <- read_fst("./data.fst", as.data.table = TRUE)
str(data)
nrow(data)
names(data)
head(data)
class(data)
library(caret)
library(ranger)

unique(data$agegrp1_cat)

prop.table(table(data$agegrp1_cat))


## Check for missing values
any(is.na(data))


### Cleaning before Machine learning
##Convert categorical predictors to factors (Classification task)
names(data)

# Specify ordinal and nominal variables
ordinal_vars <- c("agegrp", "windex5MICS", "melevel_comb", "EC1")

nominal_vars <- setdiff(names(data), ordinal_vars)

# 1. Convert ordinal variables to ordered factor
data[, (ordinal_vars) := lapply(.SD, function(x) factor(x, ordered = TRUE)),
     .SDcols = ordinal_vars]

data[ , names(data) := lapply(.SD, function(x)
  if(is.numeric(x) && !is.ordered(x)) as.factor(x) else x)]

names(data)[sapply(data, is.ordered)]
names(data)[sapply(data, is.factor)]


#Balancing Class in Target
str(data)
## Compute weights
tbl <- table(data$agegrp1_cat)
w <- 1 / tbl
w <- w / min(w)
w

# No need for Outliers
#Feature distribution
lapply(data, function(x) if(is.factor(x) && !is.ordered(x)) table(x))
library(ggplot2)
ggplot(data, aes(x = ethnicity)) + geom_bar()

lapply(data, function(x) if(is.ordered(x)) table(x))
ggplot(data, aes(x = agegrp)) + geom_bar()

#None of these apply,Because your data is categories:No normality, No skewness correction, No outlier trimming / winsorizing, No transformations (log, sqrt, z-score)

#perfectly correlated variables
#install.packages("rcompanion")
library(rcompanion)

cat_cols <- names(Filter(is.factor, data))
cat_cols

cramer_matrix <- matrix(NA,
                        nrow = length(cat_cols),
                        ncol = length(cat_cols),
                        dimnames = list(cat_cols, cat_cols))

for(i in cat_cols){
  for(j in cat_cols){
    cramer_matrix[i, j] <- cramerV(table(data[[i]], data[[j]]), 
                                   bias.correct = TRUE)
  }
}

cramer_matrix

#Correct data types

#Check for data leakage
target_var <- "agegrp1_cat"   # or your real target variable

predictors_list <- setdiff(names(data), target_var)

predictors_list
library(DescTools)

cramerV_target <- sapply(
  data[, ..predictors_list],
  function(x) cramerV(x, data[[target_var]])
)

cramerV_target <- sort(cramerV_target, decreasing = TRUE)
print(cramerV_target)


ranger::ranger

###Cross validayion


set.seed(123)

ranger::ranger

############################################
# Train/Test split (20% test)
############################################

test_index <- createDataPartition(data$agegrp1_cat, p = 0.2, list = FALSE)
test_data  <- data[test_index, ]
train_data <- data[-test_index, ]
train_weights <- w[train_data$agegrp1_cat]

############################################
# 1. Fix factor levels (IMPORTANT)
############################################
# Convert outcome to valid R variable names
train_data$agegrp1_cat <- factor(train_data$agegrp1_cat,
                                 levels = c("0", "1"),
                                 labels = c("class0", "class1"))

test_data$agegrp1_cat <- factor(test_data$agegrp1_cat,
                                levels = c("0", "1"),
                                labels = c("class0", "class1"))

############################################
# 10-fold CV control
############################################
tc <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = "final",
  classProbs = TRUE,       # <-- THIS IS THE IMPORTANT PART
  allowParallel = TRUE
)

############################################
# DEFAULT ranger (no tuning)
############################################
default_model <- train(
  agegrp1_cat ~ .,
  data = train_data,
  method = "ranger",
  trControl = tc,
  tuneGrid = data.frame(
    mtry = floor(sqrt(ncol(train_data) - 1)),
    splitrule = "gini",
    min.node.size = 1
  ),
  weights = train_weights,
  importance = "impurity"
)

############################################
# TUNED ranger (hyperparameter tuning)
############################################
tuned_grid <- expand.grid(
  mtry = c(3, 5, 7),
  splitrule = c("gini", "extratrees"),
  min.node.size = c(1, 5, 10)
)

tuned_model <- train(
  agegrp1_cat ~ .,
  data = train_data,
  method = "ranger",
  trControl = tc,
  tuneGrid = tuned_grid,
  weights = train_weights,
  importance = "impurity"
)

############################################
# Compare CV performance
############################################
default_model
tuned_model

cat("Default CV Accuracy:", max(default_model$results$Accuracy), "\n")
cat("Tuned CV Accuracy:",   max(tuned_model$results$Accuracy), "\n")

############################################
# Select the better model
############################################
best_model <-
  if (max(tuned_model$results$Accuracy) > max(default_model$results$Accuracy)) {
    cat("Using TUNED model\n")
    tuned_model
  } else {
    cat("Using DEFAULT model\n")
    default_model
  }

############################################
# Test set performance
############################################
test_pred <- predict(best_model, test_data)
test_acc <- mean(test_pred == test_data$agegrp1_cat)

cat("Final TEST accuracy:", test_acc, "\n")


############################################
# Confusion Matrix
############################################

test_pred <- predict(best_model, test_data)

confusionMatrix(
  data = test_pred,
  reference = test_data$agegrp1_cat,
  positive = "class1"    # change if your event is coded differently
)

############################################
# ROC + AUC
############################################

library(pROC)
test_prob <- predict(best_model, test_data, type = "prob")[, "class1"]

roc_obj <- roc(test_data$agegrp1_cat, test_prob)
auc(roc_obj)
plot(roc_obj, main = paste("AUC =", round(auc(roc_obj), 3)))


############################################
# Feature importancs
############################################

imp <- varImp(best_model)$importance

imp<- best_model$finalModel$variable.importance
imp_df <- data.frame(Feature = names(imp), Importance = imp)

imp_df <- imp_df[order(imp_df$Importance, decreasing = TRUE), ]

print(imp_df)

library(dplyr)
imp_df2 <- aggregate(
  Importance ~ gsub("\\..*|\\^.*", "", Feature),
  imp_df,
  sum
)

imp_df2 <- imp_df2 %>%
  dplyr::rename(
    Feature = `gsub("\\\\..*|\\\\^.*", "", Feature)`
  )

# Replace variable prefixes with readable labels
imp_df2$Feature <- imp_df2$Feature %>%
  sub("^melevel_comb", "mother's education", .) %>%
  sub("^agegrp",        "age group", .) %>%
  sub("^EC1",           "ECD stimulation", .) %>%
  sub("^HH6",           "type of residence", .) %>%
  sub("^windex5MICS",   "wealth index", .) %>%
  sub("^zone",          "geopolitical zone", .) %>%
  sub("^HL4",           "sex", .) %>%
  sub("^UCF18",         "difficulty playing", .) %>%
  sub("^CA1",           "diarrhoea", .) %>%
  sub("^UCD2G",         "hit hard", .) %>%
  sub("^UCD2K",         "beaten hard", .)

imp_df2
ggplot(imp_df2, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col() +
  coord_flip() +
  labs(title = "Ranger Feature Importance", x = "", y = "Importance")


# Optional ggplot barplot
library(ggplot2)
ggplot(imp_df2, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col() +
  coord_flip() +
  labs(title = "Ranger Feature Importance", x = "", y = "Importance")

