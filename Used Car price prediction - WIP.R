# Using advanced regression techniques to perform analysis on used car dataset

# Importing libraries
library(plyr)
library(dplyr)
library(caret)
library(glmnet)
library(ggplot2)
set.seed(1234)
# Importing data sets
X_train <- read.csv2('D:/LiU/Courses/Machine Learning - 732A99/Labs/Lab 2 Block 2/X_train.csv', sep = ',')
X_test <- read.csv('D:/LiU/Courses/Machine Learning - 732A99/Labs/Lab 2 Block 2/X_test.csv', sep = ',')
y_train <- read.csv('D:/LiU/Courses/Machine Learning - 732A99/Labs/Lab 2 Block 2/y_train.csv', sep = ',')
y_test <- read.csv('D:/LiU/Courses/Machine Learning - 732A99/Labs/Lab 2 Block 2/y_test.csv', sep = ',')


train_full <- merge(X_train, y_train, by = 'carID')
test_full <- merge(X_test, y_test, by = 'carID')

full_data <- rbind(train_full, test_full)

full_data <- select(full_data, -carID)

full_data <- mutate_at(full_data, c(7,8,9), as.double)

full_data <- full_data %>% mutate_if(is.character, ~as.factor(.))

full_data <- mutate_at(full_data, c(3), as.factor)

id <- sample(nrow(full_data), floor(nrow(full_data)*0.7))

train <- full_data[id,]
test <- full_data[-id,]

# Price of the used cars seems to be positively skewed and I think a simple log transformation should balance the distribution.
train$price <- log(train$price)
test$price <- log(test$price)

# Also some of the predictors in our training data are skewed, also it's in different scale, so we can apply the same technique there

train$mileage <- log(train$mileage)
test$mileage <- log(test$mileage)

test_response <- as.numeric(test[, 'price'])

test <- select(test, -price)

# Now let's encode the categorical columns in our data to numerical values

#for (col in c(2,3,4,5,7)) {
  
 # X_train[,col] <- as.numeric(mapvalues(x = X_train[,col], 
  #                            from = as.vector(unique(X_train[,col])),
   #                           to = 1:length(unique(X_train[,col]))))
  
#}

#for (col in c(2,3,4,5,7)) {
  
 # X_test[,col] <- as.numeric(mapvalues(x = X_test[,col], 
  #                                      from = as.vector(unique(X_test[,col])),
   #                                     to = 1:length(unique(X_test[,col]))))
  
#}

# It's time to merge our data for applying algorithms through caret package

#train_ <- merge(X_train, y_train, by='carID')
#test_ <- merge(X_test, y_test, by = 'carID')

# And now we drop the ID column as it doesn't give us any information on explaining the price of the used cars

#train_ <- select(train_, -carID)
#test_ <- select(test_, -carID)

#train_ <- mutate_at(train_, c(7,8,9), as.numeric)
#test_ <- mutate_at(test_, c(7,8,9), as.numeric)

#dummytrain <- dummyVars('~.',  train_)
#train_ <- data.frame(predict(dummytrain, train_))
#dummytest <- dummyVars('~.',  test_)
#test_ <- data.frame(predict(dummytrain, test_))

#train_ <- select(train_, -model.M6)




r_sqaured <- function(pred){
  rss <- sum((pred - test_response) ^ 2)
  tss <- sum((test_response - mean(test_response)) ^ 2)
  return(round(1-rss/tss, 3))
}

# Setting the resampling techniques using caret's trainControl function
fit_control <- trainControl(method = 'repeatedcv',
                            number = 10,
                            repeats = 4)

# Building simple linear regression model using caret's train function
simplelrfit <- train(price~.,
                     data = train,
                     method = 'lm',
                     trControl = fit_control)
predsimplelrfit <- predict.train(simplelrfit, 
                                 newdata = test)

rsq_simplelrfit <- r_sqaured(pred = predsimplelrfit)

# Now we try ridge regression to train our model and see how it works

ridgefit <- train(price~.,
                     data = train,
                     method = 'glmnet',
                     trControl = fit_control,
                     tuneGrid = expand.grid(data.frame(alpha = 0,
                                                       lambda = seq(0.0001, 0.1,0.001))))

# Optimal lambda chosen by the model is 0.0461, let's train using optimal lambda and measure our metric

optridgefit <- train(price~.,
                  data = train,
                  method = 'glmnet',
                  trControl = fit_control,
                  tuneGrid = expand.grid(data.frame(alpha = 0,
                                                    lambda = 0.0381)))

predridgefit <- predict.train(ridgefit, newdata = test)

rsq_ridgefit <- r_sqaured(predridgefit)


# Now we try Lasso regression to train our model and see how it works 

lassofit <- train(price~.,
                  data = train,
                  method = 'glmnet',
                  trControl = fit_control,
                  tuneGrid = expand.grid(data.frame(alpha = 1,
                                                    lambda = seq(0.00001, 0.01,0.0001))))

# Optimal lambda was given by the model is 0.00031, let's train our model using optimal lambda and measure our model

optlassofit <- train(price~.,
                  data = train,
                  method = 'glmnet',
                  trControl = fit_control,
                  tuneGrid = expand.grid(data.frame(alpha = 1,
                                                    lambda = 0.00031)))

predlassofit <- predict.train(lassofit, newdata = test)

rsq_lassofit <- r_sqaured(predlassofit)


# Now we are going to try decision trees on our dataset and measure the performance of our model

dtrpartfit <- train(price~.,
               data = train,
               method = 'rpart',
               trControl = fit_control,
               tuneLength =40)

# Optimal cp value chosen was cp = 0.0013 

preddtfit <- predict.train(dtrpartfit, newdata = test)

rsq_dtrpartfit <- r_sqaured(preddtfit)

# Resulting R-squared value for decison tree is 0.875


# Now we are going to try random forests on our dataset and measure the performance of our model.

library(randomForest)

set.seed(123)
#ranfofit <- train(price~.,
#                  data =  train,
#                  method='rf',
#                 trControl = fit_control,
#                  tuneGrid = expand.grid(data.frame(mtry=seq(1,9,1))))


hyper_grid <- expand.grid(mtry= seq(1,9,1),
                          node_size = seq(10, 16, 2),
                          sampe_size = c(0.55,0.632,0.70,0.80),
                          OOB_RMSE = 0)



for (i in 1:nrow(hyper_grid)) {
  ranger_rf <- ranger(price~., 
                      data = train, num.trees = 500, 
                      mtry = hyper_grid$mtry[i], 
                      min.node.size = hyper_grid$node_size[i], 
                      sample.fraction = hyper_grid$sampe_size[i],
                      seed = 123)
  
hyper_grid$OOB_RMSE[i] <- sqrt(ranger_rf$prediction.error)
  }



#hyper_grid2 <- expand.grid(mtry= seq(1,9,1),
#                           node_size = seq(1, 9, 2),
#                           sampe_size = c(0.55,0.632,0.70,0.80),
#                           OOB_RMSE = 0)


#for (i in 1:nrow(hyper_grid2)) {
#  ranger_rf2 <- ranger(price~., 
#                       data = X_traintree, 
#                       num.trees = 400, 
#                       mtry = hyper_grid2$mtry[i], 
#                       min.node.size = hyper_grid2$node_size[i], 
#                       sample.fraction = hyper_grid2$sampe_size[i],
#                       seed = 123)
#  hyper_grid2$OOB_RMSE[i] <- sqrt(ranger_rf2$prediction.error)
#  }


# Merging and then reshuffling

















