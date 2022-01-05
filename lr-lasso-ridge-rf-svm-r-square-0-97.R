# Using advanced regression techniques to perform analysis on used car dataset

# Importing libraries
library(plyr)
library(dplyr)
library(caret)
library(glmnet)
library(ggplot2)
library(ranger)
set.seed(123456)
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

id <- sample(nrow(full_data), floor(nrow(full_data)*0.75))


# Price of the used cars seems to be positively skewed and I think a simple log transformation should balance the distribution.
#full_data$price <- full_data$price^(1/4)
full_data$price <- log(full_data$price)

# Also some of the predictors in our training data are skewed, also it's in different scale, so we can apply the same technique there
full_data$mileage <- full_data$mileage^(1/4)

full_data$mpg <- log(full_data$mpg)

# We can also try some of the non-linear transformations and learn how it affects our model

full_data$tax <- sqrt(full_data$tax)

# Separate into training and test sets
train <- full_data[id,]
test <- full_data[-id,]

# Now we check if our partition has similar proportions in train and test data
train %>% group_by(brand) %>% summarise(n(), percent = n()/nrow(train)*100)
test %>% group_by(brand) %>% summarise(n(), percent = n()/nrow(test)*100)


# As we check the year column and see that some rows are present in test column but not present in training column which may cause problem as the model has never seen that data and we need to clean it before we measure our model.
train %>% group_by(year) %>% summarise(n())
test %>% group_by(year) %>% summarise(n())


# Rows where data is uneven in test data are: 161,1939,2692,3242,5144
which(train$year == 1970)
which(train$year == 2000)
which(train$year == 1997) 
train <- train[-c(161,1939,2692,3242,5144),]

`%!in%` <- Negate(`%in%`)
which(unique(train$model)  %!in% unique(test$model))
which(unique(train$fuelType)  %!in% unique(test$fuelType))

# There is one transmission which is not in test data
unique(train$transmission)  %!in% unique(test$transmission)
train <- train[-c(1159,1211),]


# Keeping y test aside 
test_response <- as.numeric(test[, 'price'])

# Keeping y test from test data frame
test <- select(test, -price)

#metric
# Defining R-Squared function for measuring performance of our predictive models.
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
                     method = 'lm')
predsimplelrfit <- predict.train(simplelrfit, 
                                 newdata = test)

rsq_simplelrfit <- r_sqaured(pred = predsimplelrfit)

# After looking at the summary of the linear regression, we see that tax column in our datset is not at all significant and so we drop it and actually it does not change any acuuracy value for any models which confirms this hypothesis.

pred_plot <- plot(x = test_response, 
                  y = predsimplelrfit, 
                  xlab = 'Actual Values',
                  ylab = 'Predicted Values',
                  main = 'Simple Linear Regression',
                  ylim = c(6,18),
                  abline(a=0,b=1, col = 'red'))

resid_simplelr <- plot(x = predsimplelrfit,
                       y = predsimplelrfit - test_response,
                       abline(0,0,col = 'red'),
                       xlab = 'Predicted Values',
                       ylab = 'Residuals',
                       main = 'Simple Linear Regression Residuals')


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
                  tuneGrid = expand.grid(data.frame(alpha = 1,
                                                    lambda = 0.00021)))

predlassofit <- predict.train(optlassofit, newdata = test)

rsq_lassofit <- r_sqaured(predlassofit)


# Now we are going to try decision trees on our dataset and measure the performance of our model

dtrpartfit <- train(price~.,
               data = train,
               method = 'rpart',
               trControl = fit_control,
               tuneLength =20)

# Optimal cp value chosen was cp = 0.0013 

preddtfit <- predict.train(dtrpartfit, newdata = test)

rsq_dtrpartfit <- r_sqaured(preddtfit)

# Resulting R-squared value for decison tree is 0.875


# Now we are going to try random forests on our dataset and measure the performance of our model.


set.seed(123)


hyper_grid <- expand.grid(mtry= seq(1,8,1),
                          node_size = seq(5, 15, 5),
                          sampe_size = c(0.55,0.632,0.70,0.80,1),
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

# Above ranger object gives us out optimal tree of mtry = 3, node size = 5, and sample size of 100% of train data.
 
# We use this optimal random forest tree to predict our house prices using test data

pred_ranger <- predict(ranger_rf, data = test)

rsq_ranforfit <- r_sqaured(pred_ranger$predictions)

pred_plot_ranfo <- plot(x = test_response, 
                  y = pred_ranger$predictions, 
                  xlab = 'Actual Values',
                  ylab = 'Predicted Values',
                  main = 'Random Forest (R-Squared - 96%)',
                  ylim = c(6,18),
                  abline(a=0,b=1, col = 'red'))

resid_ranfor <- plot(x = pred_ranger$predictions,
                       y = pred_ranger$predictions - test_response,
                       abline(0,0,col = 'red'),
                       xlab = 'Predicted Values',
                       ylab = 'Residuals',
                       main = 'Random Forest Residuals')

# As we can see, random forests has become the strongest Jedi of all the models present here and it is also robust to outliers getting us an R-Sqaured value of 0.96 for this dataset. 

gammas <- 2^(-3:3)
costs <- 2^(-3:3)
epsilon <- c(0.1,0.01,0.001)
library(e1071)

#svmtune <- tune.svm(price~.,  data = train, gamma = gammas, cost = costs,epsilon = epsilon)

# Above hyperparameter tuning of svm gives us the best values for gamma = 0.125, cost = 4, epsilon = 0.01 and then we train our optimal model on whole training data set.

svmoptimal <- svm(price~., 
                  data = train, 
                  scale = T, 
                  type = 'eps-regression', 
                  kernel = 'radial', 
                  gamma = 0.125, 
                  cost=4, 
                  epsilon = 0.01 )


predsvm <- predict(svmoptimal, newdata = test)

rsq_svm <- r_sqaured(predsvm)


brands<- train %>% 
  group_by(brand) %>% 
  summarise(length(brand))

ggplot(data = brands, aes(x = brand, y = `length(brand)`)) + 
  geom_bar( stat= 'identity', width=0.6, fill = '#9999FF', color='black') + 
  geom_text(aes(label = `length(brand)` ), vjust=1.5) +
  xlab('Brand') + 
  ylab('Number of cars') +
  theme(panel.grid.major = element_blank(), 
        panel.background = element_rect('white'),text = element_text(size = 20))


bybrand <- train %>% 
  group_by(brand, transmission) %>% 
  summarise(price = mean(price))


ggplot(data = bybrand, aes(x = brand, y = price, fill= transmission))+
  geom_bar(stat='identity', position = position_dodge2(), color = 'black') +
  xlab('Brand')+
  ylab('Average price of cars') +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect('white'),
        text = element_text(size = 20))
  


taxincur <-train %>% 
  group_by(fuelType) %>% 
  summarise(tax = tax)

ggplot(data = taxincur, aes(x = fuelType, y = tax))+
  geom_boxplot() +
  xlab('Brand')+
  ylab('Average price of cars') +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect('white'),
        text = element_text(size = 20))























































