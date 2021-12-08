# Trying the Ames dataset from Kaggle to understand how different 
# aspects of a house effect Sale prices

library(caret)
library(fastDummies)
library(ggplot2)
library(dplyr)



#Read the data 
house_data <- read.csv('D:/LiU/Courses/Machine Learning - 732A99/Labs/Lab 2 Block 2/train.csv')

# Analyzing the dataset
# Checking the percentage of missing values in each column and analyzing it

na_perc <- list()
for (i in colnames(house_data)) {
  na_perc <- c(na_perc, list(round(sum(is.na(house_data[,i])/nrow(house_data)),
                          3)*100))
}
names(na_perc) <- colnames(house_data)
# We have around 6 columns where the missing data is more than 10% 
# For columns where there is a chance to fill the missing values, we can
# do that by using appropriate methods as described in data description
# file.

na_perc[(which(unlist(na_perc) !=0))]

# Alley tells us what kind of alley is around the house and NA value indicates no alley
house_data[is.na(house_data$Alley), 'Alley'] <- 'none'

# Lot Frontage has around 17.7% missing values and according to data description file, NA means no alley access so we can put 0 in place of NA values.
house_data[is.na(house_data$LotFrontage), 'LotFrontage'] <- 0

# MasVnrType is the type of Masonary Veneer Type and NA value means there was no veneer and not that it contains NA values
house_data[is.na(house_data$MasVnrType), 'MasVnrType'] <- 'None'

# MasVnrArea refers to the installed veneer area in house and NA means that the total are is zero
house_data[is.na(house_data$MasVnrArea), 'MasVnrArea'] <- 0

# BsmtQual is the height of the basement and hence given categorical inputs where NA means no basement
house_data[is.na(house_data$BsmtQual), 'BsmtQual'] <- 'none'

# BsmtCond is the general condition of the basement and hence given categorical inputs where NA means no basement
house_data[is.na(house_data$BsmtCond), 'BsmtCond'] <- 'none'

# BsmtExposure is the walkout or garden level walls of the basement and hence given categorical inputs where NA means no basement
house_data[is.na(house_data$BsmtExposure), 'BsmtExposure'] <- 'none'

# BsmtFinType1 is the rating of finished area in basement and hence given categorical inputs where NA means no basement
house_data[is.na(house_data$BsmtFinType1), 'BsmtFinType1'] <- 'none'

# BsmtFinType2 is the rating of multiple finished area in basement and hence given categorical inputs where NA means no basement
house_data[is.na(house_data$BsmtFinType2), 'BsmtFinType2'] <- 'none'

# There is only one missing value for Electrical Systems and we can put that as the most frequent value 
house_data[1380, 'Electrical'] <- 'SBrkr'

# FireplaceQu is the quality of fireplace in house and NA means there was no fireplace which can be changed to none or no values
house_data[is.na(house_data$FireplaceQu), 'FireplaceQu'] <- 'none'

# GarageType is the garage location around the house and contain many categorical variables describing different location. NA means there is no garage.
house_data[is.na(house_data$GarageType), 'GarageType'] <- 'none'
 
# GarageYrBlt denotes the years at which the garage was built and we have no information about when the NA values were built, so we can fill them up by randomly choosing year values from the columns so as not to lose any information but it will increase the randomness in GarageYrBlt column, let's try it!
for (na in which(is.na(house_data$GarageYrBlt))) {
  house_data[na, 'GarageYrBlt'] <- sample(na.omit(unique(house_data$GarageYrBlt)) , 1) 
}
# And then we find if it is an important variable for price prediction and it shows that it relates to saleprice by 47%
cor(house_data$GarageYrBlt, house_data$SalePrice)

# GarageFinish tells us the finish quality
house_data[is.na(house_data$GarageFinish), 'GarageFinish'] <- 'none'

# GarageQual tells us the finished quality of garage in house
house_data[is.na(house_data$GarageQual), 'GarageQual'] <- 'none'

# GarageCond tells us the condition of garage in house
house_data[is.na(house_data$GarageCond), 'GarageCond'] <- 'none'

# PoolQC tells us the pool quality in house
house_data[is.na(house_data$PoolQC), 'PoolQC'] <- 'none'

# Fence tells us the fence quality around the house
house_data[is.na(house_data$Fence), 'Fence'] <- 'none'

# MiscFeature tells us more about miscellaneous features in house not covered by other categories
house_data[is.na(house_data$MiscFeature), 'MiscFeature'] <- 'none'

# Looking at the sale price we see that it is skewed towards right and we can improve it by taking log transform
ggplot(data = house_data) + geom_density(aes(x=SalePrice)) 

house_data$SalePrice <- log(house_data$SalePrice)
ggplot(data = house_data) + geom_histogram(aes(x=SalePrice), bins = 60)

# We have handled all the missing values from our precious data
anyNA(house_data) # FALSE

# Next, in order to use basic regression algorithms on our dataset, we need to encode categorical values into continuous values 
dummy_house_data <- dummy_cols(house_data)

# Now all the categorical features have been encoded to continuous values and we remove the character columns from our data set.
num_house_data_1 <- select_if(dummy_house_data, is.numeric)

# Since we have a lot of predictors, it is prefectly possible to have collinearity or zero variance predictors in our data set and we need to get rid of them for better model generalization.
# we check zero variance predictors using 'nearZerovar' function in caret
num_house_data_1 <- num_house_data_1[,-nearZeroVar(num_house_data_1)]
# This leaves us with only 128 predictors that have some variance which could be useful for explaining variance in response variable.

# After this, we check the highly correlated variables in our model and this can be done by using 'findCorrelation' function in caret and must be removed to gain useful models. We will set the threshold to 0.85 for correlation.
num_house_data_1 <- num_house_data_1[,-findCorrelation(cor(num_house_data_1), cutoff = 0.9)]
# This leaves us with 111 predictors that seem to be useful for our model
set.seed(12)
# We have to split our data into train and test sets and perform scaling on them
id <- sample(1:nrow(num_house_data_1), floor(nrow(num_house_data_1)*0.7))
train <- num_house_data_1[id,-c(1,30)]
test <- num_house_data_1[-id, -c(1,30)]

# Split into train set and then apply preprocess function to it which will apply the same processing to test set

train <- as.data.frame(scale(train))
test <- as.data.frame(scale( test))
train <- cbind(train, SalePrice = house_data$SalePrice[id])
test <- cbind(test, SalePrice = house_data$SalePrice[-id])

# First we are going to try simple linear regression and consider it as our base model from where we have to improve our model
fit_control <- trainControl(method = 'repeatedcv',
                            number = 10,
                            repeats = 4)

simplelrfit <- train(SalePrice ~. ,
                     data= train,
                     method='lm',
                     trControl = fit_control,
                     tuneGrid  = expand.grid(intercept = T))
simple_prediction <- predict.train(simplelrfit, test[,-ncol(test)])
simpleLRrmse <- RMSE(simple_prediction, test[,ncol(test)])


# Running this model removed several predictors from the model as they did not provide any useful variance. And we get the test RMSE of 0.166 on the test set against the train RMSE of 1.312. We can consider this as our base model and apply different models and compare the results.


#######################################RIDGE REGRESSION#################################
# Let's apply ridge regression into out training set now and check how it faired against simple linear regression.
ridgefit <- train(SalePrice~.,
                  data=train,
                  method = 'glmnet',
                  trControl = fit_control,
                  tuneGrid = expand.grid(data.frame(
                    alpha = rep(0, length(seq(0.1,10,1))),
                    lambda=seq(0.0001, 0.1, 0.001))))
# Here we get the optimal lambda value of lambda = 0.0541 using 4 repeated 10 cross validation which gives us the lowest rmse.
# And now we train the ridge model again with optimal alpha and lambda value and check our model on testing data.
optridgefit <- train(SalePrice~.,
                     data=train,
                     method='glmnet',
                     tuneGrid=expand.grid(data.frame(
                       alpha = 0,
                       lambda=0.0541)))
ridge_prediction <- predict.train(optridgefit, test[,-ncol(test)])
ridgermse <- RMSE(ridge_prediction, test[,ncol(test)])


#######################################LASSO REGRESSION#################################
# Let's apply lasso regression into out training set now and check how it faired against simple linear regression.
lassofit <- train(SalePrice~.,
                  data=train,
                  method = 'glmnet',
                  trControl = fit_control,
                  tuneGrid = expand.grid(data.frame(
                    alpha = rep(1, length(seq(0.1,10,1))),
                    lambda=seq(0.0001, 0.01, 0.001))))

optlassofit <- train(SalePrice~.,
                     data=train,
                     method='glmnet',
                     tuneGrid=expand.grid(data.frame(
                       alpha = 1,
                       lambda=0.001)))
lasso_prediction <- predict.train(optlassofit, test[,-ncol(test)])
lassormse <- RMSE(lasso_prediction, test[,ncol(test)])
#######################################LASSO REGRESSION#################################
# Let's apply Elastic Net regression into out training set now and check how it faired against simple linear regression.

elnetfit <- train(SalePrice~., 
                  data=train, 
                  method='glmnet', 
                  tuneLength = 5)

optelnetfit <- train(SalePrice~.,
                     data=train,
                     method='glmnet',
                     tuneGrid=expand.grid(data.frame(
                       alpha = 0.1,
                       lambda=0.004)))

elnet_prediction <- predict.train(optelnetfit, test[,-ncol(test)])
elnetrmse <- RMSE(elnet_prediction, test[,ncol(test)])




# 















