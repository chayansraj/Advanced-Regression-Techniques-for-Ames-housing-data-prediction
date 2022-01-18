# Trying the Ames dataset from Kaggle to understand how different 
# aspects of a house effect Sale prices

library(caret)
library(fastDummies)
library(ggplot2)
library(dplyr)



#Read the data 
house_data <- read.csv('train.csv')

# Analyzing the dataset
# Checking the percentage of missing values in each column and analyzing it

na_perc <- list()
for (i in colnames(house_data)) {
  na_perc <- c(na_perc, list( round(sum(is.na(house_data[,i])/nrow(house_data)),
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

# I will put categorical features in one set and numerical features in one set to tidy and understand the relations better
# Reference - https://dplyr.tidyverse.org/

# There are some columns that are wrongly added as numeric but are in fact categorical, we need to figure them out and change it
house_data[,c(18,19,20,21,48,49,50,51,52,53,55,57,60,62,78)] <- lapply(house_data[,c(18,19,20,21,48,49,50,51,52,53,55,57,60,62,78)], as.factor)

house_data<-  house_data %>% mutate_if(is.character, ~as.factor(.))
categorical_features <- house_data %>% select(where(is.factor))
numerical_features <- house_data %>% select(where(is.numeric))
numerical_features <- numerical_features[,-1]
sales <- numerical_features[,ncol(numerical_features)]

attach(numerical_features)

ggplot(data = numerical_features, aes(x=SalePrice)) +
  geom_histogram(aes(y= ..density..),
                 bins = 50, fill = '#FF9933', color='black') +
  geom_density( alpha = 0.3, stat = 'density', fill='white', color='black') +
  scale_x_continuous(label = function(l) as.integer(format(l,scientific=F)))+
  theme(panel.grid.major = element_blank(), 
        panel.background = element_rect('white'),
        text = element_text(size = 20),
        plot.title = element_text(hjust = 0.5))

# We use principal component analysis to find out which of these features explain the saleprice most effectively.

pca <- prcomp(select(numerical_features, -ncol(numerical_features)), scale. = T)
pca1 <- prcomp(numerical_features, scale. = T)
variance <- pca$sdev^2
variance_explained <- round(variance/ sum(variance)*100, 1)

ggdata <- data.frame('PCs' = 1:length(variance_explained), 
                     'variance' = variance_explained)

ggdata2 <- data.frame('PCs' = 1:length(variance_explained), 
'variance' = cumsum(variance_explained))

ggplot(data = ggdata, aes(x=PCs, y = variance)) +
  geom_bar(aes(fill = variance_explained), stat = 'identity', width = 0.8, color='black') +
  geom_text(aes(label =variance_explained), size=4, vjust=-0.8,color='black' ) +
  xlab('Principal Components') +
  ylab('Variance Explained') +
  scale_fill_gradient2(low = '#FAF66A', mid = '#FABF6A', high = '#F0600E')+
  theme(legend.position = 'none',
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect('white'),
        text = element_text(size = 20))



ggplot(data = ggdata2, aes(x=PCs, y = variance)) +
  geom_bar(aes(fill = variance_explained), stat = 'identity', width = 0.8, color='black') +
  geom_text(aes(label = cumsum(variance_explained)), size=4, vjust=-0.8,color='black' ) +
  xlab('Principal Components') +
  ylab('Variance Explained') +
  scale_fill_gradient2(low = '#F0600E', mid = '#FABF6A', high = '#FAF66A')+
  theme(legend.position = 'none',
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect('white'),
        text = element_text(size = 20))


# We can see above that some features do not contribute to variance and are hence redundant for this problem
# We now check the features that are really important in explaining variance of the data.
imp_features <- sort(abs(pca$rotation[,1]), decreasing = T)

# According to PCA, we see that first floor square feet area explains the most variance
#  Now we visualize first floor sqaure feet with respect to Sale Price

ggplot(data = numerical_features, aes(x=X1stFlrSF, y = SalePrice)) +
  geom_point(color='black', 
             size=3, 
             shape=21, 
             fill='#999966',
             alpha=0.8) + geom_smooth(method = 'lm', color='blue') +
  xlab('First floor Square feet area')+
  ylab('SalePrice')+
  scale_y_continuous(label = function(l) as.integer(format(l,scientific=F)))+
  theme(legend.position = 'none',
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect('white'),
        text = element_text(size = 20))

# The plot shows that there is a strong relationship between them and also it suggests presence of some outliers


# Now we see Total Basement SquareFeet against SalePrice
ggplot(data = numerical_features, aes(x=TotalBsmtSF, y = SalePrice)) +
  geom_point(color='black', 
             size=3, 
             shape=21, 
             fill='#339966',
             alpha=0.8) + geom_smooth(method = 'lm', color='red') +
  xlab('Total Basement Square feet area')+
  ylab('SalePrice')+
  scale_y_continuous(label = function(l) as.integer(format(l,scientific=F)))+
  theme(legend.position = 'none',
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect('white'),
        text = element_text(size = 20))
# Again we have some outliers and there is a strong relationship between the variables.


# Now we check ground living area and saleprice

ggplot() +
  geom_point(data = numerical_features, 
             aes(x= GrLivArea, 
                 y = SalePrice),
             color='black',
             fill = '#CC3333', 
             size = 3, 
             shape=21, 
             alpha=0.6) +
  geom_point(data = as.data.frame(numerical_features[c(524,1299), 
                                                     c("GrLivArea", "SalePrice")]),
             aes(x=GrLivArea, y = SalePrice), 
             color='black', 
             size=4, 
             shape=21, 
             fill='#3399FF',
             alpha=0.8)+
  geom_smooth(method = 'loess')+
  xlab('Ground Living Area')+
  ylab('SalePrice')+
  scale_color_brewer(palette = 'Set3')+
  scale_y_continuous(label = function(l) as.integer(format(l,scientific=F)))+
  scale_fill_brewer(palette = 'Spectral') +
  theme(legend.position = 'none',
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect('white'),
        text = element_text(size = 20))
# Here we see that there is a strong positive correlation between ground living area and sale price but also, we have some outliers which we shall remove as suggested by author.
numerical_features <- numerical_features[-c(which(GrLivArea>4500)), ]
categorical_features <- categorical_features[-c(524,1299), ]


# Now we look at Garage Area against Sale Price
ggplot(data = numerical_features, aes(x=GarageArea, y = SalePrice)) +
  geom_point(color='black', 
             size=3, 
             shape=21, 
             fill='#3399FF',
             alpha=0.8) + geom_smooth(method = 'lm', color='red') +
  xlab('Masonry Veneer Area')+
  ylab('SalePrice')+
  scale_y_continuous(label = function(l) as.integer(format(l,scientific=F)))+
  theme(legend.position = 'none',
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect('white'),
        text = element_text(size = 20))
# It again has a strong positive correlation with sale price

ggplot(data = numerical_features, aes(x=MasVnrArea, y = SalePrice)) +
  geom_point(color='black', 
             size=3, 
             shape=21, 
             fill='#3399FF',
             alpha=0.8) + geom_smooth(method = 'lm', color='red') +
  xlab('Garage Area')+
  ylab('SalePrice')+
  scale_y_continuous(label = function(l) as.integer(format(l,scientific=F)))+
  theme(legend.position = 'none',
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect('white'),
        text = element_text(size = 20))

# From the plot, we can see that most of the values are zero and if we check it more than 50% values are missing, maybe because this feature is not that important?


# Now we look at Finished basement square feet area against Sale Price
ggplot(data = numerical_features, aes(x=BsmtFinSF1, y = SalePrice)) +
  geom_point(color='black', 
             size=3, 
             shape=21, 
             fill='#3399FF',
             alpha=0.8) + geom_smooth(method = 'lm', color='red') +
  xlab('Basement Area')+
  ylab('SalePrice')+
  scale_y_continuous(label = function(l) as.integer(format(l,scientific=F)))+
  theme(legend.position = 'none',
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect('white'),
        text = element_text(size = 20))



# Now we look at Finished basement square feet area against Sale Price
ggplot(data = numerical_features, aes(x=LotArea, y = SalePrice)) +
  geom_point(color='black', 
             size=3, 
             shape=21, 
             fill='#3399FF',
             alpha=0.8) + geom_smooth(method = 'lm', color='red') +
  xlab('Lot Area')+
  ylab('SalePrice')+
  scale_y_continuous(label = function(l) as.integer(format(l,scientific=F)))+
  theme(legend.position = 'none',
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect('white'),
        text = element_text(size = 20))


# REsizing the numerical features according to analysis above.

numerical_features <- numerical_features[,-c(11,20,6,21)]

# Just out of curiosity, let's see how many of these columns have zero values
zeroes = list()
for(i in colnames(numerical_features)){
  zeroes = c(zeroes,list(i = length(which(numerical_features[,i] == 0))))
}
names(zeroes) = colnames(numerical_features)



# Here we have some really important numerical features as we can see, and now we see the association between some of these features and our sales.
ggplot(data = categorical_features, 
       aes(x= OverallQual, 
           y = numerical_features[,ncol(numerical_features)], 
           fill = as.factor(OverallQual))) +
  geom_boxplot( color = 'black', outlier.color = NULL) +
  xlab('OverallQual')+
  ylab('SalePrice')+
  scale_color_brewer(palette = 'Set3')+
  scale_y_continuous(label = function(l) as.integer(format(l,scientific=F)))+
  scale_fill_brewer(palette = 'Spectral') +
  theme(legend.position = 'none',
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect('white'),
        text = element_text(size = 20))

ggplot(data = numerical_features, aes(x=GarageCars, y = SalePrice, fill=as.factor(GarageCars))) +
  geom_boxplot()+
  xlab('Number of cars in garage')+
  ylab('SalePrice')+
  scale_y_continuous(label = function(l) as.integer(format(l,scientific=F)))+
  theme(legend.position = 'none',
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect('white'),
        text = element_text(size = 20))


# Since number of garage cars will always be proportional to the garage area and we check it finding the correlation between them, they are highly correlated and so we remove garage cars and keep garage area

numerical_features <- numerical_features[,-c(26)]

ggplot(data = numerical_features, aes(x=FullBath, y = SalePrice, fill=as.factor(FullBath))) +
  geom_boxplot()+
  xlab('Number of full bathrooms')+
  ylab('SalePrice')+
  scale_y_continuous(label = function(l) as.integer(format(l,scientific=F)))+
  theme(legend.position = 'none',
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect('white'),
        text = element_text(size = 20))

# In real life also, it doesn't make sense, since increasing bathrooms decreases the livable area and so does not contribute to the price
numerical_features <- numerical_features[,-c(19)]




# Now we check year built, does it feel important? I think, remodelling year would be more importat but let's check
ggplot(data = numerical_features, aes(x=YearBuilt, y = SalePrice, fill=as.factor(YearBuilt))) +
  geom_boxplot()+
  xlab('Year Built')+
  ylab('SalePrice')+
  scale_y_continuous(label = function(l) as.integer(format(l,scientific=F)))+
  theme(legend.position = 'none',
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect('white'),
        text = element_text(size = 20))
# Yes, we can see that it was week correlation with sale price.

ggplot(data = numerical_features, aes(x=TotRmsAbvGrd, y = SalePrice, fill=as.factor(TotRmsAbvGrd))) +
  geom_boxplot()+
  xlab('Number of rooms above ground')+
  ylab('SalePrice')+
  scale_y_continuous(label = function(l) as.integer(format(l,scientific=F)))+
  theme(legend.position = 'none',
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect('white'),
        text = element_text(size = 20))
# This trend shows a good relationship between number of rooms and sale price

ggplot(data = numerical_features, aes(x=GarageYrBlt,  y = SalePrice, fill=as.factor(GarageYrBlt))) +
  geom_boxplot()+
  xlab('Garage Year Built')+
  ylab('SalePrice')+
  scale_y_continuous(label = function(l) as.integer(format(l,scientific=F)))+
  theme(legend.position = 'none',
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect('white'),
        text = element_text(size = 20))

# It has a positive correlation with sale price
ggplot(data = numerical_features, aes(x=YearRemodAdd,  y = SalePrice, fill=as.factor(YearRemodAdd))) +
  geom_boxplot()+
  xlab('Garage Year Built')+
  ylab('SalePrice')+
  scale_y_continuous(label = function(l) as.integer(format(l,scientific=F)))+
  theme(legend.position = 'none',
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect('white'),
        text = element_text(size = 20))
# Not much relation 





















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















