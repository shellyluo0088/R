#################
#Multi-Linear Regression and Predictive Modeling on Ames Housing Dataset
# Xi "Shelly" Luo
# 12/1/2022
#################

##############Analysis 1 (Simple Linear Regression)#####################
## 1a) Develop SLR:

#import the Ames_neighborhoods data set 
setwd("D:/Documents")
Ames_neighbor = read.csv(file = 'Ames_neighborhoods.csv')
head(Ames_neighbor)

#try the simplest SLR first
ames_lm_1 = lm(Ames_neighbor$SalePrice ~ Ames_neighbor$GrLivArea, data = Ames_neighbor)
summary(ames_lm_1)

#scatter plot of two variables
plot(x = Ames_neighbor$GrLivArea, y = Ames_neighbor$SalePrice, main="SalePrice vs GrLivArea",
     xlab = "Above grade (ground) living area square feet", 
     ylab = "Sales Price",
     pch = 20, cex = 2, col = "grey")
abline(reg = ames_lm_1, lwd = 3, col="red")
#several outliers spotted, could be influential points


#check normality assumption
library(olsrr)
ols_plot_resid_qq(ames_lm_1)
shapiro.test(resid(ames_lm_1))

###################
#p-value is 0.446, which is more then the significance level of 0.05. 
#We failed to reject the null hypothesis 
#and conclude that the errors have a normal distribution.

# check the constant variance assumption
library(lmtest)
ols_plot_resid_fit(ames_lm_1)
bptest(ames_lm_1)


##################
#The value of the test statistic is 6.6446 with a p-value of 0.009946
#We reject the null hypothesis and conclude that the errors are heteroscedastic.
#so the constant variance assumption is violated.

#consider log transformation: use box-cox to find lambda
library(MASS)
bc = boxcox(ames_lm_1, lambda = seq(-0.25, 1.5, by = 0.05), plotit = TRUE)
bc$x[which.max(bc$y)] #optimal lambda:0.72

#find 95% CI for lambda to justify the log transformation
get_lambda_ci = function(bc, level = 0.95) {
  # lambda such that 
  # L(lambda) > L(hat(lambda)) - 0.5 chisq_{1, alpha}
  CI_values = bc$x[bc$y > max(bc$y) - qchisq(level, 1)/2]
  
  # 95 % CI 
  CI <- range(CI_values) 
  
  # label the columns of the CI
  names(CI) <- c("lower bound","upper bound")
  
  CI
}

# extract the 95% CI from the box cox object
get_lambda_ci(bc)

#let's try the square root transformation since 0.5 is contained in the CI
ames_lm_2 = lm(SalePrice^ 0.5 ~ GrLivArea, data = Ames_neighbor)
summary(ames_lm_2)

#is the transformation of response fixed the assumption violation?
bptest(ames_lm_2)
library(lmtest)
ols_plot_resid_fit(ames_lm_2)
##################
#The value of the test statistic is 0.75518 with a p-value of 0.3848
#We failed reject the null hypothesis and conclude that the constant variance assumption is hold.

#compare two models by using fitted-vs-residual plot
par(mfrow = c(1, 2))
# OLS fitted-vs-residual plot
plot(fitted(ames_lm_1), resid(ames_lm_1), 
     pch = 20, 
     xlab = 'Fitted Value', ylab = 'Residual')

abline(h=0, lwd=3, col='steelblue')

# transformed OLS fitted-vs-residual plot
plot(fitted(ames_lm_2), resid(ames_lm_2), 
     pch = 20, 
     xlab = 'Fitted Value', ylab = 'Residual')

abline(h=0, lwd=3, col='steelblue')


#check normality assumption
library(olsrr)
ols_plot_resid_qq(ames_lm_2)
shapiro.test(resid(ames_lm_2))
###################
#p-value is 0.0616, which is more then the significance level of 0.05. 
#We failed to reject the null hypothesis 
#and conclude that the errors have a normal distribution.

# plot the model
plot(SalePrice^ 0.5 ~ GrLivArea,
     data =Ames_neighbor,
     xlab = "Above grade (ground) living area square feet",
     ylab = "Sales Price^ 0.5",
     main = "SalePrice^ 0.5 vs GrLivArea",
     pch = 20,
     cex = 2,
     col = "gray")
abline(ames_lm_2, lwd = 3, col = "darkorange")

#unusual observations
outlier_test_cutoff = function(model, alpha = 0.05) {
  n = length(resid(model))
  qt(alpha/(2 * n), df = df.residual(model) - 1, lower.tail = FALSE)
}
cutoff = outlier_test_cutoff(ames_lm_2, alpha = 0.05)


outliers = which(abs(rstudent(ames_lm_2)) > cutoff)
outliers
high_inf_ids = which(cooks.distance(ames_lm_2) > 4/length(resid(ames_lm_2)))
high_inf_ids
high_lev_ids = which(hatvalues(ames_lm_2) > 2 * mean(hatvalues(ames_lm_2)))
high_lev_ids
#There are 11 high leverage, and 11 highly influential points in our model.

#remove the highly influential points then refit the model:
non_inf_ids = which(cooks.distance(ames_lm_2) <= 4/length(resid(ames_lm_2)))
ames_lm_2_fix = lm(SalePrice^ 0.5 ~ GrLivArea, 
                   data = Ames_neighbor,
                   subset = non_inf_ids)
summary(ames_lm_2_fix)

bptest(ames_lm_2_fix)
library(lmtest)
ols_plot_resid_fit(ames_lm_2_fix)

#scatter plot

plot(SalePrice^ 0.5 ~ GrLivArea, 
     data = Ames_neighbor,
     subset = non_inf_ids,
     main = "SalePrice^ 0.5 vs GrLivArea with influencial ids removed",
     xlab = 'Above grade (ground) living area square feet', 
     ylab = 'Sales price^0.5',
     pch = 20, cex = 2, col = 'gray')
abline(ames_lm_2, lwd = 3, col = "darkorange")

##1b) OLS and WLS:

# fit the OLS model of |e_i| ~ predictors. 
# NOTE: Remember to remove the response!
model_wts = lm(abs(resid(ames_lm_1)) ~ . - Ames_neighbor$SalePrice, data = Ames_neighbor)

# extract the coefficient estimates.
coef(model_wts)

# calculate the weights as 1 / (fitted values)^2
weights = 1 / fitted(model_wts)^2
print(weights)
# run WLS
model_wls = lm(Ames_neighbor$SalePrice ~ Ames_neighbor$GrLivArea, data = Ames_neighbor, weights = weights)
summary(model_wls)

#plot fitted vs residual
plot(fitted(model_wls), weighted.residuals(model_wls), 
     pch = 20, xlab = 'Fitted Value', ylab = 'Weighted Residual')

abline(h=0, lwd=3, col='steelblue')
bptest(model_wls)

#WLS model with the influential ids removed:
model_wls_fix = lm(Ames_neighbor$SalePrice ~ Ames_neighbor$GrLivArea, 
                   data = Ames_neighbor, 
                   weights = weights,
                   subset = non_inf_ids)
summary(model_wls_fix)
bptest(model_wls_fix)

#plot fitted vs residual
plot(fitted(model_wls_fix), weighted.residuals(model_wls_fix), 
     pch = 20, xlab = 'Fitted Value', ylab = 'Weighted Residual')

abline(h=0, lwd=3, col='steelblue') 
#model_wls_fix is the best because r square value is the highest among four models
# it also passed the bp test 
#95% CI for slope
confint(model_wls_fix, level = 0.95)


##############Analysis 2 (Multiple Linear Regression) #####################
#2a) check multicollinearity and fix it
#import the ames_mlr data set 
setwd("D:/Documents")
ames_mlr = read.csv(file = 'ames_mlr.csv')
head(ames_mlr)

#calculate the pairwise correlation:
library(dplyr)

# data.frame containing just the predictors
ames_preds= dplyr::select(ames_mlr, c(TotalBsmtSf, GrLivArea, FirstFlrSf, SecondFlrSf, GarageCars, GarageArea, Fireplaces, ScreenPorch))
round(cor(ames_preds), 3)
library(corrplot)
corrplot(cor(ames_preds), 
         method = 'color', order = 'hclust',  diag = FALSE,
         number.digits = 3, addCoef.col = 'black', tl.pos= 'd', cl.pos ='r')
pairs(ames_preds, col = 'dodgerblue', pch=20)
#formal diagnostic (using VIF)
library(faraway)
ames_mlr_1 = lm(log(SalePrice) ~ ames_mlr$TotalBsmtSf + ames_mlr$GrLivArea 
                + ames_mlr$FirstFlrSf + ames_mlr$SecondFlrSf + ames_mlr$GarageCars
                + ames_mlr$GarageArea + ames_mlr$Fireplaces + ames_mlr$ScreenPorch, 
                data = ames_mlr)
summary(ames_mlr_1)
vif(ames_mlr_1)
#There are three variables are highly correlated to each other, "GrLivArea", "FirstFlrSf" and "SecondFlrSf"
#Creating new variables
ames_df = ames_mlr %>% select(c(SalePrice, TotalBsmtSf, GrLivArea, FirstFlrSf, SecondFlrSf, 
                                GarageCars, GarageArea, Fireplaces,  ScreenPorch))
ames_df["TotalHouseSF"] = ames_df["TotalBsmtSf"] + ames_df["FirstFlrSf"] + ames_df["SecondFlrSf"]
ames_mlr_2 = lm(log(SalePrice) ~ TotalHouseSF + GrLivArea + GarageCars
                + GarageArea + Fireplaces + ScreenPorch, data = ames_df)
summary(ames_mlr_2)
vif(ames_mlr_2)



#2b) full model and restricted model:
#Here, we Use an F -test to compare the following two models:
#Model 1:  log(SalePricei) =β0 + β1 TotalBsmtSFi + β2 GrLivArea+β3 FirstFlrSFi +  β4 GarageCarsi+ β5 Fireplacesi + β6 ScreenPorchi;
#Model 2: log(SalePricei) =β0 + β1 TotalBsmtSFi + β2 GrLivArea+β3 FirstFlrSFi + β4 SecondFlrSFi + β5 GarageCarsi+ β6 GarageAreai + β7 Fireplacesi + β8 ScreenPorchi;
#The null and alternative hypotheses are
#H0 : βSecondFlrSF = βGarageCars = 0
#H1 : Any of βSecondFlrSF or βGarageCars not equal to zero.
restricted_model = lm(log(SalePrice) ~ TotalBsmtSf + FirstFlrSf + GrLivArea + GarageCars
                      + Fireplaces + ScreenPorch, data = ames_df)
full_model = ames_mlr_1
anova(restricted_model, full_model)
#The p-value of the test is 0.1989.
#Stats Decision: We failed to reject the null hypothesis at the α = 0.05 significance level.
#Conclusion: The two variables, SecondFrlsf and GarageArea have no significant linear
#relationship with log(SalePrice), given the other predictors are in the model. 
#As such,we prefer Model 1.


#2c)try robust regression with model 2:
library(MASS)

# IRWLS with a limit of 100 iterations.
model_hub = rlm(log(SalePrice) ~ TotalBsmtSf + FirstFlrSf + GrLivArea + GarageCars
                + Fireplaces + ScreenPorch, data = ames_df)

summary(model_hub)
library(car)
set.seed(42)
Confint(Boot(model_hub, R = 2000, method = 'residual'))
##The robust regression does not provide valid CIs
#Let's try OLS instead:
ames_mlr_3 = lm(log(SalePrice) ~ TotalBsmtSf + FirstFlrSf + GrLivArea + GarageCars
                      + Fireplaces + ScreenPorch, data = ames_df)
summary(ames_mlr_3)
confint(ames_mlr_3, level = 0.95)

#check normality assumption
library(olsrr)
ols_plot_resid_qq(ames_mlr_3)
shapiro.test(resid(ames_mlr_3)) #model didn't met the normality assumption might due to the outliers


#bp test and the residual fitted value plot
library(lmtest)
bptest(ames_mlr_3)
ols_plot_resid_fit(ames_mlr_3) #passed the bp test and the plot looks good




##############Analysis 3 (Predictive Modeling) #####################
#3a) compare models with RMSE
#import the ames_train.csv and ames_test.csv
setwd("D:/Documents")
ames_train = read.csv(file = 'ames_train.csv')
head(ames_train)
dim(ames_train)
ames_test = read.csv(file = "ames_test.csv")
head(ames_test)
dim(ames_test)


##############
##detecting and treating outliers on training data set
##############
outlier_test_cutoff = function(model, alpha = 0.05) {
  n = length(resid(model))
  qt(alpha/(2 * n), df = df.residual(model) - 1, lower.tail = FALSE)
}
cutoff = outlier_test_cutoff(model1, alpha = 0.05)


outliers = which(abs(rstudent(model1)) > cutoff)
outliers #599th observation is an outlier
high_inf_ids = which(cooks.distance(model1) > 4/length(resid(model1)))
high_inf_ids #many highly influential points, including the outlier (599th)
high_lev_ids = which(hatvalues(model1) > 2 * mean(hatvalues(model1)))
high_lev_ids #many high leverage points

#removing the highly influential points and the high leverage points
# ids for non-influential observations
ames_train_fixed = which(cooks.distance(model1) <= 4 / length(cooks.distance(model1)))
################
##Model 1: SLR
################
#setting up model1: WLS without influential ids:
model1 = lm(log(SalePrice) ~ GrLivArea, 
                   data = ames_train,
            subset = ames_train_fixed)
summary(model1)
#calculate weights and remove the influential ids:
non_inf_ids_model1 = which(cooks.distance(model1) <= 4/length(resid(model1)))
# fit the OLS model of |e_i| ~ predictors. 
# NOTE: Remember to remove the response!
model_wts_2 = lm(abs(resid(model1)) ~ . - ames_train$SalePrice, data = ames_train)

# extract the coefficient estimates.
coef(model_wts_2)

# calculate the weights as 1 / (fitted values)^2
weights = 1 / fitted(model_wts_2)^2

# run WLS
model1_wls_fix = lm(log(SalePrice) ~ GrLivArea, 
                    data = ames_train, 
                    weights = weights,
                    subset = non_inf_ids_model1)
summary(model1_wls_fix)
#for training data set, original model1(OLS) has a higher r square value, 
#thus we decided to keep the original model1
#check assumptions for model 1
bptest(model1) #passed the bp test
library(olsrr)
ols_plot_resid_qq(model1)
shapiro.test(resid(model1)) #didn't pass the normality test
#splitting the data set
X_train = subset(ames_train, select = -c(SalePrice))
y_train = subset(ames_train, select = c(SalePrice))
X_test = subset(ames_test, select = -c(SalePrice))
y_test = subset(ames_test, select = c(SalePrice))
y_train_log = log(y_train)
y_test_log = log(y_test)

#calculate test RMSE for model1:
y_pred_1 = predict(model1, X_test)
print(mean(y_pred_1))
result1= (y_test_log - y_pred_1)^2
mean1 = mean(as.numeric(unlist(result1)), na.rm = TRUE)
rmse_model1 = sqrt(mean1)  #RMSE = 0.2223933

################
##Model 2: MLR
################
model2 = lm(log(SalePrice) ~ TotalBsmtSf + FirstFlrSf + GrLivArea + GarageCars
                + Fireplaces + ScreenPorch, data = ames_train)
summary(model2) #has 0.6471 of r square value
#check assumptions for model 2
bptest(model2) #passed the bp test
library(olsrr)
ols_plot_resid_qq(model2)
shapiro.test(resid(model2)) #didn't pass the normality test

#calculate test RMSE for model1:
y_pred_2 = predict(model2, X_test)
print(mean(y_pred_2))
result2= (y_test_log - y_pred_2)^2
mean2 = mean(as.numeric(unlist(result2)), na.rm = TRUE)
rmse_model2 = sqrt(mean2)  #RMSE = 0.1654244

##############################
##Model 3: backward selection 
###############################
#Using AIC
mod_all_preds = lm(log(SalePrice) ~ ., data = ames_train)

# NOTE:  step defaults to using AIC
mod_back_aic = step(mod_all_preds, direction = 'backward')
coef(mod_back_aic) #using 18 predictors with the lowest aic: -3061.286
extractAIC(mod_back_aic)

#backward selection with bic:
n = nrow(ames_train)
mod_back_bic = step(mod_all_preds, direction = 'backward', k = log(n))
coef(mod_back_bic)
extractAIC(mod_back_bic, k = log(n))#15 predictors with lowest BIC: -2986.59

#choosing between AIC and BIC:
summary(mod_back_aic)$adj.r.squared
summary(mod_back_bic)$adj.r.squared #aic model has a bit higher adjusted r squared value

calc_loocv_rmse = function(model) {
  sqrt(mean((resid(model) / (1 - hatvalues(model))) ^ 2))
}
calc_loocv_rmse(mod_back_aic)
calc_loocv_rmse(mod_back_bic) #aic model has lower loocv_rmse, we prefer the model chosen by AIC

#fit model 3:
model3 = lm(formula = log(SalePrice) ~ LotArea + OverallQual + OverallCond + 
              YearBuilt + YearRemodAdd + BsmtFinSf1 + BsmtFinSf2 + TotalBsmtSf + 
              FirstFlrSf + GrLivArea + BsmtFullBath + BsmtHalfBath + BedroomAbvGr + 
              KitchenAbvGr + TotRmsAbvGrd + Fireplaces + GarageArea + ScreenPorch, 
            data = ames_train)
summary(model3) #86.58 r squared value

#check assumptions for model 3
bptest(model3) #didn't passed the bp test
library(olsrr)
ols_plot_resid_qq(model3)
shapiro.test(resid(model3)) #didn't pass the normality test

#calculate test RMSE for model3:
y_pred_3 = predict(model3, X_test)
print(mean(y_pred_3))
result3= (y_test_log - y_pred_3)^2
mean3 = mean(as.numeric(unlist(result3)), na.rm = TRUE)
rmse_model3 = sqrt(mean3)  #RMSE = 0.09845752



##############################
##Model 4: best subset selection 
###############################
#install.packages("leaps")
library(leaps)
mod_exhaustive = summary(regsubsets(log(SalePrice) ~ ., data = ames_train))
mod_exhaustive$which

#Extract The Best Adjusted R square Model
mod_exhaustive$adjr2
best_r2_ind = which.max(mod_exhaustive$adjr2)
mod_exhaustive$which[best_r2_ind,] #using 8 predictors

#refit the model with 8 predictors
model4 = lm(log(SalePrice) ~ LotArea + OverallQual + OverallCond + 
              YearBuilt  + BsmtFinSf1  + FirstFlrSf + GrLivArea + GarageArea, 
            data = ames_train)
summary(model4) #84.94 r square value-pretty similar with model3, but with much less variables

#calculate test RMSE for model4:
y_pred_4 = predict(model4, X_test)
print(mean(y_pred_4))
result4= (y_test_log - y_pred_4)^2
mean4 = mean(as.numeric(unlist(result4)), na.rm = TRUE)
rmse_model4 = sqrt(mean4)  #RMSE = 0.1030226

########################################
##Model 5: Principal Component Regression
########################################
library(pls)

# set random seed to 2 for reproducability
set.seed(2)

# fit the PCR model
mod_pcr = pcr(log(SalePrice) ~ ., data = ames_train, 
              scale = TRUE, 
              validation = 'CV')

summary(mod_pcr)

#scree plot
# percentage of variance explained
pve = mod_pcr$Xvar / mod_pcr$Xtotvar

# scree plot
plot(pve, type='b', pch=20, lwd=2, 
     xlab = 'Principal Component', 
     ylab = 'Percentage of Variance Explained')
#24 components model has the lowest rmse: 0.1076
which.min(RMSEP(mod_pcr)$val[1,1, ]) - 1
#coefficents for 24 components
mod_pcr$Yloadings[1:24]

#calculate RMSE on test 
# there are 24 principal components in total 
y_pred_5 = predict(mod_pcr, ames_test, ncomp = 24)
result5= (y_test_log - y_pred_5)^2
mean5 = mean(as.numeric(unlist(result5)), na.rm = TRUE)
rmse_model5 = sqrt(mean5)  #RMSE = 0.0966 #not bad rmse, but too much components



############################
##Model 6: Ridge Regression
############################

library(lmridge)
# lambda values evenly spaced on the log-scale from 10^10 to 10^-2.
grid = 10 ^ seq(10, -2 , length = 100)
# estimate ridge regression on the training data
mod_ridge = lmridge(log(SalePrice)~., 
                    data = ames_train, 
                    scaling = 'scaled', 
                    K = grid)
# extract the GCV errors and lambda that minimizes the GCV error
k_est = kest(mod_ridge)

# a plot of GCV vs. log10(lambda) 
plot(log10(mod_ridge$K), k_est$GCV, type = 'l', lwd = 2,
     xlab = expression(log[10](lambda)), ylab = 'GCV')

points(log10(mod_ridge$K), k_est$GCV, 
       pch = 19, col = 'steelblue', cex = 0.75)

# horizontal line at log10(kGCV), i.e.,
# the base 10 logarithm of the best lambda value
abline(v=log10(k_est$kGCV), lty = 'dashed', col = 'grey',
       lwd = 2)

# value of lambda chosen by GCV
k_est$kGCV
cv.plot(mod_ridge) #k = 14.1474

##improve the lambda value
# lambda values evenly spaced on the log-scale from 10^1 to 10^2.5.
grid = 10 ^ seq(1, 2.5 , length = 100)

# estimate ridge regression on the training data
mod_ridge_2 = lmridge(log(SalePrice)~., 
                      data = ames_train, 
                      scaling = 'scaled', 
                      K = grid)

# extract the GCV errors and lambda that minimizes the GCV error
k_est = kest(mod_ridge_2)

# a plot of GCV vs. log10(lambda) 
plot(log10(mod_ridge_2$K), k_est$GCV, type = 'l', lwd = 2,
     xlab = expression(log[10](lambda)), ylab = 'GCV')

points(log10(mod_ridge_2$K), k_est$GCV, 
       pch = 19, col = 'steelblue', cex = 0.75)

# horizontal line at log10(kGCV), i.e.,
# the base 10 logarithm of the best lambda value
abline(v=log10(k_est$kGCV), lty = 'dashed', col = 'grey',
       lwd = 2)

# value of lambda chosen by GCV
k_best = k_est$kGCV #k best is  12.32847

# re-fit the model using the best value of lambda according to GCV
model6 = lmridge(log(SalePrice)~., 
                 data = ames_train, 
                 scaling = 'scaled', 
                 K = k_best)
summary(model6)
coef(model6) #no coefficient is zero because ridge do not do variable selection
#but there are only 16 predictors siginificant based on the result

#calculate test RMSE for model6:
y_pred_6 = predict(model6, X_test)
print(mean(y_pred_6))
result6= (y_test_log - y_pred_6)^2
mean6 = mean(as.numeric(unlist(result6)), na.rm = TRUE)
rmse_model6 = sqrt(mean6)  #RMSE =  0.1017397




############################
##Model 7: Lasso Regression
############################
y_train = y_train_log$SalePrice
X_train = model.matrix(log(SalePrice) ~ ., data = ames_train)[, -1]
library(glmnet)
# set the random seed for reproducibility
set.seed(123)
# fit the lasso using 10-fold cross-validation to determine lambda
mod_lasso = cv.glmnet(X_train, y_train)

# the logarithm in this plot is the natural logarithm
plot(mod_lasso)

mod_lasso$lambda.min
mod_lasso$lambda.1se


# find the best lambda:
# set the random seed for reproducability
set.seed(42)

# lambda values evenly spaced on the natural log-scale from e^-1 to e^-3.
grid = (seq(0.001, 0.15, length=100))

# fit the lasso using 10-fold cross-validation to determine lambda

mod_lasso_2 = cv.glmnet(X_train, y_train, lambda = grid)

# the logarithm in this plot is the natural logarithm
plot(mod_lasso_2)

# lambda.min
mod_lasso_2$lambda.min

# lambda.1se
mod_lasso_2$lambda.1se


#report the non-zero coefficients using lambda.min
coef(mod_lasso_2, s = 'lambda.min') #26 variables

#report the non-zero coefficients using lambda.1se
coef(mod_lasso_2, s = 'lambda.1se') #14 variables

# predictions using lambda.min
# matrix of predictors with the first column of ones removed
x_test = model.matrix(log(SalePrice) ~ ., data = ames_test)[, -1]
y_test = y_test_log$SalePrice
y_pred_7_min = predict(mod_lasso_2, newx = x_test, s = 'lambda.min')

# calculate the RMSE
# quick function to calculate RMSE
rmse = function(y_true, y_pred) {
  sqrt(mean((y_true - y_pred)^2))
}
rmse(y_test, y_pred_7_min) #RMSE = 0.0971874


# predictions using lambda.1se
y_pred_7_1se = predict(mod_lasso_2, newx = x_test, s = 'lambda.1se')

# calculate the RMSE
rmse(y_test, y_pred_7_1se) #RMSE = 0.1031045

#calculate training r squared value (lambda.min)
y_train_pred_lasso <- predict(mod_lasso_2, X_train,s = 'lambda.min')
sse <- sum((y_train - y_train_pred_lasso)^2)
sst <- sum((y_train - mean(y_train))^2)
training_rsq <- 1 - sse / sst #r squared value = 0.8661379


#calculate training r squared value (lambda.1se)
y_train_pred_lasso <- predict(mod_lasso_2, X_train,s = 'lambda.1se')
sse <- sum((y_train - y_train_pred_lasso)^2)
sst <- sum((y_train - mean(y_train))^2)
training_rsq <- 1 - sse / sst #r squared value = 0.8454803




######################
#Final Model
######################

#ames_train modify:
library(tidyr)
ames_train["GarAreaPerCar"] = (ames_train["GarageArea"] / ames_train["GarageCars"])
ames_train$GarAreaPerCar %>% replace_na(0)
ames_train["GrLivAreaPerRoom"] = ames_train["GrLivArea"] / ames_train["TotRmsAbvGrd"]
ames_train["TotalHouseSF"] = ames_train["TotalBsmtSf"] + ames_train["FirstFlrSf"] + ames_train["SecondFlrSf"]
ames_train["TotalFullBath"] = ames_train["FullBath"] + ames_train["BsmtFullBath"]
ames_train["TotalHalfBath"] = ames_train["HalfBath"] + ames_train["BsmtHalfBath"]
ames_train["RemodHouseAge"] = ames_train["YearRemodAdd"] - ames_train["YearBuilt"]
ames_train['TotalPorchSF'] = ames_train['OpenPorchSf'] + ames_train['EnclosedPorch'] + ames_train['ThreeSsnPorch'] + ames_train['ScreenPorch']
ames_train["AvgQualCond"] = (ames_train["OverallQual"] + ames_train["OverallCond"]) / 2
head(ames_train)

#drop columns:
ames_train_dropped = subset(ames_train, select = -c(TotalBsmtSf,SecondFlrSf, FullBath, BsmtFullBath, HalfBath,
                                           BsmtHalfBath, YearRemodAdd, YearBuilt, OpenPorchSf, EnclosedPorch,
                                           ThreeSsnPorch,ScreenPorch,OverallQual, OverallCond, HasGarage)) 
dim(ames_train_dropped)
head(ames_train_dropped)


#ames_test modify:
library(tidyr)
ames_test["GarAreaPerCar"] = (ames_test["GarageArea"] / ames_test["GarageCars"])
ames_test$GarAreaPerCar %>% replace_na(0)
ames_test["GrLivAreaPerRoom"] = ames_test["GrLivArea"] / ames_test["TotRmsAbvGrd"]
ames_test["TotalHouseSF"] = ames_test["TotalBsmtSf"] + ames_test["FirstFlrSf"] + ames_test["SecondFlrSf"]
ames_test["TotalFullBath"] = ames_test["FullBath"] + ames_test["BsmtFullBath"]
ames_test["TotalHalfBath"] = ames_test["HalfBath"] + ames_test["BsmtHalfBath"]
ames_test["RemodHouseAge"] = ames_test["YearRemodAdd"] - ames_test["YearBuilt"]
ames_test['TotalPorchSF'] = ames_test['OpenPorchSf'] + ames_test['EnclosedPorch'] + ames_test['ThreeSsnPorch'] + ames_test['ScreenPorch']
ames_test["AvgQualCond"] = (ames_test["OverallQual"] + ames_test["OverallCond"]) / 2
head(ames_test)

#drop columns:
ames_test_dropped = subset(ames_test, select = -c(GarageArea, SecondFlrSf, FullBath, BsmtFullBath, HalfBath,
                                                    BsmtHalfBath, YearRemodAdd, YearBuilt, OpenPorchSf, EnclosedPorch,
                                                    ThreeSsnPorch,ScreenPorch,OverallQual, OverallCond, HasGarage)) 
dim(ames_test_dropped)
head(ames_test_dropped)

#fit OLS model with all predictors:
str(ames_train_dropped)
model_ols_final = lm(log(SalePrice)~ .,  data = ames_train_dropped)
summary(model_ols_final)

model_ols_final_selected = lm(log(SalePrice)~ LotArea + FirstFlrSf + KitchenAbvGr + TotalHouseSF +
                                TotalFullBath + RemodHouseAge + AvgQualCond, data = ames_train_dropped)
summary(model_ols_final_selected)
#splitting the data set
X_train_dropped = subset(ames_train_dropped, select = -c(SalePrice))
y_train = subset(ames_train_dropped, select = c(SalePrice))
X_test_dropped = subset(ames_test_dropped, select = -c(SalePrice))
y_test = subset(ames_test_dropped, select = c(SalePrice))
y_train_log = log(y_train)
y_test_log = log(y_test)
#RMSE on test data:
y_pred_final = predict(model_ols_final_selected, X_test_dropped)
print(mean(y_pred_final))
result= (y_test_log - y_pred_final)^2
mean = mean(as.numeric(unlist(result)), na.rm = TRUE)
rmse_final = sqrt(mean)  #RMSE = 0.1296568


#using best subset selection
library(leaps)
mod_exhaustive_final = summary(regsubsets(log(SalePrice) ~ ., data = ames_train_dropped))
mod_exhaustive_final$which
mod_exhaustive_final$adjr2

model_ols_best_subset = lm(log(SalePrice)~ LotArea + Fireplaces + GarageCars + TotalHouseSF +
                                TotalFullBath + RemodHouseAge + AvgQualCond, data = ames_train_dropped)
summary(model_ols_best_subset)
#RMSE on test data:
y_pred_final_best = predict(model_ols_best_subset, X_test_dropped)
print(mean(y_pred_final_best))
result= (y_test_log - y_pred_final_best)^2
mean = mean(as.numeric(unlist(result)), na.rm = TRUE)
rmse_final = sqrt(mean) #rmse = 0.1205783

#using best subset selection without dropping any variables
model_ols_without_dropping = lm(log(SalePrice)~ LotArea + YearBuilt + BsmtFinSf1 + GrLivArea
                                +TotalHouseSF + AvgQualCond + GarageArea, data = ames_train)
summary(model_ols_without_dropping)

#splitting the data set
X_train= subset(ames_train, select = -c(SalePrice))
y_train = subset(ames_train, select = c(SalePrice))
X_test = subset(ames_test, select = -c(SalePrice))
y_test = subset(ames_test, select = c(SalePrice))
y_train_log = log(y_train)
y_test_log = log(y_test)
dim(X_train)
dim(X_test)


#RMSE on test data:
y_pred = predict(model_ols_without_dropping, X_test)
print(mean(y_pred))
result= (y_test_log - y_pred)^2
mean = mean(as.numeric(unlist(result)), na.rm = TRUE)
rmse_final = sqrt(mean) #rmse = 0.1017563
###################This is our model, mission accomplished##############################
