data1<-read.csv("Data_20171121.csv",header = T)
View(data1)
#lets remove the columns with Zero variance
data1$ID<-NULL
data1$S7<-NULL
str(data1)
colSums(is.na(data1))
sum(is.na(data1))

## Replacing -99 with NA's
#install.packages("car")
library(car)
data1$P6<-recode(data1$P6,"-99.00=NA")
data1$P12<-recode(data1$P12,"-99.00=NA")
sum(is.na(data1))

library(DMwR)
## Central Imputation is done
data1<- centralImputation(data1)

sum(is.na(data1))
str(data1)
data1$Target<-factor(data1$Target)
data1$RF1<-factor(data1$RF1)
data1$RF2<-factor(data1$RF2)
data1$RF3<-factor(data1$RF3)
data1$RF4<-factor(data1$RF4)
data1$RF5<-factor(data1$RF5)
data1$PI<-factor(data1$PI)
#deviding the given data set into train and test data
set.seed(123)
rows<-seq(1,nrow(data1),1)
trainrows<-sample(rows,(0.7*nrow(data1)))
train<-data1[trainrows,]
test<-data1[-trainrows,]
dim(train)
dim(test)
library(caret)

data1_log<-glm(formula = Target~.,data = train,family = "binomial")
summary(data1_log) ## AIC Value =26670

#Predicting on train data
pred<-predict(data1_log,type = "response")

#predicitng on test data
pred1<-predict(data1_log,newdata=test,type = "response")
#Manually choose the threshold; Here, we take it as 0.5
pred_class<-ifelse(pred>0.50,1,0)
tab<-table(train$Target,pred_class)
tab

preds_test<-ifelse(pred1>0.50,1,0)
tab1<-table(test$Target,preds_test)
tab1
library(ROCR)
pred2<-prediction(pred,train$Target)
pred3<-prediction(pred1,test$Target)

#Extract performance measures (True Positive Rate and False Positive Rate) using the
#"performance()" function from the ROCR package
#??? The performance() function from the ROCR package helps us extract metrics such as True
#positive rate, False positive rate etc. from the prediction object, we created above.
#Two measures (y-axis = tpr, x-axis = fpr) are extracted
perf<-performance(pred2,x.measure ="fpr",measure = "tpr")
perf1<-performance(pred3,x.measure = "fpr",measure = "tpr")
#Plot the ROC curve using the extracted performance measures (TPR and FPR)

plot(perf,col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))
plot(perf1,col=rainbow(10),colorize=T, print.cutoffs.at=seq(0,1,0.05))
#Extract the AUC score of the ROC curve and store it in a variable named "auc"

perf_auc<-performance(pred2,measure = "auc")
perf1_auc<-performance(pred3,measure = "auc")
#Access the auc score from the performance object

auc <- perf_auc@y.values[[1]]
auc
auc1 <- perf1_auc@y.values[[1]]
auc1

# true Negative rate
specificity <- tab[2, 2]/sum(tab[2, ])
specificity
# true positive rate
sensitivity <- tab[1,1]/sum(tab[1,])
sensitivity

#precision
precision<-tab[1,1]/sum(tab[,1])
precision

accuracy <- sum(diag(tab))/sum(tab)
accuracy

# true Negative rate
specificity1 <- tab1[2, 2]/sum(tab1[2, ])
specificity1
# true positive rate
sensitivity1 <- tab1[1, 1]/sum(tab1[1, ])
sensitivity1
#precision
accuracy1 <- sum(diag(tab1))/sum(tab1)
accuracy1
#precision
precision1<-tab1[1,1]/sum(tab1[,1])
precision1

# Accuracy can often be a misleading metric, when one category accours more then other in the given data set
#Automated Computation through Caret
#Evaluation metrics for classification can be accessed through the "confusionMatrix()" function from the caret package
library(caret)
#Tuning the model
#Check the model summary to check for any insignificant variables
summary(data1_log)
#Use vif to find any multi-collinearity
library(car)
#Multi-colinearity check
#Variance inflation factor
#Improve the model using stepAIC

##PCA is mainly used for dimensionality reduction and so we cannot consider PCA for this problem
# PCA is performed only on predictors

# We do not perform regularization as there is no over fitting and underfitting between AUC vallues of train and test data
# Naive Bayes is done only when all the predictor variables are categorial variables 
# Since here there is a mixture of Numerical and categorail Navies Byes model 
#does not work well.

## Based on AUC curves and confusion matrix, 0.35 is a balanced threshold

##Build Model by removing insignificant variables

log_reg=glm(Target~IV+A+TR+QV+SO+S1+S3+S6+S9+PI+PP+P6+RF1+RF3+RF4,
            data=train,family = binomial)
summary(log_reg) #Aic= 26703
prob_train=predict(log_reg,train, type="response")
pred_class=ifelse(prob_train>0.5,1,0)
conf_matrix=table(train$Target,pred_class)
print(conf_matrix)
##OutPut
#pred_class
#   0       1
#0  15584   592
#1  6289   1540

#ROC for Train Data
library(ROCR)
pred<-prediction(prob_train,train$Target)
pref=performance(pred,measure = "tpr",x.measure = "fpr")
plot(pref,col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
perf_auc=performance(pred,measure = "auc")
auc=perf_auc@y.values[[1]]
print(auc)
## with 0.5 auc 0.7614221


#Predictions on test data 
prob_test <- predict(log_reg, test, type = "response") 
preds_test <- ifelse(prob_test > 0.5, 1,0)
table(preds_test) 
confusionMatrix(preds_test, test$Target, positive = "1") 

#ROC for Test  Data
library(ROCR)
pred<-prediction(prob_test,test$Target)
pref<-performance(pred,measure = "tpr",x.measure = "fpr")
plot(pref,col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
perf_auc=performance(pred,measure = "auc")
auc=perf_auc@y.values[[1]]
print(auc)
## with 0.5 auc 0.7603742

#Check the model summary to check for any insignificant variables 
summary(log_reg) 

#############################################################
#Use vif to find any multi-collinearity 
library(car) 
log_reg_vif<- vif(log_reg) 
log_reg_vif 
# For S3,S6,S9 VIF is greater than 10 so for next model remove those column 
# As this column has strong multi-collinearity 

log_reg_vif=glm(Target~IV+A+TR+QV+SO+S1+PI+PP+P6+RF1+RF3+RF4,
                data=train,family = binomial)
summary(log_reg_vif)  #AIC value is 26795
prob_train=predict(log_reg_vif,train, type="response")
pred_class=ifelse(prob_train>0.5,1,0)
confusionMatrix(train$Target,pred_class)
##OutPut
#pred_class
#   0       1
#0  15593   583
#1   6327   1502 

#ROC for Train Data
library(ROCR)
pred=prediction(prob_train,train$Target)
pref=performance(pred,measure = "tpr",x.measure = "fpr")
plot(pref,col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
perf_auc=performance(pred,measure = "auc")
auc=perf_auc@y.values[[1]]
print(auc)
## with 0.5 auc 0.7583294


#Predictions on test data 
prob_test <- predict(log_reg_vif, test, type = "response") 
preds_test <- ifelse(prob_test > 0.5, 1,0)
table(preds_test) 
confusionMatrix(preds_test, test$Target, positive = "1") 
#output
# Reference
#Prediction    0    1
0     6584 2785
1     240  679

#ROC for Test  Data
library(ROCR)
pred=prediction(prob_test,test$Target)
pref=performance(pred,measure = "tpr",x.measure = "fpr")
plot(pref,col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
perf_auc=performance(pred,measure = "auc")
auc=perf_auc@y.values[[1]]
print(auc)
## with 0.5 auc 0.7569183

#Check the model summary to check for any insignificant variables 
summary(log_reg_vif) 

####################################################################
#Improve the model using stepAIC
library(MASS)
log_reg_step = stepAIC(log_reg_vif, direction = "both") #26591.05
summary(log_reg_step) #AIC 26747
prob_train=predict(log_reg_step,train, type="response")
pred_class=ifelse(prob_train>0.5,1,0)
conf_matrix=  table
confusionMatrix(train$Target,pred_class)
conf_matrix
##OutPut
#pred_class
#   0       1
#0  15577    599
#1  6327    1502 

#ROC for Train Data
library(ROCR)
pred=prediction(prob_train,train$Target)
pref=performance(pred,measure = "tpr",x.measure = "fpr")
plot(pref,col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
perf_auc=performance(pred,measure = "auc")
auc=perf_auc@y.values[[1]]
print(auc)
## with 0.5 auc  0.7590351


#Predictions on test data 
prob_test <- predict(log_reg_step, test, type = "response") 
preds_test <- ifelse(prob_test > 0.5, 1,0)
table(preds_test) 
confusionMatrix(preds_test, test$Target, positive = "1") 

#ROC for Test  Data
library(ROCR)
pred=prediction(prob_test,test$Target)
perf=performance(pred,measure = "tpr",x.measure = "fpr")
plot(perf,col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
perf_auc=performance(pred,measure = "auc")
auc=perf_auc@y.values[[1]]
print(auc)
## with 0.5 auc 0.7576596

#Check the model summary to check for any insignificant variables 
summary(log_reg_step) 




