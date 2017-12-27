---
title: 'Prediction Assignment'
author: "Tong Tat"
date: "December 25, 2017"
output:
  html_document:
    highlight: tango
    keep_md: yes
    theme: yeti
  pdf_document: default
---

**Rpub Link:** [Click Here]()

## 1. Introduction

This report is for Coursera Practical Machine Learning Course's Final Project. 

The data is from this source:
[http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har.)

Basically the data is collected from sensor device such as *Jawbone Up*, *Nike FuelBand*, and *Fitbit* which are attached on belt, forearm, arm, and dumbell of 6 participants.  They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

The objective of this project is to predict the manner in which the participants did the exercise. The variable which I am predicting is called "classe". 

Our outcome variable "classe" is a factor variable with 5 levels. For this dataset, "participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in 5 different fashions:

1) exactly according to the specification (Class A)

2) throwing the elbows to the front (Class B)

3) lifting the dumbbell only halfway (Class C)

4) lowering the dumbbell only halfway (Class D)

5) throwing the hips to the front (Class E)

The report will touch on how the model is built, cross validation, out of sample error and predict the outcome for 20 different test subjects. 


## 2. Load Dataset


```r
# Load dataset. Note "" and "NA" are actually na.string for this dataset
train <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", header=TRUE, stringsAsFactors = TRUE, na.strings = c("","NA"))
test <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", header=TRUE, stringsAsFactors = TRUE, na.strings = c("","NA"))
```

## 3. Data Exploration

First, let's take a look at the summary and look for any NA.
Missing dependencies check will be done to look for any missing package that require installation. 


```r
# Check for missing dependencies and load necessary R packages
if(!require(caret)){install.packages('caret')}; library(caret)
if(!require(rattle)){install.packages('rattle')}; library(rattle)
if(!require(randomForest)){install.packages('randomForest')}; library(randomForest)
if(!require(MASS)){install.packages('MASS')}; library(MASS)
if(!require(ggplot2)){install.packages('ggplot2')}; library(ggplot2)


# Check summary of Train & Test data
# summary(train); str(train); head(train); summary(test); str(test); head(test)  

# Check NA for each columns in Train
#sapply(train,function(x) sum(is.na(x)))
```


# 4. Data Cleaning

Noticed there are lots of columns with NA values. Removing these columns and the columns with time/date.


```r
# Remove NA columns for Train
train2 <- train[ , apply(train, 2, function(x) !any(is.na(x)))]

# Remove unnecessary columns
train2 <- train2[,8:60]
```


Separate data the train data into 60% for training the model and 40% for testing the model. The model with the lowest MSE and highest AUC will be used to predict the final outcome for the 20 different test subjects.



```r
# Create Index for training
IndexTrain <- createDataPartition(y=train2$classe, p=0.6, list=FALSE)
training <- train2[IndexTrain,]
testing <- train2[-IndexTrain,]
```


# 5. Model Building

## 5.1. Decision Tree

Using the train function in the caret package, we set **method="rpart"** and train the Decision Tree model with the training data.


```r
# Train Tree Model
tree1 <- train(classe~., method="rpart", data=training)
tree1$finalModel
```

```
## n= 11776 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 11776 8428 A (0.28 0.19 0.17 0.16 0.18)  
##    2) roll_belt< 130.5 10783 7438 A (0.31 0.21 0.19 0.18 0.11)  
##      4) pitch_forearm< -33.95 936    8 A (0.99 0.0085 0 0 0) *
##      5) pitch_forearm>=-33.95 9847 7430 A (0.25 0.23 0.21 0.2 0.12)  
##       10) magnet_dumbbell_y< 436.5 8298 5936 A (0.28 0.18 0.24 0.19 0.11)  
##         20) roll_forearm< 121.5 5156 3037 A (0.41 0.18 0.19 0.16 0.062) *
##         21) roll_forearm>=121.5 3142 2131 C (0.077 0.18 0.32 0.24 0.18) *
##       11) magnet_dumbbell_y>=436.5 1549  771 B (0.036 0.5 0.043 0.23 0.19) *
##    3) roll_belt>=130.5 993    3 E (0.003 0 0 0 1) *
```

```r
# Plot Tree Model
fancyRpartPlot(tree1$finalModel, tweak=1.5)
```

![](Prediction_Assignment_files/figure-html/tree-1.png)<!-- -->

```r
# Predictions using Testing dataset
tree.pred <- predict(tree1, newdata = testing)

# ConfusionMatrix for Tree Model
tree.confuse <- confusionMatrix(tree.pred, testing$classe)
tree.confuse
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2010  619  607  594  197
##          B   40  524   42  224  202
##          C  171  375  719  468  402
##          D    0    0    0    0    0
##          E   11    0    0    0  641
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4963          
##                  95% CI : (0.4852, 0.5074)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3423          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9005  0.34519  0.52558   0.0000   0.4445
## Specificity            0.6407  0.91972  0.78141   1.0000   0.9983
## Pos Pred Value         0.4991  0.50775  0.33677      NaN   0.9831
## Neg Pred Value         0.9419  0.85412  0.88636   0.8361   0.8887
## Prevalence             0.2845  0.19347  0.17436   0.1639   0.1838
## Detection Rate         0.2562  0.06679  0.09164   0.0000   0.0817
## Detection Prevalence   0.5133  0.13153  0.27211   0.0000   0.0831
## Balanced Accuracy      0.7706  0.63246  0.65350   0.5000   0.7214
```

Based on the confusionMatrix, we can see the accuracy for Decision Tree Model is **0.4963038**.


## 5.2. Random Forest

For Random Forest model, manual tuning was done to find the optimal mtry which will be used to train the final model. The optimal mtry will have the lowest out-of-bag error.


```r
# Manually tune for Optimal mtry
mse.rfs <- rep(0, 13)
for(m in 1:13){
    set.seed(123)
    rf <- randomForest(classe ~ ., data=training, mtry=m)
    mse.rfs[m] <- rf$err.rate[500]  
}

# Plot OOB Error for each mtry
plot(1:13, mse.rfs, type="b", xlab="mtry", ylab="OOB Error")
```

![](Prediction_Assignment_files/figure-html/rf-1.png)<!-- -->

```r
mse.rfs
```

```
##  [1] 0.011294158 0.008152174 0.007387908 0.006963315 0.006623641
##  [6] 0.006283967 0.006453804 0.006283967 0.006368886 0.006878397
## [11] 0.006623641 0.006963315 0.006453804
```

```r
optimal.mtry <- which.min(mse.rfs)


# Train randomForest Model with optimal mtry
rf1 <- randomForest(classe~., data=training, mtry=optimal.mtry)
rf1
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training, mtry = optimal.mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 6
## 
##         OOB estimate of  error rate: 0.7%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3346    2    0    0    0 0.0005973716
## B   13 2258    8    0    0 0.0092145678
## C    0   16 2038    0    0 0.0077896787
## D    0    0   34 1893    3 0.0191709845
## E    0    0    3    4 2158 0.0032332564
```

```r
# Predictions using Testing dataset
rf.pred <- predict(rf1, newdata = testing)

# ConfusionMatrix for Tree Model
rf.confuse <- confusionMatrix(rf.pred, testing$classe)
rf.confuse
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232   11    0    0    0
##          B    0 1503   10    0    1
##          C    0    4 1358   18    2
##          D    0    0    0 1267    4
##          E    0    0    0    1 1435
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9935          
##                  95% CI : (0.9915, 0.9952)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9918          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9901   0.9927   0.9852   0.9951
## Specificity            0.9980   0.9983   0.9963   0.9994   0.9998
## Pos Pred Value         0.9951   0.9927   0.9826   0.9969   0.9993
## Neg Pred Value         1.0000   0.9976   0.9985   0.9971   0.9989
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1916   0.1731   0.1615   0.1829
## Detection Prevalence   0.2859   0.1930   0.1761   0.1620   0.1830
## Balanced Accuracy      0.9990   0.9942   0.9945   0.9923   0.9975
```


Based on the confusionMatrix, we can see the accuracy for Random Forest Model is **0.9934999**.


## 5.3. Gradient Boosting Model

For Gradient Boosting Model, train function in the caret package waas used and set **method="gbm"**. Verbose=FALSE is to surpress all the messages.


```r
# Train Tree Model
gbm <- train(classe~., method="gbm", data=training, verbose=FALSE)
gbm$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 52 predictors of which 43 had non-zero influence.
```

```r
# Predictions using Testing dataset
gbm.pred <- predict(gbm, newdata = testing)

# ConfusionMatrix for Tree Model
gbm.confuse <- confusionMatrix(gbm.pred, testing$classe)
gbm.confuse
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2182   40    0    1    3
##          B   28 1426   52   11   21
##          C    6   47 1302   42   14
##          D    5    1   11 1219   18
##          E   11    4    3   13 1386
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9578          
##                  95% CI : (0.9531, 0.9622)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9466          
##  Mcnemar's Test P-Value : 1.836e-09       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9776   0.9394   0.9518   0.9479   0.9612
## Specificity            0.9922   0.9823   0.9832   0.9947   0.9952
## Pos Pred Value         0.9802   0.9272   0.9227   0.9721   0.9781
## Neg Pred Value         0.9911   0.9854   0.9897   0.9898   0.9913
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2781   0.1817   0.1659   0.1554   0.1767
## Detection Prevalence   0.2837   0.1960   0.1798   0.1598   0.1806
## Balanced Accuracy      0.9849   0.9608   0.9675   0.9713   0.9782
```


Based on the confusionMatrix, we can see the accuracy for Gradient Boosting Model is **0.9578129**.


# 6. Model Selection

Based on the summary table below, we can see the model with best accuracy is the **Random Forest Model**. This model will be used to predict the final class for the 20 subjects in the test data.


```r
# Create table for comparison of Accuracy
table1 <- data.frame(
  Model=c("Random Forest","Gradient Boosting", "Decision Tree"),
  Accuracy=c(rf.confuse$overall[[1]],gbm.confuse$overall[[1]],tree.confuse$overall[[1]]),
 "ConfInv 95 Lower"=c(rf.confuse$overall[[3]],gbm.confuse$overall[[3]],tree.confuse$overall[[3]]),
 "ConfInv 95 Upper"=c(rf.confuse$overall[[4]],gbm.confuse$overall[[4]],tree.confuse$overall[[4]])
 )
table1
```

```
##               Model  Accuracy ConfInv.95.Lower ConfInv.95.Upper
## 1     Random Forest 0.9934999        0.9914623        0.9951565
## 2 Gradient Boosting 0.9578129        0.9531280        0.9621537
## 3     Decision Tree 0.4963038        0.4851800        0.5074305
```


# 7. Test Data Prediction

Applying the trained model from Random Forest, we can get the predicted class as shown below.

```r
# Predict outcome on the original Testing data set using Random Forest model
predictfinal <- predict(rf1, newdata=test, type="class")
predictfinal
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```




