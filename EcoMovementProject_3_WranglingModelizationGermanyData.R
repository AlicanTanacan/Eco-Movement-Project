### ------------ Eco Movement Project ------------ ###
### ------------- by Alican Tanaçan -------------- ###
### -- Version 3: Data Wrangling & Modelization -- ###

## In this version I try increase the amount of 'Private' observations by 
## over sampling the dependent variable. 
## It is important to remember that every private observation created this
## way is artificial, not REAL!
## As we input more private data into the dataset, our confidence for our
## models will surely increase, although we might experience a decrease
## in the performance metrics.
## Important Note: Please be careful to core selection, if your
## computer does not have 8 cores, please do NOT run the core selection
## cluster codes.

### ---- Libraries & Source ---- 
if(require("pacman") == "FALSE"){
  install.packages("pacman")
}
p_load(dplyr, ggplot2, plotly, caret, corrplot, GGally,
       doParallel, tidyverse, e1071, randomForest, caTools,
       plyr, ROSE, Hmisc, vcd, polycor, gbm)

## Take data & models from previous version 2 with source
source(file = "D:/RStudio/R Studio Working Directory/Eco Movement/EcoMovementProject_2_CleaningGermanyData.R")

### ---- Data Munging & Augmenting ----
## combining private with company
EcoData_Sample_PrivComp <- rbind(Data_Private,
                                 Data_Company)

EcoData_Sample_PrivComp %>% 
  group_by(public_access_type_id) %>% 
  summarise(count(public_access_type_id))

## Over Sampling the Private Company Data
set.seed(200)
EcoData_Sample_PrivCompOver <- ovun.sample(public_access_type_id~., 
                                           data = EcoData_Sample_PrivComp,
                                           p = 0.5,
                                           seed = 1, 
                                           method = "over")$data

EcoData_Sample_PrivCompOver %>% 
  group_by(public_access_type_id) %>% 
  summarise(count(public_access_type_id))

## Take a random sample from public that is equal to amount of company and private observations
set.seed(400)
Public_Sample <- Data_Public[sample(1:nrow(Data_Public), 1864, replace = F),]

## Merge public with private company over sampled subset
EcoData_FullSample1 <- rbind(Public_Sample,
                             EcoData_Sample_PrivCompOver)

## Change y data type to factor for data partition
EcoData_FullSample1$public_access_type_id <- as.factor(EcoData_FullSample1$public_access_type_id)

### ---- Feature Selection ----
EcoData_FullSample1 %>% 
  select(physical_type,
         power,
         Open_Hours,
         lat,
         lng,
         public_access_type_id) -> EcoGermanyData_SlctSample

### ---- Data Partition ----
intrain <- createDataPartition(y = EcoGermanyData_SlctSample$public_access_type_id, 
                               p = 0.7, 
                               list = FALSE)

EcoData_Train2 <- EcoGermanyData_SlctSample[intrain,]
EcoData_Test2 <- EcoGermanyData_SlctSample[-intrain,]

EcoData_Train2 %>% 
  group_by(public_access_type_id) %>% 
  summarise(count(public_access_type_id))
# Public: 1305
# Private: 1271
# Company: 1305

### ---- Core Selection ----
## Find how many cores are on your machine
detectCores() # Result = 8

## Create cluster with desired number of cores. If you have less than 8 cores do NOT run this code!
cl <- makeCluster(4)

## Register cluster
registerDoParallel(cl)

### ---- RF Modelization (Random Forest) ----
set.seed(4567)
## Set Train Control
RFtrctrl <- trainControl(method = "repeatedcv",
                         number = 5,
                         repeats = 2,
                         verboseIter = TRUE)

RFmodel1 <- train(public_access_type_id ~ ., 
                  EcoData_Train2,
                  method = "rf",
                  trControl = RFtrctrl)

RFmodel1

plot(RFmodel1)

## Most Important Variables
plot(varImp(RFmodel1))
# power
# Open_Hours (Regular)
# lat
# lng
# capability_remote_start_stop_capable (Y)

## Predicton on Test set
predRFmodel1 <- predict(RFmodel1, EcoData_Test2)

postResample(predRFmodel1, EcoData_Test2$public_access_type_id) -> RFmodel1metrics

RFmodel1metrics

## Confusion Matrix
RFConfMat <- confusionMatrix(predRFmodel1, EcoData_Test2$public_access_type_id) 
RFConfMat
#              Reference
# Prediction   1   2   3
#          1 535   0  13
#          2   1 544  11
#          3  23   0 535

# Accuracy: 0.971
# Kappa: 0.956

### ---- C5.0 Modelization (Decision Tree) ----
set.seed(4568)
## Set Train Control
C50trctrl <- trainControl(method = "repeatedcv", 
                          number = 5, 
                          repeats = 2,
                          preProc = c("center", "scale"),
                          verboseIter = TRUE)

C50model1 <- train(public_access_type_id~.,
                   data = EcoData_Train2,
                   method = "C5.0",
                   trControl = C50trctrl)

C50model1

plot(C50model1)

## Most Important Variables
plot(varImp(C50model1))

## Predicton on Test set
predC50model1 <- predict(C50model1, EcoData_Test2)

postResample(predC50model1, EcoData_Test2$public_access_type_id) -> C50model1metrics

C50model1metrics

## Confusion Matrix
C50ConfMat <- confusionMatrix(predC50model1, EcoData_Test2$public_access_type_id) 
C50ConfMat
#              Reference
# Prediction   1   2   3
#          1 518   0  18
#          2   4 544  17
#          3  37   0 524

# Accuracy: 0.954
# Kappa: 0.931

### ---- GBM Modelization (Gradiant Boosted Machine) ----
set.seed(4569)
## Set Train Control
GBMtrcntrl <- trainControl(method = "repeatedcv", 
                           number = 5, 
                           repeats = 2)

GBMmodel1 <- train(public_access_type_id ~ ., 
                   EcoData_Train2,
                   method = "kknn",
                   trControl = GBMtrcntrl)

GBMmodel1

## Most Important Variables
plot(varImp(GBMmodel1))

## Predicton on Test set
predGBMmodel1 <- predict(GBMmodel1, EcoData_Test2)

postResample(predGBMmodel1, EcoData_Test2$public_access_type_id) -> GBMmodel1metrics

GBMmodel1metrics

## Confusion Matrix
GBMConfMat <- confusionMatrix(predGBMmodel1, EcoData_Test2$public_access_type_id) 
GBMConfMat
#              Reference
# Prediction   1   2   3
#          1 531   0  18
#          2   3 544   6
#          3  25   0 535

# Accuracy: 0.968
# Kappa: 0.935

### ---- kNN Modelization (k-Nearest Neighbor) ----
set.seed(4570)
## Set Train Control
kNNcontrol <- trainControl(method = "repeatedcv",
                            number = 5,
                            repeats = 2,
                            preProc = c("center", "scale"))

kNNmodel1 <- train(public_access_type_id ~ ., 
                   EcoData_Train2,
                   method = "kknn",
                   trControl = kNNcontrol)

kNNmodel1

## Most Important Variables
plot(varImp(kNNmodel1))

## Predicton on Test set
predkNNmodel1 <- predict(kNNmodel1, EcoData_Test2)

postResample(predkNNmodel1, EcoData_Test2$public_access_type_id) -> kNNmodel1metrics

kNNmodel1metrics

## Confusion Matrix
kNNConfMat <- confusionMatrix(predkNNmodel1, EcoData_Test2$public_access_type_id) 
kNNConfMat
#              Reference
# Prediction   1   2   3
#          1 531   0  18
#          2   3 544   6
#          3  25   0 522

# Accuracy: 0.968
# Kappa: 0.953

## Stop Cluster
stopCluster(cl)


### ---- Model Comparison ----
## Creating data frames for performance and accuracy metrics
AccuracyMetrics <- data.frame(RFmodel1metrics, 
                              C50model1metrics,
                              GBMmodel1metrics,
                              kNNmodel1metrics)

## Transposing the data frame
AccuracyMetrics <- data.frame(t(AccuracyMetrics))

## Naming the rows
AccuracyMetrics$Algorithms <- rownames(AccuracyMetrics)

## Reordering the columns
AccuracyMetrics <- AccuracyMetrics[, c(3,1,2)]

## Arranging by Accuracy to see the best models
AccuracyMetrics %>% 
  arrange(desc(Accuracy))

#         Algorithms  Accuracy     Kappa
#    RFmodel1metrics 0.9711191 0.9566794 ***
#   GBMmodel1metrics 0.9687124 0.9530682
#   kNNmodel1metrics 0.9687124 0.9530682
#   C50model1metrics 0.9542720 0.9314141

# *** : Random Forest model performs the best! Let's improve it..