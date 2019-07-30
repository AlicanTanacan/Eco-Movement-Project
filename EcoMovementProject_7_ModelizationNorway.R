### ------------ Eco Movement Project ------------ ###
### ------------- by Alican Tanaçan -------------- ###
### ---- Version 7: Norway Data Modelization ----- ###

## Important Note: Please be careful to core selection, if your
## computer does not have 8 cores, please do NOT run the core selection
## cluster codes.

### ---- Libraries ---- 
if(require("pacman") == "FALSE"){
  install.packages("pacman")
}
p_load(dplyr, ggplot2, plotly, caret, corrplot, GGally,
       doParallel, tidyverse, e1071, randomForest, caTools,
       plyr, ROSE, Hmisc, vcd, polycor, gbm)

### --- Import Clean Norway Data ----
CleanNorwayData <- readRDS(file = "CleanNorwayData.rds")

### ---- Data Partition ----
## Check how many observations are there in Dependent Variable
CleanNorwayData %>% 
  group_by(Access_Type) %>% 
  summarise(count(Access_Type))
# Public 2454
# Private 451
# Company 809

intrain <- createDataPartition(y = CleanNorwayData$Access_Type, 
                               p = 0.7, 
                               list = FALSE)

NorwayData_Train <- CleanNorwayData[intrain,]
NorwayData_Test <- CleanNorwayData[-intrain,]

NorwayData_Train %>% 
  group_by(Access_Type) %>% 
  summarise(count(Access_Type))
# Public 1718
# Private 316
# Company 567

### ---- Core Selection ----
## Find how many cores are on your machine
detectCores() # Result = 8

## Create cluster with desired number of cores. If you have less than 8 cores do NOT run this code!
cl <- makeCluster(4)

## Register cluster
registerDoParallel(cl)

## Confirm how many cores are now assigned to R & RStudio
getDoParWorkers() # Result = 4

### ---- RF Modelization (Random Forest) ----
set.seed(4567)
## Set Train Control
RFtrctrl <- trainControl(method = "repeatedcv",
                         number = 5,
                         repeats = 2,
                         verboseIter = TRUE)

RFmodel1 <- train(Access_Type ~ ., 
                  NorwayData_Train,
                  method = "rf",
                  trControl = RFtrctrl)

RFmodel1

plot(RFmodel1)

## Most Important Variables
plot(varImp(RFmodel1))

## Predicton on Test set
predRFmodel1 <- predict(RFmodel1, NorwayData_Test)

postResample(predRFmodel1, NorwayData_Test$Access_Type) -> RFmodel1metrics

RFmodel1metrics

## Confusion Matrix
RFConfMat <- confusionMatrix(predRFmodel1, NorwayData_Test$Access_Type) 
RFConfMat
#                   Reference
# Prediction Company Private Public
# Company     136      34     37
# Private      25      58     14
# Public       81      43    685

# Accuracy: 0.789
# Kappa: 0.551

### ---- C5.0 Modelization (Decision Tree) ----
set.seed(4568)
## Set Train Control
C50trctrl <- trainControl(method = "repeatedcv", 
                          number = 5, 
                          repeats = 2,
                          preProc = c("center", "scale"),
                          verboseIter = TRUE)

C50model1 <- train(Access_Type~.,
                   data = NorwayData_Train,
                   method = "C5.0",
                   trControl = C50trctrl)

C50model1

plot(C50model1)

## Most Important Variables
plot(varImp(C50model1))

## Predicton on Test set
predC50model1 <- predict(C50model1, NorwayData_Test)

postResample(predC50model1, NorwayData_Test$Access_Type) -> C50model1metrics

C50model1metrics

## Confusion Matrix
C50ConfMat <- confusionMatrix(predC50model1, NorwayData_Test$Access_Type) 
C50ConfMat
#                   Reference
# Prediction Company Private Public
# Company     126      17     32
# Private      12      36      3
# Public      104      82    701

# Accuracy: 0.775
# Kappa: 0.481

### ---- GBM Modelization (Gradiant Boosted Machine) ----
set.seed(4569)
## Set Train Control
GBMtrcntrl <- trainControl(method = "repeatedcv", 
                           number = 5, 
                           repeats = 2)

GBMmodel1 <- train(Access_Type ~ ., 
                   NorwayData_Train,
                   method = "kknn",
                   trControl = GBMtrcntrl)

GBMmodel1

## Most Important Variables
plot(varImp(GBMmodel1))

## Predicton on Test set
predGBMmodel1 <- predict(GBMmodel1, NorwayData_Test)

postResample(predGBMmodel1, NorwayData_Test$Access_Type) -> GBMmodel1metrics

GBMmodel1metrics

## Confusion Matrix
GBMConfMat <- confusionMatrix(predGBMmodel1, NorwayData_Test$Access_Type) 
GBMConfMat
#                   Reference
# Prediction Company Private Public
# Company     149      35     47
# Private      28      55     19
# Public       65      45    670

# Accuracy: 0.785
# Kappa: 0.552

### ---- kNN Modelization (k-Nearest Neighbor) ----
set.seed(4570)
## Set Train Control
kNNcontrol <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 2,
                           preProc = c("center", "scale"))

kNNmodel1 <- train(Access_Type ~ ., 
                   NorwayData_Train,
                   method = "kknn",
                   trControl = kNNcontrol)

kNNmodel1

## Most Important Variables
plot(varImp(kNNmodel1))

## Predicton on Test set
predkNNmodel1 <- predict(kNNmodel1, NorwayData_Test)

postResample(predkNNmodel1, NorwayData_Test$Access_Type) -> kNNmodel1metrics

kNNmodel1metrics

## Confusion Matrix
kNNConfMat <- confusionMatrix(predkNNmodel1, NorwayData_Test$Access_Type) 
kNNConfMat
#                   Reference
# Prediction Company Private Public
# Company     149      35     47
# Private      28      55     19
# Public       65      45    670

# Accuracy: 0.785
# Kappa: 0.552

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
#    RFmodel1metrics 0.7897574 0.5510828 ***
#   GBMmodel1metrics 0.7852650 0.5529422
#   kNNmodel1metrics 0.7852650 0.5529422
#   C50model1metrics 0.7753819 0.4815559

# *** : Random Forest model performs the best! Let's improve it..

## Random Forest Important Variables
varImp(RFmodel1)
# Open_Hours
# physical_type
# connector_format
# power
# capability_rfid_reader
# powertype
# charging_when_closed
# lat
# lng
# amperage