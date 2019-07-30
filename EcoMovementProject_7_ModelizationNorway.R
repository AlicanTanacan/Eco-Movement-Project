### ------------ Eco Movement Project ------------ ###
### ------------- by Alican Tanaçan -------------- ###
### ---- Version 7: Norway Data Modelization ----- ###

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

## Random Forest Important Variables
varImp(RFmodel1)
