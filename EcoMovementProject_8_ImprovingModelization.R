### ------------ Eco Movement Project ------------ ###
### ------------- by Alican Tanaçan -------------- ###
### ----- Version 8: Improving Modelization ------ ###

### ---- Libraries ---- 
if(require("pacman") == "FALSE"){
  install.packages("pacman")
}
p_load(dplyr, ggplot2, plotly, caret, corrplot, GGally,
       doParallel, tidyverse, e1071, randomForest, caTools,
       plyr, ROSE, Hmisc, vcd, polycor, gbm)

### --- Import Clean Norway Data ----
CleanNorwayData <- readRDS(file = "CleanNorwayData.rds")

### ---- Feature Selection ----
CleanNorwayData %>% 
  select(Open_Hours,
         physical_type,
         connector_format,
         power,
         capability_rfid_reader,
         powertype,
         charging_when_closed,
         lat,
         lng,
         amperage,
         Access_Type) -> ReadyNorwayData

### ---- Data Partition ----
## Check how many observations are there in Dependent Variable
ReadyNorwayData %>% 
  group_by(Access_Type) %>% 
  summarise(count(Access_Type))

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

RFmodel2 <- train(Access_Type ~ ., 
                  NorwayData_Train,
                  method = "rf",
                  trControl = RFtrctrl)

RFmodel2

plot(RFmodel2)

## Most Important Variables
plot(varImp(RFmodel2))

## Predicton on Test set
predRFmodel2 <- predict(RFmodel2, NorwayData_Test)

postResample(predRFmodel2, NorwayData_Test$Access_Type) -> RFmodel2metrics

RFmodel2metrics

## Confusion Matrix
RFConfMat2 <- confusionMatrix(predRFmodel2, NorwayData_Test$Access_Type) 
RFConfMat2
