### ------------ Eco Movement Project ------------ ###
### ------------- by Alican Tanaçan -------------- ###
### ----- Version 4: Improving Modelization ------ ###

## In this version, we try to improve the performance metrics of
## our random forest model.
## Important Note: Please correct the path on source before running!
## Source function should get the version 1 of our analysis.
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

## Take merged data from version 1 with source
source(file = "D:/RStudio/R Studio Working Directory/Eco Movement/EcoMovementProject_1_MergingGermanyData.R")

### ---- Import Dataset ----
OriginalEcoData <- read.csv("EcoGermanyData.csv", stringsAsFactors = F)

### ---- Preprocessing & Feature Selection ----
## Remove auto-generated column "X"
OriginalEcoData$X <- NULL

## Save Original Dataset
EcoData <- OriginalEcoData

## Missing Value Treatment
EcoData[is.na(EcoData)] <- 0
sum(is.na(EcoData)) # No NA's

## Create a new variable that depicts open hours of chargers
EcoData %>% 
  mutate(
    Open_Hours = 
      if_else(opening_times %in% c("{\"twentyfourseven\": true }",
                                   "{\"twentyfourseven\":true}"), "Everytime",
              if_else(opening_times %in% "0", "Unknown",
                      "Regular"))) -> EcoData

## Change variable type
EcoData$Open_Hours <- as.factor(EcoData$Open_Hours)

## Remove Unknown Open Hours 
EcoData %>% 
  filter(Open_Hours != "Unknown") -> EcoData

## Select The Best Predictors for Germany Data Access Type
EcoData %>%
  select(power,
         Open_Hours,
         lat,
         lng,
         capability_remote_start_stop_capable,
         public_access_type_id) -> ReadyEcoData

## Move Dependent Variable to end
ReadyEcoData %>% 
  select(-public_access_type_id, public_access_type_id) -> ReadyEcoData

## Remove Duplicates
unique(ReadyEcoData) -> ReadyEcoData

### ---- Sampling  ----
## Amount of Observations in each level of y
ReadyEcoData %>% 
  group_by(public_access_type_id) %>% 
  summarise(count(public_access_type_id))
# 4596 Public
# 77 Private
# 1864 Company

## Change y data type to character in order to subset without empty classes
ReadyEcoData$public_access_type_id <- as.character(ReadyEcoData$public_access_type_id)

## Divide the data into 3 subsets for the levels of y
ReadyEcoData %>% 
  filter(public_access_type_id == 1) -> Data_Public

ReadyEcoData %>% 
  filter(public_access_type_id == 2) -> Data_Private

ReadyEcoData %>% 
  filter(public_access_type_id == 3) -> Data_Company

## combining private with company
EcoData_Sample_Priv_Comp <- rbind(Data_Private,
                                 Data_Company)

## Over Sampling the Private Company Data
set.seed(200)
EcoData_Sample_PrivCompOver <- ovun.sample(public_access_type_id~., 
                                           data = EcoData_Sample_Priv_Comp,
                                           p = 0.5,
                                           seed = 1, 
                                           method = "over")$data

## Take a random sample from public that is equal to amount of company observations
Public_Sample2 <- Data_Public[sample(1:nrow(Data_Public), 1864, replace = F),]

## Merge both public and company subsets.
EcoData_Sample2 <- rbind(Public_Sample2,
                         EcoData_Sample_PrivCompOver)

## Change y data type to factor for data partition
EcoData_Sample2$public_access_type_id <- as.factor(EcoData_Sample2$public_access_type_id)
EcoData_Sample2$capability_remote_start_stop_capable <- as.factor(EcoData_Sample2$capability_remote_start_stop_capable)

### ---- Data Partition ----
intrain <- createDataPartition(y = EcoData_Sample2$public_access_type_id, 
                               p = 0.7, 
                               list = FALSE)

EcoData_Train <- EcoData_Sample2[intrain,]
EcoData_Test <- EcoData_Sample2[-intrain,]

EcoData_Train %>% 
  group_by(public_access_type_id) %>% 
  summarise(count(public_access_type_id))

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

RFmodel <- train(public_access_type_id ~ ., 
                  EcoData_Train,
                  method = "rf",
                  trControl = RFtrctrl)

RFmodel

plot(RFmodel)

## Most Important Variables
plot(varImp(RFmodel))
# power
# Open_Hours (Regular)
# lat
# lng
# capability_remote_start_stop_capable (Y)

## Predicton on Test set
predRFmodel <- predict(RFmodel, EcoData_Test)

postResample(predRFmodel, EcoData_Test$public_access_type_id) -> RFmodelmetrics

RFmodelmetrics

## Confusion Matrix
RFConfMat <- confusionMatrix(predRFmodel, EcoData_Test$public_access_type_id) 
RFConfMat
#              Reference
# Prediction   1   2   3
#          1 531   0  12
#          2   1 544   4
#          3  27   0 543

# Accuracy: 0.973
# Kappa: 0.960

## This model is quite good to predict access types in Germany and only Germany.
## However, we must not forget how e generated extra private data and the amount
## of duplicates among the train dataset.

## Save the model to disk
save(RFmodel, file = "EcoGermanyRFModel.rda")

## Stop Cluster
stopCluster(cl)