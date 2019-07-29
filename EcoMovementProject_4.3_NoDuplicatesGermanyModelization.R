### ------------ Eco Movement Project ------------ ###
### ------------- by Alican Tanaçan -------------- ###
### ------ Version 4.3: Removing Duplicates ------ ###

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
OriginalEcoData <- read.csv("EcoData.csv", stringsAsFactors = F)

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

### Remove Duplicates 
unique(ReadyEcoData) -> ReadyEcoData2

## To remove duplicated rows according to a column(s):
# ReadyEcoData %>% distinct(lat, lng, .keep_all = TRUE) -> ReadyEcoData3

### ---- Sampling  ----
## Amount of Observations in each level of y
ReadyEcoData2 %>% 
  group_by(public_access_type_id) %>% 
  summarise(count(public_access_type_id))

## Change y data type to character in order to subset without empty classes
ReadyEcoData2$public_access_type_id <- as.character(ReadyEcoData2$public_access_type_id)

## Divide the data into 3 subsets for the levels of y
ReadyEcoData2 %>% 
  filter(public_access_type_id == 1) -> Data_Public

ReadyEcoData2 %>% 
  filter(public_access_type_id == 2) -> Data_Private

ReadyEcoData2 %>% 
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

EcoData_Sample_PrivCompOver %>% 
  group_by(public_access_type_id) %>% 
  summarise(count(public_access_type_id))

## Take a random sample from public that is equal to amount of company observations
Public_Sample2 <- Data_Public[sample(1:nrow(Data_Public), 999, replace = F),]

## Merge both public and company subsets. (we do not include private yet)
EcoData_Sample2 <- rbind(Public_Sample2,
                         EcoData_Sample_PrivCompOver)

## Change data types to factor for data partition
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

RFmodel2 <- train(public_access_type_id ~ ., 
                  EcoData_Train,
                  method = "rf",
                  trControl = RFtrctrl)

RFmodel2

plot(RFmodel2)

## Most Important Variables
plot(varImp(RFmodel2))
predictors(RFmodel2)

## Predicton on Test set
predRFmodel2 <- predict(RFmodel2, EcoData_Test)

postResample(predRFmodel2, EcoData_Test$public_access_type_id) -> RFmodel2metrics

RFmodel2metrics

## Confusion Matrix
RFConfMat2 <- confusionMatrix(predRFmodel2, EcoData_Test$public_access_type_id) 
RFConfMat2

# Accuracy: 0.914
# Kappa: 0.871

## Save the model to disk
save(RFmodel2, file = "EcoGermanyRFModel2.rda")

## Stop Cluster
stopCluster(cl)
