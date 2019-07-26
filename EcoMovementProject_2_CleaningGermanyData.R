### ------------ Eco Movement Project ------------ ###
### ------------- by Alican Tanaçan -------------- ###
### ---- Version 2: Cleaning Germany Dataset ----- ###

## In this version the data is cleaned and runned a recursive feature elimination with
## random forest algorithm to find out which variables define the dependent variable.
## Since private chargers' observations are very few, filtering them out was logical to
## find related variables. Later we are going to increase the amount of private 
## observations for modelization. If we can acquire more private charger examples to add 
## to the data, our models will become more reliable.
## Important Note: Please be careful to core selection, if your
## computer does not have 8 cores, please do NOT run the core selection
## cluster codes.

### ---- Libraries ----
if(require("pacman") == "FALSE"){
  install.packages("pacman")
}
p_load(dplyr, ggplot2, plotly, caret, corrplot, GGally,
       doParallel, tidyverse, e1071, randomForest, caTools,
       plyr, ROSE, Hmisc, vcd, polycor)

### ---- Import Dataset ----
OriginalEcoData <- read.csv("EcoGermanyData.csv", stringsAsFactors = F)

### ---- Data Exploration ----
summary(OriginalEcoData)

## Remove auto-generated column "X"
OriginalEcoData$X <- NULL

## Detect Missing Values
sum(is.na(OriginalEcoData)) # Total 19769 NA's

colSums(is.na(OriginalEcoData))
# operator_id has 183 NA's
# owner_id has 183 NA's
# opening_times has 4364 NA's
# charging_when_closed has 4536 NA's
# additional_geo_location has 5652 NA's
# powertype has 832 NA's
# street has 15 NA's
# administrative_area_1 has 3334 NA's
# administrative_area_2 has 670 NA's

## Amount of Observations in each level of y
OriginalEcoData %>% 
  group_by(public_access_type_id) %>% 
  summarise(count(public_access_type_id))
# 7994 Public
# 195 Private
# 2712 Company

### ---- Initial Data Visualizations ----
## Map of Germany, drawn with chargers
OriginalEcoData %>% 
  mutate(Charger_Type = case_when(public_access_type_id %in% "1" ~ "Public",
                                  public_access_type_id %in% "2" ~ "Private",
                                  public_access_type_id %in% "3" ~ "Company" )) %>% 
  plot_ly(x = ~lng,
          y = ~lat,
          color = ~as.factor(Charger_Type), 
          colors = c("black", "darkorange", "red")) %>%
  add_markers() %>%
  layout(title = "Charger Type Locations in Germany")

### ---- Preprocessing ----
## Save Original Dataset
EcoData <- OriginalEcoData

## Missing Value Treatment
EcoData[is.na(EcoData)] <- 0
sum(is.na(EcoData)) # No NA's

## Change Data Types
str(EcoData)

EcoData %>% 
  mutate_at(c("masterlocation_id",
              "charging_spots_id",
              "chargingstation_id",
              "operator_id",
              "owner_id",
              "public_access_type_id",
              "connector_types_id",
              "physical_type",
              "opening_times",
              "charging_when_closed",
              "realtime_status_available",
              "spot_status",
              "capability_remote_start_stop_capable",
              "capability_rfid_reader",
              "src_id",
              "connector_format",
              "powertype",
              "name",
              "published",
              "point",
              "street",
              "postal_code",
              "city",
              "administrative_area_1",
              "administrative_area_2"), as.factor) -> EcoData

### ---- Feature Engineering ----

## Check levels of opening_times
levels(EcoData$opening_times)

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

## Remove Variables above 53 factor levels and Identifiers for Recursive Feature Elimination
EcoData %>%
  select(-masterlocation_id,
         -charging_spots_id,
         -chargingstation_id,
         -operator_id,
         -owner_id,
         -additional_geo_location,
         -opening_times,
         -src_id,
         -powertype,
         -connector_types_id,
         -name,
         -point,
         -street,
         -postal_code,
         -city,
         -administrative_area_1,
         -administrative_area_2) -> ReadyEcoData

## Move Dependent Variable to end
ReadyEcoData %>% 
  select(-public_access_type_id, public_access_type_id) -> ReadyEcoData

### ---- Correlation & Visualizations  ----

## Correlation Matrix 
# rcorr(ReadyEcoData, type = c("pearson","spearman"))
# corrplot(ReadyEcoData) # vcd & polycor package can be used

## Open Hours to y
plot(ReadyEcoData$Open_Hours,
     ReadyEcoData$public_access_type_id) # Explanatory

## Physical Type to y
plot(ReadyEcoData$physical_type,
     ReadyEcoData$public_access_type_id) # Perhaps

## Voltage & Amperage to y
plot(ReadyEcoData$voltage, ReadyEcoData$amperage)

plot(ReadyEcoData$voltage,
     ReadyEcoData$public_access_type_id) # Perhaps

plot(ReadyEcoData$amperage,
     ReadyEcoData$public_access_type_id) # Explanatory

## Other Variables to y
plot(ReadyEcoData$charging_when_closed,
     ReadyEcoData$public_access_type_id) # Explanatory

plot(ReadyEcoData$realtime_status_available,
     ReadyEcoData$public_access_type_id) # Perhaps

plot(ReadyEcoData$spot_status,
     ReadyEcoData$public_access_type_id) # Perhaps

plot(ReadyEcoData$capability_remote_start_stop_capable,
     ReadyEcoData$public_access_type_id) # Perhaps

plot(ReadyEcoData$capability_rfid_reader,
     ReadyEcoData$public_access_type_id) # Perhaps

plot(ReadyEcoData$connector_format,
     ReadyEcoData$public_access_type_id) # Perhaps

plot(ReadyEcoData$power,
     ReadyEcoData$public_access_type_id) # Explanatory (high levels of power might point to public charger)

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

## Take a random sample from public that is equal to amount of company observations
set.seed(333)
Public_Sample <- Data_Public[sample(1:nrow(Data_Public), 1864, replace = F),]

## Merge both public and company subsets. (we do not include private yet)
EcoData_Sample <- rbind(Public_Sample,
                        Data_Company)

## Change y data type to factor for data partition
EcoData_Sample$public_access_type_id <- as.factor(EcoData_Sample$public_access_type_id)

str(EcoData_Sample)

### ---- Data Partition ----
intrain <- createDataPartition(y = EcoData_Sample$public_access_type_id, 
                               p = 0.7, 
                               list = FALSE)

EcoData_Train <- EcoData_Sample[intrain,]
EcoData_Test <- EcoData_Sample[-intrain,]

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

## Confirm how many cores are now assigned to R & RStudio
getDoParWorkers() # Result = 4

### ---- Recursive Feature Elimination ----
## Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

## Use rfe and omit the response variable
rfeResults <- rfe(EcoData_Train[,1:14], 
                  EcoData_Train$public_access_type_id, 
                  sizes = (1:14), 
                  rfeControl = ctrl)

## Get results
rfeResults

## Plot results
plot(rfeResults, type=c("g", "o"))

## Most Important Variables
varImp(rfeResults)
#                Overall
# lat           38.88346
# lng           32.62864
# Open_Hours    23.32273
# power         20.99297
# physical_type 20.84720

## Create new data set with rfe recommended features
EcoDataRFE <- EcoData[,predictors(rfeResults)]

## Add the dependent variable to EcoDataRFE
EcoDataRFE$public_access_type_id <- EcoData$public_access_type_id

## Review outcome dataset
str(EcoDataRFE)

## Predictions on Binary Level Test Data
predRFE <- predict(rfeResults, EcoData_Test)

## Model Metrics
postResample(predRFE, EcoData_Test$public_access_type_id) -> RFEmodelmetrics
RFEmodelmetrics
# Accuracy: 0.960
# Kappa: 0.921

## Confusion Matrix
RFEConfMat <- confusionMatrix(predRFE$pred, EcoData_Test$public_access_type_id)
RFEConfMat 
#           Reference
# Prediction   1   3
#          1 541  26
#          3  18 533

## Stop Cluster
stopCluster(cl)