### ------------ Eco Movement Project ------------ ###
### ------------- by Alican Tanaçan -------------- ###
### ------- Version 6: Data Cleaning & RFE ------- ###

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

### --- Import Merged Norway Data ----
OriginalNorwayEcoData <- read.csv("EcoDataNorway.csv", stringsAsFactors = F)

### ---- Preprocessing ----
## Save Original Dataset
EcoNorwayData <- OriginalNorwayEcoData

## Remove auto-generated column "X"
EcoNorwayData$X <- NULL

## Remove duplicates from the data
unique(EcoNorwayData) -> EcoNorwayData # 0 duplicates at the moment

## Detect Missing Values
sum(is.na(EcoNorwayData)) # Total 18689 NA's
colSums(is.na(EcoNorwayData))
# operator_id has 5725 NA's
# owner_id has 8367 NA's
# opening_times has 528 NA's
# charging_when_closed has 478 NA's
# connector_format has 3585 NA's
# powertype has 6 NA's

## Missing Value Treatment
EcoNorwayData[is.na(EcoNorwayData)] <- 0 # replace NA's with 0
sum(is.na(EcoNorwayData)) # No NA's

## Change Data Types
str(EcoNorwayData)

EcoNorwayData %>% 
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
              "capability_rfid_reader",
              "src_id",
              "connector_format",
              "powertype",
              "name",
              "published",
              "point"), as.factor) -> EcoNorwayData

### ---- Feature Selection & Engineering ----
## Check levels of opening_times
levels(EcoNorwayData$opening_times)

## Create a new variable that depicts open hours of chargers
EcoNorwayData %>% 
  mutate(
    Open_Hours = 
      if_else(opening_times %in% c("{\"twentyfourseven\": true }",
                                   "{\"twentyfourseven\":true}",
                                   "{\"twentyfourseven\": true}"), "Everytime",
              if_else(opening_times %in% "0", "Unknown",
                      "Regular"))) -> EcoNorwayData

EcoNorwayData$Open_Hours <- as.factor(EcoNorwayData$Open_Hours)

## Exclude Identifiers before removing duplicates
EcoNorwayData %>% 
  select(-chargingstation_id,
         -masterlocation_id,
         -charging_spots_id,
         -operator_id,
         -owner_id,
         -src_id,
         -connector_types_id) -> EcoNorwayData2

## Remove duplicates
unique(EcoNorwayData2) -> EcoNorwayData2

## Deselect Irrelevant Variables for Recursive Feature Elimination
EcoNorwayData2 %>%
  select(-opening_times,
         -name,
         -point) -> ReadyEcoNorwayData

## Change the Dependent Variable name, levels and type
ReadyEcoNorwayData %>% 
  mutate(Access_Type = case_when(public_access_type_id %in% "1" ~ "Public",
                                 public_access_type_id %in% "2" ~ "Private",
                                 public_access_type_id %in% "3" ~ "Company" )) -> ReadyEcoNorwayData

ReadyEcoNorwayData$public_access_type_id <- NULL

ReadyEcoNorwayData$Access_Type <- as.factor(ReadyEcoNorwayData$Access_Type)

## Save the ready data for data wrangling and modelization
saveRDS(ReadyEcoNorwayData, file = "CleanNorwayData.rds")

### ---- Visualizations ----
## Latitude & Longitude vs Access Type (Map of Norway, drawn with chargers)
ReadyEcoNorwayData %>% 
  plot_ly(x = ~lng,
          y = ~lat,
          color = ~as.factor(Access_Type), 
          colors = c("black", "darkorange", "red")) %>%
  add_markers() %>%
  layout(title = "Charger Type Locations in Norway")

## Open Hours vs Access Type
ggplot(ReadyEcoNorwayData, aes(Open_Hours, Access_Type)) +
  geom_count()

## Physical Type vs Access Type
ggplot(ReadyEcoNorwayData, aes(physical_type, Access_Type)) +
  geom_count()

## Spot Status vs Access Type
ggplot(ReadyEcoNorwayData, aes(spot_status, Access_Type)) +
  geom_count()

## Connector Format vs Access Type
ggplot(ReadyEcoNorwayData, aes(connector_format, Access_Type)) +
  geom_count()

## Powertype vs Access Type
ggplot(ReadyEcoNorwayData, aes(powertype, Access_Type)) +
  geom_count()

## Published vs Access Type
ggplot(ReadyEcoNorwayData, aes(published, Access_Type)) +
  geom_count()

## Voltage vs Access Type
ggplot(ReadyEcoNorwayData, aes(Access_Type, voltage)) +
  geom_violin()

## Amperage vs Access Type
ggplot(ReadyEcoNorwayData, aes(Access_Type, amperage)) +
  geom_violin()

## Power vs Access Type
ggplot(ReadyEcoNorwayData, aes(Access_Type, power)) +
  geom_violin()

### ---- Sampling ----
## Check how many observations are there in Dependent Variable
ReadyEcoNorwayData %>% 
  group_by(Access_Type) %>% 
  summarise(count(Access_Type))
# Public 2454
# Private 451
# Company 809

## Change y data type to character in order to subset without empty classes
ReadyEcoNorwayData$Access_Type <- as.character(ReadyEcoNorwayData$Access_Type)

## Divide the data into 3 subsets for the levels of y
ReadyEcoNorwayData %>% 
  filter(Access_Type == "Public") -> Data_Public

ReadyEcoNorwayData %>% 
  filter(Access_Type == "Private") -> Data_Private

ReadyEcoNorwayData %>% 
  filter(Access_Type == "Company") -> Data_Company

## Take random sample from Public and Company, same amount of Private
Public_Sample <- Data_Public[sample(1:nrow(Data_Public), 451, replace = F),]

Company_Sample <- Data_Company[sample(1:nrow(Data_Company), 451, replace = F),]

## Join all samples into a single dataset
EcoNorwayData_Sample <- rbind(Public_Sample,
                              Company_Sample,
                              Data_Private)

EcoNorwayData_Sample$Access_Type <- as.factor(EcoNorwayData_Sample$Access_Type)

EcoNorwayData_Sample %>% 
  group_by(Access_Type) %>% 
  summarise(count(Access_Type))
# Public: 451
# Private: 451
# Company: 451

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
rfeResults <- rfe(EcoNorwayData_Sample[,1:14], 
                  EcoNorwayData_Sample$Access_Type, 
                  sizes = (1:14), 
                  rfeControl = ctrl)

## Get results
rfeResults

## Predictors
predictors(rfeResults)

## Plot results
plot(rfeResults, type=c("g", "o"))

## Most Important Variables
varImp(rfeResults)
#                           Overall
# physical_type             29.489944
# Open_Hours                25.694999
# connector_format          18.076922
# power                     15.713978
# capability_rfid_reader    15.107562
# powertype                 12.776930
# charging_when_closed      12.104839
# amperage                  10.090931

## Create new data set with rfe recommended features
EcoNorwayDataRFE <- EcoNorwayData_Sample[,predictors(rfeResults)]

## Add the dependent variable to EcoDataRFE
EcoNorwayDataRFE$Access_Type <- EcoNorwayData_Sample$Access_Type

## Review outcome dataset
str(EcoNorwayDataRFE)

## Stop Cluster
stopCluster(cl)