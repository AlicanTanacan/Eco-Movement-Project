### ------------ Eco Movement Project ------------ ###
### ------------- by Alican Tanaçan -------------- ###
### --- Version 4.2: Validation on Norway Data --- ###

### ---- Libraries & Source ---- 
if(require("pacman") == "FALSE"){
  install.packages("pacman")
}
p_load(dplyr, ggplot2, plotly, caret, corrplot, GGally,
       doParallel, tidyverse, e1071, randomForest, caTools,
       plyr, ROSE, Hmisc, vcd, polycor, gbm)

### ---- Import Datasets ----
Norway_Charging_Stations <- read.csv("D:/RStudio/R Studio Working Directory/Eco Movement/charging_stations_NO.csv")
Norway_Charging_Spots <- read.csv("D:/RStudio/R Studio Working Directory/Eco Movement/charging_spots_NO.csv")
Norway_Spot_Connectors <- read.csv("D:/RStudio/R Studio Working Directory/Eco Movement/spot_connectors_NO.csv")
Norway_Master_Locations <- read.csv("D:/RStudio/R Studio Working Directory/Eco Movement/masterlocations_NO.csv")
Norway_Reverse_Geodata <- read.csv("D:/RStudio/R Studio Working Directory/Eco Movement/reverse_geodata_NO.csv")

## Save Original Datasets
df1 <- Norway_Charging_Stations
df2 <- Norway_Charging_Spots
df3 <- Norway_Spot_Connectors
df4 <- Norway_Master_Locations
df5 <- Norway_Reverse_Geodata

### ---- Preprocess Before Merging ----

## Detecting NA's at the Dependent Variables
summary(df1$public_access_type_id) # There are 24 NA's in public_access_type_id

## Exclude NA's from the Dependent Variable
df1 <- df1[!is.na(df1$public_access_type_id),]

## Check how many observations are there in Dependent Variable
df1 %>% 
  group_by(public_access_type_id) %>% 
  summarise(count(public_access_type_id))
# Public 1693
# Private 391
# Company 746

## Removing Nondescriptive, Irrelevant and Uninformative Variables
df1 %>% 
  select(chargingstation_id,
         masterlocation_id,
         physical_type,
         operator_id,
         owner_id,
         public_access_type_id,
         opening_times,
         charging_when_closed,
         additional_geo_location) -> df1

df2 %>% 
  select(chargingstation_id,
         charging_spots_id,
         realtime_status_available,
         spot_status,
         capability_remote_start_stop_capable,
         capability_rfid_reader) -> df2

df3 %>% 
  select(charging_spots_id,
         src_id,
         connector_types_id,
         connector_format,
         powertype,
         voltage,
         amperage,
         power) -> df3

df4 %>% 
  select(masterlocation_id,
         name,
         published,
         point) -> df4

df5 %>% 
  select(masterlocation_id,
         street,
         postal_code,
         city,
         administrative_area_1,
         administrative_area_2,
         lat,
         lng) -> df5

### ---- Merge Datasets ----
df12 <- merge(df1, df2, by = "chargingstation_id")

df123 <- merge(df12, df3, by = "charging_spots_id")

df1234 <- merge(df123, df4, by = "masterlocation_id")

EcoDataNorway <- merge(df1234, df5, by = "masterlocation_id")

## Save the Dataset
write.csv(EcoDataNorway, file = "EcoDataNorway.csv")

### --- Import Original Norway Data ----
OriginalNorwayEcoData <- read.csv("EcoDataNorway.csv", stringsAsFactors = F)

### ---- Preprocessing & Feature Selection ----
## Remove auto-generated column "X"
OriginalNorwayEcoData$X <- NULL

## Save Original Dataset
NorwayEcoData <- OriginalNorwayEcoData

## Missing Value Treatment
NorwayEcoData[is.na(NorwayEcoData)] <- 0
sum(is.na(NorwayEcoData)) # No NA's

## Check levels of opening_times
levels(as.factor(EcoData$opening_times))

## Create a new variable that depicts open hours of chargers
NorwayEcoData %>% 
  mutate(
    Open_Hours = 
      if_else(opening_times %in% c("{\"twentyfourseven\": true }",
                                   "{\"twentyfourseven\":true}"), "Everytime",
              if_else(opening_times %in% "0", "Unknown",
                      "Regular"))) -> NorwayEcoData

## Change variable type
NorwayEcoData$Open_Hours <- as.factor(NorwayEcoData$Open_Hours)

## Remove Unknown Open Hours 
NorwayEcoData %>% 
  filter(Open_Hours != "Unknown") -> NorwayEcoData

## Remove Nondescriptive, Irrelevant and Uninformative Variables
NorwayEcoData %>%
  select(power,
         Open_Hours,
         lat,
         lng,
         public_access_type_id) -> ReadyNorwayEcoData

## Move Dependent Variable to end
ReadyNorwayEcoData %>% 
  select(-public_access_type_id, public_access_type_id) -> ReadyNorwayEcoData

## Change Data Types for Modelization
str(ReadyNorwayEcoData)
ReadyNorwayEcoData %>% 
  mutate_at(c("Open_Hours",
              "public_access_type_id"), as.factor) -> ReadyNorwayEcoData

### ---- Predicting y with Germany Random Forest Model ----
GermanyRFModel <- load("EcoRandomForestModel.rda")

Pred_RF_Norway <- predict(RFmodel2, ReadyNorwayEcoData)

postResample(Pred_RF_Norway, ReadyNorwayEcoData$public_access_type_id) -> NorwayRFMetrics

NorwayRFMetrics

## Confusion Matrix
NorwayRFConfMat <- confusionMatrix(Pred_RF_Norway, ReadyNorwayEcoData$public_access_type_id) 
NorwayRFConfMat
