### ------------ Eco Movement Project ------------ ###
### ------------- by Alican Tanaçan -------------- ###
### ------ Version 1: Merging Germany Data ------- ###

### ---- Libraries ----
if(require("pacman") == "FALSE"){
  install.packages("pacman")
}
p_load(dplyr, ggplot2, plotly, caret, corrplot, GGally,
       doParallel, tidyverse, e1071, randomForest, caTools,
       plyr, ROSE, kknn)

### ---- Import Datasets ----
Charging_Stations <- read.csv("D:/RStudio/R Studio Working Directory/Eco Movement/charging_stations_DE.csv")
Charging_Spots <- read.csv("D:/RStudio/R Studio Working Directory/Eco Movement/charging_spots_DE.csv")
Spot_Connectors <- read.csv("D:/RStudio/R Studio Working Directory/Eco Movement/spot_connectors_DE.csv")
Master_Locations <- read.csv("D:/RStudio/R Studio Working Directory/Eco Movement/masterlocations_DE.csv")
Reverse_Geodata <- read.csv("D:/RStudio/R Studio Working Directory/Eco Movement/reverse_geodata_DE.csv")

## Save Original Datasets
df1 <- Charging_Stations
df2 <- Charging_Spots
df3 <- Spot_Connectors
df4 <- Master_Locations
df5 <- Reverse_Geodata

### ---- Preprocess Before Merging ----

## Detecting NA's at the Dependent Variables
summary(df1$public_access_type_id) # There are 8115 NA's in public_access_type_id

## Exclude NA's from the Dependent Variable
df1 <- df1[!is.na(df1$public_access_type_id),]

## Check how many observations are there in Dependent Variable
df1 %>% 
  group_by(public_access_type_id) %>% 
  summarise(count(public_access_type_id))

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

EcoGermanyData <- merge(df1234, df5, by = "masterlocation_id")

## Save the Dataset
write.csv(EcoGermanyData, file = "EcoGermanyData.csv")