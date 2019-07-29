### ------------ Eco Movement Project ------------ ###
### ------------- by Alican Tanaçan -------------- ###
### ----- Version 5: Merging Norway Datasets ----- ###

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

## Removing Nondescriptive, Irrelevant and Uninformative Variables
df1 %>% 
  select(chargingstation_id,
         masterlocation_id,
         physical_type,
         operator_id,
         owner_id,
         public_access_type_id,
         opening_times,
         charging_when_closed) -> df1

df2 %>% 
  select(chargingstation_id,
         charging_spots_id,
         realtime_status_available,
         spot_status,
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
         lat,
         lng) -> df5

### ---- Merge Datasets ----
df12 <- merge(df1, df2, by = "chargingstation_id")

df123 <- merge(df12, df3, by = "charging_spots_id")

df1234 <- merge(df123, df4, by = "masterlocation_id")

EcoDataNorway <- merge(df1234, df5, by = "masterlocation_id")

## Save the Dataset to Working Directory
write.csv(EcoDataNorway, file = "EcoDataNorway.csv")
