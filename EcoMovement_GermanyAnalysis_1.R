### ------------ Eco Movement Project ------------ ###
### ------------- by Alican Tanaçan -------------- ###
### --- Version 1: Reconstructing Germany Data --- ###

## In short, the project goal is to develop a model that can predict/classify the public_access_type_id in Germany.
## public_access_type_id  is a feature that defines whether a charging station is:
##   1 public (i.e. publicly available)
##   2 private (i.e. charging stations of private persons)
##   3 company (i.e. provided by company, often publicly available, but with restrictions like not 24/7)

### ---- Libraries ----
if(require("pacman") == "FALSE"){
  install.packages("pacman")
}
p_load(dplyr, ggplot2, plotly, caret, corrplot, GGally,
       doParallel, tidyverse, e1071, randomForest, caTools,
       plyr, ROSE, kknn)

### ---- Import Datasets ----
Charging_Stations <- read.csv("D:/RStudio/R Studio Working Directory/Eco Movement/charging_stations_DE.csv", stringsAsFactors = F)
Charging_Spots <- read.csv("D:/RStudio/R Studio Working Directory/Eco Movement/charging_spots_DE.csv", stringsAsFactors = F)
Spot_Connectors <- read.csv("D:/RStudio/R Studio Working Directory/Eco Movement/spot_connectors_DE.csv", stringsAsFactors = F)
Master_Locations <- read.csv("D:/RStudio/R Studio Working Directory/Eco Movement/masterlocations_DE.csv", stringsAsFactors = F)
Reverse_Geodata <- read.csv("D:/RStudio/R Studio Working Directory/Eco Movement/reverse_geodata_DE.csv", stringsAsFactors = F)

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
# Public 2558
# Private 197  <- too few!
# Company 1286

## Removing Nondescriptive, Irrelevant and Uninformative Variables
df1 %>% 
  select(chargingstation_id,
         masterlocation_id,
         physical_type,
         public_access_type_id,
         opening_times,
         charging_when_closed) -> df1

df2 %>% 
  select(chargingstation_id,
         charging_spots_id,
         realtime_status_available,
         spot_status,
         capability_remote_start_stop_capable,
         capability_rfid_reader) -> df2

df3 %>% 
  select(charging_spots_id,
         connector_types_id,
         connector_format,
         powertype,
         voltage,
         amperage,
         power) -> df3

df5 %>% 
  select(masterlocation_id,
         lat,
         lng) -> df5

### ---- Dataset Merging and Feature Engineering ----

## Merge Charging_Stations with Charging_Spots
df12 <- merge(df1, df2, by = "chargingstation_id")

## Count the number of charging spots on each charging station
tally(group_by(df12, chargingstation_id)) -> a

## Change the name of "n" column
names(a)[2] <- paste("totalspots")

levels(as.factor(a$totalspots)) 
# 20 levels of totalspots from 1 to 24. Some stations have 1 charging spots where some
# others have 24 spots.

## Add the new column to df12
df12a <- merge(df12, a, by = "chargingstation_id")

## Count the number of spot connectors on each charging spot
tally(group_by(df3, charging_spots_id)) -> b

## Change the name of "n" column
names(b)[2] <- paste("totalconnectors")

levels(as.factor(b$totalconnectors))
# 3 levels of totalconnectors. Spot connectors can be 1, 2 or 3.

## Add the new column to df12a
df12ab <- merge(df12a, b, by = "charging_spots_id")

## Merge df12a with Spot_Connectors
df123ab <- merge(df12ab, df3, by = "charging_spots_id")

## Merge df123a with Reverse_Geodata
c <- merge(df123ab, df5, by = "masterlocation_id")

## Take the max power, voltage, amperage
aggregate(power ~ chargingstation_id, data = c, max) -> d
names(d)[2] <- paste("maxpower")

aggregate(voltage ~ chargingstation_id, data = c, max) -> e
names(e)[2] <- paste("maxvoltage")

aggregate(amperage ~ chargingstation_id, data = c, max) -> f
names(f)[2] <- paste("maxamperage")

## Bring the columns to main dataframe (c)
g <- merge(c, d, by = "chargingstation_id")
h <- merge(g, e, by = "chargingstation_id")
EcoGermanyData <- merge(h, f, by = "chargingstation_id")

## Detect Missing Values
sum(is.na(EcoGermanyData)) # Total 9278 NA's
colSums(is.na(EcoGermanyData))
# operator_id has 183 NA's
# owner_id has 183 NA's
# opening_times has 4084 NA's
# charging_when_closed has 4362 NA's
# connector_format has 3585 NA's
# powertype has 832 NA's

## Missing Value Treatment
EcoGermanyData[is.na(EcoGermanyData)] <- 0 # replace NA's with 0
sum(is.na(EcoGermanyData)) # No NA's

## Change the Dependent Variable name, levels and type
EcoGermanyData %>% 
  mutate(Access_Type = case_when(public_access_type_id %in% "1" ~ "Public",
                                 public_access_type_id %in% "2" ~ "Private",
                                 public_access_type_id %in% "3" ~ "Company")) -> EcoGermanyData

EcoGermanyData$public_access_type_id <- NULL

## Change date types
EcoGermanyData %>% 
  mutate_at(c("masterlocation_id",
              "charging_spots_id",
              "chargingstation_id",
              "Access_Type",
              "connector_types_id",
              "physical_type",
              "opening_times",
              "charging_when_closed",
              "realtime_status_available",
              "capability_remote_start_stop_capable",
              "spot_status",
              "capability_rfid_reader",
              "connector_format",
              "powertype"), as.factor) -> EcoGermanyData

## Check levels of opening_times
levels(EcoGermanyData$opening_times)

## Create a new variable that depicts open hours of chargers
EcoGermanyData %>% 
  mutate(
    Open_Hours = 
      if_else(opening_times %in% c("{\"twentyfourseven\": true }",
                                   "{\"twentyfourseven\":true}"), "Everytime",
              if_else(opening_times %in% "0", "Unknown",
                      "Regular"))) -> EcoData

EcoData$opening_times <- NULL

EcoData$Open_Hours <- as.factor(EcoData$Open_Hours)

## Select variables for modelization
EcoData %>% 
  select(chargingstation_id,
         Access_Type,
         physical_type,
         charging_when_closed,
         realtime_status_available,
         spot_status,
         capability_remote_start_stop_capable,
         capability_rfid_reader,
         connector_types_id,
         connector_format,
         Open_Hours,
         lat,
         lng,
         maxpower,
         maxvoltage,
         maxamperage,
         totalspots,
         totalconnectors) -> ReadyEcoData

## Remove duplicates from the data
unique(ReadyEcoData) -> ReadyEcoData # 4477 duplicates at the moment

## Move Dependent Variable to end
ReadyEcoData %>% 
  select(-Access_Type, Access_Type) -> ReadyEcoData

### ---- Sampling ----
## Check how many observations are there in Dependent Variable
ReadyEcoData %>% 
  group_by(Access_Type) %>% 
  summarise(count(Access_Type))
# Public 4688
# Private 186
# Company 1550

## Change y data type to character in order to subset without empty classes
ReadyEcoData$Access_Type <- as.character(ReadyEcoData$Access_Type)

## Divide the data into 3 subsets for the levels of y
ReadyEcoData %>% 
  filter(Access_Type == "Public") -> Data_Public

ReadyEcoData %>% 
  filter(Access_Type == "Private") -> Data_Private

ReadyEcoData %>% 
  filter(Access_Type == "Company") -> Data_Company

## Set seed
set.seed(123)

## Take random sample from Public and Company, same amount of Private
Public_Sample <- Data_Public[sample(1:nrow(Data_Public), 500, replace = F),]

Company_Sample <- Data_Company[sample(1:nrow(Data_Company), 500, replace = F),]

## combining private with company
Private_Company_Sample <- rbind(Data_Private,
                                Company_Sample)

## Over Sampling the Private Company Data
Private_Company_OverSample <- ovun.sample(Access_Type~., 
                                          data = Private_Company_Sample,
                                          p = 0.5,
                                          seed = 1, 
                                          method = "over")$data

## Join all samples into a single dataset
GermanySample <- rbind(Public_Sample,
                       Private_Company_OverSample)

GermanySample$Access_Type <- as.factor(GermanySample$Access_Type)

GermanySample %>% 
  group_by(Access_Type) %>% 
  summarise(count(Access_Type))
# Public: 500
# Private: 480
# Company: 500

### ---- Removing chargingstations_id ----
## It is also possible to exclude id column and keep track of it in caret modeling.
GermanySample %>%
  select(-chargingstation_id) -> GermanySample

### ---- Data Partition ----
intrain <- createDataPartition(y = GermanySample$Access_Type, 
                               p = 0.7, 
                               list = FALSE)

GermanyTrain <- GermanySample[intrain,]
GermanyTest <- GermanySample[-intrain,]

### ---- Core Selection ----
## Find how many cores are on your machine
detectCores() # Result = 8

## Create cluster with desired number of cores. If you have less than 8 cores do NOT run this code!
cl <- makeCluster(4)

## Register cluster
registerDoParallel(cl)

### ---- Recursive Feature Elimination ----
## Set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

## Use rfe and omit the response variable
rfeResults <- rfe(GermanyTrain[,1:16], 
                  GermanyTrain$Access_Type, 
                  sizes = (1:16), 
                  rfeControl = ctrl)

## Get results
rfeResults

## Predictors
predictors(rfeResults)

## Plot results
plot(rfeResults, type=c("g", "o"))

## Most Important Variables
varImp(rfeResults)
#                                       Overall
# totalspots                           35.17151
# maxpower                             20.37431
# Open_Hours                           19.59481
# connector_types_id                   17.06702
# lng                                  16.96443
# lat                                  16.80500
# maxvoltage                           14.88043
# capability_remote_start_stop_capable 14.41079
# totalconnectors                      13.67403
# spot_status                          13.43550
# maxamperage                          13.32858
# physical_type                        12.22100
# connector_format                     11.91763
# charging_when_closed                 11.83651

### ---- Feature Selection & Data Partition ----
GermanySample %>%
  select(totalspots,
         maxpower,
         Open_Hours,
         connector_types_id,
         lng,
         lat,
         Access_Type) -> GermanySample2

## Data Partition 
intrain <- createDataPartition(y = GermanySample2$Access_Type, 
                               p = 0.7, 
                               list = FALSE)

GermanyTrain2 <- GermanySample2[intrain,]
GermanyTest2 <- GermanySample2[-intrain,]


### ---- Random Forest ----
set.seed(456)
## Set Train Control
RFtrctrl <- trainControl(method = "repeatedcv",
                         number = 5,
                         repeats = 2,
                         verboseIter = TRUE)

RFmodel <- train(Access_Type ~ ., 
                 GermanyTrain2,
                 method = "rf",
                 trControl = RFtrctrl)

RFmodel

plot(RFmodel)

## Most Important Variables
plot(varImp(RFmodel))

## Predictors
predictors(RFmodel)

## Predicton on Test set
predRFmodel <- predict(RFmodel, GermanyTest2)

postResample(predRFmodel, GermanyTest2$Access_Type) -> RFmodelmetrics

RFmodelmetrics

## Confusion Matrix
RFConfMat <- confusionMatrix(predRFmodel, GermanyTest2$Access_Type) 
RFConfMat
#              Reference
# Prediction   1   2   3
#          1 121   0  23
#          2   6 143   2
#          3  23   0 125

# Accuracy: 0.876
# Kappa: 0.814

## Stop Cluster
stopCluster(cl)

## Save the model to disk
save(RFmodel, file = "GermanyRFmodel.rda")