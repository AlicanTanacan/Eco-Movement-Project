### ------------ Eco Movement Project ------------ ###
### ------------- by Alican Tanaçan -------------- ###
### ------ Version 2: Germany Data Analysis ------ ###

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

### ---- Inspect Datasets ----

## Check how many chargingstation_id are there, how many times they occur?
n_occur <- data.frame(table(df1$chargingstation_id))
n_occur[n_occur$Freq > 1,]
# 12156 unique chargingstation_id's

## Check how many chargingstation_id are there in df2, how many times they occur?
n_occur2 <- data.frame(table(df2$chargingstation_id))
n_occur2[n_occur2$Freq > 1,]
# 11728 unique chargingstation_id's

## Check how many charging_spots_id are there in df2, how many times they occur?
n_occur3 <- data.frame(table(df2$charging_spots_id))
n_occur3[n_occur3$Freq > 1,]
# 26771 unique charging_spots_id's

## Check how many charging_spots_id are there in df3, how many times they occur?
n_occur4 <- data.frame(table(df3$charging_spots_id))
n_occur4[n_occur4$Freq > 1,]
# 26766 unique charging_spots_id's

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
         opening_times) -> df1

df2 %>% 
  select(chargingstation_id,
         charging_spots_id) -> df2

df3 %>% 
  select(charging_spots_id,
         connector_types_id,
         powertype,
         voltage,
         amperage,
         power) -> df3

df5 %>% 
  select(masterlocation_id,
         lat,
         lng) -> df5

### ---- Dataset Merging and Feature Engineering ----

### Merge Charging_Stations with Charging_Spots
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


## Count the number of connector types on each charging spot and indirectly charging station.
df12ab$chargingstation_id <- as.factor(df12ab$chargingstation_id)

## Sum totalconnectors on each charging station
aggregate(totalconnectors ~ chargingstation_id, data = df12ab, sum) -> y

## See the levels of summed total connectors on each charging stations
levels(as.factor(y$totalconnectors))

## Change the name of the column
names(y)[2] <- paste("sumconnectors")

## Add the new column to df123abx
df12aby <- merge(df12ab, y, by = "chargingstation_id")


## Create a new column `n` for the number of connector types
df3 %>% 
  group_by(charging_spots_id) %>% 
  add_count(connector_types_id) -> df3

## Change the name of the new column
names(df3)[7] <- paste("totalconnectortypes")

## Change the data type and see how many levels of connector types each charging spot have
df3$totalconnectortypes <- as.factor(df3$totalconnectortypes)
levels(df3$totalconnectortypes) # 1 and 2

## Take the maximum number of totalconnectortypes because some spots have both 1 and 2 types at the same time.
df3$totalconnectortypes <- as.numeric(df3$totalconnectortypes)

## Create a new data frame with a new column
aggregate(totalconnectortypes ~ charging_spots_id, data = df3, max) -> x

## Change the name of the column
names(x)[2] <- paste("maxconnectortypes")

## Add the new column to df3
df3x <- merge(df3, x, by = "charging_spots_id")


## Merge df12a with Spot_Connectors
df123abxy <- merge(df12aby, df3x, by = "charging_spots_id")


## Merge df123a with Reverse_Geodata
c <- merge(df123abxy, df5, by = "masterlocation_id")


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

## Check the levels of engineered features
levels(as.factor(EcoGermanyData$totalspots))
levels(as.factor(EcoGermanyData$sumconnectors))
levels(as.factor(EcoGermanyData$totalconnectors))
levels(as.factor(EcoGermanyData$totalconnectortypes))
levels(as.factor(EcoGermanyData$maxconnectortypes))

## Detect Missing Values
sum(is.na(EcoGermanyData)) # Total 5196 NA's
colSums(is.na(EcoGermanyData))
# opening_times has 43646 NA's
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
              "physical_type",
              "opening_times",
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
         Open_Hours,
         lat,
         lng,
         maxpower,
         maxvoltage,
         maxamperage,
         totalspots,
         sumconnectors,
         maxconnectortypes) -> ReadyEcoData0

## Remove duplicates from the data
unique(ReadyEcoData0) -> ReadyEcoData # 6918 duplicates removed

## Move Dependent Variable to end
ReadyEcoData %>% 
  select(-Access_Type, Access_Type) -> ReadyEcoData

### ---- Sampling ----
## Check how many observations are there in Dependent Variable
ReadyEcoData %>% 
  group_by(Access_Type) %>% 
  summarise(count(Access_Type))
# Public 2531
# Private 186
# Company 1266

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
rfeResults <- rfe(GermanyTrain[,1:10], 
                  GermanyTrain$Access_Type, 
                  sizes = (1:10), 
                  rfeControl = ctrl)

## Get results
rfeResults

## Predictors
predictors(rfeResults)

## Plot results
plot(rfeResults, type=c("g", "o"))

## Most Important Variables
varImp(rfeResults)
#                Overall
# Open_Hours    43.38186
# maxpower      30.94530
# physical_type 28.92642
# lat           20.10162
# sumconnectors 19.89555
# lng           19.49497
# maxamperage   17.20227
# totalspots    16.16551
# maxvoltage    11.40072

### ---- Feature Selection & Data Partition for Random Forest ----
GermanySample %>%
  select(totalspots,
         maxpower,
         physical_type,
         sumconnectors,
         Open_Hours,
         lng,
         lat,
         maxamperage,
         maxvoltage,
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
#            Reference
# Prediction Company Private Public
# Company     114       1     16
# Private       5     141      1
# Public       31       2    133

# Accuracy: 0.873
# Kappa: 0.810

### ---- Feature Selection & Data Partition without Over Sampling ----
ReadyEcoData$Access_Type <- as.factor(ReadyEcoData$Access_Type)

ReadyEcoData %>%
  select(totalspots,
         maxpower,
         physical_type,
         sumconnectors,
         Open_Hours,
         lng,
         lat,
         maxamperage,
         maxvoltage,
         Access_Type) -> GermanySample3

## Data Partition 
intrain <- createDataPartition(y = GermanySample3$Access_Type, 
                               p = 0.7, 
                               list = FALSE)

GermanyTrain3 <- GermanySample3[intrain,]
GermanyTest3 <- GermanySample3[-intrain,]

GermanyTrain3 %>% 
  group_by(Access_Type) %>% 
  summarise(count(Access_Type))
# Public 1772
# Private 131 
# Company 887

GermanyTest3 %>% 
  group_by(Access_Type) %>% 
  summarise(count(Access_Type))
# Public 759
# Private 55 
# Company 379

### ---- Random Forest ----
set.seed(567)
## Set Train Control
RFtrctrl <- trainControl(method = "repeatedcv",
                         number = 10,
                         repeats = 2,
                         verboseIter = TRUE)

RFmodel2 <- train(Access_Type ~ ., 
                  GermanyTrain3,
                  method = "rf",
                  trControl = RFtrctrl)

RFmodel2

plot(RFmodel2)

## Most Important Variables
plot(varImp(RFmodel2))

## Predictors
predictors(RFmodel2)

## Predicton on Test set
predRFmodel2 <- predict(RFmodel2, GermanyTest3)

postResample(predRFmodel2, GermanyTest3$Access_Type) -> RFmodel2metrics

RFmodel2metrics

## Confusion Matrix
RFConfMat2 <- confusionMatrix(predRFmodel2, GermanyTest3$Access_Type) 
RFConfMat2
#            Reference
# Prediction Company Private Public
# Company     305       3     35
# Private       1      50      2
# Public       73       2    722

# Accuracy: 0.902
# Kappa: 0.798

## Stop Cluster
stopCluster(cl)

## Save the model to disk
# save(RFmodel, file = "GermanyRFmodel.rda")