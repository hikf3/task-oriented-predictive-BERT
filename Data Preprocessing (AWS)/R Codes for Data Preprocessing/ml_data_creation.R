##########################################################################################################
############      This program will find download the ML dataset        ############################
############         and convert that to wide tables                         ###########################
#########################################################################################################

library(DBI)
library(dplyr)
library(dbplyr)
library(odbc)
library(RODBC)
library(tidyr)
library(stringr)
library(readr)
library(tidyverse)

################## Loading data tables from snowflake connection
################## "DM_DIAG_MLDATA_PHE_DISTNCT" --Phecode
DM_DIAG_MLDATA_PHE <- DBI::dbGetQuery(con,"SELECT * FROM ANALYTICS.DIABETESMELLITUSSTUDYSCHEMA.DM_DIAG_MLDATA_PHE_DISTNCT") 

DM_DIAG_MLDATA <- DM_DIAG_MLDATA_PHE

DM_DIAG_MLDATA$STATUS <-1

DM_DIAG_MLDATA_MATRIX <- spread(DM_DIAG_MLDATA, phecode, STATUS)

DM_DIAG_MLDATA_MATRIX[is.na(DM_DIAG_MLDATA_MATRIX)] <-0
#DM_DIAG_MLDATA_MATRIX1<- DM_DIAG_MLDATA_MATRIX[,2:1860]

write_csv(DM_DIAG_MLDATA_MATRIX,file.path("D:\\BERT\\ML_FEATURES_DATA\\FEATURES_DM_DIAG_PHE_MLDATA.csv.gz") )

################## "DM_DIAG_MLDATA_PHE_DISTNCT" --ICD

DM_DIAG_MLDATA_ICD <- DBI::dbGetQuery(con,"SELECT * FROM ANALYTICS.DIABETESMELLITUSSTUDYSCHEMA.DM_DIAG_MLDATA_ICD") 

DM_DIAG_MLDATA <- DM_DIAG_MLDATA_ICD

DM_DIAG_MLDATA$STATUS <-1

#DM_DIAG_MLDATA_MATRIX <- spread(DM_DIAG_MLDATA, DIAGNOSIS, STATUS)

#DM_DIAG_MLDATA_MATRIX[is.na(DM_DIAG_MLDATA_MATRIX)] <-0
#DM_DIAG_MLDATA_MATRIX1<- DM_DIAG_MLDATA_MATRIX[,2:1860
#
#

DM_DIAG_MLDATA_MATRIX <- DM_DIAG_MLDATA %>%
  pivot_wider(names_from= DIAGNOSIS, values_from=STATUS, values_fill=0)

typeof(DM_DIAG_MLDATA$DIAGNOSIS)
typeof(DM_DIAG_MLDATA$STATUS)

write_csv(DM_DIAG_MLDATA_MATRIX,file.path("D:\\BERT\\ML_FEATURES_PHE\\FEATURES_DM_DIAG_ICD_MLDATA.csv.gz") )

