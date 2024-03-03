##########################################################################################################
############      This program will find download the pretrain dataset        ############################
############         and save that as a zip file                              ###########################
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
################## DM_DIAG_PRETRAINDATA - PHE & AGE at visit
DM_DIAG_PRETRAINDATA <- DBI::dbGetQuery(con,"SELECT * FROM ANALYTICS.DIABETESMELLITUSSTUDYSCHEMA.DM_PRETRAIN_DX_AGE") 

########### save dataset 
########### 
file_name_mapped = paste("C:\\Users\\Administrator\\Documents\\GitHub\\Doctoral-Research\\BEHRT\\DATA\\PRETRAIN_DATA\\DM_PRETRAIN_DX_AGE.csv.gz", sep="")

#write_tsv(DM_DIAG_PRETRAINDATA, file.path(fine_name_mapped))

#demo<-read_tsv("D:\\BERT\\DM_DIAG_PRETRAINDATA.tsv.gz")

write_csv(DM_DIAG_PRETRAINDATA, file.path(file_name_mapped))

#demo<-read.csv("D:\\BERT\\DM_DIAG_PRETRAINDATA2.csv.gz")

################## DM_DIAG_PRETRAINDATA - PHE
#DM_DIAG_PRETRAINDATA_PHE <- DBI::dbGetQuery(con,"SELECT * FROM ANALYTICS.DIABETESMELLITUSSTUDYSCHEMA.DM_DIAG_PRETRAIN_MAP2PHE") 

#write_csv(DM_DIAG_PRETRAINDATA_PHE, file.path("D:\\BERT\\PRETRAIN_DATA\\DM_DIAG_PRETRAINDATA_PHE.csv.gz"))


pretrain <- DM_DIAG_PRETRAINDATA

summary(pretrain$AGE_AT_ENC)

hist(pretrain$AGE_AT_ENC)
