##########################################################################################################
############  This program will download the fine tuning cohort          #######################
############      DX as ICD and Phecode                                  ###########################
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

################## Loading Fine-tuning Features-DX ICD
DM_DIAG_FINETUNING_FEATURES <- DBI::dbGetQuery(con,"SELECT * FROM ANALYTICS.DIABETESMELLITUSSTUDYSCHEMA.DM_DIAG_FINETUNING_NEXT1YR")

data_features<-DM_DIAG_FINETUNING_FEATURES
data_features <- data_features[order(data_features$PATIENT_ID),]

write_csv(data_features, file.path("D:\\BERT\\FINETUNING_FEATURES\\DM_DIAG_ICD_FINETUNEDATA.csv.gz"))


################## Loading Fine-tuning Labels- DX ICD : 1 year prediction
DM_DIAG_FINETUNE_DX_NEXT1YR_LABELS <- DBI::dbGetQuery(con,"SELECT * FROM ANALYTICS.DIABETESMELLITUSSTUDYSCHEMA.DM_DIAG_FINETUNING_NEXT1YR_LABELS")

data_labels<- DM_DIAG_FINETUNE_DX_NEXT1YR_LABELS
#View(table(data_labels$PATIENT_ID))

data_labels$ENDPOINT<-ifelse(data_labels$ENDPOINT %in% c('HF','MI','REVASC','STROKE'), 'MACE',
                             ifelse(data_labels$ENDPOINT=='CKD','CKD',
                                    ifelse(data_labels$ENDPOINT=='NEUR','NEUR',
                                           ifelse(data_labels$ENDPOINT=='RET','RET',0 ))))

table(data_labels$ENDPOINT)
## CKD MACE NEUR  RET 
## 2677 4177 3855  979 
## 
# Remove duplicate rows
data_labels <- data_labels %>% distinct(PATIENT_ID, ENDPOINT, .keep_all=TRUE)
data_labels$label<-ifelse(data_labels$ENDPOINT=='NA',0,1)
data_labels<- spread(data_labels, key=ENDPOINT, value=label)
data_labels <-data_labels[,1:5]
data_labels[is.na(data_labels)]<-0
#data_labels1<-data_labels[,2:5]
write_csv(data_labels,file.path("D:\\BERT\\FINETUNING_LABELS_ICD\\LABELS_ICD_YR1_DM_DIAG_FINETUNEDATA.csv.gz") )

################## Loading Fine-tuning Labels- DX ICD : 2 year prediction
DM_DIAG_FINETUNE_DX_NEXT2YR_LABELS <- DBI::dbGetQuery(con,"SELECT * FROM ANALYTICS.DIABETESMELLITUSSTUDYSCHEMA.DM_DIAG_FINETUNING_NEXT2YR_LABELS")

data_labels<- DM_DIAG_FINETUNE_DX_NEXT2YR_LABELS
#View(table(data_labels$PATIENT_ID))

data_labels$ENDPOINT<-ifelse(data_labels$ENDPOINT %in% c('HF','MI','REVASC','STROKE'), 'MACE',
                             ifelse(data_labels$ENDPOINT=='CKD','CKD',
                                    ifelse(data_labels$ENDPOINT=='NEUR','NEUR',
                                           ifelse(data_labels$ENDPOINT=='RET','RET',0 ))))

table(data_labels$ENDPOINT)
##  CKD MACE NEUR  RET 
## 3074 4686 4288 1163
## 
# Remove duplicate rows
data_labels <- data_labels %>% distinct(PATIENT_ID, ENDPOINT, .keep_all=TRUE)
data_labels$label<-ifelse(data_labels$ENDPOINT=='NA',0,1)
data_labels<- spread(data_labels, key=ENDPOINT, value=label)
data_labels <-data_labels[,1:5]
data_labels[is.na(data_labels)]<-0
#data_labels1<-data_labels[,2:5]
write_csv(data_labels,file.path("D:\\BERT\\FINETUNING_LABELS_ICD\\LABELS_ICD_YR2_DM_DIAG_FINETUNEDATA.csv.gz") )

################## Loading Fine-tuning Labels- DX : 3 year prediction
DM_DIAG_FINETUNE_DX_NEXT3YR_LABELS <- DBI::dbGetQuery(con,"SELECT * FROM ANALYTICS.DIABETESMELLITUSSTUDYSCHEMA.DM_DIAG_FINETUNING_NEXT3YR_LABELS")

data_labels<- DM_DIAG_FINETUNE_DX_NEXT3YR_LABELS
#View(table(data_labels$PATIENT_ID))

data_labels$ENDPOINT<-ifelse(data_labels$ENDPOINT %in% c('HF','MI','REVASC','STROKE'), 'MACE',
                             ifelse(data_labels$ENDPOINT=='CKD','CKD',
                                    ifelse(data_labels$ENDPOINT=='NEUR','NEUR',
                                           ifelse(data_labels$ENDPOINT=='RET','RET',0 ))))

table(data_labels$ENDPOINT)
##  CKD MACE NEUR  RET 
## 3355 4982 4547 1277
## 
# Remove duplicate rows
data_labels <- data_labels %>% distinct(PATIENT_ID, ENDPOINT, .keep_all=TRUE)
data_labels$label<-ifelse(data_labels$ENDPOINT=='NA',0,1)
data_labels<- spread(data_labels, key=ENDPOINT, value=label)
data_labels <-data_labels[,1:5]
data_labels[is.na(data_labels)]<-0
#data_labels1<-data_labels[,2:5]
write_csv(data_labels,file.path("D:\\BERT\\FINETUNING_LABELS_ICD\\LABELS_ICD_YR3_DM_DIAG_FINETUNEDATA.csv.gz") )

############################################################################################################
#############################################################################################################
#############################################################################################################
################## Loading Fine-tuning Features-Phecode and age
DM_DIAG_FINETUNING_FEATURES_MAP2PHE <- DBI::dbGetQuery(con,"SELECT * FROM ANALYTICS.DIABETESMELLITUSSTUDYSCHEMA.DM_FINETUNE_DX_AGE")

data_features<-DM_DIAG_FINETUNING_FEATURES_MAP2PHE
data_features <- data_features[order(data_features$PATIENT_ID),]

write_csv(data_features, file.path("C:\\Users\\Administrator\\Documents\\GitHub\\Doctoral-Research\\BEHRT\\DATA\\FINETUNING_FEATURES\\DM_FINETUNEDATA_DX_AGE.csv.gz"))


################## Loading Fine-tuning Labels- Phecode : 1 year prediction
DM_DIAG_FINETUNE_Phe_NEXT1YR_LABELS <- DBI::dbGetQuery(con,"SELECT * FROM ANALYTICS.DIABETESMELLITUSSTUDYSCHEMA.DM_DIAG_FINETUNEPHE_NEXT1YR_LABELS")

data_labels<- DM_DIAG_FINETUNE_Phe_NEXT1YR_LABELS
#View(table(data_labels$PATIENT_ID))

data_labels$ENDPOINT<-ifelse(data_labels$ENDPOINT %in% c('HF','MI','REVASC','STROKE'), 'MACE',
                             ifelse(data_labels$ENDPOINT=='CKD','CKD',
                                    ifelse(data_labels$ENDPOINT=='NEUR','NEUR',
                                           ifelse(data_labels$ENDPOINT=='RET','RET',0 ))))

table(data_labels$ENDPOINT)
##  CKD MACE NEUR  RET 
## 2701 4176 3891  984 
## 
# Remove duplicate rows
data_labels <- data_labels %>% distinct(PATIENT_ID, ENDPOINT, .keep_all=TRUE)
data_labels$label<-ifelse(data_labels$ENDPOINT=='NA',0,1)
data_labels<- spread(data_labels, key=ENDPOINT, value=label)
data_labels <-data_labels[,1:5]
data_labels[is.na(data_labels)]<-0
#data_labels1<-data_labels[,2:5]
write_csv(data_labels,file.path("D:\\BERT\\FINETUNING_LABELS_PHE\\LABELS_PHE_YR1_DM_DIAG_FINETUNEDATA.csv.gz") )

################## Loading Fine-tuning Labels- Phecode : 2 year prediction
DM_DIAG_FINETUNE_Phe_NEXT2YR_LABELS <- DBI::dbGetQuery(con,"SELECT * FROM ANALYTICS.DIABETESMELLITUSSTUDYSCHEMA.DM_DIAG_FINETUNEPHE_NEXT2YR_LABELS")

data_labels<- DM_DIAG_FINETUNE_Phe_NEXT2YR_LABELS
#View(table(data_labels$PATIENT_ID))

data_labels$ENDPOINT<-ifelse(data_labels$ENDPOINT %in% c('HF','MI','REVASC','STROKE'), 'MACE',
                             ifelse(data_labels$ENDPOINT=='CKD','CKD',
                                    ifelse(data_labels$ENDPOINT=='NEUR','NEUR',
                                           ifelse(data_labels$ENDPOINT=='RET','RET',0 ))))

table(data_labels$ENDPOINT)
##  CKD MACE NEUR  RET 
## 3099 4675 4323 1165 
## 
# Remove duplicate rows
data_labels <- data_labels %>% distinct(PATIENT_ID, ENDPOINT, .keep_all=TRUE)
data_labels$label<-ifelse(data_labels$ENDPOINT=='NA',0,1)
data_labels<- spread(data_labels, key=ENDPOINT, value=label)
data_labels <-data_labels[,1:5]
data_labels[is.na(data_labels)]<-0
#data_labels1<-data_labels[,2:5]
write_csv(data_labels,file.path("D:\\BERT\\FINETUNING_LABELS_PHE\\LABELS_PHE_YR2_DM_DIAG_FINETUNEDATA.csv.gz") )


################## Loading Fine-tuning Labels- Phecode : 3 year prediction
DM_DIAG_FINETUNE_Phe_NEXT3YR_LABELS <- DBI::dbGetQuery(con,"SELECT * FROM ANALYTICS.DIABETESMELLITUSSTUDYSCHEMA.DM_DIAG_FINETUNEPHE_NEXT3YR_LABELS")

data_labels<- DM_DIAG_FINETUNE_Phe_NEXT3YR_LABELS
#View(table(data_labels$PATIENT_ID))

data_labels$ENDPOINT<-ifelse(data_labels$ENDPOINT %in% c('HF','MI','REVASC','STROKE'), 'MACE',
                             ifelse(data_labels$ENDPOINT=='CKD','CKD',
                                    ifelse(data_labels$ENDPOINT=='NEUR','NEUR',
                                           ifelse(data_labels$ENDPOINT=='RET','RET',0 ))))

table(data_labels$ENDPOINT)
##  CKD MACE NEUR  RET 
## 3378 4965 4578 1278  
## 
# Remove duplicate rows
data_labels <- data_labels %>% distinct(PATIENT_ID, ENDPOINT, .keep_all=TRUE)
data_labels$label<-ifelse(data_labels$ENDPOINT=='NA',0,1)
data_labels<- spread(data_labels, key=ENDPOINT, value=label)
data_labels <-data_labels[,1:5]
data_labels[is.na(data_labels)]<-0
#data_labels1<-data_labels[,2:5]
write_csv(data_labels,file.path("D:\\BERT\\FINETUNING_LABELS_PHE\\LABELS_PHE_YR3_DM_DIAG_FINETUNEDATA.csv.gz") )

table(rowSums(data_labels[,2:5]))
##  0     1     2     3     4 
##29634  9622  1565   197    15 
colSums(data_labels[,2:5])
