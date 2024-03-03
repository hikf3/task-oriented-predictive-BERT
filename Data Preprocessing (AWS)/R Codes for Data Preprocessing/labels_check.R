year1 <- read.csv("~/GitHub/Doctoral-Research/BEHRT/DATA/FINETUNING_LABELS_PHE/LABELS_PHE_YR1_DM_DIAG_FINETUNEDATA.csv.gz")


total_dx_1<- year1$CKD + year1$MACE + year1$NEUR + year1$RET

table(total_dx1)


#0     1     2     3     4 
#26723  8268  1360   173    15
#

total_ckd1 <- sum(year1$CKD)

year2 <- read.csv("~/GitHub/Doctoral-Research/BEHRT/DATA/FINETUNING_LABELS_PHE/LABELS_PHE_YR2_DM_DIAG_FINETUNEDATA.csv.gz")
year3 <- read.csv("~/GitHub/Doctoral-Research/BEHRT/DATA/FINETUNING_LABELS_PHE/LABELS_PHE_YR3_DM_DIAG_FINETUNEDATA.csv.gz")
#year3 <- LABELS_PHE_YR3_DM_DIAG_FINETUNEDATA.csv
#
total_dx_2<- year2$CKD + year2$MACE + year2$NEUR + year2$RET

table(total_dx_2)

#0     1     2     3     4 
#26127  8167  1887   316    42

total_dx_3<- year3$CKD + year3$MACE + year3$NEUR + year3$RET

table(total_dx_3)
#0     1     2     3     4 
#25851  7977  2196   449    66

sum(year1$CKD) #2701
sum(year2$CKD) #3099
sum(year3$CKD) #3378

sum(year1$MACE) #3991
sum(year2$MACE) #4470
sum(year3$MACE) #4746

sum(year1$NEUR) # 3891
sum(year2$NEUR) #4323
sum(year3$NEUR) #4578

sum(year1$RET) #984
sum(year2$RET) #1165
sum(year3$RET) #1278