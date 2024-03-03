-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------
-----                                 Date: October 10, 2023,Author: Humayera Islam                            ------
----- Checking the age of DM diagnosis, making sure all DM patients were diagnosed with DM >=18 yrs              ------
-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_COHORT_DM_AGE" as
select *, datediff(day,BIRTH_DATE, DMONSETDATE)/365.2425 as AGE_AT_DM
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."FINALDIABETESCOHORT"

select min(AGE_AT_DM), median(AGE_AT_DM), max(AGE_AT_DM)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_COHORT_DM_AGE"
where PATID in (select PATIENT_ID from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAIN_MAP2PHE")

select *
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_COHORT_DM_AGE"
where PATID in (select PATIENT_ID from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAIN_MAP2PHE") and AGE_AT_DM>=90
---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------
-----                                 Date: September 20, 2023,Author: Humayera Islam                    ------
----- Adding new features to the analysis. First step is to add age and sex from the demographics table  ------
---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------

---- Pretraining DM Cohort PATID and Cutoff date (to determine the learning period) -----
---- Join with DOB and SEX from Demographics table ----

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DEMO_PRETRAIN" as
select distinct b.PATIENT_ID, c.BIRTH_DATE as DOB, c.SEX
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAIN_MAP2PHE" b 
left join "DEIDENTIFIED_PCORNET_CDM"."CDM_2022_OCT"."DEID_DEMOGRAPHIC" c
on b.PATIENT_ID = c.PATID

select SEX, count(distinct PATIENT_ID)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DEMO_PRETRAIN" 
group by SEX -- F: 26,329 M: 24,664 --no missing

--adding the cutoff date to each PATID
create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DEMO_PRETRAIN" as
select distinct b.*, c.cutoff
from  "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DEMO_PRETRAIN" b 
left join "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_ENDPNT_CUTOFF" c
on b.PATIENT_ID = c.PATID

--- Compute age of patients at every encounter date (admit_date)
create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_DX_AGE" as
select b.*, c.DOB, datediff(day,c.DOB, b.ADMIT_DATE)/365.2425 as AGE_AT_ENC, c.SEX
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAIN_MAP2PHE" b
left join "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DEMO_PRETRAIN" c
on b.PATIENT_ID = c.PATIENT_ID
order by b.PATIENT_ID, b.ADMIT_DATE, b.DISCHARGE_DATE, b.POA, b.THIRD_PARTY_IND, b.DIAGNOSIS_PRIORITY

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_DX_AGE" as
select * 
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_DX_AGE"
where AGE_AT_ENC >0


-- check

select count(distinct PATIENT_ID) --50993
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_DX_AGE"
where AGE_AT_ENC >0

select count(distinct PATIENT_ID) --50993
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAIN_MAP2PHE"

select max(cnt), min(cnt), median(cnt)
from (select patient_id, count(patient_id) as cnt 
      from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_DX_AGE" 
      group by patient_id)
      
select count (distinct patient_id) --2611
from (select patient_id, count(patient_id) as cnt 
      from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_DX_AGE" 
      group by patient_id)
where cnt >=500


select count (distinct patient_id) -- 5251
from (select patient_id, count(patient_id) as cnt 
      from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_DX_AGE" 
      group by patient_id)
where cnt > 250 and cnt <500

select count (distinct patient_id) -- 43131
from (select patient_id, count(patient_id) as cnt 
      from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_DX_AGE" 
      group by patient_id)
where cnt <=250

select *
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_DX_AGE"
where PATIENT_ID = '1940'


select *
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_DX_AGE"
where AGE_AT_ENC <=1

select *
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_DX_AGE"
where AGE_AT_ENC <=0

select * from "DEIDENTIFIED_PCORNET_CDM"."CDM_2022_OCT"."DEID_DEMOGRAPHIC"
where patid = '2442839'


select * from "DEIDENTIFIED_PCORNET_CDM"."CDM_2022_OCT"."DEID_ENCOUNTER"
where patid = '2442839'
order by admit_date

select * from "DEIDENTIFIED_PCORNET_CDM"."CDM_2022_OCT"."DEID_PRESCRIBING"
where patid = '2442839'
order by RX_ORDER_DATE

-- Create a finetune feature cohort containing AGE_AT_ENC and SEX

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_FINETUNE_DX_AGE" as
select b.*
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_DX_AGE" b
where PATIENT_ID in (select PATIENT_ID from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNE_MAP2PHE" )
order by b.PATIENT_ID, b.ADMIT_DATE, b.DISCHARGE_DATE, b.POA, b.THIRD_PARTY_IND, b.DIAGNOSIS_PRIORITY

select count(distinct PATIENT_ID ) from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_FINETUNE_DX_AGE"
-- symptoms as DX
create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_FINETUNE_DX_CNT" as
select distinct "phecode", count(*) as cnt
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_FINETUNE_DX_AGE"
group by "phecode"

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_FINETUNE_DX_CNT" as
select d.*, e."phenotype"
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_FINETUNE_DX_CNT" d
left join "ONTOLOGY"."GROUPER_VALUESETS"."PHECODE_REF" e
on d."phecode" = e."phecode"


select distinct "phecode", "phecode_str" from "ONTOLOGY"."GROUPER_VALUESETS"."ICD10CM_PHECODE"
where "phecode" in (select distinct "phecode"
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_FINETUNE_DX_AGE")

select distinct "phecode", "phenotype" from "ONTOLOGY"."GROUPER_VALUESETS"."PHECODE_REF"
where "phecode" in (select distinct "phecode"
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_FINETUNE_DX_AGE")


-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------
-----                                 Date: September 20, 2023,Author: Humayera Islam                            ------
----- Adding new features to the analysis. Second step is to add medication list ordered from prescribing table  ------
-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------

-- looking into the prescribing table to take out the list of meds for the patient ids in pretraining cohort
create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_MEDS" as
select distinct PATID, ENCOUNTERID, RXNORM_CUI
from "DEIDENTIFIED_PCORNET_CDM"."CDM_2022_OCT"."DEID_PRESCRIBING"
where PATID in (select PATIENT_ID from"ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_DX_AGE") and RXNORM_CUI IS NOT NULL
order by PATID

-- merging encounter and meds table
create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_MEDS" as
select b.*, cast(c.ADMIT_DATE as date) as ADMIT_DATE, 
case when c.DISCHARGE_DATE is not null then cast(c.DISCHARGE_DATE as date) 
     else cast(c.ADMIT_DATE as date) end as DISCHARGE_DATE
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_MEDS" b
left join "DEIDENTIFIED_PCORNET_CDM"."CDM_2022_OCT"."DEID_ENCOUNTER" c
on b.PATID = c.PATID and b.ENCOUNTERID = c.ENCOUNTERID

-- create a column for age at enc using demog table
-- keep the encounters only when before cutoff
create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_MEDS" as
select m.*, datediff(day,d.DOB,m.ADMIT_DATE)/365.2425 as AGE_AT_ENC, d.SEX, datediff(day, m.discharge_date, d.cutoff) as date_diff 
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_MEDS" m
left join "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DEMO_PRETRAIN" d
on m.PATID=d.PATIENT_ID
where date_diff > 0
order by PATID, ADMIT_DATE

#updating the table with encounters with age_at_enc>0
create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_MEDS" as
select distinct PATID, ENCOUNTERID, RXNORM_CUI, ADMIT_DATE, DISCHARGE_DATE, AGE_AT_ENC, SEX
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_MEDS"
where AGE_AT_ENC >0
order by PATID, ADMIT_DATE

select min(cnt), max(cnt), median(cnt)
from (select PATID, count(PATID) as cnt
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_MEDS"
group by PATID)

select count(distinct RXNORM_CUI) as cnt 
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_MEDS"
    
select * from (select PATID, count(distinct RXNORM_CUI) as cnt
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_MEDS"
group by PATID) 
where cnt>500

select count(PATID) --1846 patients
from (select PATID, count(PATID) as cnt
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_MEDS"
group by PATID)
where cnt>500


select distinct PATID, cnt --360
from (select PATID, count(PATID) as cnt
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_MEDS"
group by PATID)
where cnt>1000

select count(distinct PATID) --4371 patients
from (select PATID, count(PATID) as cnt
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_MEDS"
group by PATID)
where cnt>250 and cnt <500


select count(distinct PATID) --43164 patients
from (select PATID, count(PATID) as cnt
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_MEDS"
group by PATID)
where cnt<=250

select count(distinct PATID)--49390
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_MEDS"

select count(distinct RXNORM_CUI) #36415
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_MEDS"


-- Create a finetune feature cohort containing MEDS, AGE_AT_ENC and SEX

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_FINETUNE_MEDS" as
select b.*
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_MEDS" b
where PATID in (select PATIENT_ID from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNE_MAP2PHE" )
order by b.PATID, b.ADMIT_DATE

select count(distinct PATID) --35776
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_FINETUNE_MEDS"

-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------
-----                                 Date: September 20, 2023,Author: Humayera Islam                            ------
----- Adding new features to the analysis. Third step is to add LOINC LABS ordered from LAB_RESULT table & result  ------
-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------
-- looking into the LABS table to take out the list of loincs and results for the patient ids in pretraining cohort
create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_LABS" as
select distinct PATID, ENCOUNTERID, LAB_LOINC, RESULT_QUAL, RESULT_NUM, RESULT_UNIT, NORM_RANGE_LOW, NORM_MODIFIER_LOW, NORM_RANGE_HIGH, NORM_MODIFIER_HIGH, RAW_LAB_NAME
from "DEIDENTIFIED_PCORNET_CDM"."CDM_2022_OCT"."DEID_LAB_RESULT_CM"
where PATID in (select PATIENT_ID from"ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_DX_AGE") and LAB_LOINC IS NOT NULL
order by PATID

--Check
select count(distinct PATID) from "DEIDENTIFIED_PCORNET_CDM"."CDM_2022_OCT"."DEID_LAB_RESULT_CM"
where LAB_PX_TYPE = 'LC' --576262

select RESULT_QUAL,count(distinct PATID) from "DEIDENTIFIED_PCORNET_CDM"."CDM_2022_OCT"."DEID_LAB_RESULT_CM"
group by RESULT_QUAL

-- create a table with list of LOINC
create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."LOINC_LABS_LIST" as
select distinct LAB_LOINC, RESULT_QUAL, RESULT_NUM, RESULT_UNIT, NORM_RANGE_LOW, NORM_MODIFIER_LOW, NORM_RANGE_HIGH, NORM_MODIFIER_HIGH, d.COMPONENT
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_LABS" c
left join "ONTOLOGY"."LOINC"."LOINC_V2_17" d
on c.LAB_LOINC = d.LOINC_NUM
order by LAB_LOINC,RESULT_NUM

select distinct LAB_LOINC, COMPONENT
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."LOINC_LABS_LIST"

select count(distinct LAB_LOINC)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."LOINC_LABS_LIST" --1953

-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------
-----                                 Date: September 20, 2023,Author: Humayera Islam                            ------
----- Adding new features to the analysis. Fourth step is to add VITALS Vitals table                             ------
-----------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------

-- looking into the VITALS table for the patient ids in pretraining cohort
create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_VITALS" as
select distinct PATID, ENCOUNTERID, HT, WT, DIASTOLIC, SYSTOLIC
from "DEIDENTIFIED_PCORNET_CDM"."CDM_2022_OCT"."DEID_VITAL" 
where PATID in (select PATIENT_ID from"ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_DX_AGE") 
order by PATID


--- phecodes
select distinct d."phecode", e."category"
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."MAPPING_PHECODE" d
left join "ONTOLOGY"."GROUPER_VALUESETS"."PHECODE_REF" e
where d."phecode" = e."phecode"
order by d."phecode"

select distinct "phecode","category", "phenotype"
from "ONTOLOGY"."GROUPER_VALUESETS"."PHECODE_REF"

## ckd
select distinct "phecode","phenotype"
from "ONTOLOGY"."GROUPER_VALUESETS"."PHECODE_REF"
where "phecode" in (728.10,  575.60,  656.20 , 781.10 , 433.50 , 473.10,  627.30,  420.20, 1004.00 , 369.20,  132.00,  728.71 , 263.00,  870.40 ,  54.00,  348.20, 723.00 , 568.10,  377.00 , 585.20)
## NEUR
select distinct "phecode","phenotype"
from "ONTOLOGY"."GROUPER_VALUESETS"."PHECODE_REF"
where "phecode" in (624.20, 428.30, 291.10, 259.10 ,385.00 ,742.20, 612.30, 475.90, 395.40 ,686.20, 604.30, 753.10 ,359.20, 368.30, 350.30, 855.00, 359.10, 749.00, 728.71 ,697.00)
## RET
select distinct "phecode","phenotype"
from "ONTOLOGY"."GROUPER_VALUESETS"."PHECODE_REF"
where "phecode" in (540.00, 530.11, 185.00, 958.00, 585.00, 382.00, 250.70, 281.00, 281.11, 443.00, 627.30, 573.30, 381.10, 536.00, 714.10, 381.11, 369.00, 565.10,  41.20, 380.40)
## MACE
select distinct "phecode","phenotype"
from "ONTOLOGY"."GROUPER_VALUESETS"."PHECODE_REF"
where "phecode" in (245.00, 617.00, 635.30, 743.13, 368.10, 255.11, 729.10, 609.20, 743.40, 414.20, 716.20, 264.00, 286.50, 474.10, 428.30, 377.10, 656.00, 609.11, 433.50, 345.11)

## number of unique patids in EHR
select count(distinct PATID) 
from "DEIDENTIFIED_PCORNET_CDM"."CDM_2023_OCT"."DEID_ENCOUNTER"


## extract race for patients
select distinct RACE, count(*) as n from"DEIDENTIFIED_PCORNET_CDM"."CDM"."DEID_DEMOGRAPHIC"
where PATID in (select distinct PATIENT_ID from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_PRETRAIN_DX_AGE")
group by RACE

select distinct RACE, count(*) as n from "DEIDENTIFIED_PCORNET_CDM"."CDM"."DEID_DEMOGRAPHIC"
where PATID in (select distinct PATIENT_ID from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_FINETUNE_DX_AGE")
group by RACE