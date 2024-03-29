---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------
-----  This code is written by Humayera Islam on January 18-19, 2023 as an adoption to extract            -----
-----  Diabetes Population from MU CDM using Snowflake.                                                   -----
-----  In case of questions contact hikf3@mail.missouri.edu                                               -----
-----  Snowflake codes are adopted frpm NU t-sql code for oracle (kindly done by Alexander Stoddard (MCW))-----

-----  In case of questions, please, contact MCW or NU via astoddard@mcw.edu or                           -----
-----  alona.furmanchuk@northwestern.edu or xsong@kumc.edu, correspondingly.                              -----
-----																				                      -----
-----  Some archived changes or modifications can be found from the following github link:                ----- 
-----  at https://github.com/kumc-bmi/nextd-study-support/commit/72a8ffc6bf63539152de16d75cf054cc42f01411 -----
-----                                                                                                     -----
-----  Extraction is based on MU CDM released on October 2023, which uses shifted dates                 -----
---------------------------------------------------------------------------------------------------------------


---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------
-----                          Part 1: Defining Denominator or Study Sample                               -----  
--------------------------------------------------------------------------------------------------------------- 
---------------------------------------------------------------------------------------------------------------
-----                People with at least two encounters recorded on different days                       -----
-----                                                                                                     -----
-----                       Encounter should meet the following requirements:                             -----
-----    Patient must be 18 years old >= age <= 89 years old during the encounter day.                    -----
-----    Encounter should be encounter types: 'AMBULATORY VISIT', 'EMERGENCY DEPARTMENT',                 -----
-----    'INPATIENT HOSPITAL STAY', 'OBSERVATIONAL STAY', 'NON-ACUTE INSTITUTIONAL STAY'.                 -----
-----                                                                                                     -----
-----          The date of the first encounter and total number of encounters is collected.               -----
---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------
-----         STEP 1.1: Create a table from encounter and demographic with all the encounters and         ----- 
-----                                       Age at First Visit                                            -----
---------------------------------------------------------------------------------------------------------------
CREATE SCHEMA DMCohort

CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."ENCS_WITH_AGE_AT_VISIT" AS
SELECT e.PATID, 
       e.ENCOUNTERID, 
       d.BIRTH_DATE,
       round(DATEDIFF(day, d.BIRTH_DATE,e.ADMIT_DATE)/365.25,2) AS AGE_AT_VISIT, 
       e.ADMIT_DATE, 
       e.ENC_TYPE
FROM "DEIDENTIFIED_PCORNET_CDM"."CDM_2023_OCT"."DEID_ENCOUNTER" e
JOIN "DEIDENTIFIED_PCORNET_CDM"."CDM_2023_OCT"."DEID_DEMOGRAPHIC" d
ON e.PATID = d.PATID


select count(distinct PATID) from "DEIDENTIFIED_PCORNET_CDM"."CDM_2023_OCT"."DEID_ENCOUNTER"

select enc_type, count(distinct PATID) from "DEIDENTIFIED_PCORNET_CDM"."CDM_2023_OCT"."DEID_ENCOUNTER"
group by enc_type
---------------------------------------------------------------------------------------------------------------
-----                     STEP 1.2: Create a table from ENCS_WITH_AGE_AT_VISIT with                       ----- 
-----                          #distinct encounter days and unique row # for each enc_id                  -----
---------------------------------------------------------------------------------------------------------------

CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."SUMMARIZED_ENCOUNTERS" AS 
SELECT PATID, ENCOUNTERID, BIRTH_DATE,ADMIT_DATE,
  count(DISTINCT ADMIT_DATE) OVER (PARTITION BY PATID) AS cnt_distinct_enc_days,
  row_number() OVER (PARTITION BY PATID ORDER BY ADMIT_DATE) AS rn,--identify the unique row for each encouter per patid
  ENC_TYPE,
  age_at_visit
FROM "ANALYTICS"."DMCOHORT"."ENCS_WITH_AGE_AT_VISIT"
WHERE age_at_visit >= 18 and age_at_visit <= 89
AND   ENC_TYPE in ('IP','ED','EI','IS','OS','AV','TH') 

#AND   ADMIT_DATE BETWEEN DATE '2010-01-01' AND '2023-10-31'


---------------------------------------------------------------------------------------------------------------
-----         STEP 1.3: Create a table from SUMMARIZED_ENCOUNTERS with                                    ----- 
-----                      cnt_distinct_enc_days >= 2                                                     -----
---------------------------------------------------------------------------------------------------------------
CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."MU_STUDY_DENOMINATOR" AS
SELECT PATID,ENCOUNTERID,BIRTH_DATE,ADMIT_DATE,cnt_distinct_enc_days,rn,ENC_TYPE,age_at_visit
FROM "ANALYTICS"."DMCOHORT"."SUMMARIZED_ENCOUNTERS" -- 13,000,482 rows
WHERE cnt_distinct_enc_days >= 2

select min(age_at_visit), max(age_at_visit)
from "ANALYTICS"."DMCOHORT"."MU_STUDY_DENOMINATOR"

select min(CNT_DISTINCT_ENC_DAYS), max(CNT_DISTINCT_ENC_DAYS)
from "ANALYTICS"."DMCOHORT"."MU_STUDY_DENOMINATOR"

select count(distinct PATID) from "DMCOHORT"."MU_STUDY_DENOMINATOR"


---------------------------------------------------------------------------------------------------------------
-----         STEP 1.4: Create a table for first visit only from MU_STUDY_DENOMINATOR                      -----
---------------------------------------------------------------------------------------------------------------

CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."ENC_FIRST_VISIT" AS
SELECT  PATID,ADMIT_DATE AS enc_first_visit,cnt_distinct_enc_days, age_at_visit as age_at_first_visit
FROM "ANALYTICS"."DMCOHORT"."MU_STUDY_DENOMINATOR" 
WHERE rn = 1  --561,369 


---------------------------------------------------------------------------------------------------------------
-----                                    Part 2: Defining Pregnancy                                       -----
---------------------------------------------------------------------------------------------------------------
-----                             People with pregnancy-related encounters                                -----
-----                                                                                                     -----
-----                       Encounter should meet the following requerements:                             -----
-----           Patient must be 18 years old >= age <= 89 years old during the encounter day.             -----
-----                                                                                                     -----
-----                 The date of the first encounter for each pregnancy is collected.                    -----
---------------------------------------------------------------------------------------------------------------

CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."PREGNANCY_DX" AS
SELECT dia.PATID      AS PATID,
       dia.ADMIT_DATE AS ADMIT_DATE
FROM "DEIDENTIFIED_PCORNET_CDM"."CDM_2023_OCT"."DEID_DIAGNOSIS" dia
JOIN "DEIDENTIFIED_PCORNET_CDM"."CDM_2023_OCT"."DEID_DEMOGRAPHIC" d 
ON dia.PATID = d.PATID
WHERE
      -- miscarriage, abortion, pregnancy, birth and pregnancy related complications diagnosis codes diagnosis codes:
     (
      -- ICD9 codes
      ((regexp_like(dia.DX,'^63[0-9]\.')
        or regexp_like(dia.DX,'^6[4-7][0-9]\.')
        or regexp_like(dia.DX,'^V2[2-3]\.')
        or regexp_like(dia.DX,'^V28\.')) AND dia.DX_TYPE = '09')
      OR
      -- ICD10 codes
      ((   regexp_like(dia.DX,'^O')
           or regexp_like(dia.DX,'^A34\.')
           or regexp_like(dia.DX,'^Z3[34]\.')
           or regexp_like(dia.DX,'^Z36')) AND dia.DX_TYPE = '10')
      )
      -- age restriction
      AND
      (
        (DATEDIFF(day, d.BIRTH_DATE,dia.ADMIT_DATE)/365.25) BETWEEN 18 AND 89
      )
      -- time frame restriction
      AND dia.ADMIT_DATE BETWEEN DATE '2010-01-01' AND CURRENT_DATE

      



CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."DELIVERY_PROC" AS
SELECT  p.PATID       AS PATID,
        p.ADMIT_DATE  AS ADMIT_DATE
FROM "DEIDENTIFIED_PCORNET_CDM"."CDM_2023_OCT"."DEID_PROCEDURES"  p
JOIN "DEIDENTIFIED_PCORNET_CDM"."CDM_2023_OCT"."DEID_DEMOGRAPHIC" d 
ON p.PATID = d.PATID
WHERE
      -- Procedure codes
      (
          -- ICD9 codes
          (
            regexp_like(p.PX,'^7[2-5]\.')
            and p.PX_TYPE = '09'
          )
          OR
          -- ICD10 codes
          (
            regexp_like(p.PX,'^10')
            and p.PX_TYPE = '10'
          )
          OR
          -- CPT codes
          (
            regexp_like(p.PX,'^59[0-9][0-9][0-9]')
            and p.PX_TYPE='CH'
          )
      )
      -- age restriction
      AND
      (
        (DATEDIFF(day, d.BIRTH_DATE,p.ADMIT_DATE)/365.25 ) BETWEEN 18 AND 89
      )
      -- time frame restriction
      AND p.ADMIT_DATE BETWEEN DATE '2010-01-01' AND CURRENT_DATE
      
CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."PREGNANCY_DX" AS
SELECT p.PATID, p.ADMIT_DATE 
FROM "ANALYTICS"."DMCOHORT"."PREGNANCY_DX" p
JOIN "ANALYTICS"."DMCOHORT"."ENC_FIRST_VISIT" e ON p.PATID=e.PATID

CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."DELIVERY_PROC" AS
SELECT d.PATID, d.ADMIT_DATE 
FROM "ANALYTICS"."DMCOHORT"."DELIVERY_PROC" d
JOIN "ANALYTICS"."DMCOHORT"."ENC_FIRST_VISIT" e ON d.PATID=e.PATID

CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."PREGNANCY_PATIDs" AS -- 117,175
SELECT p.PATID, p.ADMIT_DATE 
FROM "ANALYTICS"."DMCOHORT"."PREGNANCY_DX" p
UNION
SELECT d.PATID, d.ADMIT_DATE 
FROM "ANALYTICS"."DMCOHORT"."DELIVERY_PROC" d
order by PATID, ADMIT_DATE

-- Find separate pregnancy events (separated by >= 12 months from prior event)
CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."DISTINCT_PREG_EVENTS" AS
SELECT PATID
      ,ADMIT_DATE
      ,round(months_between(ADMIT_DATE, lag(ADMIT_DATE,1, NULL) OVER (PARTITION BY PATID ORDER BY ADMIT_DATE))) AS months_delta                             
      ,row_number() OVER (PARTITION BY PATID ORDER BY ADMIT_DATE) as rn                         
FROM "ANALYTICS"."DMCOHORT"."PREGNANCY_PATIDs"
ORDER BY PATID, ADMIT_DATE

CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."DISTINCT_PREG_EVENTS" AS
SELECT *
FROM "ANALYTICS"."DMCOHORT"."DISTINCT_PREG_EVENTS"
WHERE months_delta IS NULL OR months_delta >= 12
ORDER BY PATID, ADMIT_DATE

-- Number of distinct ADMIT_DATE
select min(cnt), max(cnt) 
from (select count(distinct ADMIT_DATE) as cnt
from "ANALYTICS"."DMCOHORT"."DISTINCT_PREG_EVENTS"
group by PATID)

select PATID 
from (select PATID, count(distinct ADMIT_DATE) as cnt
from "ANALYTICS"."DMCOHORT"."DISTINCT_PREG_EVENTS"
group by PATID)
where cnt=7

select * from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."MU_STUDY_DENOMINATOR"
where PATID='2597419'

-- Create a table with the PREGNANT EVENT PATIDs and all encounters from MU_STUDY_DENOMINATOR
CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."PREG_MERGE" AS --1,895,511
SELECT e.*, pevent.ADMIT_DATE AS PREG_EVENT, abs(DATEDIFF(day,PREG_EVENT,e.ADMIT_DATE)) AS DATEDIFF
FROM "ANALYTICS"."DMCOHORT"."MU_STUDY_DENOMINATOR" e
JOIN "ANALYTICS"."DMCOHORT"."DISTINCT_PREG_EVENTS" pevent
WHERE pevent.PATID = e.PATID

DELETE FROM "ANALYTICS"."DMCOHORT"."PREG_MERGE"
WHERE DATEDIFF > 365 --759,822

-- Mask eligible encounters within 1 year of any distinct pregnancies --12,009,952
CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."PREG_MASKED_ENCOUNTERS" AS
SELECT e.*
FROM "ANALYTICS"."DMCOHORT"."MU_STUDY_DENOMINATOR" e
WHERE NOT EXISTS (SELECT 1 FROM "ANALYTICS"."DMCOHORT"."PREG_MERGE" pevent
                 WHERE pevent.PATID = e.PATID AND pevent.ENCOUNTERID=e.ENCOUNTERID)

select min(DATEDIFF), max(DATEDIFF)
FROM "ANALYTICS"."DMCOHORT"."PREG_MERGE"
                  
SELECT count(distinct e.PATID) --24,141
FROM "ANALYTICS"."DMCOHORT"."MU_STUDY_DENOMINATOR" e
JOIN "ANALYTICS"."DMCOHORT"."DISTINCT_PREG_EVENTS" pevent
WHERE pevent.PATID = e.PATID
                  
SELECT count(distinct PATID) --782641
FROM "ANALYTICS"."DMCOHORT"."MU_STUDY_DENOMINATOR"


---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------
-----                                 Defining Diabetes Mellitus sample                                   -----
---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------
-----        People with HbA1c having two measures on different days within 2 years interval              -----
-----                                                                                                     -----
-----                         Lab should meet the following requerements:                                 -----
-----    Patient must be 18 years old >= age <= 89 years old during the lab ordering day.                 -----
-----    Lab value is >= 6.5 %.                                                                           -----
-----    Lab name is 'A1C' or                                                                             -----
-----    LOINC codes '17855-8', '4548-4','4549-2','17856-6','41995-2','59261-8','62388-4',                -----
-----    '71875-9','54039-3'                                                                              -----
-----    Lab should meet requirement for encounter types: 'AMBULATORY VISIT', 'EMERGENCY DEPARTMENT',     -----
-----    'INPATIENT HOSPITAL STAY', 'OBSERVATIONAL STAY', 'NON-ACUTE INSTITUTIONAL STAY'.                 -----
-----                                                                                                     -----
-----    In this Snowflake version of the code Patient age, encounter type and pregnancy masking is          -----
-----    accomplished by joining against the set of pregnancy masked eligible encounters                  -----
---------------------------------------------------------------------------------------------------------------

CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."ALL_A1C" AS
WITH A1C_LABS AS (
SELECT l.PATID
      ,l.LAB_ORDER_DATE
      ,l.LAB_LOINC
      ,l.RESULT_NUM
      ,l.RESULT_UNIT
      ,l.RAW_LAB_NAME
      ,l.RAW_FACILITY_CODE 
FROM "DEIDENTIFIED_PCORNET_CDM"."CDM_2023_OCT"."DEID_LAB_RESULT_CM" l
WHERE ( 
        (
         UPPER(l.RAW_LAB_NAME) like '%HEMOGLOBIN A1C%' OR 
         UPPER(l.RAW_LAB_NAME) like '%A1C%' OR 
         UPPER(l.RAW_LAB_NAME) like '%HA1C%' OR 
         UPPER(l.RAW_LAB_NAME) like '%HBA1C%'
         )
        OR
        l.LAB_LOINC IN (
                        '17855-8','4548-4','4549-2','17856-6',
                        '41995-2','59261-8','62388-4','71875-9','54039-3'
                        )
        OR
        /*--the new column RAW_FACILITY_CODE (available after v4.0) can be used to identify 
            additional lab based on local i2b2 concepts*/
        l.RAW_FACILITY_CODE in (
                                'KUH|COMPONENT_ID:2034' --KUMC specific
                                )
      )
      AND l.RESULT_NUM > 6.5
      AND UPPER(l.RESULT_UNIT) in ('%','PERCENT') --modified: 'PERCENT' can be recorded as '%'
      AND EXISTS (SELECT 1 FROM "ANALYTICS"."DMCOHORT"."PREG_MASKED_ENCOUNTERS" valid_encs
                           WHERE valid_encs.ENCOUNTERID = l.ENCOUNTERID)
)
SELECT * FROM A1C_LABS
ORDER BY PATID, LAB_ORDER_DATE
;

-- take the first date of the earlist pair of A1C results on distinct days within two years of each other
CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."A1C_FINAL_FIRSTPAIR" AS
WITH DELTA_A1C AS (
SELECT PATID
      ,LAB_ORDER_DATE
      ,CASE WHEN DATEDIFF(day,(lag(LAB_ORDER_DATE,1) over (partition by PATID order by LAB_ORDER_DATE asc)),LAB_ORDER_DATE)  BETWEEN 1 AND 365 * 2 
           THEN 1
           ELSE 0
       END AS WITHIN_TWO_YEARS
FROM "ANALYTICS"."DMCOHORT"."ALL_A1C"
), A1C_WITHIN_TWO_YEARS AS (
SELECT PATID
      ,LAB_ORDER_DATE
      ,row_number() OVER (PARTITION BY PATID ORDER BY LAB_ORDER_DATE) AS rn
FROM DELTA_A1C
WHERE WITHIN_TWO_YEARS = 1
)
SELECT PATID
      , LAB_ORDER_DATE AS EventDate
FROM A1C_WITHIN_TWO_YEARS
WHERE rn = 1;

---------------------------------------------------------------------------------------------------------------
-----     People with fasting glucose having two measures on different days within 2 years interval       -----
-----                                                                                                     -----
-----                         Lab should meet the following requerements:                                 -----
-----    Patient must be 18 years old >= age <= 89 years old during the lab ordering day.                 -----
-----    Lab value is >= 126 mg/dL.                                                                       -----
-----    (LOINC codes '1558-6',  '10450-5', '1554-5', '17865-7','35184-1' )                               -----
-----    Lab should meet requerement for encounter types: 'AMBULATORY VISIT', 'EMERGENCY DEPARTMENT',     -----
-----    'INPATIENT HOSPITAL STAY', 'EMERGENCY DEPARTMENT TO INPATIENT HOSPITAL STAY'.                    -----
-----                                                                                                     -----
-----                   The first pair of labs meeting requirements is collected.                         -----
-----   The date of the first fasting glucose lab out the first pair will be recorded as initial event.   -----
---------------------------------------------------------------------------------------------------------------
-----                                    May not be available in PCORNET                                  -----
---------------------------------------------------------------------------------------------------------------
-----    In this Snowflake version of the code Patient age, encounter type and pregnancy masking is          -----
-----    accomplished by joining against the set of pregnancy masked eligible encounters                  -----
---------------------------------------------------------------------------------------------------------------

CREATE OR REPLACE TABLE  "ANALYTICS"."DMCOHORT"."ALL_FG" AS
WITH FG_LABS AS (
SELECT l.PATID
      ,l.LAB_ORDER_DATE
      ,l.LAB_LOINC
      ,l.RESULT_NUM
      ,l.RESULT_UNIT
      ,l.RAW_LAB_NAME
      ,l.RAW_FACILITY_CODE 
FROM "DEIDENTIFIED_PCORNET_CDM"."CDM_2023_OCT"."DEID_LAB_RESULT_CM" l
WHERE (l.LAB_LOINC IN (
                       '1558-6', '10450-5', '1554-5', '17865-7', '35184-1'
                       )
       OR  
       l.RAW_FACILITY_CODE in (
                               'KUH|COMPONENT_ID:2035' -- KUMC specific
                               )
      )
      AND l.RESULT_NUM >= 126
      AND UPPER(l.RESULT_UNIT) = 'MG/DL' -- PCORNET_CDM 3.1 standardizes on uppercase lab units
      AND EXISTS (SELECT 1 FROM  "ANALYTICS"."DMCOHORT"."PREG_MASKED_ENCOUNTERS" valid_encs
                           WHERE valid_encs.ENCOUNTERID = l.ENCOUNTERID)
)
SELECT * FROM FG_LABS
ORDER BY PATID, LAB_ORDER_DATE
;

CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."FG_FINAL_FIRSTPAIR" AS
WITH DELTA_FG AS (
SELECT PATID
      ,LAB_ORDER_DATE
      ,CASE WHEN DATEDIFF(day,(lag(LAB_ORDER_DATE,1) over (partition by PATID order by LAB_ORDER_DATE asc)),LAB_ORDER_DATE) BETWEEN 1 AND 365 * 2 
           THEN 1
           ELSE 0
       END AS WITHIN_TWO_YEARS
FROM "ANALYTICS"."DMCOHORT"."ALL_FG"
), FG_WITHIN_TWO_YEARS AS (
SELECT PATID
      ,LAB_ORDER_DATE
      ,row_number() OVER (PARTITION BY PATID ORDER BY LAB_ORDER_DATE) AS rn
FROM DELTA_FG
WHERE WITHIN_TWO_YEARS = 1
)
SELECT PATID
      , LAB_ORDER_DATE AS EventDate
FROM FG_WITHIN_TWO_YEARS
WHERE rn = 1
;

---------------------------------------------------------------------------------------------------------------
-----     People with random glucose having two measures on different days within 2 years interval        -----
-----                                                                                                     -----
-----                         Lab should meet the following requerements:                                 -----
-----    Patient must be 18 years old >= age <= 89 years old during the lab ordering day.                 -----
-----    Lab value is >= 200 mg/dL.                                                                       -----
-----    (LOINC codes '2345-7', '2339-0','10450-5','17865-7','1554-5','6777-7','54246-4',                 -----
-----    '2344-0','41652-9')                                                                              -----
-----    Lab should meet requirement for encounter types: 'AMBULATORY VISIT', 'EMERGENCY DEPARTMENT',     -----
-----    'INPATIENT HOSPITAL STAY', 'OBSERVATIONAL STAY', 'NON-ACUTE INSTITUTIONAL STAY'.                 -----
-----                                                                                                     -----
---------------------------------------------------------------------------------------------------------------
-----                                    May not be available in PCORNET                                  -----
---------------------------------------------------------------------------------------------------------------
-----    In this Snowflake version of the code Patient age, encounter type and pregnancy masking is          -----
-----    accomplished by joining against the set of pregnancy masked eligible encounters                  -----
---------------------------------------------------------------------------------------------------------------

CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."ALL_RG" AS
WITH RG_LABS AS (
SELECT l.PATID
      ,l.LAB_ORDER_DATE
      ,l.LAB_LOINC
      ,l.RESULT_NUM
      ,l.RESULT_UNIT
      ,l.RAW_LAB_NAME
      ,l.RAW_FACILITY_CODE 
FROM "DEIDENTIFIED_PCORNET_CDM"."CDM_2023_OCT"."DEID_LAB_RESULT_CM" l
WHERE (
       l.LAB_LOINC IN (
                       '2345-7','2344-0','2339-0',
                       '10450-5','17865-7','1554-5','6777-7','54246-4','41652-9'
                       )
       OR
       l.RAW_FACILITY_CODE in (
                               'KUH|COMPONENT_ID:2010','KUH|COMPONENT_ID:2011','KUH|COMPONENT_ID:2012',
                               'KUH|COMPONENT_ID:2036','KUH|COMPONENT_ID:2037','KUH|COMPONENT_ID:2038',
                               'KUH|COMPONENT_ID:2039','KUH|COMPONENT_ID:2040','KUH|COMPONENT_ID:2041',
                               'KUH|COMPONENT_ID:2042','KUH|COMPONENT_ID:2043','KUH|COMPONENT_ID:3728',
                               'KUH|COMPONENT_ID:341','KUH|COMPONENT_ID:342','KUH|COMPONENT_ID:343',
                               'KUH|COMPONENT_ID:344','KUH|COMPONENT_ID:515',
                               'KUH|COMPONENT_ID:8012','KUH|COMPONENT_ID:8052','KUH|COMPONENT_ID:8087'
                               ) -- KUMC specific
      )
      AND l.RESULT_NUM >= 200
      AND UPPER(l.RESULT_UNIT) = 'MG/DL' -- PCORNET_CDM 3.1 standardizes on uppercase lab units
      AND EXISTS (SELECT 1 FROM "ANALYTICS"."DMCOHORT"."PREG_MASKED_ENCOUNTERS" valid_encs
                           WHERE valid_encs.ENCOUNTERID = l.ENCOUNTERID)
)
SELECT * FROM RG_LABS
ORDER BY PATID, LAB_ORDER_DATE
;

CREATE TABLE "ANALYTICS"."DMCOHORT"."RG_FINAL_FIRSTPAIR" AS
WITH DELTA_RG AS (
SELECT PATID
      ,LAB_ORDER_DATE
      ,CASE WHEN DATEDIFF(day,(lag(LAB_ORDER_DATE,1) over (partition by PATID order by LAB_ORDER_DATE asc)),LAB_ORDER_DATE) BETWEEN 1 AND 365 * 2 
           THEN 1
           ELSE 0
       END AS WITHIN_TWO_YEARS
FROM "ANALYTICS"."DMCOHORT"."ALL_RG"
), RG_WITHIN_TWO_YEARS AS (
SELECT PATID
      ,LAB_ORDER_DATE
      ,row_number() OVER (PARTITION BY PATID ORDER BY LAB_ORDER_DATE) AS rn
FROM DELTA_RG
WHERE WITHIN_TWO_YEARS = 1
)
SELECT PATID
      , LAB_ORDER_DATE AS EventDate
FROM RG_WITHIN_TWO_YEARS
WHERE rn = 1
;

---------------------------------------------------------------------------------------------------------------
-----     People with one random glucose & one HbA1c having both measures on different days within        -----
-----                                        2 years interval                                             -----
-----                                                                                                     -----
-----                         Lab should meet the following requirements:                                 -----
-----    Patient must be 18 years old >= age <= 89 years old during the lab ordering day.                 -----
-----    See corresponding sections above for the Lab values requerements.                                -----
-----    Lab should meet requirement for encounter types: 'AMBULATORY VISIT', 'EMERGENCY DEPARTMENT',     -----
-----    'INPATIENT HOSPITAL STAY', 'OBSERVATIONAL STAY', 'NON-ACUTE INSTITUTIONAL STAY'.                 -----
-----                                                                                                     -----
-----        The date of the first lab out the first pair will be recorded as initial event.              -----
---------------------------------------------------------------------------------------------------------------

CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."A1CRG_FINAL_FIRSTPAIR" AS
WITH A1C_RG_PAIRS AS (
select  ac.PATID
       ,CASE WHEN rg.LAB_ORDER_DATE < ac.LAB_ORDER_DATE
             THEN rg.LAB_ORDER_DATE
             ELSE ac.LAB_ORDER_DATE
        END AS EventDate
       ,row_number() over (partition by ac.PATID order by CASE WHEN rg.LAB_ORDER_DATE < ac.LAB_ORDER_DATE
                                                               THEN rg.LAB_ORDER_DATE
                                                               ELSE ac.LAB_ORDER_DATE
                                                          END) AS rn
from "ANALYTICS"."DMCOHORT"."ALL_A1C" ac
join "ANALYTICS"."DMCOHORT"."ALL_RG"  rg on ac.PATID = rg.PATID
WHERE ABS(DATEDIFF(day,rg.LAB_ORDER_DATE,ac.LAB_ORDER_DATE)) BETWEEN 1 AND 365 * 2
)
SELECT PATID
      ,EventDate
FROM A1C_RG_PAIRS
WHERE rn = 1
;

---------------------------------------------------------------------------------------------------------------
-----     People with one fasting glucose & one HbA1c having both measures on different days within       -----
-----                                        2 years interval                                             -----
-----                                                                                                     -----
-----                         Lab should meet the following requirements:                                 -----
-----    Patient must be 18 years old >= age <= 89 years old during the lab ordering day.                 -----
-----    See corresponding sections above for the Lab values requirements.                                -----
-----    Lab should meet requerement for encounter types: 'AMBULATORY VISIT', 'EMERGENCY DEPARTMENT',     -----
-----    'INPATIENT HOSPITAL STAY', 'OBSERVATIONAL STAY', 'NON-ACUTE INSTITUTIONAL STAY'.                 -----
-----                                                                                                     -----
-----        The date of the first lab out the first pair will be recorded as initial event.              -----
---------------------------------------------------------------------------------------------------------------

CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."A1CFG_FINAL_FIRSTPAIR" AS
WITH A1C_FG_PAIRS AS (
select  ac.PATID
       ,CASE WHEN fg.LAB_ORDER_DATE < ac.LAB_ORDER_DATE
             THEN fg.LAB_ORDER_DATE
             ELSE ac.LAB_ORDER_DATE
        END AS EventDate
       ,row_number() over (partition by ac.PATID order by CASE WHEN fg.LAB_ORDER_DATE < ac.LAB_ORDER_DATE
                                                               THEN fg.LAB_ORDER_DATE
                                                               ELSE ac.LAB_ORDER_DATE
                                                          END) AS rn
from "ANALYTICS"."DMCOHORT"."ALL_A1C" ac
join "ANALYTICS"."DMCOHORT"."ALL_FG"  fg on ac.PATID = fg.PATID
WHERE ABS(DATEDIFF(day,fg.LAB_ORDER_DATE,ac.LAB_ORDER_DATE)) BETWEEN 1 AND 365 * 2
)
SELECT PATID
      ,EventDate
FROM A1C_FG_PAIRS
WHERE rn = 1
;

---------------------------------------------------------------------------------------------------------------
-----               People with two visits (inpatient, outpatient, or emergency department)               -----
-----             relevant to type 1 Diabetes Mellitus or type 2 Diabetes Mellitus diagnosis              -----
-----                        recorded on different days within 2 years interval                           -----
-----                                                                                                     -----
-----                         Visit should meet the following requirements:                               -----
-----    Patient must be 18 years old >= age <= 89 years old during on the visit day.                     -----
-----    Visit should should be of encounter types: 'AMBULATORY VISIT', 'EMERGENCY DEPARTMENT',           -----
-----    'INPATIENT HOSPITAL STAY', 'OBSERVATIONAL STAY', 'NON-ACUTE INSTITUTIONAL STAY'.                 -----
-----                                                                                                     -----
-----                  The first pair of visits meeting requirements is collected.                        -----
-----     The date of the first visit out the first pair will be recorded as initial event.               -----
---------------------------------------------------------------------------------------------------------------
-----    In this Snowflake version of the code Patient age, encounter type and pregnancy masking is          -----
-----    accomplished by joining against the set of pregnancy masked eligible encounters                  -----
---------------------------------------------------------------------------------------------------------------
-- Get all visits of specified types for each patient sorted by date:

CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."DX_VISITS_INITIAL" AS
WITH DX_VISITS AS (
  SELECT d.PATID
        ,d.ADMIT_DATE
        ,d.DX
        ,d.DX_TYPE
  FROM "DEIDENTIFIED_PCORNET_CDM"."CDM_2023_OCT"."DEID_DIAGNOSIS" d
  WHERE(
        (
         (
             d.DX LIKE '250.%'
          or d.DX LIKE '357.2'
          or regexp_like(d.DX,'^362.0[1-7]')
         )
         AND DX_TYPE = '09'
        )
        OR
        (
         (
             regexp_like(d.DX,'^E1[01]\.')
          or d.DX LIKE 'E08.42'
          or d.DX LIKE 'E13.42'
         )
         AND DX_TYPE = '10'
        )
       )
       AND EXISTS (SELECT 1 FROM "ANALYTICS"."DMCOHORT"."PREG_MASKED_ENCOUNTERS" valid_encs
                   WHERE valid_encs.ENCOUNTERID = d.ENCOUNTERID)
)
SELECT * FROM DX_VISITS
ORDER BY PATID, ADMIT_DATE
;

CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."DX_VISIT_FINAL_FIRSTPAIR" AS
WITH DELTA_DX AS (
SELECT PATID
      ,ADMIT_DATE
      ,CASE WHEN DATEDIFF(day,(LAG(ADMIT_DATE,1) over (partition by PATID order by ADMIT_DATE asc)),ADMIT_DATE) BETWEEN 1 AND 365 * 2 
           THEN 1
           ELSE 0
       END AS WITHIN_TWO_YEARS
FROM "ANALYTICS"."DMCOHORT"."DX_VISITS_INITIAL"
), DX_WITHIN_TWO_YEARS AS (
SELECT PATID
      ,ADMIT_DATE
      ,row_number() OVER (PARTITION BY PATID ORDER BY ADMIT_DATE) AS rn
FROM DELTA_DX
WHERE WITHIN_TWO_YEARS = 1
)
SELECT PATID
      ,ADMIT_DATE AS EventDate
FROM DX_WITHIN_TWO_YEARS
WHERE rn = 1
;
---------------------------------------------------------------------------------------------------------------
-----            People with at least one ordered medications specific to Diabetes Mellitus               -----
-----                                                                                                     -----            
-----                         Medication should meet the following requirements:                          -----
-----     Patient must be 18 years old >= age <= 89 years old during the ordering of medication           -----
-----    Medication should relate to encounter types: 'AMBULATORY VISIT', 'EMERGENCY DEPARTMENT',         -----
-----    'INPATIENT HOSPITAL STAY', 'OBSERVATIONAL STAY', 'NON-ACUTE INSTITUTIONAL STAY'.                 -----
-----                                                                                                     -----
-----                The date of the first medication meeting requirements is collected.                  -----
---------------------------------------------------------------------------------------------------------------
--  Sulfonylurea:
-- collect meds based on matching names:

CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."SPECIFIC_MEDS" AS
SELECT a.PATID
      ,a.RX_ORDER_DATE
      ,a.RAW_RX_MED_NAME
      ,a.RXNORM_CUI
FROM "DEIDENTIFIED_PCORNET_CDM"."CDM_2023_OCT"."DEID_PRESCRIBING" a 
WHERE EXISTS (SELECT 1 FROM "ANALYTICS"."DMCOHORT"."PREG_MASKED_ENCOUNTERS" valid_encs
                       WHERE valid_encs.ENCOUNTERID = a.ENCOUNTERID) 
AND
(  
  -- Sufonylurea
   (
        UPPER(a.RAW_RX_MED_NAME) like UPPER('%Acetohexamide%') 
     or regexp_like(UPPER(a.RAW_RX_MED_NAME), UPPER('D[iy]melor')) 
     or regexp_like(UPPER(a.RAW_RX_MED_NAME), UPPER('glimep[ei]ride'))
     --This is combination of glimeperide-rosiglitazone :
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Avandaryl%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Amaryl%')
     --this is combination of glimepiride-pioglitazone:
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Duetact%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%gliclazide%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Uni Diamicron%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%glipizide%')
     --this is combination of metformin-glipizide : 
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Metaglip%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Glucotrol%')
     or regexp_like(UPPER(a.RAW_RX_MED_NAME),UPPER('Min[io]diab'))
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Glibenese%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Glucotrol XL%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Glipizide XL%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%glyburide%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Glucovance%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%glibenclamide%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%DiaBeta%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Glynase%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Micronase%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%chlorpropamide%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Diabinese%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Apo-Chlorpropamide%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Glucamide%') 
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Novo-Propamide%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Insulase%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%tolazamide%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Tolinase%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Glynase PresTab%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Tolamide%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%tolbutamide%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Orinase%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Tol-Tab%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Apo-Tolbutamide%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Novo-Butamide%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Glyclopyramide%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Deamelin-S%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Gliquidone%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Glurenorm%')
	 or UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLIBORNURIDE %')  
	 or UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLUTRIL%')  
	 or UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLIBORNURID%')  
	 or UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLIBORNURIDA%')  
	 or UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLIBORNURIDE%')  
	 or UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLIBORNURIDUM%')  
	 or UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLYMIDINE SODIUM%')  
	 or UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLYCODIAZINE%') 
	 or UPPER(a.RAW_RX_MED_NAME) like UPPER('%GONDAFON%')  
	 or UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLIDIAZINE%') 
	 or UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLYMIDINE%')
     -- RXNORM_CUI is a VARCHAR field in PCORNet CDM spec, Oracle autoconverts numbers to varchars in comparisons but is very inefficient
     -- choose the appropriate version for the predicate based on if you use a VARCHAR or numeric RXCUI in your local implementation
     -- or a.RXNORM_CUI in (3842,153843,153844,153845,197306,197307,197495,197496,197737,198291,198292,198293,198294,199245,199246,199247,199825,201056,201057,201058,201059,201060,201061,201062,201063,201064,201919,201921,201922,203289,203295,203679,203680,203681,205828,205830,205872,205873,205875,205876,205879,205880,207953,207954,207955,208012,209204,214106,214107,217360,217364,217370,218942,220338,221173,224962,227211,241604,241605,245266,246391,246522,246523,246524,250919,252259,252960,260286,260287,261351,261974,284743,285129,310488,310489,310490,310534,310536,310537,310539,313418,313419,314000,314006,315107,315239,315273,315274,315647,315648,315978,315979,315980,315987,315988,315989,315990,315991,315992,316832,316833,316834,316835,316836,317379,317637,328851,330349,331496,332029,332808,332810,333394,336701,351452,352381,352764,353028,362611,367762,368204,368586,368696,368714,369297,369304,369373,369500,369557,369562,370529,371465,371466,371467,372318,372319,372320,372333,372334,374149,374152,374635,375952,376236,376868,378730,379559,379565,379568,379570,379572,379802,379803,379804,380849,389137,391828,393405,393406,405121,429841,430102,430103,430104,430105,432366,432780,432853,433856,438506,440285,440286,440287,465455,469978,542029,542030,542031,542032,563154,564035,564036,564037,564038,565327,565408,565409,565410,565667,565668,565669,565670,565671,565672,565673,565674,565675,566055,566056,566057,566718,566720,566761,566762,566764,566765,566768,566769,568684,568685,568686,568742,569831,573945,573946,574089,574090,574571,574612,575377,600423,600447,602543,602544,602549,602550,606253,607784,607816,647208,647235,647236,647237,647239,669981,669982,669983,669984,669985,669986,669987,687730,700835,706895,706896,731455,731457,731461,731462,731463,827400,844809,844824,844827,847706,847707,847708,847710,847712,847714,847716,847718,847720,847722,847724,849585,861731,861732,861733,861736,861737,861738,861740,861741,861742,861743,861745,861747,861748,861750,861752,861753,861755,861756,861757,865567,865568,865569,865570,865571,865572,865573,865574,881404,881405,881406,881407,881408,881409,881410,881411,1007411,1007582,1008873,1120401,1125922,1128359,1130921,1132391,1132805,1135219,1135428,1147918,1153126,1153127,1155467,1155468,1155469,1155470,1155471,1155472,1156197,1156198,1156199,1156200,1156201,1157121,1157122,1157240,1157241,1157242,1157243,1157244,1157245,1157246,1157247,1157642,1157643,1157644,1165203,1165204,1165205,1165206,1165207,1165208,1165845,1169680,1169681,1170663,1170664,1171233,1171234,1171246,1171247,1171248,1171249,1171933,1171934,1173427,1173428,1175658,1175659,1175878,1175879,1175880,1175881,1176496,1176497,1177973,1177974,1178082,1178083,1179112,1179113,1183952,1183954,1183958,1185049,1185624,1309022,1361492,1361493,1361494,1361495,1384487,1428269,1741234)
     or a.RXNORM_CUI in ('1007411','1007582','1008873','102845','102846','102847','102848','102849','102850','105369','105371','105372','105373','105374','10633','10635','1153126','1153127','1155467','1155468','1155469','1155470','1155471','1155472','1156197','1156198','1156199','1156200','1156201','1157121','1157122','1157238','1157239','1157240','1157241','1157242','1157243','1157244','1157245','1157246','1157247','1157642','1157643','1157644','1157805','1157806','1165203','1165204','1165205','1165206','1165207','1165208','1165845','1169680','1169681','1170663','1170664','1171233','1171234','1171246','1171247','1171929','1171930','1171933','1171934','1171961','1171962','1173427','1173428','1175878','1175879','1175880','1175881','1176496','1176497','1177973','1177974','1178082','1178083','1179112','1179113','1179291','1179292','1183952','1183954','1183958','1361492','1361493','1361494','1361495','151615','151616','151822','153591','153592','153842','153843','153844','153845','173','197306','197307','197495','197496','197737','198291','198292','198293','198294','199245','199246','199247','199825','201056','201057','201058','201059','201060','201061','201062','201063','201064','201919','201921','201922','203289','203295','203679','203680','203681','205828','205830','205872','205873','205875','205876','205879','205880','207953','207954','207955','208012','209204','214106','214107','217360','217364','217370','218942','220338','221173','227211','2404','245266','246391','246522','246523','246524','250919','252259','252960','25789','25793','260286','260287','261351','261532','261974','285129','310488','310489','310490','310534','310536','310537','310539','313418','313419','314000','314006','315107','315239','315273','315274','315647','315648','315978','315979','315980','315987','315988','315989','315990','315991','315992','316832','316833','316834','316835','316836','317379','317637','328851','330349','331496','332029','332808','332810','333394','336701','351452','352381','353028','353626','358839','358840','362611','367762','368204','368586','368696','368714','369297','369304','369373','369500','369555','369557','369562','370529','371465','371466','371467','372318','372319','372320','372333','372334','374149','374152','374635','375952','376236','376868','378730','378822','378823','379559','379565','379568','379570','379572','379802','379803','379804','380849','389137','393405','393406','429841','430102','430103','430104','430105','432366','432780','432853','433856','438506','440285','440286','440287','4815','4816','4821','542029','542030','542031','542032','563154','563155','564035','564036','564037','564038','565327','565408','565409','565410','565667','565668','565669','565670','565671','565672','565673','565674','565675','566055','566056','566057','566718','566720','566761','566762','566764','566765','566768','566769','568684','568685','568686','568742','569831','573945','573946','574089','574090','574571','574612','575377','602543','602544','602549','602550','606253','647235','647236','647237','647239','669981','669982','669983','669984','669985','669986','669987','700835','706895','706896','731455','731457','731461','731462','731463','844809','844824','844827','847706','847707','847708','847710','847712','847714','847716','847718','847720','847722','847724','849585','861731','861732','861733','861736','861737','861738','861740','861741','861742','861743','861745','861747','861748','861750','861752','861753','861755','861756','861757','865567','865568','865569','865570','865571','865572','865573','865574','881404','881405','881406','881407','881408','881409','881410','881411','93312')
   )
OR
  --  Alpha-glucosidase inhibitor:
   (
        UPPER(a.RAW_RX_MED_NAME) like UPPER('%acarbose%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Precose%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Glucobay%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%miglitol%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Glyset%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Voglibose%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Basen%')
     -- RXNORM_CUI is a VARCHAR field in PCORNet CDM spec, Oracle autoconverts numbers to varchars in comparisons but is very inefficient
     -- choose the appropriate version for the predicate based on if you use a VARCHAR or numeric RXCUI in your local implementation
     -- or a.RXNORM_CUI in (16681,30009,137665,151826,199149,199150,200132,205329,205330,205331,209247,209248,213170,213485,213486,213487,217372,315246,315247,315248,316304,316305,316306,368246,368300,370504,372926,569871,569872,573095,573373,573374,573375,1153649,1153650,1157268,1157269,1171936,1171937,1185237,1185238,1598393,1741321)
     or a.RXNORM_CUI in ('1153649','1153650','1157268','1157269','1171936','1171937','1185237','1185238','151826','16681','199149','199150','200132','205329','205330','205331','209247','209248','213170','213485','213486','213487','217372','30009','315246','315247','315248','316304','316305','316306','368246','368300','370504','372926','569871','569872','573095','573373','573374','573375')
   )
OR
  --Glucagon-like Peptide-1 Agonists:
   (
        UPPER(a.RAW_RX_MED_NAME) like UPPER('%Lixisenatide%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Adlyxin%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Lyxumia%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Albiglutide%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Tanzeum%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Eperzan%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Dulaglutide%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Trulicity%')
     -- RXNORM_CUI is a VARCHAR field in PCORNet CDM spec, Oracle autoconverts numbers to varchars in comparisons but is very inefficient
     -- choose the appropriate version for the predicate based on if you use a VARCHAR or numeric RXCUI in your local implementation
     -- or a.RXNORM_CUI in (1440051,1440052,1440053,1440056,1534763,1534797,1534798,1534800,1534801,1534802,1534804,1534805,1534806,1534807,1534819,1534820,1534821,1534822,1534823,1534824,1551291,1551292,1551293,1551295,1551296,1551297,1551299,1551300,1551301,1551302,1551303,1551304,1551305,1551306,1551307,1551308,1593645,1649584,1649586,1659115,1659117,1803885,1803886,1803887,1803888,1803889,1803890,1803891,1803892,1803893,1803894,1803895,1803896,1803897,1803898,1803902,1803903)
     or a.RXNORM_CUI in ('1440051','1440052','1440053','1440056','1534763','1534797','1534798','1534800','1534801','1534802','1534804','1534805','1534806','1534807','1534819','1534820','1534821','1534822','1534823','1534824','1551291','1551292','1551293','1551295','1551296','1551297','1551299','1551300','1551301','1551302','1551303','1551304','1551305','1551306','1551307','1551308','1649584','1649586','1659115','1659117','1803885','1803886','1803887','1803888','1803889','1803890','1803891','1803892','1803893','1803894','1803895','1803896','1803897','1803898','1803902','1803903','1858991','1858992','1858993','1858994','1858995','1858997','1858998','1859000','1859001','1859002')
   )
OR   
  --  Dipeptidyl peptidase IV inhibitor:
   (
        UPPER(a.RAW_RX_MED_NAME) like UPPER('%alogliptin%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Kazano%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Oseni%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Nesina%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Anagliptin%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Suiny%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%linagliptin%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Jentadueto%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Jentadueto XR%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Glyxambi%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Tradjenta%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%saxagliptin%')
     --this is combination of metformin-saxagliptin :
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Kombiglyze XR%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Onglyza%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%sitagliptin%')
     --this is combination of metformin-vildagliptin :
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Eucreas%')
     --this is combination of sitagliptin-simvastatin:
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Juvisync%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Epistatin%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Synvinolin%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Zocor%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Janumet%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Janumet XR%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Januvia%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Teneligliptin%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Tenelia%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Vildagliptin%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Galvus%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Zomelis%')
     -- RXNORM_CUI is a VARCHAR field in PCORNet CDM spec, Oracle autoconverts numbers to varchars in comparisons but is very inefficient
     -- choose the appropriate version for the predicate based on if you use a VARCHAR or numeric RXCUI in your local implementation
     -- or a.RXNORM_CUI in (36567,104490,104491,152923,196503,208220,213319,368276,563653,563654,565109,568935,573220,593411,596554,621590,638596,665031,665032,665033,665034,665035,665036,665037,665038,665039,665040,665041,665042,665043,665044,669475,700516,729717,757603,757708,757709,757710,757711,757712,857974,858034,858035,858036,858037,858038,858039,858040,858041,858042,858043,858044,861769,861770,861771,861819,861820,861821,1043560,1043561,1043562,1043563,1043565,1043566,1043567,1043568,1043569,1043570,1043572,1043574,1043575,1043576,1043578,1043580,1043582,1043583,1043584,1048346,1100699,1100700,1100701,1100702,1100703,1100704,1100705,1100706,1128666,1130631,1132606,1145961,1158518,1158519,1159662,1159663,1161605,1161606,1161607,1161608,1164580,1164581,1164670,1164671,1167810,1167811,1167814,1167815,1179163,1179164,1181729,1181730,1187973,1187974,1189800,1189801,1189802,1189803,1189804,1189806,1189808,1189810,1189811,1189812,1189813,1189814,1189818,1189821,1189823,1189827,1243015,1243016,1243017,1243018,1243019,1243020,1243022,1243026,1243027,1243029,1243033,1243034,1243036,1243037,1243038,1243039,1243040,1243826,1243827,1243829,1243833,1243834,1243835,1243839,1243842,1243843,1243844,1243845,1243846,1243848,1243849,1243850,1312409,1312411,1312415,1312416,1312418,1312422,1312423,1312425,1312429,1365802,1368000,1368001,1368002,1368003,1368004,1368005,1368006,1368007,1368008,1368009,1368010,1368011,1368012,1368017,1368018,1368019,1368020,1368033,1368034,1368035,1368036,1368381,1368382,1368383,1368384,1368385,1368387,1368391,1368392,1368394,1368395,1368396,1368397,1368398,1368399,1368400,1368401,1368402,1368403,1368405,1368409,1368410,1368412,1368416,1368417,1368419,1368423,1368424,1368426,1368430,1368431,1368433,1368434,1368435,1368436,1368437,1368438,1368440,1368444,1372692,1372706,1372717,1372738,1372754,1431025,1431048,1546030,1598392,1602106,1602107,1602108,1602109,1602110,1602111,1602112,1602113,1602114,1602115,1602118,1602119,1602120,1692194,1727500,1741248,1741249,1791055,1796088,1796089,1796090,1796091,1796092,1796093,1796094,1796095,1796096,1796097,1796098,1803420)
     or a.RXNORM_CUI in ('1043560','1043561','1043562','1043563','1043565','1043566','1043567','1043568','1043569','1043570','1043572','1043574','1043575','1043576','1043578','1043580','1043582','1043583','1043584','1158518','1158519','1159662','1159663','1161605','1161606','1161607','1161608','1164580','1164581','1167814','1167815','1181729','1181730','1189800','1189801','1189802','1189803','1189804','1189806','1189808','1189810','1189811','1189814','1189818','1189821','1189823','1189827','1243826','1243827','1243829','1243833','1243834','1243835','1243839','1243842','1243843','1243844','1243845','1243846','1243848','1243849','1243850','1312409','1312411','1312415','1312416','1312418','1312422','1312423','1312425','1312429','1368008','1368009','1368012','1368019','1368020','1368035','1368036','1546030','1727500','1925495','1925496','1925497','1925498','1925500','1925501','1925504','593411','596554','621590','638596','665031','665032','665033','665034','665035','665036','665037','665038','665039','665040','665041','665042','665043','665044','700516','729717','757603','757708','757709','757710','757711','757712','857974','858034','858035','858036','858037','858038','858039','858040','858041','858042','858043','858044','861769','861770','861771','861819','861820','861821')
   )
OR
  -- Meglitinide:
   (
        UPPER(a.RAW_RX_MED_NAME) like UPPER('%nateglinide%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Starlix%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Prandin%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%NovoNorm%')
     -- RXNORM_CUI is a VARCHAR field in PCORNet CDM spec, Oracle autoconverts numbers to varchars in comparisons but is very inefficient
     -- choose the appropriate version for the predicate based on if you use a VARCHAR or numeric RXCUI in your local implementation
     -- or a.RXNORM_CUI in (213218,213219,213220,219335,226911,226912,226913,226914,274332,284529,284530,311919,314142,330385,330386,368289,374648,389139,393408,402943,402944,402959,430491,430492,446631,446632,573136,573137,573138,574042,574043,574044,574957,574958,1158396,1158397,1178121,1178122,1178433,1178434,1184631,1184632)
     or a.RXNORM_CUI in ('1157407','1157408','1158396','1158397','1161599','1161600','1178121','1178122','1178433','1178434','1184631','1184632','200256','200257','200258','213218','213219','213220','219335','226911','226912','226913','226914','274332','284529','284530','311919','314142','316630','316631','330385','330386','331433','368289','373759','374648','389139','393408','402943','402944','402959','430491','430492','446631','446632','573136','573137','573138','574042','574043','574044','574957','574958','73044','802646','802742','805670','861787','861788','861789','861790','861791','861792')
   )
OR
  --  Amylinomimetics:
   (
        UPPER(a.RAW_RX_MED_NAME) like UPPER('%Pramlintide%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Symlin%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%SymlinPen 120%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%SymlinPen 60%')
     -- RXNORM_CUI is a VARCHAR field in PCORNet CDM spec, Oracle autoconverts numbers to varchars in comparisons but is very inefficient
     -- choose the appropriate version for the predicate based on if you use a VARCHAR or numeric RXCUI in your local implementation
     -- or a.RXNORM_CUI in (139953,356773,356774,486505,582702,607296,753370,753371,759000,861034,861036,861038,861039,861040,861041,861042,861043,861044,861045,1161690,1185508,1360096,1360184,1657563,1657565,1657792)
     or a.RXNORM_CUI in ('1161690','1185508','1360096','1360184','139953','1657563','1657565','356773','356774','486505','582702','759000','861034','861036','861038','861039','861040','861041','861042','861043','861044','861045')
   )
OR
  --  Insulin:
   (
        UPPER(a.RAW_RX_MED_NAME) like UPPER('%Insulin aspart%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%NovoLog%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Insulin glulisine%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Apidra%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Insulin lispro%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Humalog%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Insulin inhaled%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Afrezza%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Regular insulin%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Humulin R%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Novolin R%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Insulin NPH%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Humulin N%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Novolin N%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Insulin detemir%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Levemir%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Insulin glargine%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Lantus%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Lantus SoloStar%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Toujeo%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Basaglar%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Insulin degludec%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Tresiba%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Insulin aspart protamine%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Insulin aspart%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Actrapid%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Hypurin%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Iletin%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Insulatard%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Insuman%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Mixtard%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%NovoMix%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%NovoRapid%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Oralin%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Abasaglar%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%V-go%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Ryzodeg%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Insulin lispro protamine%')
	 or UPPER(a.RAW_RX_MED_NAME) like UPPER('insulin lispro%')
     -- RXNORM_CUI is a VARCHAR field in PCORNet CDM spec, Oracle autoconverts numbers to varchars in comparisons but is very inefficient
     -- choose the appropriate version for the predicate based on if you use a VARCHAR or numeric RXCUI in your local implementation
     -- or a.RXNORM_CUI in (5856,6926,51428,86009,92880,92881,92942,93398,93558,93560,106888,106889,106891,106892,106894,106895,106896,106899,106900,106901,108407,108813,108814,108815,108816,108822,135805,139825,142138,150659,150831,150973,150974,150978,152598,152599,152602,152640,152644,152647,153383,153384,153386,153389,154992,203209,213442,217704,217705,217706,217707,217708,225569,226290,226291,226292,226293,226294,242120,245264,245265,247511,247512,247513,249026,249220,253181,253182,259111,260265,261111,261112,261420,261542,261551,274783,284810,285018,307383,311021,311026,311027,311028,311030,311033,311034,311035,311036,311040,311041,311048,311049,311051,311052,311059,314684,340325,340326,340327,343076,343083,343226,343258,343263,343663,349670,351297,351857,351858,351859,351860,351861,351862,351926,352385,352386,359125,359126,359127,360894,362585,362622,362777,363120,363150,363221,363534,365573,365583,365668,365670,365672,365674,365677,365679,365680,366206,372909,372910,375170,376915,378841,378857,378864,378966,379734,379740,379744,379745,379746,379747,379750,379756,379757,384982,385896,386083,386084,386086,386087,386088,386089,386091,386092,386098,388513,392660,400008,400560,405228,412453,412978,415088,415089,415090,415184,415185,440399,440650,440653,440654,451437,451439,466467,466468,467015,484320,484321,484322,485210,564390,564391,564392,564395,564396,564397,564399,564400,564401,564531,564601,564602,564603,564605,564766,564820,564881,564882,564885,564994,564995,564998,565176,565253,565254,565255,565256,573330,573331,574358,574359,575068,575137,575141,575142,575143,575146,575147,575148,575151,575626,575627,575628,575629,575679,607583,615900,615908,615909,615910,615992,616236,616237,616238,633703,636227,658226,668934,723550,724231,724343,727907,728543,731277,731280,731281,752386,752388,761522,796006,796386,801808,803192,803193,803194,816726,834989,834990,834992,835225,835226,835227,835228,835868,847186,847187,847188,847189,847191,847194,847198,847199,847200,847201,847202,847203,847204,847205,847211,847213,847230,847232,847239,847241,847252,847254,847256,847257,847259,847261,847278,847279,847343,847417,849095,865097,865098,977838,977840,977841,977842,1008501,1045051,1069670,1087799,1087800,1087801,1087802,1132383,1136628,1136712,1140739,1140763,1157459,1157460,1157461,1160696,1164093,1164094,1164095,1164824,1167138,1167139,1167140,1167141,1167142,1167934,1168563,1171289,1171291,1171292,1171293,1171295,1171296,1172691,1172692,1175624,1176722,1176723,1176724,1176725,1176726,1176727,1176728,1177009,1178119,1178120,1178127,1178128,1183426,1183427,1184075,1184076,1184077,1246223,1246224,1246225,1246697,1246698,1246699,1260529,1295992,1296093,1309028,1359484,1359581,1359684,1359700,1359712,1359719,1359720,1359855,1359856,1359934,1359936,1360036,1360058,1360172,1360226,1360281,1360383,1360435,1360482,1362705,1362706,1362707,1362708,1362711,1362712,1362713,1362714,1362719,1362720,1362721,1362722,1362723,1362724,1362725,1362726,1362727,1362728,1362729,1362730,1362731,1362732,1372685,1372741,1374700,1374701,1377831,1435649,1456746,1535271,1538910,1543200,1543201,1543202,1543203,1543205,1543206,1543207,1544488,1544490,1544568,1544569,1544570,1544571,1593805,1598498,1598618,1604538,1604539,1604540,1604541,1604543,1604544,1604545,1604546,1604550,1605101,1607367,1607643,1607992,1607993,1650256,1650260,1650262,1650264,1651315,1651572,1651574,1652237,1652238,1652239,1652240,1652241,1652242,1652243,1652244,1652639,1652640,1652641,1652642,1652643,1652644,1652645,1652646,1652647,1652648,1652754,1653104,1653106,1653196,1653197,1653198,1653200,1653202,1653203,1653204,1653206,1653209,1653449,1653468,1653496,1653497,1653499,1653506,1653899,1654060,1654190,1654192,1654341,1654348,1654379,1654380,1654381,1654651,1654850,1654855,1654857,1654858,1654862,1654863,1654866,1654909,1654910,1654911,1654912,1655063,1656705,1656706,1660643,1663228,1663229,1664772,1665830,1668430,1668441,1668442,1668448,1670007,1670008,1670009,1670010,1670011,1670012,1670013,1670014,1670015,1670016,1670017,1670018,1670020,1670021,1670022,1670023,1670024,1670025,1670404,1670405,1716525,1717038,1717039,1719496,1720524,1721033,1721039,1727493,1731314,1731315,1731316,1731317,1731318,1731319,1736613,1736859,1736860,1736861,1736862,1736863,1736864,1743273,1792701,1798387,1798388,1804446,1804447,1804505,1804506)
     or a.RXNORM_CUI in ('106888','106889','106891','1087799','108813','108814','108815','108816','108822','1157459','1160696','1164093','1164094','1164095','1167138','1167139','1167140','1167141','1167142','1167934','1168563','1171292','1171293','1171295','1171296','1172691','1172692','1175624','1176722','1176723','1176724','1176725','1176726','1176727','1176728','1177009','1178119','1178120','1178127','1178128','1184075','1184076','1184077','1359484','1359684','1359719','1359855','1359856','1359934','1360281','1360383','1360435','1362705','1362706','1362707','1362708','1362711','1362712','1362713','1362714','1362719','1362720','1362721','1362722','1362723','1362724','1362725','1362726','1362727','1362728','1362729','1362730','1362731','1362732','1372685','1372741','139825','142138','150831','150973','150978','152598','152599','152602','152640','152644','152647','153383','153384','153386','153389','1543203','1543205','1543206','1543207','1544490','1544569','1544571','1604538','1604539','1604540','1604541','1604543','1604544','1604545','1604546','1650260','1650264','1651315','1653197','1653198','1653200','1653203','1653204','1653206','1653209','1653497','1653499','1653506','1654190','1654192','1654863','1654866','1654911','1654912','1656706','1670007','1670008','1670009','1670010','1670011','1670012','1670013','1670014','1670015','1670016','1670017','1670018','1670020','1670021','1670022','1670023','1670024','1670025','1727493','1731316','1731317','1731319','1736859','1736860','1736861','1736862','1736863','1736864','1798388','1858992','1858993','1858994','1858995','1858997','1858998','1859000','1859001','1859002','1860165','1860166','1860167','1860169','1860170','1860172','1860173','1860174','1862102','203209','217704','217705','217707','217708','225569','226290','226291','226292','226293','226294','261111','261112','261551','274783','284810','285018','311021','311026','311027','311030','311033','311036','311041','343226','349670','351857','351858','351859','351860','351926','362585','362622','362777','363150','363221','363534','365573','365583','365670','365674','365677','365680','366206','372909','372910','375170','378864','379740','379744','379745','379746','379747','379750','379756','379757','384982','386083','386084','386086','386087','386088','386089','386091','386092','386098','400560','405228','484320','484321','484322','485210','564390','564391','564392','564601','564602','564603','564605','564820','564881','564885','564994','564995','564998','565176','565253','565254','565255','565256','574358','574359','575068','575137','575141','575142','575143','575146','575148','575626','575627','575628','575629','575679','607583','616236','616237','616238','6926','724343','803192','803193','803194','847198','847199','847200','847201','847204','847205','847230','847232','847239','847241','847259','847261','847279','900788','92880','92881','92942','93398','93558','93560','977838','977840','977841','977842')

   )
OR
  --  Sodium glucose cotransporter (SGLT) 2 inhibitors:
   (
        UPPER(a.RAW_RX_MED_NAME) like UPPER('%dapagliflozin%')
     or regexp_like(UPPER(a.RAW_RX_MED_NAME), UPPER('F[ao]rxiga'))
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%canagliflozin%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Invokana%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Invokamet%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Xigduo XR%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Sulisent%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%empagliflozin%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Jardiance%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Synjardy%')
     --this one is combination of linagliptin-empagliflozin, see also Dipeptidyl Peptidase IV Inhibitors section
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Glyxambi%')
     -- RXNORM_CUI is a VARCHAR field in PCORNet CDM spec, Oracle autoconverts numbers to varchars in comparisons but is very inefficient
     -- choose the appropriate version for the predicate based on if you use a VARCHAR or numeric RXCUI in your local implementation
     -- or a.RXNORM_CUI in (1373458,1373459,1373460,1373461,1373462,1373463,1373464,1373465,1373466,1373467,1373468,1373469,1373470,1373471,1373472,1373473,1422532,1486436,1486966,1486977,1486981,1488564,1488565,1488566,1488567,1488568,1488569,1488573,1488574,1493571,1493572,1534343,1534344,1534397,1540290,1540292,1545145,1545146,1545147,1545148,1545149,1545150,1545151,1545152,1545153,1545154,1545155,1545156,1545157,1545158,1545159,1545160,1545161,1545162,1545163,1545164,1545165,1545166,1545653,1545654,1545655,1545656,1545657,1545658,1545659,1545660,1545661,1545662,1545663,1545664,1545665,1545666,1545667,1545668,1546031,1592709,1592710,1592722,1593057,1593058,1593059,1593068,1593069,1593070,1593071,1593072,1593073,1593774,1593775,1593776,1593826,1593827,1593828,1593829,1593830,1593831,1593832,1593833,1593835,1598392,1598430,1602106,1602107,1602108,1602109,1602110,1602111,1602112,1602113,1602114,1602115,1602118,1602119,1602120,1655477,1664310,1664311,1664312,1664313,1664314,1664315,1664316,1664317,1664318,1664319,1664320,1664321,1664322,1664323,1664324,1664325,1664326,1664327,1664328,1665367,1665368,1665369,1683935,1727500)
     or a.RXNORM_CUI in ('1163230','1163790','1169415','1186578','1242961','1242963','1242964','1242965','1242967','1242968','1359640','1359802','1359979','1360105','1360454','1360495','1544916','1544918','1544919','1544920','1598264','1598265','1598267','1598268','1598269','1653594','1653597','1653600','1653610','1653611','1653613','1653614','1653616','1653619','1653625','1727493','1860164','1860165','1860166','1860167','1860169','1860170','1860172','1860173','1860174','475968','604751','60548','847908','847910','847911','847913','847914','847915','847916','847917','897120','897122','897123','897124','897126')
   )
OR
  --  Combinations:
  (
   a.RXNORM_CUI in ('1043567','1043574','1043582','104490','104491','1166403','1166404','1167810','1167811','1171248','1171249','1175658','1175659','1184627','1184628','1185049','1185624','1187973','1187974','1189806','1189810','1189811','1189812','1189813','1189814','1189818','1189823','1189827','1243022','1243026','1243029','1243033','1243036','1243037','1243038','1243039','1243040','1243829','1243833','1243835','1243839','1243843','1243845','1243848','1243850','1312411','1312415','1312418','1312422','1312425','1312429','1368387','1368391','1368394','1368395','1368396','1368397','1368398','1368405','1368409','1368412','1368416','1368419','1368423','1368426','1368430','1368433','1368434','1368435','1368436','1368437','1368440','1368444','1372692','1372706','1372716','1372717','1372738','1372754','152923','1545151','1545152','1545153','1545154','1545155','1545156','1545158','1545159','1545162','1545163','1545165','1545166','1593775','1593831','1593833','1593835','1602110','1602111','1602112','1602113','1602114','1602115','1602119','1602120','1796090','1796091','1796093','1796095','1796096','1796098','1810998','1810999','1811001','1811003','1811005','1811007','1811009','1811011','1811013','1940498','196503','208220','213319','284743','352764','368276','563653','563654','565109','568935','573220','607816','647208','731455','731457','731461','731462','731463','757603','805670','847706','847707','847708','847710','847712','847714','847716','847718','847720','847722','847724','849585','861732','861733','861737','861738','861741','861742','861745','861747','861750','861752','861755','861756','861757','861770','861771','861788','861789','861791','861792','861820','861821')
   )
)
;

---------------------------------------------------------------------------------------------------------------
-----         The date of the first medication of any kind will be recorded for each patient              -----
---------------------------------------------------------------------------------------------------------------
CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."INCLUSIONMEDS_FINAL" AS
SELECT PATID
      ,MIN(RX_ORDER_DATE) AS EventDate
FROM "ANALYTICS"."DMCOHORT"."SPECIFIC_MEDS"
GROUP BY PATID
;

---------------------------------------------------------------------------------------------------------------
-----           People with at least one ordered medications non-specific to Diabetes Mellitus            -----
-----                                                   &                                                 -----
-----one lab or one visit record described above. Both recorded on different days within 2 years interval.-----
-----                                                                                                     -----
-----           Medication and another encounter should meet the following requirements:                  -----
-----        Patient must be 18 years old >= age <= 89 years old during the recorded encounter            -----
-----     Encounter should relate to encounter types: 'AMBULATORY VISIT', 'EMERGENCY DEPARTMENT',         -----
-----    'INPATIENT HOSPITAL STAY', 'OBSERVATIONAL STAY', 'NON-ACUTE INSTITUTIONAL STAY'.                 -----
-----                                                                                                     -----
-----                The date of the first medication meeting requirements is collected.                  -----
---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------
-----                  People with medications non-specific to Diabetes Mellitus                          -----
-----                                 meeting one more requirement                                        -----
-----                         18 >= Age <=89 during the lab ordering day                                  -----
-----                    the date the first time med is recorded will be used                             -----
---------------------------------------------------------------------------------------------------------------
-----    In this Snowflake version of the code Patient age, encounter type and pregnancy masking is          -----
-----    accomplished by joining against the set of pregnancy masked eligible encounters                  -----
---------------------------------------------------------------------------------------------------------------

CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."INCL_RESTRICT_MEDS" AS
SELECT a.PATID
      ,a.RX_ORDER_DATE
      ,a.RAW_RX_MED_NAME
      ,a.RXNORM_CUI
FROM "DEIDENTIFIED_PCORNET_CDM"."CDM_2023_OCT"."DEID_PRESCRIBING" a
WHERE EXISTS (SELECT 1 FROM "ANALYTICS"."DMCOHORT"."PREG_MASKED_ENCOUNTERS" valid_encs
                       WHERE valid_encs.ENCOUNTERID = a.ENCOUNTERID)
AND
(
   --  Biguanide:
   (
        UPPER(a.RAW_RX_MED_NAME) like UPPER('%Glucophage%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Fortamet%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Glumetza%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Riomet%')
     or
     (
		UPPER(a.RAW_RX_MED_NAME) like UPPER('%METFORMIN %') 
		and not (
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%ACARBOSE %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%MIGLITOL%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%VOGLIBOSE%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%ALOGLIPTIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%ANAGLIPTIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%LINAGLIPTIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%SAXAGLIPTIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%SITAGLIPTIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%TENELIGLIPTIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%VILDAGLIPTIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%LIXISENATIDE%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%ALBIGLUTIDE%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%DULAGLUTIDE%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%DAPAGLIFLOZIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%CANAGLIFLOZIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%EMPAGLIFLOZIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%Sulfonylureas:%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%ACETOHEXAMIDE%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLIMEPIRIDE%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLICLAZIDE%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLIPIZIDE%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLYBURIDE%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLIBENCLAMIDE%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%CHLORPROPAMIDE %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%TOLAZAMIDE %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%TOLBUTAMIDE %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLYCLOPYRAMIDE %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLIQUIDONE %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLIBORNURIDE %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLYMIDINE SODIUM %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%PRAMLINTIDE %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%NATEGLINIDE %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%REPAGLINIDE%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLULISINE %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%REGULAR INSULIN %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%NPH %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%DETEMIR %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLARGINE %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%DEGLUDEC %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%INSULIN %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%ASPART%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%LISPRO %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%ACTRAPID%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%HYPURIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%ILETIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%INSULATARD%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%INSUMAN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%MIXTARD%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%NOVOMIX%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%NOVORAPID%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%ORALIN %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%ABASAGLAR%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%RYZODEG%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%V-GO%')
				)		
     )
     -- RXNORM_CUI is a VARCHAR field in PCORNet CDM spec, Oracle autoconverts numbers to varchars in comparisons but is very inefficient
     -- choose the appropriate version for the predicate based on if you use a VARCHAR or numeric RXCUI in your local implementation
     -- or a.RXNORM_CUI in (6809,105376,105377,151827,152161,204045,204047,235743,236325,236510,246522,250919,285065,285129,316255,316256,330861,332809,352381,352450,361841,368254,368526,371466,372803,372804,374635,378729,378730,405304,406082,406257,428759,429841,431724,432366,432780,438507,465455,485822,541766,541768,541774,541775,577093,583192,583194,583195,600868,601021,602411,605605,607999,614348,633695,645109,647241,668418,700516,729717,729919,729920,731442,757603,790326,802051,802646,802742,805670,806287,860974,860975,860976,860977,860978,860979,860980,860981,860982,860983,860984,860985,860995,860996,860997,860998,860999,861000,861001,861002,861003,861004,861005,861006,861007,861008,861009,861010,861011,861012,861014,861015,861016,861017,861018,861019,861020,861021,861022,861023,861024,861025,861026,861027,861730,861731,861736,861740,861743,861748,861753,861760,861761,861762,861763,861764,861765,861769,861770,861771,861783,861784,861785,861787,861788,861789,861790,861791,861792,861795,861796,861797,861806,861807,861808,861816,861817,861818,861819,861820,861821,861822,861823,861824,875864,875865,876009,876010,876033,899988,899989,899991,899992,899993,899994,899995,899996,899998,900000,900001,900002,977566,997965,1007411,1008476,1043561,1043562,1043563,1043565,1043566,1043567,1043568,1043569,1043570,1043572,1043574,1043575,1043576,1043578,1043580,1043582,1043583,1043584,1048346,1083665,1128666,1130631,1130713,1131491,1132606,1143649,1145961,1155467,1155468,1156197,1161597,1161598,1161599,1161600,1161601,1161602,1161603,1161604,1161605,1161606,1161607,1161608,1161609,1161610,1161611,1165205,1165206,1165845,1167810,1167811,1169920,1169923,1171244,1171245,1171254,1171255,1172629,1172630,1175016,1175021,1182890,1182891,1184627,1184628,1185325,1185326,1185653,1185654,1243016,1243017,1243018,1243019,1243020,1243027,1243034,1243826,1243827,1243829,1243833,1243834,1243835,1243839,1243842,1243843,1243844,1243845,1243846,1243848,1243849,1243850,1305366,1308857,1313354,1365405,1365406,1365802,1368381,1368382,1368383,1368384,1368385,1368392,1372716,1372738,1431024,1431025,1486436,1493571,1493572,1540290,1540292,1545146,1545147,1545148,1545149,1545150,1545157,1545161,1545164,1548426,1549776,1592709,1592710,1592722,1593057,1593058,1593059,1593068,1593069,1593070,1593071,1593072,1593073,1593774,1593775,1593776,1593826,1593827,1593828,1593829,1593830,1593831,1593832,1593833,1593835,1598393,1598394,1655477,1664311,1664312,1664313,1664314,1664315,1664323,1664326,1665367,1692194,1741248,1741249,1791055,1796088,1796089,1796092,1796094,1796097)
     or a.RXNORM_CUI in ('1008476','105376','105377','1161601','1161602','1161609','1161610','1161611','1171244','1171245','1171254','1171255','1172629','1172630','1182890','1182891','1185325','1185326','1185653','1185654','151827','152161','1807888','1807894','1807915','1807917','204045','204047','235743','285065','316255','316256','330861','332809','361841','368254','368526','372803','372804','405304','406082','406257','428759','431724','438507','541766','541768','541774','541775','583192','583194','583195','645109','647241','6809','802051','860974','860975','860976','860977','860978','860979','860980','860981','860982','860983','860984','860985','860995','860997','860998','860999','861000','861001','861002','861003','861004','861005','861006','861007','861008','861009','861010','861011','861012','861014','861015','861016','861017','861018','861019','861020','861021','861022','861023','861024','861025','861026','861027','861027','861730','875864','875865','876009','876010','876033','977566')
   )
OR
    --  Thiazolidinedione:
   (
        UPPER(a.RAW_RX_MED_NAME) like UPPER('%Avandia%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Actos%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Noscal%') 
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Re[z,s]ulin%') 
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Romozin%') 
     or (
		(UPPER(a.RAW_RX_MED_NAME) like UPPER('%ROSIGLITAZONE%') 
		 or UPPER(a.RAW_RX_MED_NAME) like UPPER('%PIOGLITAZONE%') 
		 or UPPER(a.RAW_RX_MED_NAME) like UPPER('%TROGLITAZONE%')
		 ) 
		 and not (UPPER(a.RAW_RX_MED_NAME) like UPPER('%ACARBOSE %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%MIGLITOL%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%VOGLIBOSE%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%ALOGLIPTIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%ANAGLIPTIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%LINAGLIPTIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%SAXAGLIPTIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%SITAGLIPTIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%TENELIGLIPTIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%VILDAGLIPTIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%LIXISENATIDE%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%ALBIGLUTIDE%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%DULAGLUTIDE%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%DAPAGLIFLOZIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%CANAGLIFLOZIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%EMPAGLIFLOZIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%ACETOHEXAMIDE%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLIMEPIRIDE%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLICLAZIDE%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLIPIZIDE%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLYBURIDE%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLIBENCLAMIDE%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%CHLORPROPAMIDE %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%TOLAZAMIDE %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%TOLBUTAMIDE %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLYCLOPYRAMIDE %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLIQUIDONE %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLIBORNURIDE %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLYMIDINE SODIUM %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%PRAMLINTIDE %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%NATEGLINIDE %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%REPAGLINIDE%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLULISINE %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%REGULAR INSULIN %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%NPH %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%DETEMIR %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%GLARGINE %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%DEGLUDEC %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%INSULIN %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%ASPART%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%LISPRO %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%ACTRAPID%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%HYPURIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%ILETIN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%INSULATARD%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%INSUMAN%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%MIXTARD%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%NOVOMIX%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%NOVORAPID%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%ORALIN %') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%ABASAGLAR%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%RYZODEG%') or
				UPPER(a.RAW_RX_MED_NAME) like UPPER('%V-GO%')
				)
		)
     -- RXNORM_CUI is a VARCHAR field in PCORNet CDM spec, Oracle autoconverts numbers to varchars in comparisons but is very inefficient
     -- choose the appropriate version for the predicate based on if you use a VARCHAR or numeric RXCUI in your local implementation
     -- or a.RXNORM_CUI in (202,572,1480,2242,2465,4583,4586,4615,4622,4623,4624,4625,4626,4627,4628,4630,4631,4632,4633,4634,4635,4636,4951,5316,6053,6211,6214,6216,6885,7208,7475,8925,10481,11019,11022,11024,17547,19217,20185,21336,25511,25512,25515,25517,25520,25527,25617,25851,26061,26402,28170,30957,30958,33738,39108,39125,39131,39141,39146,39147,39149,39150,39152,39847,40224,40738,41397,46981,47026,47724,48253,50216,50278,53324,59156,59191,59208,59284,59419,59427,59513,61539,67375,68213,71877,71921,72227,72610,72622,75157,75158,82578,84108,88577,97483,97568,97609,97833,97841,97848,97852,97867,97875,97877,97881,97892,97893,97894,97919,97977,97986,98066,98075,98076,114175,117305,124953,153014,153015,153722,153723,153724,153725,160190,170941,182835,199984,199985,200065,203027,203715,204377,212281,212282,213041,217954,224003,235298,235305,237601,237602,237603,237622,237623,237624,238255,240757,242206,242614,242615,246859,248528,253198,259319,259351,259382,259383,259635,260018,260111,261241,261242,261243,261266,261267,261268,261442,261455,283847,283848,283849,284132,308043,308706,311248,311259,311260,311261,312440,312441,312859,312860,312861,314063,316124,316125,316869,316870,316871,317223,317573,331478,332435,332436,336270,336271,336906,340667,353229,358499,358500,358530,358809,368230,368234,368317,373801,374252,374606,375549,375855,378729,381259,386116,391625,391627,391633,391634,391635,391636,391645,391646,391650,391654,391671,391677,391801,393133,401993,401994,420203,428722,429088,429558,429808,430181,430343,433795,435806,436189,437129,437131,437306,437391,437392,440537,441116,476352,476353,483592,483642,565366,565367,565368,572491,572492,572980,574470,574471,574472,574495,574496,574497,577093,577605,577606,578033,580285,582044,582226,601642,602012,602014,602015,602016,602017,602018,602019,602166,602543,602544,602549,602550,602593,602594,602595,605320,606253,607999,614348,615015,615016,618299,629614,629615,631212,631213,631214,631215,631216,631217,633494,647235,647236,647237,647239,687360,690417,690728,691361,691407,692793,704551,704552,706895,706896,729115,729116,730905,731455,731457,731461,731462,731463,755768,757211,792114,792115,795807,799064,833760,834152,855905,860487,860488,860489,861760,861763,861783,861795,861806,861816,861822,885217,885249,885250,885252,895940,899988,899989,899994,899996,900001,967790,968642,968643,968644,968793,968799,968800,979613,985057,990292,1007465,1007707,1008459,1009262,1010582,1010583,1010584,1010585,1010591,1011085,1011086,1011087,1014246,1021902,1022620,1023321,1025110,1041762,1041767,1041785,1049771,1087356,1120060,1121071,1121356,1121421,1121996,1123648,1130713,1131491,1135408,1135409,1141372,1143463,1144222,1144325,1144326,1146169,1147775,1149975,1150961,1153166,1153167,1153620,1153621,1153622,1153623,1153624,1157240,1157241,1157242,1157243,1157987,1157988,1161597,1161598,1161603,1161604,1162196,1162197,1162198,1162199,1163231,1163232,1163351,1163352,1163389,1163390,1169928,1169929,1175666,1175667,1181933,1181934,1233738,1234307,1237081,1239373,1246496,1291129,1291130,1291131,1291132,1291133,1294845,1297455,1301823,1302343,1302344,1302345,1302346,1302361,1302362,1302364,1305527,1305837,1307662,1308784,1310038,1313354,1362181,1362741,1368399,1368400,1368401,1368402,1368403,1368405,1368409,1368410,1368412,1368416,1368417,1368419,1368423,1368424,1368426,1368430,1368431,1368433,1368434,1368437,1368438,1368440,1368444,1374661,1374670,1384487,1424651,1424867,1425574,1425992,1426413,1431048,1436586,1439115,1440947,1441287,1485868,1490457,1492336,1493021,1493170,1493173,1494179,1494180,1494181,1494182,1494183,1494184,1494186,1494187,1494191,1495136,1547254,1548162,1552134,1593260,1593760,1593870,1600083,1601850,1605394,1605395,1608162,1649155,1649302,1661518,1663279,1663413,1670328,1670329,1670330,1670331,1670332,1720681,1720845,1722015,1724842,1733688,1738530,1743163,1743280,1746354,1746954,1805299)
     or a.RXNORM_CUI in ('1157987','1157988','1163231','1163232','1163389','1163390','153722','153723','153724','199984','199985','200065','212281','212282','213041','253198','259319','261241','261242','261243','261266','261267','261268','312440','312441','312859','312860','312861','316869','316870','316871','317573','331478','332435','332436','33738','358499','358500','358530','358809','368230','368234','368317','373801','374252','374606','378729','386116','430343','565366','565367','565368','572491','572492','572980','574470','574471','574472','574495','574496','574497','72610','84108')
   )
OR
   --  Glucagon-like Peptide-1 Agonist:
   (
        UPPER(a.RAW_RX_MED_NAME) like UPPER('%Exenatide%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Byetta%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Bydureon%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Liraglutide%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Victoza%')
     or UPPER(a.RAW_RX_MED_NAME) like UPPER('%Saxenda%')
     -- RXNORM_CUI is a VARCHAR field in PCORNet CDM spec, Oracle autoconverts numbers to varchars in comparisons but is very inefficient
     -- choose the appropriate version for the predicate based on if you use a VARCHAR or numeric RXCUI in your local implementation
     -- or a.RXNORM_CUI in (60548,475968,604751,847908,847910,847911,847913,847914,847915,847916,847917,847919,897120,897122,897123,897124,897126,1163230,1163790,1169415,1186578,1242961,1242963,1242964,1242965,1242967,1242968,1244651,1359640,1359802,1359979,1360105,1360454,1360495,1544916,1544918,1544919,1544920,1593624,1596425,1598264,1598265,1598267,1598268,1598269,1598618,1653594,1653597,1653600,1653610,1653611,1653613,1653614,1653616,1653619,1653625,1654044,1654730,1727493,1804447,1804505)
     or a.RXNORM_CUI in ('1163230','1163790','1169415','1186578','1242961','1242963','1242964','1242965','1242967','1242968','1359640','1359802','1359979','1360105','1360454','1360495','1544916','1544918','1544919','1544920','1598264','1598265','1598267','1598268','1598269','1653594','1653597','1653600','1653610','1653611','1653613','1653614','1653616','1653619','1653625','1727493','1860164','1860165','1860166','1860167','1860169','1860170','1860172','1860173','1860174','475968','604751','60548','847908','847910','847911','847913','847914','847915','847916','847917','897120','897122','897123','897124','897126')
   )
OR
   --  Combinations:
   (
    a.RXNORM_CUI in ('1169920','1169923','1175016','1175021','352450','602411','731442','806287','861761','861762','861764','861765','861784','861785','861796','861797','861807','861808','861817','861818','861823','861824','899991','899992','899993','899995','899998','900000','900002')
   )
);

CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."INCL_NON_SPEC_MEDS_FINAL"AS
WITH p1 AS ( -- Get set of patients having one med & one visit:
    SELECT x.PATID, x.RX_ORDER_DATE AS MedDate
    FROM "ANALYTICS"."DMCOHORT"."INCL_RESTRICT_MEDS" x
    JOIN "ANALYTICS"."DMCOHORT"."DX_VISITS_INITIAL"  y ON x.PATID = y.PATID
    WHERE ABS(DATEDIFF(day,y.ADMIT_DATE,x.RX_ORDER_DATE)) > 1
), p2 AS (   -- Get set of patients having one med & one HbA1c:
    SELECT x.PATID, x.RX_ORDER_DATE AS MedDate
    FROM "ANALYTICS"."DMCOHORT"."INCL_RESTRICT_MEDS" x
    JOIN "ANALYTICS"."DMCOHORT"."ALL_A1C" y ON x.PATID = y.PATID
    WHERE ABS(DATEDIFF(day,x.RX_ORDER_DATE,y.LAB_ORDER_DATE)) > 1
), p3 AS (   -- Get set of patients having one med & fasting glucose measurement:
    SELECT x.PATID, x.RX_ORDER_DATE AS MedDate
    FROM "ANALYTICS"."DMCOHORT"."INCL_RESTRICT_MEDS" x
    JOIN "ANALYTICS"."DMCOHORT"."ALL_FG" y ON x.PATID = y.PATID
    WHERE ABS(DATEDIFF(day,x.RX_ORDER_DATE, y.LAB_ORDER_DATE)) > 1
), p4 AS (   -- Get set of patients having one med & random glucose measurement:
    SELECT x.PATID, x.RX_ORDER_DATE AS MedDate
    FROM "ANALYTICS"."DMCOHORT"."INCL_RESTRICT_MEDS"x
    JOIN "ANALYTICS"."DMCOHORT"."ALL_RG" y ON x.PATID = y.PATID
    WHERE ABS(DATEDIFF(day,x.RX_ORDER_DATE,y.LAB_ORDER_DATE)) > 1
), combine_restrict_med_events AS (
    SELECT PATID, MedDate
    FROM p1
    UNION ALL
    SELECT PATID, MedDate
    FROM p2
    UNION ALL
    SELECT PATID, MedDate
    FROM p3
    UNION ALL
    SELECT PATID, MedDate
    FROM p4
)
SELECT PATID, MIN(MedDate) AS EventDate
FROM combine_restrict_med_events
GROUP BY PATID;

---------------------------------------------------------------------------------------------------------------
-----                                      Defining onset date                                            -----
---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------
-----                    Combine results from all parts of the code into final table:                     -----
---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------

CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."FINALSTATSTABLE1_LOCAL" AS 
WITH combined_onset_dates AS (
SELECT a.PATID, a.EventDate
FROM  "ANALYTICS"."DMCOHORT"."DX_VISIT_FINAL_FIRSTPAIR" a
UNION ALL
SELECT b.PATID, b.EventDate
FROM "ANALYTICS"."DMCOHORT"."INCLUSIONMEDS_FINAL"       b
UNION ALL
SELECT c.PATID, c.EventDate 
FROM "ANALYTICS"."DMCOHORT"."A1C_FINAL_FIRSTPAIR"       c
UNION ALL
SELECT d.PATID, d.EventDate
FROM "ANALYTICS"."DMCOHORT"."FG_FINAL_FIRSTPAIR"        d
UNION ALL
SELECT e.PATID, e.EventDate
FROM "ANALYTICS"."DMCOHORT"."RG_FINAL_FIRSTPAIR"        e
UNION ALL
SELECT f.PATID, f.EventDate 
FROM "ANALYTICS"."DMCOHORT"."A1CFG_FINAL_FIRSTPAIR"     f
UNION ALL
SELECT g.PATID, g.EventDate 
FROM "ANALYTICS"."DMCOHORT"."A1CRG_FINAL_FIRSTPAIR"     g
UNION ALL
SELECT k.PATID, k.EventDate
FROM "ANALYTICS"."DMCOHORT"."INCL_NON_SPEC_MEDS_FINAL" k
)
   ,DM_OnsetDates AS (
SELECT PATID, MIN(EventDate) AS DMonsetDate
FROM combined_onset_dates
GROUP BY PATID
)
-- duplicates exist due to different DEATH_SOURCE
   ,death_unique as (
SELECT PATID, MAX(DEATH_DATE) DEATH_DATE
FROM "DEIDENTIFIED_PCORNET_CDM"."CDM_2023_OCT"."DEID_DEATH"
WHERE DEATH_DATE < CURRENT_DATE
group by PATID
)
SELECT v.PATID
      ,v.ENC_FIRST_VISIT AS FirstVisit
      ,v.cnt_distinct_enc_days AS NumberOfVisits 
      ,pt.BIRTH_DATE  AS BIRTH_DATE -- modified: shifted dates back to real dates
      ,d.DEATH_DATE AS DEATH_DATE -- modified: shifted dates back to real dates
      ,dm.DMonsetDate AS DMonsetDate -- modified: shifted dates back to real dates
FROM      "ANALYTICS"."DMCOHORT"."ENC_FIRST_VISIT" v
LEFT JOIN "DEIDENTIFIED_PCORNET_CDM"."CDM_2023_OCT"."DEID_DEMOGRAPHIC" pt on pt.PATID = v.PATID
LEFT JOIN death_unique d ON v.PATID = d.PATID
LEFT JOIN DM_OnsetDates dm ON v.PATID = dm.PATID
;

select count(distinct patid) from "ANALYTICS"."DMCOHORT"."FINALSTATSTABLE1_LOCAL"  --782641
select count(distinct patid) from "ANALYTICS"."DMCOHORT"."PREG_MASKED_ENCOUNTERS" --778763


--99,624 PATIDs with DM
SELECT distinct PATID
FROM  "ANALYTICS"."DMCOHORT"."DX_VISIT_FINAL_FIRSTPAIR" --21,655
UNION
SELECT distinct PATID
FROM "ANALYTICS"."DMCOHORT"."INCLUSIONMEDS_FINAL"  --65,614     
UNION
SELECT distinct PATID
FROM "ANALYTICS"."DMCOHORT"."A1C_FINAL_FIRSTPAIR"   --14,736
UNION 
SELECT distinct PATID
FROM "ANALYTICS"."DMCOHORT"."FG_FINAL_FIRSTPAIR"       --23
UNION
SELECT distinct PATID
FROM "ANALYTICS"."DMCOHORT"."RG_FINAL_FIRSTPAIR"       --23281
UNION
SELECT distinct PATID
FROM "ANALYTICS"."DMCOHORT"."A1CFG_FINAL_FIRSTPAIR"     --161
UNION 
SELECT distinct PATID
FROM "ANALYTICS"."DMCOHORT"."A1CRG_FINAL_FIRSTPAIR"    --18550 
UNION 
SELECT distinct PATID
FROM "ANALYTICS"."DMCOHORT"."INCL_NON_SPEC_MEDS_FINAL"  --28382


---- Diabetes Cohort:

CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."FINALDIABETESCOHORT" AS 
WITH combined_onset_dates AS (
SELECT a.PATID, a.EventDate, 'DX' as SRC
FROM  "ANALYTICS"."DMCOHORT"."DX_VISIT_FINAL_FIRSTPAIR" a
UNION ALL
SELECT b.PATID, b.EventDate, 'INCLUSIONMEDS' as SRC
FROM "ANALYTICS"."DMCOHORT"."INCLUSIONMEDS_FINAL"       b
UNION ALL
SELECT c.PATID, c.EventDate, 'A1C' as SRC
FROM "ANALYTICS"."DMCOHORT"."A1C_FINAL_FIRSTPAIR"       c
UNION ALL
SELECT d.PATID, d.EventDate, 'FG' as SRC
FROM "ANALYTICS"."DMCOHORT"."FG_FINAL_FIRSTPAIR"        d
UNION ALL
SELECT e.PATID, e.EventDate, 'RG' as SRC
FROM "ANALYTICS"."DMCOHORT"."RG_FINAL_FIRSTPAIR"        e
UNION ALL
SELECT f.PATID, f.EventDate, 'A1CFG' as SRC
FROM "ANALYTICS"."DMCOHORT"."A1CFG_FINAL_FIRSTPAIR"     f
UNION ALL
SELECT g.PATID, g.EventDate, 'A1CRG' as SRC 
FROM "ANALYTICS"."DMCOHORT"."A1CRG_FINAL_FIRSTPAIR"     g
UNION ALL
SELECT k.PATID, k.EventDate, 'INCL_NON_SPEC_MEDS' as SRC
FROM "ANALYTICS"."DMCOHORT"."INCL_NON_SPEC_MEDS_FINAL" k
)
   ,DM_OnsetDates AS (
SELECT PATID, MIN(EventDate) AS DMonsetDate
FROM combined_onset_dates
GROUP BY PATID
)
-- duplicates exist due to different DEATH_SOURCE
   ,death_unique as (
SELECT PATID, MAX(DEATH_DATE) DEATH_DATE
FROM "DEIDENTIFIED_PCORNET_CDM"."CDM_2023_OCT"."DEID_DEATH"
WHERE DEATH_DATE < CURRENT_DATE
group by PATID
)

SELECT distinct c.PATID
      ,v.ENC_FIRST_VISIT AS FirstVisit
      ,v.cnt_distinct_enc_days AS NumberOfVisits 
      ,pt.BIRTH_DATE  AS BIRTH_DATE -- modified: shifted dates back to real dates
      ,d.DEATH_DATE AS DEATH_DATE -- modified: shifted dates back to real dates
      ,dm.DMonsetDate AS DMonsetDate -- modified: shifted dates back to real dates
      ,DATEDIFF(day, v.ENC_FIRST_VISIT, dm.DMonsetDate) AS DIFF_FIRSTENC_DMONSET
FROM  combined_onset_dates c
JOIN "ANALYTICS"."DMCOHORT"."ENC_FIRST_VISIT" v on v.PATID= c.PATID
LEFT JOIN "DEIDENTIFIED_PCORNET_CDM"."CDM_2023_OCT"."DEID_DEMOGRAPHIC" pt on pt.PATID = c.PATID
LEFT JOIN death_unique d ON c.PATID = d.PATID
LEFT JOIN DM_OnsetDates dm ON c.PATID = dm.PATID
ORDER BY PATID
;

-- Diabetes Cohort with SRC of the visit information
-- Might help to identify if there are any orphan records

CREATE OR REPLACE TABLE "ANALYTICS"."DMCOHORT"."FINALDIABETESCOHORT_SRC" AS 
WITH combined_onset_dates AS (
SELECT a.PATID, a.EventDate, 'DX' as SRC
FROM  "ANALYTICS"."DMCOHORT"."DX_VISIT_FINAL_FIRSTPAIR" a
UNION ALL
SELECT b.PATID, b.EventDate, 'INCLUSIONMEDS' as SRC
FROM "ANALYTICS"."DMCOHORT"."INCLUSIONMEDS_FINAL"       b
UNION ALL
SELECT c.PATID, c.EventDate, 'A1C' as SRC
FROM "ANALYTICS"."DMCOHORT"."A1C_FINAL_FIRSTPAIR"       c
UNION ALL
SELECT d.PATID, d.EventDate, 'FG' as SRC
FROM "ANALYTICS"."DMCOHORT""FG_FINAL_FIRSTPAIR"        d
UNION ALL
SELECT e.PATID, e.EventDate, 'RG' as SRC
FROM "ANALYTICS"."DMCOHORT"."RG_FINAL_FIRSTPAIR"        e
UNION ALL
SELECT f.PATID, f.EventDate, 'A1CFG' as SRC
FROM "ANALYTICS"."DMCOHORT"."A1CFG_FINAL_FIRSTPAIR"     f
UNION ALL
SELECT g.PATID, g.EventDate, 'A1CRG' as SRC 
FROM "ANALYTICS"."DMCOHORT"."A1CRG_FINAL_FIRSTPAIR"     g
UNION ALL
SELECT k.PATID, k.EventDate, 'INCL_NON_SPEC_MEDS' as SRC
FROM "ANALYTICS"."DMCOHORT"."INCL_NON_SPEC_MEDS_FINAL" k
)
   ,DM_OnsetDates AS (
SELECT PATID, MIN(EventDate) AS DMonsetDate
FROM combined_onset_dates
GROUP BY PATID
)
-- duplicates exist due to different DEATH_SOURCE
   ,death_unique as (
SELECT PATID, MAX(DEATH_DATE) DEATH_DATE
FROM "DEIDENTIFIED_PCORNET_CDM"."CDM_2023_OCT"."DEID_DEATH"
WHERE DEATH_DATE < CURRENT_DATE
group by PATID
)
    ,SRC_OF_VISIT AS (
SELECT c.PATID, SRC
FROM combined_onset_dates c
JOIN DM_OnsetDates dm 
ON c.PATID=dm.PATID and c.EventDate = dm.DMonsetDate 
)
SELECT distinct c.PATID
      ,v.ENC_FIRST_VISIT AS FirstVisit
      ,v.cnt_distinct_enc_days AS NumberOfVisits 
      ,pt.BIRTH_DATE  AS BIRTH_DATE -- modified: shifted dates back to real dates
      ,d.DEATH_DATE AS DEATH_DATE -- modified: shifted dates back to real dates
      ,dm.DMonsetDate AS DMonsetDate -- modified: shifted dates back to real dates
      ,src.SRC AS SRC_OF_VISIT
      ,DATEDIFF(day, v.ENC_FIRST_VISIT, dm.DMonsetDate) AS DIFF_FIRSTENC_DMONSET
FROM  combined_onset_dates c
JOIN "ANALYTICS"."DMCOHORT"."ENC_FIRST_VISIT" v on v.PATID= c.PATID
LEFT JOIN "DEIDENTIFIED_PCORNET_CDM"."CDM_2023_OCT"."DEID_DEMOGRAPHIC" pt on pt.PATID = c.PATID
LEFT JOIN death_unique d ON c.PATID = d.PATID
LEFT JOIN DM_OnsetDates dm ON c.PATID = dm.PATID
LEFT JOIN SRC_OF_VISIT src ON c.PATID = src.PATID
ORDER BY PATID
;

select * from "ANALYTICS"."DMCOHORT"."ENC_FIRST_VISIT"
where PATID='106340'
select * from "ANALYTICS"."DMCOHORT"."SUMMARIZED_ENCOUNTERS"
where PATID='106340'



select min(NUMBEROFVISITS), max(NUMBEROFVISITS), min(DIFF_FIRSTENC_DMONSET), max(DIFF_FIRSTENC_DMONSET)
from "ANALYTICS"."DMCOHORT"."FINALDIABETESCOHORT"

select *
from "ANALYTICS"."DMCOHORT"."FINALDIABETESCOHORT"
where DIFF_FIRSTENC_DMONSET <0 --1280 patids


select distinct DIFF_FIRSTENC_DMONSET,count(PATID)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."FINALDIABETESCOHORT"
where DIFF_FIRSTENC_DMONSET <0
group by DIFF_FIRSTENC_DMONSET --349 patids with -1 day diff
order by DIFF_FIRSTENC_DMONSET

select NUMBEROFVISITS, count(PATID)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."FINALDIABETESCOHORT"
group by NUMBEROFVISITS
