---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------
-----                                 Date: Feb 16, 2023,Author: Humayera Islam                          ------
----- Four major complications of DM: MACE (Major Adverse Cardiovascular Events), Chronic Kidney Disease,------
-----                                 Retinopathy,Neuropathy, & Death                                    ------
-----                The major complications will be identified among the DM cohort and                  ------
-----                     Their first incidence in the EHR will be recorded                              ------
-----                Each PATID will be followed upto the first incidence of the minimum of              ------
-----                             incidences among all the complications                                 ------
---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------

-------Extract all diagnosis from DEID_DIAGNOSIS for PATIDs with DM 
create or replace table create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG" as
select distinct d.PATID, d.DMONSETDATE, e.encounterid, e.enc_type, e.admit_date, d.BIRTH_DATE,d.DEATH_DATE,d.FIRSTVISIT, d.DIFF_FIRSTENC_DMONSET,
     e.DX, e.DX_TYPE, e.DX_SOURCE, e.DX_ORIGIN, e.PDX, e.DX_POA,
    datediff(day, d.DMONSETDATE,e.ADMIT_DATE) as diff_DM_ENC    
from  "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."FINALDIABETESCOHORT" d
join  "DEIDENTIFIED_PCORNET_CDM"."CDM_2022_OCT"."DEID_DIAGNOSIS" e
on d.PATID=e.PATID
where e.DX_TYPE in ('09','10')
order by d.PATID, e.ADMIT_DATE
-- 17,322,108

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG" as
select d.*, 
    case when e.DISCHARGE_DATE is not null then cast(e.DISCHARGE_DATE as date) 
       else cast(e.ADMIT_DATE as date) end as DISCHARGE_DATE
from  "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG" d
join  "DEIDENTIFIED_PCORNET_CDM"."CDM_2022_OCT"."DEID_ENCOUNTER" e
on d.PATID=e.PATID
and d.encounterid=e.encounterid
order by d.PATID, e.ADMIT_DATE
-- 17,322,105

select * from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG"
where discharge_date is null

---------------------------------------------------------------------------------------------------------------
------                                       Identify PATIDs with MACE                                   ------  
---------------------------------------------------------------------------------------------------------------

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."PAT_MACE_ENDPT" as
with mace_event as (
    select dx.PATID
          ,'MI' ENDPOINT
          ,coalesce(dx.DX_DATE,dx.ADMIT_DATE) ENDPOINT_DATE
          , dx.ENCOUNTERID , dx.ADMIT_DATE
    from "DEIDENTIFIED_PCORNET_CDM"."CDM_2022_OCT"."DEID_DIAGNOSIS" dx
    where (dx.DX_TYPE = '09' and split_part(dx.DX,'.',1) in ( '410','412')) OR
          (dx.DX_TYPE = '10' and split_part(dx.DX,'.',1) in ( 'I21','I22','I23'))
    union
    select dx.PATID
          ,'STROKE' ENDPOINT
          ,coalesce(dx.DX_DATE,dx.ADMIT_DATE) ENDPOINT_DATE
          , dx.ENCOUNTERID , dx.ADMIT_DATE
    from "DEIDENTIFIED_PCORNET_CDM"."CDM_2022_OCT"."DEID_DIAGNOSIS" dx
    where (dx.DX_TYPE = '09' and split_part(dx.DX,'.',1) in ( '431','434')) OR
          (dx.DX_TYPE = '10' and split_part(dx.DX,'.',1) in ( 'I61','I62','I63'))
    union
    select dx.PATID
          ,'HF' ENDPOINT
          ,coalesce(dx.DX_DATE,dx.ADMIT_DATE) ENDPOINT_DATE
          , dx.ENCOUNTERID , dx.ADMIT_DATE
    from  "DEIDENTIFIED_PCORNET_CDM"."CDM_2022_OCT"."DEID_DIAGNOSIS" dx
    where (dx.DX_TYPE = '09' and split_part(dx.DX,'.',1) in ( '428')) OR
          (dx.DX_TYPE = '10' and split_part(dx.DX,'.',1) in ( 'I50'))
    union
    select px.PATID
          ,'REVASC' ENDPOINT
          ,coalesce(px.PX_DATE,px.ADMIT_DATE) ENDPOINT_DATE
          ,px.ENCOUNTERID , px.ADMIT_DATE
    from "DEIDENTIFIED_PCORNET_CDM"."CDM_2022_OCT"."DEID_PROCEDURES" px
    where (px.PX_TYPE = '09' and substr(px.PX,1,4) in ( '36.0','36.1')) OR
          (px.PX_TYPE = '10' and substr(px.PX,1,3) in ( '021','027')) OR
          (px.PX_TYPE = 'CH' and
           px.PX in ( '92920','92921','92924','92925','92928','92929'
                     ,'92933','92934','92937','92938','92941','92943'
                     ,'92944','92980','92981','92982','92984','92995'
                     ,'92996','92973','92974'))
)
select m.patid, m.endpoint_date , m.ENDPOINT,m.ENCOUNTERID, m.ADMIT_DATE
from mace_event m 
where m.endpoint_date = (
                    select min(endpoint_date)
                    from mace_event p
                    where m.patid=p.patid
                    )
order by m.patid
;

---------------------------------------------------------------------------------------------------------------
------                                       Identify PATIDs with CKD                                    ------  
---------------------------------------------------------------------------------------------------------------

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."PAT_CKD_ENDPT" as
with CKD_event as (
  select dx.PATID
      ,'CKD' as ENDPOINT
      ,coalesce(dx.DX_DATE,dx.ADMIT_DATE) ENDPOINT_DATE
      ,dx.ENCOUNTERID , dx.ADMIT_DATE
  from "DEIDENTIFIED_PCORNET_CDM"."CDM_2022_OCT"."DEID_DIAGNOSIS" dx
  where (dx.DX_TYPE = '10' and
       (dx.DX like 'N18%')) 
        or 
        (dx.DX_TYPE = '09' and
       (dx.DX like '585%'))
        or
        (dx.DX_TYPE = '09' and
       (dx.DX like '250.4%'))
        or
        (dx.DX_TYPE = '10' and
       (dx.DX like 'E10.2%'))
        or
        (dx.DX_TYPE = '10' and
       (dx.DX like 'E11.2%'))
  order by dx.PATID
)
select m.patid, m.endpoint_date , m.ENDPOINT,m.ENCOUNTERID, m.ADMIT_DATE
from CKD_event m 
where m.endpoint_date = (
                    select min(endpoint_date)
                    from CKD_event p
                    where m.patid=p.patid
                    )
order by m.patid
;

 
select dx.*
from "DEIDENTIFIED_PCORNET_CDM"."CDM_2022_OCT"."DEID_DIAGNOSIS" dx
having dx.PATID = '1000393' and
       ( (dx.DX_TYPE = '10' and
       (dx.DX like 'N18%')) 
        or 
        (dx.DX_TYPE = '09' and
       (dx.DX like '585%'))
        or
        (dx.DX_TYPE = '09' and
       (dx.DX like '250.4%'))
        or
        (dx.DX_TYPE = '10' and
       (dx.DX like 'E10.2%'))
        or
        (dx.DX_TYPE = '10' and
       (dx.DX like 'E11.2%')))
       
       
---------------------------------------------------------------------------------------------------------------
------                                       Identify PATIDs with Retinopathy                             ------  
---------------------------------------------------------------------------------------------------------------      
-- retinopathy 362.0x, E10.31x-E10.35x or E11.31x-E11.35x 


create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."PAT_RET_ENDPT" as
with RET_event as (
  select dx.PATID
      ,'RET' as ENDPOINT
      ,coalesce(dx.DX_DATE,dx.ADMIT_DATE) ENDPOINT_DATE
      ,dx.ENCOUNTERID , dx.ADMIT_DATE
  from "DEIDENTIFIED_PCORNET_CDM"."CDM_2022_OCT"."DEID_DIAGNOSIS" dx
  where 
        ((dx.DX_TYPE = '09' and
       (dx.DX like '362.0%'))
        or
        (dx.DX_TYPE = '10' and
       (dx.DX like 'E10.31%'))
          or
        (dx.DX_TYPE = '10' and
       (dx.DX like 'E10.32%'))
          or
        (dx.DX_TYPE = '10' and
       (dx.DX like 'E10.33%'))
          or
        (dx.DX_TYPE = '10' and
       (dx.DX like 'E10.34%'))
          or
        (dx.DX_TYPE = '10' and
       (dx.DX like 'E10.35%'))
          or
        (dx.DX_TYPE = '10' and
       (dx.DX like 'E11.31%'))  
          or
        (dx.DX_TYPE = '10' and
       (dx.DX like 'E11.32%'))
          or
        (dx.DX_TYPE = '10' and
       (dx.DX like 'E11.33%'))
          or
        (dx.DX_TYPE = '10' and
       (dx.DX like 'E11.34%'))
          or
        (dx.DX_TYPE = '10' and
       (dx.DX like 'E11.35%')))
  order by dx.PATID
)
select m.patid, m.endpoint_date , m.ENDPOINT,m.ENCOUNTERID, m.ADMIT_DATE
from RET_event m 
where m.endpoint_date = (
                    select min(endpoint_date)
                    from RET_event p
                    where m.patid=p.patid
                    )
order by m.patid
;

#some patids had multiple encounterids for same admit_date
select *
from "DEIDENTIFIED_PCORNET_CDM"."CDM_2022_OCT"."DEID_ENCOUNTER"
where encounterid in ('20305341', '21292466')


---------------------------------------------------------------------------------------------------------------
------                                       Identify PATIDs with Neuropathy                             ------  
--------------------------------------------------------------------------------------------------------------- 

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."PAT_NEUR_ENDPT" as
with NEUR_event as (
  select dx.PATID
      ,'NEUR' as ENDPOINT
      ,coalesce(dx.DX_DATE,dx.ADMIT_DATE) ENDPOINT_DATE
      ,dx.ENCOUNTERID , dx.ADMIT_DATE
  from "DEIDENTIFIED_PCORNET_CDM"."CDM_2022_OCT"."DEID_DIAGNOSIS" dx
  where 
        (dx.DX_TYPE = '09' and
       (dx.DX like '357.2%'))
        or
        (dx.DX_TYPE = '09' and
       (dx.DX like '350.6%'))
         or
        (dx.DX_TYPE = '10' and
       (dx.DX like 'E10.4%'))
        or
        (dx.DX_TYPE = '10' and
       (dx.DX like 'E11.4%'))
           or
        (dx.DX_TYPE = '10' and
       (dx.DX like 'E12.4%'))
         or
        (dx.DX_TYPE = '10' and
       (dx.DX like 'E13.4%'))
          or
        (dx.DX_TYPE = '10' and
       (dx.DX like 'E14.4%'))
           or
        (dx.DX_TYPE = '10' and
       (dx.DX like 'G63.2%'))
           or
        (dx.DX_TYPE = '10' and
       (dx.DX like 'G62.9%'))
  order by dx.PATID
)
select m.patid, m.endpoint_date , m.ENDPOINT,m.ENCOUNTERID, m.ADMIT_DATE
from NEUR_event m 
where m.endpoint_date = (
                    select min(endpoint_date)
                    from NEUR_event p
                    where m.patid=p.patid
                    )
order by m.patid
;

select *
from "DEIDENTIFIED_PCORNET_CDM"."CDM_2022_OCT"."DEID_DIAGNOSIS" 
where DX like '362.0%'
or DX like 'E10.31%' 


---------------------------------------------------------------------------------------------------------------
------                                       Combining all four datasets                                 ------  
--------------------------------------------------------------------------------------------------------------- 


create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."PATID_EVENT_ENDPT" as
with endpt_event as (
  select m.patid,  m.endpoint_date, m.endpoint, m.encounterid, m.admit_date
  from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."PAT_MACE_ENDPT" m
  UNION
  select c.patid,  c.endpoint_date, c.endpoint, c.encounterid, c.admit_date
  from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."PAT_CKD_ENDPT" c
  UNION
  select r.patid,r.endpoint_date, r.endpoint, r.encounterid, r.admit_date
  from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."PAT_RET_ENDPT" r
  UNION
  select n.patid,n.endpoint_date, n.endpoint, n.encounterid, n.admit_date
  from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."PAT_NEUR_ENDPT" n
)
select e.patid, e.endpoint_date , e.ENDPOINT,e.ENCOUNTERID, e.ADMIT_DATE
from endpt_event e
where e.endpoint_date = (
                    select min(endpoint_date)
                    from endpt_event p
                    where e.patid=p.patid
                    )
order by e.patid

select count(distinct patid) 
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."PATID_EVENT_ENDPT"
--102861

----------------------------------------------------------------------------------------------------------------
------                                       Combining with DM Cohort                                      ------  
------ Creating a column to contain maximum discharge date for each patient if                             ------  
------ the endpoint date is not available then keeping rows that have discharge dates before cut_off dates ------  
----------------------------------------------------------------------------------------------------------------- 

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_ENDPNT" as
select d.*, e.endpoint, e.endpoint_date, row_number() OVER (PARTITION BY d.PATID ORDER BY d.ADMIT_DATE) as rn, 
count(distinct d.admit_date) OVER (PARTITION BY d.PATID) as cnt
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG" d
left join  "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."PATID_EVENT_ENDPT" e
on d.patid=e.patid and d.encounterid=e.encounterid
order by d.patid, d.admit_date, rn


create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_ENDPNT_CUTOFF" as
with endpnt as (select d.patid, min(d.endpoint_date) as cutoff_point
                from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_ENDPNT"  d
                group by d.patid), 
      cutoff as (select d.patid, max(d.discharge_date) as cutoff_extended
                from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_ENDPNT"  d
                 group by d.patid )
select  d.*, coalesce(e.cutoff_point, dateadd(day, 1, to_date(c.cutoff_extended))) as cutoff
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_ENDPNT" d
left join endpnt e on d.patid=e.patid
left join cutoff c on d.patid=c.patid
order by d.patid, d.admit_date, d.rn

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PROGNOSTIC" as
select *, datediff(day, discharge_date, cutoff) as date_diff 
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_ENDPNT_CUTOFF" 
where date_diff >0
order by patid, admit_date, rn

-- Check
select count(distinct PATID)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG" --73365

select count(distinct PATID)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PROGNOSTIC" --66791


select *
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_ENDPNT_CUTOFF"
where PATID= '1239250' 


select *
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_ENDPNT_CUTOFF"
where PATID= '2297407' 

-- Count number of visits per patient
select min(cnt), max(cnt)
from (select count(distinct encounterid) as cnt
from  "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PROGNOSTIC"
group by patid)

select cnt, count(*) from (
select count(distinct encounterid) as cnt
from  "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PROGNOSTIC"
group by patid)
group by cnt

select count(distinct PATID) as cnt
from  "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PROGNOSTIC"
where cnt>3

--------------Check if DM onset is before or after endpoint_date --------
create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PATID_ELIG" as
select PATID from (select patid, min(dmonsetdate) as dmonset, min(endpoint_date) as endpnt, datediff(day, dmonset,endpnt) as diff
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_ENDPNT_CUTOFF"
group by PATID )
 where diff > 0 or diff is NULL
 
 
--------------- Pretraining data for BERT --------------------------
create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAINDATA" as
select distinct PATID as patient_id, cast(ADMIT_DATE as date) as ADMIT_DATE, 
case when DISCHARGE_DATE is not null then cast(DISCHARGE_DATE as date) 
     else cast(ADMIT_DATE as date) end as DISCHARGE_DATE, 
concat(DX_TYPE, '_', DX) as diagnosis,
case when DX_POA= 'Y' then 1
     when DX_POA = 'N' then 2
     else 3 end as poa,
case when DX_SOURCE= 'AD' then 1
     when DX_SOURCE = 'IN' then 2
     when DX_SOURCE = 'DI' then 3
     when DX_SOURCE = 'FI' then 3
     else 4 end as third_party_ind,     
case when PDX='P' then 1 
     when PDX='S' then 2 
     else 3 end as diagnosis_priority
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PROGNOSTIC"
where PATID in (select PATID from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PATID_ELIG")
order by PATID, ADMIT_DATE, POA, diagnosis_priority, third_party_ind

select *
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAINDATA"
where patient_id='2400934'

select count(*)
from (select count(distinct admit_date) as cnt
from  "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAINDATA"
group by patient_id)
where cnt > 4

select PDX, count(*)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAINDATA"
group by PDX --S, P, NI

select DX_POA, count(*)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAINDATA"
group by DX_POA --1, N, NI, Y

select DX_SOURCE, count(*)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAINDATA"
group by DX_SOURCE --OT, FI, AD, IN, DI


select count(distinct patient_id)  --51049
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAINDATA"

select min(cnt_DX), max(cnt_DX)
from
(select patient_id, count(distinct DIAGNOSIS) as cnt_DX --51049
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAINDATA"
group by patient_id)


------------------ Fine Tuning Data ----------------------------------
create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_ALL_ENDPNT" as 
select m.patid,  m.endpoint_date, m.endpoint, m.encounterid, m.admit_date
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."PAT_MACE_ENDPT" m
UNION
select c.patid,  c.endpoint_date, c.endpoint, c.encounterid, c.admit_date
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."PAT_CKD_ENDPT" c
UNION
select r.patid,r.endpoint_date, r.endpoint, r.encounterid, r.admit_date
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."PAT_RET_ENDPT" r
UNION
select n.patid,n.endpoint_date, n.endpoint, n.encounterid, n.admit_date
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."PAT_NEUR_ENDPT" n
order by PATID, ADMIT_DATE, ENDPOINT_DATE

select ENDPOINT, count(distinct PATID)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_ALL_ENDPNT"
group by ENDPOINT

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_LAST_VISIT" as 
select patient_id, max(discharge_date) as final_visit_date
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAINDATA"
group by patient_id


--- Labels for next 1 year
create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_COMPLICATIONS_NEXT1YR" as 
select d.*, e.endpoint_date, e.endpoint, e.encounterid, e.admit_date, datediff(day, final_visit_Date, endpoint_date) as visit_time_to_comp
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_LAST_VISIT" d
left join "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_ALL_ENDPNT" e
on d.patient_id=e.patid
where visit_time_to_comp <= 365

select ENDPOINT, count(distinct PATIENT_ID)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_COMPLICATIONS_NEXT1YR"
group by ENDPOINT

select count(distinct PATIENT_ID)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_COMPLICATIONS_NEXT1YR" --14120


-- Labels for next 2 year
create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_COMPLICATIONS_NEXT2YR" as 
select d.*, e.endpoint_date, e.endpoint, e.encounterid, e.admit_date, datediff(day, final_visit_Date, endpoint_date) as visit_time_to_comp
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_LAST_VISIT" d
left join "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_ALL_ENDPNT" e
on d.patient_id=e.patid
where visit_time_to_comp <= 365*2

select ENDPOINT, count(distinct PATIENT_ID)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_COMPLICATIONS_NEXT2YR"
group by ENDPOINT


-- Labels for next 3 year
create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_COMPLICATIONS_NEXT3YR" as 
select d.*, e.endpoint_date, e.endpoint, e.encounterid, e.admit_date, datediff(day, final_visit_Date, endpoint_date) as visit_time_to_comp
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_LAST_VISIT" d
left join "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_ALL_ENDPNT" e
on d.patient_id=e.patid
where visit_time_to_comp <= 365*3

select ENDPOINT, count(distinct PATIENT_ID)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_COMPLICATIONS_NEXT3YR"
group by ENDPOINT


---- Refine PRETRAIN DATA ------
---- At least 5 and max 100 encounters on different dates for each PATID -------
---- At least 5 unique ICD codes -------


create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNING_NEXT1YR" as
select d.*
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAINDATA" d
where d.patient_id in (select patient_id from (select PATIENT_ID, count(distinct ADMIT_DATE) as cnt 
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAINDATA"
group by PATIENT_ID
having cnt >=5 and cnt <=100)) --37,021
order by patient_id


create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNING_NEXT1YR" as
select d.*
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNING_NEXT1YR" d
where d.patient_id in (select patient_id from (select PATIENT_ID, count(distinct DIAGNOSIS) as cnt_DX
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNING_NEXT1YR"
group by PATIENT_ID
having cnt_DX >=5)) --36,846
order by patient_id

select count(distinct PATIENT_ID)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNING_NEXT1YR"


--- Year 1 prediction labels
create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNING_NEXT1YR_LABELS" as
select distinct d.patient_id, e.endpoint
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNING_NEXT1YR" d
left join "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_COMPLICATIONS_NEXT1YR" e
on d.patient_id=e.patient_id
order by patient_id


select min(cnt_DX), max(cnt_DX)
from
(select patient_id, count(distinct DIAGNOSIS) as cnt_DX --51049
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNING_NEXT1YR"
group by patient_id)

select min(cnt_DX), max(cnt_DX)
from
(select patient_id, count(distinct ADMIT_DATE) as cnt_DX --51049
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNING_NEXT1YR"
group by patient_id)



-- 2 year labels fine-tuning
create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNING_NEXT2YR_LABELS" as
select distinct d.patient_id, e.endpoint
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNING_NEXT1YR" d
left join "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_COMPLICATIONS_NEXT2YR" e
on d.patient_id=e.patient_id
order by patient_id

-- 3 year labels fine-tuning
create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNING_NEXT3YR_LABELS" as
select distinct d.patient_id, e.endpoint
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNING_NEXT1YR" d
left join "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_COMPLICATIONS_NEXT3YR" e
on d.patient_id=e.patient_id
order by patient_id

select count(distinct PATIENT_ID)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNING_NEXT1YR_LABELS"

select count(distinct PATIENT_ID)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNING_NEXT2YR_LABELS"

select count(distinct PATIENT_ID)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNING_NEXT3YR_LABELS"


----------------------------------------------------------------------------------------------------------------
--------------- DX mapping to Phecode for Pretrain Cohort             ------------------------------------------
----------------------------------------------------------------------------------------------------------------

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAIN_MAP2PHE" as
select distinct PATID as patient_id, cast(ADMIT_DATE as date) as ADMIT_DATE, 
case when DISCHARGE_DATE is not null then cast(DISCHARGE_DATE as date) 
     else cast(ADMIT_DATE as date) end as DISCHARGE_DATE, 
DX_TYPE, DX,
case when DX_POA= 'Y' then 1
     when DX_POA = 'N' then 2
     else 3 end as poa,
case when DX_SOURCE= 'AD' then 1
     when DX_SOURCE = 'IN' then 2
     when DX_SOURCE = 'DI' then 3
     when DX_SOURCE = 'FI' then 3
     else 4 end as third_party_ind,     
case when PDX='P' then 1 
     when PDX='S' then 2 
     else 3 end as diagnosis_priority
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PROGNOSTIC"
where PATID in (select PATID from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PATID_ELIG")
order by PATID, ADMIT_DATE, POA, diagnosis_priority, third_party_ind

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAIN_MAP2PHE" as
select DISTINCT d.PATIENT_ID, d.ADMIT_DATE, d.DISCHARGE_DATE, e."phecode", d.POA, d.THIRD_PARTY_IND, d.DIAGNOSIS_PRIORITY
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAIN_MAP2PHE" d
left join "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."MAPPING_PHECODE" e
ON e.DX=d.DX AND e.DX_TYPE=d.DX_TYPE
where  e."phecode" is not NULL
order by d.PATIENT_ID, d.ADMIT_DATE, d.DISCHARGE_DATE, d.POA, d.THIRD_PARTY_IND, d.DIAGNOSIS_PRIORITY


select count(distinct PATIENT_ID)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAIN_MAP2PHE" --50993
----------------------------------------------------------------------------------------------------------------
--------------- Fine Tuning Cohort - DX mapping to Phecode ------------------------------------------
----------------------------------------------------------------------------------------------------------------
create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNE_MAP2PHE" as
select d.*
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAIN_MAP2PHE" d
where patient_id in (select patient_id from (select PATIENT_ID,count(distinct ADMIT_DATE) as cnt 
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAIN_MAP2PHE"
group by PATIENT_ID
having cnt >=5 and cnt <=100)) --36,539
order by patient_id

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNE_MAP2PHE" as
select d.*
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNE_MAP2PHE" d
where patient_id in (select patient_id from (select PATIENT_ID, count(distinct "phecode") as cnt_DX
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNE_MAP2PHE"
group by PATIENT_ID
having cnt_DX >=5)) --36,539
order by patient_id

select count(distinct patient_id) --36,539
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNE_MAP2PHE"

select min(cnt_DX), max(cnt_DX)
from
(select patient_id, count(distinct "phecode") as cnt_DX 
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNE_MAP2PHE"
group by patient_id)

select min(cnt_DX), max(cnt_DX)
from
(select patient_id, count(distinct ADMIT_DATE) as cnt_DX 
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNE_MAP2PHE"
group by patient_id)

----------------------------------------------------------------------------------------------------------------
--------------- Machine Learning Applications - DX mapping to Phecode ------------------------------------------
----------------------------------------------------------------------------------------------------------------

--------------- training data for ML models --------------------------

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_MLDATA_PHE" as
select distinct d.*, e."phenotype"
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNE_MAP2PHE" d
left join "ONTOLOGY"."GROUPER_VALUESETS"."PHECODE_REF" e
on d."phecode" = e."phecode"
order by PATIENT_ID, ADMIT_DATE

----- Breaking the temporal structure ----

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_MLDATA_PHE_DISTNCT" as
select distinct PATIENT_ID, "phecode"
from  "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_MLDATA_PHE"

select count(distinct "phecode") --1859 potential features
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_MLDATA_PHE_DISTNCT"

select count(distinct PATIENT_ID) --36,539
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_MLDATA_PHE_DISTNCT"

select min(cnt), max(cnt)
from (select distinct PATIENT_ID, count(*) as cnt
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_MLDATA_PHE_DISTNCT"
group by PATIENT_ID)

select * from (select distinct PATIENT_ID, count(*) as cnt
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_MLDATA_PHE_DISTNCT"
group by PATIENT_ID)
where cnt<5


select * from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_MLDATA_PHE" 
where "phecode" is null

select "phenotype", count(distinct PATIENT_ID)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_MLDATA_PHE" 
group by "phenotype"


"ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG"-- select new features for machine learning 
-- recent age for each patient
-- DM onset date
-- Birthdate
-- find age at DM
-- find time since DM diagnosis until recent visit

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."ML_FEATURES" as
select distinct PATID, DMONSETDATE, BIRTH_DATE from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG"
where PATID in (select distinct PATIENT_ID from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_MLDATA_PHE_DISTNCT" )

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."ML_FEATURES" as
select distinct a.*, b.OLDEST_ADMIT_DATE, b.RECENT_ADMIT_DATE,  b.RECENT_AGE
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."ML_FEATURES" a
left join (select PATIENT_ID, MIN(ADMIT_DATE) as OLDEST_ADMIT_DATE,MAX(ADMIT_DATE) AS RECENT_ADMIT_DATE, MAX(AGE_AT_ENC) AS RECENT_AGE
           from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_FINETUNE_DX_AGE" 
           group by PATIENT_ID) b
on a.PATID=b.PATIENT_ID

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."ML_FEATURES" as
select *, datediff(day,BIRTH_DATE, DMONSETDATE)/365.2425 as AGE_AT_DM, datediff(day,OLDEST_ADMIT_DATE,RECENT_ADMIT_DATE )/365.2425 as TIME_SINCE
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."ML_FEATURES"
order by PATID




select count(distinct PATIENT_ID) from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_FINETUNE_DX_AGE" --36,539

----------------------------------------------------------------------------------------------------------------
--------------- Machine Learning Applications - DX as ICD9/10     ------------------------------------------
----------------------------------------------------------------------------------------------------------------

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_MLDATA_ICD" as
select distinct PATIENT_ID, DIAGNOSIS
from  "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNING_NEXT1YR"

select count(distinct DIAGNOSIS) --30,794
from  "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_MLDATA_ICD" 


----------------------------------------------------------------------------------------------------------------
--------------- Fine-tuning Cohort Labels for Phecodes           ------------------------------------------
----------------------------------------------------------------------------------------------------------------

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNEPHE_NEXT1YR_LABELS" as
select distinct d.patient_id, e.endpoint
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNE_MAP2PHE" d
left join "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_COMPLICATIONS_NEXT1YR" e
on d.patient_id=e.patient_id
order by patient_id


create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNEPHE_NEXT2YR_LABELS" as
select distinct d.patient_id, e.endpoint
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNE_MAP2PHE" d
left join "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_COMPLICATIONS_NEXT2YR" e
on d.patient_id=e.patient_id
order by patient_id

create or replace table "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNEPHE_NEXT3YR_LABELS" as
select distinct d.patient_id, e.endpoint
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNE_MAP2PHE" d
left join "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_COMPLICATIONS_NEXT3YR" e
on d.patient_id=e.patient_id"ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_PRETRAIN_MAP2PHE"
order by patient_id

select count(distinct PATIENT_ID)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNEPHE_NEXT1YR_LABELS"

select count(distinct PATIENT_ID)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNEPHE_NEXT2YR_LABELS"

select count(distinct PATIENT_ID)
from "ANALYTICS"."DIABETESMELLITUSSTUDYSCHEMA"."DM_DIAG_FINETUNEPHE_NEXT3YR_LABELS"
