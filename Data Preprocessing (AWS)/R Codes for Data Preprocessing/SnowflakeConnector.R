library(DBI)
library(dplyr)
library(dbplyr)
library(odbc)
library(RODBC)
library(rstudioapi)

con <- DBI::dbConnect(odbc::odbc(),
                      "MU_CDM_DSN", 
                      authenticator='externalbrowser',
                      role='ANALYTICS',
                      warehouse='ANALYTICS_WH',
                      database='DEIDENTIFIED_PCORNET_CDM',
                      schema='CDM_C010R022')
                      #UID    = rstudioapi::askForPassword("Database user"),
                      #PWD    = rstudioapi::askForPassword("Database password"))


