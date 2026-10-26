USE capstonev2db;
/* ---------- 1. INSPECT (read-only) ---------------------------------- */
 
-- 1.1 Data type of every column
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'capstonedb'
  AND table_name IN ('patient_master','admissions','clinical_measures','utilization',
                     'discharge_notes','ref_hospitals','ref_regions')
ORDER BY table_name, ordinal_position;

-- 1.2 See the problem: days_since_last_discharge was imported as TEXT
--     Expect: 10000 blanks (first admission of each patient) and 5000 numbers like '12.0'
SELECT CASE WHEN days_since_last_discharge IS NULL THEN 'NULL'
            WHEN TRIM(days_since_last_discharge) = '' THEN 'empty text'
            ELSE 'has a value' END AS kind,
       COUNT(*) AS n
FROM utilization
GROUP BY kind;


/* ---------- 2. FIX days_since_last_discharge ------------------------ */
 
-- 2.1 List every value that is not a number
SELECT days_since_last_discharge AS bad_value, COUNT(*) AS n
FROM utilization
WHERE days_since_last_discharge IS NOT NULL
  AND days_since_last_discharge NOT REGEXP '^[0-9]+([.][0-9]+)?$'
GROUP BY days_since_last_discharge;

-- 2.2 Turn every non-number into a real NULL
SET SQL_SAFE_UPDATES = 0;
UPDATE utilization
SET days_since_last_discharge = NULL
WHERE days_since_last_discharge IS NOT NULL
  AND days_since_last_discharge NOT REGEXP '^[0-9]+([.][0-9]+)?$';
SET SQL_SAFE_UPDATES = 1;

-- 2.3 Verify: no bad values left (expect 0), and the row counts
SELECT COUNT(*) AS bad_values
FROM utilization
WHERE days_since_last_discharge IS NOT NULL
  AND days_since_last_discharge NOT REGEXP '^[0-9]+([.][0-9]+)?$';
 
SELECT COUNT(*) AS total_rows,
       COUNT(days_since_last_discharge) AS non_null_values
FROM utilization;

-- 2.4 Change the column from TEXT to whole numbers
ALTER TABLE utilization
MODIFY days_since_last_discharge INT NULL;

-- 2.5 Verify the new type and sense-check the values
SELECT COLUMN_TYPE
FROM information_schema.COLUMNS
WHERE TABLE_NAME = 'utilization'
  AND COLUMN_NAME = 'days_since_last_discharge';
  
  SELECT MIN(days_since_last_discharge) AS min_days,
       MAX(days_since_last_discharge) AS max_days,
       AVG(days_since_last_discharge) AS avg_days,
       COUNT(days_since_last_discharge) AS non_null
FROM utilization;

-- 2.6 Describe the gaps between visits
SELECT
  CASE
    WHEN days_since_last_discharge IS NULL THEN '0. No prior discharge'
    WHEN days_since_last_discharge <= 30  THEN '1. 0-30 days'
    WHEN days_since_last_discharge <= 90  THEN '2. 31-90 days'
    WHEN days_since_last_discharge <= 180 THEN '3. 91-180 days'
    WHEN days_since_last_discharge <= 365 THEN '4. 181-365 days'
    ELSE '5. Over 365 days'
    
    /* ---------- 3. FIX THE ID COLUMNS ----------------------------------- */
 
-- 3.1 Longest ID and missing IDs (expect max_len 9 / 10 / 8 / 4 and null_ids 0)

SELECT 'patient_master.patient_id'    AS col, MAX(CHAR_LENGTH(patient_id))   AS max_len, SUM(patient_id   IS NULL) AS null_ids FROM patient_master
UNION ALL
SELECT 'admissions.admission_id',      MAX(CHAR_LENGTH(admission_id)),  SUM(admission_id IS NULL) FROM admissions
UNION ALL
SELECT 'ref_hospitals.hospital_id',    MAX(CHAR_LENGTH(hospital_id)),   SUM(hospital_id  IS NULL) FROM ref_hospitals
UNION ALL
SELECT 'ref_regions.region_code',      MAX(CHAR_LENGTH(region_code)),   SUM(region_code  IS NULL) FROM ref_regions
UNION ALL
SELECT 'clinical_measures.admission_id', MAX(CHAR_LENGTH(admission_id)), SUM(admission_id IS NULL) FROM clinical_measures
UNION ALL
SELECT 'utilization.admission_id',     MAX(CHAR_LENGTH(admission_id)),  SUM(admission_id  IS NULL) FROM utilization
UNION ALL
SELECT 'discharge_notes.admission_id', MAX(CHAR_LENGTH(admission_id)),  SUM(admission_id  IS NULL) FROM discharge_notes;

-- 3.2 Duplicate IDs (each of the four must return no rows)
SELECT patient_id, COUNT(*) FROM patient_master GROUP BY patient_id HAVING COUNT(*) > 1;
SELECT admission_id, COUNT(*) FROM admissions GROUP BY admission_id HAVING COUNT(*) > 1;
SELECT hospital_id, COUNT(*) FROM ref_hospitals GROUP BY hospital_id HAVING COUNT(*) > 1;
SELECT region_code, COUNT(*) FROM ref_regions GROUP BY region_code HAVING COUNT(*) > 1;

 
-- 3.3 ID column types BEFORE the change
SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND COLUMN_NAME IN ('patient_id','admission_id','hospital_id','region_code')
ORDER BY TABLE_NAME, COLUMN_NAME;

-- 3.4  Change ID columns from TEXT to VARCHAR(20)
ALTER TABLE patient_master    MODIFY patient_id   VARCHAR(20) NOT NULL, MODIFY region_code VARCHAR(20);
ALTER TABLE admissions        MODIFY admission_id VARCHAR(20) NOT NULL, MODIFY patient_id VARCHAR(20), MODIFY hospital_id VARCHAR(20);
ALTER TABLE clinical_measures MODIFY admission_id VARCHAR(20) NOT NULL, MODIFY patient_id VARCHAR(20);
ALTER TABLE utilization       MODIFY admission_id VARCHAR(20) NOT NULL, MODIFY patient_id VARCHAR(20);
ALTER TABLE discharge_notes   MODIFY admission_id VARCHAR(20) NOT NULL, MODIFY patient_id VARCHAR(20);
ALTER TABLE ref_hospitals     MODIFY hospital_id  VARCHAR(20) NOT NULL, MODIFY region_code VARCHAR(20);
ALTER TABLE ref_regions       MODIFY region_code  VARCHAR(20) NOT NULL;   -- ONLY ADDED LINE: ref_regions was missing, and its primary key needs it

 
-- 3.5 ID column types AFTER the change (all should now say varchar(20))
SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND COLUMN_NAME IN ('patient_id','admission_id','hospital_id','region_code')
ORDER BY TABLE_NAME, COLUMN_NAME;

/* ---------- 4. PRIMARY KEYS AND INDEXES ----------------------------- */
 
-- 4.1 Primary keys (each table once only)
ALTER TABLE patient_master ADD PRIMARY KEY (patient_id);
ALTER TABLE admissions     ADD PRIMARY KEY (admission_id);
ALTER TABLE ref_hospitals  ADD PRIMARY KEY (hospital_id);
ALTER TABLE ref_regions    ADD PRIMARY KEY (region_code);

-- 4.2 [ONCE] Indexes for faster joins
CREATE INDEX ix_adm_patient  ON admissions(patient_id);
CREATE INDEX ix_adm_hospital ON admissions(hospital_id);
CREATE INDEX ix_adm_date     ON admissions(admission_date);
CREATE INDEX ix_clin_adm     ON clinical_measures(admission_id);
CREATE INDEX ix_util_adm     ON utilization(admission_id);
CREATE INDEX ix_notes_adm    ON discharge_notes(admission_id);

SELECT COLUMN_TYPE
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = 'capstonev2db' AND TABLE_NAME = 'admissions'
  AND COLUMN_NAME IN ('admission_date','discharge_date');

SELECT admission_date, discharge_date, LENGTH(admission_date) AS len
FROM admissions LIMIT 5;

SELECT COUNT(*) AS bad_dates
FROM admissions
WHERE STR_TO_DATE(admission_date, '%Y-%m-%d') IS NULL
   OR STR_TO_DATE(discharge_date, '%Y-%m-%d') IS NULL;
   
ALTER TABLE admissions
MODIFY admission_date DATE,
MODIFY discharge_date DATE;

CREATE INDEX ix_adm_date ON admissions(admission_date);

SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'capstonedb'
ORDER BY table_name, ordinal_position;

-- 4.3 Verify: every key and index now in the schema
SELECT TABLE_NAME, INDEX_NAME, COLUMN_NAME
FROM information_schema.STATISTICS
WHERE TABLE_SCHEMA = DATABASE()
ORDER BY TABLE_NAME, INDEX_NAME;

/* ---------- 5. LOOK AT THE TARGET (read-only) -------------- */
 
-- 5.1 Target distribution
SELECT readmitted_30_days, COUNT(*) AS n
FROM admissions
GROUP BY readmitted_30_days;
 
-- 5.2 Target versus the backward-looking gap column
SELECT
  a.readmitted_30_days,
  CASE WHEN u.days_since_last_discharge IS NULL THEN 'no prior discharge'
       WHEN u.days_since_last_discharge <= 30 THEN '<= 30 days'
       ELSE '> 30 days' END AS gap_bucket,
  COUNT(*) AS n
FROM admissions a
JOIN utilization u ON a.admission_id = u.admission_id
GROUP BY a.readmitted_30_days, gap_bucket
ORDER BY a.readmitted_30_days, gap_bucket;