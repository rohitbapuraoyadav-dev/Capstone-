/* =====================================================================
   CAPSTONE 1 - REQUIRED SQL ANALYSES  (MySQL 8, schema capstonedb)
   Run AFTER your cleaning steps. Run ONE statement at a time.

   COVERAGE MAP (what each requirement is demonstrated by)
   ---------------------------------------------------------------------
   INNER JOIN         view vw_admission_master (admissions x patient x clinical x utilization x notes)
   LEFT JOIN          view (hospital and region lookups) and query E3b
   GROUP BY           every analysis (E1 to E7)
   CASE statements    view bands (age, LOS, HbA1c), E4 utilizer group, E6/E7 labels
   Window functions   E3 (PARTITION BY), E5 (moving average, LAG, running total)
   Ranking functions  E1 RANK, E2 DENSE_RANK, E3 RANK, E4 ROW_NUMBER/DENSE_RANK/NTILE, E4b ROW_NUMBER

   SAMPLE ANALYSES    E1 diagnosis | E2 region | E3 LOS by hospital | E4 high utilizers | E5 rolling trends
   ===================================================================== */

USE capstonev2db;

/* ---------- D. ONE CLEAN ANALYTIC VIEW ---------------------------------
   WHY: join the 5 admission-level tables once, reuse the result everywhere.
   INNER JOIN where the child must exist (1:1 facts). LEFT JOIN for lookups,
   so no admission is dropped silently. CASE builds age / LOS / HbA1c bands.
   Implausible clinical values become NULL here instead of being deleted. */

DROP VIEW IF EXISTS vw_admission_master;
CREATE VIEW vw_admission_master AS
SELECT
    a.admission_id, a.patient_id, a.hospital_id,
    a.admission_date, a.discharge_date,
    DATE_FORMAT(a.admission_date, '%Y-%m')                          AS admission_month,
    a.admission_type, a.diagnosis, a.diagnosis_code,
    a.length_of_stay, a.discharge_disposition, a.readmitted_30_days,
    p.age, p.gender, p.region_code, p.region, p.state, p.city,
    p.coverage_plan, p.lives_alone, p.distance_to_hospital_km,
    h.hospital_name, h.bed_count, r.cost_index,
    -- clinical (implausible values become NULL instead of deleting the row)
    CASE WHEN c.bmi BETWEEN 12 AND 70            THEN c.bmi END          AS bmi_clean,
    CASE WHEN c.hba1c BETWEEN 3 AND 18           THEN c.hba1c END        AS hba1c_clean,
    CASE WHEN c.systolic_bp BETWEEN 70 AND 250   THEN c.systolic_bp END  AS systolic_bp_clean,
    CASE WHEN c.diastolic_bp BETWEEN 40 AND 150  THEN c.diastolic_bp END AS diastolic_bp_clean,
    c.medication_count,
    CASE WHEN c.hba1c IS NULL THEN 1 ELSE 0 END                          AS hba1c_was_missing,
    -- utilization
    u.er_visits, u.previous_admissions, u.outpatient_visits_12m,
    u.days_since_last_discharge, u.treatment_cost,
    -- notes
    d.physician_notes, d.follow_up_instructions, d.follow_up_within_7_days,
    -- derived bands (CASE statements)
    CASE WHEN p.age < 45 THEN '18-44' WHEN p.age < 65 THEN '45-64'
         WHEN p.age < 80 THEN '65-79' ELSE '80+' END                     AS age_band,
    CASE WHEN a.length_of_stay <= 3 THEN 'Short (1-3d)'
         WHEN a.length_of_stay <= 7 THEN 'Medium (4-7d)'
         ELSE 'Long (8d+)' END                                           AS los_band,
    CASE WHEN c.hba1c >= 9 THEN 'Poorly controlled (9+)'
         WHEN c.hba1c >= 6.5 THEN 'Diabetic range (6.5-8.9)'
         WHEN c.hba1c >= 5.7 THEN 'Pre-diabetic (5.7-6.4)'
         WHEN c.hba1c IS NULL THEN 'Not measured' ELSE 'Normal (<5.7)' END AS hba1c_band,
    CASE WHEN a.admission_date >= '2024-01-01' AND a.admission_date <= '2026-05-31'
         THEN 1 ELSE 0 END                                               AS in_reporting_window
FROM admissions a
INNER JOIN patient_master     p ON a.patient_id   = p.patient_id
INNER JOIN clinical_measures  c ON a.admission_id = c.admission_id
INNER JOIN utilization        u ON a.admission_id = u.admission_id
INNER JOIN discharge_notes    d ON a.admission_id = d.admission_id
LEFT  JOIN ref_hospitals      h ON a.hospital_id  = h.hospital_id
LEFT  JOIN ref_regions        r ON p.region_code  = r.region_code;

-- D1. Integration proof: view must keep every admission (15,000) - no silent row loss
SELECT (SELECT COUNT(*) FROM admissions)          AS admissions_rows,
       (SELECT COUNT(*) FROM vw_admission_master) AS view_rows;
       
/* ---------- E. BUSINESS ANALYSES (the 5 sample analyses) -------------- */
-- Always use the reporting window (<= 2026-05-31) for rates and trends.

-- E1. ANALYSIS 1: Readmission rate by DIAGNOSIS
--     Demonstrates: GROUP BY, RANK() ranking function (view supplies INNER + LEFT JOINs)
SELECT diagnosis,
       COUNT(*)                                   AS admissions,
       SUM(readmitted_30_days)                    AS readmissions,
       ROUND(100.0 * AVG(readmitted_30_days), 2)  AS readmit_rate_pct,
       RANK() OVER (ORDER BY AVG(readmitted_30_days) DESC) AS risk_rank
FROM vw_admission_master
WHERE in_reporting_window = 1
GROUP BY diagnosis
ORDER BY risk_rank;

-- E3b. EXTRA: LEFT JOIN on its own (hospitals with ZERO admissions would still appear)
--      Demonstrates: LEFT JOIN, GROUP BY
SELECT h.hospital_id, h.hospital_name, COUNT(a.admission_id) AS admissions
FROM ref_hospitals h
LEFT JOIN admissions a ON h.hospital_id = a.hospital_id
GROUP BY h.hospital_id, h.hospital_name
ORDER BY admissions ASC;

-- E4. ANALYSIS 4: HIGH UTILIZERS ranking (per patient totals)
--     Demonstrates: GROUP BY, CTE, ROW_NUMBER, DENSE_RANK, NTILE, CASE
WITH patient_totals AS (
    SELECT patient_id,
           COUNT(*)                         AS admissions_in_window,
           SUM(er_visits)                   AS total_er_visits,
           ROUND(SUM(treatment_cost), 2)    AS total_cost,
           SUM(length_of_stay)              AS total_days_in_hospital,
           SUM(readmitted_30_days)          AS readmissions
    FROM vw_admission_master
    WHERE in_reporting_window = 1
    GROUP BY patient_id
)
SELECT patient_id, admissions_in_window, total_er_visits, total_cost, readmissions,
       ROW_NUMBER() OVER (ORDER BY admissions_in_window DESC, total_cost DESC) AS row_no,
       DENSE_RANK() OVER (ORDER BY admissions_in_window DESC)                  AS admit_dense_rank,
       NTILE(10)    OVER (ORDER BY total_cost DESC)                            AS cost_decile,   -- 1 = top 10%
       CASE WHEN admissions_in_window >= 4 THEN 'Frequent utilizer'
            WHEN admissions_in_window >= 2 THEN 'Repeat patient'
            ELSE 'Single admission' END                                        AS utilizer_group
FROM patient_totals
ORDER BY row_no
LIMIT 25;

-- E4b. EXTRA: Top 3 costliest patients INSIDE each region
--      Demonstrates: ROW_NUMBER() OVER (PARTITION BY ...)
WITH region_patient AS (
    SELECT region, patient_id, ROUND(SUM(treatment_cost), 2) AS total_cost
    FROM vw_admission_master WHERE in_reporting_window = 1
    GROUP BY region, patient_id
), ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY region ORDER BY total_cost DESC) AS cost_rank_in_region
    FROM region_patient
)
SELECT * FROM ranked WHERE cost_rank_in_region <= 3 ORDER BY region, cost_rank_in_region;

-- E5. ANALYSIS 5: ROLLING admission trends
--     Demonstrates: window functions (moving AVG, LAG, running SUM), GROUP BY
--     monthly counts + 3-month moving average,
--     month-over-month change (LAG) and running total (cumulative SUM)
WITH monthly AS (
    SELECT admission_month,
           COUNT(*)                AS admissions,
           SUM(readmitted_30_days) AS readmissions
    FROM vw_admission_master
    WHERE in_reporting_window = 1
    GROUP BY admission_month
)
SELECT admission_month, admissions, readmissions,
       ROUND(100.0 * readmissions / admissions, 2) AS readmit_rate_pct,
       ROUND(AVG(admissions) OVER (ORDER BY admission_month
             ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 1)               AS admissions_3m_moving_avg,
       ROUND(100.0 * (admissions - LAG(admissions) OVER (ORDER BY admission_month))
             / LAG(admissions) OVER (ORDER BY admission_month), 1)       AS mom_change_pct,
       SUM(admissions) OVER (ORDER BY admission_month
             ROWS UNBOUNDED PRECEDING)                                   AS running_total_admissions
FROM monthly
ORDER BY admission_month;


-- E6. Financial: cost of the INDEX admissions (the stays that were followed by a readmission)
SELECT CASE WHEN readmitted_30_days = 1 THEN 'Readmitted' ELSE 'Not readmitted' END AS outcome,
       COUNT(*)                          AS admissions,
       ROUND(AVG(treatment_cost), 0)     AS avg_cost,
       ROUND(SUM(treatment_cost), 0)     AS total_cost
FROM vw_admission_master
WHERE in_reporting_window = 1
GROUP BY readmitted_30_days;

-- E6b. Financial: cost of the READMISSION STAYS themselves (the return visits).
--      A return visit is an admission 1-30 days after the patient's previous discharge.
--      No reporting-window filter here: the 58 late rows ARE return visits.
--      EXPECT: 2435 stays (same as the number of readmissions flagged), total about 41.1 million, average about 16,900
SELECT COUNT(*)                        AS readmission_stays,
       ROUND(SUM(treatment_cost), 0)   AS total_cost,
       ROUND(AVG(treatment_cost), 0)   AS avg_cost
FROM vw_admission_master
WHERE days_since_last_discharge BETWEEN 1 AND 30;

-- E7. Which groups to target first? Readmit rate by risk drivers (CASE bands + GROUP BY)
SELECT 'age_band' AS driver, age_band AS grp, COUNT(*) AS n, ROUND(100.0*AVG(readmitted_30_days),2) AS readmit_pct
FROM vw_admission_master WHERE in_reporting_window = 1 GROUP BY age_band
UNION ALL SELECT 'coverage_plan', coverage_plan, COUNT(*), ROUND(100.0*AVG(readmitted_30_days),2)
FROM vw_admission_master WHERE in_reporting_window = 1 GROUP BY coverage_plan
UNION ALL SELECT 'follow_up_within_7_days', CAST(follow_up_within_7_days AS CHAR), COUNT(*), ROUND(100.0*AVG(readmitted_30_days),2)
FROM vw_admission_master WHERE in_reporting_window = 1 GROUP BY follow_up_within_7_days
UNION ALL SELECT 'lives_alone', CAST(lives_alone AS CHAR), COUNT(*), ROUND(100.0*AVG(readmitted_30_days),2)
FROM vw_admission_master WHERE in_reporting_window = 1 GROUP BY lives_alone
UNION ALL SELECT 'hba1c_band', hba1c_band, COUNT(*), ROUND(100.0*AVG(readmitted_30_days),2)
FROM vw_admission_master WHERE in_reporting_window = 1 GROUP BY hba1c_band
UNION ALL SELECT 'los_band', los_band, COUNT(*), ROUND(100.0*AVG(readmitted_30_days),2)
FROM vw_admission_master WHERE in_reporting_window = 1 GROUP BY los_band
UNION ALL SELECT 'discharge_disposition', discharge_disposition, COUNT(*), ROUND(100.0*AVG(readmitted_30_days),2)
FROM vw_admission_master WHERE in_reporting_window = 1
GROUP BY discharge_disposition
ORDER BY driver, readmit_pct DESC;
