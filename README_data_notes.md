# Synthetic KP-region dataset: Smart Patient Readmission capstone

**All data is synthetic.** It contains no real patients and no real Kaiser Permanente data. Disclose this on your slides.

## Tables
| File | Type | Rows | Columns |
|---|---|---|---|
| patient_master.csv | Dimension | 10,000 | 10 |
| admissions.csv | Fact | 15,000 | 11 |
| clinical_measures.csv | Fact | 15,000 | 7 |
| utilization.csv | Fact | 15,000 | 7 |
| discharge_notes.csv | Fact | 15,000 | 5 |
| ref_hospitals.csv, ref_regions.csv | Small lookups | 23 / 9 | for star schema and hospital-level analysis |
| data_dictionary.csv | Documentation | 40 | every column explained |
| data_quality_issues_log.csv | Answer key | 35 | the deliberate problems injected (look only after you finish Phase 1) |

Keys: patient_master.patient_id (1) -> admissions.patient_id (many); admissions.admission_id (1) -> clinical_measures, utilization, discharge_notes (1:1).

## Regions (KP-style)
Northern California, Southern California, Mid-Atlantic States (MD, VA, DC), Northwest (OR / SW WA), Colorado, Georgia, Hawaii, Washington, Nevada.
Patient counts follow approximate KP membership, flattened so small regions still have enough rows (assumption). Costs use an illustrative region cost index (assumption, not KP data).
Hospital names are fictional ("<City> Medical Center (synthetic)").

## Target
`readmitted_30_days` = 1 if the same patient has another admission 1 to 30 days after this discharge. It is computed from the data, so SQL on admission/discharge dates reproduces it exactly. Overall rate is about 16%.

## Built-in patterns (so models and EDA have real signal; accuracy is deliberately imperfect)
Readmission rises with: age, ER visits, previous admissions, medication count, HbA1c of 9 or more, long stays, emergency admission, leaving against medical advice, living alone, long distance, Medicaid coverage, no follow-up within 7 days, unclear medication instructions or missed appointments in the notes. It falls with: follow-up within 7 days, completed teach-back, caregiver support.
Heart failure, COPD and chronic kidney disease have the highest rates. Small regional differences exist (e.g. Georgia a little higher).
Expected model AUC on a time-based split is roughly 0.70 to 0.75.

## Cautions
- Admissions after 31 May 2026 (58 rows, under 0.5%) are follow-on readmissions only, so avoid them in trend charts, or filter `admission_date <= '2026-05-31'`.
- Suggested time-based split: train on admissions before Oct 2025, validate Oct to Dec 2025, test 2026.
- Everything in these tables is known by discharge, so the prediction point is discharge. Do not add post-discharge columns when you build features.
- Notes were generated from templates; the themes in them are planted, so say so when you present the NLP findings.

## Regenerate
`python generate_kp_data.py --out kp_readmission_data` (seed 42 gives identical files for everyone).
