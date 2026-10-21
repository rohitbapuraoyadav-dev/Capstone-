#!/usr/bin/env python3
"""
Synthetic dataset generator: Smart Patient Readmission & Care Intervention Platform
-----------------------------------------------------------------------------------
Creates 5 related tables (+ 2 small reference tables) for a Kaiser Permanente-style
multi-region health system. ALL DATA IS SYNTHETIC. No real patients, no real KP data.

Tables
  patient_master      dimension   10,000 rows  (1 row per patient)
  admissions          fact        15,000 rows  (1 row per hospital admission)
  clinical_measures   fact        15,000 rows  (1 row per admission)
  utilization         fact        15,000 rows  (1 row per admission)
  discharge_notes     fact        15,000 rows  (1 row per admission)
  ref_hospitals / ref_regions     small lookup tables (not row-count targets)

Run:  python generate_kp_data.py --out kp_readmission_data
Same seed => identical files for every team member.
"""
import argparse
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
SEED = 42
N_PATIENTS = 10_000
N_FACT_ROWS = 15_000
WINDOW_START = date(2024, 1, 1)
INDEX_END = date(2026, 5, 31)         # last "index" admission date; follow-on admissions can run later
MIN_GAP_DAYS = 165                    # spacing between unrelated admission episodes of one patient
MAX_EPISODES = 5
MAX_CHAIN = 3                         # an episode = index admission + up to 2 readmissions
INTERCEPT = -2.85                     # tuned so overall 30-day readmission rate is about 16%
NOISE_SD = 0.85                       # unobserved randomness -> realistic (not perfect) model accuracy

# ----------------------------------------------------------------------------
# KP-style regions (membership-weighted, flattened so small regions still have enough rows)
# member_millions: approximate relative size used only to weight patient counts (assumption)
# ----------------------------------------------------------------------------
REGIONS = {
    "NCAL": dict(name="Northern California", members=4.50, cost_idx=1.20, readm_adj=0.00, dist_med=6, bmi_adj=0.0, n_hosp=4,
                 cities=[("San Francisco", "CA"), ("Oakland", "CA"), ("Sacramento", "CA"), ("San Jose", "CA"),
                         ("Fresno", "CA"), ("Santa Rosa", "CA"), ("Walnut Creek", "CA"), ("Vallejo", "CA"), ("Modesto", "CA")]),
    "SCAL": dict(name="Southern California", members=4.80, cost_idx=1.15, readm_adj=0.00, dist_med=6, bmi_adj=0.3, n_hosp=4,
                 cities=[("Los Angeles", "CA"), ("San Diego", "CA"), ("Riverside", "CA"), ("Anaheim", "CA"),
                         ("Bakersfield", "CA"), ("Fontana", "CA"), ("Irvine", "CA"), ("Palm Springs", "CA"), ("Woodland Hills", "CA")]),
    "MAS": dict(name="Mid-Atlantic States", members=0.82, cost_idx=1.05, readm_adj=0.06, dist_med=7, bmi_adj=0.8, n_hosp=3,
                cities=[("Washington", "DC"), ("Baltimore", "MD"), ("Silver Spring", "MD"), ("Rockville", "MD"), ("Largo", "MD"),
                        ("Arlington", "VA"), ("Fairfax", "VA"), ("Alexandria", "VA"), ("Richmond", "VA")]),
    "NW": dict(name="Northwest (OR / SW WA)", members=0.63, cost_idx=1.05, readm_adj=-0.05, dist_med=9, bmi_adj=0.0, n_hosp=2,
               cities=[("Portland", "OR"), ("Beaverton", "OR"), ("Salem", "OR"), ("Hillsboro", "OR"), ("Clackamas", "OR"),
                       ("Vancouver", "WA"), ("Longview", "WA")]),
    "CO": dict(name="Colorado", members=0.52, cost_idx=1.00, readm_adj=0.00, dist_med=11, bmi_adj=-1.6, n_hosp=2,
               cities=[("Denver", "CO"), ("Aurora", "CO"), ("Colorado Springs", "CO"), ("Boulder", "CO"),
                       ("Fort Collins", "CO"), ("Lakewood", "CO"), ("Pueblo", "CO"), ("Greeley", "CO")]),
    "GA": dict(name="Georgia", members=0.32, cost_idx=0.92, readm_adj=0.10, dist_med=10, bmi_adj=1.5, n_hosp=2,
               cities=[("Atlanta", "GA"), ("Marietta", "GA"), ("Decatur", "GA"), ("Lawrenceville", "GA"),
                       ("Athens", "GA"), ("Duluth", "GA")]),
    "HI": dict(name="Hawaii", members=0.27, cost_idx=1.25, readm_adj=-0.08, dist_med=9, bmi_adj=-0.5, n_hosp=2,
               cities=[("Honolulu", "HI"), ("Kapolei", "HI"), ("Kahului", "HI"), ("Lihue", "HI"), ("Hilo", "HI")]),
    "WA": dict(name="Washington", members=0.60, cost_idx=1.05, readm_adj=0.00, dist_med=10, bmi_adj=-0.3, n_hosp=2,
               cities=[("Seattle", "WA"), ("Tacoma", "WA"), ("Spokane", "WA"), ("Everett", "WA"),
                       ("Bellevue", "WA"), ("Olympia", "WA"), ("Renton", "WA")]),
    "NV": dict(name="Nevada", members=0.10, cost_idx=1.00, readm_adj=0.00, dist_med=8, bmi_adj=0.2, n_hosp=2,
               cities=[("Las Vegas", "NV"), ("Henderson", "NV"), ("North Las Vegas", "NV")]),
}

# ----------------------------------------------------------------------------
# Diagnoses (ICD-10 code, mean LOS, per-day cost, base share, age slope, readmission log-odds offset)
# ----------------------------------------------------------------------------
DX = {
    "Heart Failure":           dict(code="I50.9",  los=5.0, perdiem=3300, w=0.12, age_coef=0.9,  off=0.45),
    "COPD Exacerbation":       dict(code="J44.1",  los=4.2, perdiem=3000, w=0.11, age_coef=0.7,  off=0.40),
    "Pneumonia":               dict(code="J18.9",  los=4.3, perdiem=3100, w=0.13, age_coef=0.5,  off=0.10),
    "Diabetes Complications":  dict(code="E11.65", los=4.0, perdiem=3200, w=0.09, age_coef=-0.1, off=0.30),
    "Chronic Kidney Disease":  dict(code="N18.30", los=4.8, perdiem=3400, w=0.07, age_coef=0.7,  off=0.35),
    "Sepsis":                  dict(code="A41.9",  los=7.0, perdiem=4600, w=0.09, age_coef=0.4,  off=0.25),
    "Acute Myocardial Infarction": dict(code="I21.9", los=3.8, perdiem=5200, w=0.07, age_coef=0.5, off=0.15),
    "Stroke":                  dict(code="I63.9",  los=5.5, perdiem=4300, w=0.07, age_coef=0.7,  off=0.05),
    "Cellulitis":              dict(code="L03.90", los=3.8, perdiem=2800, w=0.06, age_coef=-0.1, off=-0.15),
    "Urinary Tract Infection": dict(code="N39.0",  los=3.2, perdiem=2700, w=0.06, age_coef=0.4,  off=-0.10),
    "GI Bleed":                dict(code="K92.2",  los=4.0, perdiem=3600, w=0.05, age_coef=0.5,  off=0.05),
    "Hypertensive Crisis":     dict(code="I16.9",  los=2.6, perdiem=3000, w=0.04, age_coef=0.0,  off=-0.05),
    "Dehydration":             dict(code="E86.0",  los=2.6, perdiem=2500, w=0.04, age_coef=0.3,  off=-0.20),
}
DM_MULT = {"Diabetes Complications": 4.0, "Chronic Kidney Disease": 1.6, "Heart Failure": 1.3,
           "Acute Myocardial Infarction": 1.3, "Stroke": 1.2}
CHRONIC_DX = {"Heart Failure", "COPD Exacerbation", "Diabetes Complications", "Chronic Kidney Disease"}
ER_SENSITIVE = {"Heart Failure", "COPD Exacerbation", "Diabetes Complications"}
ELECTIVE_OK = {"Chronic Kidney Disease", "Heart Failure", "Cellulitis", "Diabetes Complications"}
SYS_ADJ = {"Hypertensive Crisis": 55, "Sepsis": -18, "Dehydration": -10, "Acute Myocardial Infarction": 5}

ADM_TYPES = ["Emergency", "Urgent", "Elective"]
DISPOSITIONS = ["Home (self-care)", "Home with home health", "Skilled nursing facility",
                "Inpatient rehabilitation", "Left against medical advice"]
COVERAGE = ["Commercial (Employer)", "Medicare Advantage", "Medicaid", "ACA Marketplace"]

MONTH_WEIGHT = {1: 1.25, 2: 1.2, 3: 1.05, 4: 0.95, 5: 0.92, 6: 0.9, 7: 0.9, 8: 0.92, 9: 0.95, 10: 1.0, 11: 1.05, 12: 1.22}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------------------------------
# Reference + patient dimension
# ----------------------------------------------------------------------------
def make_hospitals(rng):
    rows = []
    for code, r in REGIONS.items():
        cities = [c for c in r["cities"]]
        idx = rng.permutation(len(cities))[: r["n_hosp"]]
        for i, ci in enumerate(idx, start=1):
            city, state = cities[ci]
            rows.append(dict(hospital_id=f"{code}-H{i:02d}", hospital_name=f"{city} Medical Center (synthetic)",
                             region_code=code, region=r["name"], state=state, city=city,
                             bed_count=int(rng.integers(150, 520))))
    return pd.DataFrame(rows)


def make_patients(rng):
    codes = list(REGIONS)
    members = np.array([REGIONS[c]["members"] for c in codes])
    share = members / members.sum()
    weights = 0.65 * share + 0.35 * (1.0 / len(codes))   # flatten so small regions have usable sample sizes
    weights = weights / weights.sum()
    patients = []
    for n in range(1, N_PATIENTS + 1):
        code = codes[rng.choice(len(codes), p=weights)]
        reg = REGIONS[code]
        city, state = reg["cities"][rng.integers(0, len(reg["cities"]))]
        grp = rng.choice(4, p=[0.12, 0.33, 0.40, 0.15])
        lo, hi = [(18, 39), (40, 64), (65, 84), (85, 95)][grp]
        age = int(rng.integers(lo, hi + 1))
        gender = str(rng.choice(["Female", "Male", "Other"], p=[0.52, 0.475, 0.005]))
        if age >= 65:
            cov = COVERAGE[rng.choice(4, p=[0.20, 0.68, 0.10, 0.02])]
        else:
            cov = COVERAGE[rng.choice(4, p=[0.60, 0.06, 0.22, 0.12])]
        lives_alone = int(rng.random() < 0.10 + 0.18 * (age >= 65) + 0.12 * (age >= 80))
        dist = float(np.clip(rng.lognormal(np.log(reg["dist_med"]), 0.8), 0.5, 150))
        bmi_base = float(np.clip(rng.normal(29 + reg["bmi_adj"], 6), 16, 55))
        p_dm = float(np.clip(0.10 + 0.004 * (age - 18) + 0.012 * (bmi_base - 25), 0.05, 0.65))
        diabetic = rng.random() < p_dm
        if diabetic:
            hba1c_base = float(np.clip(rng.normal(8.0, 1.5), 6.5, 14.0))
        elif rng.random() < 0.30:
            hba1c_base = float(np.clip(rng.normal(5.9, 0.2), 5.7, 6.4))
        else:
            hba1c_base = float(np.clip(rng.normal(5.3, 0.3), 4.3, 5.6))
        frailty = 0.5 * (age - 60) / 20 + rng.normal(0, 0.8) + 0.2 * diabetic
        er_rate = 0.45 * np.exp(0.55 * frailty + 0.30 * (cov == "Medicaid") + 0.2 * lives_alone)
        if rng.random() < 0.05:
            er_rate *= 3                                  # a small group of frequent ER users
        er_rate = float(min(er_rate, 4.0))
        patients.append(dict(
            patient_id=f"PAT{n:06d}", age=age, gender=gender, region_code=code, region=reg["name"],
            state=state, city=city, coverage_plan=cov, lives_alone=lives_alone,
            distance_to_hospital_km=round(dist, 1),
            # hidden (latent) attributes - not exported
            bmi_base=bmi_base, diabetic=bool(diabetic), hba1c_base=hba1c_base, frailty=float(frailty),
            er_rate=er_rate, prior_base=int(min(rng.poisson(0.35 * np.exp(0.45 * frailty)), 6)),
            sys_base=float(120 + 0.35 * (age - 50) + 0.5 * (bmi_base - 27) + rng.normal(0, 10)),
            med_base=float(1.5 + 0.045 * max(age - 30, 0) + 1.2 * diabetic + 0.6 * frailty),
            out_rate=float(max(0.5, 4.5 + 0.02 * (age - 50) - 0.9 * lives_alone - 0.8 * (cov == "Medicaid") - 0.03 * dist)),
        ))
    return patients


# ----------------------------------------------------------------------------
# Admission-level logic
# ----------------------------------------------------------------------------
def dx_weights(pt):
    w = []
    for name, d in DX.items():
        wt = d["w"] * np.exp(d["age_coef"] * (pt["age"] - 60) / 20)
        if pt["diabetic"]:
            wt *= DM_MULT.get(name, 1.0)
        w.append(wt)
    w = np.array(w)
    return w / w.sum()


def pick_disposition(rng, pt, dx):
    age = pt["age"]
    snf = 0.03 + 0.10 * (age >= 75) + 0.07 * (age >= 85) + (0.20 if dx == "Stroke" else 0)
    rehab = 0.02 + (0.30 if dx == "Stroke" else 0) + 0.02 * (age >= 65)
    hh = 0.08 + (0.10 if dx in ("Heart Failure", "COPD Exacerbation", "Cellulitis") else 0) + 0.04 * (age >= 75)
    ama = 0.012 + 0.008 * (age < 50)
    home = max(0.05, 1 - (snf + rehab + hh + ama))
    p = np.array([home, hh, snf, rehab, ama])
    p = p / p.sum()
    return DISPOSITIONS[rng.choice(5, p=p)]


def build_admission(rng, pt, adm_date, history, followon, prev_dx, hospitals_by_region, intercept, noise_sd):
    dx_names = list(DX)
    if followon and prev_dx is not None and rng.random() < 0.65:
        dx = prev_dx
    else:
        dx = dx_names[rng.choice(len(dx_names), p=dx_weights(pt))]
    d = DX[dx]
    reg = REGIONS[pt["region_code"]]

    if followon:
        p_type = [0.88, 0.12, 0.0]
    elif dx in ELECTIVE_OK:
        p_type = [0.78, 0.17, 0.05]
    else:
        p_type = [0.82, 0.18, 0.0]
    adm_type = ADM_TYPES[rng.choice(3, p=p_type)]

    mean_los = d["los"] * (1 + 0.10 * float(np.clip(pt["frailty"], -1.5, 2.5)))
    los = int(np.clip(round(1 + rng.gamma(2.2, max(mean_los - 1, 0.5) / 2.2)), 1, 30))
    disp = pick_disposition(rng, pt, dx)
    discharge_date = adm_date + timedelta(days=los)

    # earlier admissions already in the dataset (this patient only)
    prior_rows = [h for h in history if h["discharge_date"] < adm_date and (adm_date - h["admission_date"]).days <= 365]
    n_prior = len(prior_rows)
    n_prior_er = sum(h["admission_type"] == "Emergency" for h in prior_rows)
    earlier = [h for h in history if h["discharge_date"] < adm_date]
    days_since_last = (adm_date - max(h["discharge_date"] for h in earlier)).days if earlier else None
    elapsed = (adm_date - WINDOW_START).days
    base_prior = int(rng.binomial(pt["prior_base"], max(0.0, 1 - elapsed / 365)))
    previous_admissions = int(min(base_prior + n_prior, 12))
    er_visits = int(min(rng.poisson(pt["er_rate"] * (1.25 if dx in ER_SENSITIVE else 1.0)) + n_prior_er, 15))
    outpatient_visits = int(rng.poisson(pt["out_rate"]))

    # clinical measures
    bmi = float(np.clip(pt["bmi_base"] + rng.normal(0, 0.4), 14, 60))
    hba1c = float(np.clip(pt["hba1c_base"] + rng.normal(0, 0.2) + (0.5 if dx == "Diabetes Complications" else 0), 4.0, 15.0))
    sys_bp = float(np.clip(pt["sys_base"] + rng.normal(0, 6) + SYS_ADJ.get(dx, 0), 80, 230))
    dia_bp = float(np.clip(0.55 * sys_bp + 8 + rng.normal(0, 6), 40, 130))
    dia_bp = min(dia_bp, sys_bp - 20)
    med_count = int(np.clip(rng.poisson(max(0.5, pt["med_base"] + (1.0 if dx in CHRONIC_DX else 0))), 0, 25))

    # cost (USD)
    cost = 0.75 * (2500 + d["perdiem"] * los) * reg["cost_idx"] * float(rng.lognormal(0, 0.22))

    # process / social themes (drive the discharge note text AND readmission risk)
    medicaid = pt["coverage_plan"] == "Medicaid"
    dist = pt["distance_to_hospital_km"]
    unclear = rng.random() < float(np.clip(0.08 + 0.015 * max(med_count - 4, 0) + 0.07 * (pt["age"] >= 80), 0, 0.6))
    missed = rng.random() < (0.06 + 0.18 * pt["lives_alone"] + 0.08 * medicaid + 0.10 * (pt["frailty"] > 1))
    transport = rng.random() < (0.03 + 0.12 * (dist > 25) + 0.04 * (dist > 12) + 0.07 * medicaid)
    diet = rng.random() < (0.05 + 0.10 * (dx in CHRONIC_DX) + 0.05 * (bmi > 35))
    teach_back = rng.random() < float(np.clip(0.60 - 0.25 * unclear - 0.10 * (pt["age"] >= 85), 0.1, 0.9))
    caregiver = rng.random() < float(np.clip(0.70 - 0.50 * pt["lives_alone"], 0.1, 0.9))
    p_fu = (0.66 - 0.20 * transport - 0.12 * missed - 0.30 * (disp == DISPOSITIONS[4])
            + 0.10 * (disp == DISPOSITIONS[1]) + 0.08 * (dx in ("Heart Failure", "Acute Myocardial Infarction")))
    follow_up_7 = rng.random() < float(np.clip(p_fu, 0.05, 0.95))

    # readmission log-odds (what makes the label learnable, but never perfectly)
    hb = hba1c
    z = (intercept + d["off"] + reg["readm_adj"]
         + 0.35 * (pt["age"] - 60) / 20
         + 0.28 * min(er_visits, 6)
         + 0.30 * min(previous_admissions, 6)
         + 0.045 * (med_count - 5)
         + 0.30 * (hb >= 9) + 0.15 * (7 <= hb < 9)
         + 0.06 * float(np.clip(los - 4.5, -3, 10))
         + 0.40 * (adm_type == "Emergency")
         + {0: 0.0, 1: 0.05, 2: 0.15, 3: -0.05, 4: 0.9}[DISPOSITIONS.index(disp)]
         - 0.50 * follow_up_7
         + 0.40 * unclear + 0.35 * missed + 0.25 * transport + 0.15 * diet
         - 0.35 * teach_back - 0.20 * caregiver
         + 0.30 * pt["lives_alone"] + 0.008 * min(dist, 60)
         + 0.22 * medicaid
         + rng.normal(0, noise_sd))

    hosp_list = hospitals_by_region[pt["region_code"]]
    hospital_id = hosp_list[rng.integers(0, len(hosp_list))]

    return dict(
        patient_id=pt["patient_id"], hospital_id=hospital_id, admission_date=adm_date, discharge_date=discharge_date,
        admission_type=adm_type, diagnosis=dx, diagnosis_code=d["code"], length_of_stay=los, discharge_disposition=disp,
        bmi=bmi, hba1c=hba1c, systolic_bp=sys_bp, diastolic_bp=dia_bp, medication_count=med_count,
        er_visits=er_visits, previous_admissions=previous_admissions, outpatient_visits_12m=outpatient_visits,
        days_since_last_discharge=days_since_last, treatment_cost=cost, z=float(z),
        f_unclear=bool(unclear), f_missed=bool(missed), f_transport=bool(transport), f_diet=bool(diet),
        f_teach_back=bool(teach_back), f_caregiver=bool(caregiver), follow_up_7=bool(follow_up_7),
    )


def build_episode(rng, pt, start_date, history, hospitals_by_region, intercept, noise_sd):
    rows, prev_dx, adm_date = [], None, start_date
    for pos in range(MAX_CHAIN):
        row = build_admission(rng, pt, adm_date, history + rows, pos > 0, prev_dx, hospitals_by_region, intercept, noise_sd)
        rows.append(row)
        if pos == MAX_CHAIN - 1:
            break
        if rng.random() < sigmoid(row["z"]):
            gap = min(30, 2 + int(rng.exponential(9)))     # readmission 2-30 days after discharge
            adm_date = row["discharge_date"] + timedelta(days=gap)
            prev_dx = row["diagnosis"]
        else:
            break
    return rows


def day_probs():
    n_days = (INDEX_END - WINDOW_START).days + 1
    days = [WINDOW_START + timedelta(days=i) for i in range(n_days)]
    w = np.array([MONTH_WEIGHT[d.month] for d in days])
    return days, w / w.sum()


def plan_dates(rng, k, days, probs):
    if k == 1:
        return [days[rng.choice(len(days), p=probs)]]
    span = len(days) - (k - 1) * MIN_GAP_DAYS
    offs = sorted(int(x) for x in rng.integers(0, span, k))
    return [WINDOW_START + timedelta(days=offs[i] + i * MIN_GAP_DAYS) for i in range(k)]


def generate_rows(extra_rate, seed):
    rng = np.random.default_rng(seed)
    hospitals = make_hospitals(rng)
    patients = make_patients(rng)
    hospitals_by_region = hospitals.groupby("region_code")["hospital_id"].apply(list).to_dict()
    days, probs = day_probs()
    episodes = {}
    for pt in patients:
        k = 1 + min(MAX_EPISODES - 1, int(rng.poisson(extra_rate * np.exp(0.6 * pt["frailty"]))))
        history, eps = [], []
        for d0 in plan_dates(rng, k, days, probs):
            rows = build_episode(rng, pt, d0, history, hospitals_by_region, INTERCEPT, NOISE_SD)
            history += rows
            eps.append(rows)
        episodes[pt["patient_id"]] = eps
    return rng, hospitals, patients, episodes


def trim_to_target(rng, episodes, target):
    n = sum(len(ep) for eps in episodes.values() for ep in eps)
    while n > target:
        excess = n - target
        multi = [pid for pid, eps in episodes.items() if len(eps) >= 2 and len(eps[-1]) <= excess]
        if multi:
            pid = multi[rng.integers(0, len(multi))]
            n -= len(episodes[pid].pop())
            continue
        chain = [pid for pid, eps in episodes.items() if len(eps[-1]) >= 2]
        if chain:
            pid = chain[rng.integers(0, len(chain))]
            episodes[pid][-1].pop()
            n -= 1
            continue
        single = [pid for pid, eps in episodes.items() if len(eps) >= 2]
        pid = single[rng.integers(0, len(single))]
        n -= len(episodes[pid].pop())
    return n


# ----------------------------------------------------------------------------
# Discharge-note text
# ----------------------------------------------------------------------------
CLIN = {
    "Heart Failure": ["Admitted with acute decompensated heart failure; diuresed with IV furosemide with good response.",
                      "Presented with shortness of breath and leg swelling from a heart failure exacerbation; treated with IV diuretics and fluid restriction.",
                      "Heart failure flare with volume overload; net negative fluid balance achieved and transitioned to oral diuretics."],
    "COPD Exacerbation": ["Admitted with COPD exacerbation; treated with nebulized bronchodilators, steroids and supplemental oxygen.",
                          "Increased cough and wheeze consistent with COPD flare; improved on steroids and antibiotics.",
                          "COPD exacerbation requiring oxygen; weaned to room air before discharge."],
    "Pneumonia": ["Admitted with community-acquired pneumonia; completed IV antibiotics and switched to oral therapy.",
                  "Fever, cough and infiltrate on chest X-ray consistent with pneumonia; clinically improved on antibiotics.",
                  "Pneumonia treated with antibiotics and oxygen support; oxygen saturation normal on room air at discharge."],
    "Diabetes Complications": ["Admitted with uncontrolled diabetes and hyperglycemia; insulin regimen adjusted.",
                               "Hyperglycemia with elevated glucose levels; started on a revised insulin plan with diabetes education.",
                               "Diabetes-related complication managed with insulin titration and nutrition consult."],
    "Chronic Kidney Disease": ["Admitted with worsening chronic kidney disease and fluid retention; nephrology consulted.",
                               "Kidney function monitored closely; medications adjusted for renal dosing.",
                               "CKD with electrolyte imbalance corrected during stay; renal diet reviewed."],
    "Sepsis": ["Admitted with sepsis from an infectious source; treated with IV fluids and broad-spectrum antibiotics.",
               "Sepsis responding to antibiotics and fluid resuscitation; cultures reviewed and therapy narrowed.",
               "Presented with fever and low blood pressure; sepsis protocol followed with clinical improvement."],
    "Acute Myocardial Infarction": ["Admitted with acute myocardial infarction; managed with antiplatelet therapy and cardiology follow-up.",
                                    "Chest pain with elevated troponin; treated per cardiac protocol and started on secondary prevention medications.",
                                    "Heart attack managed medically with new cardiac medications and rehabilitation referral."],
    "Stroke": ["Admitted with acute ischemic stroke; started on antithrombotic therapy with physical therapy evaluation.",
               "Stroke with residual weakness; neurology consulted and rehabilitation planned.",
               "Cerebral infarction managed with antiplatelets and statin; speech and physical therapy involved."],
    "Cellulitis": ["Admitted with lower-extremity cellulitis; treated with IV antibiotics and wound care.",
                   "Skin infection with erythema and swelling improved on antibiotics and elevation.",
                   "Cellulitis responding to antibiotics; wound care teaching provided."],
    "Urinary Tract Infection": ["Admitted with complicated urinary tract infection; treated with IV then oral antibiotics.",
                                "UTI with confusion and fever; improved after antibiotics and hydration.",
                                "Urinary infection treated; urine culture guided antibiotic choice."],
    "GI Bleed": ["Admitted with gastrointestinal bleeding; transfused as needed and endoscopy performed.",
                 "GI bleed with drop in hemoglobin; stabilized with fluids and acid suppression.",
                 "Upper GI bleed managed with endoscopic evaluation and medication review."],
    "Hypertensive Crisis": ["Admitted with severely elevated blood pressure; medications adjusted with gradual control.",
                            "Hypertensive urgency with headache; blood pressure lowered and regimen optimised.",
                            "Elevated blood pressure readings corrected after medication titration."],
    "Dehydration": ["Admitted with dehydration and electrolyte disturbance; treated with IV fluids.",
                    "Poor oral intake with dehydration; fluids and electrolytes replaced.",
                    "Dehydration with weakness resolved after IV hydration."],
}
COURSE = ["Hospital course was uncomplicated.", "Hospital course notable for transient hypotension that resolved.",
          "Pain controlled with acetaminophen.", "Vital signs stable at discharge.", "Physical therapy cleared patient for discharge.",
          "Labs reviewed and improving at discharge."]
DISP_TEXT = {
    DISPOSITIONS[0]: ["Discharged home.", "Patient discharged home in stable condition."],
    DISPOSITIONS[1]: ["Discharged home with home health nursing.", "Home health nurse arranged after discharge."],
    DISPOSITIONS[2]: ["Transferred to a skilled nursing facility for ongoing care.", "Discharged to skilled nursing facility."],
    DISPOSITIONS[3]: ["Transferred to inpatient rehabilitation.", "Discharged to rehabilitation facility."],
    DISPOSITIONS[4]: ["Patient left against medical advice before planned teaching was completed.", "Patient chose to leave against medical advice."],
}
T_UNCLEAR = ["Patient expressed confusion about the new medication schedule.",
             "Medication changes made at discharge; patient unsure which home medications to continue.",
             "Pharmacy noted patient did not fully understand updated dosing instructions."]
T_MISSED = ["Patient has missed several recent clinic appointments.", "History of missed follow-up visits noted by primary care.",
            "Two prior no-shows to outpatient follow-up."]
T_TRANSPORT = ["Patient reports difficulty arranging transportation to appointments.", "Limited access to a ride for clinic visits.",
               "Lives far from the clinic and has no reliable transport."]
T_ALONE = ["Patient lives alone with limited support at home.", "No caregiver at home after discharge."]
T_DIET = ["Reviewed diet and fluid restrictions; patient admits difficulty adhering.", "Counseled on dietary changes; poor adherence reported."]
T_TEACH = ["Teach-back completed with patient and family; understanding confirmed.",
           "Discharge education delivered with teach-back; patient explained the plan clearly."]
T_CARE = ["Daughter will help with medications and appointments.", "Spouse attended discharge teaching and will assist at home.",
          "Family member will stay with patient for the first week."]
T_FU_YES = ["Follow-up appointment arranged before discharge.", "Care manager will call the patient within 72 hours."]
T_FU_NO = ["No follow-up appointment was booked before discharge.", "Patient asked to call the clinic to schedule follow-up."]


def pick(rng, items):
    return str(items[rng.integers(0, len(items))])


def make_note(rng, r, pt):
    parts = [pick(rng, CLIN[r["diagnosis"]])]
    if rng.random() < 0.5:
        parts.append(pick(rng, COURSE))
    parts.append(pick(rng, DISP_TEXT[r["discharge_disposition"]]))
    themes = []
    if r["f_unclear"]:
        themes.append(pick(rng, T_UNCLEAR))
    if r["f_missed"]:
        themes.append(pick(rng, T_MISSED))
    if r["f_transport"]:
        themes.append(pick(rng, T_TRANSPORT))
    if pt["lives_alone"] and rng.random() < 0.7:
        themes.append(pick(rng, T_ALONE))
    if r["f_diet"]:
        themes.append(pick(rng, T_DIET))
    if r["f_teach_back"]:
        themes.append(pick(rng, T_TEACH))
    if r["f_caregiver"] and not pt["lives_alone"]:
        themes.append(pick(rng, T_CARE))
    themes.append(pick(rng, T_FU_YES if r["follow_up_7"] else T_FU_NO))
    rng.shuffle(themes)
    return " ".join(parts + themes)


def follow_up_text(rng, r):
    dx, disp = r["diagnosis"], r["discharge_disposition"]
    if r["follow_up_7"]:
        opts = ["PCP follow-up within 7 days", "Telehealth check-in within 72 hours, then clinic visit within 7 days",
                "Care manager call within 48 hours and PCP visit within 7 days"]
        if dx in ("Heart Failure", "Acute Myocardial Infarction"):
            opts.append("Cardiology visit within 7 days")
        if dx in ("COPD Exacerbation", "Pneumonia"):
            opts.append("Pulmonary or PCP visit within 7 days")
        if disp == DISPOSITIONS[1]:
            opts.append("Home health nurse visit within 48 hours; PCP visit within 7 days")
    else:
        opts = ["PCP follow-up in 2-4 weeks", "Specialist follow-up in 2-3 weeks", "Follow up as needed",
                "No follow-up appointment scheduled", "Patient to call clinic to schedule follow-up"]
    return pick(rng, opts)


# ----------------------------------------------------------------------------
# Data dictionary
# ----------------------------------------------------------------------------
DICTIONARY = [
    ("patient_master", "patient_id", "text (PK)", "Unique patient key, PAT000001 to PAT010000"),
    ("patient_master", "age", "int", "Age in years (18-95) at start of study window"),
    ("patient_master", "gender", "text", "Female / Male / Other"),
    ("patient_master", "region_code", "text", "KP-style region code: NCAL, SCAL, MAS, NW, CO, GA, HI, WA, NV"),
    ("patient_master", "region", "text", "Region name (see ref_regions)"),
    ("patient_master", "state", "text", "US state or DC of the patient's city"),
    ("patient_master", "city", "text", "City in the region's service area"),
    ("patient_master", "coverage_plan", "text", "Commercial (Employer), Medicare Advantage, Medicaid, ACA Marketplace"),
    ("patient_master", "lives_alone", "int 0/1", "1 = lives alone (social-risk factor)"),
    ("patient_master", "distance_to_hospital_km", "float", "Distance from home to nearest hospital in km"),
    ("admissions", "admission_id", "text (PK)", "Unique admission key, ADM0000001 onward, ordered by admission date"),
    ("admissions", "patient_id", "text (FK)", "Links to patient_master"),
    ("admissions", "hospital_id", "text (FK)", "Links to ref_hospitals; hospital is always in the patient's region"),
    ("admissions", "admission_date", "date", "Admission date"),
    ("admissions", "discharge_date", "date", "Discharge date (= admission_date + length_of_stay)"),
    ("admissions", "admission_type", "text", "Emergency / Urgent / Elective"),
    ("admissions", "diagnosis", "text", "Primary diagnosis group"),
    ("admissions", "diagnosis_code", "text", "ICD-10-CM code for the diagnosis group"),
    ("admissions", "length_of_stay", "int", "Days in hospital"),
    ("admissions", "discharge_disposition", "text", "Where the patient went after discharge"),
    ("admissions", "readmitted_30_days", "int 0/1 (TARGET)", "1 = patient has another admission 1-30 days after THIS discharge"),
    ("clinical_measures", "admission_id", "text (PK/FK)", "One row per admission"),
    ("clinical_measures", "patient_id", "text (FK)", "Links to patient_master"),
    ("clinical_measures", "bmi", "float", "Body mass index (kg/m2)"),
    ("clinical_measures", "hba1c", "float", "HbA1c percent; missing if not measured"),
    ("clinical_measures", "systolic_bp", "float", "Systolic blood pressure mmHg"),
    ("clinical_measures", "diastolic_bp", "float", "Diastolic blood pressure mmHg"),
    ("clinical_measures", "medication_count", "int", "Number of active medications at discharge"),
    ("utilization", "admission_id", "text (PK/FK)", "One row per admission"),
    ("utilization", "patient_id", "text (FK)", "Links to patient_master"),
    ("utilization", "er_visits", "int", "ER visits in the 12 months before this admission"),
    ("utilization", "previous_admissions", "int", "Hospital admissions in the 12 months before this admission (history before Jan 2024 is synthetic)"),
    ("utilization", "outpatient_visits_12m", "int", "Outpatient visits in the prior 12 months"),
    ("utilization", "days_since_last_discharge", "float", "Days since previous discharge in the dataset; blank if none"),
    ("utilization", "treatment_cost", "float (USD)", "Total cost of this admission in US dollars (synthetic, region-adjusted)"),
    ("discharge_notes", "admission_id", "text (PK/FK)", "One row per admission"),
    ("discharge_notes", "patient_id", "text (FK)", "Links to patient_master"),
    ("discharge_notes", "physician_notes", "text", "Synthetic discharge summary (blank for about 2% of rows)"),
    ("discharge_notes", "follow_up_instructions", "text", "Follow-up plan given at discharge"),
    ("discharge_notes", "follow_up_within_7_days", "int 0/1", "1 = follow-up care booked within 7 days of discharge"),
]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # find the smallest extra_rate that gives at least 15,000 rows, then trim to exactly 15,000
    extra_rate = 0.20
    while True:
        rng, hospitals, patients, episodes = generate_rows(extra_rate, SEED)
        total = sum(len(ep) for eps in episodes.values() for ep in eps)
        if total >= N_FACT_ROWS:
            break
        extra_rate = round(extra_rate + 0.01, 2)
    trim_to_target(rng, episodes, N_FACT_ROWS)

    pt_by_id = {p["patient_id"]: p for p in patients}
    rows = [r for eps in episodes.values() for ep in eps for r in ep]
    adm = pd.DataFrame(rows)
    adm = adm.sort_values(["admission_date", "patient_id"]).reset_index(drop=True)
    adm.insert(0, "admission_id", [f"ADM{i:07d}" for i in range(1, len(adm) + 1)])

    # label derived from the data itself: a later admission within 30 days of discharge
    adm = adm.sort_values(["patient_id", "admission_date"]).reset_index(drop=True)
    next_adm = adm.groupby("patient_id")["admission_date"].shift(-1)
    gap = (pd.to_datetime(next_adm) - pd.to_datetime(adm["discharge_date"])).dt.days
    adm["readmitted_30_days"] = ((gap >= 0) & (gap <= 30)).astype(int)
    adm = adm.sort_values("admission_id").reset_index(drop=True)

    # notes (needs row + patient)
    note_rng = np.random.default_rng(SEED + 1)
    notes_text, fu_text = [], []
    for _, r in adm.iterrows():
        pt = pt_by_id[r["patient_id"]]
        notes_text.append(make_note(note_rng, r, pt))
        fu_text.append(follow_up_text(note_rng, r))

    # ---------------- tables ----------------
    patient_master = pd.DataFrame(patients)[
        ["patient_id", "age", "gender", "region_code", "region", "state", "city",
         "coverage_plan", "lives_alone", "distance_to_hospital_km"]]
    admissions = adm[["admission_id", "patient_id", "hospital_id", "admission_date", "discharge_date", "admission_type",
                      "diagnosis", "diagnosis_code", "length_of_stay", "discharge_disposition", "readmitted_30_days"]].copy()
    clinical = adm[["admission_id", "patient_id", "bmi", "hba1c", "systolic_bp", "diastolic_bp", "medication_count"]].copy()
    clinical[["bmi", "hba1c", "systolic_bp", "diastolic_bp"]] = clinical[["bmi", "hba1c", "systolic_bp", "diastolic_bp"]].round(1)
    utilization = adm[["admission_id", "patient_id", "er_visits", "previous_admissions", "outpatient_visits_12m",
                       "days_since_last_discharge", "treatment_cost"]].copy()
    utilization["treatment_cost"] = utilization["treatment_cost"].round(2)
    notes = pd.DataFrame({"admission_id": adm["admission_id"], "patient_id": adm["patient_id"],
                          "physician_notes": notes_text, "follow_up_instructions": fu_text,
                          "follow_up_within_7_days": adm["follow_up_7"].astype(int)})

    # ---------------- deliberate, documented data-quality issues ----------------
    iss_rng = np.random.default_rng(SEED + 2)
    log = []

    def log_issue(table, key, col, issue, old, new):
        log.append(dict(table=table, record_id=key, column=col, issue=issue, original_value=old, injected_value=new))

    n = len(clinical)
    # (a) missing values
    is_dm = admissions["diagnosis"].eq("Diabetes Complications").values
    miss_hba1c = iss_rng.random(n) < np.where(is_dm, 0.01, 0.10)
    miss_bmi = iss_rng.random(n) < 0.03
    miss_bp = iss_rng.random(n) < 0.01
    clinical.loc[miss_hba1c, "hba1c"] = np.nan
    clinical.loc[miss_bmi, "bmi"] = np.nan
    clinical.loc[miss_bp, ["systolic_bp", "diastolic_bp"]] = np.nan
    blank_notes = iss_rng.random(n) < 0.02
    notes.loc[blank_notes, "physician_notes"] = np.nan
    log.append(dict(table="clinical_measures", record_id="(many)", column="hba1c, bmi, systolic_bp, diastolic_bp",
                    issue="random missing values (hba1c about 9%, bmi 3%, blood pressure 1%)", original_value="", injected_value="blank"))
    log.append(dict(table="discharge_notes", record_id="(many)", column="physician_notes",
                    issue="blank notes (about 2%)", original_value="", injected_value="blank"))

    # (b) impossible values (a handful, so cleaning rules can catch them)
    ok = clinical.index[clinical["bmi"].notna() & clinical["hba1c"].notna() & clinical["systolic_bp"].notna()].to_numpy()
    pick_idx = iss_rng.choice(ok, size=26, replace=False)
    for i in pick_idx[:6]:
        old = clinical.at[i, "bmi"]; new = float(iss_rng.choice([3.1, 4.2, 5.0]))
        clinical.at[i, "bmi"] = new
        log_issue("clinical_measures", clinical.at[i, "admission_id"], "bmi", "impossibly low BMI", old, new)
    for i in pick_idx[6:12]:
        old = clinical.at[i, "bmi"]; new = float(iss_rng.choice([78.4, 85.0, 92.3]))
        clinical.at[i, "bmi"] = new
        log_issue("clinical_measures", clinical.at[i, "admission_id"], "bmi", "implausibly high BMI", old, new)
    for i in pick_idx[12:17]:
        old = clinical.at[i, "hba1c"]; new = float(iss_rng.choice([21.5, 24.0, 28.8]))
        clinical.at[i, "hba1c"] = new
        log_issue("clinical_measures", clinical.at[i, "admission_id"], "hba1c", "impossible HbA1c (above 20%)", old, new)
    for i in pick_idx[17:23]:
        s, d_ = clinical.at[i, "systolic_bp"], clinical.at[i, "diastolic_bp"]
        clinical.at[i, "systolic_bp"], clinical.at[i, "diastolic_bp"] = d_, s
        log_issue("clinical_measures", clinical.at[i, "admission_id"], "systolic_bp/diastolic_bp",
                  "systolic and diastolic swapped (systolic lower than diastolic)", f"{s}/{d_}", f"{d_}/{s}")
    cost_idx = iss_rng.choice(len(utilization), size=10, replace=False)
    for i in cost_idx:
        old = utilization.at[i, "treatment_cost"]; new = round(float(old) * float(iss_rng.uniform(4, 7)), 2)
        utilization.at[i, "treatment_cost"] = new
        log_issue("utilization", utilization.at[i, "admission_id"], "treatment_cost", "extreme cost outlier", old, new)

    # ---------------- reference tables ----------------
    ref_regions = pd.DataFrame([dict(region_code=c, region=r["name"],
                                     states=", ".join(sorted({s for _, s in r["cities"]})),
                                     assumed_member_millions=r["members"], cost_index=r["cost_idx"])
                                for c, r in REGIONS.items()])
    ref_hospitals = hospitals

    # ---------------- write ----------------
    files = {
        "patient_master.csv": patient_master, "admissions.csv": admissions, "clinical_measures.csv": clinical,
        "utilization.csv": utilization, "discharge_notes.csv": notes, "ref_hospitals.csv": ref_hospitals,
        "ref_regions.csv": ref_regions, "data_quality_issues_log.csv": pd.DataFrame(log),
        "data_dictionary.csv": pd.DataFrame(DICTIONARY, columns=["table", "column", "type", "description"]),
    }
    for name, df in files.items():
        df.to_csv(os.path.join(out_dir, name), index=False)
    print(f"extra_rate used: {extra_rate}")
    for name, df in files.items():
        print(f"{name:32s} rows={len(df):6d} cols={df.shape[1]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kp_readmission_data")
    main(ap.parse_args().out)
