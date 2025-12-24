import pandas as pd
from pathlib import Path
from typing import Dict, Iterable, List


MIMIC_BASE_DEFAULT = Path("Dataset/mimic-iv-3.1/mimic-iv-3.1")


def _load_core_tables(base: Path) -> Dict[str, pd.DataFrame]:
    """Load core MIMIC-IV tables needed for the HTN ICU cohort."""
    hosp = base / "hosp"
    icu = base / "icu"

    patients = pd.read_csv(hosp / "patients.csv")
    admissions = pd.read_csv(
        hosp / "admissions.csv",
        parse_dates=["admittime", "dischtime", "deathtime", "edregtime", "edouttime"],
    )
    diagnoses_icd = pd.read_csv(hosp / "diagnoses_icd.csv")
    d_icd = pd.read_csv(hosp / "d_icd_diagnoses.csv")
    prescriptions = pd.read_csv(
        hosp / "prescriptions.csv",
        parse_dates=["starttime", "stoptime"],
    )
    omr = pd.read_csv(
        hosp / "omr.csv",
        parse_dates=["chartdate"],
    )
    d_labitems = pd.read_csv(hosp / "d_labitems.csv")
    labevents = pd.read_csv(
        hosp / "labevents.csv",
        usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
        parse_dates=["charttime"],
    )
    icustays = pd.read_csv(
        icu / "icustays.csv",
        parse_dates=["intime", "outtime"],
    )

    return {
        "patients": patients,
        "admissions": admissions,
        "diagnoses_icd": diagnoses_icd,
        "d_icd": d_icd,
        "prescriptions": prescriptions,
        "omr": omr,
        "d_labitems": d_labitems,
        "labevents": labevents,
        "icustays": icustays,
    }


def _select_htn_admissions(diagnoses_icd: pd.DataFrame, d_icd: pd.DataFrame) -> pd.DataFrame:
    """Return unique (subject_id, hadm_id) pairs with any hypertension diagnosis."""
    # Simple definition: any ICD long_title mentioning "hypertens"
    htn_codes = d_icd[
        d_icd["long_title"].str.contains("hypertens", case=False, na=False)
    ]["icd_code"].unique()

    htn_diag = diagnoses_icd[diagnoses_icd["icd_code"].isin(htn_codes)]
    htn_adm = htn_diag[["subject_id", "hadm_id"]].drop_duplicates()
    return htn_adm


def _flag_comorbidities(
    diagnoses_icd: pd.DataFrame,
    d_icd: pd.DataFrame,
    keywords: Dict[str, Iterable[str]],
) -> pd.DataFrame:
    """
    Build simple comorbidity flags by ICD long_title keyword search.

    keywords: mapping from flag name to list of substrings to search in long_title.
    """
    pieces: List[pd.DataFrame] = []
    for flag, patterns in keywords.items():
        pattern = "|".join(patterns)
        codes = d_icd[
            d_icd["long_title"].str.contains(pattern, case=False, na=False)
        ]["icd_code"].unique()
        diag_flag = diagnoses_icd[diagnoses_icd["icd_code"].isin(codes)]
        sub = diag_flag[["subject_id", "hadm_id"]].drop_duplicates()
        sub[flag] = 1
        pieces.append(sub)

    if not pieces:
        return pd.DataFrame(columns=["subject_id", "hadm_id"])

    comorbid = pieces[0]
    for df in pieces[1:]:
        comorbid = comorbid.merge(
            df, on=["subject_id", "hadm_id"], how="outer", validate="one_to_one"
        )

    # Fill NaN flags with 0
    flag_cols = [c for c in comorbid.columns if c not in ("subject_id", "hadm_id")]
    comorbid[flag_cols] = comorbid[flag_cols].fillna(0).astype(int)
    return comorbid


def _build_acei_arb_flags(
    icu_htn: pd.DataFrame,
    prescriptions: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each ICU stay, flag exposure to ACEI / ARB during the ICU stay window.
    """
    acei_names = [
        "LISINOPRIL",
        "CAPTOPRIL",
        "ENALAPRIL",
        "BENAZEPRIL",
        "FOSINOPRIL",
        "MOEXIPRIL",
        "PERINDOPRIL",
        "QUINAPRIL",
        "RAMIPRIL",
        "TRANDOLAPRIL",
    ]
    arb_names = [
        "LOSARTAN",
        "VALSARTAN",
        "IRBESARTAN",
        "OLMESARTAN",
        "CANDESARTAN",
        "TELMISARTAN",
        "EPROSARTAN",
        "AZILSARTAN",
    ]

    # Limit prescriptions to our cohort admissions to reduce size
    cohort_hadm = icu_htn[["subject_id", "hadm_id"]].drop_duplicates()
    pres = prescriptions.merge(
        cohort_hadm, on=["subject_id", "hadm_id"], how="inner"
    )

    # Guard against missing times
    pres = pres.dropna(subset=["starttime", "stoptime"])

    pres["drug_upper"] = pres["drug"].astype(str).str.upper()
    acei_mask = False
    for name in acei_names:
        acei_mask |= pres["drug_upper"].str.contains(name, na=False)
    arb_mask = False
    for name in arb_names:
        arb_mask |= pres["drug_upper"].str.contains(name, na=False)

    pres["is_acei"] = acei_mask
    pres["is_arb"] = arb_mask

    # Merge prescriptions with ICU stays for time window overlap
    pres_icu = pres.merge(
        icu_htn[["subject_id", "hadm_id", "stay_id", "intime", "outtime"]],
        on=["subject_id", "hadm_id"],
        how="inner",
    )

    # Keep only orders overlapping the ICU stay
    overlap = (pres_icu["stoptime"] >= pres_icu["intime"]) & (
        pres_icu["starttime"] <= pres_icu["outtime"]
    )
    pres_icu = pres_icu[overlap]

    exposure = pres_icu.groupby("stay_id")[["is_acei", "is_arb"]].any().astype(int)
    exposure = exposure.rename(
        columns={"is_acei": "acei_exposed", "is_arb": "arb_exposed"}
    ).reset_index()

    return icu_htn.merge(exposure, on="stay_id", how="left")


def _extract_omr_baseline(icu_htn: pd.DataFrame, omr: pd.DataFrame) -> pd.DataFrame:
    """
    Extract baseline BMI and clinic blood pressure from OMR.

    For each ICU stay, we take the last outpatient measurement on or before
    ICU admission date.
    """
    wanted = ["BMI (kg/m2)", "Blood Pressure"]
    omr_use = omr[omr["result_name"].isin(wanted)].copy()
    if omr_use.empty:
        return pd.DataFrame(
            {
                "stay_id": icu_htn["stay_id"].unique(),
                "bmi_omr": pd.NA,
                "clinic_bp": pd.NA,
                "clinic_sbp": pd.NA,
                "clinic_dbp": pd.NA,
            }
        )

    icu = icu_htn[["subject_id", "stay_id", "intime"]].copy()
    icu["intime_date"] = icu["intime"].dt.normalize()

    merged = omr_use.merge(icu, on="subject_id", how="inner")
    merged = merged[merged["chartdate"].notna()]
    merged = merged[merged["chartdate"] <= merged["intime_date"]]
    if merged.empty:
        return pd.DataFrame(
            {
                "stay_id": icu_htn["stay_id"].unique(),
                "bmi_omr": pd.NA,
                "clinic_bp": pd.NA,
                "clinic_sbp": pd.NA,
                "clinic_dbp": pd.NA,
            }
        )

    merged = merged.sort_values(["stay_id", "result_name", "chartdate"])
    last = merged.groupby(["stay_id", "result_name"]).tail(1)

    pivot = last.pivot(index="stay_id", columns="result_name", values="result_value").reset_index()
    pivot = pivot.rename(
        columns={
            "BMI (kg/m2)": "bmi_omr",
            "Blood Pressure": "clinic_bp",
        }
    )

    # Ensure BMI is numeric
    if "bmi_omr" in pivot.columns:
        pivot["bmi_omr"] = pd.to_numeric(pivot["bmi_omr"], errors="coerce")

    # Parse systolic / diastolic from clinic blood pressure string "SBP/DBP"
    if "clinic_bp" in pivot.columns:
        bp_split = pivot["clinic_bp"].astype(str).str.split("/", n=1, expand=True)
        pivot["clinic_sbp"] = pd.to_numeric(bp_split[0], errors="coerce")
        pivot["clinic_dbp"] = pd.to_numeric(bp_split[1], errors="coerce")
    else:
        pivot["clinic_sbp"] = pd.NA
        pivot["clinic_dbp"] = pd.NA

    return pivot[["stay_id", "bmi_omr", "clinic_bp", "clinic_sbp", "clinic_dbp"]]


def _extract_lab_baseline(
    icu_htn: pd.DataFrame,
    labevents: pd.DataFrame,
    d_labitems: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract baseline lab values around ICU admission (±24h window).

    Currently:
      - creatinine_baseline
      - bun_baseline
      - glucose_baseline
      - sodium_baseline
    """
    lab_keywords: Dict[str, Iterable[str]] = {
        "creatinine": ["Creatinine"],
        "bun": ["Urea Nitrogen"],
        "glucose": ["Glucose"],
        "sodium": ["Sodium"],
    }

    item_to_var: Dict[int, str] = {}
    for var_name, patterns in lab_keywords.items():
        pattern = "|".join(patterns)
        mask = (
            d_labitems["fluid"].str.lower().eq("blood")
            & d_labitems["label"].str.contains(pattern, case=False, na=False)
        )
        itemids = d_labitems.loc[mask, "itemid"].unique()
        for itemid in itemids:
            item_to_var[int(itemid)] = var_name

    if not item_to_var:
        return pd.DataFrame(
            {
                "stay_id": icu_htn["stay_id"].unique(),
                "creatinine_baseline": pd.NA,
                "bun_baseline": pd.NA,
                "glucose_baseline": pd.NA,
                "sodium_baseline": pd.NA,
            }
        )

    lab = labevents[labevents["itemid"].isin(item_to_var.keys())].copy()

    icu = icu_htn[["subject_id", "hadm_id", "stay_id", "intime"]].copy()
    lab = lab.merge(icu, on=["subject_id", "hadm_id"], how="inner")
    lab = lab.dropna(subset=["charttime", "valuenum"])

    # Restrict to a ±24h window around ICU admission
    lab["dt"] = (lab["charttime"] - lab["intime"]).abs()
    lab = lab[lab["dt"] <= pd.Timedelta("1 days")]
    if lab.empty:
        return pd.DataFrame(
            {
                "stay_id": icu_htn["stay_id"].unique(),
                "creatinine_baseline": pd.NA,
                "bun_baseline": pd.NA,
                "glucose_baseline": pd.NA,
                "sodium_baseline": pd.NA,
            }
        )

    lab["var_name"] = lab["itemid"].map(item_to_var)
    lab = lab.dropna(subset=["var_name"])

    lab = lab.sort_values(["stay_id", "var_name", "dt"])
    first = lab.groupby(["stay_id", "var_name"]).head(1)

    pivot = first.pivot(index="stay_id", columns="var_name", values="valuenum").reset_index()
    pivot = pivot.rename(
        columns={
            "creatinine": "creatinine_baseline",
            "bun": "bun_baseline",
            "glucose": "glucose_baseline",
            "sodium": "sodium_baseline",
        }
    )

    for col in [
        "creatinine_baseline",
        "bun_baseline",
        "glucose_baseline",
        "sodium_baseline",
    ]:
        if col not in pivot.columns:
            pivot[col] = pd.NA

    return pivot[
        [
            "stay_id",
            "creatinine_baseline",
            "bun_baseline",
            "glucose_baseline",
            "sodium_baseline",
        ]
    ]


def build_htn_icu_cohort(
    base: Path | str = MIMIC_BASE_DEFAULT,
) -> pd.DataFrame:
    """
    Build a hypertension ICU cohort table for causal analysis.

    One row per ICU stay with:
      - keys: subject_id, hadm_id, stay_id
      - baseline: age, sex, race
      - mechanistic risk factors: bmi_omr, clinic_sbp, clinic_dbp,
        creatinine_baseline, bun_baseline, glucose_baseline, sodium_baseline
      - comorbidities: diabetes, ckd, cad, heart_failure (0/1)
      - exposure: acei_exposed, arb_exposed (0/1)
      - outcomes: hospital_mortality (0/1), icu_los_days (float)
    """
    base_path = Path(base)
    tables = _load_core_tables(base_path)

    patients = tables["patients"]
    admissions = tables["admissions"]
    diagnoses_icd = tables["diagnoses_icd"]
    d_icd = tables["d_icd"]
    prescriptions = tables["prescriptions"]
    omr = tables["omr"]
    d_labitems = tables["d_labitems"]
    labevents = tables["labevents"]
    icustays = tables["icustays"]

    # 1. Hypertension admissions and ICU stays
    htn_adm = _select_htn_admissions(diagnoses_icd, d_icd)
    icu_htn = icustays.merge(
        htn_adm, on=["subject_id", "hadm_id"], how="inner"
    )

    # 2. Baseline demographics (age, gender, race)
    demo = icu_htn.merge(
        patients[["subject_id", "gender", "anchor_age"]],
        on="subject_id",
        how="left",
    ).merge(
        admissions[["subject_id", "hadm_id", "race", "hospital_expire_flag"]],
        on=["subject_id", "hadm_id"],
        how="left",
    )

    demo = demo.rename(
        columns={
            "anchor_age": "age",
            "gender": "sex",
            "hospital_expire_flag": "hospital_mortality",
        }
    )

    # 3. Simple comorbidity flags from ICD titles
    comorbid_keywords = {
        "diabetes": ["diabetes"],
        "ckd": ["chronic kidney disease"],
        "cad": ["coronary atherosclerosis", "ischemic heart disease"],
        "heart_failure": ["heart failure"],
    }
    comorbid = _flag_comorbidities(diagnoses_icd, d_icd, comorbid_keywords)

    cohort = demo.merge(comorbid, on=["subject_id", "hadm_id"], how="left")
    for col in ["diabetes", "ckd", "cad", "heart_failure"]:
        if col not in cohort.columns:
            cohort[col] = 0
        else:
            cohort[col] = cohort[col].fillna(0).astype(int)

    # 4. Mechanistic baseline variables from OMR and labevents
    omr_baseline = _extract_omr_baseline(cohort, omr)
    lab_baseline = _extract_lab_baseline(cohort, labevents, d_labitems)

    cohort = cohort.merge(omr_baseline, on="stay_id", how="left")
    cohort = cohort.merge(lab_baseline, on="stay_id", how="left")

    # 5. Exposure flags: ACEI / ARB during ICU stay
    cohort = _build_acei_arb_flags(cohort, prescriptions)
    cohort["acei_exposed"] = cohort["acei_exposed"].fillna(0).astype(int)
    cohort["arb_exposed"] = cohort["arb_exposed"].fillna(0).astype(int)

    # 6. Outcome: LOS in days (from icustays.los)
    cohort["icu_los_days"] = cohort["los"]

    return cohort[
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "age",
            "sex",
            "race",
            "bmi_omr",
            "clinic_bp",
            "clinic_sbp",
            "clinic_dbp",
            "creatinine_baseline",
            "bun_baseline",
            "glucose_baseline",
            "sodium_baseline",
            "diabetes",
            "ckd",
            "cad",
            "heart_failure",
            "acei_exposed",
            "arb_exposed",
            "hospital_mortality",
            "icu_los_days",
        ]
    ]


def example_ps_weighting(cohort: pd.DataFrame):
    """
    Minimal example of propensity-score weighting for
    ACEI exposure -> hospital mortality within HTN ICU cohort.

    Returns a dict with ATE estimate and some basic info.
    """
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    df = cohort.dropna(
        subset=[
            "age",
            "sex",
            "diabetes",
            "ckd",
            "cad",
            "heart_failure",
            "acei_exposed",
            "hospital_mortality",
        ]
    ).copy()

    # Simple binary encoding for sex
    df["sex_female"] = (df["sex"] == "F").astype(int)

    confounders = [
        "age",
        "sex_female",
        "diabetes",
        "ckd",
        "cad",
        "heart_failure",
    ]

    X = df[confounders].values
    T = df["acei_exposed"].values.astype(int)
    Y = df["hospital_mortality"].values.astype(int)

    # Estimate propensity scores
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(X, T)
    ps = ps_model.predict_proba(X)[:, 1]

    # Stabilized IPW
    p_t = T.mean()
    w = T * p_t / ps + (1 - T) * (1 - p_t) / (1 - ps)

    # Weighted means
    mu1 = np.sum(w * T * Y) / np.sum(w * T)
    mu0 = np.sum(w * (1 - T) * Y) / np.sum(w * (1 - T))
    ate = mu1 - mu0

    return {
        "n": int(len(df)),
        "treated_rate": float(T.mean()),
        "outcome_rate": float(Y.mean()),
        "ate_acei_on_hosp_mortality": float(ate),
    }
