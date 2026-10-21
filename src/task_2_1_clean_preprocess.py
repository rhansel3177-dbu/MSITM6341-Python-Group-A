"""
Midterm Project - Task 2.1: Acquire, Clean, and Preprocess Data
================================================================
Dataset : IBM Telco Customer Churn (WA_Fn-UseC_-Telco-Customer-Churn.csv)
Goal    : Prepare the data for EDA (Task 2.2) and for predicting whether a
          customer will churn (Task 2.3, binary classification, target = Churn).

Outputs (written to data/processed/):
    telco_churn_clean.csv      -> cleaned, human-readable data (use for EDA)
    telco_churn_train.csv      -> encoded + scaled training set (use for ML)
    telco_churn_test.csv       -> encoded + scaled test set     (use for ML)

Run (from the repo root):  python src/task_2_1_clean_preprocess.py
Requires: pandas, numpy, scikit-learn
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42                      # fixed seed so results are reproducible
ROOT_DIR = Path(__file__).resolve().parent.parent       # repo root
RAW_FILE = ROOT_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT_DIR = ROOT_DIR / "data" / "processed"               # generated files (git-ignored)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def section(title):
    """Print a visible header so the console output is easy to follow."""
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ==============================================================================
# (a) DATA ACQUISITION
# ==============================================================================
# Source : Kaggle - "Telco Customer Churn" by BlastChar
#          https://www.kaggle.com/datasets/blastchar/telco-customer-churn
#          (originally an IBM Cognos Analytics sample dataset)
# Method : File-based. The CSV was downloaded manually from the Kaggle page
#          ("Download" button, free Kaggle account required) and saved in the
#          repo's data/ folder. No API call is needed.
#          Alternative with the Kaggle CLI:
#              kaggle datasets download -d blastchar/telco-customer-churn --unzip
# Content: 7,043 customers x 21 columns. Each row is one customer of a
#          fictional telecom company: demographics, services subscribed,
#          account/billing info, and whether they left in the last month (Churn).
section("(a) DATA ACQUISITION")

# Fail early with a clear message if the CSV isn't next to this script.
if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"Raw data not found: {RAW_FILE}\n"
        "Download it from Kaggle (link above) and place it in the data/ folder."
    )

df = pd.read_csv(RAW_FILE)
print(f"Loaded {RAW_FILE.name}: {df.shape[0]} rows x {df.shape[1]} columns")
print(df.head())
print("\nColumn types as loaded:")
print(df.dtypes)


# ==============================================================================
# (b) DATA CLEANING
# ==============================================================================
section("(b) DATA CLEANING")

# --- b1. Strip stray whitespace from all text columns -------------------------
# This is a defensive step: leading or trailing spaces would make "Yes" and
# "Yes " count as different categories.
text_cols = df.select_dtypes(include=["object", "string"]).columns
for col in text_cols:
    df[col] = df[col].str.strip()

# --- b2. Fix an invalid data type: TotalCharges -------------------------------
# TotalCharges should be numeric but pandas read it as text, which means some
# values are not numbers. Converting with errors="coerce" turns those invalid
# entries into NaN so we can find them.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# --- b3. Check missing values -------------------------------------------------
print("Missing values per column (after type fix):")
missing = df.isnull().sum()
print(missing[missing > 0])

blank_rows = df[df["TotalCharges"].isnull()]
print(f"\nRows with missing TotalCharges: {len(blank_rows)}")
print(blank_rows[["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]])

# DECISION: impute TotalCharges = 0 rather than drop or use the mean/median.
# Every one of these customers has tenure == 0: they just signed up and have
# not been billed yet, so their true total really is $0. Imputing 0 records
# that true value and keeps all 7,043 customers. Dropping 11 rows (0.16%) would
# also be acceptable but discards valid customers, and mean/median imputation
# would wrongly give brand-new customers thousands of dollars in charges.
assert (blank_rows["tenure"] == 0).all(), "Unexpected: blank TotalCharges with tenure > 0"
df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
print(f"Missing values remaining: {df.isnull().sum().sum()}")

# --- b4. Remove duplicates ----------------------------------------------------
# Check both full-row duplicates and duplicate customer IDs. Each customer
# should appear exactly once.
full_dupes = df.duplicated().sum()
id_dupes = df["customerID"].duplicated().sum()
print(f"\nFully duplicated rows: {full_dupes}")
print(f"Duplicated customerIDs: {id_dupes}")
# DECISION: keep the de-duplication step even though none were found, so the
# pipeline stays safe if the data is refreshed later.
df = df.drop_duplicates(subset="customerID", keep="first")

# --- b5. Check for invalid entries --------------------------------------------
# Numeric sanity checks: no negative tenure or charges, and tenure within the
# 0-72 month range documented for this dataset.
print("\nNumeric range checks:")
print(f"  Negative tenure:          {(df['tenure'] < 0).sum()}")
print(f"  Tenure above 72 months:   {(df['tenure'] > 72).sum()}")
print(f"  Negative MonthlyCharges:  {(df['MonthlyCharges'] < 0).sum()}")
print(f"  Negative TotalCharges:    {(df['TotalCharges'] < 0).sum()}")

# Logical consistency check: TotalCharges should be close to
# tenure x MonthlyCharges. Prices change over time, so we only flag customers
# whose total is wildly off (outside 50%-150% of the expected value).
expected_total = df["tenure"] * df["MonthlyCharges"]
ratio = np.where(expected_total > 0, df["TotalCharges"] / expected_total, 1.0)
inconsistent = ((ratio < 0.5) | (ratio > 1.5)).sum()
print(f"  TotalCharges inconsistent with tenure x MonthlyCharges: {inconsistent}")
print(df.loc[(ratio < 0.5) | (ratio > 1.5),
             ["customerID", "tenure", "MonthlyCharges", "TotalCharges"]])
# DECISION: keep these 2 rows. Both are short-tenure customers whose plan price
# likely changed after their first bill (an upgrade or downgrade), which is a
# plausible real-world pattern, not a data-entry error.

# Categorical checks: print the allowed values of every category column to
# confirm there are no typos or unexpected labels. (Re-select the text columns
# now, because TotalCharges is no longer text after the type fix.)
category_cols = df.select_dtypes(include=["object", "string"]).columns.drop("customerID")
print("\nCategory values per column:")
for col in category_cols:
    print(f"  {col:<17} {sorted(df[col].unique())}")

# DECISION: recode SeniorCitizen from 0/1 to "Yes"/"No". Every other yes/no
# column uses text labels, so this keeps the cleaned EDA file consistent and
# makes charts readable. It is encoded back to 0/1 in preprocessing.
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

# --- b6. Simplify redundant categories ----------------------------------------
# DECISION: recode "No internet service" (six add-on columns) and "No phone
# service" (MultipleLines) to "No". That information is already captured by
# the InternetService and PhoneService columns, so keeping it would only create
# redundant dummy columns later. "No" is accurate: the customer does not have
# that add-on.
internet_addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]
for col in internet_addons:
    df[col] = df[col].replace("No internet service", "No")
df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

# DECISION: keep customerID in the cleaned file for traceability. It is a
# unique identifier with no predictive value, so it is dropped before modeling.
print(f"\nCleaned dataset shape: {df.shape}")
df.to_csv(OUT_DIR / "telco_churn_clean.csv", index=False)
print("Saved telco_churn_clean.csv (use this for EDA in Task 2.2)")


# ==============================================================================
# (c) DATA PREPROCESSING
# ==============================================================================
# Three techniques are used (the requirement is at least 2):
#   1. Feature engineering - create new, more informative columns
#   2. Encoding            - turn categorical text into numbers
#   3. Scaling             - put numeric features on a common scale
section("(c) DATA PREPROCESSING")

model_df = df.drop(columns=["customerID"]).copy()

# --- Technique 1: FEATURE ENGINEERING -----------------------------------------
# NumServices: how many add-on services the customer pays for (0-8). Customers
# who are more "embedded" in the product may be less likely to leave.
service_cols = ["PhoneService", "MultipleLines"] + internet_addons
model_df["NumServices"] = (model_df[service_cols] == "Yes").sum(axis=1)

# AvgMonthlySpend: TotalCharges / tenure, i.e. what they actually paid per month
# on average. Differs from MonthlyCharges when prices or plans have changed.
# For tenure 0, fall back to the current MonthlyCharges (avoids divide-by-zero).
model_df["AvgMonthlySpend"] = (
    model_df["TotalCharges"] / model_df["tenure"]
).where(model_df["tenure"] > 0, model_df["MonthlyCharges"])

# TenureGroup: bucket tenure into customer-lifecycle stages. Churn risk is
# usually highest in the first year, so a group label can capture that pattern
# more directly than a raw month count.

# DECISION: bins of 0-12, 13-24, 25-48 and 49-72 months. The first two years
# get their own groups because that is where churn behavior changes fastest.
model_df["TenureGroup"] = pd.cut(
    model_df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12m", "13-24m", "25-48m", "49-72m"],
).astype(str)

print("New engineered features (first 5 rows):")
print(model_df[["tenure", "TenureGroup", "NumServices",
                "MonthlyCharges", "AvgMonthlySpend"]].head())

# --- Technique 2: ENCODING ----------------------------------------------------
# a) Binary (two-value) columns -> 0/1 with a simple map.
#    This includes the target, Churn (Yes = 1 = customer left).
binary_cols = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
               "MultipleLines", "PaperlessBilling", "Churn"] + internet_addons
yes_no = {"Yes": 1, "No": 0}
for col in binary_cols:
    if col == "gender":
        # Which gender gets 1 is arbitrary; there's no order or ranking
        # implied. It's just a two-value flag.
        model_df[col] = model_df[col].map({"Male": 1, "Female": 0})
    else:
        model_df[col] = model_df[col].map(yes_no)

# b) Multi-category nominal columns -> one-hot encoding (dummy variables).

#    DECISION: one-hot rather than label encoding. These categories have no
#    natural order ("Electronic check" is not "greater than" "Mailed check"),
#    so label encoding them as 0/1/2/3 would invent a false ranking.

#    DECISION: drop_first=True removes one dummy per column to avoid perfect
#    multicollinearity, which matters for Logistic Regression.
multi_cols = ["InternetService", "Contract", "PaymentMethod", "TenureGroup"]
model_df = pd.get_dummies(model_df, columns=multi_cols, drop_first=True, dtype=int)

print(f"\nShape after encoding: {model_df.shape}")
print("Encoded columns:", list(model_df.columns))
assert model_df.isnull().sum().sum() == 0, "Encoding produced missing values"

# --- Train/test split (before scaling, to prevent data leakage) ---------------
# DECISION: split before scaling. The scaler must learn its mean/std from the
# training data only. If it were fit on the full dataset, information from the
# test set would leak into training and make model scores look better than
# they really are.

# DECISION: 80/20 split with stratify=y, which keeps the same churn ratio
# (~26.5%) in both sets. This matters because the classes are imbalanced.
X = model_df.drop(columns=["Churn"])
y = model_df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"\nTrain: {X_train.shape}, churn rate {y_train.mean():.1%}")
print(f"Test:  {X_test.shape}, churn rate {y_test.mean():.1%}")

# --- Technique 3: SCALING -----------------------------------------------------
# The continuous columns are on very different scales (tenure 0-72, monthly
# charges ~18-119, total charges 0-8,685). StandardScaler converts each to
# mean 0 / std 1 so that no feature dominates just because its numbers are
# bigger. This matters for distance- and gradient-based models (Logistic
# Regression, KNN, SVM); tree models like Random Forest are unaffected but not
# harmed.

# DECISION: scale only the continuous columns and leave the 0/1 encoded
# columns as they are, so they stay easy to interpret.
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges",
                "NumServices", "AvgMonthlySpend"]
scaler = StandardScaler()
X_train = X_train.copy()
X_test = X_test.copy()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])   # fit on train only
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])         # reuse train stats

print("\nScaled numeric features (train set) - mean ~0, std ~1:")
print(X_train[numeric_cols].describe().loc[["mean", "std", "min", "max"]].round(2))

# --- Save the model-ready data ------------------------------------------------
train_out = X_train.assign(Churn=y_train)
test_out = X_test.assign(Churn=y_test)
train_out.to_csv(OUT_DIR / "telco_churn_train.csv", index=False)
test_out.to_csv(OUT_DIR / "telco_churn_test.csv", index=False)

section("SUMMARY")
print(f"Raw data:          {RAW_FILE.name} (7,043 x 21)")
print(f"Cleaned data:      telco_churn_clean.csv {df.shape}")
print(f"Model-ready train: telco_churn_train.csv {train_out.shape}")
print(f"Model-ready test:  telco_churn_test.csv  {test_out.shape}")
print("Cleaning:      11 blank TotalCharges imputed as 0 (tenure-0 customers), "
      "0 duplicates, 0 invalid ranges, redundant categories merged")
print("Preprocessing: feature engineering (3 features), binary + one-hot "
      "encoding, standard scaling (fit on train only)")
