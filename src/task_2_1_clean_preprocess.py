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

Structure: each cleaning and preprocessing step is its own function, and
main() at the bottom runs them in order. Read main() first for the overview.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
RANDOM_STATE = 42                      # fixed seed so results are reproducible
ROOT_DIR = Path(__file__).resolve().parent.parent       # repo root
RAW_FILE = ROOT_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT_DIR = ROOT_DIR / "data" / "processed"               # generated files (git-ignored)
CLEAN_FILE = OUT_DIR / "telco_churn_clean.csv"
TRAIN_FILE = OUT_DIR / "telco_churn_train.csv"
TEST_FILE = OUT_DIR / "telco_churn_test.csv"

# Column groups used by more than one step
INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]
SERVICE_COLS = ["PhoneService", "MultipleLines"] + INTERNET_ADDONS
BINARY_COLS = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
               "MultipleLines", "PaperlessBilling", "Churn"] + INTERNET_ADDONS
MULTI_COLS = ["InternetService", "Contract", "PaymentMethod", "TenureGroup"]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges",
                "NumServices", "AvgMonthlySpend"]
TARGET = "Churn"


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

def load_data(path):
    """Load the raw CSV and print a first look at its shape and column types."""
    # Fail early with a clear message if the CSV isn't in the data/ folder.
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found: {path}\n"
            "Download it from Kaggle (link above) and place it in the data/ folder."
        )

    df = pd.read_csv(path)
    print(f"Loaded {path.name}: {df.shape[0]} rows x {df.shape[1]} columns")
    print(df.head())
    print("\nColumn types as loaded:")
    print(df.dtypes)
    return df


# ==============================================================================
# (b) DATA CLEANING
# ==============================================================================

def strip_whitespace(df):
    """b1. Strip leading/trailing spaces from every text column.

    This is a defensive step: leading or trailing spaces would make "Yes" and
    "Yes " count as different categories.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()
    return df


def fix_total_charges_type(df):
    """b2. Convert TotalCharges from text to numeric.

    TotalCharges should be numeric but pandas read it as text, which means some
    values are not numbers. Converting with errors="coerce" turns those invalid
    entries into NaN so we can find them.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def handle_missing_values(df):
    """b3. Report missing values and impute blank TotalCharges with 0.

    DECISION: impute TotalCharges = 0 rather than drop or use the mean/median.
    Every one of these customers has tenure == 0: they just signed up and have
    not been billed yet, so their true total really is $0. Imputing 0 records
    that true value and keeps all 7,043 customers. Dropping 11 rows (0.16%)
    would also be acceptable but discards valid customers, and mean/median
    imputation would wrongly give brand-new customers thousands of dollars in
    charges.
    """
    print("Missing values per column (after type fix):")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    blank_rows = df[df["TotalCharges"].isnull()]
    print(f"\nRows with missing TotalCharges: {len(blank_rows)}")
    print(blank_rows[["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]])

    assert (blank_rows["tenure"] == 0).all(), "Unexpected: blank TotalCharges with tenure > 0"
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    print(f"Missing values remaining: {df.isnull().sum().sum()}")
    return df


def remove_duplicates(df):
    """b4. Report and remove duplicate rows and duplicate customer IDs.

    Each customer should appear exactly once.
    DECISION: keep the de-duplication step even though none were found, so the
    pipeline stays safe if the data is refreshed later.
    """
    print(f"\nFully duplicated rows: {df.duplicated().sum()}")
    print(f"Duplicated customerIDs: {df['customerID'].duplicated().sum()}")
    return df.drop_duplicates(subset="customerID", keep="first")


def check_numeric_ranges(df):
    """b5a. Check for negative values and tenure outside the 0-72 month range."""
    print("\nNumeric range checks:")
    print(f"  Negative tenure:          {(df['tenure'] < 0).sum()}")
    print(f"  Tenure above 72 months:   {(df['tenure'] > 72).sum()}")
    print(f"  Negative MonthlyCharges:  {(df['MonthlyCharges'] < 0).sum()}")
    print(f"  Negative TotalCharges:    {(df['TotalCharges'] < 0).sum()}")


def check_charges_consistency(df):
    """b5b. Flag customers whose TotalCharges doesn't match tenure x MonthlyCharges.

    Prices change over time, so we only flag customers whose total is wildly
    off (outside 50%-150% of the expected value).
    DECISION: keep the flagged rows. Both are short-tenure customers whose plan
    price likely changed after their first bill (an upgrade or downgrade),
    which is a plausible real-world pattern, not a data-entry error.
    """
    expected_total = df["tenure"] * df["MonthlyCharges"]
    ratio = np.where(expected_total > 0, df["TotalCharges"] / expected_total, 1.0)
    flagged = (ratio < 0.5) | (ratio > 1.5)
    print(f"  TotalCharges inconsistent with tenure x MonthlyCharges: {flagged.sum()}")
    print(df.loc[flagged, ["customerID", "tenure", "MonthlyCharges", "TotalCharges"]])


def check_category_values(df):
    """b5c. Print the values of every category column to spot typos.

    Text columns are selected here, after the type fix, so TotalCharges is no
    longer included.
    """
    category_cols = df.select_dtypes(include=["object", "string"]).columns.drop("customerID")
    print("\nCategory values per column:")
    for col in category_cols:
        print(f"  {col:<17} {sorted(df[col].unique())}")


def standardize_categories(df):
    """b6. Make category labels consistent and remove redundant values.

    DECISION: recode SeniorCitizen from 0/1 to "Yes"/"No". Every other yes/no
    column uses text labels, so this keeps the cleaned EDA file consistent and
    makes charts readable. It is encoded back to 0/1 in preprocessing.

    DECISION: recode "No internet service" (six add-on columns) and "No phone
    service" (MultipleLines) to "No". That information is already captured by
    the InternetService and PhoneService columns, so keeping it would only
    create redundant dummy columns later. "No" is accurate: the customer does
    not have that add-on.
    """
    df = df.copy()
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    for col in INTERNET_ADDONS:
        df[col] = df[col].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    return df


def clean_data(df):
    """Run every cleaning step in order and return the cleaned DataFrame."""
    df = strip_whitespace(df)
    df = fix_total_charges_type(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    check_numeric_ranges(df)
    check_charges_consistency(df)
    check_category_values(df)
    df = standardize_categories(df)
    print(f"\nCleaned dataset shape: {df.shape}")
    return df


# ==============================================================================
# (c) DATA PREPROCESSING
# ==============================================================================
# Three techniques are used (the requirement is at least 2):
#   1. Feature engineering - create new, more informative columns
#   2. Encoding            - turn categorical text into numbers
#   3. Scaling             - put numeric features on a common scale

def engineer_features(df):
    """Technique 1: add NumServices, AvgMonthlySpend, and TenureGroup.

    NumServices: how many add-on services the customer pays for (0-8).
    Customers who are more "embedded" in the product may be less likely to
    leave.

    AvgMonthlySpend: TotalCharges / tenure, i.e. what they actually paid per
    month on average. Differs from MonthlyCharges when prices or plans have
    changed. For tenure 0, fall back to the current MonthlyCharges (avoids
    divide-by-zero).

    TenureGroup: bucket tenure into customer-lifecycle stages. Churn risk is
    usually highest in the first year, so a group label can capture that
    pattern more directly than a raw month count.
    DECISION: bins of 0-12, 13-24, 25-48 and 49-72 months. The first two years
    get their own groups because that is where churn behavior changes fastest.
    """
    df = df.copy()
    df["NumServices"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    df["AvgMonthlySpend"] = (
        df["TotalCharges"] / df["tenure"]
    ).where(df["tenure"] > 0, df["MonthlyCharges"])
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"],
    ).astype(str)

    print("New engineered features (first 5 rows):")
    print(df[["tenure", "TenureGroup", "NumServices",
              "MonthlyCharges", "AvgMonthlySpend"]].head())
    return df


def encode_binary(df):
    """Technique 2a: map two-value columns to 0/1.

    This includes the target, Churn (Yes = 1 = customer left).
    """
    df = df.copy()
    yes_no = {"Yes": 1, "No": 0}
    for col in BINARY_COLS:
        if col == "gender":
            # Which gender gets 1 is arbitrary; there's no order or ranking
            # implied. It's just a two-value flag.
            df[col] = df[col].map({"Male": 1, "Female": 0})
        else:
            df[col] = df[col].map(yes_no)
    return df


def encode_one_hot(df):
    """Technique 2b: one-hot encode multi-category nominal columns.

    DECISION: one-hot rather than label encoding. These categories have no
    natural order ("Electronic check" is not "greater than" "Mailed check"),
    so label encoding them as 0/1/2/3 would invent a false ranking.

    DECISION: drop_first=True removes one dummy per column to avoid perfect
    multicollinearity, which matters for Logistic Regression.
    """
    df = pd.get_dummies(df, columns=MULTI_COLS, drop_first=True, dtype=int)
    print(f"\nShape after encoding: {df.shape}")
    print("Encoded columns:", list(df.columns))
    assert df.isnull().sum().sum() == 0, "Encoding produced missing values"
    return df


def split_data(df):
    """Split into stratified 80/20 train and test sets.

    DECISION: split before scaling. The scaler must learn its mean/std from the
    training data only. If it were fit on the full dataset, information from
    the test set would leak into training and make model scores look better
    than they really are.

    DECISION: 80/20 split with stratify=y, which keeps the same churn ratio
    (~26.5%) in both sets. This matters because the classes are imbalanced.
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {X_train.shape}, churn rate {y_train.mean():.1%}")
    print(f"Test:  {X_test.shape}, churn rate {y_test.mean():.1%}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Technique 3: standardize continuous columns (fit on train only).

    The continuous columns are on very different scales (tenure 0-72, monthly
    charges ~18-119, total charges 0-8,685). StandardScaler converts each to
    mean 0 / std 1 so that no feature dominates just because its numbers are
    bigger. This matters for distance- and gradient-based models (Logistic
    Regression, KNN, SVM); tree models like Random Forest are unaffected but
    not harmed.

    DECISION: scale only the continuous columns and leave the 0/1 encoded
    columns as they are, so they stay easy to interpret.
    """
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])  # fit on train only
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])        # reuse train stats

    print("\nScaled numeric features (train set) - mean ~0, std ~1:")
    print(X_train[NUMERIC_COLS].describe().loc[["mean", "std", "min", "max"]].round(2))
    return X_train, X_test


def preprocess_data(clean_df):
    """Run every preprocessing step and return model-ready train/test sets.

    customerID is kept in the cleaned file for traceability, but it is a unique
    identifier with no predictive value, so it is dropped here before modeling.
    """
    df = clean_df.drop(columns=["customerID"])
    df = engineer_features(df)
    df = encode_binary(df)
    df = encode_one_hot(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test = scale_features(X_train, X_test)
    train_df = X_train.assign(**{TARGET: y_train})
    test_df = X_test.assign(**{TARGET: y_test})
    return train_df, test_df


# ==============================================================================
# SAVE AND SUMMARIZE
# ==============================================================================

def print_summary(clean_df, train_df, test_df):
    """Print a short recap of what the pipeline produced."""
    section("SUMMARY")
    print(f"Raw data:          {RAW_FILE.name} (7,043 x 21)")
    print(f"Cleaned data:      {CLEAN_FILE.name} {clean_df.shape}")
    print(f"Model-ready train: {TRAIN_FILE.name} {train_df.shape}")
    print(f"Model-ready test:  {TEST_FILE.name}  {test_df.shape}")
    print("Cleaning:      11 blank TotalCharges imputed as 0 (tenure-0 customers), "
          "0 duplicates, 0 invalid ranges, redundant categories merged")
    print("Preprocessing: feature engineering (3 features), binary + one-hot "
          "encoding, standard scaling (fit on train only)")


def main():
    """Run the full Task 2.1 pipeline: acquire -> clean -> preprocess -> save."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    section("(a) DATA ACQUISITION")
    raw_df = load_data(RAW_FILE)

    section("(b) DATA CLEANING")
    clean_df = clean_data(raw_df)
    clean_df.to_csv(CLEAN_FILE, index=False)
    print(f"Saved {CLEAN_FILE.name} (use this for EDA in Task 2.2)")

    section("(c) DATA PREPROCESSING")
    train_df, test_df = preprocess_data(clean_df)
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print_summary(clean_df, train_df, test_df)


if __name__ == "__main__":
    main()
