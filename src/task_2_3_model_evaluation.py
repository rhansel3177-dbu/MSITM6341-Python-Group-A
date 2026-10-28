"""
Midterm Project - Task 2.3: Build and Evaluate a Machine Learning Model
=======================================================================
Dataset : IBM Telco Customer Churn (already cleaned and preprocessed in Task 2.1)
Goal    : Predict whether a customer will churn (binary classification,
          target = Churn, 1 = customer left, 0 = stayed) and compare two
          algorithms: Logistic Regression and Random Forest.

Inputs (written by src/task_2_1_clean_preprocess.py to data/processed/):
    telco_churn_train.csv      -> encoded + scaled training set (80%)
    telco_churn_test.csv       -> encoded + scaled test set     (20%)

Outputs:
    outputs/figures/*.png         -> confusion matrices, ROC curves, threshold
                                     analysis, feature importance
    outputs/model_comparison.csv  -> all metrics for every model

Run (from the repo root):  python src/task_2_3_model_evaluation.py
Requires: pandas, numpy, matplotlib, seaborn, scikit-learn

Structure: three small classes do the work (load -> train -> evaluate), and
main() at the bottom runs them in order. Read main() first for the overview.
A fourth class, ChurnPredictor, scores one new customer; it is used by the
Streamlit app (app.py) and not by main().
"""

import contextlib
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                  # no window needed; figures are only saved to PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GridSearchCV

# Task 2.1 functions are reused by ChurnPredictor so a new customer goes through
# exactly the same cleaning, encoding, and scaling as the training data.
import task_2_1_clean_preprocess as preprocessing

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
RANDOM_STATE = 42                      # same seed as Task 2.1, so results are reproducible
ROOT_DIR = Path(__file__).resolve().parent.parent       # repo root
DATA_DIR = ROOT_DIR / "data" / "processed"
TRAIN_FILE = DATA_DIR / "telco_churn_train.csv"
TEST_FILE = DATA_DIR / "telco_churn_test.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
COMPARISON_FILE = OUTPUT_DIR / "model_comparison.csv"

TARGET = "Churn"
CLASS_LABELS = ["Stayed", "Churned"]   # 0 -> Stayed, 1 -> Churned
CV_FOLDS = 5
# DECISION: tune for F1 instead of accuracy. A model can reach ~73% accuracy by
# predicting "stays" for everyone, so accuracy would reward ignoring churners.
# F1 balances recall (catching churners) and precision (not crying wolf).
TUNING_SCORING = "f1"
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)  # 0.05, 0.10, ..., 0.95
DEFAULT_THRESHOLD = 0.5

# Chart colors: one fixed color per model, used in every figure
LR_COLOR = "#2a78d6"                   # blue   -> Logistic Regression
RF_COLOR = "#eb6834"                   # orange -> Random Forest
THIRD_COLOR = "#1baf7a"                # aqua   -> third line in the threshold chart
GRAY = "#8a8984"

# Example customers for ChurnPredictor, in the cleaned (human-readable) format
# of Task 2.1. TotalCharges = tenure x MonthlyCharges.
# High risk: brand new, month-to-month, fiber optic, electronic check, no
# security or support -> the strongest churn drivers found by both models.
HIGH_RISK_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": "Yes", "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 190.0,
}
# Low risk: long-time customer on a two-year contract with automatic payment.
LOW_RISK_CUSTOMER = {
    "gender": "Male", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "Yes",
    "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 65.0,
    "TotalCharges": 3900.0,
}


def section(title):
    """Print a visible header so the console output is easy to follow."""
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def run_quietly(func, *args):
    """Call func(*args) without printing anything to the console.

    The Task 2.1 functions print progress for the command-line pipeline; the
    app and ChurnPredictor call them many times, so their output is hidden.

    Parameters:
        func (callable): function to call
        *args: arguments passed to func
    Returns:
        whatever func returns
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args)


# ==============================================================================
# 1. LOAD THE DATA
# ==============================================================================

class ChurnDataLoader:
    """Load the model-ready train and test sets created in Task 2.1.

    The data is NOT re-split or re-scaled here. Task 2.1 already did a
    stratified 80/20 split and fit the scaler on the training set only, so
    reusing its files keeps the test set truly unseen (no data leakage).
    """

    def __init__(self, train_path=TRAIN_FILE, test_path=TEST_FILE):
        """
        Parameters:
            train_path (Path): location of telco_churn_train.csv
            test_path (Path):  location of telco_churn_test.csv
        """
        self.train_path = train_path
        self.test_path = test_path

    def _split_features_target(self, df):
        """Split one DataFrame into features (X) and target (y).

        Parameters:
            df (DataFrame): a processed dataset that includes the Churn column
        Returns:
            X (DataFrame), y (Series)
        """
        X = df.drop(columns=[TARGET])
        y = df[TARGET]
        return X, y

    def load(self):
        """Read both CSVs and return X_train, X_test, y_train, y_test.

        Returns:
            X_train (DataFrame), X_test (DataFrame), y_train (Series), y_test (Series)
        Exits the script with a clear message if a file is missing.
        """
        for path in [self.train_path, self.test_path]:
            if not path.exists():
                print(f"ERROR: processed data not found: {path}")
                print("Run the Task 2.1 script first (from the repo root):")
                print("    python src/task_2_1_clean_preprocess.py")
                raise SystemExit(1)

        X_train, y_train = self._split_features_target(pd.read_csv(self.train_path))
        X_test, y_test = self._split_features_target(pd.read_csv(self.test_path))

        print(f"Train set: {X_train.shape[0]} customers x {X_train.shape[1]} features")
        print(f"Test set:  {X_test.shape[0]} customers x {X_test.shape[1]} features")
        print(f"Churn rate - train: {y_train.mean():.1%}, test: {y_test.mean():.1%} "
              "(similar, because Task 2.1 used a stratified split)")
        return X_train, X_test, y_train, y_test


# ==============================================================================
# 2. TRAIN AND TUNE THE MODELS
# ==============================================================================

class ChurnModelTrainer:
    """Build, train, and tune Logistic Regression and Random Forest models.

    DECISION: class_weight="balanced" for both models because only ~26.5% of
    customers churn. Without it, the models learn that "stays" is almost
    always the safe guess and miss many churners. "balanced" makes each
    mistake on a churner count about 2.8x more than a mistake on a stayer.
    """

    def __init__(self, random_state=RANDOM_STATE):
        """
        Parameters:
            random_state (int): seed used by every model and the grid search
        """
        self.random_state = random_state
        self.best_params = {}          # filled in by tune_models()

    def build_logistic_regression(self, C=1.0):
        """Create an untrained Logistic Regression model.

        Parameters:
            C (float): inverse regularization strength (smaller = simpler model)
        Returns:
            LogisticRegression
        """
        # max_iter=1000 so the solver always has enough steps to converge.
        return LogisticRegression(C=C, class_weight="balanced", max_iter=1000,
                                  random_state=self.random_state)

    def build_random_forest(self, n_estimators=100, max_depth=None):
        """Create an untrained Random Forest model.

        Parameters:
            n_estimators (int): number of trees
            max_depth (int or None): maximum depth of each tree (None = no limit)
        Returns:
            RandomForestClassifier
        """
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                      class_weight="balanced", n_jobs=-1,
                                      random_state=self.random_state)

    def train_majority_baseline(self, X_train, y_train):
        """Train a "predict everyone stays" model to compare against.

        It shows how high accuracy can be with zero useful predictions, which
        is why accuracy alone is misleading on imbalanced data.

        Parameters:
            X_train (DataFrame), y_train (Series): training data
        Returns:
            DummyClassifier (always predicts the most common class, 0 = Stayed)
        """
        return DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

    def train_baselines(self, X_train, y_train):
        """Train both models with their default settings (no tuning).

        Parameters:
            X_train (DataFrame), y_train (Series): training data
        Returns:
            dict: {"Logistic Regression": model, "Random Forest": model}
        """
        lr = self.build_logistic_regression().fit(X_train, y_train)
        rf = self.build_random_forest().fit(X_train, y_train)
        print("Trained baseline Logistic Regression (C=1.0)")
        print("Trained baseline Random Forest (100 trees, no depth limit)")
        return {"Logistic Regression": lr, "Random Forest": rf}

    def _grid_search(self, model, param_grid, X_train, y_train):
        """Run a 5-fold GridSearchCV on the training set and return the best model.

        DECISION: the search only sees the training set. The test set is kept
        aside for the final evaluation, so its scores stay an honest estimate
        of how the model does on new customers.

        Parameters:
            model: untrained scikit-learn estimator
            param_grid (dict): hyperparameter values to try
            X_train (DataFrame), y_train (Series): training data
        Returns:
            GridSearchCV (fitted; .best_estimator_ is refit on all training data)
        """
        search = GridSearchCV(model, param_grid, scoring=TUNING_SCORING,
                              cv=CV_FOLDS, n_jobs=-1)
        search.fit(X_train, y_train)
        return search

    def tune_models(self, X_train, y_train):
        """Tune C for Logistic Regression and n_estimators/max_depth for Random Forest.

        Parameters:
            X_train (DataFrame), y_train (Series): training data
        Returns:
            dict: {"Logistic Regression": best model, "Random Forest": best model}
        """
        # DECISION: C on a log scale from strong (0.01) to weak (100) regularization.
        lr_grid = {"C": [0.01, 0.1, 1, 10, 100]}
        # DECISION: include shallow trees. A fully grown tree memorizes the
        # training data, so limiting depth is the main way to reduce overfitting.
        rf_grid = {"n_estimators": [100, 200, 300],
                   "max_depth": [5, 10, 15, None]}

        tuned = {}
        for name, model, grid in [
            ("Logistic Regression", self.build_logistic_regression(), lr_grid),
            ("Random Forest", self.build_random_forest(), rf_grid),
        ]:
            n_combos = int(np.prod([len(values) for values in grid.values()]))
            print(f"\n{name}: trying {n_combos} combinations x {CV_FOLDS} folds ...")
            search = self._grid_search(model, grid, X_train, y_train)
            self.best_params[name] = search.best_params_
            print(f"  Best parameters: {search.best_params_}")
            print(f"  Best cross-validated {TUNING_SCORING.upper()}: {search.best_score_:.3f}")
            tuned[name] = search.best_estimator_
        return tuned


# ==============================================================================
# 3. EVALUATE THE MODELS
# ==============================================================================

class ChurnModelEvaluator:
    """Compute test-set metrics and save the evaluation charts."""

    def __init__(self, X_test, y_test, figure_dir=FIGURE_DIR):
        """
        Parameters:
            X_test (DataFrame), y_test (Series): held-out test data
            figure_dir (Path): folder where PNG files are saved
        """
        self.X_test = X_test
        self.y_test = y_test
        self.figure_dir = figure_dir
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, filename):
        """Save a figure as PNG and close it (frees memory, no display needed).

        Parameters:
            fig (Figure): the matplotlib figure
            filename (str): file name inside the figure folder
        """
        path = self.figure_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path.relative_to(ROOT_DIR)}")

    def churn_probabilities(self, model):
        """Return the predicted probability of churn (class 1) for each test customer.

        Parameters:
            model: fitted classifier with predict_proba
        Returns:
            numpy array of probabilities between 0 and 1
        """
        return model.predict_proba(self.X_test)[:, 1]

    def compute_metrics(self, model):
        """Compute accuracy, precision, recall, F1, and ROC AUC on the test set.

        Parameters:
            model: fitted classifier
        Returns:
            dict: metric name -> value
        """
        y_pred = model.predict(self.X_test)
        return {
            "Accuracy": accuracy_score(self.y_test, y_pred),
            # zero_division=0: the "everyone stays" model never predicts churn,
            # so its precision is undefined; we report it as 0 instead of a warning.
            "Precision": precision_score(self.y_test, y_pred, zero_division=0),
            "Recall": recall_score(self.y_test, y_pred),
            "F1": f1_score(self.y_test, y_pred),
            "ROC AUC": roc_auc_score(self.y_test, self.churn_probabilities(model)),
        }

    def print_classification_report(self, name, model):
        """Print scikit-learn's per-class precision/recall/F1 report.

        Parameters:
            name (str): model name for the heading
            model: fitted classifier
        """
        print(f"\n--- {name} ---")
        print(classification_report(self.y_test, model.predict(self.X_test),
                                    target_names=CLASS_LABELS, digits=3))

    def plot_confusion_matrix(self, name, model, filename):
        """Save a confusion matrix heatmap with "Stayed"/"Churned" labels.

        Parameters:
            name (str): model name for the title
            model: fitted classifier
            filename (str): PNG file name
        """
        cm = confusion_matrix(self.y_test, model.predict(self.X_test))
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                    linewidths=2, linecolor="white", annot_kws={"size": 14}, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {name}")
        self._save(fig, filename)

    def roc_points(self, model):
        """Return the points of the ROC curve and the AUC for one model.

        Parameters:
            model: fitted classifier
        Returns:
            fpr (array), tpr (array), auc (float)
        """
        probs = self.churn_probabilities(model)
        fpr, tpr, _ = roc_curve(self.y_test, probs)
        return fpr, tpr, roc_auc_score(self.y_test, probs)

    def metrics_at_threshold(self, model, threshold):
        """Compute metrics and the confusion matrix for one decision threshold.

        Parameters:
            model: fitted classifier
            threshold (float): predict churn if probability >= threshold
        Returns:
            dict: Accuracy, Precision, Recall, F1, and "Confusion matrix"
                  (2x2 array: rows = actual, columns = predicted, Stayed first)
        """
        y_pred = (self.churn_probabilities(model) >= threshold).astype(int)
        return {
            "Accuracy": accuracy_score(self.y_test, y_pred),
            # At very high thresholds the model may flag nobody; precision is then
            # reported as 0 (zero_division=0) instead of raising a warning.
            "Precision": precision_score(self.y_test, y_pred, zero_division=0),
            "Recall": recall_score(self.y_test, y_pred),
            "F1": f1_score(self.y_test, y_pred, zero_division=0),
            "Confusion matrix": confusion_matrix(self.y_test, y_pred, labels=[0, 1]),
        }

    def plot_roc_curves(self, models, filename="roc_curves.png"):
        """Plot ROC curves for several models on one chart, with AUC in the legend.

        Parameters:
            models (dict): name -> fitted classifier
            filename (str): PNG file name
        """
        colors = [LR_COLOR, RF_COLOR]
        fig, ax = plt.subplots(figsize=(6, 5))
        for (name, model), color in zip(models.items(), colors):
            fpr, tpr, auc = self.roc_points(model)
            ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1,
                label="Random guessing (AUC = 0.500)")
        ax.set_xlabel("False positive rate (stayers wrongly flagged)")
        ax.set_ylabel("True positive rate (churners caught)")
        ax.set_title("ROC Curves - Tuned Models (Test Set)")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        self._save(fig, filename)

    def threshold_analysis(self, name, model, filename="threshold_analysis.png"):
        """Show how accuracy, precision, and recall change with the decision threshold.

        A model outputs a churn probability; the threshold decides when that
        probability counts as "will churn". scikit-learn uses 0.5 by default.

        DECISION / INTERPRETATION: a LOWER threshold flags more customers as
        churners. That catches more of the real churners (recall goes up) but
        also flags more loyal customers by mistake (false alarms, so precision
        goes down). A HIGHER threshold does the opposite. The business picks
        the threshold based on what costs more: losing a customer, or giving a
        retention offer to someone who would have stayed anyway.

        Parameters:
            name (str): model name for the title
            model: fitted classifier
            filename (str): PNG file name
        Returns:
            DataFrame: one row per threshold with accuracy, precision, recall
        """
        rows = []
        for threshold in THRESHOLDS:
            # Precision drops to 0 at the highest thresholds because the model
            # flags nobody there (see metrics_at_threshold).
            metrics = self.metrics_at_threshold(model, threshold)
            rows.append({"Threshold": threshold, "Accuracy": metrics["Accuracy"],
                         "Precision": metrics["Precision"], "Recall": metrics["Recall"]})
        results = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for metric, color in [("Accuracy", LR_COLOR), ("Precision", RF_COLOR),
                              ("Recall", THIRD_COLOR)]:
            ax.plot(results["Threshold"], results[metric], color=color, linewidth=2,
                    marker="o", markersize=4, label=metric)
        ax.axvline(DEFAULT_THRESHOLD, color=GRAY, linestyle="--", linewidth=1)
        ax.text(DEFAULT_THRESHOLD + 0.01, 0.03, "default 0.5", color="#52514e")
        ax.set_xlabel("Decision threshold (predict churn if probability >= threshold)")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"Threshold Analysis - {name}")
        ax.legend(loc="center right")
        ax.grid(alpha=0.3)
        self._save(fig, filename)
        return results

    def top_features(self, lr_model, rf_model, feature_names, top_n=10):
        """Return the top Logistic Regression coefficients and Random Forest importances.

        Parameters:
            lr_model (LogisticRegression), rf_model (RandomForestClassifier): fitted models
            feature_names (list): column names in the same order as the model input
            top_n (int): number of features to return per model
        Returns:
            (Series, Series): top LR coefficients (ranked by absolute value,
            sign kept) and top RF importances, both indexed by feature name
        """
        coefs = pd.Series(lr_model.coef_[0], index=feature_names)
        top_coefs = coefs.loc[coefs.abs().sort_values(ascending=False).index[:top_n]]
        importances = pd.Series(rf_model.feature_importances_, index=feature_names)
        top_importances = importances.sort_values(ascending=False).head(top_n)
        return top_coefs, top_importances

    def plot_feature_importance(self, lr_model, rf_model, feature_names,
                                filename="feature_importance.png", top_n=10):
        """Plot the top features for both models as horizontal bar charts.

        Logistic Regression: coefficients, ranked by absolute value. Because
        Task 2.1 scaled the numeric columns, coefficients are roughly
        comparable. Positive = raises churn risk, negative = lowers it.
        Random Forest: feature_importances_ (how much each feature reduces
        impurity across all trees). Always positive, so they show how much a
        feature matters but not in which direction.

        Parameters:
            lr_model (LogisticRegression), rf_model (RandomForestClassifier): fitted models
            feature_names (list): column names in the same order as the model input
            filename (str): PNG file name
            top_n (int): number of features to show per model
        Returns:
            (Series, Series): top LR coefficients and top RF importances
        """
        top_coefs, top_importances = self.top_features(lr_model, rf_model,
                                                       feature_names, top_n)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Largest at the top: barh draws from the bottom, so reverse the order.
        lr_plot = top_coefs.iloc[::-1]
        bar_colors = [RF_COLOR if value > 0 else LR_COLOR for value in lr_plot]
        ax1.barh(lr_plot.index, lr_plot.values, color=bar_colors)
        ax1.axvline(0, color=GRAY, linewidth=1)
        ax1.set_title(f"Logistic Regression - top {top_n} coefficients\n"
                      "(orange = raises churn risk, blue = lowers it)")
        ax1.set_xlabel("Coefficient (log-odds per unit / per std. dev.)")

        rf_plot = top_importances.iloc[::-1]
        ax2.barh(rf_plot.index, rf_plot.values, color=RF_COLOR)
        ax2.set_title(f"Random Forest - top {top_n} feature importances")
        ax2.set_xlabel("Importance (mean decrease in impurity)")

        for ax in (ax1, ax2):
            ax.grid(axis="x", alpha=0.3)
        self._save(fig, filename)
        return top_coefs, top_importances

    def compare_models(self, models):
        """Build one table with every metric for every model.

        Parameters:
            models (dict): name -> fitted classifier
        Returns:
            DataFrame: one row per model, one column per metric
        """
        table = pd.DataFrame({name: self.compute_metrics(model)
                              for name, model in models.items()}).T
        table.index.name = "Model"
        return table.round(3)


# ==============================================================================
# 4. PREDICT ONE NEW CUSTOMER (used by the Streamlit app)
# ==============================================================================

class ChurnPredictor:
    """Predict the churn probability of one new customer.

    The customer is entered in the cleaned, human-readable format of Task 2.1
    (for example Contract = "Two year", SeniorCitizen = "Yes"). It then goes
    through the same Task 2.1 functions as the training data: feature
    engineering, binary and one-hot encoding, and scaling with the training
    set's mean and standard deviation.
    """

    def __init__(self, models, raw_path=preprocessing.RAW_FILE):
        """
        Parameters:
            models (dict): name -> fitted classifier (e.g. the tuned models)
            raw_path (Path): location of the raw Kaggle CSV
        """
        self.models = models
        raw_df = run_quietly(preprocessing.load_data, raw_path)
        self.clean_df = run_quietly(preprocessing.clean_data, raw_df)
        # The form fields: every cleaned column except the ID and the target.
        self.input_columns = [col for col in self.clean_df.columns
                              if col not in ("customerID", TARGET)]
        self.X_train_unscaled, self.categories = self._rebuild_training_features()
        self.feature_columns = list(self.X_train_unscaled.columns)

    def _rebuild_training_features(self):
        """Rebuild Task 2.1's training features as they were just before scaling.

        DECISION: Task 2.1 does not save its fitted scaler. Instead of copying
        its code, we call the same Task 2.1 functions with the same
        random_state split, so preprocessing.scale_features later learns
        exactly the same mean/std as the saved training file.

        Returns:
            X_train (DataFrame): unscaled training features
            categories (dict): column -> sorted list of categories, for every
                               one-hot encoded column
        """
        df = run_quietly(preprocessing.engineer_features,
                         self.clean_df.drop(columns=["customerID"]))
        categories = {col: sorted(df[col].unique()) for col in preprocessing.MULTI_COLS}
        df = run_quietly(preprocessing.encode_binary, df)
        df = run_quietly(preprocessing.encode_one_hot, df)
        X_train, _, _, _ = run_quietly(preprocessing.split_data, df)
        return X_train, categories

    def to_model_input(self, customer):
        """Turn one customer dict into a one-row, model-ready DataFrame.

        Parameters:
            customer (dict): one value per column in self.input_columns
        Returns:
            DataFrame: one row with the same columns, order, and scaling as
                       telco_churn_train.csv (without Churn)
        """
        row = pd.DataFrame([customer])
        # encode_binary also maps the target, so a placeholder Churn value is
        # added here; it is removed by the reindex below and never used.
        row[TARGET] = "No"
        row = run_quietly(preprocessing.engineer_features, row)
        # DECISION: one row holds only one category per column, so get_dummies
        # (with drop_first=True) would not know the other categories and could
        # drop the wrong one. Giving it the full category list from training
        # makes it create exactly the training dummy columns.
        for col, values in self.categories.items():
            row[col] = pd.Categorical(row[col], categories=values)
        row = run_quietly(preprocessing.encode_binary, row)
        row = run_quietly(preprocessing.encode_one_hot, row)
        row = row.reindex(columns=self.feature_columns, fill_value=0)
        _, row = run_quietly(preprocessing.scale_features, self.X_train_unscaled, row)
        return row

    def predict_customer(self, customer):
        """Predict churn for one customer with every model.

        Parameters:
            customer (dict): one value per column in self.input_columns
        Returns:
            DataFrame: one row per model with "Stay probability",
                       "Churn probability", and "Prediction" (at threshold 0.5)
        """
        X = self.to_model_input(customer)
        rows = []
        for name, model in self.models.items():
            churn_prob = float(model.predict_proba(X)[0, 1])
            rows.append({
                "Model": name,
                "Stay probability": 1 - churn_prob,
                "Churn probability": churn_prob,
                "Prediction": CLASS_LABELS[int(churn_prob >= DEFAULT_THRESHOLD)],
            })
        return pd.DataFrame(rows)

    def generate_random_customer(self, random_state=None):
        """Pick a random real customer from the cleaned data.

        DECISION: sample a real customer instead of drawing each field at
        random, so the combination always makes sense (for example, no
        streaming add-ons for a customer without internet).

        Parameters:
            random_state (int or None): seed, or None for a new customer each time
        Returns:
            dict: one value per column in self.input_columns (plain Python types)
        """
        row = self.clean_df.sample(1, random_state=random_state).iloc[0]
        customer = {}
        for col in self.input_columns:
            value = row[col]
            # Convert numpy numbers to plain int/float so Streamlit widgets accept them.
            customer[col] = value.item() if hasattr(value, "item") else value
        return customer


# ==============================================================================
# MAIN
# ==============================================================================

def train_all_models(X_train, y_train):
    """Train the "everyone stays" baseline, both baselines, and both tuned models.

    Parameters:
        X_train (DataFrame), y_train (Series): training data
    Returns:
        all_models (dict): display name -> fitted model, in table order
        trainer (ChurnModelTrainer): holds the best parameters found
    """
    trainer = ChurnModelTrainer()
    majority = trainer.train_majority_baseline(X_train, y_train)
    baselines = trainer.train_baselines(X_train, y_train)
    tuned = trainer.tune_models(X_train, y_train)
    all_models = {
        "Baseline: predict everyone stays": majority,
        "Baseline Logistic Regression": baselines["Logistic Regression"],
        "Baseline Random Forest": baselines["Random Forest"],
        "Tuned Logistic Regression": tuned["Logistic Regression"],
        "Tuned Random Forest": tuned["Random Forest"],
    }
    return all_models, trainer


def pick_best_model(comparison):
    """Pick the best model by the average of recall and F1 (not accuracy).

    DECISION: for churn, missing a customer who leaves (false negative) costs
    more than a retention offer sent to a customer who would stay (false
    positive). Recall measures how many churners we catch. F1 is added so a
    model cannot win just by flagging everyone (that gives recall 1.0 but
    very poor precision). Accuracy is ignored because "everyone stays"
    already scores ~73.5%.

    DECISION: only the tuned models are candidates. The baselines are there
    to show what tuning changed, not to be deployed.

    Parameters:
        comparison (DataFrame): output of ChurnModelEvaluator.compare_models
    Returns:
        str: name of the best model
    """
    candidates = comparison.loc[["Tuned Logistic Regression", "Tuned Random Forest"]]
    score = (candidates["Recall"] + candidates["F1"]) / 2
    return score.idxmax()


def print_summary(comparison, best_name, y_test, threshold_results):
    """Print a short plain-English summary of the results.

    Parameters:
        comparison (DataFrame): metrics table
        best_name (str): name of the chosen model
        y_test (Series): test target, used to count churners
        threshold_results (DataFrame): output of threshold_analysis
    """
    section("SUMMARY (plain English)")
    n_churners = int(y_test.sum())
    lr = comparison.loc["Tuned Logistic Regression"]
    rf = comparison.loc["Tuned Random Forest"]
    dummy = comparison.loc["Baseline: predict everyone stays"]
    best = comparison.loc[best_name]

    print(f"The test set has {len(y_test)} customers, {n_churners} of whom churned.")
    print(f"- Tuned Logistic Regression found {lr['Recall']:.0%} of churners "
          f"(about {round(lr['Recall'] * n_churners)} of {n_churners}). "
          f"When it flagged someone, it was right {lr['Precision']:.0%} of the time.")
    print(f"- Tuned Random Forest found {rf['Recall']:.0%} of churners "
          f"(about {round(rf['Recall'] * n_churners)} of {n_churners}). "
          f"When it flagged someone, it was right {rf['Precision']:.0%} of the time.")
    print(f"- A model that predicts 'everyone stays' is {dummy['Accuracy']:.0%} accurate "
          "but finds 0% of churners. That is why accuracy alone is misleading here.")
    print(f"\nBest model: {best_name} (recall {best['Recall']:.3f}, "
          f"F1 {best['F1']:.3f}, ROC AUC {best['ROC AUC']:.3f}).")
    print("It was chosen on recall and F1, not accuracy: missing a customer who is "
          "about to leave costs more than a retention offer to someone who would "
          "have stayed, and F1 makes sure the model is not just flagging everyone.")

    low = threshold_results.loc[np.isclose(threshold_results["Threshold"], 0.3)].iloc[0]
    default = threshold_results.loc[
        np.isclose(threshold_results["Threshold"], DEFAULT_THRESHOLD)].iloc[0]
    print(f"\nThreshold trade-off for the best model: at the default 0.5 it catches "
          f"{default['Recall']:.0%} of churners with {default['Precision']:.0%} precision; "
          f"lowering the threshold to 0.3 catches {low['Recall']:.0%} "
          f"but precision drops to {low['Precision']:.0%} (more false alarms).")


def main():
    """Run the full Task 2.3 pipeline: load -> train -> tune -> evaluate -> save."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    section("1. LOAD DATA (from Task 2.1)")
    X_train, X_test, y_train, y_test = ChurnDataLoader().load()

    section("2-3. TRAIN BASELINES AND TUNE HYPERPARAMETERS "
            f"(GridSearchCV, {CV_FOLDS}-fold, scoring={TUNING_SCORING})")
    all_models, _ = train_all_models(X_train, y_train)
    tuned = {"Logistic Regression": all_models["Tuned Logistic Regression"],
             "Random Forest": all_models["Tuned Random Forest"]}
    evaluator = ChurnModelEvaluator(X_test, y_test)

    section("4. CLASSIFICATION REPORTS AND CONFUSION MATRICES (test set)")
    for name, model in list(all_models.items())[1:]:          # skip the dummy model
        evaluator.print_classification_report(name, model)
        filename = "confusion_matrix_" + name.lower().replace(" ", "_") + ".png"
        evaluator.plot_confusion_matrix(name, model, filename)

    section("5. ROC CURVES (tuned models)")
    evaluator.plot_roc_curves({"Logistic Regression": tuned["Logistic Regression"],
                               "Random Forest": tuned["Random Forest"]})

    section("6. MODEL COMPARISON TABLE (test set)")
    comparison = evaluator.compare_models(all_models)
    print(comparison.to_string())
    comparison.to_csv(COMPARISON_FILE)
    print(f"\nSaved {COMPARISON_FILE.relative_to(ROOT_DIR)}")
    best_name = pick_best_model(comparison)

    section(f"7. THRESHOLD ANALYSIS ({best_name})")
    threshold_results = evaluator.threshold_analysis(best_name, all_models[best_name])
    print(threshold_results.round(3).to_string(index=False))

    section("8. FEATURE IMPORTANCE (tuned models)")
    top_coefs, top_importances = evaluator.plot_feature_importance(
        tuned["Logistic Regression"], tuned["Random Forest"], list(X_train.columns))
    print("Logistic Regression - top coefficients (+ raises churn risk, - lowers it):")
    print(top_coefs.round(3).to_string())
    print("\nRandom Forest - top feature importances:")
    print(top_importances.round(3).to_string())

    print_summary(comparison, best_name, y_test, threshold_results)


if __name__ == "__main__":
    main()
