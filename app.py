"""
Telco Customer Churn - Midterm Project Presentation App (Streamlit)
===================================================================
A visual walk-through of the project in four pages:
    1. Dataset                      -> what the data looks like
    2. Cleaning and Preprocessing   -> Task 2.1 steps and their decisions
    3. Model Results                -> Task 2.3 metrics, ROC, threshold, features
    4. Predict a Customer           -> score one customer with both tuned models

The app does NOT re-implement any cleaning, encoding, training, or prediction.
Everything comes from the project scripts:
    src/task_2_1_clean_preprocess.py   (imported as `preprocessing`)
    src/task_2_3_model_evaluation.py   (imported as `modeling`)

Run (from the repo root):  streamlit run app.py
Requires: streamlit, plotly, pandas, numpy, scikit-learn (see requirements.txt)
"""

import inspect
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# The project scripts live in src/, so add it to the import path.
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))
import task_2_1_clean_preprocess as preprocessing  # noqa: E402
import task_2_3_model_evaluation as modeling  # noqa: E402

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
PAGES = ["1. Dataset", "2. Cleaning and Preprocessing",
         "3. Model Results", "4. Predict a Customer"]

# Theme colors (must match .streamlit/config.toml)
BACKGROUND = "#0A0A0A"
TEXT = "#E8F5E9"
GREEN = "#00E676"                      # main data color
DARK_GREEN = "#1B5E20"                 # accent
LIGHT_GREEN = "#69F0AE"                # accent
RED = "#FF5252"                        # DECISION: red is used ONLY for "Churned"
MUTED = "#7A8A7D"                      # reference lines, secondary text
OUTCOME_COLORS = {"Stayed": GREEN, "Churned": RED}

# The two models shown on the results and prediction pages
TUNED_MODELS = ["Tuned Logistic Regression", "Tuned Random Forest"]
DUMMY_MODEL = "Baseline: predict everyone stays"

# One-line description of every raw column. Edit freely; the rest of the
# variable table (type, range, categories) is computed from the data.
VARIABLE_DESCRIPTIONS = {
    "customerID": "Unique customer identifier (not used for modeling)",
    "gender": "Customer gender",
    "SeniorCitizen": "Whether the customer is 65 or older",
    "Partner": "Whether the customer has a partner",
    "Dependents": "Whether the customer has dependents (children, etc.)",
    "tenure": "Number of months the customer has been with the company",
    "PhoneService": "Has a phone line",
    "MultipleLines": "Has more than one phone line",
    "InternetService": "Type of internet connection",
    "OnlineSecurity": "Subscribes to the online security add-on",
    "OnlineBackup": "Subscribes to the online backup add-on",
    "DeviceProtection": "Subscribes to the device protection add-on",
    "TechSupport": "Subscribes to the premium tech support add-on",
    "StreamingTV": "Streams TV through the internet service",
    "StreamingMovies": "Streams movies through the internet service",
    "Contract": "Contract term",
    "PaperlessBilling": "Receives bills electronically",
    "PaymentMethod": "How the customer pays",
    "MonthlyCharges": "Current monthly bill in USD",
    "TotalCharges": "Total amount billed so far in USD",
    "Churn": "TARGET: whether the customer left in the last month",
}

# Readable labels for the prediction form
FIELD_LABELS = {
    "gender": "Gender", "SeniorCitizen": "Senior citizen (65+)", "Partner": "Has partner",
    "Dependents": "Has dependents", "tenure": "Tenure (months)",
    "PhoneService": "Phone service", "MultipleLines": "Multiple lines",
    "InternetService": "Internet service", "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
    "TechSupport": "Tech support", "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies", "Contract": "Contract",
    "PaperlessBilling": "Paperless billing", "PaymentMethod": "Payment method",
    "MonthlyCharges": "Monthly charges (USD)",
}


# ==============================================================================
# DATA AND MODELS (cached so they load / train only once)
# ==============================================================================

def ensure_processed_data():
    """Run the Task 2.1 pipeline if the processed train/test files are missing."""
    if modeling.TRAIN_FILE.exists() and modeling.TEST_FILE.exists():
        return
    with st.spinner("Processed data not found - running the Task 2.1 cleaning pipeline..."):
        modeling.run_quietly(preprocessing.main)


@st.cache_data(show_spinner=False)
def load_raw_data():
    """Load the raw Kaggle CSV with Task 2.1's load_data().

    Returns:
        DataFrame: 7,043 rows x 21 columns, exactly as downloaded
    """
    return modeling.run_quietly(preprocessing.load_data, preprocessing.RAW_FILE)


@st.cache_data(show_spinner=False)
def load_clean_data():
    """Clean the raw data with Task 2.1's clean_data().

    Returns:
        DataFrame: cleaned, human-readable data (same as telco_churn_clean.csv)
    """
    return modeling.run_quietly(preprocessing.clean_data, load_raw_data())


@st.cache_data(show_spinner=False)
def load_model_data():
    """Load the model-ready train/test sets with Task 2.3's ChurnDataLoader.

    Returns:
        X_train, X_test (DataFrame), y_train, y_test (Series)
    """
    return modeling.run_quietly(modeling.ChurnDataLoader().load)


@st.cache_resource(show_spinner="Training and tuning the models (about 30 seconds, only once)...")
def train_models():
    """Train all five models with Task 2.3's train_all_models().

    DECISION: st.cache_resource keeps the fitted models in memory, so the
    grid search runs once per app session instead of on every click.

    Returns:
        all_models (dict): display name -> fitted model
        best_params (dict): model name -> best hyperparameters from GridSearchCV
    """
    X_train, _, y_train, _ = load_model_data()
    all_models, trainer = modeling.run_quietly(modeling.train_all_models, X_train, y_train)
    return all_models, trainer.best_params


@st.cache_resource(show_spinner=False)
def get_evaluator():
    """Create the Task 2.3 evaluator for the test set.

    Returns:
        ChurnModelEvaluator
    """
    _, X_test, _, y_test = load_model_data()
    return modeling.ChurnModelEvaluator(X_test, y_test)


@st.cache_resource(show_spinner=False)
def get_comparison_table():
    """Compute the metrics table for all five models on the test set.

    Returns:
        DataFrame: one row per model, one column per metric
    """
    all_models, _ = train_models()
    return get_evaluator().compare_models(all_models)


@st.cache_resource(show_spinner="Preparing the predictor...")
def get_predictor():
    """Create a Task 2.3 ChurnPredictor that uses both tuned models.

    Returns:
        ChurnPredictor
    """
    all_models, _ = train_models()
    return modeling.ChurnPredictor({
        "Logistic Regression": all_models["Tuned Logistic Regression"],
        "Random Forest": all_models["Tuned Random Forest"],
    })


# ==============================================================================
# SMALL HELPERS
# ==============================================================================

def style_figure(fig, height=380):
    """Apply the black-and-green theme to a Plotly figure.

    Parameters:
        fig (Figure): Plotly figure
        height (int): figure height in pixels
    Returns:
        Figure: the same figure, styled
    """
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND,
        font_color=TEXT, colorway=[GREEN, LIGHT_GREEN, DARK_GREEN], height=height,
        # Extra top margin so the title and the legend row never overlap.
        margin=dict(l=10, r=10, t=90, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
    )
    return fig


def show_chart(fig):
    """Display a Plotly figure at full container width."""
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def with_outcome(df):
    """Add an "Outcome" column (Stayed / Churned) based on the Churn column.

    Parameters:
        df (DataFrame): data with Churn = "Yes" / "No"
    Returns:
        DataFrame: copy with the extra column
    """
    return df.assign(Outcome=df["Churn"].map({"No": "Stayed", "Yes": "Churned"}))


def docstring_paragraphs(func, prefix):
    """Return the paragraphs of a function's docstring that start with prefix.

    Used to show the DECISION notes written in the Task 2.1 code, so the app
    never goes out of sync with the code.

    Parameters:
        func (callable): function whose docstring is read
        prefix (str): e.g. "DECISION"
    Returns:
        list of str: matching paragraphs joined into single lines
    """
    paragraphs = (inspect.getdoc(func) or "").split("\n\n")
    return [" ".join(p.split()) for p in paragraphs if p.strip().startswith(prefix)]


def docstring_summary(func):
    """Return the first line of a function's docstring (its summary)."""
    return (inspect.getdoc(func) or "").splitlines()[0]


def confusion_matrix_figure(cm, title):
    """Draw a 2x2 confusion matrix as a green heatmap with Stayed/Churned labels.

    Parameters:
        cm (array): 2x2 confusion matrix, rows = actual, columns = predicted
        title (str): chart title
    Returns:
        Figure
    """
    labels = modeling.CLASS_LABELS
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels, colorscale=[[0, "#0F1F12"], [1, GREEN]],
        showscale=False, xgap=3, ygap=3, hovertemplate="Actual %{y}<br>Predicted %{x}"
                                                       "<br>%{z} customers<extra></extra>"))
    # Dark text on bright cells and light text on dark cells, so every number is readable.
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            value = int(cm[i][j])
            fig.add_annotation(x=predicted, y=actual, text=f"<b>{value:,}</b>",
                               showarrow=False, font=dict(size=22, color=BACKGROUND
                               if value > cm.max() / 2 else TEXT))
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Actual")
    fig.update_yaxes(autorange="reversed")
    return style_figure(fig, height=340)


# ==============================================================================
# PAGE 1: DATASET
# ==============================================================================

def build_variable_table(df):
    """Describe every column: name, type, range or categories, and description.

    Parameters:
        df (DataFrame): cleaned data
    Returns:
        DataFrame: one row per variable
    """
    rows = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_type = "Numeric"
            values = f"{df[col].min():,.2f} to {df[col].max():,.2f}"
        else:
            col_type = "Categorical"
            categories = sorted(df[col].unique())
            # IDs have thousands of values, so show a count instead of a list.
            values = (", ".join(categories) if len(categories) <= 6
                      else f"{len(categories):,} unique values")
        rows.append({"Variable": col, "Type": col_type, "Range / categories": values,
                     "Description": VARIABLE_DESCRIPTIONS.get(col, "")})
    return pd.DataFrame(rows)


def distribution_figure(df, column):
    """Plot one variable's distribution colored by churn outcome.

    Parameters:
        df (DataFrame): cleaned data with an Outcome column
        column (str): variable to plot
    Returns:
        Figure: histogram for numeric columns, grouped bar chart otherwise
    """
    if pd.api.types.is_numeric_dtype(df[column]):
        # Stacked (not overlaid) so red and green never blend into a third color.
        fig = px.histogram(df, x=column, color="Outcome", nbins=40, barmode="stack",
                           color_discrete_map=OUTCOME_COLORS,
                           category_orders={"Outcome": ["Stayed", "Churned"]})
        fig.update_yaxes(title="Customers")
    else:
        counts = df.groupby([column, "Outcome"]).size().reset_index(name="Customers")
        fig = px.bar(counts, x=column, y="Customers", color="Outcome", barmode="group",
                     color_discrete_map=OUTCOME_COLORS,
                     category_orders={"Outcome": ["Stayed", "Churned"]})
    fig.update_layout(title=f"Distribution of {column} by churn outcome", bargap=0.15)
    return style_figure(fig, height=420)


def render_eda_section(clean_df):
    """Placeholder for the Task 2.2 Exploratory Data Analysis.

    TODO (Task 2.2 owners): replace the st.info() below with your EDA charts
    and findings. You receive the cleaned, human-readable data; use
    with_outcome(clean_df) to get a Stayed/Churned column, and style_figure()
    + show_chart() for Plotly charts so they match the app's theme.

    Parameters:
        clean_df (DataFrame): cleaned data (same as telco_churn_clean.csv)
    """
    st.info("The exploratory data analysis and visualizations from Task 2.2 "
            "will be added here by our teammates.", icon=":material/construction:")


def page_dataset():
    """Page 1: the raw dataset, its variables, and their distributions."""
    raw_df = load_raw_data()
    clean_df = load_clean_data()

    st.title("Telco Customer Churn")
    st.markdown(
        "**Source:** [Kaggle - Telco Customer Churn (BlastChar)]"
        "(https://www.kaggle.com/datasets/blastchar/telco-customer-churn), "
        "an IBM sample dataset of a fictional telecom company. Each row is one "
        "customer; **Churn** says whether they left in the last month."
    )

    # Missing values only appear after TotalCharges is converted to a number
    # (the blanks are spaces in the CSV), so reuse Task 2.1's two steps.
    typed_df = preprocessing.fix_total_charges_type(preprocessing.strip_whitespace(raw_df))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers (rows)", f"{raw_df.shape[0]:,}")
    col2.metric("Columns", raw_df.shape[1])
    col3.metric("Churn rate", f"{(raw_df['Churn'] == 'Yes').mean():.1%}")
    col4.metric("Missing values found", int(typed_df.isna().sum().sum()))

    st.subheader("Variables")
    st.caption("Built from the cleaned data, so TotalCharges is numeric and "
               "SeniorCitizen reads Yes/No.")
    st.dataframe(build_variable_table(clean_df), hide_index=True, width="stretch")

    st.subheader("Explore a variable")
    options = [col for col in clean_df.columns if col != "customerID"]
    column = st.selectbox("Variable", options, index=options.index("tenure"))
    show_chart(distribution_figure(with_outcome(clean_df), column))

    st.subheader("Exploratory Data Analysis (Task 2.2)")
    render_eda_section(clean_df)


# ==============================================================================
# PAGE 2: CLEANING AND PREPROCESSING
# ==============================================================================

def show_step(title, func, metrics=None):
    """Show one Task 2.1 step: its summary, key numbers, and DECISION notes.

    Parameters:
        title (str): heading for the step
        func (callable): the Task 2.1 function (its docstring is displayed)
        metrics (list of (label, value)): numbers to show as metric cards
    """
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.caption(f"`{func.__name__}()` - {docstring_summary(func)}")
        if metrics:
            for column, (label, value) in zip(st.columns(len(metrics)), metrics):
                column.metric(label, value)
        for decision in docstring_paragraphs(func, "DECISION"):
            st.markdown(f"> **DECISION:** {decision.removeprefix('DECISION:').strip()}")


def page_cleaning():
    """Page 2: every Task 2.1 cleaning and preprocessing step with its decision."""
    raw_df = load_raw_data()
    clean_df = load_clean_data()
    X_train, X_test, _, _ = load_model_data()
    typed_df = preprocessing.fix_total_charges_type(preprocessing.strip_whitespace(raw_df))
    blank = typed_df[typed_df["TotalCharges"].isna()]

    st.title("Cleaning and Preprocessing")
    st.markdown("Every step below comes from `src/task_2_1_clean_preprocess.py`. "
                "The decision notes are read directly from the code's docstrings.")

    st.header("Before and after")
    col1, col2, col3 = st.columns(3)
    col1.metric("Raw data", f"{raw_df.shape[0]:,} x {raw_df.shape[1]}")
    col2.metric("Cleaned data", f"{clean_df.shape[0]:,} x {clean_df.shape[1]}")
    col3.metric("Model-ready (train + test)",
                f"{X_train.shape[0]:,} + {X_test.shape[0]:,} x {X_train.shape[1]}",
                help="Features only; the Churn target is a separate column.")

    st.header("Cleaning")
    show_step("1. Strip whitespace", preprocessing.strip_whitespace)
    show_step("2. Fix TotalCharges type and impute blanks",
              preprocessing.handle_missing_values,
              [("Blank TotalCharges", len(blank)),
               ("Of those with tenure 0", f"{(blank['tenure'] == 0).sum()} of {len(blank)}"),
               ("Imputed value", "$0")])
    show_step("3. Remove duplicates", preprocessing.remove_duplicates,
              [("Duplicate rows", int(typed_df.duplicated().sum())),
               ("Duplicate customer IDs", int(typed_df["customerID"].duplicated().sum()))])
    show_step("4. Check charges consistency", preprocessing.check_charges_consistency)
    no_internet = int((typed_df[preprocessing.INTERNET_ADDONS] == "No internet service")
                      .sum().sum())
    no_phone = int((typed_df["MultipleLines"] == "No phone service").sum())
    show_step("5. Standardize categories", preprocessing.standardize_categories,
              [('"No internet service" -> "No"', f"{no_internet:,} values"),
               ('"No phone service" -> "No"', f"{no_phone:,} values"),
               ("SeniorCitizen", "0/1 -> No/Yes")])

    st.header("Engineered features")
    engineered = modeling.run_quietly(preprocessing.engineer_features, clean_df)
    for feature in ["NumServices", "AvgMonthlySpend", "TenureGroup"]:
        for text in docstring_paragraphs(preprocessing.engineer_features, f"{feature}:"):
            st.markdown(f"- **{feature}:** {text.removeprefix(feature + ':').strip()}")

    shares = (with_outcome(engineered).groupby("TenureGroup")["Outcome"]
              .value_counts(normalize=True).rename("Share").reset_index())
    fig = px.bar(shares, x="TenureGroup", y="Share", color="Outcome",
                 color_discrete_map=OUTCOME_COLORS, text_auto=".0%",
                 category_orders={"Outcome": ["Stayed", "Churned"],
                                  "TenureGroup": ["0-12m", "13-24m", "25-48m", "49-72m"]})
    fig.update_layout(title="Churn rate by tenure group: new customers leave most",
                      yaxis_tickformat=".0%", bargap=0.3)
    show_chart(style_figure(fig))

    st.header("Encoding, splitting, and scaling")
    show_step("One-hot encoding", preprocessing.encode_one_hot)
    show_step("Stratified 80/20 split", preprocessing.split_data)
    show_step("Standard scaling", preprocessing.scale_features)


# ==============================================================================
# PAGE 3: MODEL RESULTS
# ==============================================================================

def page_model_results():
    """Page 3: comparison table, confusion matrices, ROC, threshold, features."""
    all_models, best_params = train_models()
    evaluator = get_evaluator()
    comparison = get_comparison_table()

    st.title("Model Results")
    st.markdown(
        "Two algorithms, **Logistic Regression** and **Random Forest**, both with "
        '`class_weight="balanced"` because only 26.5% of customers churn. '
        "Tuned with 5-fold `GridSearchCV` on the training set (scoring = F1). "
        "All numbers below are on the **held-out test set** (1,409 customers)."
    )

    st.header("Comparison")
    styled = (comparison.style.format("{:.3f}")
              .highlight_max(axis=0, props=f"background-color: {DARK_GREEN}; "
                                           f"color: {TEXT}; font-weight: bold;"))
    st.dataframe(styled, width="stretch")
    st.caption(
        "Green = best value in each column. The 'everyone stays' baseline is 73.5% "
        "accurate but catches no churners, which is why we judge the models on "
        "recall and F1, not accuracy.  \n"
        f"Best parameters - Logistic Regression: {best_params['Logistic Regression']}; "
        f"Random Forest: {best_params['Random Forest']}"
    )

    st.header("Confusion matrices (threshold 0.5)")
    for column, name in zip(st.columns(2), TUNED_MODELS):
        cm = evaluator.metrics_at_threshold(all_models[name], 0.5)["Confusion matrix"]
        with column:
            show_chart(confusion_matrix_figure(cm, name))

    st.header("ROC curves")
    fig = go.Figure()
    for name, color, dash in zip(TUNED_MODELS, [GREEN, LIGHT_GREEN], ["solid", "dash"]):
        fpr, tpr, auc = evaluator.roc_points(all_models[name])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC = {auc:.3f})",
                                 line=dict(color=color, width=2, dash=dash)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guessing (0.500)",
                             line=dict(color=MUTED, width=1, dash="dot")))
    fig.update_layout(xaxis_title="False positive rate (stayers wrongly flagged)",
                      yaxis_title="True positive rate (churners caught)")
    show_chart(style_figure(fig, height=440))

    st.header("Choose the decision threshold")
    st.markdown(
        "A model outputs a churn **probability**; the threshold decides when that "
        "counts as *will churn*. **Lower** the threshold and you catch more real "
        "churners (recall goes up), but you also flag more loyal customers by "
        "mistake (precision goes down). **Raise** it and the opposite happens. "
        "The right setting depends on what costs more: losing a customer, or "
        "sending a retention offer to someone who would have stayed."
    )
    col_model, col_slider = st.columns([1, 2])
    name = col_model.selectbox("Model", TUNED_MODELS)
    threshold = col_slider.slider("Threshold", 0.05, 0.95, 0.50, step=0.05)
    current = evaluator.metrics_at_threshold(all_models[name], threshold)
    default = evaluator.metrics_at_threshold(all_models[name], modeling.DEFAULT_THRESHOLD)

    cards = st.columns(4)
    for card, metric in zip(cards, ["Recall", "Precision", "Accuracy", "F1"]):
        card.metric(metric, f"{current[metric]:.1%}",
                    delta=f"{current[metric] - default[metric]:+.1%} vs 0.5"
                    if threshold != modeling.DEFAULT_THRESHOLD else None)
    cm = current["Confusion matrix"]
    col_cm, col_text = st.columns([3, 2])
    with col_cm:
        show_chart(confusion_matrix_figure(cm, f"{name} at threshold {threshold:.2f}"))
    with col_text:
        caught, missed, false_alarms = cm[1][1], cm[1][0], cm[0][1]
        st.markdown(
            f"#### In plain English\n"
            f"- Catches **{caught} of {caught + missed}** customers who really left.\n"
            f"- Misses **{missed}** churners (they leave without warning).\n"
            f"- Flags **{false_alarms}** customers who would have stayed "
            f"(false alarms that would receive a retention offer)."
        )

    st.header("What drives churn? Top 10 features")
    X_train, _, _, _ = load_model_data()
    top_coefs, top_importances = evaluator.top_features(
        all_models["Tuned Logistic Regression"], all_models["Tuned Random Forest"],
        list(X_train.columns))
    col_lr, col_rf = st.columns(2)

    lr_plot = top_coefs.iloc[::-1]          # largest at the top of a horizontal bar chart
    # DECISION: positive coefficients push toward "Churned", so they get the churn
    # red; negative ones push toward "Stayed" and get the stay green.
    fig = go.Figure(go.Bar(x=lr_plot.values, y=lr_plot.index, orientation="h",
                           marker_color=[RED if v > 0 else GREEN for v in lr_plot.values]))
    fig.update_layout(title="Logistic Regression coefficients<br><sup>red = pushes toward "
                            "Churned, green = pushes toward Stayed</sup>",
                      xaxis_title="Coefficient")
    with col_lr:
        show_chart(style_figure(fig, height=440))

    rf_plot = top_importances.iloc[::-1]
    fig = go.Figure(go.Bar(x=rf_plot.values, y=rf_plot.index, orientation="h",
                           marker_color=GREEN))
    fig.update_layout(title="Random Forest feature importances<br><sup>how much each "
                            "feature matters (no direction)</sup>",
                      xaxis_title="Importance")
    with col_rf:
        show_chart(style_figure(fig, height=440))


# ==============================================================================
# PAGE 4: PREDICT A CUSTOMER
# ==============================================================================

def fill_form(customer):
    """Copy a customer dict into the form widgets' session state.

    Used as a button callback: callbacks run before the page is redrawn, so
    the widgets show the new values right away.

    Parameters:
        customer (dict): values in the cleaned Task 2.1 format
    """
    for col, value in customer.items():
        if col in FIELD_LABELS:
            st.session_state[f"input_{col}"] = value


def fill_random_customer():
    """Button callback: fill the form with a random real customer."""
    fill_form(get_predictor().generate_random_customer())


def form_select(container, predictor, col, disabled=False):
    """Draw a selectbox whose options come from the cleaned data.

    Parameters:
        container: Streamlit column or container to draw in
        predictor (ChurnPredictor): provides the cleaned data
        col (str): column name
        disabled (bool): grey out the widget
    Returns:
        str: the selected value
    """
    options = sorted(predictor.clean_df[col].unique())
    return container.selectbox(FIELD_LABELS[col], options, key=f"input_{col}",
                               disabled=disabled)


def probability_figure(results):
    """Draw one stacked bar per model: stay probability (green) vs churn (red).

    Parameters:
        results (DataFrame): output of ChurnPredictor.predict_customer
    Returns:
        Figure
    """
    fig = go.Figure()
    for outcome, column, color in [("Stayed", "Stay probability", GREEN),
                                   ("Churned", "Churn probability", RED)]:
        fig.add_trace(go.Bar(
            y=results["Model"], x=results[column], name=f"{outcome} ({column.lower()})",
            orientation="h", marker_color=color, text=results[column].map("{:.0%}".format),
            textposition="inside", insidetextanchor="middle",
            textfont=dict(color=BACKGROUND, size=16)))
    fig.add_vline(x=modeling.DEFAULT_THRESHOLD, line_dash="dot", line_color=TEXT)
    fig.update_layout(barmode="stack", xaxis=dict(range=[0, 1], tickformat=".0%"),
                      yaxis=dict(autorange="reversed"), bargap=0.35)
    return style_figure(fig, height=260)


def page_predict():
    """Page 4: enter one customer and score them with both tuned models."""
    predictor = get_predictor()

    # First visit: start from the low-risk example so every widget has a value.
    if "input_tenure" not in st.session_state:
        fill_form(modeling.LOW_RISK_CUSTOMER)
    # DECISION: in the cleaned data, customers without internet have every
    # add-on set to "No" (and no multiple lines without a phone line), so the
    # form enforces the same rule before the widgets are drawn.
    if st.session_state["input_InternetService"] == "No":
        for col in preprocessing.INTERNET_ADDONS:
            st.session_state[f"input_{col}"] = "No"
    if st.session_state["input_PhoneService"] == "No":
        st.session_state["input_MultipleLines"] = "No"

    st.title("Predict a Customer")
    st.markdown("Fill in a customer profile, or load an example, then press **Predict**.")

    b1, b2, b3, _ = st.columns([1, 1, 1, 2])
    b1.button("Random customer", on_click=fill_random_customer, width="stretch")
    b2.button("High risk example", on_click=fill_form, args=(modeling.HIGH_RISK_CUSTOMER,),
              width="stretch")
    b3.button("Low risk example", on_click=fill_form, args=(modeling.LOW_RISK_CUSTOMER,),
              width="stretch")

    customer = {}
    col_people, col_services, col_account = st.columns(3)
    with col_people:
        st.subheader("Customer")
        for col in ["gender", "SeniorCitizen", "Partner", "Dependents"]:
            customer[col] = form_select(col_people, predictor, col)
    with col_services:
        st.subheader("Services")
        customer["PhoneService"] = form_select(col_services, predictor, "PhoneService")
        customer["MultipleLines"] = form_select(col_services, predictor, "MultipleLines",
                                                disabled=customer["PhoneService"] == "No")
        customer["InternetService"] = form_select(col_services, predictor, "InternetService")
        no_internet = customer["InternetService"] == "No"
        for col in preprocessing.INTERNET_ADDONS:
            customer[col] = form_select(col_services, predictor, col, disabled=no_internet)
    with col_account:
        st.subheader("Account")
        customer["tenure"] = st.slider(FIELD_LABELS["tenure"], 0, 72, key="input_tenure")
        customer["MonthlyCharges"] = st.slider(FIELD_LABELS["MonthlyCharges"], 18.0, 120.0,
                                               step=0.05, key="input_MonthlyCharges")
        for col in ["Contract", "PaperlessBilling", "PaymentMethod"]:
            customer[col] = form_select(col_account, predictor, col)
        # DECISION: TotalCharges is not asked; it is estimated as tenure x
        # MonthlyCharges, which is how it is built for a customer whose price
        # never changed.
        customer["TotalCharges"] = customer["tenure"] * customer["MonthlyCharges"]
        st.metric("Total charges (tenure x monthly)", f"${customer['TotalCharges']:,.2f}")

    if st.button("Predict", type="primary", width="stretch"):
        results = predictor.predict_customer(customer)
        st.header("Prediction")
        for card, (_, row) in zip(st.columns(len(results)), results.iterrows()):
            card.metric(row["Model"], row["Prediction"],
                        f"{row['Churn probability']:.0%} churn risk", delta_color="off")
        show_chart(probability_figure(results))
        st.caption(
            'These are risk scores, not calibrated real-world probabilities: '
            'class_weight="balanced" makes the models treat churners as if they '
            "were as common as stayers, which pushes scores upward. Use them to "
            "rank customers by risk; the label uses the default 0.5 threshold "
            "(dotted line)."
        )


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Set up the page, draw the sidebar navigation, and show the chosen page."""
    st.set_page_config(page_title="Telco Churn Project", page_icon=":material/insights:",
                       layout="wide")
    ensure_processed_data()

    st.sidebar.title("Telco Churn")
    st.sidebar.caption("MSITM 6341 - Group A midterm project")
    page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("Data: IBM Telco Customer Churn (Kaggle). "
                       "Models: Logistic Regression and Random Forest (scikit-learn).")

    if page == PAGES[0]:
        page_dataset()
    elif page == PAGES[1]:
        page_cleaning()
    elif page == PAGES[2]:
        page_model_results()
    else:
        page_predict()


main()
