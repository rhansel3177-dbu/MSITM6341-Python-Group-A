
# Task 2.2 - Exploratory Data Analysis (EDA)
# Telco Customer Churn Midterm Project

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Find the project folder and cleaned dataset
project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "data" / "processed" / "telco_churn_clean.csv"

# Load the cleaned dataset from Task 2.1
df = pd.read_csv(file_path)

# Check that the data loaded correctly
print("=== Dataset Shape ===")
print(df.shape)

print("\n=== First 5 Rows ===")
print(df.head())
# Display basic descriptive statistics for numerical columns
print("\n=== Basic Descriptive Statistics ===")
print(df.describe())
# Display column names, data types, and missing-value information
print("\n=== Dataset Information ===")
df.info()
# Analyze customer churn distribution and class imbalance
print("\n=== Customer Churn Distribution ===")
print(df["Churn"].value_counts())

print("\n=== Customer Churn Percentage ===")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))

# Analyze correlations among numerical variables
print("\n=== Correlation Matrix ===")

numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

correlation_matrix = df[numeric_columns].corr()

print(correlation_matrix.round(3))

# Analyze correlations with customer churn
# Convert Churn from Yes/No to 1/0 for analysis
df["ChurnNumeric"] = df["Churn"].map({"Yes": 1, "No": 0})

print("\n=== Correlation with Customer Churn ===")

churn_columns = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "ChurnNumeric"
]

churn_correlation = df[churn_columns].corr()

print(churn_correlation["ChurnNumeric"].round(3))

# Identify potential outliers using the IQR method
print("\n=== Outlier Analysis (IQR Method) ===")

for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print(f"\n{column}:")
    print(f"Lower Bound: {lower_bound:.2f}")
    print(f"Upper Bound: {upper_bound:.2f}")
    print(f"Number of Potential Outliers: {len(outliers)}")

# Visualization 1: Histogram of Monthly Charges

# Create a folder to save our charts
output_folder = project_root / "outputs" / "figures"
output_folder.mkdir(parents=True, exist_ok=True)

# Create the histogram
plt.figure(figsize=(9, 6))

plt.hist(df["MonthlyCharges"], bins=20, edgecolor="black")

plt.title("Distribution of Monthly Charges")
plt.xlabel("Monthly Charges (USD)")
plt.ylabel("Number of Customers")

plt.tight_layout()

# Save the chart as a PNG image
plt.savefig(output_folder / "monthly_charges_histogram.png", dpi=300)

plt.close()

print("\nHistogram saved successfully!") 


# Visualization 2: Bar Chart of Churn Rate by Contract Type

# Calculate the percentage of customers who churned
# for each contract type
contract_churn = (
    df.groupby("Contract")["ChurnNumeric"].mean() * 100
)

# Arrange contract types in a meaningful order
contract_order = ["Month-to-month", "One year", "Two year"]
contract_churn = contract_churn.reindex(contract_order)

# Create the bar chart
plt.figure(figsize=(9, 6))
bars = plt.bar(contract_churn.index, contract_churn.values,
               edgecolor="black")

plt.title("Customer Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.ylim(0, 50)

# Display the percentage above each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.1f}%",
        ha="center"
    )

plt.tight_layout()

# Save the chart
plt.savefig(output_folder / "contract_churn_bar_chart.png", dpi=300)
plt.close()

print("\nContract churn bar chart saved successfully!")
print(contract_churn.round(2))


# Visualization 3: Box Plot of Customer Tenure by Churn Status

# Separate tenure values for retained and churned customers
tenure_no_churn = df[df["Churn"] == "No"]["tenure"]
tenure_churn = df[df["Churn"] == "Yes"]["tenure"]

# Create the box plot
plt.figure(figsize=(9, 6))

plt.boxplot(
    [tenure_no_churn, tenure_churn],
    tick_labels=["No Churn", "Churn"],
    patch_artist=True
)

plt.title("Customer Tenure by Churn Status")
plt.xlabel("Customer Churn Status")
plt.ylabel("Tenure (Months)")

plt.tight_layout()

# Save the chart
plt.savefig(output_folder / "tenure_churn_boxplot.png", dpi=300)
plt.close()

print("\nTenure box plot saved successfully!")

# Display median tenure for each group
print("\nMedian Tenure by Churn Status:")
print(df.groupby("Churn")["tenure"].median())


# Visualization 4: Correlation Heatmap

# Use the correlation matrix that includes customer churn
heatmap_data = df[churn_columns].corr()

# Create the heatmap
plt.figure(figsize=(9, 7))

plt.imshow(heatmap_data, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation Coefficient")

# Add variable names to both axes
plt.xticks(
    range(len(churn_columns)),
    churn_columns,
    rotation=45,
    ha="right"
)
plt.yticks(range(len(churn_columns)), churn_columns)

# Display correlation values inside the heatmap
for i in range(len(churn_columns)):
    for j in range(len(churn_columns)):
        value = heatmap_data.iloc[i, j]
        plt.text(
            j, i, f"{value:.2f}",
            ha="center",
            va="center",
            color="black"
        )

plt.title("Correlation Heatmap of Customer Churn Variables")
plt.tight_layout()

# Save the heatmap
plt.savefig(output_folder / "churn_correlation_heatmap.png", dpi=300)
plt.close()

print("\nCorrelation heatmap saved successfully!")
