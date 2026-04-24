import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob 
import os

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


CLINICAL_FEATURES_FILE = "output/subject_level_gait_features_clustered.csv"
APPLE_WATCH_FILE = "HealthExport_*.csv"


def load_clinical_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    needed_cols = ["stride_asymmetry_ratio", "Group"]
    df = df.dropna(subset=needed_cols).copy()

    # PD = 1, Control = 0
    df["label"] = df["Group"].apply(lambda x: 1 if x == 1 else 0)

    return df


def train_logistic_regression(df: pd.DataFrame):
    X = df[["stride_asymmetry_ratio"]].values
    y = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\nLogistic Regression using asymmetry only")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    return model, scaler


def parse_range_midpoint(value):
    """
    Apple export sometimes stores values like:
    '29.7-31.2'
    '1.23-2.3'
    '-'
    We take the midpoint if it's a range.
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value == "-" or value == "":
        return np.nan

    if "-" in value:
        parts = value.split("-")
        try:
            nums = [float(p) for p in parts if p.strip() != ""]
            if len(nums) == 2:
                return (nums[0] + nums[1]) / 2
        except ValueError:
            return np.nan

    try:
        return float(value)
    except ValueError:
        return np.nan


def load_apple_watch_data(file_pattern: str) -> pd.DataFrame:
    file_paths = sorted(glob.glob(file_pattern))

    if not file_paths:
        raise FileNotFoundError(f"No files matched pattern: {file_pattern}")

    all_dfs = []

    for path in file_paths:
        print("Loading Apple Watch file from:", path)
        df = pd.read_csv(path)

        df.columns = df.columns.str.strip()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        df["walking_asymmetry_percent"] = pd.to_numeric(
            df["Walking Asymmetry (%)"], errors="coerce"
        )

        df["double_support_mid"] = df["Double Support Time (%)"].apply(parse_range_midpoint)
        df["step_length_mid"] = df["Step Length (in)"].apply(parse_range_midpoint)
        df["walking_speed_mid"] = df["Walking Speed (mi/hr)"].apply(parse_range_midpoint)

        df["stride_asymmetry_ratio"] = df["walking_asymmetry_percent"] / 100.0
        df["source_file"] = os.path.basename(path)

        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates()

    return combined_df


def apply_model_to_apple_watch(df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    clean_df = df.dropna(subset=["stride_asymmetry_ratio"]).copy()

    X = clean_df[["stride_asymmetry_ratio"]].values
    X_scaled = scaler.transform(X)

    clean_df["impairment_probability"] = model.predict_proba(X_scaled)[:, 1]
    clean_df["mobility_score"] = clean_df["impairment_probability"] * 100

    def classify_score(score):
        if score < 30:
            return "Stable"
        elif score < 60:
            return "Moderate"
        else:
            return "High Variability"

    clean_df["mobility_category"] = clean_df["mobility_score"].apply(classify_score)

    return clean_df

def summarize_months(df: pd.DataFrame):
    df["month"] = df["Date"].dt.month

    monthly_summary = df.groupby(df["Date"].dt.to_period("M")).agg(
        mean_asymmetry_percent=("walking_asymmetry_percent", "mean"),
        mean_impairment_probability=("impairment_probability", "mean"),
        mean_mobility_score=("mobility_score", "mean"),
        days_recorded=("Date", "count"),
    )

    print("\nMonthly Summary:")
    print(monthly_summary)


def make_plots(df: pd.DataFrame):
    # Plot 1: asymmetry over time
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["walking_asymmetry_percent"], marker="o")
    plt.title("Walking Asymmetry Over Time")
    plt.xlabel("Date")
    plt.ylabel("Walking Asymmetry (%)")
    plt.tight_layout()
    plt.savefig("apple_watch_asymmetry_over_time.png", dpi=300)
    plt.close()

    # Plot 2: impairment probability over time
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["impairment_probability"], marker="o")
    plt.title("Impairment Probability Over Time")
    plt.xlabel("Date")
    plt.ylabel("Impairment Probability")
    plt.tight_layout()
    plt.savefig("apple_watch_impairment_probability_over_time.png", dpi=300)
    plt.close()

    # Plot 3: mobility score over time
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["mobility_score"], marker="o")
    plt.title("Mobility Score Over Time")
    plt.xlabel("Date")
    plt.ylabel("Mobility Score (0-100)")
    plt.tight_layout()
    plt.savefig("apple_watch_mobility_score_over_time.png", dpi=300)
    plt.close()

    # Plot 4: month comparison
    temp = df.copy()
    temp["month_name"] = temp["Date"].dt.month.map({1: "January", 2: "February", 3: "March"})

    vals = []
    labels = []
    for month in ["January", "February", "March"]:
        month_vals = temp[temp["month_name"] == month]["walking_asymmetry_percent"].dropna()
        if len(month_vals) > 0:
            vals.append(month_vals)
            labels.append(month)

    if vals:
        plt.figure(figsize=(8, 6))
        plt.boxplot(vals, tick_labels=labels)
        plt.title("Walking Asymmetry Comparison by Month")
        plt.ylabel("Walking Asymmetry (%)")
        plt.tight_layout()
        plt.savefig("apple_watch_month_comparison.png", dpi=300)
        plt.close()

def main():
    clinical_df = load_clinical_data(CLINICAL_FEATURES_FILE)
    model, scaler = train_logistic_regression(clinical_df)

    apple_df = load_apple_watch_data(APPLE_WATCH_FILE)
    results_df = apply_model_to_apple_watch(apple_df, model, scaler)

    print("\nApple Watch preview:")
    print(results_df[
        ["Date", "walking_asymmetry_percent", "stride_asymmetry_ratio",
        "impairment_probability", "mobility_score", "mobility_category"]
    ].head(10))

    summarize_months(results_df)

    results_df.to_csv("apple_watch_logistic_results.csv", index=False)
    print("\nSaved: apple_watch_logistic_results.csv")

    make_plots(results_df)
    print("Saved plots:")
    print("- apple_watch_asymmetry_over_time.png")
    print("- apple_watch_pd_probability_over_time.png")
    print("- apple_watch_month_comparison.png")


if __name__ == "__main__":
    main()