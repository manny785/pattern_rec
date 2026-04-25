import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_DIR = "data"   
OUTPUT_DIR = "output" 
DEMOGRAPHICS_FILE = os.path.join(DATA_DIR, "demographics.txt")

RAW_COLUMNS = [
    "time",
    "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8",
    "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8",
    "total_left",
    "total_right",
]

FORCE_THRESHOLD = 20.0
MIN_GAP_SEC = 0.30
OUTLIER_Z_THRESHOLD = 3.0


def load_demographics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+|\t+", engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df


def parse_filename(filename: str) -> dict:
    base = os.path.basename(filename).replace(".txt", "")
    match = re.match(r"([A-Za-z]{2}(?:Co|Pt)\d{2})_(\d{2})", base)
    if not match:
        raise ValueError(f"Unexpected filename format: {filename}")

    return {
        "subject_id": match.group(1),
        "walk_num": match.group(2),
        "file_name": os.path.basename(filename),
    }


def read_walk_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+|\t+", header=None, engine="python")
    if df.shape[1] != 19:
        raise ValueError(f"{path} has {df.shape[1]} columns, expected 19")
    df.columns = RAW_COLUMNS
    return df


def detect_contact_times(force_signal: pd.Series, time_signal: pd.Series,
                         threshold: float = FORCE_THRESHOLD,
                         min_gap_sec: float = MIN_GAP_SEC) -> np.ndarray:
    loaded = (force_signal > threshold).astype(int)
    rising_idx = np.where((loaded.shift(1, fill_value=0) == 0) & (loaded == 1))[0]

    if len(rising_idx) == 0:
        return np.array([])

    contact_times = time_signal.iloc[rising_idx].to_numpy()

    filtered = [contact_times[0]]
    for t in contact_times[1:]:
        if t - filtered[-1] >= min_gap_sec:
            filtered.append(t)

    return np.array(filtered)


def compute_stride_times(contact_times: np.ndarray) -> np.ndarray:
    if len(contact_times) < 2:
        return np.array([])
    return np.diff(contact_times)


def safe_cv(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    mean_val = np.mean(values)
    if mean_val == 0:
        return np.nan
    return (np.std(values, ddof=1) / mean_val) * 100 if len(values) > 1 else 0.0


def extract_stride_features(df: pd.DataFrame) -> dict:
    time_signal = df["time"]
    left_force = df["total_left"]
    right_force = df["total_right"]

    left_contacts = detect_contact_times(left_force, time_signal)
    right_contacts = detect_contact_times(right_force, time_signal)

    left_stride_times = compute_stride_times(left_contacts)
    right_stride_times = compute_stride_times(right_contacts)

    left_mean = np.mean(left_stride_times) if len(left_stride_times) > 0 else np.nan
    right_mean = np.mean(right_stride_times) if len(right_stride_times) > 0 else np.nan

    left_cv = safe_cv(left_stride_times)
    right_cv = safe_cv(right_stride_times)

    stride_asymmetry_ratio = (
        abs(left_mean - right_mean) / ((left_mean + right_mean) / 2)
        if pd.notna(left_mean) and pd.notna(right_mean) and (left_mean + right_mean) > 0
        else np.nan
    )

    return {
        "left_stride_cv": left_cv,
        "right_stride_cv": right_cv,
        "stride_asymmetry_ratio": stride_asymmetry_ratio,
    }


def build_trial_feature_table(data_dir: str, demographics: pd.DataFrame) -> pd.DataFrame:
    all_rows = []
    raw_files = glob.glob(os.path.join(data_dir, "*.txt"))

    for path in raw_files:
        name = os.path.basename(path)

        if name in {"demographics.txt", "format.txt"}:
            continue

        if not re.match(r"^[A-Za-z]{2}(Co|Pt)\d{2}_\d{2}\.txt$", name):
            continue

        info = parse_filename(path)

        # normal walking only
        if info["walk_num"] != "01":
            continue

        walk_df = read_walk_file(path)
        feature_row = extract_stride_features(walk_df)

        row = {**info, **feature_row}
        all_rows.append(row)

    feature_df = pd.DataFrame(all_rows)

    merged = feature_df.merge(
        demographics,
        left_on="subject_id",
        right_on="ID",
        how="left"
    )
    return merged


def build_subject_feature_table(trial_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average multiple trials for the same subject instead of discarding them.
    """
    numeric_feature_cols = [
        "left_stride_cv",
        "right_stride_cv",
        "stride_asymmetry_ratio",
        "Age",
        "Height",
        "Weight",
        "HoehnYahr",
        "UPDRS",
        "UPDRSM",
        "TUAG",
    ]

    existing_numeric_cols = [col for col in numeric_feature_cols if col in trial_df.columns]

    group_cols = ["subject_id"]

    aggregated = (
        trial_df.groupby(group_cols, as_index=False)[existing_numeric_cols]
        .mean()
    )

    metadata_cols = ["subject_id", "ID", "Study", "Group", "Subjnum", "Gender"]
    existing_metadata_cols = [col for col in metadata_cols if col in trial_df.columns]

    metadata = trial_df[existing_metadata_cols].drop_duplicates(subset=["subject_id"])

    subject_df = aggregated.merge(metadata, on="subject_id", how="left")
    return subject_df


def remove_outliers(subject_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    clean_df = subject_df.dropna(subset=feature_cols).copy()

    z_scores = (clean_df[feature_cols] - clean_df[feature_cols].mean()) / clean_df[feature_cols].std(ddof=0)
    mask = (np.abs(z_scores) < OUTLIER_Z_THRESHOLD).all(axis=1)

    filtered_df = clean_df[mask].copy()

    print("\nOutlier removal summary:")
    print(f"Subjects before outlier removal: {len(clean_df)}")
    print(f"Subjects after outlier removal:  {len(filtered_df)}")
    print(f"Subjects removed:               {len(clean_df) - len(filtered_df)}")

    return filtered_df


def run_kmeans(subject_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str], StandardScaler]:
    feature_cols = [
        "left_stride_cv",
        "right_stride_cv",
        "stride_asymmetry_ratio",
    ]

    filtered_df = remove_outliers(subject_df, feature_cols)

    X = filtered_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    filtered_df["cluster"] = kmeans.fit_predict(X_scaled)

    return filtered_df, X_scaled, feature_cols, scaler


def make_plots(clustered_df: pd.DataFrame, X_scaled: np.ndarray) -> None:
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plot_df = clustered_df.copy()
    plot_df["PC1"] = X_pca[:, 0]
    plot_df["PC2"] = X_pca[:, 1]

    # Plot 1: by cluster
    plt.figure(figsize=(8, 6))
    for cluster_value in sorted(plot_df["cluster"].unique()):
        subset = plot_df[plot_df["cluster"] == cluster_value]
        plt.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {cluster_value}", alpha=0.7)

    plt.title("PCA of Subject-Level Gait Features (Colored by K-means Cluster)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"kmeans_pca_clusters.png"), dpi=300)
    plt.close()

    # Plot 2: by true group
    plt.figure(figsize=(8, 6))
    group_map = {1: "PD", 2: "Control"}

    for group_value in sorted(plot_df["Group"].dropna().unique()):
        subset = plot_df[plot_df["Group"] == group_value]
        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            label=group_map.get(group_value, f"Group {group_value}"),
            alpha=0.7
        )
    

    plt.title("PCA of Subject-Level Gait Features (Colored by True Group)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"truegroup_pca.png"), dpi=300)
    plt.close()

    # Plot 3: asymmetry boxplot
    plt.figure(figsize=(8, 6))
    pd_group = plot_df[plot_df["Group"] == 1]["stride_asymmetry_ratio"].dropna()
    co_group = plot_df[plot_df["Group"] == 2]["stride_asymmetry_ratio"].dropna()

    plt.boxplot([pd_group, co_group], tick_labels=["PD", "Control"])
    plt.title("Stride Asymmetry Ratio by Group")
    plt.ylabel("Asymmetry Ratio")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"stride_asymmetry_boxplot.png"), dpi=300)
    plt.close()

        # Plot 4: Asymmetry vs Variability scatter
    plt.figure(figsize=(8, 6))

    # combine left + right variability into one metric
    plot_df["mean_stride_cv"] = (
        plot_df["left_stride_cv"] + plot_df["right_stride_cv"]
    ) / 2

    group_map = {1: "PD", 2: "Control"}

    for group_value in sorted(plot_df["Group"].dropna().unique()):
        subset = plot_df[plot_df["Group"] == group_value]

        plt.scatter(
            subset["stride_asymmetry_ratio"],
            subset["mean_stride_cv"],
            label=group_map.get(group_value, f"Group {group_value}"),
            alpha=0.7
        )

    plt.xlabel("Stride Asymmetry Ratio")
    plt.ylabel("Mean Stride CV (%)")
    plt.title("Asymmetry vs Variability (PD vs Control)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"asymmetry_vs_variability.png"))
    plt.close()


def print_summary(clustered_df: pd.DataFrame) -> None:
    print("\nCluster counts:")
    print(clustered_df["cluster"].value_counts())

    print("\nCluster vs Group:")
    print(pd.crosstab(clustered_df["cluster"], clustered_df["Group"]))

    print("\nAverage features by true group:")
    summary_cols = [
        "left_stride_cv",
        "right_stride_cv",
        "stride_asymmetry_ratio",
    ]
    print(clustered_df.groupby("Group")[summary_cols].mean())

    print("\nSubject counts by true group:")
    print(clustered_df["Group"].value_counts(dropna=False))


def main():
    demographics = load_demographics(DEMOGRAPHICS_FILE)

    trial_df = build_trial_feature_table(DATA_DIR, demographics)
    print("Trial-level feature preview:")
    print(trial_df.head())

    subject_df = build_subject_feature_table(trial_df)
    print("\nSubject-level feature preview:")
    print(subject_df.head())

    clustered_df, X_scaled, feature_cols, scaler = run_kmeans(subject_df)

    print_summary(clustered_df)

    clustered_df.to_csv(os.path.join(OUTPUT_DIR, "subject_level_gait_features_clustered.csv" ),index=False)
    print("\nSaved: subject_level_gait_features_clustered.csv")

    make_plots(clustered_df, X_scaled)
    print("Saved plots:")
    print("- kmeans_pca_clusters.png")
    print("- truegroup_pca.png")
    print("- stride_asymmetry_boxplot.png")


if __name__ == "__main__":
    main()