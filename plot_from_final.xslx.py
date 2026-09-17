import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = "DNPMSA_RLALIGN_CEHOMSA.xlsx"

# Read the file without header
df = pd.read_excel(file_path, header=None)

# Identify metrics row: the row whose first cell is "Input" and one of its cells is "Sequences"
metrics_row_idx = None
header_row_idx = None

for idx, row in df.iterrows():
    row_str = row.astype(str).str.lower()
    if row_str.iloc[0] == "input" and "sequences" in row_str.values:
        metrics_row_idx = idx
        break

# Header row (method names): the first row whose first cell is "Input" and does not contain "Sequences"
for idx, row in df.iterrows():
    row_str = row.astype(str).str.lower()
    if row_str.iloc[0] == "input" and "sequences" not in row_str.values:
        header_row_idx = idx
        break

if header_row_idx is None or metrics_row_idx is None:
    raise ValueError("Header row or metrics row not found.")

# Extract method names (header row) and metrics (metrics row)
methods = df.iloc[header_row_idx].astype(str).str.lower()
metrics = df.iloc[metrics_row_idx].astype(str).str.lower()

# Main data between header row and metrics row (excluding those rows)
data_df = df.iloc[header_row_idx+1:metrics_row_idx].copy().reset_index(drop=True)

# Convert dataset names to string to avoid issues on x-axis
datasets = data_df.iloc[:, 0].astype(str)

# Invalid names to exclude from the list of methods
invalid_names = {
    "input", "nan", "sequences", "length", "sp", "sp-norm", "em", "al", "cs",
    "avgentropy", "gappercent", "ceho-msa"   # Separate the proposed method
}

# Set of unique method names (excluding proposed method and invalid names)
method_names = set(methods.iloc[1:]) - invalid_names

# Target metrics
target_metrics = ["sp-norm", "cs"]

# Proposed method name
proposed_method = "ceho-msa"

# Print NaN report for each method and metric
print("="*50)
print("NaN report for each method and metric:")
for metric_name in target_metrics:
    print(f"\n--- Metric: {metric_name.upper()} ---")
    proposed_col = None
    for i in range(len(methods)):
        if methods[i] == proposed_method and metrics[i] == metric_name:
            proposed_col = i
            break
    if proposed_col is None:
        print(f"Column {proposed_method} for metric {metric_name} not found.")
        continue

    # Check NaN values for proposed method
    y_prop = pd.to_numeric(data_df.iloc[:, proposed_col], errors='coerce')
    prop_nan_count = y_prop.isna().sum()
    print(f"{proposed_method.upper()}: {prop_nan_count} NaN values")

    for method in method_names:
        col = None
        for i in range(len(methods)):
            if methods[i] == method and metrics[i] == metric_name:
                col = i
                break
        if col is None:
            continue
        y_m = pd.to_numeric(data_df.iloc[:, col], errors='coerce')
        nan_count = y_m.isna().sum()
        if nan_count > 0:
            print(f"{method}: {nan_count} NaN values")

print("="*50)

# Generate plots
for metric_name in target_metrics:
    # Find column for CEHO-MSA for this metric
    proposed_col = None
    for i in range(len(methods)):
        if methods[i] == proposed_method and metrics[i] == metric_name:
            proposed_col = i
            break

    if proposed_col is None:
        print(f"Column {proposed_method} for metric {metric_name} not found.")
        continue

    for method in method_names:
        # Find column for this method for the current metric
        col = None
        for i in range(len(methods)):
            if methods[i] == method and metrics[i] == metric_name:
                col = i
                break
        if col is None:
            print(f"Method {method} for metric {metric_name} not found.")
            continue

        # Convert to numeric (NaN values are preserved)
        y_method = pd.to_numeric(data_df.iloc[:, col], errors='coerce')
        y_proposed = pd.to_numeric(data_df.iloc[:, proposed_col], errors='coerce')

        # Create figure
        plt.figure(figsize=(28, 10))
        ax = plt.gca()

        # Plot main lines (NaN points cause breaks in the line)
        plt.plot(datasets, y_method, label=method, linewidth=1.5, color="#7f8c8d", alpha=0.9)
        plt.plot(datasets, y_proposed, label=proposed_method.upper(), linewidth=2.2, color="#1f77b4")

        # Find positions of NaN for both methods
        nan_mask_method = y_method.isna()
        nan_mask_proposed = y_proposed.isna()

        # If there are any NaN values, add red markers
        if nan_mask_method.any() or nan_mask_proposed.any():
            # Get current y-axis limits to determine marker position
            y_min, y_max = ax.get_ylim()
            marker_y = y_min - 0.05 * (y_max - y_min)  # Slightly below the minimum

            # Marker for the method
            if nan_mask_method.any():
                plt.scatter(datasets[nan_mask_method], 
                            [marker_y] * nan_mask_method.sum(),
                            marker='x', color='red', s=80, 
                            label='NaN (method)', zorder=5)
            # Marker for the proposed method
            if nan_mask_proposed.any():
                plt.scatter(datasets[nan_mask_proposed], 
                            [marker_y] * nan_mask_proposed.sum(),
                            marker='x', color='orange', s=80, 
                            label=f'NaN ({proposed_method.upper()})', zorder=5)

            # Adjust y-axis limits so markers are visible
            plt.ylim(bottom=marker_y - 0.05 * (y_max - y_min))

            # Add note in title
            plt.title(f"{method} vs {proposed_method.upper()} ({metric_name.upper()}) - NaN points marked with x",
                      fontsize=14)
        else:
            plt.title(f"{method} vs {proposed_method.upper()} ({metric_name.upper()})", fontsize=16)

        plt.xticks(rotation=90, fontsize=5, alpha=0.35)
        plt.yticks(fontsize=10)
        plt.xlabel("Datasets")
        plt.ylabel(metric_name.upper())
        plt.grid(alpha=0.25)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"comparison_{method}_{metric_name}.png", dpi=300)
        plt.close()

print("All plots saved successfully.")
