# Location: project/code/ml/pipeline/06_plot_cluster_viz.py | PROJECT_ROOT depth: 3
# (formerly project/code/ml/plot_cluster_viz.py)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# ---------- Path Configuration (dynamic) ----------
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

SCRIPT_DIR = Path(__file__).resolve().parent
# Location: ml/pipeline/06_plot_cluster_viz.py -> 3 levels up = project/
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

# 2026-11-22: the named parquet is produced by pipeline/04_rename_clusters.py and
# no longer lives in out_of_use/ (04's DST_DIR used to point at the deprecated
# folder, which made this script fail with FileNotFoundError).
# CAVEAT: the cluster IDS inside it are the legacy 03 numbering
# (0 = short_high_tip / 1 = long_low_tip), which is the REVERSE of the robust
# clustering. Everything below is keyed on `cluster_name`, so the colours and
# the figures are correct; do not relabel axes with raw cluster ids.
INPUT_CLUSTERED = PROJECT_ROOT / "data" / "output" / "anomaly_features" / "anomalies_clustered_named.parquet"
INPUT_SUMMARY = PROJECT_ROOT / "data" / "output" / "anomaly_features" / "cluster_money_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "anomaly_features" / "viz"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {"long_low_tip": "#EF553B", "short_high_tip": "#636EFA"}

# ---------- Load data ----------
print("Loading data...")
df = pd.read_parquet(INPUT_CLUSTERED)
summary_df = pd.read_csv(INPUT_SUMMARY)

# Inverse transform to real values
df['fare'] = np.expm1(df['log_fare'])
df['tips'] = np.expm1(df['log_tips'])
df['miles'] = np.expm1(df['log_miles'])
df['tip_rate'] = df['tips'] / df['fare']

# ---------- Figure 1: Grouped bar chart (dual axis) - Optimized ----------
fig, ax1 = plt.subplots(figsize=(9, 6))
ax2 = ax1.twinx()

clusters = ['long_low_tip', 'short_high_tip']
x = np.arange(len(clusters))
width = 0.25  # Reduce width to avoid overlap

# Define data
median_fare = [210.90, 134.69]
median_tip = [0.00, 26.18]
zero_tip_pct = [56.26, 0.21]
median_miles = [62.48, 34.40]

# Left axis: Amount (USD)
bar1 = ax1.bar(x - width*1.5, median_fare, width, label='Median Fare ($)', color='#2CA02C', alpha=0.8, edgecolor='black', linewidth=0.5)
bar2 = ax1.bar(x - width/2, median_tip, width, label='Median Tip ($)', color='#1F77B4', alpha=0.8, edgecolor='black', linewidth=0.5)
ax1.set_ylabel('Amount (USD)', fontsize=11, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(clusters, fontsize=11)
ax1.set_ylim(0, 260)

# Right axis: Percentage & Miles
bar3 = ax2.bar(x + width/2, zero_tip_pct, width, label='Zero Tip (%)', color='#D62728', alpha=0.7, edgecolor='black', linewidth=0.5)
bar4 = ax2.bar(x + width*1.5, median_miles, width, label='Median Miles', color='#FF7F0E', alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.set_ylabel('Percentage (%) / Miles', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 75)

# Function to add value labels
def add_labels(axes, bars, values, offset=3):
    for bar, val in zip(bars, values):
        height = bar.get_height()
        label_val = f'{val:.2f}' if height > 0 else '0'
        axes.text(bar.get_x() + bar.get_width()/2., height + offset,
                label_val, ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add labels to bars
add_labels(ax1, bar1, median_fare)
add_labels(ax1, bar2, median_tip)
add_labels(ax2, bar3, zero_tip_pct)
add_labels(ax2, bar4, median_miles)

# Combine legends
bars_list = [bar1, bar2, bar3, bar4]
labels = [b.get_label() for b in bars_list]
ax1.legend(bars_list, labels, loc='upper left', bbox_to_anchor=(1.08, 1), frameon=False)

plt.title('Key Metrics Comparison by Cluster', fontsize=14, pad=15)
fig.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/1_key_metrics_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved optimized 1_key_metrics_comparison.png")

# ---------- Figure 2: Tip rate boxplot ----------
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='cluster_name', y='tip_rate', palette=COLORS, width=0.5)
plt.ylabel('Tip Rate (Tips / Fare)')
plt.xlabel('Cluster')
plt.title('Tip Rate Distribution by Cluster')
plt.ylim(-0.05, 0.8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/2_tip_rate_boxplot.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved 2_tip_rate_boxplot.png")

# ---------- Figure 3: Miles violin plot ----------
plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x='cluster_name', y='miles', palette=COLORS, inner="quartile", width=0.7)
plt.ylabel('Miles')
plt.xlabel('Cluster')
plt.title('Miles Distribution by Cluster (Capped at 100)')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/3_miles_violinplot.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved 3_miles_violinplot.png")

# ---------- Figure 4: Zero tip rate bar chart ----------
zero_tip_data = summary_df[summary_df['cluster_name'].isin(clusters)][['cluster_name', 'zero_tip_pct']]
plt.figure(figsize=(6, 5))
sns.barplot(data=zero_tip_data, x='cluster_name', y='zero_tip_pct', palette=COLORS)
plt.ylabel('Zero Tip Percentage (%)')
plt.xlabel('Cluster')
plt.title('Zero Tip Rate by Cluster')
for i, v in enumerate(zero_tip_data['zero_tip_pct']):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/4_zero_tip_bar.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved 4_zero_tip_bar.png")

print("All visualizations saved to:", OUTPUT_DIR)