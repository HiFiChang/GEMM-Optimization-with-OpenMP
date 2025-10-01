import os
import matplotlib
# Use a non-interactive backend to avoid Qt/xcb issues in headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

results_dir = "results"
csv_file_path = os.path.join(results_dir, "results.csv")
output_image_path = os.path.join(results_dir, "performance_comparison.png")

# --- Read Data ---
try:
    data = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"Error: The file {csv_file_path} was not found.")
    print("Please make sure you have run the benchmark script first.")
    exit()

# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 8))

# Get the list of unique versions, sorting them for consistent plot order
versions = sorted(data['Remarks'].unique())

for version in versions:
    # Filter data for the current version
    version_data = data[data['Remarks'] == version]
    
    # Plot N vs. Average_GFLOPS/s
    ax.plot(version_data['N'], version_data['Average_GFLOPS/s'], marker='o', linestyle='-', label=version)

# --- Customize Plot ---
ax.set_title('Performance Comparison of GEMM Versions', fontsize=18, fontweight='bold')
ax.set_xlabel('Matrix Size (N)', fontsize=12)
ax.set_ylabel('Performance (Average GFLOPS/s)', fontsize=12)

# Use a logarithmic scale for the x-axis for better visualization of N values
# ax.set_xscale('log')
# Use sorted unique N values for ordered tick positions; spacing will be logarithmic by design
# ax.set_xticks(sorted(data['N'].unique()))
# Format tick labels as plain numbers instead of scientific notation on the log axis
ax.get_xaxis().set_major_formatter(ScalarFormatter())
plt.xticks(rotation=45)


ax.legend(title='Version', fontsize=10)
ax.grid(True, which="both", ls="--")

# Improve layout
plt.tight_layout()

# --- Save and Show Plot ---
try:
    plt.savefig(output_image_path, dpi=300)
    print(f"Plot saved successfully to {output_image_path}")
except Exception as e:
    print(f"Error saving plot: {e}")

# plt.show()
