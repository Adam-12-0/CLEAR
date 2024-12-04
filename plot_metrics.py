import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Define a function to generate and save graphs
def generate_and_save_graphs(csv_file_path):
    # Define perfect scores for normalization
    perfect_scores = {
        "Average PSNR": 100.0,  # Assuming perfect PSNR as a normalized 100% baseline
        "Average SSIM": 1.0,    # SSIM perfect value
        "Average VIF": 1.0      # VIF perfect value
    }
    
    # Load the CSV file
    data = pd.read_csv(csv_file_path)

    # Remove rows where PSNR has 'inf'
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["Average PSNR"])

    # Normalize the metrics as percentages
    for metric, perfect_value in perfect_scores.items():
        data[f"{metric} (%)"] = (data[metric] / perfect_value) * 100

    # Create output directory for the graphs
    output_dir = os.path.dirname(csv_file_path)
    graphs_dir = os.path.join(output_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    # 1. Grouped Bar Chart: Average Metric Comparison Across Noise Types
    normalized_avg_metrics = data.groupby("Noise Type")[
        [f"{metric} (%)" for metric in perfect_scores.keys()]
    ].mean()
    bar_graph_path = os.path.join(graphs_dir, "Average_Metric_Comparison_Across_Noise_Types.png")
    ax = normalized_avg_metrics.plot(kind='bar', figsize=(10, 6), width=0.8, color=['skyblue', 'lightgreen', 'salmon'])
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', fontsize=10)
    plt.title('Average Metric Comparison Across Noise Types (Normalized Percentages)', fontsize=14)
    plt.ylabel('Percentage', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.legend(title='Metrics', fontsize=10)
    plt.tight_layout()
    plt.savefig(bar_graph_path)
    plt.close()

    # 2. Histograms: Average Noisiness Percentage by Action Category
    category_histograms = data.groupby(["Noise Type", "Video Category"])[
        [f"{metric} (%)" for metric in perfect_scores.keys()]
    ].mean()
    average_percentages = category_histograms.mean(axis=1).reset_index()
    average_percentages.columns = ["Noise Type", "Video Category", "Average Noisiness (%)"]
    noise_types = data["Noise Type"].unique()
    for noise in noise_types:
        histogram_path = os.path.join(graphs_dir, f"Average_Noisiness_Percentage_by_Action_Category_{noise}.png")
        subset = average_percentages[average_percentages["Noise Type"] == noise]
        subset = subset.sort_values(by="Average Noisiness (%)", ascending=False)
        plt.figure(figsize=(14, 6))
        plt.bar(subset["Video Category"], subset["Average Noisiness (%)"], color='skyblue')
        plt.title(f'Average Noisiness Percentage by Action Category ({noise})', fontsize=14)
        plt.ylabel('Average Noisiness (%)', fontsize=12)
        plt.xlabel('Action Categories', fontsize=12)
        plt.xticks(rotation=45, fontsize=6, ha='right')  # Diagonal labels with smaller font
        plt.tight_layout()
        plt.savefig(histogram_path)
        plt.close()

    # 3. Line Graph: Noise-Specific Impact by Metric (Normalized Percentages)
    normalized_metric_impact = data.groupby("Noise Type")[
        [f"{metric} (%)" for metric in perfect_scores.keys()]
    ].mean()
    line_graph_path = os.path.join(graphs_dir, "Noise_Specific_Impact_by_Metric.png")
    normalized_metric_impact.plot(kind='line', figsize=(10, 6), marker='o')
    plt.title('Noise-Specific Impact by Metric (Normalized Percentages)')
    plt.ylabel('Average Metric Percentage')
    plt.xticks(range(len(normalized_metric_impact)), normalized_metric_impact.index, rotation=45)
    plt.legend(title='Metrics')
    plt.tight_layout()
    plt.savefig(line_graph_path)
    plt.close()

    print(f"Graphs saved in: {graphs_dir}")

# Example usage
csv_file_path = "metrics/Speckle/metrics_summary.csv"  # Replace this with your CSV file path
generate_and_save_graphs(csv_file_path)
