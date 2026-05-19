import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import r2_score
from data.utils import *
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

def plot_apds(model_name, actual_apds_list, apd_preds_list):
    """
    Plots the distribution of errors between predicted and actual APDs for three test datasets.
    """
    apd_test1, apd_test2, apd_test3 = actual_apds_list
    apd_p_test1, apd_p_test2, apd_p_test3 = apd_preds_list

    error_test1_perc = np.abs(apd_p_test1 - apd_test1) * 100 / apd_test1
    error_test2_perc = np.abs(apd_p_test2 - apd_test2) * 100 / apd_test2
    error_test3_perc = np.abs(apd_p_test3 - apd_test3) * 100 / apd_test3

    error_test1 = np.abs(apd_p_test1 - apd_test1) / 5000
    error_test2 = np.abs(apd_p_test2 - apd_test2) / 5000
    error_test3 = np.abs(apd_p_test3 - apd_test3) / 5000

    df_test1_2 = pd.DataFrame()
    df_test1_2[["APD " + str(10 * i) for i in range(1, 11)]] = error_test1
    df_test2_2 = pd.DataFrame()
    df_test2_2[["APD " + str(10 * i) for i in range(1, 11)]] = error_test2
    df_test3_2 = pd.DataFrame()
    df_test3_2[["APD " + str(10 * i) for i in range(1, 11)]] = error_test3

    color_test1 = "#1f77b4"  
    color_test2 = "#ff7f0e"  
    color_test3 = "#2ca02c"  

    fig, ax = plt.subplots(figsize=(13, 5))

    for i, column in enumerate(df_test1_2.columns):
        for df_, color, offset, label in zip([df_test1_2, df_test2_2, df_test3_2],
                                             [color_test1, color_test2, color_test3],
                                             [-0.2, 0, 0.2],
                                             ['Test 1', 'Test 2', 'Test 3']):
            parts = ax.violinplot(df_[column], positions=[i + offset], showmeans=False, showmedians=False, showextrema=False, widths=0.18)
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
                pc.set_edgecolor('black')
                pc.set_linewidth(0.5)

            bp = ax.boxplot(df_[column], positions=[i + offset], notch=False, patch_artist=True, zorder=10, widths=0.1, showfliers=False)
            for patch in bp['boxes']:
                patch.set_facecolor('white')
                patch.set_edgecolor('black')
                patch.set_linewidth(0.5)
            for whisker in bp['whiskers']:
                whisker.set(color='black', linestyle='-', linewidth=0.5)
            for cap in bp['caps']:
                cap.set(color='black', linestyle='-', linewidth=0.5)
            for median in bp['medians']:
                median.set(color='black', linestyle='-', linewidth=0.5)

    # Aesthetic adjustments
    ax.set_xticks(np.arange(len(df_test1_2.columns)))
    ax.set_xticklabels(df_test1_2.columns, rotation=0, ha='center', fontsize=6)
    ax.set_ylabel('APD Error (s)', fontsize=14, color='black')
    ax.set_title(f"Distribution of Errors in APDs Predicted by {model_name}", fontsize=14, color='black')

    ax.tick_params(axis='y', labelsize=6)

    legend_elements = [
        Patch(facecolor=color_test1, edgecolor='gray', alpha=0.6, label='Test Dataset 1'),
        Patch(facecolor=color_test2, edgecolor='gray', alpha=0.6, label='Test Dataset 2'),
        Patch(facecolor=color_test3, edgecolor='gray', alpha=0.6, label='Test Dataset 3')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=14)
    plt.ylim(0, 0.35)
    plt.tight_layout()

    # Save the figure with loc and model name in the filename
    plt.savefig(f"./results/{model_name}_apd_range_preds.pdf", format="pdf")
    

    # Assuming sum_error_test1, sum_error_test2, and sum_error_test3 are already calculated
    sum_error_test1 = np.abs(error_test1).mean(axis=1)
    sum_error_test2 = np.abs(error_test2).mean(axis=1)
    sum_error_test3 = np.abs(error_test3).mean(axis=1)

    sum_error_test1_perc = np.abs(error_test1_perc).mean(axis=1)
    sum_error_test2_perc = np.abs(error_test2_perc).mean(axis=1)
    sum_error_test3_perc = np.abs(error_test3_perc).mean(axis=1)

    fig, ax = plt.subplots(figsize=(5, 5))

    for i, (data, color) in enumerate(zip([sum_error_test1, sum_error_test2, sum_error_test3], [color_test1, color_test2, color_test3])):
        
        parts = ax.violinplot(data, positions=[i], showmeans=False, showmedians=False,
                            showextrema=False, widths=0.3)
        for pc in parts['bodies']:
            verts = pc.get_paths()[0].vertices
            mean_x = np.mean(verts[:, 0])
            verts[:, 0] = np.clip(verts[:, 0], mean_x, np.inf)  
            pc.set_facecolor(color)
            pc.set_alpha(0.4)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)

        bp = ax.boxplot(data, positions=[i], notch=False, patch_artist=True,
                        widths=0.12, showfliers=False)
        for patch in bp['boxes']:
            patch.set_facecolor('white')
            patch.set_edgecolor('black')
            patch.set_linewidth(0.5)
        for whisker in bp['whiskers']:
            whisker.set(color='black', linestyle='-', linewidth=0.5)
        for cap in bp['caps']:
            cap.set(color='black', linestyle='-', linewidth=0.5)
        for median in bp['medians']:
            median.set(color='black', linestyle='-', linewidth=0.5)

        max_points = 200
        if len(data) > max_points:
            scatter_data = np.random.choice(data, size=max_points, replace=False)
        else:
            scatter_data = data
        x_jitter = np.random.uniform(low=i-0.18, high=i-0.05, size=len(scatter_data))
        ax.scatter(x_jitter, scatter_data, color=color, alpha=0.5, s=0.8, zorder=0)

        # Text annotation: median ± std
        median_value = np.mean(data)
        std_ = np.std(data)
        ax.text(i, np.max(data) * 1.01, f'{median_value:.4f} ± {std_:.4f}', 
                ha='center', va='bottom', color=color, fontsize=11)

    # ---------------------
    # Aesthetic adjustments
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Test Dataset 1', 'Test Dataset 2', 'Test Dataset 3'], rotation=30, ha='right', fontsize=12)
    ax.set_ylabel('Mean APD Error (s)', fontsize=14, color='black')
    ax.set_title(f"Mean APD Errors for {model_name}", fontsize=14, color='black')
    ax.set_ylim(0, 0.175)
    ax.set_yticks(np.arange(0, 0.175 + 0.001, 0.03))

    plt.tight_layout()
    plt.savefig(f"./results/{model_name}_apds_mean.pdf", format="pdf")

    # Print statistics
    # print("APDs")
    # print(f" Test 1 Mean:", sum_error_test1.mean(), "Std:", sum_error_test1.std(), "Max:", sum_error_test1.max())
    # print(f" Test 2 Mean:", sum_error_test2.mean(), "Std:", sum_error_test2.std(), "Max:", sum_error_test2.max())
    # print(f" Test 3 Mean:", sum_error_test3.mean(), "Std:", sum_error_test3.std(), "Max:", sum_error_test3.max())

    # print(f"Test Perc 1 Mean:", sum_error_test1_perc.mean(), "Std:", sum_error_test1_perc.std(), "Max:", sum_error_test1.max())
    # print(f"Test Perc 2 Mean:", sum_error_test2_perc.mean(), "Std:", sum_error_test2_perc.std(), "Max:", sum_error_test2.max())
    # print(f"Test Perc 3 Mean:", sum_error_test3_perc.mean(), "Std:", sum_error_test3_perc.std(), "Max:", sum_error_test3.max())


def MEA_plot(model_name,preds_list,intras_list):
    """
    Plots the Mean Absolute Error (MEA) between predicted and actual values for three test datasets.
    """

    color_test1 = "#1f77b4"  
    color_test2 = "#ff7f0e"  
    color_test3 = "#2ca02c"  

    preds1,preds2,preds3 = preds_list
    intra1,intra2,intra3 = intras_list

    sum_error_test1 = np.mean(np.abs(preds1-intra1), axis = 1)
    sum_error_test2 = np.mean(np.abs(preds2-intra2), axis = 1)
    sum_error_test3 = np.mean(np.abs(preds3-intra3), axis = 1)

    fig, ax = plt.subplots(figsize=(5, 5))

    for i, (data, color) in enumerate(zip([sum_error_test1, sum_error_test2, sum_error_test3], [color_test1, color_test2, color_test3])):
        parts = ax.violinplot(data, positions=[i], showmeans=False, showmedians=False,
                            showextrema=False, widths=0.3)
        for pc in parts['bodies']:
            verts = pc.get_paths()[0].vertices
            mean_x = np.mean(verts[:, 0])
            verts[:, 0] = np.clip(verts[:, 0], mean_x, np.inf)  
            pc.set_facecolor(color)
            pc.set_alpha(0.4)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)

        bp = ax.boxplot(data, positions=[i], notch=False, patch_artist=True,
                        widths=0.12, showfliers=False)
        for patch in bp['boxes']:
            patch.set_facecolor('white')
            patch.set_edgecolor('black')
            patch.set_linewidth(0.5)
        for whisker in bp['whiskers']:
            whisker.set(color='black', linestyle='-', linewidth=0.5)
        for cap in bp['caps']:
            cap.set(color='black', linestyle='-', linewidth=0.5)
        for median in bp['medians']:
            median.set(color='black', linestyle='-', linewidth=0.5)

        max_points = 200
        if len(data) > max_points:
            scatter_data = np.random.choice(data, size=max_points, replace=False)
        else:
            scatter_data = data
        x_jitter = np.random.uniform(low=i-0.18, high=i-0.05, size=len(scatter_data))
        ax.scatter(x_jitter, scatter_data, color=color, alpha=0.5, s=0.8, zorder=0)

        # ---------------------
        # Text annotation: median ± std
        median_value = np.mean(data)
        std_ = np.std(data)
        ax.text(i, np.max(data) * 1.01, f'{median_value:.4f} ± {std_:.4f}', 
                ha='center', va='bottom', color=color, fontsize=11)


    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Test Dataset 1', 'Test Dataset 2', 'Test Dataset 3'], rotation=30, ha='right', fontsize=12)
    ax.set_ylabel('MEA', fontsize=14, color='black')
    ax.set_title(f"MEA for {model_name}", fontsize=14, color='black')
    ax.set_ylim(0, 0.175)
    ax.set_yticks(np.arange(0, 0.175 + 0.001, 0.03))

    plt.tight_layout()
    plt.savefig(f"./results/{model_name}_MEA.pdf", format="pdf")

def act_vs_pred_plot_with_residual(x_values, y_values, label, color, plot_name):
    """
    Scatter plot of actual vs predicted values with correlation,
    plus a residual subplot below sharing x-axis.
    """

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    color = colors[label]
    r2_ = r2_score(x_values.flatten(), y_values.flatten())
    print(f"r2_score: {r2_:.4f}")

    # Calculate correlation coefficient
    correlation = np.corrcoef(x_values.flatten(), y_values.flatten())[0, 1]
    print(f"correlation:{correlation:.4f}")

    # Flatten values for calculations
    x_flat_full = x_values[:, 100:-100]
    y_flat_full = y_values[:, 100:-100]

    # Sample points for plotting
    x_list, y_list = [], []
    for x_row, y_row in zip(x_flat_full, y_flat_full):
        n_points = 1000 if label == 0 else 4000
        idx = np.random.choice(len(x_row), size=n_points, replace=False)
        x_list.append(x_row[idx])
        y_list.append(y_row[idx])
    
    x_flat = np.concatenate(x_list)
    y_flat = np.concatenate(y_list)

    # Compute residuals
    residuals = y_flat - x_flat

    # Create figure with GridSpec: 2 rows (scatter + residual), 1 col
    fig = plt.figure(figsize=(5, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.1)

    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_residual = fig.add_subplot(gs[1, 0], sharex=ax_scatter)

    # ---------------------
    # Main scatter plot
    ax_scatter.scatter(x_flat, y_flat, s=0.1, alpha=0.05, color=color, rasterized=True)
    ax_scatter.plot([-0.2, 1.2], [-0.2, 1.2], color='gray', linestyle='--', linewidth=1)

    ax_scatter.set_xlim(-0.2, 1.2)
    ax_scatter.set_ylim(-0.2, 1.2)
    ax_scatter.set_ylabel('Predicted Values')
    # ax_scatter.legend(fontsize=8)
    ax_scatter.text(0.05, 0.95, f'Corr = {correlation:.2f}\nR² = {r2_:.2f}', transform=ax_scatter.transAxes,
                    fontsize=12, ha='left', va='top', color='black')
    ax_scatter.set_title(f'Test Dataset {label+1}', fontsize=14, pad=10)

    # Remove x tick labels from scatter plot
    plt.setp(ax_scatter.get_xticklabels(), visible=False)

    # ---------------------
    # Residual plot
    ax_residual.scatter(x_flat, residuals, s=0.05, alpha=0.05, color=color, rasterized=True)
    ax_residual.axhline(0, color='gray', linestyle='--', linewidth=1)  
    ax_residual.set_ylabel('Residuals')
    ax_residual.set_xlabel('Actual Values')

    # Optional: adjust y limits to be symmetric for residuals
    ax_residual.set_ylim(-0.95, 0.95)
    
    plt.tight_layout()
    plt.savefig(f"./results/{plot_name}.pdf", format="pdf")

def get_sample_indices(data_length, total_samples, seed=42):
    """
    Generates sample indices evenly spaced across the sorted data.
    
    Args:
    - data_length: The total length of the data.
    - total_samples: The total number of samples to be drawn.
    - seed: An integer to seed the random number generator for repeatability.
    
    Returns:
    - indices: A list of selected indices.
    """
    if seed is not None:
        np.random.seed(seed)
    
    step_size = max(1, data_length // total_samples)
    indices = np.arange(0, data_length, step_size)[:total_samples]
    
    return indices

def plot_samples(plot_name, intras_normalized, preds, total_samples=10, sample_indices=[], test_number=0):
    """
    Plots a specified number of intras_normalized and preds data.
    
    Args:
    - name: Name of the plot.
    - intras_normalized: The normalized data to plot.
    - preds: The predicted data to plot.
    - total_samples: Total number of samples to plot.
    - save_dir: Directory to save the plot.
    """
    intras_normalized = np.array(intras_normalized)[5:-10]
    preds = np.array(preds)[5:-10]
    
    if len(sample_indices)==0:
        data_length = min(len(intras_normalized), len(preds))
        sample_indices = get_sample_indices(data_length, total_samples)

    # Find global y-axis limits
    y_min = -0.2
    y_max = 1.3
    
    
    fig, axes = plt.subplots(nrows=1, ncols=len(sample_indices), figsize=(5 * len(sample_indices), 7))
    for i, idx in enumerate(sample_indices):
        
        smooth_preds = savgol_filter(preds[idx], window_length=51, polyorder=3)
        smooth_intra = savgol_filter(intras_normalized[idx], window_length=51, polyorder=3)

        axes[i].plot(np.arange(len(smooth_intra)), smooth_intra, color='black',label="Ground Truth iAP")
        axes[i].plot(np.arange(len(smooth_preds)), smooth_preds, color='#1f77b4', linestyle='-', linewidth=2, label="Reconstructed iAP with PIAP-Mamba")
        
        axes[i].set_ylim([y_min, y_max])
        t_max = 1.6
        N = len(intras_normalized[idx])
        step = 2000  
        xticks_idx = np.arange(0, N+1, step)
        xticks_time = xticks_idx * t_max / (N - 1)
        axes[i].set_xticks(xticks_idx)
        axes[i].set_xticklabels([f"{t:.1f}" for t in xticks_time], rotation=0, fontsize=11)
        axes[i].set_xlabel("Time (s)")
    
    
    fig.suptitle("Comparison of Predicted and Ground Truth iAPs Across Multiple Samples", fontsize=30, x=0.5, y=0.98)
    fig.text(0.5, 0.88, f"Test Dataset {test_number+1}", fontsize=28, ha='center', va='top', color='black')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,loc='upper left', bbox_to_anchor=(0.02, 0.99), ncol=1, fontsize=28,frameon=False )
    fig.text(0.01, 0.4, "Normalized iAP", fontsize=28, rotation=90, va='center', ha='left', color='black')

    plt.tight_layout(rect=[0.015, 0, 1, 0.8])  
    plt.savefig(f"./results/{plot_name}.pdf", format="pdf")
    plt.close(fig)

   
    
    