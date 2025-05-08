import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Set the style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10

# Custom color palettes
method_colors = {
    'CBD+EXGD': '#1f77b4',  # blue
    'EXGD': '#ff7f0e',      # orange
    'HBD': '#2ca02c',       # green
    'CBD': '#d62728',       # red
    'CMBD': '#9467bd',      # purple
    'OFCD': '#8c564b',      # brown
    'FedProx': '#e377c2',   # pink
    'Clustering': '#7f7f7f', # gray
    'TimeWeighted': '#bcbd22', # olive
    'Divergence': '#17becf', # cyan
    'Hybrid': '#ff9896'     # light red
}

dataset_colors = {
    'MNIST': '#1f77b4',     # blue
    'FMNIST': '#ff7f0e',    # orange
    'CIFAR10': '#2ca02c'    # green
}

dataset_markers = {
    'MNIST': 'o',
    'FMNIST': 's',
    'CIFAR10': '^'
}

# Define which metrics are better when higher or lower
higher_better = {
    'storage_efficiency': True,
    'bandwidth_usage': False,
    'learning_accuracy': True,
    'deduplication_rate': True,
    'model_training_time': False,
    'system_latency': False,
    'transactions_per_block': True,
    'max_nodes': True,
    'loss_values': False
}

# Human-readable metric names for plots
metric_names = {
    'storage_efficiency': 'Storage Efficiency (%)',
    'bandwidth_usage': 'Bandwidth Usage',
    'learning_accuracy': 'Learning Accuracy (%)',
    'deduplication_rate': 'Deduplication Rate (%)',
    'model_training_time': 'Model Training Time (s)',
    'system_latency': 'System Latency (ms)',
    'transactions_per_block': 'Transactions per Block',
    'max_nodes': 'Max Nodes',
    'loss_values': 'Loss Values'
}

def load_data():
    """
    Load data from CSV files
    Returns:
        accuracy_df, summary_df, comparison_df
    """
    # Load standard files if needed
    try:
        accuracy_df = pd.read_csv('results_20250422_142155/accuracy_by_method_dataset.csv')
    except:
        accuracy_df = None
        
    try:
        summary_df = pd.read_csv('results_20250422_142155/comprehensive_summary.csv')
    except:
        summary_df = None
    
    # Load comparison table - the main focus
    comparison_df = pd.read_csv('results_20250422_142155/interim_results/comparison_table.csv')
    
    return accuracy_df, summary_df, comparison_df

def transform_comparison_data(comparison_df):
    """
    Transform the comparison dataframe to make it easier to work with
    Returns:
        long_df: A long-format dataframe with Method, Dataset, Metric, and Value columns
    """
    # Melt the dataframe to get a long format
    long_df = pd.melt(comparison_df, id_vars=['Metric'], var_name='Method_Dataset', value_name='Value')
    
    # Extract Method and Dataset from the Method_Dataset column
    long_df[['Method', 'Dataset']] = long_df['Method_Dataset'].str.split('_', n=1, expand=True)
    
    return long_df
def plot_accuracy_comparison(accuracy_df):
    """
    Create a grouped bar chart showing accuracy by method for each dataset
    """
    # Melt the dataframe to long format for easier plotting
    melted_df = pd.melt(accuracy_df, id_vars=['Method'], var_name='Dataset', value_name='Accuracy')
    
    # Sort by accuracy within each dataset
    melted_df = melted_df.sort_values(['Dataset', 'Accuracy'], ascending=[True, False])
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Set the width of each bar and the spacing between groups
    bar_width = 0.25
    datasets = melted_df['Dataset'].unique()
    methods = accuracy_df['Method'].unique()
    
    # Calculate positions for each bar
    positions = np.arange(len(methods))
    
    # Plot each dataset as a group of bars
    for i, dataset in enumerate(datasets):
        dataset_data = melted_df[melted_df['Dataset'] == dataset]
        offset = (i - 1) * bar_width
        bars = ax.bar(positions + offset, dataset_data['Accuracy'], 
                      bar_width, label=dataset, color=dataset_colors[dataset],
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add data labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Customize the plot
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Method Accuracy Comparison Across Datasets')
    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend(title='Dataset')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add horizontal lines for reference
    for y in [20, 40, 60, 80]:
        ax.axhline(y=y, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('method_accuracy_comparison.png', dpi=300, bbox_inches='tight')

def plot_heatmap(comparison_df):
    """
    Create a heatmap showing all methods, metrics, and datasets
    """
    # Transform data for heatmap creation
    methods = ['CBD+EXGD', 'EXGD', 'HBD', 'CBD', 'CMBD', 'OFCD', 'FedProx', 
             'Clustering', 'TimeWeighted', 'Divergence', 'Hybrid']
    datasets = ['MNIST', 'FMNIST', 'CIFAR10']
    metrics = comparison_df['Metric'].unique()
    
    # Create a separate heatmap for each dataset
    for dataset in datasets:
        # Create a matrix for the heatmap: methods as rows, metrics as columns
        heatmap_data = np.zeros((len(methods), len(metrics)))
        
        # Fill the matrix with values
        for i, method in enumerate(methods):
            for j, metric in enumerate(metrics):
                col_name = f"{method}_{dataset}"
                value = comparison_df.loc[comparison_df['Metric'] == metric, col_name].values[0]
                heatmap_data[i, j] = value
        
        # For metrics where lower is better, invert the values for the heatmap
        # to ensure that "good" values are consistently shown as "hot" colors
        for j, metric in enumerate(metrics):
            if not higher_better[metric]:
                # Find min and max for normalization
                col_min = np.min(heatmap_data[:, j])
                col_max = np.max(heatmap_data[:, j])
                # Invert but keep the same range
                heatmap_data[:, j] = col_max - (heatmap_data[:, j] - col_min)
        
        # Create the heatmap
        plt.figure(figsize=(14, 10))
        ax = sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=[metric_names[m] for m in metrics],
            yticklabels=methods,
            linewidths=0.5,
            cbar_kws={'label': 'Performance (higher is better)'}
        )
        
        # Add arrows or markers to indicate the best method for each metric
        for j, metric in enumerate(metrics):
            col = comparison_df.loc[comparison_df['Metric'] == metric, [f"{method}_{dataset}" for method in methods]].values[0]
            best_idx = np.argmax(col) if higher_better[metric] else np.argmin(col)
            ax.add_patch(plt.Rectangle((j, best_idx), 1, 1, fill=False, edgecolor='red', lw=2))
        
        # Set title and labels
        plt.title(f'Performance Comparison for {dataset} Dataset', fontsize=18)
        plt.xlabel('Metrics', fontsize=14)
        plt.ylabel('Methods', fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        plt.xticks(rotation=45, ha='right')
        plt.savefig(f'heatmap_{dataset}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_radar_charts(comparison_df):
    """
    Create radar charts comparing methods for each dataset
    """
    # Get unique methods, datasets, and metrics
    methods = ['CBD+EXGD', 'EXGD', 'HBD', 'CBD', 'CMBD', 'OFCD', 'FedProx', 
               'Clustering', 'TimeWeighted', 'Divergence', 'Hybrid']
    datasets = ['MNIST', 'FMNIST', 'CIFAR10']
    metrics = comparison_df['Metric'].unique()
    
    # Normalize the data across methods for each metric and dataset
    normalized_data = comparison_df.copy()
    
    for dataset in datasets:
        for metric in metrics:
            # Get all values for this metric and dataset
            values = [normalized_data.loc[normalized_data['Metric'] == metric, f"{method}_{dataset}"].values[0] for method in methods]
            
            # Calculate min and max
            min_val = min(values)
            max_val = max(values)
            
            # Skip normalization if all values are the same
            if max_val == min_val:
                continue
                
            # Normalize to 0-1 range (considering if higher is better or worse)
            for method in methods:
                col = f"{method}_{dataset}"
                val = normalized_data.loc[normalized_data['Metric'] == metric, col].values[0]
                
                if higher_better[metric]:
                    norm_val = (val - min_val) / (max_val - min_val)
                else:
                    norm_val = 1 - ((val - min_val) / (max_val - min_val))
                    
                normalized_data.loc[normalized_data['Metric'] == metric, col] = norm_val
    
    # Create radar charts for each dataset, showing top 5 methods
    for dataset in datasets:
        # Calculate average performance across metrics for each method
        avg_performance = {}
        for method in methods:
            col = f"{method}_{dataset}"
            avg_performance[method] = normalized_data[col].mean()
        
        # Sort methods by average performance and get top 5
        top_methods = sorted(avg_performance.items(), key=lambda x: x[1], reverse=True)[:5]
        top_methods = [m[0] for m in top_methods]
        
        # Set up the radar chart
        metrics_for_radar = metrics
        N = len(metrics_for_radar)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Add metric labels around the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric_names[m] for m in metrics_for_radar], fontsize=9)
        
        # Add concentric circles for scale
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.set_ylim(0, 1)
        
        # Plot data for each method
        for method in top_methods:
            col = f"{method}_{dataset}"
            values = [normalized_data.loc[normalized_data['Metric'] == metric, col].values[0] for metric in metrics_for_radar]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, label=method, color=method_colors[method])
            ax.fill(angles, values, color=method_colors[method], alpha=0.1)
        
        # Add legend and title
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title(f'Top Methods Performance on {dataset} Dataset', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(f'radar_chart_{dataset}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_metric_comparison(comparison_df, highlight_methods=None):
    """
    Create bar charts comparing all methods across datasets for each metric
    """
    # Get unique methods, datasets, and metrics
    methods = ['CBD+EXGD', 'EXGD', 'HBD', 'CBD', 'CMBD', 'OFCD', 'FedProx', 
               'Clustering', 'TimeWeighted', 'Divergence', 'Hybrid']
    datasets = ['MNIST', 'FMNIST', 'CIFAR10']
    metrics = comparison_df['Metric'].unique()
    
    # If highlight_methods is not provided, use the top method for each metric
    if highlight_methods is None:
        highlight_methods = {}
        for metric in metrics:
            best_method = None
            best_value = float('-inf') if higher_better[metric] else float('inf')
            
            for method in methods:
                # Calculate average across datasets
                avg_value = np.mean([
                    comparison_df.loc[comparison_df['Metric'] == metric, f"{method}_{dataset}"].values[0]
                    for dataset in datasets
                ])
                
                if higher_better[metric] and avg_value > best_value:
                    best_value = avg_value
                    best_method = method
                elif not higher_better[metric] and avg_value < best_value:
                    best_value = avg_value
                    best_method = method
                    
            highlight_methods[metric] = best_method
    
    # Create a grouped bar chart for each metric
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Set up bar positions
        x = np.arange(len(methods))
        bar_width = 0.25
        opacity = 0.8
        
        # Plot bars for each dataset
        bars = []
        for i, dataset in enumerate(datasets):
            values = [comparison_df.loc[comparison_df['Metric'] == metric, f"{method}_{dataset}"].values[0] for method in methods]
            
            # Highlight the best method if needed
            colors = [method_colors[method] if method == highlight_methods.get(metric) else method_colors[method] for method in methods]
            alphas = [1.0 if method == highlight_methods.get(metric) else 0.7 for method in methods]
            
            bar = ax.bar(x + (i - 1) * bar_width, values, bar_width,
                         alpha=opacity, color=dataset_colors[dataset],
                         label=dataset, edgecolor='black', linewidth=1)
            bars.append(bar)
            
            # Add value labels
            for j, rect in enumerate(bar):
                height = rect.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, rotation=90)
        
        # Add labels and legend
        ax.set_xlabel('Methods')
        ax.set_ylabel(metric_names[metric])
        ax.set_title(f'Comparison of {metric_names[metric]} Across Methods and Datasets')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        
        # If lower is better, add note
        if not higher_better[metric]:
            ax.text(0.02, 0.02, 'Note: Lower values are better for this metric',
                    transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        ax.legend()
        
        # Highlight the best method in the legend
        if metric in highlight_methods:
            highlight_patch = mpatches.Patch(color='gold', alpha=0.3, 
                                            label=f'Best: {highlight_methods[metric]}')
            handles, labels = ax.get_legend_handles_labels()
            handles.append(highlight_patch)
            ax.legend(handles=handles)
        
        plt.tight_layout()
        plt.savefig(f'metric_comparison_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_method_profiles(comparison_df):
    """
    Create detailed profile plots for each method showing performance across all metrics and datasets
    """
    # Get unique methods, datasets, and metrics
    methods = ['CBD+EXGD', 'EXGD', 'HBD', 'CBD', 'CMBD', 'OFCD', 'FedProx', 
               'Clustering', 'TimeWeighted', 'Divergence', 'Hybrid']
    datasets = ['MNIST', 'FMNIST', 'CIFAR10']
    metrics = comparison_df['Metric'].unique()
    
    # Create a profile visualization for each method
    for method in methods:
        # Create a grid of subplots (3 rows for datasets, columns for metrics)
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, len(metrics), figure=fig)
        
        # Normalize metric values across all methods for fair comparison
        normalized_data = {}
        
        for metric in metrics:
            # Get all values for this metric across all methods and datasets
            all_values = []
            for m in methods:
                for d in datasets:
                    col = f"{m}_{d}"
                    all_values.append(comparison_df.loc[comparison_df['Metric'] == metric, col].values[0])
            
            min_val = min(all_values)
            max_val = max(all_values)
            
            # Skip if all values are the same
            if max_val == min_val:
                normalized_data[metric] = {f"{m}_{d}": 0.5 for m in methods for d in datasets}
                continue
            
            # Normalize based on whether higher or lower is better
            normalized_data[metric] = {}
            for m in methods:
                for d in datasets:
                    col = f"{m}_{d}"
                    val = comparison_df.loc[comparison_df['Metric'] == metric, col].values[0]
                    
                    if higher_better[metric]:
                        norm_val = (val - min_val) / (max_val - min_val)
                    else:
                        norm_val = 1 - ((val - min_val) / (max_val - min_val))
                        
                    normalized_data[metric][col] = norm_val
        
        # Plot each metric for each dataset
        for i, dataset in enumerate(datasets):
            for j, metric in enumerate(metrics):
                ax = fig.add_subplot(gs[i, j])
                
                # Get normalized values for all methods for this metric and dataset
                values = [normalized_data[metric][f"{m}_{dataset}"] for m in methods]
                
                # Get actual value for the current method
                actual_value = comparison_df.loc[comparison_df['Metric'] == metric, f"{method}_{dataset}"].values[0]
                
                # Highlight where the current method ranks
                method_idx = methods.index(method)
                
                # Create bar chart
                bars = ax.bar(range(len(methods)), values, alpha=0.6, color='lightgray')
                
                # Highlight the current method
                bars[method_idx].set_color(method_colors[method])
                bars[method_idx].set_alpha(1.0)
                
                # Add the actual value for the current method
                ax.text(method_idx, values[method_idx], f'{actual_value:.2f}', 
                        ha='center', va='bottom', fontsize=8)
                
                # Set titles and labels
                if i == 0:
                    ax.set_title(metric_names[metric], fontsize=10, pad=5)
                
                if j == 0:
                    ax.set_ylabel(dataset, fontsize=12)
                
                # Remove most of the clutter
                ax.set_xticks([])
                ax.set_yticks([0, 0.5, 1])
                ax.set_yticklabels(['0', '0.5', '1'], fontsize=8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Add a note about what's better
                if i == 2 and j == 0:
                    if higher_better[metric]:
                        ax.text(0, -0.3, 'Higher is better', fontsize=8, transform=ax.transAxes)
                    else:
                        ax.text(0, -0.3, 'Lower is better', fontsize=8, transform=ax.transAxes)
        
        plt.suptitle(f'Performance Profile: {method}', fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(f'method_profile_{method}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_composite_scores(comparison_df):
    """
    Calculate and visualize composite performance scores 
    by normalizing and weighting metrics
    """
    # Get unique methods, datasets, and metrics
    methods = ['CBD+EXGD', 'EXGD', 'HBD', 'CBD', 'CMBD', 'OFCD', 'FedProx', 
               'Clustering', 'TimeWeighted', 'Divergence', 'Hybrid']
    datasets = ['MNIST', 'FMNIST', 'CIFAR10']
    metrics = comparison_df['Metric'].unique()
    
    # Define metric weights (could be adjusted based on importance)
    metric_weights = {
        'storage_efficiency': 1.0,
        'bandwidth_usage': 1.0,
        'learning_accuracy': 1.5,  # Give more weight to accuracy
        'deduplication_rate': 1.0,
        'model_training_time': 0.8,
        'system_latency': 1.0,
        'transactions_per_block': 0.8,
        'max_nodes': 0.5,
        'loss_values': 1.2
    }
    
    # Normalize and calculate weighted scores
    normalized_data = {}
    for metric in metrics:
        # Get all values for this metric across all methods and datasets
        all_values = []
        for method in methods:
            for dataset in datasets:
                col = f"{method}_{dataset}"
                all_values.append(comparison_df.loc[comparison_df['Metric'] == metric, col].values[0])
        
        min_val = min(all_values)
        max_val = max(all_values)
        
        # Skip if all values are the same
        if max_val == min_val:
            normalized_data[metric] = {f"{method}_{dataset}": 0.5 for method in methods for dataset in datasets}
            continue
        
        # Normalize based on whether higher or lower is better
        normalized_data[metric] = {}
        for method in methods:
            for dataset in datasets:
                col = f"{method}_{dataset}"
                val = comparison_df.loc[comparison_df['Metric'] == metric, col].values[0]
                
                if higher_better[metric]:
                    norm_val = (val - min_val) / (max_val - min_val)
                else:
                    norm_val = 1 - ((val - min_val) / (max_val - min_val))
                    
                normalized_data[metric][col] = norm_val * metric_weights[metric]
    
    # Calculate composite scores for each method and dataset
    composite_scores = {}
    for method in methods:
        for dataset in datasets:
            key = f"{method}_{dataset}"
            scores = [normalized_data[metric][key] for metric in metrics]
            # Calculate weighted average
            total_weight = sum(metric_weights.values())
            composite_scores[key] = sum(scores) / total_weight
    
    # Create a dataframe for easier plotting
    score_data = []
    for method in methods:
        for dataset in datasets:
            key = f"{method}_{dataset}"
            score_data.append({
                'Method': method,
                'Dataset': dataset,
                'Score': composite_scores[key]
            })
    
    score_df = pd.DataFrame(score_data)
    
    # Plot composite scores by method and dataset
    plt.figure(figsize=(14, 8))
    
    # Create grouped bar chart
    ax = sns.barplot(
        data=score_df,
        x='Method',
        y='Score',
        hue='Dataset',
        palette=dataset_colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=1
    )
    
    # Add value labels
    for i, container in enumerate(ax.containers):
        ax.bar_label(container, fmt='%.2f', fontsize=8)
    
    # Add a horizontal line at the mean score
    mean_score = score_df['Score'].mean()
    plt.axhline(y=mean_score, color='gray', linestyle='--', alpha=0.7)
    plt.text(0, mean_score + 0.01, f'Average: {mean_score:.2f}', fontsize=10)
    
    # Customize the plot
    plt.title('Composite Performance Score by Method and Dataset', fontsize=16)
    plt.xlabel('Method')
    plt.ylabel('Composite Score (higher is better)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Dataset')
    
    plt.tight_layout()
    plt.savefig('composite_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a heatmap of composite scores
    score_pivot = score_df.pivot(index='Method', columns='Dataset', values='Score')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        score_pivot, 
        annot=True, 
        fmt='.2f', 
        cmap='YlGnBu',
        linewidths=0.5,
        cbar_kws={'label': 'Composite Score'}
    )
    
    plt.title('Composite Performance Score Heatmap', fontsize=16)
    plt.tight_layout()
    plt.savefig('composite_scores_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return score_df

def plot_dataset_comparison(comparison_df):
    """
    Create visualizations comparing datasets performance across methods
    """
    # Get unique methods and metrics
    methods = ['CBD+EXGD', 'EXGD', 'HBD', 'CBD', 'CMBD', 'OFCD', 'FedProx', 
               'Clustering', 'TimeWeighted', 'Divergence', 'Hybrid']
    datasets = ['MNIST', 'FMNIST', 'CIFAR10']
    metrics = comparison_df['Metric'].unique()
    
    # Create a comparison for accuracy vs efficiency across datasets
    plt.figure(figsize=(12, 8))
    
    # Extract accuracy and efficiency data
    for method in methods:
        accuracies = []
        efficiencies = []
        latencies = []
        
        for dataset in datasets:
            # Get accuracy, efficiency, and latency
            accuracy = comparison_df.loc[
                comparison_df['Metric'] == 'learning_accuracy', 
                f"{method}_{dataset}"
            ].values[0]
            
            efficiency = comparison_df.loc[
                comparison_df['Metric'] == 'storage_efficiency', 
                f"{method}_{dataset}"
            ].values[0]
            
            latency = comparison_df.loc[
                comparison_df['Metric'] == 'system_latency', 
                f"{method}_{dataset}"
            ].values[0]
            
            accuracies.append(accuracy)
            efficiencies.append(efficiency)
            latencies.append(latency)
        
        # Scale latency for marker size (smaller latency = larger marker)
        max_latency = max(latencies)
        sizes = [100 * (1 - (lat / max_latency * 0.7)) for lat in latencies]
        
        # Plot with different markers for each dataset
        for i, dataset in enumerate(datasets):
            plt.scatter(
                efficiencies[i], 
                accuracies[i],
                s=sizes[i],
                color=method_colors[method],
                marker=dataset_markers[dataset],
                alpha=0.7,
                edgecolor='black',
                linewidth=1,
                label=f"{method} ({dataset})" if i == 0 else ""
            )
            
            # Add method label
            plt.annotate(
                method,
                (efficiencies[i], accuracies[i]),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=8,
                ha='left'
            )
    
    # Add dataset annotations to show which marker is which dataset
    legend_elements = [
        plt.Line2D([0], [0], marker=marker, color='black', label=dataset,
                  markerfacecolor='gray', markersize=8, linestyle='None')
        for dataset, marker in dataset_markers.items()
    ]
    
    plt.legend(handles=legend_elements, title='Datasets', loc='upper left')
    
    # Add a second legend for methods
    method_patches = [
        mpatches.Patch(color=color, label=method)
        for method, color in method_colors.items()
    ]
    
    second_legend = plt.legend(
        handles=method_patches, 
        title='Methods',
        loc='upper right',
        bbox_to_anchor=(1.15, 1)
    )
    
    plt.gca().add_artist(second_legend)
    
    # Customize the plot
    plt.xlabel('Storage Efficiency (%)')
    plt.ylabel('Learning Accuracy (%)')
    plt.title('Accuracy vs. Efficiency Across Datasets and Methods', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add note about marker size
    plt.figtext(0.5, 0.01, 'Note: Marker size indicates lower latency (larger = faster)',
               ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig('accuracy_vs_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_tradeoff_matrix(summary_df):
    """
    Create a matrix of scatter plots showing relationships between different metrics
    """
    # Select metrics to plot
    metrics = ['Accuracy (%)', 'Deduplication Rate (%)', 'Storage Efficiency (%)', 'System Latency (ms)']
    
    # Create a simple pairplot instead of PairGrid to avoid complexity
    g = sns.pairplot(
        summary_df,
        vars=metrics,
        hue='Dataset',
        palette=dataset_colors,
        height=3,
        plot_kws={'s': 60, 'alpha': 0.7, 'edgecolor': 'black', 'linewidth': 0.5},
        diag_kws={'color': ".3", 'alpha': 0.5}
    )
    
    # Customize the diagonals to be histograms
    for i, var in enumerate(metrics):
        ax = g.axes[i, i]
        ax.clear()
        sns.histplot(summary_df, x=var, ax=ax, color=".3", alpha=0.5)
    
    # Add more customization to make the plot clearer
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            ax = g.axes[i, j]
            
            # Add grid lines
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # If this is a scatter plot (not on the diagonal)
            if i != j:
                # For the bottom row, add more descriptive x labels
                if i == len(metrics) - 1:
                    ax.set_xlabel(metrics[j], fontsize=10)
                else:
                    ax.set_xlabel('')
                
                # For the leftmost column, add more descriptive y labels
                if j == 0:
                    ax.set_ylabel(metrics[i], fontsize=10)
                else:
                    ax.set_ylabel('')
    
    # Add a title
    g.fig.suptitle('Relationships Between Performance Metrics', y=1.02, fontsize=16)
    
    # Adjust the legend
    g.add_legend(title='Dataset', frameon=True)
    
    # Tighten the layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('metric_relationship_matrix.png', dpi=300, bbox_inches='tight')
    
def plot_dataset_performance(summary_df):
    """
    Create a faceted plot showing each dataset with accuracy vs latency, 
    and bubble size representing storage efficiency
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    datasets = ['MNIST', 'FMNIST', 'CIFAR10']
    
    for i, dataset in enumerate(datasets):
        dataset_df = summary_df[summary_df['Dataset'] == dataset]
        
        # Create scatter plot
        for method in dataset_df['Method'].unique():
            method_data = dataset_df[dataset_df['Method'] == method]
            
            # Size based on storage efficiency, scaled for visibility
            size = method_data['Storage Efficiency (%)'].values[0] * 5
            
            # Color based on method
            color = method_colors[method]
            
            axes[i].scatter(
                method_data['System Latency (ms)'], 
                method_data['Accuracy (%)'],
                s=size,
                color=color,
                alpha=0.7,
                edgecolor='black',
                linewidth=1,
                label=method
            )
        
        # Add method labels to points
        for j, row in dataset_df.iterrows():
            axes[i].annotate(
                row['Method'],
                xy=(row['System Latency (ms)'], row['Accuracy (%)']),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=8,
                ha='left'
            )
        
        # Customize subplot
        axes[i].set_title(f'{dataset} Dataset')
        axes[i].set_xlabel('System Latency (ms)')
        if i == 0:
            axes[i].set_ylabel('Accuracy (%)')
        
        # Set axis limits
        axes[i].set_xlim(30, 50)
        
        # Add grid
        axes[i].grid(True, linestyle='--', alpha=0.6)
    
    # Add a common legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                         markersize=8, label=method) for method, color in method_colors.items()]
    
    fig.legend(handles=handles, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.12))
    
    plt.suptitle('Performance Tradeoff by Dataset: Accuracy vs. Latency vs. Storage Efficiency', fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('dataset_performance_tradeoff.png', dpi=300, bbox_inches='tight')
def plot_method_progression(summary_df, metrics=['Accuracy (%)', 'System Latency (ms)']):
    """
    Create a visualization showing how methods have improved based on their 
    characteristics (simplified visualization of what might be a time series)
    """
    # We'll use a scatter plot with arrows to show "progression" of methods
    # Use the first metric on x-axis and second metric on y-axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group methods into "families" for visualization
    method_families = {
        'Basic': ['CBD', 'EXGD'],
        'Improved': ['HBD', 'CMBD', 'OFCD'],
        'Advanced': ['CBD+EXGD', 'FedProx', 'Clustering', 'TimeWeighted', 'Divergence', 'Hybrid']
    }
    
    family_colors = {
        'Basic': '#ff9999',
        'Improved': '#99ff99',
        'Advanced': '#9999ff'
    }
    
    # Calculate average metrics for each method
    method_metrics = summary_df.groupby('Method')[metrics].mean().reset_index()
    
    # Assign family to each method
    method_metrics['Family'] = method_metrics['Method'].apply(
        lambda m: next((family for family, methods in method_families.items() if m in methods), 'Other')
    )
    
    # Plot points
    for family, color in family_colors.items():
        family_data = method_metrics[method_metrics['Family'] == family]
        
        # Create scatter plot for this family
        ax.scatter(
            family_data[metrics[0]], 
            family_data[metrics[1]],
            label=family,
            color=color,
            s=100,
            alpha=0.7,
            edgecolor='black',
            linewidth=1
        )
        
        # Add method labels
        for _, row in family_data.iterrows():
            ax.annotate(
                row['Method'],
                xy=(row[metrics[0]], row[metrics[1]]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10
            )
    
    # Customize the plot
    ax.set_xlabel(metrics[0])
    ax.set_ylabel(metrics[1])
    ax.set_title(f'Method Evolution: {metrics[0]} vs {metrics[1]}')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # If the second metric is latency, indicate lower is better
    if 'Latency' in metrics[1]:
        ax.text(
            0.02, 0.02, 
            'Lower latency is better ?',
            transform=ax.transAxes,
            fontsize=10,
            ha='left'
        )
        
        # Draw an arrow pointing to lower latency
        ax.annotate(
            '', 
            xy=(0.15, 0.02), 
            xytext=(0.02, 0.02),
            arrowprops=dict(arrowstyle='->', color='black'),
            transform=ax.transAxes
        )
    
    # Add annotations showing evolutionary connections
    # Basic to Improved
    ax.annotate(
        '',
        xy=(method_metrics[method_metrics['Method'] == 'HBD'][metrics[0]].values[0],
            method_metrics[method_metrics['Method'] == 'HBD'][metrics[1]].values[0]),
        xytext=(method_metrics[method_metrics['Method'] == 'CBD'][metrics[0]].values[0],
                method_metrics[method_metrics['Method'] == 'CBD'][metrics[1]].values[0]),
        arrowprops=dict(arrowstyle='->', color='gray', lw=1, alpha=0.6)
    )
    
    # Basic to Advanced
    ax.annotate(
        '',
        xy=(method_metrics[method_metrics['Method'] == 'CBD+EXGD'][metrics[0]].values[0],
            method_metrics[method_metrics['Method'] == 'CBD+EXGD'][metrics[1]].values[0]),
        xytext=(method_metrics[method_metrics['Method'] == 'EXGD'][metrics[0]].values[0],
                method_metrics[method_metrics['Method'] == 'EXGD'][metrics[1]].values[0]),
        arrowprops=dict(arrowstyle='->', color='gray', lw=1, alpha=0.6)
    )
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'method_evolution_{metrics[0].split()[0].lower()}_vs_{metrics[1].split()[0].lower()}.png', dpi=300, bbox_inches='tight')

def plot_top_methods_detailed(summary_df, top_n=5):
    """
    Create a detailed comparison of the top N methods across all metrics
    """
    # Calculate average performance across all datasets for each method
    method_avg = summary_df.groupby('Method').agg({
        'Accuracy (%)': 'mean',
        'Deduplication Rate (%)': 'mean',
        'Storage Efficiency (%)': 'mean',
        'System Latency (ms)': 'mean'
    }).reset_index()
    
    # For latency, lower is better, so we'll invert it for ranking
    method_avg['Latency_Inverted'] = 100 - method_avg['System Latency (ms)']
    
    # Normalize each metric to a 0-1 scale
    metrics = ['Accuracy (%)', 'Deduplication Rate (%)', 'Storage Efficiency (%)', 'Latency_Inverted']
    for metric in metrics:
        min_val = method_avg[metric].min()
        max_val = method_avg[metric].max()
        method_avg[f'{metric}_norm'] = (method_avg[metric] - min_val) / (max_val - min_val)
    
    # Calculate combined score (simple average of normalized metrics)
    method_avg['Combined_Score'] = method_avg[[f'{m}_norm' for m in metrics]].mean(axis=1)
    
    # Sort by combined score and select top N methods
    top_methods = method_avg.sort_values('Combined_Score', ascending=False).head(top_n)['Method'].tolist()
    
    # Filter the data to include only top methods
    top_data = summary_df[summary_df['Method'].isin(top_methods)]
    
    # Create a figure with 4 subplots (one for each metric)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    metrics_to_plot = ['Accuracy (%)', 'Deduplication Rate (%)', 'Storage Efficiency (%)', 'System Latency (ms)']
    titles = ['Accuracy (%)', 'Deduplication Rate (%)', 'Storage Efficiency (%)', 'System Latency (ms) - Lower is better']
    
    # Calculate the minimum latency for reference
    min_latency = summary_df['System Latency (ms)'].min()
    
    for i, (ax, metric, title) in enumerate(zip(axes.flatten(), metrics_to_plot, titles)):
        # For each dataset, plot the metric for each method
        for dataset in ['MNIST', 'FMNIST', 'CIFAR10']:
            dataset_data = top_data[top_data['Dataset'] == dataset]
            
            # Sort the data by the metric (descending, except for latency)
            ascending = metric == 'System Latency (ms)'
            dataset_data = dataset_data.sort_values(metric, ascending=ascending)
            
            # Get the position for each method
            methods_ordered = dataset_data['Method'].tolist()
            positions = [methods_ordered.index(m) for m in dataset_data['Method']]
            
            # Plot the data
            ax.scatter(
                positions, 
                dataset_data[metric],
                label=dataset if i == 0 else "",
                color=dataset_colors[dataset],
                marker=dataset_markers[dataset],
                s=100,
                alpha=0.7,
                edgecolor='black',
                linewidth=1
            )
            
            # Connect points for the same dataset
            ax.plot(
                positions,
                dataset_data[metric],
                color=dataset_colors[dataset],
                alpha=0.4,
                linestyle='-'
            )
        
        # Customize the subplot
        ax.set_title(title)
        ax.set_xticks(range(len(top_methods)))
        ax.set_xticklabels([m for m in dataset_data['Method']], rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # For system latency, indicate that lower is better
        if metric == 'System Latency (ms)':
            ax.axhline(y=min_latency, color='green', linestyle='--', alpha=0.5)
            ax.text(0, min_latency, 'Better', color='green', va='bottom', ha='left')
    
    # Add a legend to the first subplot
    axes[0, 0].legend(title='Dataset')
    
    plt.suptitle('Detailed Comparison of Top Methods Across All Metrics', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('top_methods_detailed.png', dpi=300, bbox_inches='tight')
    

def plot_method_comparison_by_dataset(summary_df, metric='Accuracy (%)'):
    """
    Create a grouped bar chart comparing methods across datasets for a specific metric
    """
    # Pivot the data to get methods as rows and datasets as columns
    pivot_df = summary_df.pivot_table(
        index='Method', 
        columns='Dataset', 
        values=metric,
        aggfunc='mean'
    ).reset_index()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set width of bars
    bar_width = 0.25
    positions = np.arange(len(pivot_df))
    
    # Plot bars for each dataset
    for i, dataset in enumerate(['MNIST', 'FMNIST', 'CIFAR10']):
        offset = (i - 1) * bar_width
        bars = ax.bar(
            positions + offset, 
            pivot_df[dataset], 
            bar_width, 
            label=dataset, 
            color=dataset_colors[dataset],
            alpha=0.8,
            edgecolor='black',
            linewidth=1
        )
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=8,
                rotation=90
            )
    
    # Customize the plot
    ax.set_ylabel(metric)
    ax.set_title(f'Method Comparison by Dataset: {metric}')
    ax.set_xticks(positions)
    ax.set_xticklabels(pivot_df['Method'], rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='Dataset')
    
    plt.tight_layout()
    plt.savefig(f'method_comparison_{metric.split()[0].lower()}.png', dpi=300, bbox_inches='tight')
    
def create_all_visualizations():
    """
    Generate all visualizations
    """
    # Load data
    accuracy_df, summary_df, comparison_df = load_data()
    
    # Transform comparison data for easier use
    long_df = transform_comparison_data(comparison_df)
    # Plot method comparison for different metrics
    for metric in ['Accuracy (%)', 'Deduplication Rate (%)', 'Storage Efficiency (%)', 'System Latency (ms)']:
        plot_method_comparison_by_dataset(summary_df, metric)
    # Generate visualizations
    plot_heatmap(comparison_df)
    plot_radar_charts(comparison_df)
    plot_metric_comparison(comparison_df)
    plot_method_profiles(comparison_df)
    score_df = plot_composite_scores(comparison_df)
    plot_dataset_comparison(comparison_df)
    plot_dataset_performance(summary_df)
    # Plot method progression
    plot_method_progression(summary_df)
    plot_method_progression(summary_df, ['Accuracy (%)', 'Storage Efficiency (%)'])
    plot_accuracy_comparison(accuracy_df)
    plot_tradeoff_matrix(summary_df)
    plot_top_methods_detailed(summary_df)
if __name__ == "__main__":
    create_all_visualizations()