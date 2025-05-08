import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import argparse
import traceback  # Make sure this import is included
from datetime import datetime

# Import the classes from your main code
from improved_code import (
    FederatedLearningSimulation,
    run_simulations,
    plot_metrics_comparison,
    create_comparison_table,
    analyze_privacy_impact,
    analyze_convergence,
    analyze_scalability
)
def run_single_experiment(config_name, config, results_dir):
    """Run a single experiment with error handling"""
    print(f"\n{'='*50}")
    print(f"Running simulation: {config_name}")
    print(f"{'='*50}")
    
    try:
        simulation = FederatedLearningSimulation(**config)
        simulation.simulate_federated_learning()
        return simulation
    except Exception as e:
        print(f"Error running simulation {config_name}: {e}")
        print(traceback.format_exc())
        
        # Log error to file
        error_file = os.path.join(results_dir, "error_log.txt")
        with open(error_file, 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Error in simulation: {config_name}\n")
            f.write(f"Configuration: {config}\n")
            f.write(f"Error: {e}\n")
            f.write(traceback.format_exc())
            f.write(f"{'='*50}\n")
        
        return None
def run_comprehensive_experiments(num_clients=50, num_rounds=10, output_dir='results',
                               dataset_filter='all', privacy_filter='all', distribution_filter='all'):
    """
    Run comprehensive experiments with all deduplication methods on all datasets
    
    Args:
        num_clients: Number of clients in federated learning
        num_rounds: Number of communication rounds
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{output_dir}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to {results_dir}")
    
    # Define all deduplication methods - order by priority
    deduplication_methods = [
        'CBD+EXGD', 'EXGD', 'HBD', 'CBD', 'CMBD', 'OFCD',
        'FedProx', 'Clustering', 'TimeWeighted', 'Divergence', 'Hybrid'
    ]
    # deduplication_methods = ['CBD+EXGD', 'EXGD']  # Start with key methods
    
    # Add other methods if needed for a comprehensive test
    if dataset_filter == 'MNIST' or dataset_filter == 'all':
        deduplication_methods.extend(['HBD', 'CBD', 'CMBD', 'OFCD'])
    
    all_datasets = ['MNIST', 'FMNIST', 'CIFAR10']
    datasets = all_datasets if dataset_filter == 'all' else [dataset_filter.upper()]

    all_distributions = ['iid', 'non-iid']
    distributions = all_distributions if distribution_filter == 'all' else [distribution_filter]

    all_privacy_methods = ['none', 'dp', 'secure_agg']
    privacy_methods = all_privacy_methods if privacy_filter == 'all' else [privacy_filter]
    
    # Dictionary to store all results
    all_results = {}
    
    # Build all configurations
    all_configs = {}
    for dataset in datasets:
        for method in deduplication_methods:
            for privacy in privacy_methods:
                for dist in distributions:
                    # Skip some combinations to reduce experiment time
                    if method not in ['CBD+EXGD', 'EXGD'] and privacy != 'none':
                        continue
                        
                    # Build config name
                    config_name = f"{method}_{dataset}"
                    if privacy != 'none':
                        config_name += f"_{privacy}"
                    if dist == 'non-iid':
                        config_name += "_NonIID"

                    # Build configuration dictionary
                    all_configs[config_name] = {
                        'num_clients': num_clients,
                        'num_rounds': num_rounds,
                        'dataset_name': dataset,
                        'deduplication_method': method,
                        'privacy_method': privacy,
                        'data_distribution': dist
                    }

                    # Add non_iid_alpha only when relevant
                    if dist == 'non-iid':
                        all_configs[config_name]['non_iid_alpha'] = 0.5

                    print(f"→ Configured: {config_name} "
                        f"(method={method}, privacy={privacy}, dist={dist})")
    
    # Run experiments one by one instead of all at once
    for config_name, config in all_configs.items():
        result = run_single_experiment(config_name, config, results_dir)
        
        if result is not None:
            all_results[config_name] = result
            
            # Save intermediate results
            try:
                if len(all_results) > 1:
                    # Generate comparison plots for completed experiments
                    interim_dir = os.path.join(results_dir, "interim_results")
                    os.makedirs(interim_dir, exist_ok=True)
                    plot_metrics_comparison(all_results, output_dir=interim_dir)
                    
                    # Create comparison table
                    comparison_table = create_comparison_table(all_results)
                    table_path = os.path.join(interim_dir, "comparison_table.csv")
                    comparison_table.to_csv(table_path, index=False)
            except Exception as e:
                print(f"Error saving intermediate results: {e}")
    
    # Generate final results if we have any successful experiments
    if all_results:
        try:
            # Create overall comparison across all experiments
            print("\n\nGenerating overall comparison...")
            
            # Generate comprehensive comparison plots
            overall_output_dir = f"{results_dir}/Overall"
            os.makedirs(overall_output_dir, exist_ok=True)
            plot_metrics_comparison(all_results, output_dir=overall_output_dir)
            
            # Create summary table
            summary_data = {
                'Configuration': [],
                'Dataset': [],
                'Method': [],
                'Privacy': [],
                'Distribution': [],
                'Accuracy (%)': [],
                'Deduplication Rate (%)': [],
                'Storage Efficiency (%)': [],
                'System Latency (ms)': []
            }
            
            for config_name, result in all_results.items():
                # Parse configuration name to extract components
                parts = config_name.split('_')
                
                # Default values
                method = parts[0]
                dataset = parts[1] if len(parts) > 1 else "Unknown"
                privacy = "none"
                distribution = "iid"
                
                # Check for specific markers in the config name
                if "NonIID" in config_name:
                    distribution = "non-iid"
                if "CBD+EXGD" in config_name:
                    method = "CBD+EXGD"
                    if len(parts) > 2:
                        dataset = parts[2]
                if "dp" in config_name:
                    privacy = "dp"
                if "secure_agg" in config_name or "SecAgg" in config_name:
                    privacy = "secure_agg"
                
                # Get average metrics from the last rounds
                last_n = min(5, len(result.metrics['learning_accuracy']))
                if last_n > 0:
                    avg_accuracy = np.mean(result.metrics['learning_accuracy'][-last_n:])
                    avg_dedup_rate = np.mean(result.metrics['deduplication_rate'][-last_n:])
                    avg_storage = np.mean(result.metrics['storage_efficiency'][-last_n:])
                    avg_latency = np.mean(result.metrics['system_latency'][-last_n:])
                    
                    # Add to summary data
                    summary_data['Configuration'].append(config_name)
                    summary_data['Dataset'].append(dataset)
                    summary_data['Method'].append(method)
                    summary_data['Privacy'].append(privacy)
                    summary_data['Distribution'].append(distribution)
                    summary_data['Accuracy (%)'].append(avg_accuracy)
                    summary_data['Deduplication Rate (%)'].append(avg_dedup_rate)
                    summary_data['Storage Efficiency (%)'].append(avg_storage)
                    summary_data['System Latency (ms)'].append(avg_latency)
            
            # Create summary DataFrame
            if summary_data['Configuration']:
                summary_df = pd.DataFrame(summary_data)
                
                # Save summary table
                summary_path = f"{results_dir}/comprehensive_summary.csv"
                summary_df.to_csv(summary_path, index=False)
                print(f"Comprehensive summary saved to {summary_path}")
                
                # Create pivot tables if we have enough data
                if len(summary_df) > 1:
                    # Method vs Dataset (Accuracy)
                    try:
                        accuracy_pivot = pd.pivot_table(
                            summary_df, 
                            values='Accuracy (%)', 
                            index='Method', 
                            columns='Dataset',
                            aggfunc='mean'
                        )
                        accuracy_pivot.to_csv(f"{results_dir}/accuracy_by_method_dataset.csv")
                    except Exception as e:
                        print(f"Error creating accuracy pivot: {e}")
        except Exception as e:
            print(f"Error generating final results: {e}")
            print(traceback.format_exc())
    
    print(f"\nAll experiments completed. Results saved to {results_dir}")
    return all_results, results_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run comprehensive federated learning experiments')
    parser.add_argument('--clients', type=int, default=100, help='Number of clients')
    parser.add_argument('--rounds', type=int, default=10, help='Number of communication rounds')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                    help='Dataset to run (MNIST, FMNIST, CIFAR10, or all)')
    parser.add_argument('--privacy', type=str, default='none',
                        help='Privacy method to use (none, dp, secure_agg, or all)')
    parser.add_argument('--distribution', type=str, default='iid',
                        help='Data distribution to use (iid, non-iid, or all)')
    parser.add_argument('--quick', action='store_true', help='Run only essential methods for quick testing')
    parser.add_argument('--deduplication', type=str, default='CBD+EXGD',
                    choices=['CBD+EXGD', 'EXGD', 'HBD', 'CBD', 'CMBD', 'OFCD', 
                             'FedProx', 'Clustering', 'TimeWeighted', 'Divergence', 'Hybrid'],
                    help='Deduplication method to use')
    args = parser.parse_args()
    
    print(f"Starting experiments with {args.clients} clients and {args.rounds} rounds")
    
    try:
        all_results, results_dir = run_comprehensive_experiments(
            num_clients=args.clients,
            num_rounds=args.rounds,
            output_dir=args.output,
            dataset_filter=args.dataset,
            privacy_filter=args.privacy,
            distribution_filter=args.distribution
        )
        
        print(f"Experiments completed successfully. Results available in {results_dir}")
    except Exception as e:
        print(f"Fatal error running experiments: {e}")
        print(traceback.format_exc())