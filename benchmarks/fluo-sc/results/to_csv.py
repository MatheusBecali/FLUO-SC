"""
Autor: Matheus Becali Rocha
Email: matheusbecali@gmail.com

Description:
This script is designed to extracts metrics from 
a text file and save to CSV
"""

import os
import csv
import numpy as np

# Function to extract metrics from a 'metrics.txt' file
def extract_metrics(file_path):
    """
    Extracts key performance metrics from a text file.

    Parameters:
        file_path (str): Path to the file containing the metrics.

    Returns:
        dict: A dictionary containing extracted metrics (Loss, Accuracy, Balanced Accuracy, Recall, Precision, F1-Score).
    """
    metrics = {}
    with open(file_path, 'r') as file:
        for line in file:
            if 'Loss:' in line:
                metrics['Loss'] = float(line.split(':')[-1].strip())
            elif 'Accuracy:' in line:
                metrics['Accuracy'] = float(line.split(':')[-1].strip())
            elif 'Balanced accuracy:' in line:
                metrics['Balanced Accuracy'] = float(line.split(':')[-1].strip())
            elif 'Recall:' in line:
                metrics['Recall'] = float(line.split(':')[-1].strip())
            elif 'Precision:' in line:
                metrics['Precision'] = float(line.split(':')[-1].strip())
            elif 'F1-Score:' in line:
                metrics['F1-Score'] = float(line.split(':')[-1].strip())
    return metrics

# Configurable parameters for the experiment
_model_names = ("resnet-50", "mobilenet", "densenet-121", "regnety32")
_methods = ('PUFES20', 'PadUfesPreTrained_CP', 'CPPreTrained_FLR')
_CandNC = True

# Base path to results folders
base_folder = 'YOUR_PATH/fluo-sc/results/'

# Optimizer name
_optimizer = "SGD"
# Alternatively: _optimizer = "CP_Adam"

# Metrics to include in the CSV
metrics_header = ['Fold', 'Loss', 'Accuracy', 'Balanced Accuracy', 'Recall', 'Precision', 'F1-Score']

# Experiment type
_type = 'MelVsSek'

for _model_name in _model_names:
    # Define the output CSV file path
    output_csv = f'{_type}/{_optimizer}/{_type}_{_model_name}_all_metrics.csv'

    for _method in _methods:
        # Collect metrics from each fold
        all_metrics = []
        for _fold in range(1, 6):
            # Construct the folder path for the current fold
            folder = os.path.join(
                base_folder, 
                f'{_type}/{_optimizer}/{_method}_CandNC_{_CandNC}_{_type}_{_model_name}_fold_{_fold}'
            )
            metric_file = os.path.join(folder, 'test_pred', 'metrics.txt')
            
            # Check if the metrics file exists
            if os.path.exists(metric_file):
                # Extract metrics from the file
                metrics = extract_metrics(metric_file)
                metrics['Fold'] = f'{_method}_{_type}_{_model_name}_fold_{_fold}'
                all_metrics.append(metrics)
            else:
                print(f"File {metric_file} not found.")

        if all_metrics:
            # Compute averages and standard deviations
            metrics_np = {key: [] for key in metrics_header[1:]}  # Exclude 'Fold'
            for metrics in all_metrics:
                for key in metrics_header[1:]:
                    metrics_np[key].append(metrics[key])

            averages = {key: np.mean(metrics_np[key]) for key in metrics_np}
            std_devs = {key: np.std(metrics_np[key]) for key in metrics_np}

        # Save metrics to the CSV file
        file_exists = os.path.isfile(output_csv)

        with open(output_csv, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metrics_header)

            # Write header if file does not exist
            if not file_exists:
                writer.writeheader()
                
            # Write individual metrics for each fold
            for metrics in all_metrics:
                writer.writerow(metrics)

            # Combine averages and standard deviations into a single line
            combined_metrics = {'Fold': 'Average_and_Std'}
            for key in metrics_header[1:]:
                combined_metrics[key] = f"${averages[key]:.4f} \\pm {std_devs[key]:.4f}$"

            writer.writerow(combined_metrics)

        print(f"Metrics saved to {output_csv}")
