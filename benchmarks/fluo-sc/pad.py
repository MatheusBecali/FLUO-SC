# -*- coding: utf-8 -*-
"""
Original Autor: André Pacheco
Email: pacheco.comp@gmail.com

-------------------------------------

Modified by: Matheus Becali Rocha
Email: matheusbecali@gmail.com
"""

import sys
import os

# Include paths for dependencies
sys.path.insert(0, '../../')  # Path to deep-tasks folder
sys.path.insert(0, '/mnt/hdd/matheusbecali/fluo-sc-code/my_models')  # Path to custom models folder

# The path to Raug. You may find it here: https://github.com/paaatcha/raug
RAUG_PATH = "/mnt/hdd/matheusbecali/raug"

BASE_PATH = os.path.abspath(os.path.join(os.getcwd(), "../.."))
DIR_PATH = os.getcwd()

sys.path.insert(0, RAUG_PATH)

import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import time

from raug.loader import get_data_loader
from raug.train import fit_model
from raug.eval import test_model
from raug.utils.loader import get_labels_frequency
from my_model import set_model
from aug_pad import ImgTrainTransform, ImgEvalTransform
from sacred import Experiment
from sacred.observers import FileStorageObserver


# Starting sacred experiment
ex = Experiment()

@ex.config
def cnfg():
    """
    Configuration function for Sacred. Defines default parameters for the experiment.
    """
    # Dataset and training configuration
    _folder = 1
    _CandNC = True
    _dataset_name = 'PAD-UFES-20'
    _method = "UV"
    _type = "NvsM"
    _base_path = f"{BASE_PATH}/dataset/{_dataset_name}/"

    # Define dataset-specific paths
    if _dataset_name == 'PAD-UFES-20':
        try:
            _csv_path_train = os.path.join(_base_path, f"pad-ufes-20_parsed_{_type}_folders.csv")
            _csv_path_test = os.path.join(_base_path, f"pad-ufes-20_parsed_{_type}_test.csv")
        except:
            raise Exception(f"The type {_type} of {_dataset_name} is not available!")
        _imgs_folder_train = os.path.join(_base_path, "imgs")
    else:
        _base_path = f"{BASE_PATH}/dataset/{_dataset_name}/{_type}/"
        try:
            _csv_path_train = os.path.join(_base_path, f"{_method}_sc_{_type}_folders.csv")
            _csv_path_test = os.path.join(_base_path, f"{_method}_sc_{_type}_test.csv")
        except:
            raise Exception(f"The type {_type} of {_dataset_name} is not available!")
        _imgs_folder_train = os.path.join(_base_path, f"{_method}")

    # Additional training parameters

    _use_meta_data = False # Unused parameters (originally designed for clinical data experiments with padding).
    _neurons_reducer_block = 0 # Unused parameters (originally designed for clinical data experiments with padding).
    _comb_method = None # Unused parameters (originally designed for clinical data experiments with padding).
    _comb_config = None # Unused parameters (originally designed for clinical data experiments with padding).

    _batch_size = 30
    _epochs = 2
    _best_metric = "loss"
    _pretrained = True
    _lr_init = 0.0001
    _sched_factor = 0.1
    _sched_min_lr = 1e-6
    _sched_patience = 10
    _early_stop = 15
    _metric_early_stop = None
    _weights = "frequency"
    _model_name = 'resnet-50'

    # Save path configuration
    if _pretrained == "PadUfesPreTrained":
        _save_folder = f"results/{_type}/{_pretrained}_{_method}_CandNC_{_CandNC}_{_type}_{_model_name}_fold_{_folder}"
    elif _pretrained == "CLIPreTrained":
        _save_folder = f"results/{_type}/{_pretrained}_{_method}_CandNC_{_CandNC}_{_type}_{_model_name}_fold_{_folder}"
    else:
        _save_folder = f"results/{_type}/{_method}_CandNC_{_CandNC}_{_type}_{_model_name}_fold_{_folder}"

    # This is used to configure the sacred storage observer. In brief, it says to sacred to save its stuffs in
    # _save_folder. You don't need to worry about that.
    SACRED_OBSERVER = FileStorageObserver(_save_folder)
    ex.observers.append(SACRED_OBSERVER)

@ex.automain
def main (_folder, _CandNC, _type, _csv_path_train, _imgs_folder_train, _lr_init, _sched_factor, _sched_min_lr, _sched_patience,
          _batch_size, _epochs, _early_stop, _weights, _model_name, _pretrained, _save_folder, _csv_path_test,
          _best_metric, _neurons_reducer_block, _comb_method, _comb_config, _metric_early_stop):
    """
    Main function for training, validation, and testing.
    """

    # Set paths for saving metrics
    _metric_options = {
        'save_all_path': os.path.join(_save_folder, "best_metrics"),
        'pred_name_scores': 'predictions_best_test.csv',
        'normalize_conf_matrix': True}
    _checkpoint_best = os.path.join(_save_folder, 'best-checkpoint/best-checkpoint.pth')

    # Loading the csv file
    csv_all_folders = pd.read_csv(_csv_path_train)

    print("-" * 50)
    print("- Loading validation data...")
    val_csv_folder = csv_all_folders[ (csv_all_folders['folder'] == _folder) ]
    train_csv_folder = csv_all_folders[ csv_all_folders['folder'] != _folder ]

    # Loading validation data
    try:
        val_imgs_id = val_csv_folder['ImageName'].values
        val_imgs_path = val_csv_folder['path'].values
    except:
        val_imgs_id = val_csv_folder['img_id'].values
        val_imgs_path = ["{}/{}".format(_imgs_folder_train, img_id) for img_id in val_imgs_id]

    val_labels = val_csv_folder['diagnostic_number'].values

    print("-- No metadata")
    val_meta_data = None
    val_data_loader = get_data_loader(
        val_imgs_path,
        val_labels,
        val_meta_data,
        transform=ImgEvalTransform(),
        batch_size=_batch_size,
        shuf=True,
        num_workers=16,
        pin_memory=True
    )
    print("-- Validation partition loaded with {} images".format(len(val_data_loader)*_batch_size))

    print("- Loading training data...")
    try:
        train_imgs_id = train_csv_folder['ImageName'].values
        train_imgs_path = train_csv_folder['path'].values
    except:
        train_imgs_id = train_csv_folder['img_id'].values
        train_imgs_path = ["{}/{}".format(_imgs_folder_train, img_id) for img_id in train_imgs_id]
    train_labels = train_csv_folder['diagnostic_number'].values

    print("-- No metadata")
    train_meta_data = None
    train_data_loader = get_data_loader(
        train_imgs_path,
        train_labels,
        train_meta_data,
        transform=ImgTrainTransform(),
        batch_size=_batch_size,
        shuf=True,
        num_workers=16,
        pin_memory=True
    )
    print("-- Training partition loaded with {} images".format(len(train_data_loader)*_batch_size))

    print("-"*50)
    ####################################################################################################################

    # Set class weights
    try:
        if _CandNC:
            ser_lab_freq = get_labels_frequency(train_csv_folder, "Class", "ImageName")
        else:
            ser_lab_freq = get_labels_frequency(train_csv_folder, "Lesion", "ImageName")
    except:
        ser_lab_freq = get_labels_frequency(train_csv_folder, "diagnostic", "img_id")

    _labels_name = ser_lab_freq.index.values
    _freq = ser_lab_freq.values
    print(ser_lab_freq)

    ####################################################################################################################
    # Set up the model
    print("- Loading", _model_name)

    # Initialize the model with the specified configuration
    model = set_model(
        _model_name,
        len(_labels_name),  # Number of output classes
        neurons_reducer_block=_neurons_reducer_block,  # Optional layer for reducing neurons
        comb_method=_comb_method,  # Combination method, e.g., concat, metanet, etc.
        comb_config=_comb_config,  # Additional combination configurations
        pretrained=_pretrained  # Whether to use a pretrained model
    )
    ####################################################################################################################

    # Load pretrained weights if specified
    if _pretrained == 'PadUfesPreTrained':
        print("- Loading - Pre-Trained PAD-UFES-20")
        model.load_state_dict(
            torch.load(f"{DIR_PATH}/results/{_type}/PUFES20_CandNC_{_CandNC}_{_type}_{_model_name}_fold_{_folder}/best-checkpoint/best-checkpoint.pth",
                    map_location=torch.device('cuda')
            )["model_state_dict"]
        )
    elif _pretrained == 'CLIPreTrained':
        print("- Loading - Pre-Trained CLI")
        model.load_state_dict(
            torch.load(f"{DIR_PATH}/results/{_type}/PadUfesPreTrained_CLI_CandNC_{_CandNC}_{_type}_{_model_name}_fold_{_folder}/best-checkpoint/best-checkpoint.pth",
                    map_location=torch.device('cuda')
            )["model_state_dict"]
        )
    else:
        pass
    ####################################################################################################################

    # Configure class weights based on label frequencies, if required
    if _weights == 'frequency':
        _weights = (_freq.sum() / _freq).round(3)

    # Define the loss function with optional class weights
    loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(_weights).cuda())
    # Configure the optimizer
    # Example (commented): SGD optimizer
    # optimizer = optim.SGD(model.parameters(), lr=_lr_init, momentum=0.9, weight_decay=0.001)

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=_lr_init, weight_decay=0.001)

    # Define the learning rate scheduler
    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=_sched_factor,  # Factor by which the learning rate will be reduced
        min_lr=_sched_min_lr,  # Minimum learning rate
        patience=_sched_patience  # Number of epochs with no improvement before reducing LR
    )
    ####################################################################################################################

    print("- Starting the training phase...")
    print("-" * 50)

    # Train the model
    fit_model(
        model, train_data_loader, val_data_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=_epochs,
        epochs_early_stop=_early_stop,
        save_folder=_save_folder,
        initial_model=None,
        metric_early_stop=_metric_early_stop,
        device=None,  # Automatically selects GPU or CPU
        schedule_lr=scheduler_lr,
        config_bot=None,
        model_name="CNN",
        resume_train=False,  # Start training from scratch
        history_plot=True,  # Plot training history
        val_metrics=["balanced_accuracy"],  # Validation metrics
        best_metric=_best_metric  # Metric to monitor for early stopping
    )
    ####################################################################################################################

    # Testing the validation partition
    print("- Evaluating the validation partition...")
    test_model(
        model, val_data_loader,
        checkpoint_path=_checkpoint_best,  # Path to the best model checkpoint
        loss_fn=loss_fn,
        save_pred=True,  # Save predictions
        partition_name='eval',  # Label for this evaluation partition
        metrics_to_comp='all',  # Compute all metrics
        class_names=_labels_name,  # Names of the classes
        metrics_options=_metric_options,  # Additional metric configurations
        apply_softmax=True,  # Apply softmax to model outputs
        verbose=False
    )
    ####################################################################################################################

    ####################################################################################################################

    # Load test data
    print("- Loading test data...")
    csv_test = pd.read_csv(_csv_path_test)

    # Handle dataset column structure differences
    try:
        test_imgs_id = csv_test['ImageName'].values
        test_imgs_path = csv_test['path'].values
    except:
        test_imgs_id = csv_test['img_id'].values
        test_imgs_path = ["{}/{}".format(_imgs_folder_train, img_id) for img_id in test_imgs_id]

    # Extract test labels
    test_labels = csv_test['diagnostic_number'].values
    test_meta_data = None  # Placeholder for metadata
    print("-- No metadata")

    # Update metric options for the test partition
    _metric_options = {
        'save_all_path': os.path.join(_save_folder, "test_pred"),  # Save test predictions here
        'pred_name_scores': 'predictions.csv',  # Filename for prediction scores
        'normalize_conf_matrix': True  # Normalize confusion matrix
    }

    # Create a data loader for the test set
    test_data_loader = get_data_loader(
        test_imgs_path, test_labels, test_meta_data,
        transform=ImgEvalTransform(),
        batch_size=_batch_size,
        shuf=False,  # Do not shuffle test data
        num_workers=16,
        pin_memory=True
    )
    print("-" * 50)

    # Evaluate the model on the test partition
    print("\n- Evaluating the test partition...")
    test_model(
        model, test_data_loader,
        checkpoint_path=None,  # Use the current model without a specific checkpoint
        metrics_to_comp="all",  # Compute all metrics
        class_names=_labels_name,  # Names of the classes
        metrics_options=_metric_options,  # Additional metric configurations
        save_pred=True,  # Save predictions
        verbose=False
    )
    ####################################################################################################################
