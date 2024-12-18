# -*- coding: utf-8 -*-
"""
Original Author: André Pacheco
Email: pacheco.comp@gmail.com

Script to prepare data for training, validation, and testing on the PAD-UFES-20 dataset.
"""

import sys
import os
import pandas as pd
from raug.utils.loader import split_k_folder_csv, label_categorical_to_number

# Include the path to the deep-tasks folder
sys.path.insert(0, '../../.')

# Import the RAUG_PATH constant
from constants import RAUG_PATH
sys.path.insert(0, RAUG_PATH)

# Get the current working directory
BASE_PATH = os.getcwd()

#########################################
"""
Modified by: Matheus Becali Rocha
Email: matheusbecali@gmail.com
"""

# Define the column names for clinical data
clin_ = ["Lesion", "Class", "ImageName", "path"]

# Define the imaging method: Use "CLI" (Clinical) or "FLUO" (Fluorescence)
m = "CLI"

# Alternatively, use a list of methods:
# method_ = ["FLUO", "CLI"]

# Define the type of classification:
# Options include "CandNC", "CarcinomasVsOthers", "CarcinomasVsCeratoses",
# "CarcinomaVsACK", "MelVsSek", "MelVsNev", "MelVsNevSek"
_type = 'CandNC'

# Alternatively, use a list of classification types:
# _types = ["CandNC", "CarcinomasVsOthers", "CarcinomasVsCeratoses",
#           "CarcinomaVsACK", "MelVsSek", "MelVsNev", "MelVsNevSek"]

# Loop through multiple types if using a list:
# for _type in _types:

# Loop through multiple methods if using a list:
# for m in method_:

# Read the CSV file for the selected method and type
data_csv = pd.read_csv(os.path.join(BASE_PATH, f"data_{m}_{_type}.csv"))

# Split the dataset into 6 folders for cross-validation
data = split_k_folder_csv(
    data_csv,
    "Lesion",
    save_path=None,
    k_folder=6,
    seed_number=8
)

# Separate the test set (folder 6)
data_test = data[data['folder'] == 6]

# Separate the training set (folders 1-5)
data_train = data[data['folder'] != 6]

# Save the test dataset to a CSV file
data_test.to_csv(
    os.path.join(BASE_PATH, f"{m}_sc_{_type}_test.csv"),
    index=False
)

# Convert categorical labels in the test set to numeric labels
label_categorical_to_number(
    os.path.join(BASE_PATH, f"{m}_sc_{_type}_test.csv"),
    "Lesion",
    col_target_number="diagnostic_number",
    save_path=os.path.join(BASE_PATH, f"{m}_sc_{_type}_test.csv")
)

# Reset the index of the training dataset
data_train = data_train.reset_index(drop=True)

# Split the training dataset into 5 folders for further cross-validation
data_train = split_k_folder_csv(
    data_train,
    "Lesion",
    save_path=None,
    k_folder=5,
    seed_number=8
)

# Convert categorical labels in the training set to numeric labels
label_categorical_to_number(
    data_train,
    "Lesion",
    col_target_number="diagnostic_number",
    save_path=os.path.join(BASE_PATH, f"{m}_sc_{_type}_folders.csv")
)

#########################################
