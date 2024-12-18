#!/bin/bash

# Function to run the experiment
# Arguments:
#   1. folder: The folder index (e.g., 1, 2, ..., 5).
#   2. model_name: The name of the model to be used (e.g., resnet-50, mobilenet).
#   3. method: The method or configuration for the experiment (e.g., PUFES20, CLI, FLUO).
#   4. types: The type of classification or task (e.g., CandNC, SixClass).
#   5. dataset_name: The name of the dataset (e.g., PAD-UFES-20, FLUO-SC).
#   6. pretrained: Pretrained configuration (True or specific pretrained model name).

run_experiment() {
  local folder=$1
  local model_name=$2
  local method=$3
  local types=$4
  local dataset_name=$5
  local pretrained=$6

  # Execute the Python script with the provided arguments
  python pad.py with _folder="$folder" _CandNC=True _model_name="$model_name" _method="$method" _type="$types" _dataset_name="$dataset_name" _pretrained="$pretrained"
}

# Define the models, types, methods, datasets, and pretrained configurations
# Uncomment and modify the variables below to run experiments for different settings

# models=("resnet-50" "mobilenet" "densenet-121" "regnety32")
# types=("CandNC" "CarcinomaVsOthers" "CarcinomasVsCeratoses" "CarcinomaVsACK" "MelVsSek" "MelVsNev" "MelVsNevSek")
# methods=("PUFES20" "CLI" "FLUO")
# datasets=("PAD-UFES-20" "FLUO-SC")
# pretraineds=(True "PadUfesPreTrained" "CLIPreTrained")

# Currently used settings
models=("resnet-50")  # List of models
types=("CandNC")      # Task type
methods=("PUFES20")   # Experiment method
datasets=("PAD-UFES-20")  # Dataset name
pretraineds=(True)    # Pretrained configuration

# Run experiments for each configuration
for type in "${types[@]}"; do
    for model in "${models[@]}"; do
        # Experiments for PUFES20 with PAD-UFES-20 dataset
        for folder in {1..5}; do
            run_experiment "$folder" "$model" "PUFES20" "$type" "PAD-UFES-20" True
        done

        # Uncomment the blocks below to include additional experiments:

        # Experiments for CLI using PadUfesPreTrained
        # for folder in {1..5}; do
        #     run_experiment "$folder" "$model" "CLI" "$type" "FLUO-SC" "PadUfesPreTrained"
        # done

        # Experiments for FLUO with FLUO-SC using CLIPreTrained
        # for folder in {1..5}; do
        #     run_experiment "$folder" "$model" "FLUO" "$type" "FLUO-SC" "CLIPreTrained"
        # done

        # Experiments for CLI
        # for folder in {1..5}; do
        #     run_experiment "$folder" "$model" "CLI" "$type" "FLUO-SC" True
        # done

        # Experiments for FLUO
        # for folder in {1..5}; do
        #     run_experiment "$folder" "$model" "FLUO" "$type" "FLUO-SC" True
        # done
    done
done
