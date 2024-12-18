# -*- coding: utf-8 -*-
"""
Autor: Matheus Becali Rocha
Email: matheusbecali@gmail.com

Description:
This script generates CSV files containing metadata for images of lesions
stored in subdirectories. Each CSV file corresponds to a subdirectory and
includes details such as lesion name, class (C or NC), image name, and
the full image path.
"""

import os
import csv

def _createCSV():
    """
    Generates CSV files for each folder in the current working directory,
    excluding Python scripts and existing CSV files. The CSV files will
    contain metadata for images stored within subdirectories.

    The structure of the generated CSV includes:
        - Lesion: The name of the lesion (derived from the folder name).
        - Class: 'C' for certain lesions, 'NC' for others based on a predefined rule.
        - ImageName: The name of the image file.
        - Path: The full path to the image file.
    """
    # Get the current working directory
    source_folder = os.getcwd()

    print("Start creating a CSV file for all lesions and respective folders")

    # List all files and folders in the current directory
    method_type = [f for f in os.listdir(source_folder)]

    # Iterate through each item in the directory
    for method in method_type:
        # Skip Python scripts and existing CSV files
        if method.endswith('.py') or method.endswith('.csv'):
            continue

        # Open a new CSV file for writing
        with open(os.path.join(source_folder, f'data_{method}.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            # Write the header row
            writer.writerow(['Lesion', 'Class', 'ImageName', 'Path'])

            # Construct the path to the current folder
            lesions_name_path = os.path.join(source_folder, method)

            # Iterate through the subfolders representing lesion names
            for lesions_name in os.listdir(lesions_name_path):
                # Determine the class of the lesion
                if lesions_name in ['ACK', 'SEK', 'NEV']:
                    lesion_class = 'NC'  # Non-cancer lesions
                else:
                    lesion_class = 'C'   # Cancer lesions

                # Construct the path to the folder containing images
                lesions_imgs_path = os.path.join(lesions_name_path, lesions_name)

                # Iterate through each image file in the folder
                for img_names in os.listdir(lesions_imgs_path):
                    # Construct the full path to the image
                    img_path = os.path.join(lesions_imgs_path, img_names)

                    # Write the lesion metadata to the CSV
                    writer.writerow([lesions_name, lesion_class, img_names, img_path])

if __name__ == "__main__":
    _createCSV()
