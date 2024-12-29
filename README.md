
<p align="center">
  <img src="https://github.com/MatheusBecali/FLUO-SC/blob/main/githubBanner/Banner_FLUO-SC.png?raw=true" alt="Fluo-sc Banner"/>
</p>

# FLUO-SC-Code

This repository contains the code developed for the [**Fluorescence Images of Skin Lesions and Automated Diagnosis Using Convolutional Neural Networks**](https://doi.org/10.1016/j.pdpdt.2024.104462) paper. It provides tools and scripts for skin lesion analysis and classification using deep learning models.

If you encounter any issues, bugs, or have suggestions, feel free to open an issue or contact me directly. Contributions are always welcome!

---

## 🚀 Features

- Implementations of convolutional neural networks (CNNs) for fluorescence image analysis.
- Scripts for benchmarking experiments across multiple datasets.
- Easy-to-use integration with the **Sacred** experiment management library.

---

## 📋 Dependencies

The code is developed in Python and leverages the following major libraries:

- **PyTorch**: For deep learning model implementation.
- **Scikit-Learn**: For supplementary machine learning utilities.
- **Raug**: For a streamlined training pipeline.

### About Raug

**Raug** is a simple pipeline designed to train deep neural models using PyTorch. It was developed by André Pacheco and is available at [https://github.com/paaatcha/raug](https://github.com/paaatcha/raug). For more details on its usage and features, please refer to the [original repository](https://github.com/paaatcha/raug).

---

To set up the environment, ensure Python is installed on your system. Then, install the required dependencies using:

```bash
pip install -r requirements.txt
```

---

## 📂 Repository Structure

### Core Directories

- **`my_models`**: Contains CNN model implementations.
- **`benchmarks`**: Includes scripts for running experiments on each dataset.
- **`dataset`**: Contains the folders of the datasets used.
### Sacred Integration

The code uses **Sacred**, a framework for managing machine learning experiments. While you don't need prior knowledge of Sacred to run the code, it is a powerful tool that we recommend learning for better experiment organization.

If you prefer not to use Sacred, you can modify parameters directly in the script files.

---

## 📊 Datasets Links

The datasets used in this project are openly available for download on **Data Mendeley**:

- **[PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1)**: Dataset of skin lesion images, including metadata, for automated diagnosis.
- **[FLUO-SC](https://data.mendeley.com/datasets/s8n68jj678/1)**: Dataset containing fluorescence images of skin lesions.

Please refer to the respective dataset pages for more information on their content and usage.

--- 

## 🛠 Usage Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/MatheusBecali/FLUO-SC/
cd FLUO-SC
```

### Step 2: Configure Dataset Paths

For each benchmark, you need to set the dataset path in the corresponding scripts under the `benchmarks` folder.

### Step 3: Run Experiments

To execute an experiment, use the `pad.py` script with the following command:

```bash
python pad.py with _folder="$folder" _CandNC=True _model_name="$model_name" _method="$method" _type="$types" _dataset_name="$dataset_name" _pretrained="$pretrained"
```

#### Parameters:
1. **`_folder`**: Folder index (e.g., `1`, `2`, ..., `5`).
2. **`_model_name`**: Model architecture (e.g., `resnet-50`, `mobilenet`).
3. **`_method`**: Experimental method or configuration (e.g., `PUFES20`, `CLI`, `FLUO`).
4. **`_type`**: Classification task type (e.g., `CandNC`, `SixClass`).
5. **`_dataset_name`**: Dataset name (e.g., `PAD-UFES-20`, `FLUO-SC`).
6. **`_pretrained`**: Pretrained configuration (`True` for default, or a specific pretrained model like `PadUfesPreTrained`).

---

## 🧩 Tips and Recommendations

- **Sacred Experiment Management**: Sacred is a robust tool for organizing experiments, tracking configurations, and saving results. If you're unfamiliar, refer to the [Sacred documentation](https://sacred.readthedocs.io/en/stable/).
- **Custom Modifications**: You can modify scripts in the `benchmarks` folder to adapt to specific datasets or experimental setups.
- **Reproducibility**: Ensure your dataset paths and system configurations are correctly set to reproduce results.

---

## 📫 Contact

For any inquiries, please contact me at **matheusbecali@gmail.com** or open an issue in this repository.

---

## 📝 Credits and Acknowledgements

Some of the codes in this repository are adaptations from André Pacheco's repository, available at [https://github.com/paaatcha/my-thesis](https://github.com/paaatcha/my-thesis). 

These adaptations were essential for:
- Ensuring reproducibility of previously obtained results with the PAD-UFES-20 dataset.
- Modifying and extending the original implementations to incorporate the transfer learning approach.

We are grateful for the foundation provided by André Pacheco's work, which significantly contributed to the success of this project.

--- 

