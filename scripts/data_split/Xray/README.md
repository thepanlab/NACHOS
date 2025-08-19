# Project Files Overview

_Last updated: August 8, 2025_

This repository contains a set of Jupyter notebooks and CSVs for preparing and analyzing a chest X‑ray (or similar image) dataset. Tasks covered include copying/renaming files, creating randomized train/val/test folders, resizing/transforming images, and summarizing unique patients.

## Contents at a glance

- **img_subset_creation.ipynb** — Utilities to build a smaller working subset of images from larger datasets. Includes helpers to **copy** and **rename** matching files and to **sample** rows from CSVs to define a subset.
- **randomize_folders.ipynb** — Tools to **randomly split** images (with patient-aware logic) and **distribute** them into new directory structures (e.g., split_1 vs split_2). Logs what moved where for traceability.
- **copy_and_rename2.ipynb** — End‑to‑end routines to **find files across multiple datasets**, **filter** by rules, then **copy/rename/reshape** images into a unified structure; can export CSV logs.
- **transform_analysis.ipynb** — Image **inspection & processing** helpers (e.g., print shapes, compute folder sizes) and batch **processing** of images in a folder.
- **img_resizing.ipynb** — Focused **resizing pipeline** for standardizing image dimensions across a directory; (older and folded into some of the other stuff) 
- **modified_data_analysis.ipynb** — Analysis & plotting of the **datasets**, including counting occurrences 
- **unique_patient_counts.csv** — A two‑column table (`patientid`, `Dataset`) listing each patient occurrence used for downstream uniqueness checks.
- **unique_patient_summary.csv** — A three‑column table (`patientid`, `Dataset`, `UniquePatientCount`) with per‑patient rollups across datasets (for deduplication/leakage analysis).

## Notebooks — key functions & typical inputs/outputs

### `img_subset_creation.ipynb`
Key functions: `create_subset`, `copy_files`, `rename_and_copy_files`, `sample_dataframes`  


### `randomize_folders.ipynb`
Key functions: `random_split`, `random_split2`, `collect_image_sources`, `distribute_files_and_log[_patient]`, `rename_filename`  


### `copy_and_rename2.ipynb`
Key functions: `create_filter`, `create_exclusive_filter`, `parallel_file_search`, `process_and_relocate_images`, `reshape_and_save_image`, `export_dataframe_to_csv`  


### `transform_analysis.ipynb` & `img_resizing.ipynb`
Key functions: `process_images_in_folder`, `process_image`, `print_image_shapes`, `get_dir_size`, `convert_size`  


### `modified_data_analysis.ipynb`
Key functions: `count_occurrences`, `create_plot`, `create_aggregated_plot`, `plot_multicol_aggregated_findings`, `process_and_save_unique_patients_summary`, `process_dataset_folders_with_average_images`  


## Data files

- `unique_patient_counts.csv`  
  Columns: `patientid`, `Dataset` (222,128 rows). Each row is a patient occurrence in a dataset.

- `unique_patient_summary.csv`  
  Columns: `patientid`, `Dataset`, `UniquePatientCount` (222,132 rows). Aggregated per‑patient counts by dataset.


