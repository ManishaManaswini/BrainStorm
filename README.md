# EEG Data Processing Pipeline

This repository contains an EEG data processing pipeline designed to load, preprocess, extract features, measure creativity, and visualize EEG data.

## Table of Contents

- [Overview](#overview)
- [Components](#components)
  - [Loading Data](#loading-data)
  - [Preprocessing](#preprocessing)
  - [Feature Extraction](#feature-extraction)
  - [Creativity Measurement](#creativity-measurement)
  - [Comparing Datasets](#comparing-datasets)
  - [Visualization](#visualization)
- [Getting Started](#getting-started)

## Overview

The pipeline offers a comprehensive suite of tools to analyze EEG data, especially in the context of creativity measurements. It integrates various modules to handle data from loading to visualization.

## Components

### Loading Data

Module: `load_data.py`

- **Class**: `LoadData`
  - Load and visualize EEG data.
  - Convert data to MNE format.
  - Save and load datasets.

### Preprocessing

Module: `preprocess.py`

- **Class**: `PrePro`
  - Downsample EEG data.
  - Apply bandpass and notch filters.
  - Segment data into epochs.
  - Visualize preprocessed data.

### Feature Extraction

Module: `extract_features.py`

- **Class**: `Features`
  - Extract time-frequency representation.
  - Calculate features for each epoch.
  - Visualize extracted features.

### Creativity Measurement

Module: `measure_creativity.py`

- **Class**: `MeasCre`
  - Evaluate creativity based on linguistic attributes.
  - Extract features like word uniqueness, syntax, rhyme, and phonetic properties.

### Comparing Datasets

Module: `compare_datasets.py`

- **Class**: `CompareDatasets`
  - Compare and visualize different EEG datasets.
  - Understand variations and similarities between datasets.

### Visualization

Various visualization tools are provided throughout the modules to assist in data analysis and interpretation.

## Getting Started

The main driver script (`main.py`) integrates and runs various components of the pipeline. To execute the pipeline:

1. Ensure all dependencies are installed.
2. Update the data paths in the `main.py` script.
3. Run the `main.py` script.

---

This README provides an overview of the EEG pipeline. For detailed documentation on each module, function, and class, refer to the comments and docstrings within the respective Python files.
