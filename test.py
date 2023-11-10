import os
import time
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from load_data import LoadData
from preprocess import PrePro
from extract_features import FeatEx
from measure_creativity import MeasCre
from find_correlations import FindCorrelation
from compare_datasets import CompareDatasets
import warnings
warnings.filterwarnings("ignore", message="tostring() is deprecated")

corr = FindCorrelation()

ind_extracted_features = corr.load_eeg_features("E:/Master_Thesis/EEG_Analysis/Features_Output/Ind/Ind_extracted_features.pkl")

ind_extracted_features.to_csv("E:/Master_Thesis/EEG_Analysis/Features_Output/Ind/Ind_extracted_features.csv", encoding='utf-8', index=False)




