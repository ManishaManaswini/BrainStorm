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

def save_figures(figs, titles, name, save_path):
    save_path = os.path.join(save_path, name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i, (fig, title) in enumerate(zip(figs, titles)):
        # Replace spaces and special characters in the title with underscores
        title = title.replace(' ', '_')
        for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
            title = title.replace(char, '_')
        # Generate the file name using the provided name, index, and title
        file_name = f'{name}_{i}_{title}.png'
        file_path = os.path.join(save_path, file_name)
        fig.savefig(file_path)
        plt.close(fig)


"""Load Baseline Data."""
loader = LoadData("E:/Master_Thesis/EEG_Analysis/Pipeline_Input/CSV/Base")
baseline_dataset = loader.load_files()
n = 1 # Slice the dataset upto nth minute
for file in baseline_dataset:
    df = baseline_dataset[file]['data']
    sfreq = baseline_dataset[file]['sfreq']
    start_time, end_time = 0, n * 60  # specify start and end times in seconds
    df = df.crop(tmin=start_time, tmax=end_time)
    baseline_dataset[file] = {'data':df,'sfreq':sfreq}
loader.save_dataset(baseline_dataset,"E:/Master_Thesis/EEG_Analysis//Pipeline_Input/FIF/Base")
baseline_dataset = loader.load_dataset("E:/Master_Thesis/EEG_Analysis/Pipeline_Input/FIF/Base")

save_path = "E:/Master_Thesis/EEG_Analysis/Plots/"

# Instantiate the PrePro class for baseline dataset
prepro = PrePro(baseline_dataset)
print("RAW Data")
# prepro.plot_channel_subplots(baseline_dataset, "RAW Data")

baseline_dataset = prepro.apply_downsampling()
# prepro.plot_channel_subplots(baseline_dataset, "RAW Data")

# Apply notch filter at a specific frequency (e.g., 50 Hz)
baseline_dataset = prepro.apply_notch_filter(freq=50)
print("Notch filtered data")
# prepro.plot_channel_subplots(baseline_dataset, "Notch filtered data")

# Apply bandpass filter with a low cutoff of 1 Hz and high cutoff of 40 Hz
baseline_dataset = prepro.apply_bandpass_filter(low_freq=1, high_freq=40)
print("Bandpass filtered data")
# prepro.plot_channel_subplots(baseline_dataset, "Bandpass filtered data")

baseline_dataset = prepro.apply_artifact_removal()
print("Artifact rejected data")
# prepro.plot_channel_subplots(baseline_dataset, "Artifact rejected data")

# Save the preprocessed baseline dataset
loader.save_dataset(baseline_dataset,"E:/Master_Thesis/EEG_Analysis/Preprocessed_Output/Base")


"""Load Cognitive Load Dataset"""

save_path = "E:/Master_Thesis/EEG_Analysis/Plots/CogLoad/"
# loader = LoadData("E:/Master_Thesis/EEG_Analysis/Pipeline_Input/CSV/CogLoad")
# CogLoad_dataset = loader.load_files()
# loader.save_dataset(CogLoad_dataset,"E:/Master_Thesis/EEG_Analysis/Pipeline_Input/FIF/CogLoad")
# baseline_dataset = loader.load_dataset("E:/Master_Thesis/EEG_Analysis/Preprocessed_Output/Base")
# CogLoad_dataset = loader.load_dataset("E:/Master_Thesis/EEG_Analysis/Pipeline_Input/FIF/CogLoad")
#
# # Instantiate the PrePro class for CogLoad dataset
# prepro = PrePro(CogLoad_dataset)
# # print("RAW Data")
# figs, titles = prepro.plot_channel_subplots(CogLoad_dataset, "RAW Data")
# save_figures(figs, titles, 'CogLoad_raw_data', save_path)
#
# # Apply downsampling to the mne object
# CogLoad_dataset = prepro.apply_downsampling()
#
# # Apply notch filter at a specific frequency (e.g., 60 Hz)
# CogLoad_dataset = prepro.apply_notch_filter(freq=50)
# print("Notch filtered data")
#
# # Apply bandpass filter with a low cutoff of 1 Hz and high cutoff of 40 Hz
# CogLoad_dataset = prepro.apply_bandpass_filter(low_freq=1, high_freq=40)
# print("Bandpass filtered data")
#
# CogLoad_dataset = prepro.apply_artifact_removal()
# print("Artifact rejected data")
#
# # CogLoad_dataset = prepro.manual_component_selection()
# # print("Manual artifact rejection")
# # prepro.plot_channel_subplots(CogLoad_dataset, "Manually curated data")
#
# # Calculate epochs for all the data and add it to the dataset
# CogLoad_dataset = prepro.apply_epoching(CogLoad_dataset, epoch_duration=5)
#
# # Apply rejection criteria to remove epochs exceeding a threshold
# CogLoad_dataset = prepro.apply_rejection(CogLoad_dataset)
#
# print("Epochs")
#
# CogLoad_dataset = prepro.apply_baseline_correction(baseline_dataset)
# figs, titles = prepro.plot_channel_subplots(CogLoad_dataset, "Baseline corrected data")
# save_figures(figs, titles, 'CogLoad_corr_data', save_path)
#
# if prepro.check_normality(plot_option=False):
#     CogLoad_dataset = prepro.apply_standardization() # Apply standardization to the data if normally distributed
#     figs, titles = prepro.plot_channel_subplots(CogLoad_dataset, "Standardized data")
#     save_figures(figs, titles, 'CogLoad_stan_data', save_path)
# else:
#     CogLoad_dataset = prepro.apply_normalization() # Apply normalization to the data if not normally distributed
#     figs, titles = prepro.plot_channel_subplots(CogLoad_dataset, "Normalised data")
#     save_figures(figs, titles, 'CogLoad_norm_data', save_path)
#
# CogLoad_dataset = prepro.apply_normalization() # Apply normalization to the data if not normally distributed
# figs, titles = prepro.plot_channel_subplots(CogLoad_dataset, "Normalised data")
# save_figures(figs, titles, 'CogLoad_norm_data', save_path)
#
# CogLoad_dataset = prepro.apply_amplitude_normalization() # Apply normalization to the data if not normally distributed
# figs, titles = prepro.plot_channel_subplots(CogLoad_dataset, "Amp Normalised data")
# save_figures(figs, titles, 'CogLoad_amp_norm_data', save_path)
#
#
# # Save the preprocessed CogLoad writing task dataset
# loader.save_dataset(CogLoad_dataset,"E:/Master_Thesis/EEG_Analysis/Preprocessed_Output/CogLoad")
#
# loader = LoadData("E:/Master_Thesis/EEG_Analysis/Preprocessed_Output/CogLoad")
# CogLoad_dataset = loader.load_dataset("E:/Master_Thesis/EEG_Analysis/Preprocessed_Output/CogLoad")
#
# # Create an instance of the FeatEx class
# featex = FeatEx()
# # Extract features from the EEG data
# CogLoad_extracted_features = featex.extract_features(CogLoad_dataset)
# loader.save_features_to_file(CogLoad_extracted_features,"E:/Master_Thesis/EEG_Analysis/Features_Output/CogLoad/CogLoad_extracted_features.pkl")
# CogLoad_extracted_features = loader.load_features_from_file("E:/Master_Thesis/EEG_Analysis/Features_Output/CogLoad/CogLoad_extracted_features.pkl")
#
# total_times = []
# for file, data in CogLoad_extracted_features.items():
#     total_times.append(data['total_epochs']*5)

# Extract creativity features from text data
analyzer = MeasCre()
creativity_features = analyzer.analyze_creativity_multiple_files("E:/Master_Thesis/EEG_Analysis/txt_files/CogLoad", total_times, type="sent")
# loader.save_features_to_file(creativity_features,"E:/Master_Thesis/EEG_Analysis/Features_Output/CogLoad/sent/CogLoad_sent_creativity_features.pkl")
CogLoad_creativity_features = loader.load_features_from_file("E:/Master_Thesis/EEG_Analysis/Features_Output/CogLoad/sent/CogLoad_sent_creativity_features.pkl")

# Save all the creativity score plots
parameters = ["creativity_scores", "readability_scores"]
figs, titles = analyzer.plot_all_creativity_features(CogLoad_creativity_features, parameters)
save_figures(figs, titles, 'creativity_features', save_path)
#
#
# # Save all TFR value plots
# figs, titles = featex.plot_avg_tfr_heatmap(CogLoad_extracted_features)
# save_figures(figs, titles, 'avg_tfr_heatmaps', save_path)
#
# figs, titles = featex.plot_all_tfr_values(CogLoad_extracted_features)
# save_figures(figs, titles, 'all_tfr_values', save_path)
#
# figs, titles = featex.plot_avg_tfr_values(CogLoad_extracted_features)
# save_figures(figs, titles, 'avg_tfr_values', save_path)
#
# # Save band power plots
# band_power = ['alpha_power']
# figs, titles = featex.plot_all_power_band_ratios(CogLoad_extracted_features, band_power)
# save_figures(figs, titles, 'alpha_power_band', save_path)
# #
# band_power = ['delta_power']
# figs, titles = featex.plot_all_power_band_ratios(CogLoad_extracted_features, band_power)
# save_figures(figs, titles, 'delta_power_band', save_path)
#
# band_power = ['theta_power']
# figs, titles = featex.plot_all_power_band_ratios(CogLoad_extracted_features, band_power)
# save_figures(figs, titles, 'theta_power_band', save_path)
# #
# # band_power = ['low_beta_power']
# figs, titles = featex.plot_all_power_band_ratios(CogLoad_extracted_features, band_power)
# save_figures(figs, titles, 'low_beta_power_band', save_path)
# #
# # band_power = ['high_beta_power']
# figs, titles = featex.plot_all_power_band_ratios(CogLoad_extracted_features, band_power)
# save_figures(figs, titles, 'high_beta_power_band', save_path)
# #
# # # Save psd features
# figs, titles = featex.plot_avg_psd_features(CogLoad_extracted_features, ['mean_power'])
# save_figures(figs, titles, 'avg_mean_psd_features', save_path)
#
# figs, titles = featex.plot_avg_psd_features(CogLoad_extracted_features, ['max_power'])
# save_figures(figs, titles, 'avg_max_psd_features', save_path)
#
# # Save coherence features
# figs, titles = featex.plot_avg_coherence_features(CogLoad_extracted_features)
# save_figures(figs, titles, 'avg_coherence_features', save_path)
#
# # Save all epoch stat plots
# figs, titles = featex.plot_avg_epoch_features(CogLoad_extracted_features)
# save_figures(figs, titles, 'avg_epoch_features', save_path)

# """Find correlation"""
#
# cor_finder= FindCorrelation()
# correlations, corr_df = cor_finder.run(path1="E:/Master_Thesis/EEG_Analysis/Features_Output/CogLoad/CogLoad_extracted_features.pkl",
#                                 path2="E:/Master_Thesis/EEG_Analysis/Features_Output/CogLoad/sent/CogLoad_sent_creativity_features.pkl")
#
# # Set the threshold value
# threshold = 0.3
#
# # Filter the features based on the threshold
# significant_features = corr_df[
#     (corr_df['creativity_scores'].abs() >= threshold) |
#     (corr_df['readability_scores'].abs() >= threshold) |
#     (corr_df['complexity_scores'].abs() >= threshold)
#     ]
#
# corr_df.to_csv("E:/Master_Thesis/EEG_Analysis/Features_Output/CogLoad/sent/CogLoad_sent_correlations.csv")
# figs, titles = cor_finder.plot_correlation_matrix(corr_df)
# save_figures(figs, titles, 'correlation_matrix', save_path)
# print(significant_features)

"""Load NoCogLoad Data."""

save_path = "E:/Master_Thesis/EEG_Analysis/Plots/NoCogLoad/"
loader = LoadData("E:/Master_Thesis/EEG_Analysis/Pipeline_Input/CSV/NoCogLoad")
NoCogLoad_dataset = loader.load_files()
loader.save_dataset(NoCogLoad_dataset,"E:/Master_Thesis/EEG_Analysis/Pipeline_Input/FIF/NoCogLoad")
baseline_dataset = loader.load_dataset("E:/Master_Thesis/EEG_Analysis/Preprocessed_Output/Base")
NoCogLoad_dataset = loader.load_dataset("E:/Master_Thesis/EEG_Analysis/Pipeline_Input/FIF/NoCogLoad")

# Instantiate the PrePro class for NoCogLoad dataset
prepro = PrePro(NoCogLoad_dataset)
print("RAW Data")
figs, titles = prepro.plot_channel_subplots(NoCogLoad_dataset, "RAW Data")
save_figures(figs, titles, 'NoCogLoad_raw_data', save_path)

NoCogLoad_dataset = prepro.apply_downsampling()

# Apply notch filter at a specific frequency (e.g., 60 Hz)
NoCogLoad_dataset = prepro.apply_notch_filter(freq=50)
print("Notch filtered data")

# Apply bandpass filter with a low cutoff of 1 Hz and high cutoff of 40 Hz
NoCogLoad_dataset = prepro.apply_bandpass_filter(low_freq=1, high_freq=40)
print("Bandpass filtered data")

NoCogLoad_dataset = prepro.apply_artifact_removal()
print("Artifact rejected data")

# Calculate epochs for all the data and add it to the dataset
NoCogLoad_dataset = prepro.apply_epoching(NoCogLoad_dataset, epoch_duration=5)

# Apply rejection criteria to remove epochs exceeding a threshold
NoCogLoad_dataset = prepro.apply_rejection(NoCogLoad_dataset)

print("Epochs")

NoCogLoad_dataset = prepro.apply_baseline_correction(baseline_dataset)
figs, titles = prepro.plot_channel_subplots(NoCogLoad_dataset, "Baseline corrected data")
save_figures(figs, titles, 'NoCogLoad_corr_data', save_path)
#
if prepro.check_normality(plot_option=False):
    NoCogLoad_dataset = prepro.apply_standardization() # Apply standardization to the data if normally distributed
    figs, titles = prepro.plot_channel_subplots(NoCogLoad_dataset, "Standardized data")
    save_figures(figs, titles, 'NoCogLoad_stan_data', save_path)
else:
    NoCogLoad_dataset = prepro.apply_normalization() # Apply normalization to the data if not normally distributed
    figs, titles = prepro.plot_channel_subplots(NoCogLoad_dataset, "Normalised data")
    save_figures(figs, titles, 'NoCogLoad_norm_data', save_path)

NoCogLoad_dataset = prepro.apply_combined_normalization() # Apply normalization to the data if not normally distributed
figs, titles = prepro.plot_channel_subplots(NoCogLoad_dataset, "Normalised data")
save_figures(figs, titles, 'NoCogLoad_norm_data', save_path)

# Save the preprocessed NoCogLoad writing task dataset
loader.save_dataset(NoCogLoad_dataset,"E:/Master_Thesis/EEG_Analysis/Preprocessed_Output/NoCogLoad")

loader = LoadData("E:/Master_Thesis/EEG_Analysis/Preprocessed_Output/NoCogLoad")
NoCogLoad_dataset = loader.load_dataset("E:/Master_Thesis/EEG_Analysis/Preprocessed_Output/NoCogLoad")

# Create an instance of the FeatEx class
featex = FeatEx()
# Extract features from the EEG data
NoCogLoad_extracted_features = featex.extract_features(NoCogLoad_dataset)
loader.save_features_to_file(NoCogLoad_extracted_features,"E:/Master_Thesis/EEG_Analysis/Features_Output/NoCogLoad/NoCogLoad_extracted_features.pkl")

NoCogLoad_extracted_features = loader.load_features_from_file("E:/Master_Thesis/EEG_Analysis/Features_Output/NoCogLoad/NoCogLoad_extracted_features.pkl")

total_times = []
for file, data in NoCogLoad_extracted_features.items():
    total_times.append(data['total_epochs']*5)

# Extract creativity features from text data
analyzer = MeasCre()
creativity_features = analyzer.analyze_creativity_multiple_files("E:/Master_Thesis/EEG_Analysis/txt_files/NoCogLoad", total_times, type="sent")
loader.save_features_to_file(creativity_features,"E:/Master_Thesis/EEG_Analysis/Features_Output/NoCogLoad/sent/NoCogLoad_sent_creativity_features.pkl")
NoCogLoad_creativity_features = loader.load_features_from_file("E:/Master_Thesis/EEG_Analysis/Features_Output/NoCogLoad/sent/NoCogLoad_sent_creativity_features.pkl")

# Save all the creativity score plots
parameters = ["creativity_scores", "complexity_scores"]
figs, titles = analyzer.plot_all_creativity_features(NoCogLoad_creativity_features, parameters)
save_figures(figs, titles, 'creativity_features', save_path)

# Save all TFR value plots
figs, titles = featex.plot_avg_tfr_heatmap(NoCogLoad_extracted_features)
save_figures(figs, titles, 'avg_tfr_heatmaps', save_path)

figs, titles = featex.plot_all_tfr_values(NoCogLoad_extracted_features)
save_figures(figs, titles, 'all_tfr_values', save_path)

figs, titles = featex.plot_avg_tfr_values(NoCogLoad_extracted_features)
save_figures(figs, titles, 'avg_tfr_values', save_path)

# Save band power plots
band_power = ['alpha_power']
figs, titles = featex.plot_all_power_band_ratios(NoCogLoad_extracted_features, band_power)
save_figures(figs, titles, 'alpha_power_band', save_path)

band_power = ['delta_power']
figs, titles = featex.plot_all_power_band_ratios(NoCogLoad_extracted_features, band_power)
save_figures(figs, titles, 'delta_power_band', save_path)

band_power = ['theta_power']
figs, titles = featex.plot_all_power_band_ratios(NoCogLoad_extracted_features, band_power)
save_figures(figs, titles, 'theta_power_band', save_path)

band_power = ['low_beta_power']
figs, titles = featex.plot_all_power_band_ratios(NoCogLoad_extracted_features, band_power)
save_figures(figs, titles, 'low_beta_power_band', save_path)
#
band_power = ['high_beta_power']
figs, titles = featex.plot_all_power_band_ratios(NoCogLoad_extracted_features, band_power)
save_figures(figs, titles, 'high_beta_power_band', save_path)
#
# Save psd features
figs, titles = featex.plot_avg_psd_features(NoCogLoad_extracted_features, ['mean_power'])
save_figures(figs, titles, 'avg_mean_psd_features', save_path)

figs, titles = featex.plot_avg_psd_features(NoCogLoad_extracted_features, ['max_power'])
save_figures(figs, titles, 'avg_max_psd_features', save_path)

# Save coherence features
figs, titles = featex.plot_avg_coherence_features(NoCogLoad_extracted_features)
save_figures(figs, titles, 'avg_coherence_features', save_path)

# Save all epoch stat plots
figs, titles = featex.plot_avg_epoch_features(NoCogLoad_extracted_features)
save_figures(figs, titles, 'avg_epoch_features', save_path)

"""Find correlation"""

cor_finder= FindCorrelation()
correlations, corr_df = cor_finder.run(path1="E:/Master_Thesis/EEG_Analysis/Features_Output/NoCogLoad/NoCogLoad_extracted_features.pkl",
                                path2="E:/Master_Thesis/EEG_Analysis/Features_Output/NoCogLoad/sent/NoCogLoad_sent_creativity_features.pkl")

# Set the threshold value
threshold = 0.3

# Filter the features based on the threshold
significant_features = corr_df[
    (corr_df['creativity_scores'].abs() >= threshold) |
    (corr_df['readability_scores'].abs() >= threshold) |
    (corr_df['complexity_scores'].abs() >= threshold)
    ]

corr_df.to_csv("E:/Master_Thesis/EEG_Analysis/Features_Output/NoCogLoad/sent/NoCogLoad_sent_correlations.csv")
figs, titles = cor_finder.plot_correlation_matrix(corr_df)
save_figures(figs, titles, 'correlation_matrix', save_path)
print(significant_features)

"""Load Individual Data."""

save_path = "E:/Master_Thesis/EEG_Analysis/Plots/Ind/"
loader = LoadData("E:/Master_Thesis/EEG_Analysis/Pipeline_Input/CSV/Ind")
ind_dataset = loader.load_files()
loader.save_dataset(ind_dataset,"E:/Master_Thesis/EEG_Analysis/Pipeline_Input/FIF/Ind")
baseline_dataset = loader.load_dataset("E:/Master_Thesis/EEG_Analysis/Preprocessed_Output/Base")
ind_dataset = loader.load_dataset("E:/Master_Thesis/EEG_Analysis/Pipeline_Input/FIF/Ind")

# Instantiate the PrePro class for Ind dataset
prepro = PrePro(ind_dataset)
print("RAW Data")
figs, titles = prepro.plot_channel_subplots(ind_dataset, "RAW Data")
save_figures(figs, titles, 'ind_raw_data', save_path)

ind_dataset = prepro.apply_downsampling()

# Apply notch filter at a specific frequency (e.g., 60 Hz)
ind_dataset = prepro.apply_notch_filter(freq=50)
print("Notch filtered data")

# Apply bandpass filter with a low cutoff of 1 Hz and high cutoff of 40 Hz
ind_dataset = prepro.apply_bandpass_filter(low_freq=1, high_freq=40)
print("Bandpass filtered data")

ind_dataset = prepro.apply_artifact_removal()
print("Artifact rejected data")

# Calculate epochs for all the data and add it to the dataset
ind_dataset = prepro.apply_epoching(ind_dataset, epoch_duration=5)

# Apply rejection criteria to remove epochs exceeding a threshold
# ind_dataset = prepro.apply_rejection(ind_dataset)

print("Epochs")
# prepro.visualize_epochs(ind_dataset)

ind_dataset = prepro.apply_baseline_correction(baseline_dataset)
figs, titles = prepro.plot_channel_subplots(ind_dataset, "Baseline corrected data")
save_figures(figs, titles, 'ind_1_corr_data', save_path)

if prepro.check_normality(plot_option=False):
    ind_dataset = prepro.apply_standardization() # Apply standardization to the data if normally distributed
    figs, titles = prepro.plot_channel_subplots(ind_dataset, "Standardized data")
    save_figures(figs, titles, 'ind_stan_data', save_path)
else:
    ind_dataset = prepro.apply_normalization() # Apply normalization to the data if not normally distributed
    figs, titles = prepro.plot_channel_subplots(ind_dataset, "Normalised data")
    save_figures(figs, titles, 'ind_norm_data', save_path)

ind_dataset = prepro.apply_combined_normalization() # Apply normalization to the data if not normally distributed
figs, titles = prepro.plot_channel_subplots(ind_dataset, "Normalised data")
save_figures(figs, titles, 'ind_norm_data', save_path)

# Save the preprocessed Indvidual writing task dataset
loader.save_dataset(ind_dataset,"E:/Master_Thesis/EEG_Analysis/Preprocessed_Output/Ind")

loader = LoadData("E:/Master_Thesis/EEG_Analysis/Preprocessed_Output/Ind")
Ind_dataset = loader.load_dataset("E:/Master_Thesis/EEG_Analysis/Preprocessed_Output/Ind")

# Create an instance of the FeatEx class
featex = FeatEx()

# Extract features from the EEG data
Individual_extracted_features = featex.extract_features(Ind_dataset)
loader.save_features_to_file(Individual_extracted_features,"E:/Master_Thesis/EEG_Analysis/Features_Output/Ind/Ind_extracted_features.pkl")

Ind_extracted_features = loader.load_features_from_file("E:/Master_Thesis/EEG_Analysis/Features_Output/Ind/Ind_extracted_features.pkl")

total_times = []
for file, data in Ind_extracted_features.items():
    total_times.append(data['total_epochs']*5)

# Extract creativity features from text data
analyzer = MeasCre()
creativity_features = analyzer.analyze_creativity_multiple_files("E:/Master_Thesis/EEG_Analysis/txt_files/Ind", total_times, type="sent")
loader.save_features_to_file(creativity_features,"E:/Master_Thesis/EEG_Analysis/Features_Output/Ind/sent/Ind_sent_creativity_features.pkl")
Ind_creativity_features = loader.load_features_from_file("E:/Master_Thesis/EEG_Analysis/Features_Output/Ind/sent/Ind_sent_creativity_features.pkl")

# Save all the creativity score plots
parameters = ["creativity_scores", "complexity_scores"]
figs, titles = analyzer.plot_all_creativity_features(Ind_creativity_features, parameters)
save_figures(figs, titles, 'creativity_features', save_path)


# Save all TFR value plots
figs, titles = featex.plot_avg_tfr_heatmap(Ind_extracted_features)
save_figures(figs, titles, 'avg_tfr_heatmaps', save_path)

figs, titles = featex.plot_all_tfr_values(Ind_extracted_features)
save_figures(figs, titles, 'all_tfr_values', save_path)

figs, titles = featex.plot_avg_tfr_values(Ind_extracted_features)
save_figures(figs, titles, 'avg_tfr_values', save_path)

# Save band power plots
band_power = ['alpha_power']
figs, titles = featex.plot_all_power_band_ratios(Ind_extracted_features, band_power)
save_figures(figs, titles, 'alpha_power_band', save_path)

band_power = ['delta_power']
figs, titles = featex.plot_all_power_band_ratios(Ind_extracted_features, band_power)
save_figures(figs, titles, 'delta_power_band', save_path)

band_power = ['theta_power']
figs, titles = featex.plot_all_power_band_ratios(Ind_extracted_features, band_power)
save_figures(figs, titles, 'theta_power_band', save_path)

band_power = ['low_beta_power']
figs, titles = featex.plot_all_power_band_ratios(Ind_extracted_features, band_power)
save_figures(figs, titles, 'low_beta_power_band', save_path)

band_power = ['high_beta_power']
figs, titles = featex.plot_all_power_band_ratios(Ind_extracted_features, band_power)
save_figures(figs, titles, 'high_beta_power_band', save_path)

# Save psd features
figs, titles = featex.plot_avg_psd_features(Ind_extracted_features, ['mean_power'])
save_figures(figs, titles, 'avg_mean_psd_features', save_path)

figs, titles = featex.plot_avg_psd_features(Ind_extracted_features, ['max_power'])
save_figures(figs, titles, 'avg_max_psd_features', save_path)

# Save coherence features
figs, titles = featex.plot_avg_coherence_features(Ind_extracted_features)
save_figures(figs, titles, 'avg_coherence_features', save_path)

# Save all epoch stat plots
figs, titles = featex.plot_avg_epoch_features(Ind_extracted_features)
save_figures(figs, titles, 'avg_epoch_features', save_path)

"""Find correlation"""

cor_finder= FindCorrelation()
correlations, corr_df = cor_finder.run(path1="E:/Master_Thesis/EEG_Analysis/Features_Output/Ind/Ind_extracted_features.pkl",
                                path2="E:/Master_Thesis/EEG_Analysis/Features_Output/Ind/sent/Ind_sent_creativity_features.pkl")

# Set the threshold value
threshold = 0.3

# Filter the features based on the threshold
significant_features = corr_df[
    (corr_df['creativity_scores'].abs() >= threshold) |
    (corr_df['readability_scores'].abs() >= threshold) |
    (corr_df['complexity_scores'].abs() >= threshold)
    ]

corr_df.to_csv("E:/Master_Thesis/EEG_Analysis/Features_Output/Ind/sent/Ind_sent_correlations.csv")
figs, titles = cor_finder.plot_correlation_matrix(corr_df)
save_figures(figs, titles, 'correlation_matrix', save_path)
print(significant_features)


"""Compare Datasets"""

save_path = "E:/Master_Thesis/EEG_Analysis/Plots/ComparisonNorm/"
compare = CompareDatasets()

figs, titles = compare.plot_avg_tfr_values(Ind_extracted_features, "Individual Writing")
save_figures(figs, titles, 'avg_tfr_values_Ind', save_path)

figs, titles = compare.plot_avg_tfr_values(CogLoad_extracted_features, "Collaborative Writing with Cognitive Load")
save_figures(figs, titles, 'avg_tfr_values_Load', save_path)

figs, titles = compare.plot_avg_tfr_values(NoCogLoad_extracted_features, "Collaborative Writing with No Cognitive Load")
save_figures(figs, titles, 'avg_tfr_values_NoLoad', save_path)

figs, titles = compare.plot_avg_power_band_ratios(Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features, ['alpha_power'])
save_figures(figs, titles, 'avg_alpha_band_power', save_path)
#
figs, titles = compare.plot_avg_power_band_ratios(Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features, ['delta_power'])
save_figures(figs, titles, 'avg_delta_band_power', save_path)
#
figs, titles = compare.plot_avg_power_band_ratios(Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features, ['theta_power'])
save_figures(figs, titles, 'avg_theta_band_power', save_path)
#
figs, titles = compare.plot_avg_power_band_ratios(Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features, ['low_beta_power'])
save_figures(figs, titles, 'avg_low_beta_band_power', save_path)
#
figs, titles = compare.plot_avg_power_band_ratios(Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features, ['high_beta_power'])
save_figures(figs, titles, 'avg_high_beta_band_power', save_path)

figs, titles = compare.plot_avg_power_band_ratios(Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features, ['alpha_delta_ratio'])
save_figures(figs, titles, 'avg_alpha_delta_ratio_ratios', save_path)

figs, titles = compare.plot_avg_power_band_ratios(Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features, ['alpha_theta_ratio'])
save_figures(figs, titles, 'avg_alpha_theta_ratio_ratios', save_path)

figs, titles = compare.plot_avg_spectral_entropy(Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features)
save_figures(figs, titles, 'avg_spectral_entropy', save_path)
#
figs, titles = compare.plot_avg_psd_features(Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features, ['mean_power'])
save_figures(figs, titles, 'avg_psd_mean_power', save_path)
#
figs, titles = compare.plot_avg_psd_features(Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features, ['median_power'])
save_figures(figs, titles, 'avg_psd_median_power', save_path)
#
figs, titles = compare.plot_avg_psd_features(Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features, ['max_power'])
save_figures(figs, titles, 'avg_psd_max_power', save_path)

figs, titles = compare.plot_avg_coherence_features(Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features)
save_figures(figs, titles, 'avg_coherence_features', save_path)

figs, titles = compare.plot_avg_epoch_features(Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features, ['epoch_mean'])
save_figures(figs, titles, 'avg_epoch_mean', save_path)

figs, titles = compare.plot_avg_epoch_features(Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features, ['epoch_kurtosis'])
save_figures(figs, titles, 'avg_epoch_kurtosis', save_path)

figs, titles = compare.plot_avg_epoch_features(Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features, ['epoch_skewness'])
save_figures(figs, titles, 'avg_epoch_skewness', save_path)