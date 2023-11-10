import ast
import itertools
import os
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import f_oneway
from load_data import LoadData
from preprocess import PrePro
from extract_features import FeatEx
from measure_creativity import MeasCre

metrics = ["creativity_scores", "readability_scores", "complexity_scores"]

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    if isinstance(list(d.keys())[0], int):
        for k, v in d.items():
                new_key = ''
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
    else:
        for k, v in d.items():
            new_key = parent_key + sep + str(k) if parent_key else str(k)
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
    return dict(items)

def flatten_eeg_features(eeg_features_dict):
    dfs = []
    ch_names = ["AF7", "AF8", "TP9", "TP10"]

    for file, features in eeg_features_dict.items():
        data = []

        # Determine the number of epochs based on the length of one of the features
        total_epochs = features['total_epochs']

        for i in range(total_epochs):
            epoch_data = {'file': file, 'epoch': i}

            # Flatten power_band_ratios, psd_features, and statistical_features
            for feature_name in ['power_band_ratios', 'psd_features', 'statistical_features']:
                for channel_name, channel_data in features.get(feature_name, {})[i].items():
                    for metric, value in channel_data.items():
                        key = f"{channel_name}_{feature_name}_{metric}"
                        epoch_data[key] = value

            # Handle tfr_features
            if 'tfr_features' in features:
                tfr_data_epoch = features['tfr_features'].data[:, :, i]
                tfr_data_epoch_flat = tfr_data_epoch.flatten()
                for j, value in enumerate(tfr_data_epoch_flat):
                    epoch_data[f'tfr_features_{j}'] = value

            # Handle spectral_entropy, coherence_features, and epoch_features
            for feature_name in ['spectral_entropy', 'coherence_features', 'epoch_features']:
                if feature_name in features:
                    if feature_name == 'epoch_features':
                        for metric, values in features[feature_name].items():
                            for idx, channel_name in enumerate(ch_names):
                                epoch_data[f"{channel_name}_{metric}"] = values[i][idx]
                    else:
                        for channel_name, channel_value in features[feature_name][i].items():
                            epoch_data[f"{feature_name}_{channel_name}"] = channel_value

            data.append(epoch_data)
        dfs.append(pd.DataFrame(data))

    df = pd.concat(dfs, ignore_index=True)
    return df

def interpolate_scores(epoch_scores, n_epochs):
    """Interpolates missing scores across epochs."""
    x = list(epoch_scores.keys())
    y = list(epoch_scores.values())
    all_epochs = list(range(n_epochs))
    interpolated_values = np.interp(all_epochs, x, y, left=np.nan, right=np.nan)
    return {epoch: score for epoch, score in zip(all_epochs, interpolated_values)}

def flatten_creativity_features(creativity_features_dict, total_times):
    data = []

    for i, file in enumerate(creativity_features_dict.keys()):
        total_time = total_times[i]
        n_epochs = total_time // 5

        # Extracting the times and data from the creativity scores
        times = creativity_features_dict[file]['creativity_scores']['times']
        creativity_scores_data = creativity_features_dict[file]['creativity_scores']['data']
        readability_scores_data = creativity_features_dict[file]['readability_scores']['data']
        complexity_scores_data = creativity_features_dict[file]['complexity_scores']['data']

        # Mapping scores to epochs based on times
        creativity_score_mapping = {int(t // 5): score for t, score in zip(times, creativity_scores_data)}
        readability_score_mapping = {int(t // 5): score for t, score in zip(times, readability_scores_data)}
        complexity_score_mapping = {int(t // 5): score for t, score in zip(times, complexity_scores_data)}

        # Interpolating missing scores
        interpolated_creativity_scores = interpolate_scores(creativity_score_mapping, n_epochs)
        interpolated_readability_scores = interpolate_scores(readability_score_mapping, n_epochs)
        interpolated_complexity_scores = interpolate_scores(complexity_score_mapping, n_epochs)

        for time_point in range(n_epochs):
            epoch_data = {'file': file, 'epoch': time_point}
            epoch_data['creativity_scores'] = interpolated_creativity_scores[time_point]
            epoch_data['readability_scores'] = interpolated_readability_scores[time_point]
            epoch_data['complexity_scores'] = interpolated_complexity_scores[time_point]

            data.append(epoch_data)

    return pd.DataFrame(data)


def compute_correlations_for_merged_data(merged_dataset):
    """Computes correlations for the entire merged dataset."""

    # Select only numeric columns
    numeric_data = merged_dataset.select_dtypes(include=[np.number])

    # Average the numeric data over epochs
    avg_data = numeric_data.groupby('epoch').mean()

    # Compute correlations for each feature with the scores
    correlations = avg_data.corr()

    return correlations


def plot_correlation_matrix_for_subset(correlations, feature_start, feature_end):
    """Plots the correlation matrix for a specific range of features and all metrics."""

    # Get a slice of the correlations for the range of features
    feature_slice = correlations.iloc[feature_start:feature_end, :]

    # Get the last three rows of the correlations (the metrics)
    metrics_slice = correlations.loc[["creativity_scores", "readability_scores", "complexity_scores"], :]

    # Concatenate the feature slice with the metrics slice
    subset = pd.concat([feature_slice, metrics_slice])

    # Create a mask to only show the lower triangle of the matrix (since it's mirrored around the diagonal)
    mask = np.triu(np.ones_like(subset, dtype=bool))

    # Set up the matplotlib figure
    plt.figure(figsize=(8, 6))

    # Draw the heatmap without a color bar
    sns.heatmap(subset, mask=mask, cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar=False,
                annot=True, annot_kws={"size": 6})

    plt.title(f"Correlation Matrix for features {feature_start} to {feature_end}")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrices(correlations_df, chunk_size=10):
    """Plots correlation matrices for chunks of features in the dataframe."""

    # Get the total number of features (excluding the metrics)
    total_features = correlations_df.shape[0] - 3

    # Calculate the number of chunks
    num_chunks = total_features // chunk_size
    if total_features % chunk_size != 0:
        num_chunks += 1

    # Plot correlation matrix for each chunk of features
    for i in range(num_chunks):
        feature_start = i * chunk_size
        feature_end = min((i + 1) * chunk_size, total_features)
        plot_correlation_matrix_for_subset(correlations_df, feature_start, feature_end)


loader = LoadData("D:/EEG data/Features_Output/Ind/")
extracted_features = loader.load_features_from_file("D:/EEG data/Features_Output/Ind/extracted_features.pkl")
# sent_creative_features = loader.load_features_from_file("D:/EEG data/Features_Output/Ind/sent/sent_creativity_features.pkl")
# word_creative_features = loader.load_features_from_file("D:/EEG data/Features_Output/Ind/word/word_creativity_features.pkl")

total_times = []
for file, data in extracted_features.items():
    total_times.append(data['total_epochs']*5)

# eeg_features_df = flatten_eeg_features(extracted_features)
# eeg_features_df.to_csv("D:/EEG data/Features_Output/Ind/flattened_eeg_features.csv", sep=',', index=False, encoding='utf-8')
eeg_features_df = pd.read_csv("D:/EEG data/Features_Output/Ind/flattened_eeg_features.csv")

# sent_creative_features_df = flatten_creativity_features(sent_creative_features, total_times)
# sent_creative_features_df.to_csv("D:/EEG data/Features_Output/Ind/flattened_sent_creative_features.csv", sep=',', index=False, encoding='utf-8')
# sent_creative_features_df = pd.read_csv("D:/EEG data/Features_Output/Ind/flattened_sent_creative_features.csv")
#
# merged_dataset = pd.merge(eeg_features_df, sent_creative_features_df, on=['file', 'epoch'])
#
# merged_dataset.to_csv("D:/EEG data/Features_Output/Ind/sent_merged_features.csv")

merged_dataset = pd.read_csv("D:/EEG data/Features_Output/Ind/sent_merged_features.csv")

sent_correlation_df = compute_correlations_for_merged_data(merged_dataset)

sent_correlation_df.to_csv("D:/EEG data/Features_Output/Ind/sent_correlation.csv")

# sent_correlation_df = pd.read_csv("D:/EEG data/Features_Output/Ind/sent_correlation.csv")

plot_correlation_matrices(sent_correlation_df)

# word_creative_features_df = flatten_creativity_features(word_creative_features, total_times)
# word_creative_features_df.to_csv("D:/EEG data/Features_Output/Ind/flattened_word_creative_features.csv", sep=',', index=False, encoding='utf-8')
# word_creative_features_df = pd.read_csv("D:/EEG data/Features_Output/Ind/flattened_word_creative_features.csv")
#
# merged_dataset = pd.merge(eeg_features_df, word_creative_features_df, on=['file', 'epoch'])
#
# merged_dataset.to_csv("D:/EEG data/Features_Output/Ind/word_merged_features.csv")

merged_dataset = pd.read_csv("D:/EEG data/Features_Output/Ind/word_merged_features.csv")

word_correlation_df = compute_correlations_for_merged_data(merged_dataset)

word_correlation_df.to_csv("D:/EEG data/Features_Output/Ind/word_correlation.csv")

# word_correlation_df = pd.read_csv("D:/EEG data/Features_Output/Ind/word_correlation.csv")

# plot_correlation_matrices(word_correlation_df)