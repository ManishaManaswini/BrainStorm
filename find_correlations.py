import os
import pickle
from typing import Iterable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class FindCorrelation:
    def __init__(self):
        self.total_times = []
        self.datasets = {}
        self.merged_data = pd.DataFrame()

    def flatten_eeg_features(self, eeg_features_dict):
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

    def flatten_creativity_features(self, creativity_features_dict, total_times):
        data = []

        for i, file in enumerate(creativity_features_dict.keys()):
            total_time = int(total_times[i])  # Convert total_time to int before division
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
            interpolated_creativity_scores = self.interpolate_scores(creativity_score_mapping, n_epochs)
            interpolated_readability_scores = self.interpolate_scores(readability_score_mapping, n_epochs)
            interpolated_complexity_scores = self.interpolate_scores(complexity_score_mapping, n_epochs)

            for time_point in range(n_epochs):
                epoch_data = {'file': file, 'epoch': time_point}
                epoch_data['creativity_scores'] = interpolated_creativity_scores[time_point]
                epoch_data['readability_scores'] = interpolated_readability_scores[time_point]
                epoch_data['complexity_scores'] = interpolated_complexity_scores[time_point]

                data.append(epoch_data)

        return pd.DataFrame(data)

    def interpolate_scores(self, score_mapping, n_epochs):
        scores = np.array([score_mapping.get(i, np.nan) for i in range(n_epochs)])  # Convert scores to a NumPy array
        isnan = np.isnan(scores)
        if isnan.all():
            return scores
        isnan_indices = np.where(isnan)[0]  # Indices where isnan is True
        not_isnan_indices = np.where(~isnan)[0]  # Indices where isnan is False
        try:
            scores[isnan] = np.interp(isnan_indices, not_isnan_indices, scores[not_isnan_indices])
        except Exception as e:
            print("isnan:", isnan)
            print("~isnan:", ~isnan)
            print("scores:", scores)
            raise e
        return scores


    def load_creativity_features(self, path, total_times):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            df = self.flatten_creativity_features(data, total_times)
            self.datasets["creativity_features"] = df
        return df

    def load_eeg_features(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            for file, dict in data.items():
                self.total_times.append(dict['total_epochs'] * 5)
            df = self.flatten_eeg_features(data)
            self.datasets["eeg_features"] = df
        return df

    def merge_datasets(self):
        # Merging on 'file' and 'epoch' columns
        self.merged_data = pd.merge(self.datasets["eeg_features"], self.datasets["creativity_features"], on=['file', 'epoch'], how='inner')

    def calculate_correlations(self):
        correlations = {}
        for target in ['creativity_scores', 'readability_scores', 'complexity_scores']:
            correlations[target] = {}
            for column in self.merged_data.columns:
                if column not in ['file','epoch', 'creativity_scores', 'readability_scores', 'complexity_scores']:
                    correlations[target][column] = \
                    spearmanr(self.merged_data[column].values, self.merged_data[target].values)[0]
        return correlations

    @staticmethod
    def plot_correlation_matrix(corr_df, title='Correlation matrix'):
        """Plots a correlation matrix for given features."""
        # Split the features into batches of 10
        n = 10
        figs = []
        titles = []
        feature_batches = [list(corr_df.index)[i:i + n] for i in range(0, corr_df.shape[0], n)]

        # For each batch of features, plot a correlation matrix
        for i, features in enumerate(feature_batches):
            missing_features = [f for f in features if f not in corr_df.index]
            if missing_features:
                print(f"Batch {i} contains missing features: {missing_features}")
                continue

            fig = plt.figure(figsize=(12, 12))
            sns.heatmap(corr_df.loc[features, ['creativity_scores', 'readability_scores', 'complexity_scores']],
                        annot=True, fmt=".2f", cmap="Blues")
            plt.title(f'{title} (features {i * n + 1} to {min((i + 1) * n, len(corr_df.index))})')
            # plt.yticks(rotation=45)
            plt.tight_layout()  # Ensure all elements fit within the figure
            figs.append(fig)
            titles.append(f"file_{i * n + 1}-{min((i + 1) * n, len(corr_df.index))}")
        return figs, titles


    def transform_data(self, df):
        transformed_data = {}

        for feature in df.drop(['epoch'], axis=1).columns:
            if isinstance(df[feature].iloc[0], Iterable):
                feature_values = df[feature].apply(pd.Series).stack().values
            else:
                feature_values = df[feature].values
            transformed_data[feature] = feature_values

        for target in ['creativity_scores', 'readability_scores', 'complexity_scores']:
            if isinstance(df[target].iloc[0], Iterable):
                target_scores = df[target].apply(pd.Series).stack().values
            else:
                target_scores = df[target].values
            transformed_data[target] = target_scores

        return pd.DataFrame(transformed_data)

    def run(self, path1, path2):
        self.load_eeg_features(path1)
        self.load_creativity_features(path2, self.total_times)
        self.merge_datasets()
        self.merged_data = self.transform_data(self.merged_data)
        feature_names = self.merged_data.columns.drop(
            ['file', 'creativity_scores', 'readability_scores', 'complexity_scores'])
        target_names = ['creativity_scores', 'readability_scores', 'complexity_scores']
        correlations = self.calculate_correlations()
        corr_df = pd.DataFrame(correlations)

        return correlations, corr_df





