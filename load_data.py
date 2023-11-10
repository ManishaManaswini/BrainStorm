import json
import os
import pickle
import pandas as pd
import numpy as np
import datetime
import mne
import math
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings("ignore", message="tostring() is deprecated")

class LoadData:
    def __init__(self, path):
        self.path = path

    def load_csv(self, file_path, columns):
        data = pd.read_csv(file_path, usecols=columns)
        data = data.dropna()  # Remove rows with NA values
        return data

    def calculate_sfreq(self, timestamps):
        try:
            timestamps_datetime = [datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f") for ts in timestamps]
        except ValueError:
            raise ValueError("Invalid timestamps: Cannot calculate sampling frequency.")

        time_diff = np.diff(timestamps_datetime)
        avg_time_diff = np.mean(time_diff)

        if math.isnan(avg_time_diff.total_seconds()):
            raise ValueError("Invalid timestamps: Cannot calculate sampling frequency.")

        sfreq = int(1 / avg_time_diff.total_seconds())

        return sfreq

    def visualize_eeg_data(self, dataset):
        fig, axes = plt.subplots(len(dataset), figsize=(12, 6 * len(dataset)), sharex=True)
        fig.subplots_adjust(hspace=8)  # Adjust the spacing between subplots
        fig.tight_layout(pad=8.0)

        for i, (filename, data) in enumerate(dataset.items()):
            eeg_data = data['data'].get_data()
            channels = data['data'].ch_names[1:]  # Exclude the "TimeStamp" channel
            num_channels = len(channels)

            for j, channel in enumerate(channels):
                ax = axes[i] if len(dataset) > 1 else axes[j]
                ax.hist(eeg_data[j], bins=50, alpha=0.7, label=channel)

                # Add mean, std, skew, and kurtosis text to the plot
                mean = np.mean(eeg_data[j])
                std = np.std(eeg_data[j])
                skewness = skew(eeg_data[j])  # Use skew from scipy.stats
                kurt = kurtosis(eeg_data[j])
                ax.text(0.6, 0.8, f"Mean: {mean:.2f}\nStd: {std:.2f}\nSkewness: {skewness:.2f}\nKurtosis: {kurt:.2f}",
                        transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

            ax.set_title(f"EEG Data - File: {filename} (sfreq: {data['sfreq']})")
            ax.set_xlabel("Time")
            ax.set_ylabel("Amplitude")
            ax.legend()

        plt.tight_layout()
        plt.show()

    def convert_to_mne(self, data, sfreq):
        # Convert data to MNE format
        # Data shape is (n_channels, n_samples)
        raw = mne.io.RawArray(data, info=mne.create_info(ch_names=["AF7", "AF8", "TP9", "TP10"],
                                                         sfreq=sfreq, ch_types='eeg'))
        return raw

    def save_features_to_file(self, all_features, file_path):
        """Save the all_features dictionary to a file using pickle."""
        with open(file_path, 'wb') as f:
            pickle.dump(all_features, f)

    def load_features_from_file(self,file_path):
        """Load the all_features dictionary from a file using pickle."""
        with open(file_path, 'rb') as f:
            all_features = pickle.load(f)
        return all_features

    def save_dataset(self, dataset, filepath):

        for file, data in dataset.items():
            raw = data['data']
            sfreq = data['sfreq']
            raw.save(f'{filepath}/{file}_raw.fif', overwrite=True)
            if "epochs" in data.keys():
                epochs = data['epochs']
                epochs.save(f'{filepath}/{file}_epo.fif', overwrite=True)
            with open(f'{filepath}/{file}_sfreq.txt', 'w', encoding='utf-8') as f:
                f.write(str(sfreq))

    def load_dataset(self, filepath):
        dataset = {}
        fif_files = []
        epo_files = []
        epo_files = []
        txt_files = []
        files = []
        for file in os.listdir(filepath):
            if ".fif" in file:
                if "raw" in file:
                    fif_files.append(mne.io.read_raw_fif(f'{filepath}/{file}'))
                    s = file
                    c = '_'
                    n = [pos for pos, char in enumerate(s) if char == c][1]
                    file = file[0:n]
                    files.append(file)
                else:
                    epo_files.append(mne.read_epochs(f'{filepath}/{file}'))
            else:
                with open(f'{filepath}/{file}', 'r', encoding='utf-8') as f:
                    txt_files.append(float(f.read()))

        if len(epo_files) != 0:
            for i in range(len(files)):
                dataset[files[i]] = {'data': fif_files[i], 'epochs': epo_files[i], 'sfreq': txt_files[i]}
        else:
            for i in range(len(files)):
                dataset[files[i]] = {'data': fif_files[i], 'sfreq': txt_files[i]}

        return dataset

    def load_files(self):
        data_dir = self.path
        file_names = os.listdir(data_dir)
        fields = ["TimeStamp", "RAW_AF7", "RAW_AF8", "RAW_TP9", "RAW_TP10"]
        dataset = {}
        for file in file_names:
            df = self.load_csv(os.path.join(data_dir, file), fields)
            sfreq = self.calculate_sfreq(df['TimeStamp'])
            df = self.convert_to_mne(df.drop('TimeStamp', axis=1).values.T, sfreq)

            s = file
            c = '_'
            n = [pos for pos, char in enumerate(s) if char == c][1]
            file = file[0:n]
            dataset[file] = {'data':df,'sfreq':sfreq}

        #self.visualize_eeg_data(dataset)
        return dataset

