import itertools
import time
import pandas as pd
import mne
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from mne.viz import plot_epochs
from scipy.stats import kurtosis, skew


class PrePro:
    def __init__(self, dataset):
        self.dataset = dataset
        sfreqs = []
        for file, data_info in self.dataset.items():
            sfreqs.append(data_info['sfreq'])
            print(data_info['sfreq'])
        self.min_sfrq = min(sfreqs)

    def apply_downsampling(self):

        if (self.min_sfrq < 250):
            sfreq = self.min_sfrq  # set downsampling freq to min sampling freq in dataset
        else:
            sfreq = 250  # If the minimum sampling freq in the dataset is greater than 250, set downsampling freq to 250

        for file, data_info in self.dataset.items():
            data = data_info['data'].copy()
            data_downsampled = data.copy().resample(sfreq)
            self.dataset[file]['data'] = data_downsampled
        return self.dataset

    def apply_bandpass_filter(self, low_freq, high_freq):
        for file, data_info in self.dataset.items():
            data = data_info['data'].copy()
            ch_names = data_info['data'].info['ch_names']
            ch_types = ['eeg'] * len(ch_names)
            info = mne.create_info(ch_names=ch_names, sfreq=self.dataset[file]['sfreq'],
                                   ch_types=ch_types)
            picks = mne.pick_types(info, meg=False, eeg=True)
            if len(picks) == 0:
                continue
            data_filtered = data.copy()
            data_filtered.load_data().filter(low_freq, high_freq, picks=picks)
            self.dataset[file]['data'] = data_filtered
        return self.dataset

    def apply_notch_filter(self, freq):
        for file, data_info in self.dataset.items():
            data = data_info['data'].copy()
            ch_names = data_info['data'].info['ch_names']
            ch_types = ['eeg'] * len(ch_names)
            info = mne.create_info(ch_names=ch_names, sfreq=self.dataset[file]['sfreq'],
                                   ch_types=ch_types)
            picks = mne.pick_types(info, meg=False, eeg=True)
            if len(picks) == 0:
                continue
            data_notch_filtered = data.copy()
            data_notch_filtered.load_data().notch_filter(freq, picks=picks)
            self.dataset[file]['data'] = data_notch_filtered
        return self.dataset

    def apply_artifact_removal(self):
        for file, data_info in self.dataset.items():
            data = data_info['data'].copy()

            # High-pass filter the data before ICA
            data.filter(l_freq=1, h_freq=None)

            # Create pseudo-EOG channels
            # NOTE: The choice of EEG channels to create the pseudo-EOG channels
            # depends on your specific electrode montage and should be adjusted accordingly
            pseudo_eog = data.get_data()[0, :] - data.get_data()[1, :]  # Difference between the first two EEG channels
            eog_info = mne.create_info(['EOG'], data.info['sfreq'])
            eog_raw = mne.io.RawArray(pseudo_eog.reshape(1, -1), eog_info)

            # Apply the same filtering to the pseudo-EOG channel as the original data
            eog_raw.filter(l_freq=1, h_freq=None, picks='EOG')

            # Create new Raw object that includes the EEG and pseudo-EOG channels
            data_with_eog = mne.io.RawArray(np.vstack([data.get_data(), eog_raw.get_data()]),
                                            mne.create_info(data.ch_names + ['EOG'], data.info['sfreq'],
                                                            ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eog']))

            # Fit ICA
            ica = mne.preprocessing.ICA(random_state=0, method='fastica')
            ica.fit(data_with_eog)

            # Apply the ICA to the data
            data_artifact_removed = ica.apply(data_with_eog)

            # SSP for artifact removal
            projs_eog, _ = mne.preprocessing.compute_proj_eog(data_artifact_removed, n_eeg=1, reject=None)
            data_ssp_applied = data_artifact_removed.copy().add_proj(projs_eog).apply_proj()

            data_ssp_applied = data_ssp_applied.drop_channels(['EOG'])

            self.dataset[file]['data'] = data_ssp_applied

        return self.dataset

    def manual_component_selection(self):
        for file, data_info in self.dataset.items():
            data = data_info['data'].copy()

            # Set standard montage for Muse headset
            ch_pos = dict(
                TP9=[-0.5, 0, 0.01],
                AF7=[-0.5, 0.5, 0.02],
                AF8=[0.5, 0.5, 0.03],
                TP10=[0.5, 0, 0.04]
            )

            montage = mne.channels.make_dig_montage(ch_pos, coord_frame='head')
            data.set_montage(montage)

            # Fit ICA
            ica = mne.preprocessing.ICA(random_state=0, method='fastica')
            ica.fit(data)

            print("Please close the components window after selecting components to exclude.")

            # Interactive plot. User can click on components to view properties.
            # In the properties window, there's a 'Reject' button to exclude the component.
            ica.plot_properties(data)

            # Apply the ICA to the data
            data_artifact_removed = ica.apply(data)

            self.dataset[file]['data'] = data_artifact_removed

        return self.dataset

    def apply_epoching(self, dataset, epoch_duration):

        for file, data_info in dataset.items():
            # Access the data and sample frequency from the dictionary
            data = data_info['data']
            sfreq = data_info['sfreq']

            # Apply epoching to segment the data with a fixed duration for each epoch
            # Duration of each epoch in seconds

            # Calculate the number of samples per epoch based on the epoch duration and sample frequency
            num_samples_per_epoch = int(epoch_duration * sfreq)

            # Calculate the number of epochs based on the available data size and samples per epoch
            num_epochs = int(np.floor(data.n_times / num_samples_per_epoch))

            # Calculate the total number of samples for the available epochs
            total_samples = num_samples_per_epoch * num_epochs

            # Trim the data to ensure an even number of samples
            trimmed_data = data.copy().crop(tmax=(total_samples - 1) / sfreq)  # Subtract 1 from total_samples

            # Reshape the data to the format (n_epochs, n_channels, n_samples)
            reshaped_data = trimmed_data.get_data().reshape((num_epochs, data.get_data().shape[0], -1))

            # Create a list of channel names corresponding to the data columns
            channel_names = data_info['data'].info['ch_names']

            # Create an info object from the original data
            info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')

            # Create artificial events for each epoch
            events = np.zeros((num_epochs, 3), dtype=int)
            events[:, 0] = np.arange(0, num_epochs) * num_samples_per_epoch
            events[:, 2] = 1

            # Create metadata as a DataFrame for absolute start and end times
            metadata = pd.DataFrame({'start': np.arange(0, num_epochs) * epoch_duration,
                                     'end': (np.arange(0, num_epochs) + 1) * epoch_duration})

            # Create EpochsArray with the reshaped data and metadata
            epochs = mne.EpochsArray(reshaped_data, info=info, events=events, event_id=None, metadata=metadata,
                                     baseline=(None, None))

            # Append the epochs to the list
            dataset[file] = {'data': data_info['data'], 'epochs': epochs, 'sfreq': data_info['sfreq']}

        return dataset

    def apply_rejection(self, dataset, threshold=200):
        for file, data_info in dataset.items():
            reject_criteria = dict(eeg=threshold)
            data_info['epochs'].drop_bad(reject=reject_criteria)
            dataset[file] = {'data': data_info['data'], 'epochs': data_info['epochs'], 'sfreq': data_info['sfreq']}
        return dataset

    def apply_baseline_correction(self, baseline_dataset):
        for file, data_info in self.dataset.items():
            epochs = data_info['epochs'].copy()

            # Get the corresponding baseline data
            baseline_data = baseline_dataset[file]['data'].copy()

            # Calculate the mean of the baseline data
            baseline_mean = baseline_data.get_data().mean(axis=1, keepdims=True)

            # Subtract the baseline mean from the epochs data
            epochs_data = epochs.get_data()
            epochs_baseline_corrected_data = epochs_data - baseline_mean

            # Create a new Epochs object with the baseline corrected data
            epochs_baseline_corrected = mne.EpochsArray(epochs_baseline_corrected_data, epochs.info,
                                                        events=epochs.events, event_id=epochs.event_id, on_missing='ignore')

            # Update epochs in the dataset
            self.dataset[file]['epochs'] = epochs_baseline_corrected

        return self.dataset

    def check_normality(self, plot_option=False, alpha=0.05):
        p_values = []
        for file, data_info in self.dataset.items():
            data = data_info['data'].copy().get_data()

            # Flatten the data and check if there are enough samples for normality test
            data_flat = data.flatten()
            if data_flat.shape[0] >= 8:
                # Perform normality test
                _, p = stats.normaltest(data_flat)
                p_values.append(p)

            # Plot the distribution of the EEG data
            if plot_option:
                plt.hist(data_flat, bins=100)
                plt.title(f'EEG data distribution for file: {file}')
                yield plt

        # If no normality test was performed (because all data had fewer than 8 samples), return None
        if not p_values:
            return None

        # Apply Bonferroni correction for multiple comparisons
        alpha_adjusted = alpha / len(p_values)

        # Check if the data is normally distributed
        is_normal = all(p > alpha_adjusted for p in p_values)

        return is_normal

    # def apply_normalization(self):
    #     for file, data_info in self.dataset.items():
    #         epochs = data_info['epochs'].copy()
    #
    #         # Get the data from the Epochs object
    #         epochs_data = epochs.get_data()
    #
    #         # Normalize the data
    #         epochs_data_normalized = (epochs_data - np.min(epochs_data, axis=2, keepdims=True)) / (
    #                 np.max(epochs_data, axis=2, keepdims=True) - np.min(epochs_data, axis=2, keepdims=True))
    #
    #         # Create a new Epochs object with the normalized data
    #         epochs_normalized = mne.EpochsArray(epochs_data_normalized, epochs.info, events=epochs.events,
    #                                             event_id=epochs.event_id)
    #
    #         # Update epochs in the dataset
    #         self.dataset[file]['epochs'] = epochs_normalized
    #
    #     return self.dataset
    #
    # def apply_amplitude_normalization(self):
    #     for file, data_info in self.dataset.items():
    #         epochs = data_info['epochs'].copy()
    #
    #         # Get the data from the Epochs object
    #         epochs_data = epochs.get_data()
    #
    #         # Normalize the data based on amplitude across channels
    #         epochs_data_normalized = (epochs_data - np.min(epochs_data, axis=1, keepdims=True)) / (
    #                 np.max(epochs_data, axis=1, keepdims=True) - np.min(epochs_data, axis=1, keepdims=True))
    #
    #         # Create a new Epochs object with the normalized data
    #         epochs_normalized = mne.EpochsArray(epochs_data_normalized, epochs.info, events=epochs.events,
    #                                             event_id=epochs.event_id)
    #
    #         # Update epochs in the dataset
    #         self.dataset[file]['epochs'] = epochs_normalized
    #
    #     return self.dataset

    def apply_combined_normalization(self):
        for file, data_info in self.dataset.items():
            epochs = data_info['epochs'].copy()

            # Get the data from the Epochs object
            epochs_data = epochs.get_data()

            # Normalize the data across time domain (axis=2)
            epochs_data_time_normalized = (epochs_data - np.min(epochs_data, axis=2, keepdims=True)) / (
                    np.max(epochs_data, axis=2, keepdims=True) - np.min(epochs_data, axis=2, keepdims=True))

            # Normalize the data based on amplitude across channels (axis=1) on the time-normalized data
            epochs_data_amplitude_normalized = (epochs_data_time_normalized - np.min(epochs_data_time_normalized,
                                                                                     axis=1, keepdims=True)) / (
                                                       np.max(epochs_data_time_normalized, axis=1,
                                                              keepdims=True) - np.min(epochs_data_time_normalized,
                                                                                      axis=1, keepdims=True))

            # Create a new Epochs object with the doubly normalized data
            epochs_normalized = mne.EpochsArray(epochs_data_amplitude_normalized, epochs.info, events=epochs.events,
                                                event_id=epochs.event_id)

            # Update epochs in the dataset
            self.dataset[file]['epochs'] = epochs_normalized

        return self.dataset

    def apply_standardization(self):
        for file, data_info in self.dataset.items():
            epochs = data_info['epochs'].copy()

            # Get the data from the Epochs object
            epochs_data = epochs.get_data()

            # Standardize the data
            epochs_data_standardized = (epochs_data - np.mean(epochs_data, axis=2, keepdims=True)) / np.std(epochs_data,
                                                                                                            axis=2,
                                                                                                            keepdims=True)

            # Create a new Epochs object with the standardized data
            epochs_standardized = mne.EpochsArray(epochs_data_standardized, epochs.info, events=epochs.events,
                                                  event_id=epochs.event_id)

            # Update epochs in the dataset
            self.dataset[file]['epochs'] = epochs_standardized

        return self.dataset

    def visualize_epochs(self, dataset):
        n = len(dataset)
        fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
        for i, (file, data_info) in enumerate(dataset.items()):
            data_info['epochs'].plot_image(axes=axes[i], show=False)
            axes[i][0].set_title(file)
        yield plt

    def plot_channel_subplots(self, dataset, plot_name):
        figs = []
        titles = []
        for i in range(4):

            if i == 0:
                ch = "AF7"
            if i == 1:
                ch = "AF8"
            if i == 2:
                ch = "TP9"
            if i == 3:
                ch = "TP10"

            num_files = len(dataset)
            colormap = plt.cm.get_cmap('tab10')  # Use a colormap with a larger set of distinct colors

            fig, axs = plt.subplots(num_files, 1, figsize=(12, 8), sharex=True)

            plot_names = []
            for idx, (file, data_info) in enumerate(dataset.items()):
                channel_data = data_info['data'].get_data()[i, :]

                # s = file
                # c = '_'
                # n = [pos for pos, char in enumerate(s) if char == c][1]
                # file = file[0:n]

                plot_names.append(file)

                color = colormap(idx % colormap.N)  # Assign a unique color based on the colormap
                axs[idx].plot(channel_data, color=color)  # Set unique color for each subplot
                axs[idx].tick_params(axis='x', rotation=45)  # Rotate x-axis labels

            axs[-1].set_xlabel('Samples')

            # Add a common y-axis label for all subplots
            fig.text(0.04, 0.5, 'Amplitude (uV)', va='center', rotation='vertical')

            plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between subplots

            # Create a custom legend with the plot names and corresponding colors
            handles = [plt.Line2D([], [], color=colormap(idx % colormap.N), label=name) for idx, name in
                       enumerate(plot_names)]
            plt.legend(handles=handles, loc="lower right")

            title_ax = axs[-1].twinx()  # Create a separate subplot for the title
            title_ax.set_axis_off()  # Turn off the axis for the title subplot
            title_ax.set_title(f"{ch}-{plot_name}", fontsize=16, fontweight='bold', pad=600)
            figs.append(fig)
            titles.append(f"{ch}_{plot_name}")
        return figs, titles

    def plot_single_epoch_multiple_files(self, epoch_index):
        files = list(self.dataset.keys())

        # Split files into groups of at most 3
        file_groups = [files[i:i + 3] for i in range(0, len(files), 3)]

        figs = []
        titles = []

        for group in file_groups:
            # Create a figure for each group of 3 files
            fig, axs = plt.subplots(4, len(group), figsize=(10 * len(group), 12), sharex=True)

            for i, file in enumerate(group):
                data_info = self.dataset[file]
                data = data_info['epochs'].get_data()  # Extract numpy array from RawArray

                if epoch_index < len(data):
                    for j in range(4):  # Assume there are 4 channels
                        if j == 0:
                            ch = "AF7"
                        if j == 1:
                            ch = "AF8"
                        if j == 2:
                            ch = "TP9"
                        if j == 3:
                            ch = "TP10"
                        axs[j, i].plot(data[epoch_index, j, :])
                        axs[j, i].set_ylabel(f'{ch}')
                        if j == 0:
                            axs[j, i].set_title(f'{file}')

            plt.xlabel('Samples')
            fig.suptitle(f'Data from epoch {epoch_index + 1}', y=0.98)
            plt.tight_layout()
            figs.append(fig)
            titles.append(f"epoch_{epoch_index + 1}")
        return figs, titles
