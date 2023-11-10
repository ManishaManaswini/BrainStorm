
import numpy as np
import matplotlib.pyplot as plt

class CompareDatasets:

    @staticmethod
    def plot_avg_power_band_ratios(Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features, parameters, channel_names=["AF7", "AF8", "TP9", "TP10"]):
        colormap = plt.cm.get_cmap('tab10')
        figs = []
        titles = []

        # Determine the longest time axis among the three datasets
        max_epochs = max(
            max([features['total_epochs'] for _, features in Ind_extracted_features.items()]),
            max([features['total_epochs'] for _, features in NoCogLoad_extracted_features.items()]),
            max([features['total_epochs'] for _, features in CogLoad_extracted_features.items()])
        )- 20
        time_axis = np.linspace(0, max_epochs * 5, max_epochs)  # Assume each epoch is 5 seconds long

        datasets = [Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features, ]
        dataset_names = ["Individual", "NoLoad", "CogLoad"]

        for channel_name in channel_names:
            fig, ax = plt.subplots(figsize=(12, 8))

            for data, name in zip(datasets, dataset_names):
                avg_values = {ratio: np.zeros(max_epochs) for ratio in parameters}
                std_values = {ratio: np.zeros(max_epochs) for ratio in parameters}

                for _, features in data.items():
                    feature_data = features['power_band_ratios']

                    for ratio_name in parameters:
                        ratio_values = [epoch_data[channel_name][ratio_name] for epoch_data in feature_data.values()][:max_epochs]
                        padded_ratio_values = np.zeros(max_epochs)
                        padded_ratio_values[:len(ratio_values)] = ratio_values
                        avg_values[ratio_name] += padded_ratio_values

                for ratio_name in parameters:
                    avg_values[ratio_name] /= len(data)

                for _, features in data.items():
                    feature_data = features['power_band_ratios']

                    for ratio_name in parameters:
                        ratio_values = [epoch_data[channel_name][ratio_name] for epoch_data in feature_data.values()][:max_epochs]
                        padded_ratio_values = np.zeros(max_epochs)
                        padded_ratio_values[:len(ratio_values)] = ratio_values
                        std_values[ratio_name] += (padded_ratio_values - avg_values[ratio_name]) ** 2

                for ratio_name in parameters:
                    std_values[ratio_name] = np.sqrt(std_values[ratio_name] / len(data))

                for ratio_name in parameters:
                    ax.plot(time_axis, avg_values[ratio_name], label=f"{ratio_name} - {name}",
                            color=colormap((parameters.index(ratio_name) + datasets.index(data)) % colormap.N))
                    ax.fill_between(time_axis,
                                    avg_values[ratio_name] - std_values[ratio_name],
                                    avg_values[ratio_name] + std_values[ratio_name],
                                    color=colormap((parameters.index(ratio_name) + datasets.index(data)) % colormap.N), alpha=0.3)

            ax.set_title(f"{channel_name} - Average Band power (SD)")
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Power (dB)')
            ax.legend()
            figs.append(fig)
            titles.append(f"{channel_name}_avg_with_std")

        return figs, titles


    @staticmethod
    def plot_avg_spectral_entropy(Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features, channel_names=["AF7", "AF8", "TP9", "TP10"]):
        colormap = plt.cm.get_cmap('tab10')
        figs = []
        titles = []

        max_epochs = max(
            max([features['total_epochs'] for _, features in Ind_extracted_features.items()]),
            max([features['total_epochs'] for _, features in NoCogLoad_extracted_features.items()]),
            max([features['total_epochs'] for _, features in CogLoad_extracted_features.items()])
        )- 20
        time_axis = np.linspace(0, max_epochs * 5, max_epochs)

        datasets = [Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features]
        dataset_names = ["Individual", "NoLoad", "CogLoad"]

        for channel_name in channel_names:
            fig, ax = plt.subplots(figsize=(12, 8))

            for data, name in zip(datasets, dataset_names):
                avg_values = np.zeros(max_epochs)
                std_values = np.zeros(max_epochs)

                for _, features in data.items():
                    feature_data = features['spectral_entropy']
                    values = [epoch_data[channel_name] for epoch_data in feature_data.values()][:max_epochs]
                    padded_values = np.zeros(max_epochs)
                    padded_values[:len(values)] = values
                    avg_values += padded_values

                avg_values /= len(data)

                for _, features in data.items():
                    feature_data = features['spectral_entropy']
                    values = [epoch_data[channel_name] for epoch_data in feature_data.values()][:max_epochs]
                    padded_values = np.zeros(max_epochs)
                    padded_values[:len(values)] = values
                    std_values += (padded_values - avg_values) ** 2

                std_values = np.sqrt(std_values / len(data))

                ax.plot(time_axis, avg_values, label=f"Average - {name}",
                        color=colormap(dataset_names.index(name) % colormap.N))
                ax.fill_between(time_axis, avg_values - std_values, avg_values + std_values,
                                color=colormap(dataset_names.index(name) % colormap.N), alpha=0.3)

            ax.set_title(f"{channel_name} - Average Spectral Entropy with Std Deviation")
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Spectral Entropy')
            ax.legend()
            figs.append(fig)
            titles.append(f"{channel_name}_avg_with_std")

        return figs, titles



    @staticmethod
    def plot_avg_psd_features(Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features, parameters, channel_names=["AF7", "AF8", "TP9", "TP10"]):
        colormap = plt.cm.get_cmap('tab10')
        figs = []
        titles = []

        max_epochs = max(
            max([features['total_epochs'] for _, features in Ind_extracted_features.items()]),
            max([features['total_epochs'] for _, features in NoCogLoad_extracted_features.items()]),
            max([features['total_epochs'] for _, features in CogLoad_extracted_features.items()])
        )- 20
        time_axis = np.linspace(0, max_epochs * 5, max_epochs)

        datasets = [Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features]
        dataset_names = ["Individual", "NoLoad", "CogLoad"]

        for channel_name in channel_names:
            fig, ax = plt.subplots(figsize=(12, 8))

            for data, name in zip(datasets, dataset_names):
                avg_values = {freq: np.zeros(max_epochs) for freq in parameters}
                std_values = {freq: np.zeros(max_epochs) for freq in parameters}

                for _, features in data.items():
                    feature_data = features['psd_features']

                    for freq in parameters:
                        freq_values = [epoch_data[channel_name][freq] for epoch_data in feature_data.values()][:max_epochs]
                        padded_freq_values = np.zeros(max_epochs)
                        padded_freq_values[:len(freq_values)] = freq_values
                        avg_values[freq] += padded_freq_values

                for freq in parameters:
                    avg_values[freq] /= len(data)

                for _, features in data.items():
                    feature_data = features['psd_features']

                    for freq in parameters:
                        freq_values = [epoch_data[channel_name][freq] for epoch_data in feature_data.values()][:max_epochs]
                        padded_freq_values = np.zeros(max_epochs)
                        padded_freq_values[:len(freq_values)] = freq_values
                        std_values[freq] += (padded_freq_values - avg_values[freq]) ** 2

                for freq in parameters:
                    std_values[freq] = np.sqrt(std_values[freq] / len(data))

                for freq in parameters:
                    ax.plot(time_axis, avg_values[freq], label=f"{freq} - {name}",
                            color=colormap((parameters.index(freq) + datasets.index(data)) % colormap.N))
                    ax.fill_between(time_axis,
                                    avg_values[freq] - std_values[freq],
                                    avg_values[freq] + std_values[freq],
                                    color=colormap((parameters.index(freq) + datasets.index(data)) % colormap.N), alpha=0.3)

            ax.set_title(f"{channel_name} - Average PSD (SD)",  fontsize=16, fontweight='bold')
            ax.set_xlabel('Time (s)',  fontsize=10, fontweight='bold')
            ax.set_ylabel('PSD (dB)',  fontsize=10, fontweight='bold')
            legend = ax.legend(fontsize=10)
            for text in legend.get_texts():
                text.set_fontweight('bold')
            figs.append(fig)
            titles.append(f"{channel_name}_avg_with_std")

        return figs, titles


    @staticmethod
    def plot_avg_coherence_features(Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features,
                                    channel_pairs=[("AF7_TP9"), ("AF8_TP10")]):
        colormap = plt.cm.get_cmap('tab10')
        figs = []
        titles = []

        max_epochs = max(
            max([features['total_epochs'] for _, features in Ind_extracted_features.items()]),
            max([features['total_epochs'] for _, features in NoCogLoad_extracted_features.items()]),
            max([features['total_epochs'] for _, features in CogLoad_extracted_features.items()])
        )- 20
        time_axis = np.linspace(0, max_epochs * 5, max_epochs)

        datasets = [Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features]
        dataset_names = ["Individual", "NoLoad", "CogLoad"]

        for channel_pair in channel_pairs:
            fig, ax = plt.subplots(figsize=(12, 8))

            for data, name in zip(datasets, dataset_names):
                avg_values = np.zeros(max_epochs)
                std_values = np.zeros(max_epochs)

                for _, features in data.items():
                    feature_data = features['coherence_features']
                    values = [epoch_data[channel_pair] for epoch_data in feature_data.values()][:max_epochs]
                    padded_values = np.zeros(max_epochs)
                    padded_values[:len(values)] = values
                    avg_values += padded_values

                avg_values /= len(data)

                for _, features in data.items():
                    feature_data = features['coherence_features']
                    values = [epoch_data[channel_pair] for epoch_data in feature_data.values()][:max_epochs]
                    padded_values = np.zeros(max_epochs)
                    padded_values[:len(values)] = values
                    std_values += (padded_values - avg_values) ** 2

                std_values = np.sqrt(std_values / len(data))

                ax.plot(time_axis, avg_values, label=f"Average - {name}",
                        color=colormap(dataset_names.index(name) % colormap.N))
                ax.fill_between(time_axis, avg_values - std_values, avg_values + std_values,
                                color=colormap(dataset_names.index(name) % colormap.N), alpha=0.3)

            ax.set_title(f"{channel_pair} - Average Coherence (SD)",  fontsize=16, fontweight='bold')
            ax.set_xlabel('Time (s)',  fontsize=10, fontweight='bold')
            ax.set_ylabel('Coherence',  fontsize=10, fontweight='bold')
            legend = ax.legend(fontsize=10)
            for text in legend.get_texts():
                text.set_fontweight('bold')
            figs.append(fig)
            titles.append(f"{channel_pair}_avg_with_std")

        return figs, titles


    @staticmethod
    def plot_avg_tfr_values(extracted_features, dataset_name, channel_names=["AF7", "AF8", "TP9", "TP10"]):
        colormap = plt.cm.get_cmap('RdBu_r')
        figs = []
        titles = []

        for channel_name in channel_names:
            # Collect TFR data for this channel
            tfr_data = []

            for _, features in extracted_features.items():
                tfr = features['tfr_features']
                channel_idx = tfr.ch_names.index(channel_name)
                tfr_data.append(tfr.data[channel_idx])

            # Average the TFR data across all files
            avg_tfr = np.mean(np.array(tfr_data), axis=0)

            fig, ax = plt.subplots(figsize=(12, 5))
            im = ax.imshow(avg_tfr, aspect='auto', origin='lower', cmap=colormap,
                           extent=[np.min(tfr.times), np.max(tfr.times), np.min(tfr.freqs), np.max(tfr.freqs)])
            ax.set_title(f"{channel_name} - Average TFR - {dataset_name}",  fontsize=16, fontweight='bold')
            ax.set_xlabel('Time (s)',  fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency (Hz)', fontsize=10, fontweight='bold')
            fig.colorbar(im, ax=ax)

            # fig.suptitle(f"{channel_name} TFR Average", fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            figs.append(fig)
            titles.append(f"{channel_name}_avg_TFR_comparison")

        return figs, titles

    @staticmethod
    def plot_avg_epoch_features(Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features, parameters):
        colormap = plt.cm.get_cmap('tab10')
        figs = []
        titles = []

        max_epochs = max(
            max([features['total_epochs'] for _, features in Ind_extracted_features.items()]),
            max([features['total_epochs'] for _, features in NoCogLoad_extracted_features.items()]),
            max([features['total_epochs'] for _, features in CogLoad_extracted_features.items()])
        )
        time_axis = np.linspace(0, max_epochs * 5, max_epochs)

        datasets = [Ind_extracted_features, NoCogLoad_extracted_features, CogLoad_extracted_features]
        dataset_names = ["Individual", "NoLoad", "CogLoad"]

        fig, ax = plt.subplots(figsize=(12, 8))
        for data, name in zip(datasets, dataset_names):
            avg_values = {param: np.zeros(max_epochs) for param in parameters}
            std_values = {param: np.zeros(max_epochs) for param in parameters}

            for _, features in data.items():
                feature_data = features['epoch_features']

                for param in parameters:
                    # Adjust here to handle numpy arrays
                    param_values = feature_data[param].mean(axis=1)
                    padded_param_values = np.zeros(max_epochs)
                    padded_param_values[:len(param_values)] = param_values
                    avg_values[param] += padded_param_values

            for param in parameters:
                avg_values[param] /= len(data)

            for _, features in data.items():
                feature_data = features['epoch_features']

                for param in parameters:
                    param_values = feature_data[param].mean(axis=1)
                    padded_param_values = np.zeros(max_epochs)
                    padded_param_values[:len(param_values)] = param_values
                    std_values[param] += (padded_param_values - avg_values[param]) ** 2

            for param in parameters:
                std_values[param] = np.sqrt(std_values[param] / len(data))

            for param in parameters:
                ax.plot(time_axis, avg_values[param], label=f"{param} - {name}",
                        color=colormap((parameters.index(param) + datasets.index(data)) % colormap.N))
                ax.fill_between(time_axis,
                                avg_values[param] - std_values[param],
                                avg_values[param] + std_values[param],
                                color=colormap((parameters.index(param) + datasets.index(data)) % colormap.N),
                                alpha=0.3)

        ax.set_title(f"Average Epoch Features with Std Deviation")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Epoch Feature Value')
        ax.legend()
        figs.append(fig)
        titles.append("epoch_features_avg_with_std")

        return figs, titles


