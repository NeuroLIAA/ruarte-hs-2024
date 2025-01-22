from matplotlib import pyplot as plt
from os import path
import numpy as np
import pandas as pd
from . import utils
import seaborn as sns

class SaccadeAmplitude:

    def __init__(self, dataset_name, scanpath_metadata, results_dir, human_scanpaths_dir, max_scanpath_length, filters_mss,colors_dict):
        self.dataset_name = dataset_name
        self.max_scanpath_length = max_scanpath_length
        self.scanpath_metadata = scanpath_metadata
        self.dataset_results_dir = results_dir
        self.human_scanpaths_dir = human_scanpaths_dir
        self.filters_mss = filters_mss
        self.color_model_mapping = colors_dict

    def compute(self):
        saccade_amplitudes = pd.concat(list(self.scanpath_metadata.apply(self.compute_saccade_amplitude_for_subject, axis=1))).reset_index(drop=True)
        # For each model and each MSS i need the saccade number that contains 90% of the data
        saccade_numbers = saccade_amplitudes[["Model","MSS","Saccade Number"]].groupby(["Model","MSS"]).quantile(0.9).reset_index()
        # For each model and MSS keep only the saccades that are in the quantile, so i need the data that has a Saccade Number equal or less than the quantile
        saccade_amplitudes = pd.merge(saccade_amplitudes,saccade_numbers, on=["Model","MSS"])
        saccade_amplitudes = saccade_amplitudes[saccade_amplitudes["Saccade Number_x"] <= saccade_amplitudes["Saccade Number_y"]]
        saccade_amplitudes = saccade_amplitudes.drop(columns=["Saccade Number_y"])
        saccade_amplitudes = saccade_amplitudes.rename(columns={"Saccade Number_x":"Saccade Number"})
        self.plot(saccade_amplitudes)




    def compute_saccade_amplitude_for_subject(self, subject_info):
        subject_path = path.join(self.dataset_results_dir, subject_info["Model"], 'Scanpaths.json') if subject_info["Model"] != "Humans" else path.join(self.human_scanpaths_dir, subject_info["Scanpath_file"])
        subject_scanpaths = utils.load_dict_from_json(subject_path)
        saccade_amplitudes = []

        for trial_name in subject_scanpaths.keys():            
            mss = len(subject_scanpaths[trial_name]["memory_set"])
            target_found = subject_scanpaths[trial_name]["target_found"]
            if mss in self.filters_mss[subject_info["Model"]] or target_found == False:
                continue
            x_coor_normalization = 1024/subject_scanpaths[trial_name]["image_width"]/32 if subject_info["Model"] == "Humans" else 1
            y_coor_normalization = 768/subject_scanpaths[trial_name]["image_height"]/32 if subject_info["Model"] == "Humans" else 1
            x_diff = np.diff(subject_scanpaths[trial_name]["X"])* x_coor_normalization
            y_diff = np.diff(subject_scanpaths[trial_name]["Y"])* y_coor_normalization
            saccade_amplitude = np.sqrt(np.square(x_diff) + np.square(y_diff)) # And i need to fill this until self.max_scanpath_length - 1 with NaNs
            for fixation_rank, saccade_amplitude in enumerate(saccade_amplitude):
                saccade_amplitudes.append([mss, subject_info["Model"], fixation_rank+1,saccade_amplitude])
        return pd.DataFrame(saccade_amplitudes, columns=["MSS","Model","Saccade Number","Saccade Amplitude"])
    
    def plot(self, saccade_amplitudes_df):
        mss_values = np.sort(saccade_amplitudes_df.MSS.unique())
        mss_amounts = len(mss_values)
        # Amount of characters of the longest label
        fig, axs = plt.subplots( mss_amounts,1, sharey=True, figsize=(4,2.5*mss_amounts))
        if mss_amounts == 1: axs = np.array([axs])
        # I need to plot the mean with a line and the SEM should be a shaded area around the line
        # For each row in the dataframe i need to plot only until the Saccade Number
        for i, mss in enumerate(mss_values):
            data = saccade_amplitudes_df[saccade_amplitudes_df["MSS"] == mss]
            sns.boxplot(x="Saccade Number", y="Saccade Amplitude", data=data,ax=axs[i], showfliers=False,hue="Model",palette=self.color_model_mapping,legend=False)
            axs[i].set_xlabel("Saccade Number")
            axs[i].set_ylabel("Saccade Amplitude")
            axs[i].set_title(f"MSS {mss}")
        fig.suptitle(self.dataset_name + ' Dataset')
        plt.tight_layout()
        plt.savefig(path.join(self.dataset_results_dir, f"Saccade Amplitude.png"))
        plt.close()
