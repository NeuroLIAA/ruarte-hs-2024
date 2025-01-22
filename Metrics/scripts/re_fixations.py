from matplotlib import pyplot as plt
from os import path
import numpy as np
import pandas as pd
from . import utils
import seaborn as sns

class ReFixations:

    def __init__(self, dataset_name, scanpath_metadata, results_dir, human_scanpaths_dir, max_scanpath_length, filters_mss,colors_dict):
        self.dataset_name = dataset_name
        self.max_scanpath_length = max_scanpath_length
        self.scanpath_metadata = scanpath_metadata
        self.dataset_results_dir = results_dir
        self.human_scanpaths_dir = human_scanpaths_dir
        self.filters_mss = filters_mss
        self.color_model_mapping = colors_dict

    def compute(self):
        self.amount_subj_per_model = self.scanpath_metadata.groupby("Model").size().to_dict()
        re_fixations = pd.concat(list(self.scanpath_metadata.apply(self.compute_re_fixations_for_subject, axis=1))).reset_index(drop=True)
        
        # Get unique list of Images
        images = re_fixations["Image"].unique()
        # Map the images to the corresponding index
        re_fixations["Image"] = re_fixations["Image"].map({image: index for index, image in enumerate(images)})

        # I need to count the repeated values for each image, model, MSS, re fixation x and re fixation y
        re_fixations = re_fixations.groupby(["Image", "Model", "MSS", "Re Fixation X", "Re Fixation Y", "Scanpath Length"]).size().reset_index(name="Re Fixations")
        # Divide the count by the scanpath length to get the percentage of re fixations
        re_fixations["Re Fixations"] = re_fixations["Re Fixations"] / re_fixations["Scanpath Length"]
        # Remove the scanpath length column
        re_fixations = re_fixations.drop(columns=["Scanpath Length"])

        self.plot_humans_vs_models(re_fixations)
        self.plot_bars_per_image(re_fixations)
        #self.plot_hist_humans_vs_models(re_fixations)





    def compute_re_fixations_for_subject(self, subject_info):
        """
        Compute re-fixations for a given subject based on their scanpaths.

        Args:
            subject_info (dict): Information about the subject including model type and scanpath file.

        Returns:
            pandas.DataFrame: DataFrame containing information about re-fixations, including trial name, memory set size,
                            subject model, and coordinates of re-fixations.
        """

        # Construct path to the scanpaths file based on subject information
        subject_path = (
            path.join(self.dataset_results_dir, subject_info["Model"], 'Scanpaths.json')
            if subject_info["Model"] != "Humans"
            else path.join(self.human_scanpaths_dir, subject_info["Scanpath_file"])
        )

        # Load scanpaths data from JSON file
        subject_scanpaths = utils.load_dict_from_json(subject_path)

        # Initialize list to store re-fixations
        re_fixations = []



        # Iterate through each trial in the scanpaths data
        for trial_name, trial_data in subject_scanpaths.items():
            mss = len(trial_data["memory_set"])
            target_found = trial_data["target_found"]
            scanpath_length = len(trial_data["X"])

            if mss in self.filters_mss[subject_info["Model"]] or not target_found:             
                continue

            # Calculate normalization factors
            if subject_info["Model"] == "Humans":
                x_coor_normalization = 1024/subject_scanpaths[trial_name]["image_width"] / 32
                y_coor_normalization = 768/subject_scanpaths[trial_name]["image_height"] / 32
            else:
                x_coor_normalization = 1
                y_coor_normalization = 1

            fixations = [(int(x * x_coor_normalization), int(y * y_coor_normalization)) for x, y in zip(trial_data["X"], trial_data["Y"])]

            # Set to store coordinates of previously detected re-fixations
            repeated_values = set()

            for fixation_rank, fixation in enumerate(fixations):
                repeated_values_for_fixation = set()
                for possible_refixations in range(fixation_rank + 1, len(fixations)):
                    if (
                        abs(fixation[0] - fixations[possible_refixations][0]) <= 1
                        and abs(fixation[1] - fixations[possible_refixations][1]) <= 1
                    ):
                        if fixations[possible_refixations] in repeated_values:
                            continue
                        re_fixations.append([trial_name, mss, subject_info["Model"], fixation[0], fixation[1],scanpath_length])
                        repeated_values_for_fixation.add(fixations[possible_refixations])

                repeated_values.update(repeated_values_for_fixation)

        return pd.DataFrame(re_fixations, columns=["Image", "MSS", "Model", "Re Fixation X", "Re Fixation Y",'Scanpath Length'])
    
    def plot_humans_vs_models(self, re_fixations_df):
        # Group by Model, MSS, Image, Re Fixation X, and Re Fixation Y and take the mean of the Re Fixations
        re_fixations_df = re_fixations_df.groupby(["Model", "MSS", "Image", "Re Fixation X", "Re Fixation Y"]).mean().reset_index()


        # Images have a 32 x 24 grid, so there are 32*24 = 768 possible re fixations
        # Now i need to map the Re Fixation X and Re Fixation Y columns into a single column according to the 32 x 24 grid
        re_fixations_df["Location"] = re_fixations_df["Re Fixation X"] + re_fixations_df["Re Fixation Y"] * 32

        # Now if i take into account the images and the re fixations, i have 768*len(images) possible re fixations
        # I need to transform the Re Fixation column into a single value that represents the re fixation in the 768*len(images) space
        re_fixations_df["Location"] = re_fixations_df["Location"] + re_fixations_df["Image"] * 768

        # Now i can drop the Re Fixation X and Re Fixation Y and Image columns
        re_fixations_df = re_fixations_df.drop(columns=["Re Fixation X", "Re Fixation Y","Image"])

        # Create a MultiIndex with all combinations of Model, MSS, and Location
        labels = re_fixations_df['Model'].unique()
        mss_values = np.sort(re_fixations_df.MSS.unique())

        multi_index = pd.MultiIndex.from_product([labels, mss_values, list(range(768*32*24))], names=['Model', 'MSS', 'Location'])
        all_combinations_df = pd.DataFrame(index=multi_index).reset_index()
        re_fixations_df = pd.merge(all_combinations_df, re_fixations_df, on=["Model", "MSS", "Location"], how="left").fillna(0)
        # Order by Model, MSS, and Location
        re_fixations_df = re_fixations_df.sort_values(by=["Model", "MSS", "Location"])

        mss_amounts = len(mss_values)
        
        models = labels[labels != "Humans"]
        # Amount of characters of the longest label
        char_amount = max(map(len,models))
        fig, axs = plt.subplots( mss_amounts,len(models), figsize=(0.5*len(models)+char_amount*0.15+4.5*len(models),4.5*mss_amounts))
        if mss_amounts == 1: axs = np.array([axs])
        if len(models) == 1: axs = np.array([axs])


        for j, model in enumerate(models):
            color = self.color_model_mapping[model]
            for i, mss in enumerate(mss_values):                
                human_data = re_fixations_df[re_fixations_df["MSS"] == mss]
                human_data = human_data[human_data["Model"] == "Humans"]
                data = re_fixations_df[re_fixations_df["MSS"] == mss]
                data = data[data["Model"] == model]
                # Scatter plot of model vs humans
                axs[i][j].scatter(data["Re Fixations"], human_data["Re Fixations"], color=color)
                axs[i][j].set_xlabel("Mean Prob of Re Fixation Model")
                axs[i][j].set_ylabel("Mean Prob of Re Fixation Humans")
                axs[i][j].set_title(f"MSS {mss} - {model} vs Humans")
        fig.suptitle(self.dataset_name + ' Dataset')

        plt.tight_layout()
        plt.savefig(path.join(self.dataset_results_dir, f"Re Fixations.png"))
        plt.close()

    def plot_bars_per_image(self,re_fixations_df):


        re_fixations_df = re_fixations_df.drop(columns=["Re Fixation X", "Re Fixation Y"])

        re_fixations_df = re_fixations_df.groupby(["Model", "MSS", "Image"]).sum().reset_index()

        # Divide re fixations by the amount of subjects per model, which is the same as taking the mean
        re_fixations_df["Re Fixations"] = re_fixations_df["Re Fixations"] / re_fixations_df["Model"].map(self.amount_subj_per_model)


        human_data = re_fixations_df[re_fixations_df["Model"] == "Humans"]
        human_data = human_data.sort_values(by="Re Fixations").reset_index(drop=True)
        # Get a Map that goes from Image to the index of the rowç
        image_map = {image: index for index, image in enumerate(human_data["Image"])}

        re_fixations_df["Image"] = re_fixations_df["Image"].map(image_map)

        

        labels = re_fixations_df['Model'].unique()
        mss_values = np.sort(re_fixations_df.MSS.unique())
        mss_amounts = len(mss_values)

        # Amount of characters of the longest label
        char_amount = max(map(len,labels))
        fig, axs = plt.subplots( mss_amounts,1,sharey=True, figsize=(0.5*len(labels)+char_amount*0.15+4.5,4.5*mss_amounts))
        if mss_amounts == 1: axs = np.array([axs])

        for i, mss in enumerate(mss_values):
            data = re_fixations_df[re_fixations_df["MSS"] == mss]
            sns.barplot(x="Image", y="Re Fixations", hue="Model", data=data, ax=axs[i],palette=self.color_model_mapping)
            axs[i].set_xlabel("Image Index")
            axs[i].set_ylabel("Mean Re Fixations / Scanpath Length")
            axs[i].set_title(f"MSS {mss}")
        fig.suptitle(self.dataset_name + ' Dataset')
        plt.tight_layout()
        plt.savefig(path.join(self.dataset_results_dir, f"Re Fixations Barplot.png"))
        plt.close()