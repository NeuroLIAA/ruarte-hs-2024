import multimatch_gaze as mm
import numpy as np
import matplotlib.pyplot as plt
from . import utils
from scipy.stats import pearsonr,spearmanr
from os import path
import pandas as pd
from .. import constants
import importlib

class Multimatch:
    def __init__(self, dataset_name, scanpath_metadata, results_dir, human_scanpaths_dir, max_scanpath_length, filters_mss,colors_dict):
        self.dataset_name = dataset_name

        self.scanpath_metadata = scanpath_metadata
        self.scanpath_metadata["Index to compute"] = self.scanpath_metadata.index
        self.scanpath_metadata.loc[self.scanpath_metadata["Model"] != "Humans", "Index to compute"] = -1
        self.dataset_results_dir = results_dir
        self.human_scanpaths_dir = human_scanpaths_dir
        self.filters_mss = filters_mss
        self.color_model_mapping = colors_dict
        #Primero viene el load human y despues el add model (en el viejo)


    def compute(self):
        print('Computing multimatch for ' + self.dataset_name + ' dataset')        
        # Do a cross join between human_metadata and the series of the unique models (excluding humans)
        model_metadata = self.scanpath_metadata[self.scanpath_metadata["Model"] != "Humans"]
        model_sizes = pd.DataFrame(model_metadata["Model"], columns=["Model"]).apply(lambda x: self.get_screen_and_receptive_size(x), axis=1)
        if path.isfile(path.join(self.dataset_results_dir, 'Multimatch_human_mean_per_image.csv')):
            saved_mm_humans = pd.read_csv(path.join(self.dataset_results_dir, 'Multimatch_human_mean_per_image.csv'))
            computed_models = saved_mm_humans["Model"].unique()
        else:
            computed_models = []
            saved_mm_humans = pd.DataFrame(columns=["MSS","Image","Model","MMvec","MMdir","MMlen","MMpos","MMtime"])
        # If the models in computed_models are the same as the ones in model_metadata, then mm_humans_df = saved_mm_humans
        if all(model_metadata["Model"].isin(computed_models)):
            mm_humans_df = saved_mm_humans
        else:
            human_metadata = self.scanpath_metadata[self.scanpath_metadata["Model"] == "Humans"]
            # Remove the last row (the last subject multimatch is computed against all the others, so it is not needed)
            human_metadata = human_metadata.iloc[:-1]
            human_metadata = human_metadata.merge(model_sizes[~model_metadata["Model"].isin(computed_models)], how="cross")
            mm_humans_df = pd.concat(list(human_metadata.apply(lambda x: self.compute_on_subject(x), axis=1))).reset_index(drop=True)
            mm_humans_df = mm_humans_df.groupby(["MSS","Image","Model"]).mean().reset_index()
            if not saved_mm_humans.empty:
                mm_humans_df = pd.concat([mm_humans_df,saved_mm_humans]).reset_index(drop=True)        
            mm_humans_df.to_csv(path.join(self.dataset_results_dir, 'Multimatch_human_mean_per_image.csv'), index=False)
        mm_humans_df = mm_humans_df[mm_humans_df["Model"].isin(model_metadata["Model"])].reset_index(drop=True)
        if path.isfile(path.join(self.dataset_results_dir, 'Multimatch_models.csv')):
            saved_mm_models = pd.read_csv(path.join(self.dataset_results_dir, 'Multimatch_models.csv'))
            computed_models = saved_mm_models["Model"].unique()
        else:
            computed_models = []
            saved_mm_models = pd.DataFrame(columns=["MSS","Image","Model","MMvec","MMdir","MMlen","MMpos","MMtime"])        
        if all(model_metadata["Model"].isin(computed_models)):
            mm_models_df = saved_mm_models
        else:
            model_metadata = pd.concat([model_metadata,model_sizes], axis=1)
            model_metadata = model_metadata[~model_metadata["Model"].isin(computed_models)]
            mm_models_df = pd.concat(list(model_metadata.apply(lambda x: self.compute_on_subject(x), axis=1))).reset_index(drop=True)    
            mm_models_df = mm_models_df.groupby(["MSS","Image","Model"]).mean().reset_index()
            if not saved_mm_models.empty:
                mm_models_df = pd.concat([mm_models_df,saved_mm_models]).reset_index(drop=True)
            mm_models_df.to_csv(path.join(self.dataset_results_dir, 'Multimatch_models.csv'), index=False)
        # mm_df should be the inner join of mm_models_df and mm_humans_df using "Image","Model" and "MSS" as keys
        mm_df = mm_models_df.merge(mm_humans_df, on=["Image","Model","MSS"], suffixes=('_model', '_humans'))
        # Drop the time values     
        mm_df = mm_df.drop(["MMtime_model", "MMtime_humans"], axis=1)
        # Add 1 column with the mean of the values of the columns starting with MM and ending with model and another with the same but for humans
        mm_df["AvgMM_model"] = mm_df.filter(like="MM").filter(like="_model").mean(axis=1)
        mm_df["AvgMM_humans"] = mm_df.filter(like="MM").filter(like="_humans").mean(axis=1)

        # Con esto ya tengo la misma info que en multimatch_human_mean_per_image, pero para todos los modelos y para hacer el plot.
        # Ahora tengo que guardar los resultados en metrics.json
        # Puedo guardar este dataframe también (en la carpeta de resultados general como un csv), y meter un loader y un if arriba que se fije que estén todos los modelos en el df cargado, sino que compute lo que falta.
        self.save_results(constants.FILENAME, mm_df)
        self.plot_subplots(mm_df)
        


    def get_screen_and_receptive_size(self, model_info):
        model_scanpaths = utils.load_dict_from_json(path.join(self.dataset_results_dir, model_info["Model"], 'Scanpaths.json'))
        trial_info = list(model_scanpaths.values())[0]
        screen_size = utils.get_dims(trial_info, trial_info, key='image')
        receptive_size = utils.get_dims(trial_info, trial_info, key='receptive')
        return pd.Series([screen_size, receptive_size,model_info["Model"]], index=["ScreenSize", "ReceptiveSize","Model to compare"])

    def compute_on_subject(self, scanpaths_info):
        subjects_to_compare = self.scanpath_metadata[self.scanpath_metadata["Model"] == "Humans" ]
        subjects_to_compare = subjects_to_compare[subjects_to_compare["Index to compute"] > scanpaths_info["Index to compute"]]
        subject_multimatch_values = pd.concat(list(subjects_to_compare.apply(lambda x: self.compute_multimatch_on_subject_pair(scanpaths_info, x), axis=1))).reset_index(drop=True)
        subject_multimatch_values["Model"] = scanpaths_info["Model to compare"]
        return subject_multimatch_values

        
    def compute_multimatch_on_subject_pair(self, subject_info, subject_to_compare_info):
        screen_size = subject_info["ScreenSize"]
        receptive_size = subject_info["ReceptiveSize"]
        subject_path = path.join(self.dataset_results_dir, subject_info["Model"], subject_info["Scanpath_file"]) if subject_info["Model"] != "Humans" else path.join(self.human_scanpaths_dir, subject_info["Scanpath_file"])
        subject_scanpaths = utils.load_dict_from_json(subject_path)
        subject_to_compare_path = path.join(self.dataset_results_dir, subject_to_compare_info["Model"], subject_to_compare_info["Scanpath_file"]) if subject_to_compare_info["Model"] != "Humans" else path.join(self.human_scanpaths_dir, subject_to_compare_info["Scanpath_file"])
        subject_to_compare_scanpaths = utils.load_dict_from_json(subject_to_compare_path)
        # Get a list of the keys that are in both dictionaries
        common_images = list(set(subject_scanpaths.keys()) & set(subject_to_compare_scanpaths.keys()))
        # Filter the images where their "memory_set" has a size not present in filters_mss
        common_images = list(filter(lambda x: len(subject_scanpaths[x]["memory_set"]) not in self.filters_mss[subject_info["Model to compare"]] if 'memory_set' in subject_scanpaths[x] else True, common_images))

        # Get the multimatch for each image
        multimatch_values_per_image = list(map(lambda x: self.compute_multimatch(subject_scanpaths[x], subject_to_compare_scanpaths[x], screen_size, receptive_size) + [x], common_images))

        multimatch_values_per_image = list(filter(None, multimatch_values_per_image))
        multimatch_values_per_image = pd.DataFrame(multimatch_values_per_image, columns=["MSS","MMvec", "MMdir", "MMlen", "MMpos", "MMtime", "Image"])

        multimatch_values_per_image = multimatch_values_per_image.dropna().reset_index(drop=True)
        # MSS column should be ints
        multimatch_values_per_image["MSS"] = multimatch_values_per_image["MSS"].astype(int)  
        return multimatch_values_per_image

    def compute_multimatch(self, trial_info, trial_to_compare_info, screen_size, receptive_size):
        target_found = trial_info['target_found'] and trial_to_compare_info['target_found']
        if not target_found:
           return []

        trial_scanpath_X, trial_scanpath_Y = trial_info['X'], trial_info['Y']
        trial_to_compare_scanpath_X, trial_to_compare_scanpath_Y = trial_to_compare_info['X'], trial_to_compare_info['Y']



        # Rescale accordingly
        trial_scanpath_X, trial_scanpath_Y = utils.rescale_and_crop(trial_info, screen_size, receptive_size)
        trial_to_compare_scanpath_X, trial_to_compare_scanpath_Y = utils.rescale_and_crop(trial_to_compare_info, screen_size, receptive_size)

        trial_scanpath_length            = len(trial_scanpath_X)
        trial_to_compare_scanpath_length = len(trial_to_compare_scanpath_X)
        

        # Multimatch can't be computed for scanpaths with length shorter than 3
        if trial_scanpath_length < 3 or trial_to_compare_scanpath_length < 3:
            return []

        trial_scanpath_time            = utils.get_scanpath_time(trial_info, trial_scanpath_length)
        trial_to_compare_scanpath_time = utils.get_scanpath_time(trial_to_compare_info, trial_to_compare_scanpath_length)


        trial_scanpath = np.array(list(zip(trial_scanpath_X, trial_scanpath_Y, trial_scanpath_time)), dtype=[('start_x', '<f8'), ('start_y', '<f8'), ('duration', '<f8')])
        trial_to_compare_scanpath = np.array(list(zip(trial_to_compare_scanpath_X, trial_to_compare_scanpath_Y, trial_to_compare_scanpath_time)), dtype=[('start_x', '<f8'), ('start_y', '<f8'), ('duration', '<f8')])

        return [len(trial_info['memory_set']) if 'memory_set' in trial_info else 1] + mm.docomparison(trial_scanpath, trial_to_compare_scanpath, (screen_size[1], screen_size[0]))

    def plot_subplots(self, mm_df):

        mss_values = np.sort(mm_df.MSS.unique())
        mss_amounts = len(mss_values)

        number_of_models = len(mm_df[mm_df["Model"] != "Humans"]["Model"].unique())
        fig, axs = plt.subplots(mss_amounts, number_of_models, sharex=True, sharey=True, figsize=(1 + 3.5*number_of_models, 4*mss_amounts))

        if number_of_models == 1: axs = np.array([axs])
        if mss_amounts == 1: axs = np.array([axs])
        axs=axs.reshape((mss_amounts,number_of_models))
        # Group mm_df by model and mss and apply add_to_plot to each row (image)
        grouped = mm_df.groupby(["Model", "MSS"])
        model_indexes = mm_df["Model"].unique()        
        grouped.apply(lambda x: self.add_to_plot(axs[mss_values.tolist().index(x["MSS"].iloc[0])][model_indexes.tolist().index(x["Model"].iloc[0])],x))

        axs[0][0].set(ylabel='Human multimatch mean')
        axs[-1][int(axs.shape[1]/2)].set(xlabel='Model vs human multimatch mean')
        fig.suptitle(self.dataset_name + ' dataset')
        plt.tight_layout()
        plt.savefig(path.join(self.dataset_results_dir, 'Multimatch against humans.png'))
        plt.close(fig)

    def add_to_plot(self, ax, mm_values):        
        model_name = mm_values["Model"].iloc[0]
        color = self.color_model_mapping[model_name]
        # Sum the values of the columns starting with MM and ending with model and 
        mm_means_model = mm_values["AvgMM_model"]
        mm_means_humans = mm_values["AvgMM_humans"]
        # Set same scale for every dataset
        ax.set_ylim(0.65, 1.0)
        ax.set_xlim(0.65, 1.0)
        # Plot multimatch
        ax.scatter(mm_means_model, mm_means_humans, color=color, alpha=0.5)
        # Plot linear regression
        x_linear   = np.array(mm_means_model)[:, np.newaxis]
        m, _, _, _ = np.linalg.lstsq(x_linear, mm_means_humans, rcond=None)
        ax.plot(mm_means_model, m * x_linear, linestyle=(0, (5, 5)), color='purple', alpha=0.6)
        ax.set_title(model_name + ' MSS ' + str(mm_values["MSS"].iloc[0]))
        #Plot the diagonal
        ax.plot([0.65, 1.0], [0.65, 1.0], linestyle='dashed', c='.3')
        ax.label_outer()
        ax.set_box_aspect(1)

    
    def save_results(self, filename, mm_df):
        dataset_metrics_file = path.join(self.dataset_results_dir, filename)
        dataset_metrics      = utils.load_dict_from_json(dataset_metrics_file)
        mm_df.groupby(["Model", "MSS"]).apply(lambda x: self.save_model_results(dataset_metrics, x))
        utils.save_to_json(dataset_metrics_file, dataset_metrics)
    
    def save_model_results(self, dataset_metrics, multimatch_values):
        model_vs_humans_mean = multimatch_values["AvgMM_model"].values
        humans_mean          = multimatch_values["AvgMM_humans"].values

        corr_coef_pvalue = pearsonr(model_vs_humans_mean, humans_mean)
        
        # From the "AvgMM_model" column, get the mean only, and from the rest of the columns get the mean and std in a single row within a list
        mm_model_values = pd.Series(multimatch_values.filter(like="MM").filter(like="_model").agg(lambda x: [[x.mean().round(3), x.std().round(3)]],axis=0).round(3).iloc[0])
        mm_model_values["AvgMM_model"] = mm_model_values["AvgMM_model"][0]
        


        
        mm_model_values["Corr"] = round(corr_coef_pvalue[0],3)
        mm_humans_values = pd.Series(multimatch_values.filter(like="MM").filter(like="_humans").agg(lambda x: [[x.mean().round(3), x.std().round(3)]],axis=0).iloc[0])
        mm_humans_values["AvgMM_humans"] = mm_humans_values["AvgMM_humans"][0]
        # Remove _model and _humans from the columns
        mm_model_values.index = mm_model_values.index.str.replace("_model", "")
        mm_humans_values.index = mm_humans_values.index.str.replace("_humans", "")

        metrics = dict(zip(mm_model_values.index, mm_model_values.values))

        hmetrics = dict(zip(mm_humans_values.index, mm_humans_values.values))
        model = multimatch_values["Model"].iloc[0]
        mss = multimatch_values["MSS"].iloc[0]
        if model not in dataset_metrics:
            dataset_metrics[model] = {}
        utils.update_dict(dataset_metrics[model], "MSS "+str(mss), metrics)
        if "Humans" not in dataset_metrics:
            dataset_metrics["Humans"] = {}
        # Subjects' score is computed in grid size
        utils.update_dict(dataset_metrics["Humans"], "MSS " +str(mss), hmetrics)

    

    
class MultimatchSubjects(Multimatch):
    def __init__(self, dataset_name, scanpath_metadata, results_dir, human_scanpaths_dir, max_scanpath_length, filters_mss,colors_dict):
        self.dataset_name = dataset_name
        self.scanpath_metadata = scanpath_metadata
        self.dataset_results_dir = results_dir
        self.human_scanpaths_dir = human_scanpaths_dir
        self.filters_mss = filters_mss
        self.color_model_mapping = colors_dict
        self.models = self.scanpath_metadata['Model'][self.scanpath_metadata['Model'] != 'Humans'].unique()
        # Remove data from self.scanpath_metadata that is from models
        self.scanpath_metadata = self.scanpath_metadata[~self.scanpath_metadata['Model'].isin(self.models)]
        
        self.general_results_dir = path.dirname(self.dataset_results_dir)


    def get_models_metadata(self,subject_scanpaths_file):
        subject_name = subject_scanpaths_file.split('/')[-1][:-15]
        return pd.DataFrame(list(map(lambda x: self.run_model_for_subject(x,subject_name), self.models)),columns=["Scanpath_file","Model"])

    def run_model_for_subject(self,model_name,subject_name):

        model_results_path = path.join(self.dataset_results_dir, model_name,"subjects_predictions", "subject_"+ subject_name, 'Subject_scanpaths_only_memset.json')
        if not path.isfile(model_results_path):
            model = importlib.import_module('Model.main')
            model_filters_mss = self.filters_mss.get(model_name, [])
            model.main(self.dataset_name,model_name, subject_name,filters_mss=model_filters_mss,results_folder = self.general_results_dir)

        return [path.join("subjects_predictions","subject_"+ subject_name, 'Subject_scanpaths_only_memset.json'),model_name]

    def compute_on_subject(self, scanpaths_info):
        subjects_to_compare = self.scanpath_metadata[self.scanpath_metadata["Model"] == scanpaths_info["Model"]]
        subjects_to_compare = subjects_to_compare[subjects_to_compare["Index to compute"] > scanpaths_info["Index to compute"]]
        subject_multimatch_values = pd.concat(list(subjects_to_compare.apply(lambda x: self.compute_multimatch_on_subject_pair(scanpaths_info, x), axis=1))).reset_index(drop=True)
        subject_multimatch_values["Model"] = scanpaths_info["Model to compare"]
        return subject_multimatch_values
    
    def compute(self):
        print('Computing multimatch for ' + self.dataset_name + ' dataset')        
        # Do a cross join between human_metadata and the series of the unique models (excluding humans)
        model_metadata = pd.concat(list(map(lambda x: self.get_models_metadata(x), self.scanpath_metadata["Scanpath_file"].unique()))).reset_index(drop=True)
        self.scanpath_metadata = pd.concat([self.scanpath_metadata,model_metadata], axis=0).reset_index(drop=True)
        self.scanpath_metadata["Index to compute"] = self.scanpath_metadata.index
        model_metadata = self.scanpath_metadata[self.scanpath_metadata["Model"] != "Humans"]
        model_sizes = pd.DataFrame(self.models, columns=["Model"]).apply(lambda x: self.get_screen_and_receptive_size(x), axis=1)
        if path.isfile(path.join(self.dataset_results_dir, 'Multimatch_human_mean_per_image.csv')):
            saved_mm_humans = pd.read_csv(path.join(self.dataset_results_dir, 'Multimatch_human_mean_per_image.csv'))
            computed_models = saved_mm_humans["Model"].unique()
        else:
            computed_models = []
            saved_mm_humans = pd.DataFrame(columns=["MSS","Image","Model","MMvec","MMdir","MMlen","MMpos","MMtime"])
        # If the models in computed_models are the same as the ones in self.models, then mm_humans_df = saved_mm_humans
        if all(model_sizes["Model to compare"].isin(computed_models)):
            mm_humans_df = saved_mm_humans
        else:
            human_metadata = self.scanpath_metadata[self.scanpath_metadata["Model"] == "Humans"]
            # Remove the last row (the last subject multimatch is computed against all the others, so it is not needed)
            human_metadata = human_metadata.iloc[:-1]
            human_metadata = human_metadata.merge(model_sizes[~model_sizes["Model to compare"].isin(computed_models)], how="cross")
            mm_humans_df = pd.concat(list(human_metadata.apply(lambda x: self.compute_on_subject(x), axis=1))).reset_index(drop=True)
            mm_humans_df = mm_humans_df.groupby(["MSS","Image","Model"]).mean().reset_index()
            if not saved_mm_humans.empty:
                mm_humans_df = pd.concat([mm_humans_df,saved_mm_humans]).reset_index(drop=True)        
            mm_humans_df.to_csv(path.join(self.dataset_results_dir, 'Multimatch_human_mean_per_image.csv'), index=False)
        mm_humans_df = mm_humans_df[mm_humans_df["Model"].isin(model_metadata["Model"])].reset_index(drop=True)
        if path.isfile(path.join(self.dataset_results_dir, 'Multimatch_models.csv')):
            saved_mm_models = pd.read_csv(path.join(self.dataset_results_dir, 'Multimatch_models.csv'))
            computed_models = saved_mm_models["Model"].unique()
        else:
            computed_models = []
            saved_mm_models = pd.DataFrame(columns=["MSS","Image","Model","MMvec","MMdir","MMlen","MMpos","MMtime"])        
        if all(model_metadata["Model"].isin(computed_models)):
            mm_models_df = saved_mm_models
        else:
            # Group by model and remove the row that has the highest "Index to compute" value (the last one), then reset the index with drop=True
            model_metadata = model_metadata.groupby("Model").apply(lambda x: x.iloc[:-1]).reset_index(drop=True)
            model_metadata = model_metadata.merge(model_sizes, left_on="Model", right_on="Model to compare")            
            model_metadata = model_metadata[~model_metadata["Model"].isin(computed_models)]
            mm_models_df = pd.concat(list(model_metadata.apply(lambda x: self.compute_on_subject(x), axis=1))).reset_index(drop=True)    
            mm_models_df = mm_models_df.groupby(["MSS","Image","Model"]).mean().reset_index()
            if not saved_mm_models.empty:
                mm_models_df = pd.concat([mm_models_df,saved_mm_models]).reset_index(drop=True)
            mm_models_df.to_csv(path.join(self.dataset_results_dir, 'Multimatch_models.csv'), index=False)
        # mm_df should be the inner join of mm_models_df and mm_humans_df using "Image","Model" and "MSS" as keys
        mm_df = mm_models_df.merge(mm_humans_df, on=["Image","Model","MSS"], suffixes=('_model', '_humans'))
        # Drop the time values     
        mm_df = mm_df.drop(["MMtime_model", "MMtime_humans"], axis=1)
        # Add 1 column with the mean of the values of the columns starting with MM and ending with model and another with the same but for humans
        mm_df["AvgMM_model"] = mm_df.filter(like="MM").filter(like="_model").mean(axis=1)
        mm_df["AvgMM_humans"] = mm_df.filter(like="MM").filter(like="_humans").mean(axis=1)
        # Con esto ya tengo la misma info que en multimatch_human_mean_per_image, pero para todos los modelos y para hacer el plot.
        # Ahora tengo que guardar los resultados en metrics.json
        # Puedo guardar este dataframe también (en la carpeta de resultados general como un csv), y meter un loader y un if arriba que se fije que estén todos los modelos en el df cargado, sino que compute lo que falta.

        self.save_results(constants.FILENAME, mm_df)
        
        self.plot_subplots(mm_df)