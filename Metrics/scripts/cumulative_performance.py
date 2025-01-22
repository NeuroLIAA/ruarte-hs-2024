import matplotlib.pyplot as plt
import numpy as np
from . import utils
from scipy import integrate
from os import path
import pandas as pd
import importlib
from .. import constants
class CumulativePerformance:
    def __init__(self, dataset_name, scanpath_metadata, results_dir, human_scanpaths_dir, max_scanpath_length, filters_mss,colors_dict):
        self.dataset_name = dataset_name
        self.max_scanpath_length = min(max_scanpath_length,25)
        self.scanpath_metadata = scanpath_metadata
        self.dataset_results_dir = results_dir
        self.human_scanpaths_dir = human_scanpaths_dir
        self.filters_mss = filters_mss
        self.color_model_mapping = colors_dict        

    def compute(self):
        """
        Main computation handler that calculates cumulative performance metrics.
        """
        print(f"Computing cumulative performance for {self.dataset_name} dataset")
        cumulative_performance_df = pd.concat(self.scanpath_metadata.apply(self.compute_on_subject, axis=1).tolist())
        
        if self.dataset_name == 'Interiors':
            cumulative_performance_df, cumulative_performance_interiors_df = self.handle_interiors_dataset(cumulative_performance_df)
        else:
            cumulative_performance_interiors_df = pd.DataFrame()
        
        cumulative_performance_df = self.aggregate_performance(cumulative_performance_df)
        self.process_results(cumulative_performance_df, cumulative_performance_interiors_df)

    def handle_interiors_dataset(self, df):
        """
        Handles special case for 'Interiors' dataset.
        """
        interiors_df = df[df['Model'] == 'Humans'].copy()
        df = df[df['Model'] != 'Humans']
        interiors_df["Cumulative_performance"] = interiors_df["Cumulative_performance"].apply(lambda x: x[-1])
        interiors_df = interiors_df.groupby(['MSS', 'Scanpath_Length', 'Model','Color']).agg(list).reset_index()
        return df, interiors_df
 
    def aggregate_performance(self, df):
        """
        Aggregate performance data and compute mean and standard deviation.
        """
        df.drop(columns=['Scanpath_Length'],inplace=True)
        df = df.groupby(['MSS', 'Model','Color']).agg({"Cumulative_performance": [lambda x: np.mean(np.vstack(x), axis=0), lambda x: np.std(np.vstack(x), axis=0)]}).reset_index()
        df.columns = ['MSS', 'Model','Color', 'Cumulative_performance_mean', 'Cumulative_performance_std']
        
        return df
    
    def process_results(self,cumulative_performance_df,cumulative_performance_interiors_df):
        """
        Processes the cumulative performance results and generates plots.
        """
        self.plot_subplots(cumulative_performance_df,cumulative_performance_interiors_df,"Cumulative Performance")
        cumulative_performance_df.drop(columns=['Color','Cumulative_performance_std'],inplace=True)
        if self.dataset_name == 'Interiors':
            cumulative_performance_interiors_df.drop(columns=['Color','Scanpath_Length'],inplace=True)
            cumulative_performance_interiors_df["Cumulative_performance"] = cumulative_performance_interiors_df["Cumulative_performance"].apply(lambda x: np.mean(x,axis=0))
            cumulative_performance_interiors_df = cumulative_performance_interiors_df.groupby(['MSS','Model']).agg({'Cumulative_performance': list}).reset_index()
            cumulative_performance_interiors_df.columns = ['MSS','Model','Cumulative_performance_mean']       
            # Reorder the columns of cumulative_performance_interiors_df into the same order as cumulative_performance_df
            cumulative_performance_interiors_df = cumulative_performance_interiors_df[cumulative_performance_df.columns]
            cumulative_performance_df = pd.concat([cumulative_performance_df,cumulative_performance_interiors_df],axis=0)
        
        self.save_results(constants.FILENAME,cumulative_performance_df)

    def compute_on_subject(self, scanpaths_info):
        scanpaths_path = path.join(self.human_scanpaths_dir, scanpaths_info['Scanpath_file']) if scanpaths_info['Model'] =="Humans" else path.join(self.dataset_results_dir, scanpaths_info['Model'], 'Scanpaths.json')        
        cumulative_performance = self.get_cumulative_performance_df(scanpaths_info['Model'],scanpaths_path)
        cumulative_performance = cumulative_performance.groupby(['MSS','Scanpath_Length','Model']).agg({'Targets_found': 'mean'}).rename(columns={'Targets_found':'Cumulative_performance'}).reset_index()
        cumulative_performance['Color'] = cumulative_performance['Model'].apply(lambda x: self.color_model_mapping[x])
        return cumulative_performance
    
    def get_cumulative_performance_df(self,model_name,scanpaths_path):
        cumulative_performance = self.load_filter_and_compute_metric(model_name,scanpaths_path)
        return cumulative_performance
    
    def load_filter_and_compute_metric(self,model_name,scanpaths_path):
        scanpaths = utils.load_dict_from_json(scanpaths_path)
        model_filters_mss = self.filters_mss.get(model_name,[])
        useful_scanpaths = list(filter(lambda x: not len(x['memory_set']) in model_filters_mss if 'memory_set' in x else True, scanpaths.values()))
        cumulative_performance = list(map(lambda x: self.target_found_array(x, model_name), useful_scanpaths))
        return pd.DataFrame(cumulative_performance, columns=['Targets_found','MSS','Scanpath_Length','Model'])

    def plot_subplots(self,cumulative_performance_df,cumulative_performance_interiors_df,metric = "Cumulative Performance"):
        #Sort the MSS values into an array
        mss_values = np.sort(cumulative_performance_df.MSS.unique())
        labels = cumulative_performance_df['Model'].unique()
        # Amount of characters of the longest label
        char_amount = max(map(len,labels))
        mss_amounts = len(mss_values)
        fig, axs = plt.subplots(1,mss_amounts, sharex=True, sharey=True, figsize=(char_amount*0.1+3*mss_amounts,3))
        if mss_amounts == 1: axs = np.array([axs])
        # apply function to each row of the dataframe
        cumulative_performance_df.apply(lambda x: self.plot(x,axs[int(np.where(mss_values == x['MSS'])[0])],metric), axis=1)
        list(map(lambda x: x[1].set_title(f"MSS {mss_values[x[0]]}"), list(zip(range(mss_amounts), axs))))
        axs[0].set_ylabel(f'{metric}')
        axs[len(axs)//2].set_xlabel('Fixation Number')
        fig.suptitle(self.dataset_name + ' Dataset')

        plt.tight_layout()
        plt.savefig(path.join(self.dataset_results_dir, f'{metric}.png'))
        plt.close()

    def target_found_array(self, scanpath, model):
        """ 
        Compute cumulative performance based on a given scanpath
        """
        # Iterate through each scanpath
        scanpath_length = len(scanpath['X'])
        # Initialize array to hold the number of targets found at each fixation number
        if self.dataset_name == 'Interiors' and model == 'Humans':
            targets_found_at_fixation_number = np.zeros(scanpath['max_fixations'],dtype=int)
        else:
            targets_found_at_fixation_number = np.zeros(self.max_scanpath_length,dtype=int)
        # Check if the scanpath is within the max length and if target was found
        if scanpath_length <= self.max_scanpath_length and scanpath['target_found'] and scanpath_length > 0:
            # Update the count of targets found at each fixation number
            targets_found_at_fixation_number[scanpath_length-1:] += 1
        return (targets_found_at_fixation_number,len(scanpath['memory_set']) if 'memory_set' in scanpath else 1,len(targets_found_at_fixation_number),model)


    def plot(self,cumulative_performance, ax, metric):
        model = cumulative_performance['Model'] if type(cumulative_performance) == pd.Series else cumulative_performance['Model'].iloc[0]
        color = cumulative_performance['Color'] if type(cumulative_performance) == pd.Series else cumulative_performance['Color'].iloc[0]
        linewidth =  2 if model == "Humans" else 1

        if model == 'Humans' and self.dataset_name == 'Interiors':
            list_of_lists = cumulative_performance["Cumulative_performance"].tolist()
            ax.boxplot(list_of_lists, notch=True, vert=True, whiskerprops={'linestyle': (0, (5, 10)), 'color': color}, capprops={'color': color}, \
                boxprops={'color': color}, flierprops={'marker': '+', 'markeredgecolor': color}, medianprops={'color': color}, positions=[3,5,9,13])
        else:
            perf_mean = cumulative_performance['Cumulative_performance_mean']
            perf_std = cumulative_performance['Cumulative_performance_std']
            if metric == "Cumulative Performance":
                ax.plot(range(1, self.max_scanpath_length + 1), perf_mean, label=model, c=color,linewidth=linewidth)
                ax.fill_between(range(1, self.max_scanpath_length + 1), perf_mean - perf_std, perf_mean + perf_std, alpha=0.2, color=color)
            elif metric == "Coefficient of Variation":
                coefficient_of_variation = perf_std / perf_mean.clip(1e-6)
                #append a 0 to the coefficient of variation to match the length of the cumulative performance
                ax.plot(range(1, self.max_scanpath_length + 1), coefficient_of_variation, label=model, c=color,linewidth=linewidth)
          
    
    def save_results(self, filename,cumulative_performance_df):
        dataset_metrics_file = path.join(self.dataset_results_dir, filename)
        dataset_metrics      = utils.load_dict_from_json(dataset_metrics_file)
        cumulative_performance_df.apply(lambda x: self.compute_auc(x,dataset_metrics), axis=1)
        utils.save_to_json(dataset_metrics_file, dataset_metrics)
        
    def compute_auc(self, cumulative_performance, dataset_metrics = {}):
        values = cumulative_performance['Cumulative_performance_mean']
        subject = cumulative_performance['Model']
        mss = cumulative_performance['MSS']
        fixations = np.linspace(0, 1, num=len(values))        
        auc = integrate.trapezoid(y=values, x=fixations)
        if subject not in dataset_metrics:
            dataset_metrics[subject] = {}
        utils.update_dict(dataset_metrics[subject], "MSS "+str(mss), {'AUCperf': np.round(auc, 3)})
        return auc        

class CumulativePerformanceSubjects(CumulativePerformance):


    def __init__(self, dataset_name, scanpath_metadata, results_dir, human_scanpaths_dir, max_scanpath_length, filters_mss,colors_dict):
        super().__init__(dataset_name, scanpath_metadata, results_dir, human_scanpaths_dir, max_scanpath_length, filters_mss,colors_dict)
        self.models = self.scanpath_metadata['Model'][self.scanpath_metadata['Model'] != 'Humans'].unique()
        # Remove data from self.scanpath_metadata that is from models
        self.scanpath_metadata = self.scanpath_metadata[~self.scanpath_metadata['Model'].isin(self.models)]   
        self.general_results_dir = path.dirname(self.dataset_results_dir)


    def process_results(self, cumulative_performance_df,cumulative_performance_interiors_df):
        self.plot_subplots(cumulative_performance_df,cumulative_performance_interiors_df,"Coefficient of Variation")
        super().process_results(cumulative_performance_df,cumulative_performance_interiors_df)
        
    
    def get_cumulative_performance_df(self,human_model_name,scanpaths_path):
        subject_name = scanpaths_path.split('/')[-1][:-15]
        model_cumulative_performance = pd.concat(list(map(lambda x: self.run_model_for_subject(x,subject_name), self.models)))
        cumulative_performance = self.load_filter_and_compute_metric(human_model_name,scanpaths_path)
        cumulative_performance = pd.concat([cumulative_performance,model_cumulative_performance])
        return cumulative_performance


    def run_model_for_subject(self,model_name,subject_name):
        model_filters_mss = self.filters_mss.get(model_name,[])
        model_results_path = path.join(self.dataset_results_dir, model_name,"subjects_predictions", "subject_"+ subject_name, 'Subject_scanpaths_only_memset.json')
        if not path.isfile(model_results_path):
            model = importlib.import_module('Model.main')
            model.main(self.dataset_name,model_name, subject_name,filters_mss=model_filters_mss,results_folder = self.general_results_dir)

        return self.load_filter_and_compute_metric(model_name,model_results_path)

