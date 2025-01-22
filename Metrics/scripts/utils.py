import json
import random
import re
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, path, scandir, makedirs
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from .. import constants
from collections import defaultdict
from matplotlib import gridspec

def plot_table(dfs, title, save_path, filename):
    for key,df in dfs.items():
        df = df.infer_objects(copy=False).fillna("-")

        # Calculate the maximum number of characters per row
        max_characters_per_row = df.apply(lambda row: max([len(str(value)) for value in row]), axis=1).max()

        # Calculate the appropriate figsize based on the total number of characters

        figsize = (max_characters_per_row * 0.02, 1 + len(df) * 0.25)
        fig, ax = plt.subplots(figsize=figsize)

        ax.axis('off')

        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            rowLabels=df.index,
            colColours=['peachpuff'] * len(df.columns),
            rowColours=['peachpuff'] * len(df.index),
            loc='center'
        )



        # Set column widths based on the maximum width of the contents
        for i, col in enumerate(df.columns):
            table.auto_set_column_width([i])
            
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.5, 1.5)

        fig.suptitle(title + " " + key, fontsize=16)
        
        file_name_temp = filename.split(".")
        file_name_temp = file_name_temp[0] + "_" + key + "." + file_name_temp[1]
        # Save the figure with bbox_inches='tight'
        plt.savefig(path.join(save_path, file_name_temp), bbox_inches='tight')
        
        plt.clf()
        plt.close(fig)

def average_results(datasets_results_dict, save_path, filename):
    results_average = {}
    metrics = ['AUCperf', 'Corr', 'AvgMM', 'AUChsp', 'NSShsp', 'IGhsp', 'LLhsp']
    number_of_datasets = len(datasets_results_dict)
    for dataset in datasets_results_dict:
        dataset_res   = datasets_results_dict[dataset]

        for model in dataset_res:
            if not model in results_average:
                results_average[model] = {}
            
            for mss in dataset_res[model]:
                if not mss in results_average[model]:
                    results_average[model][mss] = {}

                for metric in metrics:
                    if metric in dataset_res[model][mss]:
                        if metric in results_average[model][mss]:
                            results_average[model][mss][metric] += dataset_res[model][mss][metric] / number_of_datasets
                        else:
                            results_average[model][mss][metric] = dataset_res[model][mss][metric] / number_of_datasets
    
    final_table = create_table(results_average)
    save_to_json(path.join(save_path, filename), results_average)

    return final_table

def create_table(results_dict):
    add_score(results_dict)

    # Create a table with the results
    table = pd.DataFrame.from_dict({(model, mss): results_dict[model][mss] for model in results_dict for mss in results_dict[model]}, orient='index')

    # Move Score to the last position
    table = table[[col for col in table.columns if col != 'Score'] + ['Score']]

    # Divide the table into different tables based on the mss
    tables = {}
    for mss in table.index.get_level_values(1).unique():
        tables[mss] = table.xs(mss, level=1)
    for key in tables:
        tables[key] = tables[key].sort_values(by='Score', ascending=False)
    return tables

def scores_plots(results_dict, colors_dict, save_path, filename):
    amounts_mss = set()
    excluded_models = ['Humans', 'gold_standard', 'center_bias', 'uniform']
    useful_models = [model for model in results_dict if not (model in excluded_models) and (model in colors_dict)]
    
    model_count_per_mss = {}  # Track how many models exist per MSS
    for model in useful_models:        
        for mss in results_dict[model]:
            amounts_mss.add(mss)
            if mss not in model_count_per_mss:
                model_count_per_mss[mss] = 0
            model_count_per_mss[mss] += 1  # Count models for each MSS
    
    amounts_mss = list(amounts_mss)
    amounts_mss.sort()

    # Define width ratios based on the number of models in each mss
    width_ratios = [model_count_per_mss[mss] for mss in amounts_mss]

    # Create the figure and GridSpec with dynamic width_ratios
    fig = plt.figure(figsize=(2*len(amounts_mss) + len(useful_models)*0.15, 2))
    gs = gridspec.GridSpec(1, len(amounts_mss), width_ratios=width_ratios)

    axs = []
    for idx, mss in enumerate(amounts_mss):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_title(str(mss))
        ax.set_ylabel('Score')
        ax.set_xticks(range(len(useful_models)))
        ax.set_xticklabels(useful_models, rotation=45)
        # Hide x labels
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
        axs.append(ax)
    dot_size = 100  # Adjust size or calculate dynamically
    # Plot the scores for each model
    for idx, mss in enumerate(amounts_mss):
        ax = axs[idx]
        
        scores = []
        x_values = []  # Numerical x-values for the plot
        colors = []
        index = 0
        # Plot the scatter dots for each model
        for i, model in enumerate(useful_models):
            if mss in results_dict[model]:
                scores.append(results_dict[model][mss]['Score'])
                x_values.append(index)  # Use the index of the model as the x-value
                colors.append(colors_dict[model])
                index += 1
        # Order the three lists by the scores
        x_values = [x for x in sorted(range(len(scores)), key=lambda x: scores[x])]
        colors = [colors[i] for i in x_values]
        scores = [scores[i] for i in x_values]
        x_values = list(range(len(scores)))

        for i in range(len(scores)):
            ax.scatter(x_values[i], scores[i], s=dot_size, color=colors[i], zorder=2)
        # Plot the dashed line connecting the points
        ax.plot(x_values, scores, color='black', zorder=1, linestyle='dashed')
        # The y axis should end at 0 and begin at the minimum score
        ax.set_ylim(min(scores) - 0.02, 0)

    plt.tight_layout()
    # The background should be transparent
    plt.savefig(path.join(save_path, filename), transparent=True)

def rose_plot(results_dict,colors_dict,save_path,filename):

    metrics = ['AUCperf', 'AvgMM', 'Corr', 'AUChsp', 'NSShsp', 'IGhsp', 'LLhsp']
    reference_models = {'AUCperf': 'Humans', 'AvgMM': 'Humans', \
        'AUChsp': 'gold_standard', 'NSShsp': 'gold_standard', 'IGhsp': 'gold_standard', 'LLhsp': 'gold_standard'}
    
    excluded_models = ['Humans', 'gold_standard','center_bias','uniform']

    scores = defaultdict(defaultdict)
    # Number of metrics
    num_metrics = len(metrics)

    amounts_mss = set()

    for model in [model for model in results_dict if (not model in excluded_models) and model in colors_dict]:        
        for mss in results_dict[model]:
            data = {}
            amounts_mss.add(mss)            
            metrics_values = results_dict[model][mss]
            for metric in metrics:
                if not metric in reference_models:
                    data[metric] = metrics_values[metric] -1 
                else:
                    reference_value = results_dict[reference_models[metric]][mss][metric]
                    if metric in ['AUCperf']:
                        data[metric] = (- abs(reference_value - metrics_values[metric]))
                    else:
                        data[metric] = (metrics_values[metric] - reference_value) / reference_value
                data[metric] = np.round(data[metric], 3)
            scores[mss][model] = data
    amounts_mss = list(amounts_mss)
    amounts_mss.sort()
    # Define angles for each metric (evenly spaced)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Set up the figure
    fig, axs = plt.subplots(1,len(amounts_mss),figsize=(3*len(amounts_mss), 3), subplot_kw=dict(polar=True))

    if len(amounts_mss) == 1:
        axs = np.array([axs])
    # Function to add the plot for each model
    for mss in scores:
        models = list(scores[mss].keys())
        # Sort models by their score
        models.sort(key=lambda x: results_dict[x][mss]['Score'])
        if len(models) > 3:
            alphas = np.full(len(models), 0.5)
        else:
            alphas = np.full(len(models), 1.0)
        alphas[-1] = 1.0
        for model in models:
            values = list(scores[mss][model].values())
            values += values[:1]  # Complete the loop
            ax = axs[amounts_mss.index(mss)]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model,color=colors_dict[model],alpha=alphas[models.index(model)])

            
            
    # Adjust label spacing with labelpad
    for ax in axs:
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('center')
            label.set_verticalalignment('center')
        # You can tweak spacing here by adjusting the position using 'pad'
        ax.tick_params(pad=20)
        ax.set_title(str(amounts_mss[axs.tolist().index(ax)]))
        # Hide values
        ax.set_yticklabels([])

    plt.tight_layout()
    plt.savefig(path.join(save_path, filename), transparent=True)


def add_score(results_dict):
    # For each metric, different models are used as reference values
    reference_models = {'AUCperf': 'Humans', 'AvgMM': 'Humans', \
        'AUChsp': 'gold_standard', 'NSShsp': 'gold_standard', 'IGhsp': 'gold_standard', 'LLhsp': 'gold_standard'}

    # Only the average across dimensions is used for computing the score
    excluded_metrics = ['MMvec', 'MMdir', 'MMpos', 'MMlen','MMtime']
    
    for model in results_dict:
        for mss in results_dict[model]:
            score = 0.0
            number_of_metrics = 0
            metrics_values = results_dict[model][mss]
            valid_metrics  = [metric_name for metric_name in metrics_values if metric_name not in excluded_metrics]
            for metric in valid_metrics:
                if not metric in reference_models:
                    score += metrics_values[metric] -1 
                else:    
                    reference_value = results_dict[reference_models[metric]][mss][metric]
                    if metric == 'AUCperf':
                        # AUCperf is expressed as 1 subtracted the absolute difference between Human and model's AUCperf, maximizing the score of those models who were closest to human subjects
                        score += - abs(reference_value - metrics_values[metric])
                    else:
                        score += (metrics_values[metric] - reference_value) / reference_value
                metrics_values[metric] = np.round(metrics_values[metric], 3)
                number_of_metrics += 1
            results_dict[model][mss]['Score'] = np.round(score / number_of_metrics, 3)



def create_dirs(filepath):
    dir_ = path.dirname(filepath)
    if len(dir_) > 0 and not path.exists(dir_):
        makedirs(dir_)  

def dir_is_too_heavy(path):
    nmbytes = sum(d.stat().st_size for d in scandir(path) if d.is_file()) / 2**20
    
    return nmbytes > constants.MAX_DIR_SIZE

def is_contained_in(json_file,csv_file,column_name):
    if not (path.exists(csv_file) and path.exists(json_file)):
        return False
    
    dict = load_dict_from_json(json_file)
    csv_column = pd.read_csv(csv_file)[column_name].unique()

    return all( elem in csv_column for elem in dict.keys())

def list_files_ending(path, extension):
    folder_files = listdir(path)
    filtered_files = []
    for file in folder_files:
        if file.endswith(extension):
            filtered_files.append(file)
    
    return filtered_files


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def get_dirs(path_):
    files = listdir(path_)
    dirs  = [dir_ for dir_ in files if path.isdir(path.join(path_, dir_))]

    return dirs

def divide_by_memory_set_size(trials_dict):
    scanpaths_per_mss = {}
    for (key,val) in trials_dict.items():
        if not "memory_set" in val:
            mss = 1
        else:
            mss = len(val["memory_set"])
        if not mss in scanpaths_per_mss:
            scanpaths_per_mss[mss] = {key : val}
        else:
            scanpaths_per_mss[mss][key] = val
    return scanpaths_per_mss



def get_random_subset(trials_dict, size):
    if len(trials_dict) <= size:
        return trials_dict
    
    random.seed(constants.RANDOM_SEED)

    return dict(random.sample(trials_dict.items(), size))

def update_dict(dic, key, data):
    if key in dic:
        dic[key].update(data)
    else:
        dic[key] = data

def load_pickle(pickle_filepath):
    with open(pickle_filepath, 'rb') as fp:
        return pickle.load(fp)  

def load_dict_from_json(json_file_path):
    if not path.exists(json_file_path):
        return {}
    else:
        with open(json_file_path, 'r') as json_file:
            return json.load(json_file)

def save_to_pickle(data, filepath):
    create_dirs(filepath)

    with open(filepath, 'wb') as fp:
        pickle.dump(data, fp)

def save_to_json(filepath, data):
    create_dirs(filepath)

    with open(filepath, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def save_to_csv(data, filepath):
    create_dirs(filepath)

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

def get_dims(model_trial, subject_trial, key):
    " Lower bound for receptive size "
    if key == 'receptive' and model_trial[key + '_width'] > subject_trial[key + '_width']:
        return (subject_trial[key + '_height'], subject_trial[key + '_width'])
    
    return (model_trial[key + '_height'], model_trial[key + '_width'])

def get_scanpath_time(trial_info, length):
    if 'T' in trial_info:
        scanpath_time = [t * 0.0001 for t in trial_info['T']]
    else:
        # Dummy
        scanpath_time = [0.3] * length
    
    return scanpath_time

def rescale_and_crop(trial_info, new_size, receptive_size):
    trial_scanpath_X = [rescale_coordinate(x, trial_info['image_width'], new_size[1]) for x in trial_info['X']]
    trial_scanpath_Y = [rescale_coordinate(y, trial_info['image_height'], new_size[0]) for y in trial_info['Y']]

    image_size       = (trial_info['image_height'], trial_info['image_width'])
    target_bbox      = trial_info['target_bbox']
    target_bbox      = [rescale_coordinate(target_bbox[i], image_size[i % 2 == 1], new_size[i % 2 == 1]) for i in range(len(target_bbox))]

    trial_scanpath_X, trial_scanpath_Y = collapse_fixations(trial_scanpath_X, trial_scanpath_Y, receptive_size)
    trial_scanpath_X, trial_scanpath_Y = crop_scanpath(trial_scanpath_X, trial_scanpath_Y, target_bbox, receptive_size)

    return trial_scanpath_X, trial_scanpath_Y        

def rescale_coordinate(value, old_size, new_size):
    return int((value / old_size) * new_size)

def between_bounds(target_bbox, fix_y, fix_x, receptive_size):
    return target_bbox[0] <= fix_y + receptive_size[0] // 2 and target_bbox[2] >= fix_y - receptive_size[0] // 2 and \
        target_bbox[1] <= fix_x + receptive_size[1] // 2 and target_bbox[3] >= fix_x - receptive_size[1] // 2

def crop_scanpath(scanpath_x, scanpath_y, target_bbox, receptive_size):
    index = 0
    for fixation in zip(scanpath_y, scanpath_x):
        if between_bounds(target_bbox, fixation[0], fixation[1], receptive_size):
            break
        index += 1
    
    cropped_scanpath_x = list(scanpath_x[:index + 1])
    cropped_scanpath_y = list(scanpath_y[:index + 1])
    return cropped_scanpath_x, cropped_scanpath_y

def collapse_fixations(scanpath_x, scanpath_y, receptive_size):
    collapsed_scanpath_x = list(scanpath_x)
    collapsed_scanpath_y = list(scanpath_y)
    index = 0
    while index < len(collapsed_scanpath_x) - 1:
        abs_difference_x = [abs(fix_1 - fix_2) for fix_1, fix_2 in zip(collapsed_scanpath_x, collapsed_scanpath_x[1:])]
        abs_difference_y = [abs(fix_1 - fix_2) for fix_1, fix_2 in zip(collapsed_scanpath_y, collapsed_scanpath_y[1:])]

        if abs_difference_x[index] < receptive_size[1] / 2 and abs_difference_y[index] < receptive_size[0] / 2:
            new_fix_x = (collapsed_scanpath_x[index] + collapsed_scanpath_x[index + 1]) / 2
            new_fix_y = (collapsed_scanpath_y[index] + collapsed_scanpath_y[index + 1]) / 2
            collapsed_scanpath_x[index] = new_fix_x
            collapsed_scanpath_y[index] = new_fix_y
            del collapsed_scanpath_x[index + 1]
            del collapsed_scanpath_y[index + 1]
        else:
            index += 1

    return collapsed_scanpath_x, collapsed_scanpath_y

def aggregate_scanpaths(subjects_scanpaths_path, image_name, excluded_subject='None'):
    subjects_scanpaths_files = sorted_alphanumeric(listdir(subjects_scanpaths_path))

    scanpaths_X = []
    scanpaths_Y = []
    for subject_file in subjects_scanpaths_files:
        if excluded_subject in subject_file:
            continue
        subject_scanpaths = load_dict_from_json(path.join(subjects_scanpaths_path, subject_file))
        if image_name in subject_scanpaths:
            trial = subject_scanpaths[image_name]
            scanpaths_X += trial['X']
            scanpaths_Y += trial['Y']

    scanpaths_X = np.array(scanpaths_X)
    scanpaths_Y = np.array(scanpaths_Y)

    return scanpaths_X, scanpaths_Y

def search_bandwidth(values, shape, splits=5):
    """ Perform a grid search to look for the optimal bandwidth (i.e. the one that maximizes log-likelihood) """
    # Define search space (values estimated from previous executions)
    if np.log(shape[0] * shape[1]) < 10:
        bandwidths = 10 ** np.linspace(-1, 1, 100)
    else:
        bandwidths = np.linspace(15, 70, 200)
    
    n_splits = min(values.shape[0], splits)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths}, n_jobs=1, cv=n_splits)
    grid.fit(values)

    return grid.best_params_['bandwidth']

def load_center_bias_fixations(model_size):
    center_bias_fixs = load_dict_from_json(constants.CENTER_BIAS_FIXATIONS)
    scanpaths_X = np.array([rescale_coordinate(x, constants.CENTER_BIAS_SIZE[1], model_size[1]) for x in center_bias_fixs['X']])
    scanpaths_Y = np.array([rescale_coordinate(y, constants.CENTER_BIAS_SIZE[0], model_size[0]) for y in center_bias_fixs['Y']])

    return scanpaths_X, scanpaths_Y

def gaussian_kde(scanpaths_X, scanpaths_Y, shape, bandwidth=None):
    values = np.vstack([scanpaths_Y, scanpaths_X]).T

    if bandwidth is None:
        bandwidth = search_bandwidth(values, shape)

    X, Y = np.mgrid[0:shape[0], 0:shape[1]] + 0.5
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    gkde   = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(values)
    scores = np.exp(gkde.score_samples(positions))

    return scores.reshape(shape)

def plot_legends_and_labels(dataset_results,colors_dict, save_path, filename):
    excluded_models = ['Humans', 'gold_standard','center_bias','uniform']
    mss_amounts = set()
    useful_models = [model for model in dataset_results if not (model in excluded_models) and (model in colors_dict)]
    for model in useful_models:
        for mss in dataset_results[model]:
            mss_amounts.add(mss)
    mss_amounts = list(mss_amounts)
    mss_amounts.sort()
    fig, axs = plt.subplots(1,len(mss_amounts), figsize=(1 + 3*len(mss_amounts), 8))
    if len(mss_amounts) == 1:
        axs = np.array([axs])
    for model in useful_models:
        for mss in dataset_results[model]:
            ax = axs[mss_amounts.index(mss)]
            ax.plot([], [], label=model, color=colors_dict[model])

    for ax in axs:
        ax.legend(loc='center')
        ax.axis('off')
    plt.savefig(path.join(save_path, filename))
    plt.clf()
    plt.close(fig)

