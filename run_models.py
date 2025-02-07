import constants
import argparse
import utils
import importlib
import Metrics.main as metrics_module


def main(datasets, models, metrics, results_folder,force_execution = False,filters_mss = {}):
    for dataset_name in datasets:
        for model_name in models:
            if force_execution: utils.delete_precomputed_results(dataset_name, model_name,results_folder)
            model_filters_mss = filters_mss.get(model_name,[])
            if utils.found_precomputed_results(dataset_name, model_name,results_folder):
                print('Found precomputed results for ' + model_name + ' on ' + dataset_name + ' dataset')
                continue
                
            print('Running ' + model_name + ' on ' + dataset_name + ' dataset')
            model = importlib.import_module('Model.main')
            model.main(dataset_name, model_name,filters_mss=model_filters_mss,results_folder=results_folder)
    
    if metrics:
        metrics_module.main(datasets, models,metrics,results_folder,filters_mss)

if __name__ == "__main__":
    available_models   = utils.get_files(constants.MODELS_PATH)
    available_datasets = utils.get_dirs(constants.DATASETS_PATH)
    available_metrics  = constants.AVAILABLE_METRICS
    parser = argparse.ArgumentParser(description='Run the model with a given set of configs on specific datasets and compute the corresponding metrics')
    parser.add_argument('--d', '--datasets', type=str, nargs='*', default=available_datasets, help='Names of the datasets on which to run the models. \
        Values must be in list: ' + str(available_datasets))
    parser.add_argument('--m', '--models', type=str, nargs='*', default=available_models, help='Names of the configs to run. \
        Values must be in list: ' + str(available_models))
    parser.add_argument('--mts', '--metrics', type=str, nargs='*', default=available_metrics, help='Names of the metrics to compute. \
        Values must be in list: ' + str(available_metrics) + '. Leave blank to not run any. WARNING: If not precomputed, human scanpath prediction (hsp) will take a LONG time!')
    parser.add_argument('--f', '--force', action='store_true', help='Deletes all precomputed results and forces models\' execution.')

    args = parser.parse_args()
    invalid_models   = not all(model in available_models for model in args.m)
    invalid_datasets = not all(dataset in available_datasets for dataset in args.d)
    invalid_metrics  = not all(metric in available_metrics for metric in args.mts)
    if (not args.m or invalid_models) or (not args.d or invalid_datasets) or invalid_metrics:
        raise ValueError('Invalid set of models, datasets or metrics')

    main(args.d, args.m, args.mts,constants.RESULTS_PATH, args.f)