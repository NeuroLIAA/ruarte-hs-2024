from os import path,remove
import json
import numpy as np
import run_models as rm
from copy import deepcopy
from sklearn.model_selection import ParameterGrid

def main():
    config_folder = path.join("Model","configs")

    base_config_file = "elm_base" #puede ser otro pero este es el de elm

    searches = {
                "peripheral_visibility":{
                    "peripheral_exponent":np.array([0.125,0.2,0.3,0.4,0.5,1.000]),
                    "fovea_filter":np.array([True],dtype=bool),
                },
                "wm_fixed_limit" :{
                    "history_size":np.array([1,2,4,6,8,12],dtype=np.int32),
                    "prior_as_fixation":np.array([True],dtype=bool),
                    "peripheral_exponent":np.array([0.2]),
                    "fovea_filter":np.array([True],dtype=bool),
                },
                "wm_degradation":{
                    "history_size":np.array([2,4,8,16],dtype=np.int32),
                    "history_degradation":np.array([True],dtype=bool),
                    "prior_as_fixation":np.array([True],dtype=bool),
                    "peripheral_exponent":np.array([0.2]),
                    "fovea_filter":np.array([True],dtype=bool),
                },
                "target_selection":{
                    "history_size":np.array([8],dtype=np.int32),
                    "history_degradation":np.array([True],dtype=bool),
                    "prior_as_fixation":np.array([True],dtype=bool),
                    "peripheral_exponent":np.array([0.2]),
                    "fovea_filter":np.array([True],dtype=bool),
                    "target_index_selector":np.array(["Random","CorrectTarget","LikelihoodMean","MinEntropy2D","MinEntropy"]),
                },


    }


    filters_mss_experiments = { "target_selection":[1],
                    "primacy_recency":[1,2],
                    "seen_targets_weights":[1,2],
                    "peripheral_visibility":[2,4],
                    "wm_fixed_limit":[2,4],
                    "forget_prior":[2,4],
                    "wm_degradation":[2,4],}

    with open(path.join(config_folder,base_config_file+".json"), 'r') as json_file:
        base_config =  json.load(json_file)



    datasets=["train_set"]
    metrics =["perf","mm","hsp"]
    models = [base_config_file]
    filters_mss = {}

    for search_name in searches.keys():

        search = searches[search_name]
        config = deepcopy(base_config)
        excluded_hyperparameters = []

        for hyperparameter in list(ParameterGrid(search))[0].keys():
            # If it has the same value for all the combinations, it is not necessary to include it in the config file
            if len(set(map(lambda x: x[hyperparameter],list(ParameterGrid(search)))))==1:
                excluded_hyperparameters.append(hyperparameter)     

        for combination in list(ParameterGrid(search)):
            config_name = search_name
            for hyperparameter in combination.keys():
                if isinstance(combination[hyperparameter], (np.int32)):
                    combination[hyperparameter] = int(combination[hyperparameter])
                if isinstance(combination[hyperparameter], (np.bool_)):
                    combination[hyperparameter] = bool(combination[hyperparameter])
                if hyperparameter in excluded_hyperparameters:
                    continue    
                config_name = f'{config_name}-{"".join(list(map(lambda x: x[0].capitalize(),hyperparameter.split("_"))))}_'
                if isinstance(combination[hyperparameter], (np.float64,np.float32)):
                    config_name = config_name+f'{combination[hyperparameter]:.3f}'
                else:
                    config_name = config_name+f'{"".join(c for c in str(combination[hyperparameter]).capitalize() if c.isupper() or c.isdigit())}'
            filters_mss[config_name] = filters_mss_experiments[search_name]         
            config["name"] = config_name        
            config.update(combination)
            with open(path.join(config_folder,config_name+".json"), 'w') as json_file:
                json.dump(config, json_file, indent=4)
            models.append(config_name)

    models = list(set(models))
    models.sort()

    rm.main(datasets,models,metrics,path.join("Experiments", "paper"),filters_mss = filters_mss)


    for config_name in models:
        if config_name != base_config_file and path.exists(path.join(config_folder,config_name+".json")):
            remove(path.join(config_folder,config_name+".json"))
        
if __name__ == "__main__":
    main()