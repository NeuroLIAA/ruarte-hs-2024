from os import path
DATASETS_PATH = 'Datasets'
MODELS_PATH   = path.join('Model', 'configs')
METRICS_PATH  = 'Metrics'
RESULTS_PATH  = 'Results'

AVAILABLE_METRICS = ['perf', 'mm', 'hsp','perfs','mms','sa','rf'] # Cumulative performance; Multi-Match; Human scanpath prediction

