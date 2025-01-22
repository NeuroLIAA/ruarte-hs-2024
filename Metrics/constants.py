from os import path
from .scripts.multimatch import Multimatch,MultimatchSubjects
from .scripts.human_scanpath_prediction import HumanScanpathPrediction
from .scripts.cumulative_performance import CumulativePerformance, CumulativePerformanceSubjects
from .scripts.saccade_amplitude import SaccadeAmplitude
from .scripts.re_fixations import ReFixations
# All paths are relative to root

DATASETS_PATH = 'Datasets'


RANDOM_SEED = 1234
MAX_DIR_SIZE = 10 # MBytes, for probability maps dirs

FILENAME = 'Metrics.json'

# Constants for baseline models
CENTER_BIAS_PATH      = path.join('Metrics', 'center_bias')
CENTER_BIAS_FIXATIONS = path.join(CENTER_BIAS_PATH, 'cat2000_fixations.json')
CENTER_BIAS_SIZE      = (1080, 1920)

# To ensure models have the same color in the plots across all datasets

HUMANS_COLOR  = '#000000'

NAME_METRIC_MAP = {
    'perf': CumulativePerformance,
    'hsp': HumanScanpathPrediction,
    'mm': Multimatch,
    'perfs': CumulativePerformanceSubjects,
    'mms': MultimatchSubjects,
    'sa' : SaccadeAmplitude,
    'rf' : ReFixations
}