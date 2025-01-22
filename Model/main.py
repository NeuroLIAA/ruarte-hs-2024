from .scripts import loader, constants
import argparse
from .visualsearch import visual_searcher as vs
import sys
from os import path

" Runs visualsearch/visual_searcher.py according to the supplied parameters "

def setup_and_run(dataset_name, config_name, image_name, image_range, human_subject, number_of_processes, results_folder, config_dir,filters_mss,follow_human_scanpath=False):
     dataset_path = path.join(constants.DATASETS_PATH, dataset_name)
     trials_properties_file = path.join(dataset_path, 'trials_properties.json')

     dataset_info      = loader.load_dataset_info(dataset_path)
     human_scanpaths   = loader.load_human_scanpaths(dataset_info['scanpaths_dir'], human_subject)
     config            = loader.load_config(config_dir, config_name, constants.IMAGE_SIZE, dataset_info['max_scanpath_length'], number_of_processes, human_scanpaths, follow_human_scanpath)
     output_path       = path.join(results_folder, path.join(dataset_name + '_dataset', config_name))
     output_path       = loader.create_output_folders(output_path, image_name, image_range, human_subject)
     checkpoint        = loader.load_checkpoint(output_path, config)
     trials_properties = loader.load_trials_properties(trials_properties_file, image_name, image_range, human_scanpaths, checkpoint)
     loader.print_config(config)
     if human_scanpaths:
          visual_searcher = vs.VisualSearcherSubject(config, dataset_info, trials_properties, output_path, human_scanpaths,constants.SIGMA,filters_mss,follow_human_scanpath)
     else:
          visual_searcher = vs.VisualSearcher(config, dataset_info, trials_properties, output_path, constants.SIGMA,filters_mss)
     # Agregado para tener un resultado intermedio
     total_founds, total_scanpaths = visual_searcher.run()
     return total_founds, total_scanpaths

""" Main method, added to be polymorphic with respect to the other models """
def main(dataset_name, config="elm", human_subject=None,filters_mss=[], results_folder ="Results", config_dir=constants.CONFIG_DIR,follow_human_scanpath=False):
     print(config)
     # Agregado para tener un resultado intermedio
     total_founds, total_scanpaths= setup_and_run(dataset_name, config_name=config, image_name=None, image_range=None, human_subject=human_subject,
                                                  number_of_processes='all', results_folder=results_folder, config_dir=config_dir,filters_mss=filters_mss,follow_human_scanpath=follow_human_scanpath)
     return total_founds, total_scanpaths

if __name__ == "__main__":
     parser = argparse.ArgumentParser(description='Run the nnIBS Visual Search model')
     parser.add_argument('-dataset', type=str, help='Name of the dataset on which to run the model. Value must be one of cIBS, COCOSearch18, IVSN or MCS.')
     parser.add_argument('--cfg', '--config', type=str, default='default', help='Name of configuration setup. Examples: greedy, ssim, ivsn. Default is ivsn.', metavar='cfg')
     group = parser.add_mutually_exclusive_group()
     group.add_argument('--img', '--image_name', type=str, default=None, help='Name of the image on which to run the model', metavar='img')
     group.add_argument('--rng', '--range', type=int, nargs=2, default=None, help='Range of image numbers on which to run the model. \
          For example, 1 100 runs the model on the image 1 through 100', metavar='rng')
     parser.add_argument('--m', '--multiprocess', nargs='?', const='all', default=1, \
          help='Number of processes on which to run the model. Leave blank to use all cores available.')
     parser.add_argument('--s', '--save_prob_map', action='store_true', \
          help='Save probability map for each saccade. If human_subject is provided, this will always be true.')
     parser.add_argument('--h', '--human_subject', type=int, default=None, help='Human subject on which the model will follow its scanpaths, saving the probability map for each saccade.\
          Useful for computing different metrics. See "Kümmerer, M. & Bethge, M. (2021), State-of-the-Art in Human Scanpath Prediction" for more information')
     parser.add_argument('--md','--mode', type=str, default='default', choices=['default', 'experiments'], 
                         help='Mode in which to run the model. "default" is for using a specific model, "experiments" mode is used for running on several configuration same model.')
     args = parser.parse_args()

     if (isinstance(args.m, str) and args.m != 'all') and int(args.m) < 1:
          print('Invalid value for --multiprocess argument')
          sys.exit(-1)

     setup_and_run(args.dataset, args.cfg, args.img, args.rng, args.h, args.m, args.s, args.md)