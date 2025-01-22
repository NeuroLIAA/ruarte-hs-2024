import numpy as np
from ..utils import utils

class GreedyModel:
    def __init__(self, grid_size, visibility_map, norm_cdf_tolerance, number_of_processes, save_probability_maps,plot_heatmap):
        self.save_probability_maps = save_probability_maps
        self.plot_heatmap = plot_heatmap

    def next_fixation(self, posterior, image_name, fixation_number, output_path,heatmap_directory,current_fixation):
        " Given the posterior for each cell in the grid, this function computes the next fixation by searching for the maximum values from it "
        """ Input:
                posterior (2D array of floats) : matrix the size of the grid containing the posterior probability for each cell
            Output:
                next_fix (int, int) : cell chosen to be the next fixation
            
            (The rest of the input arguments are used to save the probability map to a CSV file.)
        """
        coordinates = np.where(posterior == np.amax(posterior))
        next_fix    = (coordinates[0][0], coordinates[1][0])

        if self.save_probability_maps:   
            if self.plot_heatmap:
                utils.save_csv_heatmap(heatmap_directory,f'{fixation_number}_{current_fixation[0]}_{current_fixation[1]}.csv',posterior)
            else:
                utils.save_probability_map(output_path, image_name, posterior, fixation_number)

        return coordinates[0][0], coordinates[1][0], 0