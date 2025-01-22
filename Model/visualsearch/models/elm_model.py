import numpy as np
from ..utils import utils



class ElmModel:
    def __init__(self, grid_size, visibility_map, norm_cdf_tolerance, number_of_processes, save_probability_maps,plot_heatmap):
        self.grid_size              = grid_size
        self.visibility_map         = visibility_map
        self.save_probability_maps = save_probability_maps
        self.plot_heatmap = plot_heatmap

    def next_fixation(self, posterior, image_name, fixation_number, output_path,heatmap_directory,current_fixation):
        
        posterior_repeated = np.tile(posterior[:, :, np.newaxis, np.newaxis], (1, 1, self.grid_size[0], self.grid_size[1]))
        # Compute the expected information gain map
        expected_ig_map = 1/2 * np.sum(posterior_repeated*self.visibility_map.fovea_map,axis=(0,1))
        # For borders we can complete the parts of the image that are not visible with mirrored values of the image so that the sum takes into account more values in the borders.
        

        # Get the fixation which minimizes the expected entropy
        coordinates = np.where(expected_ig_map == np.amax(expected_ig_map))

        # Save the entropy map reduction
        if self.save_probability_maps:   
            if self.plot_heatmap:
                utils.save_csv_heatmap(heatmap_directory,f'{fixation_number}_{current_fixation[0]}_{current_fixation[1]}.csv',expected_ig_map)
            else:
                utils.save_probability_map(output_path, image_name, expected_ig_map, fixation_number)

        return (coordinates[0][0], coordinates[1][0],np.amax(expected_ig_map))