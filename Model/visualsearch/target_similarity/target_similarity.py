import numpy as np
from os import path
from skimage import io
from ..utils import utils
import pandas as pd



class TargetSimilarity():
    def __init__(self, image_name,stim_name,target_name, image, target, target_bbox, visibility_map, scale_factor, additive_shift, grid, seed, number_of_processes, save_similarity_maps, target_similarity_dir,maps_save_directory,fovea_filter):
        # Set the seed for generating random noise
        np.random.seed(seed)

        self.number_of_processes   = number_of_processes
        self.save_similarity_maps  = save_similarity_maps
        self.grid                  = grid
        self.image_name            = image_name
        self.target_similarity_dir = target_similarity_dir
        self.stim_name             = stim_name
        self.target_name           = target_name
        self.maps_save_directory   = maps_save_directory
        self.visibility_map        = visibility_map
        self.fovea_filter          = fovea_filter

        self.create_target_similarity_map(image, target, target_bbox, scale_factor, additive_shift)

    def create_target_similarity_map(self, image, target, target_bbox, scale_factor, additive_shift):
        " Creates the target similarity map for a given image, target and visibility map.  "
        """ Input:
                image  (2D array) : search image
                target (2D array) : target image
                target_bbox (array)  : bounding box (upper left row, upper left column, lower right row, lower right column) of the target in the image
                visibility_map (VisibilityMap) : visibility map which indicates how focus decays over distance from the fovea
                scale_factor   (int) : modulates the inverse of the visibility and prevents the variance from diverging
                additive_shift (int) : modulates the inverse of the visibility and prevents the variance from diverging
            Output:
                sigma, mu (4D arrays) : values of the normal distribution for each possible fixation in the grid. It's based on target similarity and visibility
        """
        grid_size = self.grid.size()

        # Initialize mu, where each cell has a value of 0.5 if the target is present and -0.5 otherwise
        self.mu = np.zeros(shape=(grid_size[0], grid_size[1], grid_size[0], grid_size[1])) - 0.5

        if not (target_bbox is None) and (self.target_name == self.stim_name):
            self.mu[target_bbox[0]: target_bbox[2]+ 1, target_bbox[1]: target_bbox[3]+ 1] = np.zeros(shape=grid_size) + 0.5
        file_path = f'{self.maps_save_directory}/target_mask.png' 
        # Initialize sigma
        self.sigma = np.ones(shape=self.mu.shape)
        # Variance now depends on the visibility
        self.sigma = self.sigma / (self.visibility_map.normalized_at_every_fixation() * scale_factor + additive_shift)
              
        # If precomputed, load target similarity map
        save_path = path.join(self.target_similarity_dir, self.__class__.__name__)
        filename  = self.image_name[:-4] + '_' + self.stim_name[:-4] +'.png'
        file_path = path.join(save_path, filename)
        if path.exists(file_path):
            target_similarity_map = io.imread(file_path)
        else:
            if not utils.is_coloured(image) and utils.is_coloured(target):
                target = utils.to_grayscale(target)
            # Calculate target similarity based on a specific method  
            print('Building target similarity map...')
            target_similarity_map = self.compute_target_similarity(image, target, target_bbox)
            utils.save_similarity_map(save_path, filename, target_similarity_map)
                
        
        # Add target similarity and visibility info to mu
        self.add_info_to_mu(target_similarity_map)

        return

    def compute_target_similarity(self, image, target, target_bbox):
        """ Each subclass calculates the target similarity map with its own method """
        pass

    def add_info_to_mu(self, target_similarity_map):
        """ Once target similarity has been computed, its information is added to mu, alongside the visibility map """
        # Reduce to grid
        target_similarity_map = self.grid.reduce(target_similarity_map, mode='max')

        # Convert values to the interval [-0.5, 0.5] 
        target_similarity_map = target_similarity_map - np.min(target_similarity_map)
        target_similarity_map = target_similarity_map / np.max(target_similarity_map) - 0.5
        # Make it the same shape as mu
        grid_size = self.grid.size()
        target_similarity_map = np.tile(target_similarity_map[:, :, np.newaxis, np.newaxis], (1, 1, grid_size[0], grid_size[1]))

        # Modify mu in order to incorporate target similarity and visibility
        # The real target has a lot of weight within the fovea, but not in the peripheral vision
        # The distractors (target_similarity_map with high values) have a lot of weight in the peripheral vision, but not in the fovea, they are discarded within the fovea
        if self.fovea_filter:
            self.mu = self.mu * (self.visibility_map.normalized_fovea_at_every_fixation() + 0.5) + target_similarity_map * (1 - self.visibility_map.normalized_fovea_at_every_fixation() + 0.5)
        else:
            self.mu = self.mu * (self.visibility_map.normalized_at_every_fixation() + 0.5) + target_similarity_map * (1 - self.visibility_map.normalized_at_every_fixation() + 0.5)
        # Convert values to the interval [-0.5, 0.5]

        self.mu = self.mu / 2

        return
    
    def at_fixation(self, fixation,fixation_number):
        " Given a fixation in the grid, it returns the target similarity map, represented as a 2D array of scalars with added random noise "
        """ Input:
                fixation (int, int) : cell in the grid on which the observer is fixating
            Output:
                target_similarity_map (2D array of floats) : matrix the size of the grid, where each value is a scalar which represents how similar the position is to the target
        """
        grid_size = self.grid.size()
        # For backwards compatibility with MATLAB, it's necessary to transpose the matrix
        random_noise = np.transpose(np.random.standard_normal((grid_size[1], grid_size[0])))
        visual_evidence =  self.mu[:, :, fixation[0], fixation[1]] + self.sigma[:, :, fixation[0], fixation[1]] * random_noise
        fovea = self.visibility_map.at_fixation_fovea(fixation) * visual_evidence

        peripheral_visibility = self.visibility_map.at_fixation(fixation) * visual_evidence
        if self.fovea_filter:
            visual_evidence_foveated = np.maximum(fovea, peripheral_visibility)
        else:
            visual_evidence_foveated = fovea
        if self.save_similarity_maps:
            utils.save_csv_heatmap(self.maps_save_directory + f'/{self.stim_name[:-4]}/visual_evidence_foveated',f'{fixation_number}_{fixation[0]}_{fixation[1]}.csv',visual_evidence_foveated)
        return visual_evidence_foveated
