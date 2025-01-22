import numpy as np
from abc import ABC, abstractmethod

class HistoryVisualEvidenceFactory(ABC):
    def __init__(self,prior_as_fixation, max_saccades, grid_size, image_prior, history_size):
        if prior_as_fixation:
            self.likelihood = np.log(image_prior)
        else:
            self.likelihood = np.zeros(shape=grid_size)
        self.history_size = history_size
        self.prior_as_fixation = prior_as_fixation

    def get_history_visual_evidence(self):
        return self.history_visual_evidence
    
    @abstractmethod
    def compute_likelihood(self, visual_evidence_at_fixation, current_fixation,fixation_number, history_size):
        pass
    
    def compute_posterior_unnormalized(self,image_prior,visual_evidence_at_fixation,current_fixation,fixation_number):
        likelihood = self.compute_likelihood(visual_evidence_at_fixation,current_fixation,fixation_number,self.history_size)

        if self.prior_as_fixation:
            likelihood_times_prior =  np.exp(likelihood)
        else:
            likelihood_times_prior =  np.exp(likelihood + np.log(image_prior))

        
        return likelihood_times_prior
class NoHistoryVisualEvidence(HistoryVisualEvidenceFactory):
    def __init__(self,prior_as_fixation, max_saccades, grid_size, image_prior, history_size):
        super().__init__(prior_as_fixation, max_saccades, grid_size, image_prior,history_size)
        self.history_visual_evidence = None
    
    def compute_likelihood(self, visual_evidence_at_fixation, current_fixation,fixation_number, history_size):
        return self.likelihood + visual_evidence_at_fixation

    def update_values(self,selected_visual_evidence,fixation_number):
        self.likelihood = self.likelihood + selected_visual_evidence 

class SimpleHistoryVisualEvidence(HistoryVisualEvidenceFactory):
    def __init__(self,prior_as_fixation, max_saccades, grid_size, image_prior, history_size):
        super().__init__(prior_as_fixation, max_saccades, grid_size, image_prior,history_size)
        if prior_as_fixation:
            self.history_visual_evidence = np.tile(np.log(image_prior), (history_size, 1, 1))
        else:
            self.history_visual_evidence = np.zeros(shape=(history_size, grid_size[0], grid_size[1]))
 
    def compute_likelihood(self,  visual_evidence_at_fixation, current_fixation,fixation_number, history_size):
        return self.history_visual_evidence[0] + visual_evidence_at_fixation
    
    def update_values(self,selected_visual_evidence,fixation_number):
        grid_size = self.history_visual_evidence.shape[1:]
        self.history_visual_evidence = np.append(self.history_visual_evidence,np.zeros(shape=(1,grid_size[0],grid_size[1])), axis=0)
        self.history_visual_evidence = self.history_visual_evidence + selected_visual_evidence 
        self.history_visual_evidence = self.history_visual_evidence[1:] #I discard the oldest fixation info

class DegradedHistoryVisualEvidence(HistoryVisualEvidenceFactory):
    def __init__(self,prior_as_fixation, max_saccades, grid_size, image_prior, history_size):
        super().__init__(prior_as_fixation, max_saccades, grid_size, image_prior,history_size)
        self.history_visual_evidence = np.zeros(shape=(max_saccades+1,grid_size[0],grid_size[1]))
        if prior_as_fixation:
            self.history_visual_evidence[0] = np.log(image_prior)


    def compute_likelihood(self, visual_evidence_at_fixation, current_fixation,fixation_number,history_size):
        degradation = np.exp(np.arange(-fixation_number-1, 1) / history_size)[:, np.newaxis, np.newaxis] #Degradation using an exponential        
        self.history_visual_evidence[fixation_number+1] = visual_evidence_at_fixation #Index 0 is reserved for the prior
        #Even if history_visual_evidence is changed here and is used outside this function, it is not a problem because the correct value is used after this function returns
        likelihood = np.sum(self.history_visual_evidence[:fixation_number+2]*degradation,axis=0)
        return likelihood

    def update_values(self,selected_visual_evidence,fixation_number):
        self.history_visual_evidence[fixation_number+1] = selected_visual_evidence 
