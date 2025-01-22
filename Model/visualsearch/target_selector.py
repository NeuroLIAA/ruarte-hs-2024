from abc import ABC, abstractmethod
import numpy as np
import EntropyHub as EH

random_generator = np.random.RandomState(42)

class TargetSelector(ABC):
    def __init__(self, memset_weights,memory_set_names,target_name):
        self.memset_weights = memset_weights
        self.memory_set_names = memory_set_names
        self.target_name = target_name

    @abstractmethod
    def select(self,posteriors_unnormalized):
        pass
    
    def get_posteriors_unnormalized(self,visual_evidences,image_prior,current_fixation,fixation_number,visual_evidence_history_factory):
        posteriors_unnormalized = np.array(list(map(lambda x: visual_evidence_history_factory.compute_posterior_unnormalized(image_prior,x.at_fixation(current_fixation,fixation_number),current_fixation,fixation_number),visual_evidences)))
        return posteriors_unnormalized

class MinEntropy2D(TargetSelector):
    def __init__ (self,memset_weights,memory_set_names,target_name):
        super().__init__(memset_weights,memory_set_names,target_name)
    
    def select(self,posteriors_unnormalized):
        posteriors = posteriors_unnormalized / np.sum(posteriors_unnormalized, axis=(1, 2), keepdims=True)
        entropies_2d = np.array(list(map(lambda x: EH.DistEn2D(x),posteriors)))
        standardized = (entropies_2d - np.mean(entropies_2d))/np.std(entropies_2d) if len(entropies_2d) > 1 else entropies_2d
        return np.argmin(standardized*self.memset_weights)

class MinEntropy(TargetSelector):
    def __init__(self,memset_weights,memory_set_names,target_name):
        super().__init__(memset_weights,memory_set_names,target_name)

    def select(self,posteriors_unnormalized):
        posteriors = posteriors_unnormalized / np.sum(posteriors_unnormalized, axis=(1, 2), keepdims=True)
        entropies = -np.sum(posteriors * np.log(posteriors), axis=(1, 2))
        standardized = (entropies - np.mean(entropies))/np.std(entropies) if len(entropies) > 1 else entropies
        return np.argmin(standardized*self.memset_weights)

class Random(TargetSelector):
    def __init__(self,memset_weights,memory_set_names,target_name):
        super().__init__(memset_weights,memory_set_names,target_name)

    def select(self,posteriors_unnormalized):
        return random_generator.choice(len(self.memory_set_names),p=self.memset_weights)

class CorrectTarget(TargetSelector):
    def __init__(self,memset_weights,memory_set_names,target_name):
        super().__init__(memset_weights,memory_set_names,target_name)

    def select(self,posteriors_unnormalized):
        return np.argmax(list(map(lambda x : x==self.target_name,self.memory_set_names)))
    
class LikelihoodMean(TargetSelector):
    def __init__(self,memset_weights,memory_set_names,target_name):
        super().__init__(memset_weights,memory_set_names,target_name)

    def select(self,posteriors_unnormalized):
        return 0
    
    def get_posteriors_unnormalized(self,visual_evidences,image_prior,current_fixation,fixation_number,visual_evidence_history_factory):
        visual_evidences_at_fixation = np.array(list(map(lambda x: x.at_fixation(current_fixation,fixation_number),visual_evidences)))
        mean_visual_evidences = np.mean(visual_evidences_at_fixation,axis=0)
        posteriors_unnormalized = visual_evidence_history_factory.compute_posterior_unnormalized(image_prior,mean_visual_evidences,current_fixation,fixation_number)
        # Add new dimension to posteriors_unnormalized to make it compatible with the rest of the code
        posteriors_unnormalized = np.expand_dims(posteriors_unnormalized, axis=0)
        return posteriors_unnormalized