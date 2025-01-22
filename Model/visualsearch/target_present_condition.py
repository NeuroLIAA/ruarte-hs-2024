from abc import ABC, abstractmethod
from .utils.utils import are_within_boundaries
class TargetPresentCondition(ABC):

    @abstractmethod
    def end_trial(self, target_bbox, current_fixation, fixation_number, fixations):
        pass


class ConsecutiveFix(TargetPresentCondition):
    def end_trial(self, target_bbox, current_fixation, fixation_number, fixations):
        next_fix = fixations[fixation_number + 1]
        return (next_fix == current_fixation).all()


class Oracle(TargetPresentCondition):
    def end_trial(self, target_bbox, current_fixation, fixation_number, fixations):
        return not target_bbox is None and are_within_boundaries(
        current_fixation, current_fixation, (target_bbox[0], target_bbox[1]),
        (target_bbox[2] + 1, target_bbox[3] + 1)
    )
