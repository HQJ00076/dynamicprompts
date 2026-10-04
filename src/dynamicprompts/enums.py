from enum import Enum


class SamplingMethod(Enum):
    RANDOM = "random"
    COMBINATORIAL = "combinatorial"
    CYCLICAL = "cycle"
    DECAY_RANDOM = "decay_random"

    def is_nonfinite(self):
        return self in NON_FINITE_SAMPLING_METHODS


NON_FINITE_SAMPLING_METHODS = {SamplingMethod.RANDOM, SamplingMethod.CYCLICAL, SamplingMethod.DECAY_RANDOM}
