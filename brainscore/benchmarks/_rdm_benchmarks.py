import numpy as np

from brainio_base.assemblies import array_is_element, walk_coords
from brainscore.benchmarks import BenchmarkBase, ceil_score
from brainscore.benchmarks.screen import place_on_screen
from brainscore.model_interface import BrainModel


class RDMBenchmark(BenchmarkBase):
    def __init__(self, identifier, assembly, similarity_metric, visual_degrees, number_of_trials, region, time_bins, **kwargs):
        super(RDMBenchmark, self).__init__(identifier=identifier, **kwargs)
        self._assembly = assembly
        self._similarity_metric = similarity_metric
        self.region = region
        self.timebins = time_bins
        self._visual_degrees = visual_degrees
        self._number_of_trials = number_of_trials

    def __call__(self, candidate: BrainModel):
        candidate.start_recording(self.region, time_bins=self.timebins)
        stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                       source_visual_degrees=self._visual_degrees)
        source_assembly = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
        if 'time_bin' in source_assembly.dims:
            source_assembly = source_assembly.squeeze('time_bin')  # static case for these benchmarks
        raw_score = self._similarity_metric(source_assembly, self._assembly)
        return explained_variance(raw_score, self.ceiling)



def explained_variance(score, ceiling):
    ceiled_score = ceil_score(score, ceiling)
    # ro(X, Y)
    # = (r(X, Y) / sqrt(r(X, X) * r(Y, Y)))^2
    # = (r(X, Y) / sqrt(r(Y, Y) * r(Y, Y)))^2  # assuming that r(Y, Y) ~ r(X, X) following Yamins 2014
    # = (r(X, Y) / r(Y, Y))^2
    r_square = np.power(ceiled_score.raw.sel(aggregation='center').values /
                        ceiled_score.ceiling.sel(aggregation='center').values, 2)
    ceiled_score.__setitem__({'aggregation': score['aggregation'] == 'center'}, r_square,
                             _apply_raw=False)
    return ceiled_score