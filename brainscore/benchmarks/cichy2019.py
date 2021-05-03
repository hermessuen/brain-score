import brainscore
from brainscore.benchmarks._neural_common import NeuralBenchmark, average_repetition, apply_keep_attrs
from brainscore.benchmarks._rdm_benchmarks import RDMBenchmark, explained_variance
from brainscore.metrics.ceiling import InternalConsistency, RDMConsistency
from brainscore.metrics.rdm import RDMCrossValidated
from brainscore.metrics.regression import CrossRegressedCorrelation, mask_regression, ScaledCrossRegressedCorrelation, \
    pls_regression, pearsonr_correlation
from brainscore.benchmarks import BenchmarkBase, ceil_score
from brainscore.metrics.rdm import RDMCrossValidated

from brainscore.utils import LazyLoad
import numpy as np
from brainio_base.assemblies import walk_coords
from brainscore.metrics import Score
import pandas as pd

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

VISUAL_DEGREES = 8
NUMBER_OF_TRIALS = 50




def AruCichy2019RDM():
    assembly = LazyLoad(lambda: load_assembly(average_repetitions=False))

    similarity_metric = RDMCrossValidated(crossvalidation_kwargs=dict(stratification_coord=None))
    ceiler = RDMConsistency()

    ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
    return RDMBenchmark(identifier=f'aru.Cichy2019-rdm', version=1, assembly=assembly,
                           similarity_metric=similarity_metric,
                           visual_degrees=VISUAL_DEGREES, ceiling_func=lambda: ceiling,
                           parent='Kamila', number_of_trials=1, region='IT',
                        time_bins=[(62.5, 156.2)])


def load_assembly(average_repetitions):
    assembly = brainscore.get_assembly(name='aru.Cichy2019')

    # fix the off by 1 error with the stimulus set
    # assembly = assembly['image_id'] + 1

    if average_repetitions:
        assembly = average_repetition(assembly)
    return assembly