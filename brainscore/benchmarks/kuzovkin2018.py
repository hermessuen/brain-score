import brainscore
from brainscore.benchmarks._neural_common import NeuralBenchmark, average_repetition, apply_keep_attrs
from brainscore.metrics.ceiling import InternalConsistency, RDMConsistency
from brainscore.metrics.rdm import RDMCrossValidated
from brainscore.metrics.regression import CrossRegressedCorrelation, mask_regression, ScaledCrossRegressedCorrelation, \
    pls_regression, pearsonr_correlation

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



def AruKuzovkin2018PLS():
    assembly_repetition = LazyLoad(lambda: load_assembly(average_repetitions=False))
    assembly = LazyLoad(lambda: load_assembly(average_repetitions=True))

    # similarity_metric = CrossRegressedCorrelation(
    #     regression=pls_regression(),
    #     correlation=pearsonr_correlation(),
    #     crossvalidation_kwargs=dict(stratification_coord=None)
    # )

    similarity_metric = RDMCrossValidated(crossvalidation_kwargs=dict(stratification_coord=None))

    # sub-select the entire IT region
    # x = assembly['region']
    # idx = x.data == 20

    # sub-select only important it electrodes for IT: 1094
    idx = np.zeros((len(assembly.neuroid_id)), dtype=bool)
    idx[1094] = True
    idx[847] = True

    new_assembly = assembly[:, idx, :]

    new_assembly = type(new_assembly)(new_assembly.values, coords={coord: (dims, values if coord != 'region' else ['IT'] * len(new_assembly['region'])) for
                                   coord, dims, values in walk_coords(new_assembly)}, dims=new_assembly.dims)


    new_assembly.attrs = assembly.attrs

    new_assembly = new_assembly.where(new_assembly['time_bin_start'] < 150, drop=True)
    new_assembly = new_assembly.where(new_assembly['time_bin_start'] > 50, drop=True)
    new_assembly = new_assembly.mean('time_bin')
    new_assembly = new_assembly.expand_dims('time_bin')
    new_assembly.coords['time_bin'] = pd.MultiIndex.from_tuples([(62.5, 156.2)], names=['time_bin_start', 'time_bin_end'])
    new_assembly = new_assembly.squeeze('time_bin')

    new_assembly.attrs = assembly.attrs

    ceiling = Score([1, np.nan], coords={'aggregation': ['center', 'error']}, dims=['aggregation'])
    return NeuralBenchmark(identifier=f'aru.Kuzovkin2018-pls', version=1, assembly=new_assembly,
                           similarity_metric=similarity_metric,
                           visual_degrees=VISUAL_DEGREES, ceiling_func=lambda: ceiling,
                           parent='Kamila', number_of_trials=1)

def load_assembly(average_repetitions):
    assembly = brainscore.get_assembly(name='aru.Kuzovkin2018')
    if average_repetitions:
        assembly = average_repetition(assembly)

    # assembly.reset_index('time', drop=True, inplace=True)

    return assembly