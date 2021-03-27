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
from pathlib import Path

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

VISUAL_DEGREES = 8
NUMBER_OF_TRIALS = 50

storage_loc = Path(
    "D:/MIT/EcogData/brainscore_img_elec_time_70hz150_mni_ba_sid_subtrbaseline/brainscore_img_elec_time_70hz150_mni_ba_sid_subtrbaseline")
#neural_responses = np.load(storage_loc / "neural_responses.npy")
#categories = np.load(storage_loc / 'stimgroups.npy')
#brodmann_areas = np.load(storage_loc / "brodmann_areas.npy")
#mni = np.load(storage_loc / "mni_coordinates.npy")


it_idx = [2560, 2561, 2562, 3588, 3589, 10757, 6663, 6664, 6665, 10758, 2573, 2574, 8206, 8207, 6674, 7195, 10818, 5699, 5700, 10819, 1105, 1106, 3700, 3701, 4224, 4225, 8848, 8849, 9362, 6810, 154, 4765, 4766, 9920, 6354, 5335, 5336, 6365, 6366, 6367, 5346, 5347, 8940, 9967, 9968, 7925, 7926, 4349, 4350, 7431, 7432, 5899, 5900, 4887, 4888, 7453, 7454, 1354, 1355, 332, 333, 6988, 6989, 9062, 9063, 10599, 10600, 10093, 3950, 3951, 5488, 5489, 10094, 2423, 2424, 2434, 2435, 8579, 8580, 6542, 6543, 10174, 10175, 10176, 3017, 3018, 5585, 5586, 2020, 2021, 2022, 3575, 3576, 6141, 6142, 2559]

# bounding box IT MNI coordinates:
it_left_a_x = [-29, -20]
it_left_a_y = [-28, -7]
it_left_a_z = [-16, -2]

it_left_b_x = [-19, -13]
it_left_b_y = [-8, 3]
it_left_b_z = [-19, -15]

it_right_a_x = [13, 19]
it_right_a_y = [-30, -8]
it_right_a_z = [-16, -2]

it_right_b_x = [20, 29]
it_right_b_y = [-9, 2]
it_right_b_z = [-20, -16]


# def get_IT_MNI():
#     it_left_a_idx = list(np.where((mni[:, 0] >= it_left_a_x[0]) & (mni[:, 0] <= it_left_a_x[1]) &
#                                   (mni[:, 1] >= it_left_a_y[0]) & (mni[:, 1] <= it_left_a_y[1]) & (mni[:, 2] >= it_left_a_z[0]) & (mni[:, 2] <= it_left_a_z[1]))[0])
# 
#
#     it_left_b_idx = list(np.where((mni[:, 0] >= -19) & (mni[:, 0] <= -13) & (mni[:, 1] >=  -8) & (mni[:, 1] <=  3) & (mni[:, 2] >= -19) & (mni[:, 2] <= -15))[0])
#
#     it_right_a_idx = list(np.where((mni[:, 0] >=  13) & (mni[:, 0] <=  19) & (mni[:, 1] >= -30) & (mni[:, 1] <= -8) & (mni[:, 2] >= -16) & (mni[:, 2] <=  -2))[0])
#     it_right_b_idx = list(np.where((mni[:, 0] >=  20) & (mni[:, 0] <=  29) & (mni[:, 1] >=  -9) & (mni[:, 1] <=  2) & (mni[:, 2] >= -20) & (mni[:, 2] <= -16))[0])
#
#     it_idx = list(set(it_left_a_idx + it_left_b_idx + it_right_a_idx + it_right_b_idx))
#     return it_idx


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
    # idx = np.zeros((len(assembly.neuroid_id)), dtype=bool)
    # idx[1094] = True
    # idx[847] = True

    #idx = get_IT_MNI()
    idx = it_idx

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

