#   Copyright 2021 Phasecraft Ltd.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import collections
import json
import logging
import os
from itertools import chain

import cirq
import numpy as np
from cirq.google import optimized_for_sycamore

VERSION = "0.2.0"

module_logger = logging.getLogger("fhvqe.settings")
Files = collections.namedtuple("Files", ["experiment_folder",
                                         "results_folder",
                                         "processed_folder",
                                         "metadata_filename",
                                         "jobs_filename",
                                         "processed_filename",
                                         "settings_filename",
                                         "samples_filename",
                                         "tflo_samples_filename",
                                         "benchmark_filename",
                                         "calibration_filename",
                                         "ec_filename",
                                         "tflo_filename",
                                         "logger",
                                         "stdout_logger"])
DEFAULT = 1
if DEFAULT:
    GATESET = "default"

PROJECT_ID = "fermi-hubbard-vqe"

# top folder
EXPERIMENTS_FOLDER = "experiments"
# folder with results as downloaded from the engine
RESULTS_FOLDER = "results"
# folder with processed results
PROCESSED_FOLDER = "processed"
# anything recorded during experiment
LIVE_FOLDER = "live"
# project folders (can add extra_desc) for the typical experiments
VQE_FOLDER = "vqe"
HEATMAPS_FOLDER = "heatmaps"
# results file containing raw data for the given job + prog
RESULTS_JSON = "results.json"
# any metadata saved along with the raw results and at runtime
METADATA_JSON = "metadata.json"
# processed file containing processed data from experiment
PROCESSED_JSON = "processed.json"
# the initial settings for this experiments
SETTINGS_JSON = "settings.json"
# benchmark for this experiment (if run)
BENCHMARK_JSON = "benchmark.json"
# calibration for this experiment
CALIBRATION_JSON = "calibration.json"
# samples from the last iteration measured in the computational basis
SAMPLES_FILE = "samples.txt"
# samples from TFLO measured in the computational basis
TFLO_SAMPLES_FILE = "tflo_samples.txt"
# logger data
LOGGER = "fhvqe.log"
# stdout log copy
STDOUT_LOGGER = "stdout.log"
# TFLO file
TFLO_JSON = "tflo.json"
# error correction file
EC_JSON = "error_correction.json"

def generate_experiment_data_structure(nh, nv, project_folder,
                                       experiments_folder = EXPERIMENTS_FOLDER,
                                       results_folder = RESULTS_FOLDER,
                                       processed_folder = PROCESSED_FOLDER,
                                       live_folder = LIVE_FOLDER,
                                       general_metadata_file = METADATA_JSON,
                                       results_file = RESULTS_JSON,
                                       metadata_file = METADATA_JSON,
                                       processed_file = PROCESSED_JSON,
                                       settings_file = SETTINGS_JSON,
                                       samples_file = SAMPLES_FILE,
                                       tflo_samples_file = TFLO_SAMPLES_FILE,
                                       calibration_file = CALIBRATION_JSON,
                                       benchmark_file = BENCHMARK_JSON,
                                       ec_file = EC_JSON,
                                       tflo_file = TFLO_JSON,
                                       logger_name = LOGGER,
                                       stdout_logger_name = STDOUT_LOGGER,
                                       extra_desc = None):
    """Creates necessary folders and paths for files saving the data.

    Args:
        nh -- number of horizontal sites
        nv -- number of vertical sites
        project_folder -- name of project (vqe/heatmaps/single)
        experiments_folder -- main experiment folder (default EXPERIMENTS_FOLDER)
        results_folder -- folder containing experiment results and
                          circuit information (default RESULTS_FOLDER)
        processed_folder -- processed data subfolder (default PROCESSED_FOLDER)
        live_folder -- live data subfolder (default LIVE_FOLDER)
        general_metadata_file -- file containing general metadata about the
                        experiment, chip descriptions, calibrations etc
                        (default METADATA_JSON)
        results_file -- filename to be used for results files (RESULTS_JSON)
        metadata_file -- filename to be used for metadata accompanying
                         results files (METADATA_JSON)
        processed_file -- filename to be used for processed data (PROCESSED_JSON)
        settings_file -- experiment configuration file (default SETTINGS_JSON)
        samples_file -- samples measured in the computational basis (default SAMPLES_FILE)

        extra_desc -- extra description to name of project (default None)
    """
    if extra_desc:
        project_folder = extra_desc
    new_name = project_folder
    keep_asking = True
    while keep_asking:
        project_folder = os.path.join(experiments_folder, new_name)
        if not os.path.exists(project_folder):
            os.makedirs(project_folder)
        experiment_folder = os.path.join(project_folder, f"{nh}x{nv}")
        keep_asking = os.path.exists(experiment_folder)
        break
        if keep_asking:
            new_name = input("Folder already exists, do you want to give a different name: N/[name]\n")
        print(new_name)
        if new_name in ["n", "N"]:
            break
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    jobs_filename = os.path.join(experiment_folder, f"jobs_{nh}x{nv}.json")
    metadata_filename = os.path.join(experiment_folder, general_metadata_file)
    settings_filename = os.path.join(experiment_folder, settings_file)
    calibration_filename = os.path.join(experiment_folder, calibration_file)
    benchmark_filename = os.path.join(experiment_folder, benchmark_file)
    ec_filename = os.path.join(experiment_folder, ec_file)
    tflo_filename = os.path.join(experiment_folder, tflo_file)
    results_folder = os.path.join(experiment_folder, results_folder)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    samples_filename = os.path.join(experiment_folder, samples_file)
    tflo_samples_filename = os.path.join(experiment_folder, tflo_samples_file)
    processed_folder = os.path.join(experiment_folder, processed_folder)
    logger_filename = os.path.join(experiment_folder, logger_name)
    stdout_logger_filename = os.path.join(experiment_folder, stdout_logger_name)
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    processed_filename = os.path.join(processed_folder, processed_file)
    return Files(experiment_folder, results_folder, processed_folder,
                 metadata_filename, jobs_filename, processed_filename,
                 settings_filename, samples_filename, tflo_samples_filename, benchmark_filename,
                 calibration_filename, ec_filename, tflo_filename,
                 logger_filename, stdout_logger_filename)


# standard single layer settings...
standard_settings = { (2, 1): (2, 1, 2, 10000, [[0.464, 0.785]], -1.237),
                      (3, 1): (2, 1, 2, 10000, [[0.362, 0.598, 0.622]], -2.25),
                      (4, 1): (3, 1, 2, 10000, [[0.361, 0.593, 0.516]], -3.01),
                      (5, 1): (4, 1, 2, 10000, [[0.356, 0.566, 0.563]], -4.06),
                      (6, 1): (4, 1, 2, 10000, [[0.332, 0.540, 0.542]], -4.91),
                      (7, 1): (3, 1, 2, 10000, [[0.351, 0.5, 0.5]], 0.0),
                      (8, 1): (3, 1, 2, 10000, [[0.351, 0.5, 0.5]], 0.0),
                      (12, 1): (3, 1, 2, 10000, [[0.351, 0.5, 0.5]], 0.0),
                      (2, 2): (2, 1, 2, 10000, [[0.289, 0.511, 0.511]], -3.60),
                      (2, 3): (4, 1, 2, 10000, [[0.327, 0.709, 0.238, 0.238]], -5.685),
                      (3, 2): (4, 1, 2, 10000, [[0.327, 0.238, 0.238, 0.709]], -5.685),
                      (2, 4): (6, 1, 2, 10000, [[0.317, 0.243, 0.567, 0.500]], -7.72),
                      (4, 2): (6, 1, 2, 10000, [[0.317, 0.567, 0.500, 0.243]], -7.72),
                      (3, 3): (6, 1, 2, 10000, [[0.277, 0.471, 0.409, 0.441, 0.451]], -9.53)
                    }


spsa_settings = {
                  (2,1): { "trials": [100, 1000, 10000],
                           "grad": [1, 1, 2],
                           "max": [200, 60, 60]
                         },
                  (3,1): { "trials": [100, 1000, 10000],
                           "grad": [1, 1, 2],
                           "max": [200, 200, 200]
                         },
                  (4,1): { "trials": [100, 1000, 10000],
                           "grad": [1, 1, 2],
                           "max": [200, 500, 200]
                         },
                  (5,1): { "trials": [100, 1000, 10000],
                           "grad": [1, 1, 2],
                           "max": [500, 100, 100]
                         },
                  (6,1): { "trials": [100, 1000, 10000],
                           "grad": [1, 1, 2],
                           "max": [500, 100, 100]
                         },
                  (7,1): { "trials": [100, 1000, 10000],
                           "grad": [1, 1, 2],
                           "max": [1000, 200, 100]
                         },
                  (2,2): { "trials": [100, 1000, 10000],
                           "grad": [1, 1, 2],
                           "max": [700, 60, 60]
                         }
                }
#TODO load settings from a csv file


def get_qubits_sim(nh, nv):
    return (cirq.GridQubit.rect(nh, 2*nv), "JW")


def get_qubits_default(nh, nv, chip="rainbow"):
    qubits_dict_rainbow = {
                             (2,1): [cirq.GridQubit(4,1), cirq.GridQubit(4,2), cirq.GridQubit(5,1), cirq.GridQubit(5,2)],
                             (3,1): [cirq.GridQubit(4,1), cirq.GridQubit(4,2), cirq.GridQubit(4,3), cirq.GridQubit(5,1), cirq.GridQubit(5,2), cirq.GridQubit(5,3)],
                             (4,1): [cirq.GridQubit(4,1), cirq.GridQubit(4,2), cirq.GridQubit(4,3),cirq.GridQubit(4,4), cirq.GridQubit(5,1), cirq.GridQubit(5,2), cirq.GridQubit(5,3), cirq.GridQubit(5,4)],
                             (5,1): [cirq.GridQubit(4,1), cirq.GridQubit(4,2), cirq.GridQubit(4,3), cirq.GridQubit(4,4), cirq.GridQubit(4,5), cirq.GridQubit(5,1), cirq.GridQubit(5,2), cirq.GridQubit(5,3), cirq.GridQubit(5,4), cirq.GridQubit(5,5)],
                             (6,1): [cirq.GridQubit(4,1), cirq.GridQubit(4,2), cirq.GridQubit(4,3), cirq.GridQubit(4,4), cirq.GridQubit(4,5), cirq.GridQubit(4,6), cirq.GridQubit(5,1), cirq.GridQubit(5,2), cirq.GridQubit(5,3), cirq.GridQubit(5,4), cirq.GridQubit(5,5), cirq.GridQubit(5,6)],
                             (7,1): [cirq.GridQubit(4,1), cirq.GridQubit(4,2), cirq.GridQubit(4,3), cirq.GridQubit(4,4), cirq.GridQubit(4,5), cirq.GridQubit(4,6), cirq.GridQubit(4,7), cirq.GridQubit(5,1), cirq.GridQubit(5,2), cirq.GridQubit(5,3), cirq.GridQubit(5,4), cirq.GridQubit(5,5), cirq.GridQubit(5,6), cirq.GridQubit(5,7)],
                             (8,1): [cirq.GridQubit(4,1), cirq.GridQubit(4,2), cirq.GridQubit(4,3), cirq.GridQubit(4,4), cirq.GridQubit(4,5), cirq.GridQubit(4,6), cirq.GridQubit(4,7), cirq.GridQubit(4,8), cirq.GridQubit(5,1), cirq.GridQubit(5,2), cirq.GridQubit(5,3), cirq.GridQubit(5,4), cirq.GridQubit(5,5), cirq.GridQubit(5,6), cirq.GridQubit(5,7), cirq.GridQubit(5,8)],
                             (12,1): [cirq.GridQubit(4,1), cirq.GridQubit(4,2), cirq.GridQubit(4,3), cirq.GridQubit(4,4), cirq.GridQubit(4,5), cirq.GridQubit(4,6), cirq.GridQubit(4,7), cirq.GridQubit(4,8), cirq.GridQubit(4,9), cirq.GridQubit(4,10), cirq.GridQubit(4,11), cirq.GridQubit(4,12), cirq.GridQubit(5,1), cirq.GridQubit(5,2), cirq.GridQubit(5,3), cirq.GridQubit(5,4), cirq.GridQubit(5,5), cirq.GridQubit(5,6), cirq.GridQubit(5,7), cirq.GridQubit(5,8), cirq.GridQubit(5,9), cirq.GridQubit(5,10), cirq.GridQubit(5,11), cirq.GridQubit(5,12)],
                             (2,2): [cirq.GridQubit(4,1), cirq.GridQubit(4,2), cirq.GridQubit(4,3),cirq.GridQubit(4,4), cirq.GridQubit(5,1), cirq.GridQubit(5,2), cirq.GridQubit(5,3), cirq.GridQubit(5,4)],
                             (2,3): [cirq.GridQubit(4,1), cirq.GridQubit(4,2), cirq.GridQubit(4,3), cirq.GridQubit(4,4), cirq.GridQubit(4,5), cirq.GridQubit(4,6), cirq.GridQubit(5,1), cirq.GridQubit(5,2), cirq.GridQubit(5,3), cirq.GridQubit(5,4), cirq.GridQubit(5,5), cirq.GridQubit(5,6)],
                             (2,4): [cirq.GridQubit(4,1), cirq.GridQubit(4,2), cirq.GridQubit(4,3), cirq.GridQubit(4,4), cirq.GridQubit(4,5), cirq.GridQubit(4,6), cirq.GridQubit(4,7), cirq.GridQubit(4,8), cirq.GridQubit(5,1), cirq.GridQubit(5,2), cirq.GridQubit(5,3), cirq.GridQubit(5,4), cirq.GridQubit(5,5), cirq.GridQubit(5,6), cirq.GridQubit(5,7), cirq.GridQubit(5,8)],
                          }
    return (qubits_dict_rainbow[(nh, nv)], "JW")


# Helper function to determine the valid positions for a (width * height) rectangle in a set of qubits
def valid_rect_positions(qubits, rect_width, rect_height):
    valid_rect_positions = set(qubits)
    for start_qubit in list(qubits):
        rect = cirq.GridQubit.rect(rect_height, rect_width, start_qubit.row, start_qubit.col)
        for rect_qubit in rect:
            if rect_qubit not in qubits:
                valid_rect_positions.remove(start_qubit)
                break
    return valid_rect_positions


# Determine the quality of a rectangle at a given position, based on calibration data
def rect_quality(calibration, position, width, height, metric_name, benchmark):
    def _update_quality(quality,q1,q2):
        # Check for 2-qubit metric first; if it isn't a 2-qubit metric, try two 1-qubit metrics.
        # Assumes the 1-qubit metric is an error that can be multiplied together.
        metric_value = metric.get((q1,q2),[1])[0]
        if metric_value == 1:
            metric_value = 1 - (1-metric.get((q1,),[1])[0]) * (1-metric.get((q2,),[1])[0])
#        if metric_name == 'single_qubit_p00_error' or metric_name == 'single_qubit_p11_error' or metric_name == 'parallel_p00_error' or metric_name == 'parallel_p11_error':
#            metric_value = metric.get((q1,q2),[1])[0]
        return quality * (1-metric_value)

    #metric_name = 'two_qubit_sqrt_iswap_gate_xeb_average_error_per_cycle'
    #metric_name = 'two_qubit_parallel_sqrt_iswap_gate_xeb_average_error_per_cycle'
    if metric_name in calibration:
        metric = calibration[metric_name]

        quality = 1
        for x in range(position.col, position.col+width-1):
            for y in range(position.row, position.row+height-1):
                quality = _update_quality(quality, cirq.GridQubit(y,x),cirq.GridQubit(y,x+1))
                quality = _update_quality(quality, cirq.GridQubit(y,x),cirq.GridQubit(y+1,x))

        y = position.row+height-1
        for x in range(position.col, position.col+width-1):
            quality = _update_quality(quality,cirq.GridQubit(y,x),cirq.GridQubit(y,x+1))

        x = position.col+width-1
        for y in range(position.row, position.row+height-1):
            quality = _update_quality(quality,cirq.GridQubit(y,x),cirq.GridQubit(y+1,x))
        return quality

    if benchmark is not None:
        module_logger.debug(f"Testing position ({position.col}, {position.row}), {width}, {height}")
        error = 0
        with open(benchmark, "r") as read_file:
            energies = json.load(read_file)
        for x in range(position.col, position.col+width-1):
            for y in range(position.row, position.row+height-1):
                error += energies[str((y, x))]["error"]
        module_logger.debug(f"Total error: {error}")
        return -error
    return -1



def valid_zigzag_positions(qubits, n, generate_zigzag, second_position):
    valid_zigzag_pos = set(qubits)
    for start_qubit in list(qubits):
        start_qubit2 = second_position(start_qubit)
        qubits1 = generate_zigzag(start_qubit, n)
        qubits2 = generate_zigzag(start_qubit2, n)
        for (qubit1, qubit2) in zip(qubits1, qubits2):
            if qubit1 not in qubits:
                valid_zigzag_pos.remove(start_qubit)
                break
            if qubit2 not in qubits:
                valid_zigzag_pos.remove(start_qubit)
                break
    return valid_zigzag_pos


# Determine the quality of a rectangle at a given position, based on calibration data
def zigzag_quality(calibration, position1, position2, width, height,
                   metric_name, benchmark, generate_zigzag):
    def _update_quality(quality,q1,q2):
        # Check for 2-qubit metric first; if it isn't a 2-qubit metric, try two 1-qubit metrics.
        # Assumes the 1-qubit metric is an error that can be multiplied together.
        metric_value = metric.get((q1,q2),[1])[0]
        if metric_value == 1:
            metric_value = metric.get((q2,q1),[1])[0]
        if metric_value == 1:
            metric_value = 1 - (1-metric.get((q1,),[1])[0]) * (1-metric.get((q2,),[1])[0])
        return quality * (1-metric_value)

    #metric_name = 'two_qubit_sqrt_iswap_gate_xeb_average_error_per_cycle'
    #metric_name = 'two_qubit_parallel_sqrt_iswap_gate_xeb_average_error_per_cycle'
    n = height
    if metric_name in calibration:
        metric = calibration[metric_name]

        quality = 1
        qubits1_list = generate_zigzag(position1, n)
        qubits2_list = generate_zigzag(position2, n)
        qubit1 = next(qubits1_list)
        qubit2 = next(qubits2_list)
        for (next_qubit1, next_qubit2) in zip(qubits1_list, qubits2_list):
            quality = _update_quality(quality, qubit1, next_qubit1)
            quality = _update_quality(quality, qubit2, next_qubit2)
            qubit1 = next_qubit1
            qubit2 = next_qubit2
        return quality

    return -1


# Finds the best rectangle of qubits, based on calibration data.
def find_best_rect(calibration, qubits, width, height, metric_name, benchmark):
    best_quality = -10000
    best_position = (0,0)
    flip_orientation = False

    # Check the rectangle in "normal" orientation
    positions = valid_rect_positions(qubits, width, height)
    for position in positions:
        quality = rect_quality(calibration, position, width, height, metric_name, benchmark)
        if quality > best_quality:
            best_quality = quality
            best_position = position

    # Check the rectangle in "flipped" orientation
    positions = valid_rect_positions(qubits, height, width)
    for position in positions:
        quality = rect_quality(calibration, position, height, width, metric_name, benchmark)
        if quality > best_quality:
            best_quality = quality
            best_position = position
            flip_orientation = True

    module_logger.info(f'Best pos: {best_position}, flip: {flip_orientation}')
    n = height
    if flip_orientation:
        qubit_list = [cirq.GridQubit(best_position.row,best_position.col+i) for i in range(n)] + [cirq.GridQubit(best_position.row+1,best_position.col+i) for i in range(n)]
    else:
        qubit_list = [cirq.GridQubit(best_position.row+i,best_position.col) for i in range(n)] + [cirq.GridQubit(best_position.row+i,best_position.col+1) for i in range(n)]
    return best_position, best_quality, qubit_list


def generate_grid_qubits(position, orientation):
    # Use a grid of qubits starting at a particular position
    rows, cols, top, left = [int(x) for x in position.split()]
    qubits = []
    if cols > rows:
        for r in range(rows):
            for c in range(cols):
                qubits.append(cirq.GridQubit(r+top, c+left))
    else:
        for c in range(cols):
            for r in range(rows):
                qubits.append(cirq.GridQubit(r+top, c+left))
    qubit_map = "JW"
    return qubits, qubit_map


def zigzag_functions(direction):
    def _right_down(position, n):
        for _ in range(n//2):
            yield position
            position = cirq.GridQubit(position.row, position.col+1)
            yield position
            position = cirq.GridQubit(position.row+1, position.col)
        if n % 2 == 1:
            yield position
            position = cirq.GridQubit(position.row, position.col+1)
    def _right_up(position, n):
        for _ in range(n//2):
            yield position
            position = cirq.GridQubit(position.row, position.col+1)
            yield position
            position = cirq.GridQubit(position.row-1, position.col)
        if n % 2 == 1:
            yield position
            position = cirq.GridQubit(position.row, position.col+1)
    def _down_right(position, n):
        for _ in range(n//2):
            yield position
            position =  cirq.GridQubit(position.row+1, position.col)
            yield position
            position = cirq.GridQubit(position.row, position.col+1)
        if n % 2 == 1:
            yield position
            position =  cirq.GridQubit(position.row+1, position.col)
    def _up_right(position, n):
        for  _ in range(n//2):
            yield position
            position = cirq.GridQubit(position.row-1, position.col)
            yield position
            position = cirq.GridQubit(position.row, position.col+1)
        if n % 2 == 1:
            yield position
            position = cirq.GridQubit(position.row-1, position.col)
    def _second_position_right_down(position):
        return cirq.GridQubit(position.row-1, position.col+1)
    def _second_position_down_right(position):
        return cirq.GridQubit(position.row+1, position.col-1)
    def _second_position_right_up(position):
        return cirq.GridQubit(position.row+1, position.col+1)
    def _second_position_up_right(position):
        return cirq.GridQubit(position.row-1, position.col-1)
    direction_functions = {
                           "rd": (_right_down, _second_position_right_down),
                           "dr": (_down_right, _second_position_down_right),
                           "ru": (_right_up, _second_position_right_up),
                           "ur": (_up_right, _second_position_up_right)
                         }
    return direction_functions[direction]

def generate_zigzag_qubits(position, direction, n):
    generate_zigzag, second_position = zigzag_functions(direction)
    qubit_list = list(generate_zigzag(position, n)) +\
                 list(generate_zigzag(second_position(position), n))
    return qubit_list, "JW"

def find_best_zigzag(calibration, qubits, width, height, metric_name, benchmark):
    best_quality = -10000
    best_position = (0,0)
    flip_orientation = False

    n = height

    # Check the rectangle in "normal" orientation
    pos_str = ''
    for direction in ["rd", "ru", "ur", "dr"]:
        generate_zigzag, second_position = zigzag_functions(direction)
        positions = valid_zigzag_positions(qubits, n, generate_zigzag,
                                           second_position)
#        for pos in positions:
#            pos_str += f'"zigzag {pos.row} {pos.col} {direction}", '
#        continue
        for position in positions:
            quality = zigzag_quality(calibration, position,
                                     second_position(position), width,
                                     height, metric_name, benchmark,
                                     generate_zigzag)
            if quality > best_quality:
                best_quality = quality
                best_position = position
                flip_orientation = (direction, second_position(position),
                                    generate_zigzag)
 
# Uncomment (and code above) to print out allowed zigzag positions
#    print(pos_str)
#    quit()

    module_logger.info(f'Best pos: {best_position}, direction: {flip_orientation[0]}')
    qubit_list = list(flip_orientation[2](best_position, n)) +\
                 list(flip_orientation[2](flip_orientation[1], n))
    module_logger.info(f'Best pos: {qubit_list}')
    return best_position, best_quality, qubit_list

# Returns a list of the best qubits to use, in a format suitable for use in other parts of the code
# Only works for nx1, 1xn and 2x2.
def find_best_qubit_list(calibration, qubits, width_sites, height_sites,
                         metric_name = 'two_qubit_parallel_sqrt_iswap_gate_xeb_average_error_per_cycle',
                         quality_function=find_best_rect, benchmark=None,
                         **kwargs):
    # Reorganise 2x2 to fit our optimised implementation
    if width_sites == 2 and height_sites >= 2:
        width_sites = height_sites * 2
        height_sites = 1
    if width_sites == 1 or height_sites == 1:
        if width_sites == 1:
            n = height_sites
        else:
            n = width_sites
        best_position, best_quality, qubit_list = quality_function(calibration,
                                qubits, 2, n, metric_name, benchmark, **kwargs)
    else:
        module_logger.error(f"ERROR: can't find best qubits for a {width_sites} x {height_sites} lattice")
        return None

    module_logger.info(f'Best qubits by {metric_name}: {qubit_list}. Quality: {best_quality}')
    return (qubit_list, "JW")


def remap_qubits(qubits, mapping, mapping_args):
    return [qubits[mapping(*mapping_args, i)] for i in range(len(qubits))]


QUBITS_ASSIGNMENT = { "default": get_qubits_default,
                      "sim": get_qubits_sim
                    }
