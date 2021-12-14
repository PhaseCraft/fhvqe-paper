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

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import re
from fhvqe.settings import rect_quality
import cirq

from fhvqe.tools import round_sig


class NumpyEncoder(json.JSONEncoder):
    """Helper class for encoding numpy into json.

    TODO: expand as needed for other types.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def process_spsa_json(filename, label, option="energy"):
    """Process SPSA json file into pandas dataframe.
    """
    with open(filename) as json_file:
        file_data = json.load(json_file)
    values = []
    if option=="energy":
        if len(file_data) == 1: # SPSA - FIXME: hack to be used only while MGD file format is different
            for result in file_data:
                values += result[3]
        else: # MGD
            values = file_data[2]
        val_pd = pd.DataFrame(values, columns=[label])
    if option=="parameters":
        if len(file_data) == 1: # SPSA - FIXME: hack to be used only while MGD file format is different
            for result in file_data:
                values += result[4]
        else: # MGD
            values = file_data[3]
#           val_pd = pd.DataFrame(values, columns=[label])
        if len(values[0]) != 1: # Fix for format of MGD file
            values = [[x] for x in values]
        return values
    if option=="details":
        for result in file_data:
            for details in result[2]:
                nmeas = 0
                post_nmeas = 0
                energies = {}
                for key, val in details.items():
                    if key != "E":
                        energies[key] = val["Em"]
                        nmeas += val["num_trials"]
                        post_nmeas += val["post-processed"]
                energies["nmeas"] = nmeas
                energies["pnmeas"] = post_nmeas
                values.append(energies)
        val_pd = pd.DataFrame(values)
    if option=="timestamps":
        all_stamps = []
        for result in file_data:
            for details in result[2]:
                all_stamps.append(details["end_timestamp"])
        val_pd = all_stamps
    if option=="parameters":
        given_params = [] # parameters for all iterations
        if len(file_data) == 1: # SPSA
            for mod_spsa_results in file_data: # for set number of measurements
                for details in mod_spsa_results[2]:
                    # details contains info on a specific gradient evaluation
                    # pick any measurement
                    measurement_list = next(iter(details.values()))
                    gradient_params = [[None, None] for _ in range(len(measurement_list)//2)]
                    for measurement in measurement_list:
                        if measurement["sgn"] == 1:
                            element_loc = 0
                        if measurement["sgn"] == -1:
                            element_loc = 1
                        gradient_params[measurement["grad"]][element_loc] = measurement["params"]
                        # parameters for specific gradient
                    given_params.append(gradient_params)
            val_pd = given_params # small cheat
        else:
            val_pd = file_data[3]
            if len(val_pd[0]) != 1: # Fix for format of MGD file
                val_pd = [[x] for x in val_pd]
    return val_pd


def plot_spsa(energy_pds, fig = None, ax = None, rolling=False, subplots=False):
    """Displays SPSA optimization plots.

    Args:
        energy_pds -- pandas dataframe with spsa optimization per column
        fig -- previous pyplot figure to add current data to (default None)
        ax -- previous pyplot axes to add current data to (default None)
        rolling -- displaying rolling average (default False)
        subplots -- display in single versus subplots (default False)

    Returns:
        pyplot figure and axes of the plot
    """
#    fig = plt.figure()
#    fig, ax = plt.subplots()
    if fig is None or ax is None:
        if subplots:
            fig, ax = plt.subplots(energy_pds.shape[1])
        else:
            fig, ax = plt.subplots()
    energy_pds.plot(ax=ax, marker='.', colormap='Accent', subplots=subplots)
    if rolling:
        ax.plot(energy_pds.rolling(5).mean(), color='r')
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.legend(loc="best")
    #plt.show()
    return fig, ax



def show_heatmap(data, smoothed=False, x_min=0.0, x_max=1.0, y_min=0.0,
                 y_max=1.0, custom_x=None, custom_y=None, xlabel="Onsite parameter",
                 ylabel="Hopping parameter", title=None, **kwargs):
    """Displays a heatmap of data.

    Displays a heatmap corresponding to some data, assumed to correspond to
    a square array of energies. If smoothed is True, uses an interpolation method.

    Args:
        data -- numpy array of some data (if flattened list, then converts to square)
        smoothed -- whether interpolation is used (default False)
        x_min -- minimum x value (default 0.0)
        x_max -- maximum x value (default 1.0)
        y_min -- minimum y value (default 0.0)
        y_max -- maximum y value (default 1.0)
        custom_x -- custom x tick labels (default None)
        custom_y -- custom y tick labels (default None)
        xlabel -- x axis name (default "Onsite parameter")
        ylabel -- y axis name (default "Hopping parameter")
        title -- graph title (default None)

    Returns:
        pyplot figure and axes of the plot
    """
    fig, ax = plt.subplots()
    plt.rcParams.update({"font.size": 16})

    if data.ndim == 1: # assumes square, and reshapes
        size = int(math.sqrt(len(data)))
        data = np.reshape(data, (size, size))
    if smoothed:
        kwargs["interpolation"] = "bicubic"
    im = ax.imshow(data, cmap=plt.get_cmap('cividis'), origin='lower', **kwargs)

    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticks(np.arange(data.shape[1]))
    if custom_x is not None: xlabels=custom_x
    else:
        step = (x_max - x_min) / (data.shape[1] - 1)
        x_labels = [round_sig(x_min + k * step) if not k % 2 else "" for k in range(data.shape[1])]
    if custom_y is not None: ylabels=custom_y
    else:
        step = (y_max - y_min) / (data.shape[0] - 1)
        y_labels = [round_sig(y_min + k * step) for k in range(data.shape[0])]
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None: ax.set_title(title)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Energy", rotation=-90, va="bottom")
    fig.tight_layout()
    return fig, ax


def display_benchmark_json(filename, calib_filename, label="benchmarks"):
    """Process SPSA json file into pandas dataframe.
    """
    def _convert(key):
        new_key = []
        for match in re.findall(r"\((.*?), (.*?)\)", key[1:-1]):
            a, b = map(int, match)
            new_key.append(cirq.GridQubit(a,b))
        return tuple(new_key)
    with open(filename) as json_file:
        file_data = json.load(json_file)
    with open(calib_filename) as json_file:
        import ast
        calib_data = ast.literal_eval(json.load(json_file))["_metric_dict"]

    metric_name = 'two_qubit_sqrt_iswap_gate_xeb_average_error_per_cycle'
    calibration = {key: {_convert(key2):val2 for key2, val2 in val.items()}
                            for key, val in calib_data.items()}
    val_pd = pd.DataFrame(file_data)#, columns=[label])
    new_dict = {"0":{}, "1":{}, "2":{}, "3":{}, "4":{}, "calib":{}, "overall":{}}
    for key, val in file_data.items():
        qubits = [int(qubit) for qubit in filter(None, re.split("[(),\s]",key))]
        quality = rect_quality(calibration, cirq.GridQubit(*qubits), 2, 2, metric_name)
        new_dict["calib"][key] = quality
        for key2, val2 in val.items():
            if key2 == "error":
                continue
            new_dict[key2][key] = abs((val2["actual"] - val2["calculated"]))
    val_pd = pd.DataFrame.from_dict(new_dict)#, columns=[label])
    return val_pd
