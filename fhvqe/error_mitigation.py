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

from typing import Callable

import collections
import copy
import datetime
import json
import logging
import time
import os
from random import choices

import cirq
import numpy as np
import scipy
import cirq_google as cg
from cirq_google import optimized_for_sycamore

import fhvqe.experiment
import fhvqe.circuit
from fhvqe.circuit import sqrt_iswap_gate
from fhvqe.experiment import (energy_calculation_wrap,
                              extract_values,
                              extract_values_exact,
                              run_executables, run_executables_exact)
from fhvqe.processing import NumpyEncoder

module_logger = logging.getLogger("fhvqe.error_mitigation")


ERROR_CORRECTION_DICTIONARY = {
                                "occupation_number": False,
                                "spin_type": False,
                              }


def sample_error_mitigation_func(error_dict, nocc_1=0, nocc_2=0, nocc=0):
    """Create appropriate error correction function on sample level.

    Args:
        error_dict -- dictionary of error mitigation strategies to
                      be included in the function
        nocc_1 -- number of fermions of spin up type (default 0)
        nocc_2 -- number of fermions of spin down type (default 0)
        nocc -- number of fermions in the system (default 0)

    Returns:
        Error correction function.
    """
    def _do_nothing(args):
        return args
    error_func = _do_nothing
    if error_dict["occupation_number"]:
        error_func = error_detect_samples_wrap(nocc)
    if error_dict["spin_type"]:
        error_func = error_detect_spins_samples_wrap(nocc_1, nocc_2)
    return error_func
    

# TRAINING WITH FERMIONIC LINEAR OPTICS

def mitigate_by_tflo(qubits, theta, measurement_set, exact_measurement_set,
                     num_layers=0, num_params=0, min_energy=0, n=2, U=2,
                     filename=None, extract_values_func=extract_values,
                     save_tflo_samples=False, tflo_samples_filename="tflo_samples.txt", **kwargs):
    tflo_data = {}
    params_list = []
    module_logger.info(f"About to mitigate, theta={theta}")

    print(f"---Starting TFLO")
    tick = time.perf_counter()
    
    # Generate either a fixed set of points or random ones, if the points aren't provided.
    if kwargs["mitigation_kwargs"]["tflo_points"] is None:
        if num_layers == 1:
            params_list = [[[0, a*np.pi/4, b*np.pi/4]] for a in range(8) for b in range(8)]
        else:
            np.random.set_seed(1)
            params_list = np.random.random_sample((num_tests, num_layers, num_params)) * 2 * np.pi
            for i in range(num_correction_pts):
                for j in range(num_layers):
                    params_list[i][j][0] = 0
            
    else:
        print(f"Using saved points")
        params_list = kwargs["mitigation_kwargs"]["tflo_points"]
        
        params_list.append(copy.deepcopy(theta)) # evaluate at final VQE params but in FLO
        for j in range(num_layers):
            params_list[-1][j][0] = 0
        params_list.append(theta) # evaluate at final VQE params
        params_list_neg = [[[-z for z in y] for y in x] for x in params_list]
        params_list = params_list + params_list_neg
    
    num_tests = len(params_list)

    module_logger.debug(f"About to run TFLO, params={params_list}")

    nmeas = 20000    # use 20000 shots

    energy_calculation_noisy = energy_calculation_wrap(extract_values_func=extract_values_func)
    energy_calculation_exact = energy_calculation_wrap(run_executables_exact,
                                                       extract_values_exact)

    noisy_E = energy_calculation_noisy(params_list,
                                         (num_tests,
                                          [(i, 0) for i in range(num_tests)]),
                                         qubits,
                                         measurement_set,
                                         nmeas,
                                         num_layers=num_layers,
                                         num_params=num_params,
                                         batch_evals=num_tests,
                                         save_samples=save_tflo_samples,
                                         samples_filename=tflo_samples_filename,
                                         **kwargs)
    measured_values = copy.deepcopy(fhvqe.experiment.measured_values)
    tflo_data["noisy_measured_values"] = measured_values

    tock = time.perf_counter()
    print(f"---Time for noisy calculation: {tock-tick}")
    tick = time.perf_counter()

    # output the noisy energies measured
    print(f"***TFLON: ", end='')
    for idx,e in enumerate(noisy_E):
        print(f'{e}', end='')
        if idx < len(noisy_E)-1:
            print(f',', end='')
        else:
            print(f" {time.time()}")

    noisy = {}
    for measurement in measurement_set:
        m_type = measurement.type
        noisy[m_type] = np.array([x["Em"] for x in measured_values[m_type]])

    print(f"Noisy energies by type: {noisy}")

    # Compute exact energies if this option is set
    if kwargs["mitigation_kwargs"]["tflo_exact_energies"] is None:
        # Turn off corrections to sqrt(iswap) gates for exact calculation
        set_sqrt_iswap_corrections(None, None, False)

        fhvqe.circuit.insert_synthetic_errors = False
        fhvqe.circuit.recompilation = False

        tock = time.perf_counter()
        print(f"---Time between calcs: {tock-tick}")
        tick = time.perf_counter()

        exact_E = energy_calculation_exact(params_list,
                                           (num_tests,
                                            [(i, 0) for i in range(num_tests)]),
                                           qubits,
                                           exact_measurement_set,
                                           nmeas,
                                           num_layers=num_layers,
                                           num_params=num_params,
                                           batch_evals=num_tests,
                                           run_args = {"qubit_order": qubits})
        exact_measured_values = copy.deepcopy(fhvqe.experiment.measured_values)
        tflo_data["exact_measured_values"] = exact_measured_values

        tock = time.perf_counter()
        print(f"---Time for exact calculation: {tock-tick}")
        tick = time.perf_counter()

        exact = {}
        for measurement in measurement_set:
            m_type = measurement.type
            exact[m_type] = np.array([x["Em"] for x in exact_measured_values[m_type]])
            if __debug__:
                print(f"Exact {m_type}: avg error: {np.mean(np.abs(noisy[m_type] - exact[m_type]))}")

        print(f"***TFLOE: ", end='')
        for idx,e in enumerate(exact_E):
            print(f'{e}', end='')
            if idx < len(exact_E)-1:
                print(f',', end='')
            else:
                print('')
    else:
        print(f"Using saved exact energies")
        exact_E = kwargs["mitigation_kwargs"]["tflo_exact_energies"]



# POSTSELECTION ON OCCUPATION NUMBER AND SPIN TYPE

def error_detect_samples_wrap(nocc):
    """Wrapper for postselecting on occupation number."""
    def error_detect_samples(results):
        """Postselect samples in results with the right occupation number.

        Args:
            results -- numpy array of shape ??
            nocc -- occupation number

        Returns:
            Postselected numpy array
        """
        good_indices = np.where(np.sum(results, axis=1) == nocc)[0]
        return results[good_indices]
    return error_detect_samples


def error_detect_spins_samples_wrap(nocc_1, nocc_2):
    """Wrapper for postselecting on occupation number and spin type."""
    def error_detect_spins_samples(results):
        """Postselect samples in results with the right occupation number and spin-type.

        Args:
            results -- numpy array of shape ??
            nocc_1 -- occupation number of spin up
            nocc_2 -- occupation number of spin down

        Returns:
            Postselected numpy array
        """
        size = results.shape[1]//2
        good_indices = np.where((np.sum(results[:,:size], axis=1) == nocc_1) &
                                (np.sum(results[:,size:], axis=1) == nocc_2))[0]
        return results[good_indices]
    return error_detect_spins_samples
