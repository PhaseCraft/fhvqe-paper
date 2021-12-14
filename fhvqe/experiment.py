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
import copy
import datetime
import logging
import time
from collections import defaultdict
from itertools import chain

import cirq
import uncertainties
import cirq_google as cg
import numpy as np
import pandas as pd
import uncertainties.unumpy as unp
from cirq.google import optimized_for_sycamore
from uncertainties import ufloat

import fhvqe.circuit
import fhvqe.error_mitigation
from fhvqe.circuit import (ansatz, ansatz_multilayer_circuit,
                           ansatz_multilayer_circuit_merge,
                           ansatz_multistep, prepH, prepV, prepV2wrap)
from fhvqe.tools import map_site_to_JW

module_logger = logging.getLogger("fhvqe.experiment")
Measurements = collections.namedtuple("Measurements", "pairs prep analysis")
Circuits = collections.namedtuple("Circuits", "device, initial, final, ansatz, type, analysis")
# subbatch described a sub-groupation of a single batch, for example if the
  # measurement is part of the same gradient evaluation
# descriptor is some extra descriptor of the measurement/circuit, for example
  # the sign -1/+1/0 which means that it's the datapoint -1 delta away from some
  # parameters/ +1 delta away from some parameters/at parameters (all giving
  # different set of theta).
# type is measurement type of the circuit
# batchiteration is the iterator over different thetas (for a single theta we
  # have multiple circuits corresponding to different measurement needed to be
  # taken)
Descriptions = collections.namedtuple("Descriptions", "subbatch, descriptor, type, analysis, batchiteration")

measured_values = {}

def start_logger(logfile="fhvqe.log"):
    """Initializes the logger for fhvqe module.
    """
    # create logger with 'fhvqe'
    logger = logging.getLogger('fhvqe')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def analyze(measurement_type, pairs, nh, nv, t=1, U=2, **kwargs):
    """Returns function which analyzes a numpy array of measurement results.

    Function it returns assumes results are in numpy array of shape ??

    Args:
        measurement_type -- onsite, horizontal or vertical measurement
        pairs -- pairs of qubits which should be combined for measurement
        nh -- number of horizontal sites
        nv -- number of vertical sites
        t -- hopping interaction parameter
        U -- on-site interaction parameter

    Returns:
        Function for analysis of results
    """
    if __debug__:
        module_logger.debug(f"Preparing analysis function for {measurement_type}")
    def _parity_indices(q1, q2, nh):
        """Internal function giving parity for qubits q1 to q2."""
        index_list = range(q1+1, q2)
        return index_list

    def analyzeO(results):
        """Analyzes onsite measurement results.
        """
        sum_tot = 0
        for (q1, q2) in pairs:
            res = np.mean(results[q1] * results[q2])
            sum_tot += res
#        print(f'Onsite energy: {U*sum_tot}')
        return  U * sum_tot
    def analyzeH(results):
        """Analyzes horizontal measurement results."""
        sum_tot = 0
        for (q1, q2) in pairs:
            res = np.mean(results[q2] - results[q1])
            sum_tot += res
#        print(f'H energy: {-t*sum_tot}')
        return -t * sum_tot
    def analyzeV(results):
        """Analyzes vertical measurement results (applies parity corrections)."""
        sum_tot = 0
        size = results.shape[1]
        for (q1, q2) in pairs:
            res = results[q2] - results[q1]
            parity = 0
            for q in _parity_indices(q1, q2, nh):
                parity += results[q]
            parity = 1 - (parity % 2) * 2
            sum_tot += np.mean(res * parity)
#        print(f'V energy: {-t*sum_tot}')
        return -t * sum_tot
    if measurement_type == "onsite":
        return analyzeO
    if measurement_type == "vert":
        return analyzeV
    if measurement_type in ["horiz", "vert0", "vert1"]:
        return analyzeH


def analyze_exact(measurement_type, pairs, nh, nv, t=1, U=2, **kwargs):
    """Returns function which analyzes a numpy array of measurement results.

    Function it returns assumes results are in numpy array of shape ??

    Args:
        measurement_type -- onsite, horizontal or vertical measurement
        pairs -- pairs of qubits which should be combined for measurement
        nh -- number of horizontal sites
        nv -- number of vertical sites
        t -- hopping interaction parameter
        U -- on-site interaction parameter

    Returns:
        Function for analysis of results
    """
    if __debug__:
        module_logger.debug(f"Preparing exact analysis function for {measurement_type}")
    def _parity_indices(q1, q2, nh):
        """Internal function giving parity for qubits q1 to q2."""
        index_list = range(q1+1, q2)
        return index_list

    total_len = nh*nv*2
    import itertools
    lst = list(itertools.product([0, 1], repeat = total_len - 2))
    if measurement_type == "onsite":
        all_indices = []
        for (q1, q2) in pairs:
            all_indices += [cirq.big_endian_bits_to_int(element[:q1]+(1,)+element[q1:q2-1]+(1,)+element[q2-1:]) for element in lst]
    if measurement_type in ["horiz"]:
        all_indices_plus = []
        all_indices_minus = []
        for (q1, q2) in pairs:
            all_indices_plus += [cirq.big_endian_bits_to_int(element[:q1]+(0,)+element[q1:q2-1]+(1,)+element[q2-1:]) for element in lst]
            all_indices_minus += [cirq.big_endian_bits_to_int(element[:q1]+(1,)+element[q1:q2-1]+(0,)+element[q2-1:]) for element in lst]
    if measurement_type in ["vert0", "vert1", "vert"]:
        all_indices_plus = []
        all_indices_minus = []
        for (q1, q2) in pairs:
            parity = 1
            for element in lst:
                parity = 1 - (sum(element[q1:q2-1]) % 2) * 2
                if parity == 1:
                    all_indices_plus.append(cirq.big_endian_bits_to_int(element[:q1]+(0,)+element[q1:q2-1]+(1,)+element[q2-1:]))
                    all_indices_minus.append(cirq.big_endian_bits_to_int(element[:q1]+(1,)+element[q1:q2-1]+(0,)+element[q2-1:]))
                else:
                    all_indices_minus.append(cirq.big_endian_bits_to_int(element[:q1]+(0,)+element[q1:q2-1]+(1,)+element[q2-1:]))
                    all_indices_plus.append(cirq.big_endian_bits_to_int(element[:q1]+(1,)+element[q1:q2-1]+(0,)+element[q2-1:]))
    def analyzeO(results):
        """Analyzes onsite measurement results.
        """
        sum_tot = np.sum(np.abs(results[all_indices])**2)
#        print(f'Onsite: {U * sum_tot}')
        return  U * sum_tot
    def analyzeH(results):
        """Analyzes horizontal measurement results."""
        sum_tot = np.sum(np.abs(results[all_indices_plus])**2) - np.sum(np.abs(results[all_indices_minus])**2)
#        print(f'Horiz: {-t * sum_tot}')
        return -t * sum_tot
    def analyzeV(results):
        """Analyzes vertical measurement results (applies parity corrections)."""
        #TODO: fix this one... parity needs to be handled..
        sum_tot = np.sum(np.abs(results[all_indices_plus])**2) - np.sum(np.abs(results[all_indices_minus])**2)
        return -t * sum_tot
    if measurement_type == "onsite":
        return analyzeO
    if measurement_type == "vert":
        return analyzeV
    if measurement_type in ["horiz", "vert0", "vert1"]:
        return analyzeH

def analyze_exact_mgd(measurement_type, pairs, nh, nv, t=1, U=2, **kwargs):
    """Returns function which analyzes a numpy array of measurement results.

    Function it returns assumes results are in numpy array of shape ??

    Args:
        measurement_type -- onsite, horizontal or vertical measurement
        pairs -- pairs of qubits which should be combined for measurement
        nh -- number of horizontal sites
        nv -- number of vertical sites
        t -- hopping interaction parameter
        U -- on-site interaction parameter

    Returns:
        Function for analysis of results
    """
    if __debug__:
        module_logger.debug(f"Preparing exact analysis function for {measurement_type}")
    def _parity_indices(q1, q2, nh):
        """Internal function giving parity for qubits q1 to q2."""
        index_list = range(q1+1, q2)
        return index_list
        
    print(f"analyze_exact_mgd {measurement_type}")

    total_len = nh*nv*2
    import itertools
    lst = list(itertools.product([0, 1], repeat = total_len - 2))
    if measurement_type == "onsite":
        all_indices = []
        for (q1, q2) in pairs:
            all_indices += [cirq.big_endian_bits_to_int(element[:q1]+(1,)+element[q1:q2-1]+(1,)+element[q2-1:]) for element in lst]
    if measurement_type in ["horiz"]:
        all_indices_plus = []
        all_indices_minus = []
        for (q1, q2) in pairs:
            all_indices_plus += [cirq.big_endian_bits_to_int(element[:q1]+(0,)+element[q1:q2-1]+(1,)+element[q2-1:]) for element in lst]
            all_indices_minus += [cirq.big_endian_bits_to_int(element[:q1]+(1,)+element[q1:q2-1]+(0,)+element[q2-1:]) for element in lst]
    if measurement_type in ["vert0", "vert1", "vert"]:
        all_indices_plus = []
        all_indices_minus = []
        for (q1, q2) in pairs:
            parity = 1
            for element in lst:
                parity = 1 - (sum(element[q1:q2-1]) % 2) * 2
                if parity == 1:
                    all_indices_plus.append(cirq.big_endian_bits_to_int(element[:q1]+(0,)+element[q1:q2-1]+(1,)+element[q2-1:]))
                    all_indices_minus.append(cirq.big_endian_bits_to_int(element[:q1]+(1,)+element[q1:q2-1]+(0,)+element[q2-1:]))
                else:
                    all_indices_minus.append(cirq.big_endian_bits_to_int(element[:q1]+(0,)+element[q1:q2-1]+(1,)+element[q2-1:]))
                    all_indices_plus.append(cirq.big_endian_bits_to_int(element[:q1]+(1,)+element[q1:q2-1]+(0,)+element[q2-1:]))
    def analyzeO(results):
        """Analyzes onsite measurement results.
        """
        sum_tot = ufloat(np.sum(np.abs(results[all_indices])**2), np.sqrt(len(results[all_indices])))
#        print(f'Onsite: {U * sum_tot}')
        return  U * sum_tot
    def analyzeH(results):
        """Analyzes horizontal measurement results."""
        sum_tot = ufloat(np.sum(np.abs(results[all_indices_plus])**2) - np.sum(np.abs(results[all_indices_minus])**2), np.sqrt(len(results[all_indices_plus])) )
#        print(f'Horiz: {-t * sum_tot}')
        return -t * sum_tot
    def analyzeV(results):
        """Analyzes vertical measurement results (applies parity corrections)."""
        #TODO: fix this one... parity needs to be handled..
        sum_tot = ufloat(np.sum(np.abs(results[all_indices_plus])**2) - np.sum(np.abs(results[all_indices_minus])**2), np.sqrt(len(results[all_indices_plus])) )
        return -t * sum_tot
    if measurement_type == "onsite":
        return analyzeO
    if measurement_type == "vert":
        return analyzeV
    if measurement_type in ["horiz", "vert0", "vert1"]:
        return analyzeH
        

def analyze_mgd(measurement_type, pairs, nh, nv, t=1, U=2, **kwargs):
    """Returns function which analyzes a numpy array of measurement results.

    Function it returns assumes results are in numpy array of shape ??

    Args:
        measurement_type -- onsite, horizontal or vertical measurement
        pairs -- pairs of qubits which should be combined for measurement
        nh -- number of horizontal sites
        nv -- number of vertical sites
        t -- hopping interaction parameter
        U -- on-site interaction parameter

    Returns:
        Function for analysis of results
    """
    if __debug__:
        module_logger.debug(f"Preparing analysis function for {measurement_type}")
    def _parity_indices(q1, q2, nh):
        """Internal function giving parity for qubits q1 to q2."""
        index_list = range(q1+1, q2)
        return index_list

    def analyzeO(results):
        """Analyzes onsite measurement results.
        """
        sum_tot = ufloat(0.,0.)
        for (q1, q2) in pairs:
            res = np.mean(results[q1] * results[q2])
            std = np.std(results[q1] * results[q2], ddof=1)
            std /= np.sqrt(len(results[q1]))
            sum_tot += ufloat(res, std)
        return U * sum_tot
    def analyzeH(results):
        """Analyzes horizontal measurement results."""
        sum_tot = ufloat(0.,0.)
        for (q1, q2) in pairs:
            res = np.mean(results[q2] - results[q1])
            std = np.std(results[q2] - results[q1], ddof=1)
            std /= np.sqrt(len(results[q1]))
            sum_tot += ufloat(res, std)
        return -t * sum_tot
    def analyzeV(results):
        """Analyzes vertical measurement results (applies parity corrections)."""
        sum_tot = ufloat(0.,0.)
        size = results.shape[1]
        for (q1, q2) in pairs:
            res = results[q2] - results[q1]
            parity = 0
            for q in _parity_indices(q1, q2, nh):
                parity += results[q]
            parity = 1 - (parity % 2) * 2
            std = np.std(res * parity, ddof=1) / np.sqrt(len(res))
            res = np.mean(res * parity)
            sum_tot += ufloat(res, std)
        return -t * sum_tot
    if measurement_type == "onsite":
        return analyzeO
    if measurement_type == "vert":
        return analyzeV
    if measurement_type in ["horiz", "vert0", "vert1"]:
        return analyzeH


def create_measurements(nh, nv, offset, measurement_type):
    """Creates necessary measurement details for a given type on a given lattice.

    Given the lattice size, whether odd or even pairs are being measured,
    and the measurement type, this function returns a namedtuple
    with the pairs of qubits to be measured, the circuit preparation
    function and the measurement_type to be passed to the analysis
    function.
    The measurement_type can be:
    "onsite", "horiz", "vert", "vert0", "vert1"

    Args:
        nh -- number of horizontal sites
        nv -- number of vertical sites
        offset -- offset taking care of odd vs even pairing
        measurement_type -- onsite, horizontal or vertical measurement

    Returns:
        Measurements namedtuple with measurement
        (pairs, preparation circuit, analysis type)
    """
    n = nh * nv
    if measurement_type == "onsite":
        pairs = [(i, i+n) for i in range(n)]
        prep = None
    if measurement_type == "horiz":
        pairs = [(i+j, i+j+1) for i in range(0, 2*n, nh) for j in range(offset,nh-1,2)]
        prep = prepH
    if measurement_type == "vert":
        pairst = [(i*nh+j, (i+1)*nh+j) for i in range(offset, nv-1, 2) for j in range(nh)]
        pairst += [(i*nh+j+n, (i+1)*nh+j+n) for i in range(offset, nv-1, 2) for j in range(0, nh)]
        pairs = [ (map_site_to_JW(nh, nv, site1), map_site_to_JW(nh, nv, site2)) for (site1, site2) in pairst]
        prep = prepV
    if measurement_type == "vert0":
        pairs = [(i+j, i+j+1) for i in range(0, 2*n, n) for j in range(1,n-1,2)]
        prep = prepV
    if measurement_type == "vert1":
        pairs = [(i+j, i+j+1) for i in range(0, 2*n, n) for j in range(1,n-1,2)]
        prep = prepV2wrap(nh, nv)
    print(f"Prepped {measurement_type}, pairs={pairs}")
    return Measurements(pairs=pairs, prep=prep, analysis=measurement_type)


def create_all(nh, nv):
    """Creates all the necessary measurement details for a given lattice.

    This creates the measurement pairs assuming no extra details
    on the connection of the qubits.

    Args:
        nh -- number of horizontal sites
        nv -- number of vertical sites

    Returns:
        Dictionary of Measurement namedtuples.
    """
    measurement_types = ["onsite0"]
    if nh > 1:
        measurement_types.append("horiz0")
    if nh > 2:
        measurement_types.append("horiz1")
    if nv > 1:
        if nh == 2:
            measurement_types.append("vert02")
        else:
            measurement_types.append("vert0")
    if nv > 2 or nh == 2:
        if nh == 2:
            measurement_types.append("vert12")
        else:
            measurement_types.append("vert1")
    measurements = {}
    print(f"Meas types {measurement_types}")
    for measurement_type in measurement_types:
        measurements[measurement_type]=create_measurements(nh, nv,
                                                           int(measurement_type[-1]),
                                                           measurement_type[:-1])
    return measurements


def create_executable(init_prog, ansatz_def, theta,
                      optimizer=optimized_for_sycamore,
                      optimizer_kwargs=None,
                      prep=None, **kwargs):
    """Create executable program with initial and final circuits and optimizations.


    Args:
        init_prog -- initial state preparation
        ansatz_def -- ansatz chosen for the circuit
        theta -- parameters circuits is being evaluated for
        optimizer -- optimization function (default optimized_for_sycamore)
        prep -- measurement preparation circuit (default None)

    Returns:
        Executable program
    """
    def _check_one_two_qubit_moments(circuit):
        "Checks whether a moment only either contains one qubits gates or two."
        try:
            for i, moment in enumerate(circuit):
#                print('Moment {}:\n{}'.format(i,moment))
                num_qubits = len(list(moment)[0].qubits)
                for op in moment:
                    assert num_qubits == len(op.qubits)
#            print("all good")
        except:
            print("bad news, at least one of the moments is misaligned!")
    prog = []

    if "remap" in kwargs:
        qubits = kwargs["remap"]

    prog = cirq.Circuit()

    prog.append(init_prog)
    prog.append(ansatz_def)
    
    # add measurement prep
    if prep is not None:
        prog.append(prep)
        
    if __debug__:
        module_logger.debug(f"number of moments preoptimization: {len(prog)}")
        module_logger.debug(f"depth preoptimization: {len(cirq.Circuit(prog.all_operations()))}")
    
    # finally compile
    if optimizer is not None:
        prog = optimizer(prog, **optimizer_kwargs)

    if __debug__:
        module_logger.debug(f"number of moments post-optimization: {len(prog)}")
        module_logger.debug(f"depth post-optimization: {len(cirq.Circuit(prog.all_operations()))}")

    if __debug__:
        _check_one_two_qubit_moments(prog)
        
    # add measurements (do last so they are the final things to be executed)
    prog.append(cirq.measure(*qubits, key='x'))
    if __debug__:
        try:
            cirq.google.Sycamore.validate_circuit(prog)
        except:
            print("oh no, circuit is not sycamore compliant!")
            for i, moment in enumerate(prog):
                print("Moment {}:\n{}".format(i,moment))
            quit()
            
    return prog


def run_executables(circuits, run_args, qc):
    """Actually runs a set of quantum circuits.
    Args:
        circuits -- the quantum circuits to execute
        run_args -- metadata for the circuits
        qc -- Engine instance to run the circuits on
    Returns:
        The result of running the circuits.
    """
    prog_id = None
    job_id = None
    if "program_id" in run_args:
        timestr = datetime.datetime.now().strftime("%H%M%S-%f")
        temp_prog_id = run_args["program_id"]
        run_args["program_id"] = run_args["program_id"] + "-" + timestr
        prog_id = run_args["program_id"]
        job_id = run_args["job_id"]
    results = qc.run_batch(circuits,
                           **run_args)
    if "program_id" in run_args:
        run_args["program_id"] = temp_prog_id
    return results, prog_id, job_id


def run_executables_exact(circuits, run_args, qc):
    """Run a set of quantum circuits using the exact simulator.
    Args:
        circuits -- the quantum circuits to execute
        run_args -- metadata for the circuits
        qc -- not used
    Returns:
        The result of running the circuits.
    """
    from cirq import Simulator
    simulator = Simulator()
    results = []
    for circuit in circuits:
        circuit2 = cirq.Circuit(circuit.moments[:-1])  # run the circuit except for the final layer of measurements
        results.append(simulator.simulate(circuit2, qubit_order=run_args["qubit_order"]))# **run_args))
    return results, None, None


def prepare_executables(batch_params, descriptions, measurement_set, run_args,
                        num_trials, qubits, optimizer=None,
                        optimizer_kwargs=None):
    """Prepares a set of quantum circuits for execution, by including the final
       measurement transformations required.
    Args:
        batch_params -- parameters for the circuits
        descriptions -- additional description
        measurement_set -- which measurement types to use
        run_args -- additional metadata
        num_trials -- the number of shots to use
        qubits -- which qubits to run on
        optimizer -- which optimizer to use
        optimizer_kwargs -- additional optimizer arguments
    Returns:
        Prepared circuits and additional details
    """
    extra_params = []
    circuits = []
    circuits_details = []
    module_logger.info("Preparing executables.")
    
    for i, (params, details) in enumerate(zip(batch_params, descriptions)):
        if __debug__:
            module_logger.debug(f"ansatz with params: {params}")
        meas = measurement_set[0]
        ansatz_compiled_no_hop, ansatz_compiled_with_hop = ansatz_multilayer_circuit_merge(meas.ansatz,
                                                          params, qubits)
        for meas in measurement_set:
            meas_prep = meas.final
            ansatz_compiled = ansatz_compiled_with_hop
            if meas.final is not None and not isinstance(meas.final, cirq.circuits.circuit.Circuit):
                meas_prep = cirq.Circuit(ansatz(meas.final, [params[-1][-1]], qubits))
                ansatz_compiled = ansatz_compiled_no_hop
            circ = create_executable(meas.initial,
                                     ansatz_compiled,
                                     params,
                                     prep=meas_prep,
                                     qubits=qubits,
                                     remap=qubits,
                                     optimizer=optimizer,
                                     optimizer_kwargs=optimizer_kwargs)
            if run_args is None: run_args = {}
            
            circuits.append(circ)
            circuits_details.append(Descriptions(*details + (meas.type, meas.analysis, i)))
            extra_params.append(None)

    run_args["params_list"] = extra_params
    run_args["repetitions"] = num_trials
    return circuits, run_args, circuits_details


def save_results(data, samples_filename):
    """Save retrieved samples to a text file."""
    
    results = [''.join(str(x) for x in result) + '\n' for result in data]
    with open(samples_filename, 'a') as results_file:
        results_file.writelines(results)
        results_file.write('\n\n')


def extract_values(batch_evals, thetas, circuits_details, results,
                   sample_error_mitigation=None,
                   noise_matrix=None,
                   value_error_mitigation=None,
                   samples_filename=None,
                   save_samples=False,
                   **kwargs):
    """Extract and analyse results returned from running the quantum circuits.
    """
    E = np.zeros(batch_evals)
    global measured_values
    measured_values = {}
    E_each_evaluation = {}

    for (circ_desc, result) in zip(circuits_details, results):
    
        if circ_desc.subbatch not in E_each_evaluation:
            E_each_evaluation[circ_desc.subbatch] = {}
        if circ_desc.descriptor not in E_each_evaluation[circ_desc.subbatch]:
            E_each_evaluation[circ_desc.subbatch][circ_desc.descriptor] = 0.0
        params = thetas[circ_desc.batchiteration]
        if isinstance(result, list):
            data = result[0].measurements['x'].astype(int)
        else:
            data = result.measurements['x'].astype(int)
        num_trials = data.shape[0]
        if sample_error_mitigation is not None:
            mitigation_args = []
            if "correct_m_type" in kwargs:
                mitigation_args += [noise_matrix, circ_desc.type]
            data = sample_error_mitigation(data, *mitigation_args)
            
        # Save samples from the final onsite measurement to a file if needed
        if save_samples and samples_filename is not None and circ_desc.descriptor == 0 and circ_desc.type == "onsite0":
            save_results(data, samples_filename)
            
        Em = circ_desc.analysis(data.transpose())
        
        E[circ_desc.batchiteration] += Em
        vals = {"Em": Em,
                "num_trials": num_trials,
                "post-processed": data.shape[0],
                "params": params,
                "grad": circ_desc.subbatch,
                "sgn": circ_desc.descriptor,
                }
        measured_values.setdefault(circ_desc.type, []).append(vals)
        E_each_evaluation[circ_desc.subbatch][circ_desc.descriptor] += Em
        
    measured_values["all_E"] = E_each_evaluation
    return E, measured_values


def extract_values_mgd(batch_evals, thetas, circuits_details, results,
                   sample_error_mitigation=None,
                   noise_matrix=None,
                   value_error_mitigation=None,
                   samples_filename=None,
                   save_samples=False,
                   **kwargs):
    """Extract and analyse results returned from running the quantum circuits,
       in a form suitable for use with BayesMGD.
    """
    E = np.zeros(batch_evals, dtype=uncertainties.core.Variable)
    global measured_values
    measured_values = {}
    E_each_evaluation = {}
    for (circ_desc, result) in zip(circuits_details, results):
        if circ_desc.subbatch not in E_each_evaluation:
            E_each_evaluation[circ_desc.subbatch] = {}
        if circ_desc.descriptor not in E_each_evaluation[circ_desc.subbatch]:
            E_each_evaluation[circ_desc.subbatch][circ_desc.descriptor] = 0
        params = thetas[circ_desc.batchiteration]
        if isinstance(result, list):
            data = result[0].measurements['x'].astype(int)
        else:
            data = result.measurements['x'].astype(int)
        num_trials = data.shape[0]
        if sample_error_mitigation is not None:
            mitigation_args = []
            if "correct_m_type" in kwargs:
                mitigation_args += [noise_matrix, circ_desc.type]
            data = sample_error_mitigation(data, *mitigation_args)

        # Save samples from the final onsite measurement to a file if needed
        if save_samples and samples_filename is not None and circ_desc.descriptor == 0 and circ_desc.type == "onsite0":
            save_results(data, samples_filename)

        Em = circ_desc.analysis(data.transpose())
        E[circ_desc.batchiteration] += Em
        vals = {"Em": Em.nominal_value,
                "num_trials": num_trials,
                "post-processed": data.shape[0],
                "params": params,
                "grad": circ_desc.subbatch,
                "sgn": circ_desc.descriptor,
                }
        measured_values.setdefault(circ_desc.type, []).append(vals)
        E_each_evaluation[circ_desc.subbatch][circ_desc.descriptor] += Em.nominal_value
    measured_values["all_E"] = E_each_evaluation
    return E, measured_values


def extract_values_exact(batch_evals, thetas, circuits_details, results,
                         sample_error_mitigation=None,
                         noise_matrix=None,
                         value_error_mitigation=None, **kwargs):
    """Extract and analyse results obtained from running an exact simulation."""
    
    E = np.zeros(batch_evals)
    global measured_values
    measured_values = {}
    E_each_evaluation = {}
    for (circ_desc, result) in zip(circuits_details, results):
        if circ_desc.subbatch not in E_each_evaluation:
            E_each_evaluation[circ_desc.subbatch] = {}
        if circ_desc.descriptor not in E_each_evaluation[circ_desc.subbatch]:
            E_each_evaluation[circ_desc.subbatch][circ_desc.descriptor] = 0.0
        params = thetas[circ_desc.batchiteration]
        data = result.state_vector()
        Em = circ_desc.analysis(data)
        if isinstance(Em, uncertainties.core.AffineScalarFunc):
            E[circ_desc.batchiteration] += Em.nominal_value
        else:
            E[circ_desc.batchiteration] += Em
        vals = {"Em": Em,
                "num_trials": 0,
                "post-processed": (0,),
                "params": params,
                "grad": circ_desc.subbatch,
                "sgn": circ_desc.descriptor,
                }
        measured_values.setdefault(circ_desc.type, []).append(vals)
        E_each_evaluation[circ_desc.subbatch][circ_desc.descriptor] += Em
    measured_values["all_E"] = E_each_evaluation
    return E, measured_values


def energy_calculation_wrap(run_executables_func=run_executables,
                            extract_values_func=extract_values):
    def energy_calculation(thetas, description, qubits, measurement_set,
                           num_trials,
                           batch_evals = 0, num_layers = 0,
                           num_params = 0,
                           optimizer = None,
                           optimizer_kwargs = None,
                           mitigation_kwargs = {},
                           run_args = None,
                           save_samples=False,
                           samples_filename=""):
        """Objective function which returns energy calculations for a list of angles.

        Args:
            thetas -- parameters for which energy is evaluated
            qubits -- qubits on which the program is being executed
            measurement_set -- set of measurements to be carried out
            num_trials -- number of shots

        Returns:
            Tuple containing evaluated energy values, and a dictionary with
            further details on results, parameters etc.
        """
        global measured_values
        
        params = np.array(thetas)
        batch_evals = description[0] #description.number_of_circuits_in_batch
        descriptions = description[1] #description.descriptions
        params = params.reshape(batch_evals, num_layers, num_params)
    
        tick = time.perf_counter()
        print(f"---Preparing executables")
        circuits, run_args, circuits_details = prepare_executables(params, descriptions,
                                                      measurement_set,
                                                      run_args,
                                                      num_trials,
                                                      qubits,
                                                      optimizer=optimizer,
                                                      optimizer_kwargs=optimizer_kwargs)
        start_timestamp = datetime.datetime.now()
        tock = time.perf_counter()
        print(f"---Running executables, {len(circuits)} in batch, delta={tock-tick}")
        tick = time.perf_counter()
        results, program_id, job_id = run_executables_func(circuits, run_args,
                                                     measurement_set[0].device)
        tock = time.perf_counter()
        print(f"---Extracting values delta={tock-tick}")
        tick = time.perf_counter()
        E, mv = extract_values_func(batch_evals, params,
                                                 circuits_details, results,
                                                 save_samples=save_samples,
                                                 samples_filename=samples_filename,
                                                 **mitigation_kwargs)
        tock = time.perf_counter()
        print(f"---Extracted values delta={tock-tick}")
        measured_values = mv
        measured_values["start_timestamp"] = str(start_timestamp)
        measured_values["end_timestamp"] = str(datetime.datetime.now())
        if "program_id" in run_args:
            measured_values["program_id"] = program_id
            measured_values["job_id"] = job_id
        return E
    return energy_calculation


def insert_x_gates_unused(circuit):
    """Inserts X gates on qubits that are unused for a long period of time,
       in an attempt to cancel out unwanted Z rotations."""
    output_circuit = copy.deepcopy(circuit)
    
    all_qubits = circuit.all_qubits()
    
    for i, moment in enumerate(output_circuit):
        if not is_two_qubit_gate_layer(moment) and i < len(output_circuit) - 3:
            unused_qubits = set(all_qubits)
            for gate in moment:
                unused_qubits.discard(gate.qubits[0])
                
            for gate in output_circuit[i+1]:
                unused_qubits.discard(gate.qubits[0])
                unused_qubits.discard(gate.qubits[1])
                
            for next_layer in range(i+1, len(output_circuit)):
                if not is_two_qubit_gate_layer(output_circuit[next_layer]):
                    break
            for gate in output_circuit[next_layer]:
                unused_qubits.discard(gate.qubits[0])
                
            output1 = cirq.Circuit()
            output2 = cirq.Circuit()
            for qubit in unused_qubits:
                output1.append(cirq.X.on(qubit))
                output2.append(cirq.X.on(qubit))
            for gate in output_circuit[i]:
                output1.append(gate)
            for gate in output_circuit[next_layer]:
                output2.append(gate)
                
            output_circuit[i] = output1[0]
            output_circuit[next_layer] = output2[0]
            
    return output_circuit
    
def is_two_qubit_gate_layer(moment):
    """Helper function to check whether a moment in a circuit contains
       one- or two-qubit gates."""
       
    if len(next(iter(moment)).qubits) == 1:
        return False
    return True
    
def insert_x_gates_all(circuit):
    """Inserts layers of X gates before and after every other layer of 2-qubit gates,
       in an attempt to cancel out unwanted Z rotations."""
       
    all_qubits = circuit.all_qubits()
    x_layer = cirq.Circuit()
    for qubit in all_qubits:
        x_layer.append(cirq.X.on(qubit))

    output_circuit = cirq.Circuit()
        
    two_qubit_gate_layer_count = 0
    for i, moment in enumerate(circuit):
        if is_two_qubit_gate_layer(moment):
            two_qubit_gate_layer_count += 1
        if is_two_qubit_gate_layer(moment) and two_qubit_gate_layer_count % 2 == 1:
            output_circuit.append(x_layer)
        output_circuit.append(moment)
        if is_two_qubit_gate_layer(moment) and two_qubit_gate_layer_count % 2 == 1:
            output_circuit.append(x_layer)
        
    return output_circuit
    
    
def optimize_one_qubit_layer(layer):
    """Optimizes a set of moments in a quantum circuit which contain 1-qubit gates only.
       Combines each sequence of 1-qubit gates into one gate."""
       
    if len(layer) == 1:
        return layer
        
    matrices = {}
    needs_update = {}
    first_gate = {}
        
    # Prepare a list of matrices corresponding to the 1-qubit gate acting on each qubit.
    for moment in layer:
        for gate in moment:
            qubit = gate.qubits[0]
            if matrices.get(qubit) is None:
                matrices[qubit] = np.matrix(cirq.unitary(gate))
                first_gate[qubit] = gate
            else:
                matrices[qubit] = np.matrix(cirq.unitary(gate)) @ matrices[qubit]
                needs_update[qubit] = True
            
    # Work out a hardware-native gate decomposition for each 1-qubit matrix.
    output_layer = cirq.Circuit()
    for qubit, matrix in matrices.items():
        if not np.allclose(matrix, [[1,0],[0,1]]):
            if needs_update.get(qubit):
                gate = cirq.single_qubit_matrix_to_phxz(matrix)
                if gate is not None:
                    output_layer.append(gate.on(qubit))
            else:
                output_layer.append(first_gate[qubit])
                
    return output_layer
    
def check_circuit(circuit):
    """Checks whether a quantum circuit is in the desired form that alternates
       between layers of 1- and 2-qubit gates."""
    for i, moment in enumerate(circuit):
        has_one_qubit_gate = False
        has_two_qubit_gate = False
        for gate in moment:
            if len(gate.qubits) == 1:
                has_one_qubit_gate = True
            if len(gate.qubits) == 2:
                has_two_qubit_gate = True
        if has_one_qubit_gate and has_two_qubit_gate:
            print(f"ERROR: moment {i} contains one and two qubit gates")
            for i, moment in enumerate(circuit):
                print("Moment {}:\n{}".format(i,moment))
            quit()

def optimize_in_layers(circuit, **kwargs):
    """ Optimizes a quantum circuit for Sycamore, but preserving all 2-qubit gate layers.
        Assumes that we're given a circuit which has some layers of 1-qubit gates, and some layers of
        2-qubit gates. Combines and optimizes the 1-qubit layers, but leaves the 2-qubit gate layers alone.

    Args:
        circuit -- circuit to optimize

    Returns:
        optimized circuit
    """
    output_circuit = cirq.Circuit()
    output_layer = cirq.Circuit()
    
    tick = time.perf_counter()
    
    if __debug__:
        check_circuit(circuit)
    
    # Optimize the layers of 1-qubit gates in the circuit one-by-one.
    for i, moment in enumerate(circuit):
        num_qubits = len(next(iter(moment)).qubits)
        if num_qubits <= 1:
            output_layer.append(moment)
        else:
            output_circuit.append(optimize_one_qubit_layer(output_layer))
            output_circuit.append(moment)
            output_layer = cirq.Circuit()

    output_circuit.append(optimize_one_qubit_layer(output_layer))
 
    # Insert X gates for error mitigation if required.
    insert_xs = kwargs.get("add_x_gates")
    
    if insert_xs is None:
        return output_circuit
    
    if insert_xs == "unused":
        circuit_with_xs = insert_x_gates_unused(output_circuit)
    elif insert_xs == "all":
        circuit_with_xs = insert_x_gates_all(output_circuit)
    else:
        circuit_with_xs = output_circuit
    
    # Now optimise the circuit again to combine any added X gates with other
    # 1-qubit gates.
    final_output_circuit = cirq.Circuit()
    output_layer = cirq.Circuit()
    
    for i, moment in enumerate(circuit_with_xs):
        num_qubits = len(next(iter(moment)).qubits)
        if num_qubits == 1:
            output_layer.append(moment)
        else:
            output_layer = optimize_one_qubit_layer(output_layer)
            final_output_circuit.append(output_layer)
            final_output_circuit.append(moment)
            output_layer = cirq.Circuit()

    final_output_circuit.append(optimize_one_qubit_layer(output_layer))
    
    tock = time.perf_counter()
    
    return final_output_circuit


NAMED_OBJECTIVE = {
                    "energy": energy_calculation_wrap,
                  }
