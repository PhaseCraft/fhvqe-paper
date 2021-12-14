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

import argparse
import json
import logging
import os
import sys
import time

import cirq
import cirq_google as cg
import numpy as np
import pandas as pd
import scipy
from google.protobuf.json_format import MessageToDict
from numpy import pi as pi
import fhvqe.circuit

from fhvqe.circuit import (NAMED_ANSATZ, ansatz_multilayer_circuit,
                           any_h_by_v_explicit,
                           hopping_measurement_ansatz,
                           initial_state_diff_spin_types)
                           
from fhvqe.error_mitigation import (ERROR_CORRECTION_DICTIONARY,
                                    mitigate_by_tflo,
                                    sample_error_mitigation_func)
                                    
from fhvqe.experiment import (NAMED_OBJECTIVE, Circuits,
                              analyze, analyze_exact, analyze_mgd, analyze_exact_mgd,
                              create_all, create_executable,
                              extract_values_exact, extract_values, extract_values_mgd,
                              optimize_in_layers, run_executables_exact,
                              start_logger)
from fhvqe.optimization import NAMED_OPT
from fhvqe.processing import NumpyEncoder
from fhvqe.settings import (PROJECT_ID, QUBITS_ASSIGNMENT, VERSION, VQE_FOLDER,
                            find_best_qubit_list, find_best_rect, find_best_zigzag,
                            generate_experiment_data_structure,
                            generate_grid_qubits,
                            generate_zigzag_qubits,
                            remap_qubits,
                            standard_settings)
from fhvqe.tools import color_qubits_in_grid, map_JW_to_site, map_site_to_JW

optimizer_def = { "syc": [cg.optimized_for_sycamore, {"new_device": cg.Sycamore,
                                                      "optimizer_type": "sqrt_iswap"}],
                  "syc_layers": [optimize_in_layers, {}],
                  "syc_layers_x1": [optimize_in_layers, {"add_x_gates": "unused"}],
                  "syc_layers_x2": [optimize_in_layers, {"add_x_gates": "all"}],
                  "none": [lambda x: x, {}]
                }
                
module_logger = logging.getLogger("fhvqe.vqe_experiment")
class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()

# Convert a circuit to a text string, moment by moment.
def moment_diagram(circ):
    str = ''
    for i, moment in enumerate(circ):
        str += f"Moment {i}:\n{moment}\n"
    return str

def vqe_experiment(on_chip=False,
                   processor_name="rainbow",
                   processor_gateset="fsim", #cg.FSIM_GATESET,
                   optimize_circuit="syc_layers", #cg.optimized_for_sycamore,
                   qubits=None,
                   nh=2,
                   nv=1,
                   t=1.,
                   U=2.,
                   nocc=2,
                   nocc_1=None,
                   nocc_2=None,
                   opt_algo = "mod_spsa",
                   num_trials_setting=None, #[100, 1000, 10000],
                   grad_evals_setting=None,#[1, 1, 2],
                   max_evals_setting=None, #[500, 100, 100],
                   split=False,
                   opt_args=None,
                   d = 1,
                   initial_params=None,
                   measurement_func=None,
                   initial_prog_mapping=None,
                   chosen_ansatz=None,
                   error_correction={},
                   file_desc=None,
                   settings_file=None,
                   objective_func="energy",
                   objective_args=None,
                   exact=False,
                   benchmark=None,
                   save_samples=False,
                   save_tflo_samples=False,
                   notes="",
                   collect_data=False,
                   **kwargs
                   ):
    """Runs the vqe experiment with given configuration off or on chip.

    Simluation/chip settings:
    on_chip -- Running the experiment on chip or simulated (default False)
    processor_name -- QPU being used (default "rainbow")
    processor_gateset -- QPU gateset being used (default "fsim")
    optimize_circuit -- Circuit optimizer being used (default "syc")
    qubits -- qubits on chip to be used (default None)

    Hamiltonian settings:
    nh -- number of horizontal sites (default 2)
    nv -- number of vertical sites (default 1)
    t -- hopping interaction parameter (default 1)
    U -- on-site interaction parameter (default 2)

    State settings:
    nocc -- total number of fermions in the system (default 2)
    nocc_1 -- number of spin-up fermions in the system (default None)
    nocc_2 -- number of spin-down fermions in the system (default None)

    Optimization settings:
    opt_algo -- optimization algorithm (spsa/mod_spsa), (default mod_spsa)
    num_trials_setting -- the number of trials to be used for spsa (default [100, 1000, 10000])
    grad_evals_setting -- number of gradient evaluations to be used for spsa (default [1, 1, 2])
    max_evals_setting -- number of circuit evaluations to be used for spsa (default [500, 100, 100])
    split -- running the circuits within a single energy evaluations batch or split (default False)
    opt_args -- spsa hyperparameters (default None)
    save_samples -- whether to save the last measurement results to a file (default False)

    Ansatz settings:
    d -- number of ansatz layers (default 1)
    initial_params -- initial ansatz parameters (default None)
    measurement_func -- function that creates the measurement settings (default None)
    initial_prog_mapping -- remapping needed for the initial state which is the ground state
                            of the non-interacting Hamiltonian (default None)
    chosen_ansatz -- ansatz that is being used (default None)

    Error correction settings:
    error_correction -- dictionary of booleans specifying which error mitigation
                        techniques, if any, are used (default None)

    Housekeeping:
    file_desc -- file descriptor to be attached to the files in the project (default None)
    settings_file -- configuration file that is being used (default None)
    """
    # files and time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if file_desc is None:
        extra_desc = timestr
    else:
        if len(file_desc) > 40:
            print("Please limit the file_desc to less than 40 characters")
            sys.exit()
        extra_desc = file_desc + "-" + time.strftime("%m%d-%H%M%S")
    tick = time.perf_counter()

    # data location
    data_struct = generate_experiment_data_structure(nh, nv, VQE_FOLDER,
                                                     extra_desc=extra_desc)

    settings = locals()
    experiment_metadata = settings.copy()

    logger = start_logger(logfile=data_struct.logger)
    Tee(data_struct.stdout_logger, "a")

    experiment_metadata["tag"] = VERSION
    experiment_metadata["start_time"] = timestr

    error_correction_settings = ERROR_CORRECTION_DICTIONARY
    error_correction_settings.update(error_correction)
    error_correction_metadata = {}
    error_correction_metadata["error_correction"] = error_correction_settings

    optimizer = optimizer_def[optimize_circuit][0]
    optimizer_kwargs = optimizer_def[optimize_circuit][1]

    ## SETTING THE DEVICE
    if on_chip:
        engine = cirq.google.Engine(project_id=PROJECT_ID)
        processor = engine.get_processor(processor_name)
        experiment_metadata["device"] = processor_name
        proc_gateset = cg.NAMED_GATESETS[processor_gateset]
        device = processor.get_device([proc_gateset])
        spec = processor.get_device_specification()
        experiment_metadata["specification"] = MessageToDict(spec)
        latest_calibration = processor.get_current_calibration()
        experiment_metadata["calibration"] = pd.DataFrame(latest_calibration.__dict__).to_json()
        print(device)
        qc = engine
    else:
        qc = cirq.Simulator()
        experiment_metadata["device"] = "simulator"
        device = None



    # SETTING VARIOUS PARAMETERS
    # benchmarks
    benchmark_file = None
    if on_chip:
        with open(data_struct.calibration_filename, 'w') as outfile:
            json.dump(experiment_metadata["calibration"], outfile, cls=NumpyEncoder, indent=4)
        # benchmark chosen qubits
        if benchmark == "set":
            total_qs = len(qubits)
            up = qubits[:total_qs//2]
            down = qubits[total_qs//2:]
            energies_qubits = {}
            for (q1, q2), (q3, q4) in zip(zip(up[0:total_qs//2-1], up[1:]),
                                          zip(down[0:total_qs//2-1], down[1:])):
                try:
                    energies_qubits[str(position)] =benchmark_single_qubits(qubits=[[q1, q2, q3, q4]],
                                                                   project_id=PROJECT_ID,
                                                                   processor_name=processor_name,
                                                                   processor_gateset=processor_gateset)
                except:
                    break
            experiment_metadata["benchmark"] = energies_qubits
        if benchmark == "full":
            energies_qubits = benchmark_all_qubits(project_id=PROJECT_ID,
                                                   processor_name=processor_name,
                                                   processor_gateset=processor_gateset,
                                                   save_file=False)
            experiment_metadata["benchmark"] = energies_qubits
            with open(data_struct.benchmark_filename, 'w') as outfile:
                json.dump(experiment_metadata["benchmark"], outfile, cls=NumpyEncoder, indent=4)
            benchmark_file = data_struct.benchmark_filename

    # number of parameters in layer
    num_params = 1 + (nh>1) + (nh>2) + (nv>1) + (nv>2)
    experiment_metadata["num_params"] = num_params
    # load standard settings for occupation number, t, U, ideal parameters and
    # energy
    nocc_t, t_t, U_t, _ , sim_params, sim_E = standard_settings[(nh, nv)]
    experiment_metadata["expected_params"] = sim_params
    experiment_metadata["expected_energy"] = sim_E
    # occupation number
    if nocc is None:
        nocc = nocc_t
        experiment_metadata["nocc"] = nocc
    # t
    # saving samples...
    samples_filename = ""
    if save_samples:
        samples_filename = data_struct.samples_filename
    if t is None:
        t = t_t
        experiment_metadata["t"] = t
    # U
    if U is None:
        U = U_t
        experiment_metadata["U"] = U
    # occupation number of spin up/spin down
    if nocc_1 is None:
        nocc_1 = nocc//2 + nocc%2
        nocc_2 = nocc//2
        experiment_metadata["nocc_1"] = nocc_1
        experiment_metadata["nocc_2"] = nocc_2
    # number of layers in ansatz
    if d is None:
        d = len(params)
        experiment_metadata["d"] = d
    # qubits on which we're running
    if qubits is None:
        qubits, qubit_map = QUBITS_ASSIGNMENT["default"](nh, nv)
    elif qubits[:4] == "best":
        if on_chip:
            all_qubits = device.qubit_set()
            if len(qubits) >= 6:
                metric_name = qubits[5:]
                if metric_name[:16] == "benchmark_metric":
                    if metric_name[17:21] == "time":
                        timestr = metric_name[22:]
                        benchmark_file = "experiments/benchmark/{timestr}/benchmark.json"
                    else:
                        if benchmark != "full":
                            print("If benchmark is being used without a preset file, full benchmark must be run! Set 'benchmark' option in configuration file to 'full'")
                            sys.exit()
            else:
                metric_name = "two_qubit_parallel_sqrt_iswap_gate_xeb_average_error_per_cycle"
            quality_function = find_best_rect
            if chosen_ansatz in ["one_by_n_zigzag", "two_by_n_zigzag"]:
                quality_function = find_best_zigzag
            qubits, qubit_map = find_best_qubit_list(latest_calibration, all_qubits,
                                          nh, nv, metric_name,
                                          benchmark=benchmark_file,
                                          quality_function=quality_function)
            print(f"Using 'best' qubits {qubits}")
        else:
            qubits, qubit_map = QUBITS_ASSIGNMENT["default"](nh, nv)
    elif qubits[:6] == "zigzag":
        qubits_details = qubits.split()
        row = int(qubits_details[1])
        col = int(qubits_details[2])
        orientation = qubits_details[3]
        print(f"Start at: ({row}, {col}), with {orientation}")
        qubits, qubit_map = generate_zigzag_qubits(cirq.GridQubit(row, col), orientation, nh*nv)
        print(f"Using qubits {qubits}")
    else:
        # Use a grid of qubits starting at a particular position
        print(f"Qubits: {qubits}")
        qubits, qubit_map = generate_grid_qubits(qubits, None)
        print(f"Using qubits {qubits}")

    print(f"Using qubits, {qubit_map}:{qubits}")
    if qubit_map == "site":
        qubits = remap_qubits(qubits, map_site_to_JW, [nh, nv])
        print(f"Qubits, remapped, using:{qubits}")

    # Show the qubits that we're actually using in the experiment.
    if on_chip:
        print(color_qubits_in_grid(str(device), qubits))

    experiment_metadata["qubits"] = str(qubits)
    # initial parameters to start optimization at
    if initial_params is None:
        if d > 0:
            initial_params = np.full((d, num_params), 1 / (d * t))
        else:
            initial_params = []
    params = initial_params
    experiment_metadata["params"] = params
    # ansatz that is being used
    experiment_metadata["chosen_ansatz"] = chosen_ansatz
    if chosen_ansatz is None:
        chosen_ansatz = any_h_by_v_explicit(qubits, nh, nv)
        experiment_metadata["chosen_ansatz"] = "h_by_v_explicit"
    elif chosen_ansatz in ["one_by_n_zigzag", "two_by_n_zigzag"]:
        chosen_ansatz = NAMED_ANSATZ[chosen_ansatz](qubits, nh, nv)
    else:
        chosen_ansatz = NAMED_ANSATZ[chosen_ansatz]
        
    # measurements being applied to find energy
    measurement_func = create_all
    
    # initial state mapping
    if initial_prog_mapping == None or initial_prog_mapping == "JW":
        initial_prog_mapping = None
    measurements = measurement_func(nh, nv)
    kwargs = {}
    # loading previous parameters for spsa
    if opt_algo == "spsa" or opt_algo == "mod_spsa":
        if (objective_func in ["set_parameters", "exact_parameters", "parameters_list"]):
            print("Loading parameters...")
            with open("parameters.json") as json_file: ## LOAD PARAMETERS
                loaded_params = json.load(json_file)
                kwargs["given_params"] = loaded_params
    # result analysis functions
    if exact:
        if opt_algo == "mgd" or opt_algo == "bayes_mgd":
            analysis_fns = analyze_exact_mgd
        else:
            analysis_fns = analyze_exact
    else:
        if opt_algo == "mgd" or opt_algo == "bayes_mgd":
            analysis_fns = analyze_mgd
        else:
            analysis_fns = analyze

    ## SAVE SETTINGS
    with open(data_struct.settings_filename, 'w') as outfile:
        json.dump(settings, outfile, cls=NumpyEncoder, indent=4)



    ## ON CHIP JOB DETAILS AND RUN ARGUMENTS
    job_desc = {}
    temp_run_args = {}
    tflo_run_args = {}
    run_args=None
    if exact:
        retrieve_objective = NAMED_OBJECTIVE[objective_func](run_executables_func=run_executables_exact,
                                                             extract_values_func=extract_values_exact)
        run_args = {}
        run_args["qubit_order"] = qubits
    elif opt_algo == "mgd" or opt_algo == "bayes_mgd":
        retrieve_objective = NAMED_OBJECTIVE[objective_func](extract_values_func=extract_values_mgd)
    else:
        retrieve_objective = NAMED_OBJECTIVE[objective_func]()
    if on_chip:
        job_id = f"{extra_desc}"
        program_id = job_id
#        experiment_metadata["job_id"] = job_id
#        experiment_metadata["program_id"] = program_id
        job_labels= {
                     "username": os.getlogin(),
                     "vqe": f"{nh}x{nv}",
                     "compressed": "no",
                     "opt-alg":opt_algo,
                     "program": "main"
                    }
        run_args = {
                    "job_id":job_id,
                    "program_id": program_id,
                    "job_labels":job_labels,
                    "processor_ids":[processor.processor_id],
                    "gate_set":proc_gateset
                    }
        kwargs.update({"run_args": run_args})
        job_desc = {
                    "job_id": job_id,
                    "program_id": program_id,
                    "qubits": str(qubits),
                   }
        # noise run arguments
        temp_run_args = {
                         "job_id":job_id,
                         "program_id": program_id,
                         "job_labels": {
                                        "username": os.getlogin(),
                                        "vqe": f"{nh}x{nv}",
                                        "program": "mitigation"
                                       },
                         "processor_ids":[processor.processor_id],
                         "gate_set":proc_gateset
                        }
        tflo_run_args = dict(temp_run_args)
        tflo_run_args["job_labels"]["program"] = "tflo_calib"
        tflo_run_args["program_id"] = "tflo-" + tflo_run_args["program_id"]

    # Get which pairs we want to measure - needed for error correction
    pairs_set = set()
    pairs_dict = {}
    for (measurement_type, measurement) in measurements.items():
        for pair in measurement.pairs:
            pairs_set.add(pair)
        pairs_dict[measurement_type] = measurement.pairs
    pairs_set = list(pairs_set)

    print(f"Got pairs set {pairs_set}")

    ## ERROR CORRECTION SETTINGS
    
    mitigation_kwargs = {}
    
    postselection = error_correction_settings["occupation_number"] or error_correction_settings["spin_type"]
    if postselection:
        mitigation_kwargs["sample_error_mitigation"] = sample_error_mitigation_func(error_correction_settings,
                                 nocc=nocc,
                                 nocc_1=nocc_1,
                                 nocc_2=nocc_2)

    if error_correction_settings["tflo"]:
        mitigation_kwargs["tflo_points"] = error_correction_settings.get("tflo_points")
        mitigation_kwargs["tflo_exact_energies"] = error_correction_settings.get("tflo_exact_energies")


    ## INITIAL STATE SETTINGS

    initial_prog_instructions =  initial_state_diff_spin_types(nh, nv, t, nocc_1,
                                                  nocc_2, remap=qubits,
                                                  mapping=initial_prog_mapping)
    initial_prog = cirq.Circuit(initial_prog_instructions())
    initial_prog = optimize_in_layers(initial_prog, **{})
    temp_prog = cirq.Circuit(initial_prog_instructions())
    experiment_metadata["initial_state_circuit"] = cirq.to_json(temp_prog.moments)
    experiment_metadata["initial_state_circuit_text"] = moment_diagram(temp_prog)

    tock = time.perf_counter()
    print(f"---Starting to do calibration: delta {tock-tick}")
    tick = time.perf_counter()

    ## MEASUREMENT AND CIRCUIT SETTINGS
    measurement_set = []
    experiment_metadata["measurement_circuit"] = {}
    
    # check onsite is the first element of the dictionary!
    if "onsite0" not in measurements:
        print("Onsite type has to be in measurements")

    measurement_type = "onsite0"
    measurement = measurements[measurement_type]
    ansatz_compiled = ansatz_multilayer_circuit(chosen_ansatz, params, qubits)
    reqs_circ = create_executable(cirq.Circuit(),
                             ansatz_compiled,
                             params,
                             prep = cirq.Circuit(),
                             qubits = qubits,
                             remap = qubits,
                             optimizer = optimizer,
                             optimizer_kwargs = optimizer_kwargs)
    reqs_circ2 = create_executable(initial_prog,
                             ansatz_compiled,
                             params,
                             prep = cirq.Circuit(),
                             qubits = qubits,
                             remap = qubits,
                             optimizer = optimizer,
                             optimizer_kwargs = optimizer_kwargs)

    tock = time.perf_counter()
    print(f"---Starting to set up measurements: delta {tock-tick}")
    tick = time.perf_counter()
        
    initial_prog = cirq.Circuit(initial_prog_instructions())
    initial_prog = optimize_in_layers(initial_prog, **{})
    
    # In some cases, we need to merge the final measurement transformation
    # with a previous layer of hopping terms.
    if nh == 2 and nv == 1:
        special_measurement_type = "horiz0"
        measurement_ansatz = hopping_measurement_ansatz(qubits, nh, nv, 0)
    if nh > 2 and nv == 1:
        special_measurement_type = "horiz1"
        measurement_ansatz = hopping_measurement_ansatz(qubits, nh, nv, 1)
    if nv >= 2:
        special_measurement_type = "vert02"
        measurement_ansatz = hopping_measurement_ansatz(qubits, nh, nv, 1)

    for (measurement_type, measurement) in measurements.items():
        if measurement.prep:
            if measurement_type == special_measurement_type:
                measurement_prog = measurement_ansatz
                measurement_prog_instructions = lambda *args: []
            else:
                measurement_prog_instructions = measurement.prep(measurement.pairs, qubits)
                measurement_prog = cirq.Circuit(measurement_prog_instructions())
                measurement_prog = optimize_in_layers(measurement_prog, **{})
        else:
            measurement_prog = None
        measurement_set.append(Circuits(qc,
                                        initial_prog,
                                        measurement_prog,
                                        chosen_ansatz,
                                        measurement_type,
                                        analysis_fns(measurement.analysis,
                                                     measurement.pairs,
                                                     nh, nv,
                                                     t=t, U=U)
                                        ))
        if measurement.prep:
            temp_prog = cirq.Circuit(measurement_prog_instructions())
            if __debug__:
                module_logger.debug(f"{measurement_type}:\n{temp_prog}")
            experiment_metadata["measurement_circuit"][measurement_type] = cirq.to_json(temp_prog.moments)
            experiment_metadata["measurement_circuit_text"] = moment_diagram(temp_prog)
        else:
            experiment_metadata["measurement_circuit"][measurement_type] = ""
            experiment_metadata["measurement_circuit_text"] = ""
        if __debug__ and measurement_type != special_measurement_type:
            module_logger.debug(f"Executables, measurement {measurement_type}:")
            ansatz_compiled = ansatz_multilayer_circuit(chosen_ansatz, params, qubits)
            circ = create_executable(initial_prog,
                                     ansatz_compiled,
                                     params,
                                     prep = measurement_prog,
                                     qubits = qubits,
                                     remap = qubits,
                                     optimizer = optimizer,
                                     optimizer_kwargs = optimizer_kwargs,
                                     print_gates=True)
            module_logger.debug(moment_diagram(circ))

    ## EXTRA ARGUMENTS FOR OPTIMIZATION
    # saving samples...
    samples_filename = ""
    if save_samples:
        samples_filename = data_struct.samples_filename
    tflo_samples_filename = ""
    if save_tflo_samples:
        tflo_samples_filename = data_struct.tflo_samples_filename
    # modified SPSA
    if opt_algo == "mod_spsa":
        measurement_sets = [measurement_set]*len(num_trials_setting)
        opt_details = zip(num_trials_setting, grad_evals_setting,
                          max_evals_setting, measurement_sets)
        kwargs["spsa_details"] = opt_details
        kwargs["spsa_kwargs"] = dict(split=split, spsa_args=opt_args, repetitions=repetitions,
                                     save_file=True, save_filename=data_struct.processed_filename,
                                     num_layers=len(params), num_params=len(params[0]))
        kwargs["save_file"] = True
        kwargs["save_filename"] = data_struct.processed_filename
        args = (qubits, measurement_set, 10000, 1,
                d, num_params, optimizer, optimizer_kwargs,
                mitigation_kwargs, run_args, save_samples,
                samples_filename)
    # standard SPSA
    if opt_algo == "spsa":
        opt_details = measurement_set
        kwargs["grad_evals"] = grad_evals_setting[0]
        kwargs["max_evals"] = max_evals_setting[0]
        kwargs["opt_args"] = opt_args
        kwargs["save_file"] = True
        kwargs["save_filename"] = data_struct.processed_filename
        kwargs["num_layers"] = len(params)
        kwargs["num_params"] = len(params[0])
        args = (qubits, measurement_set, 10000, 1,
                d, num_params, optimizer, optimizer_kwargs,
                mitigation_kwargs, run_args, save_samples,
                samples_filename)
    # mgd and bayes_mgd
    if opt_algo == "mgd" or opt_algo == "bayes_mgd":
        kwargs["max_evals"] = max_evals_setting[0]
        kwargs["split"] = split
        kwargs["save_file"] = True
        kwargs["save_filename"] = data_struct.processed_filename
        kwargs["opt_args"] = opt_args
        args = (qubits, measurement_set, num_trials_setting[0], 1,
                d, num_params, optimizer, optimizer_kwargs,
                mitigation_kwargs, run_args, save_samples,
                samples_filename)
        params = [param for subparams in params for param in subparams]

    # no optimization
    if opt_algo == "none":
        kwargs["nmeas"] = num_trials_setting[0]
        kwargs["split"] = split
        kwargs["opt_args"] = opt_args
        args = (qubits, measurement_set, num_trials_setting[0], 1,
                d, num_params, optimizer, optimizer_kwargs,
                mitigation_kwargs, run_args, save_samples,
                samples_filename)
        kwargs["save_file"] = True
        kwargs["save_filename"] = data_struct.processed_filename
    if opt_algo not in ["none", "mod_spsa", "spsa", "mgd", "bayes_mgd"]:
        args = ((1, [(1,0)]), qubits, measurement_set, 10000, 1,
                d, num_params, optimizer, optimizer_kwargs,
                mitigation_kwargs, run_args, save_samples,
                samples_filename)
        params = [param for subparams in params for param in subparams]
        kwargs["maxiter"] = 100

    # RUN EXPERIMENTS
    with open(data_struct.jobs_filename, 'w') as outfile:
        json.dump(job_desc, outfile, indent=4)
        
    tock = time.perf_counter()
    print(f"---Starting to run VQE: delta {tock-tick}")
    tick = time.perf_counter()

    start_time = time.strftime("%Y%m%d-%H%M%S")
    experiment_metadata["optimization_start_time"] = start_time
    result = scipy.optimize.minimize(retrieve_objective, params, method=NAMED_OPT[opt_algo],
                                     options=kwargs, args=args)
    theta = result.x
    min_energy = result.fun
    end_time = time.strftime("%Y%m%d-%H%M%S")
    experiment_metadata["optimization_end_time"] = end_time
    print(f"***{min_energy} {time.time()}")

    tock = time.perf_counter()
    print(f"---VQE complete: delta {tock-tick}")
    tick = time.perf_counter()
    
    # Mitigate errors by training with fermionic linear optics,
    # unless we're running with exact measurements anyway.
    if error_correction_settings["tflo"] and not exact:
        # Prepare exact analysis functions.
        exact_measurement_set = []
        for (measurement_type, measurement) in measurements.items():
            if measurement.prep:
                measurement_prog = measurement.prep(measurement.pairs, qubits)
            else:
                measurement_prog = None
            exact_measurement_set.append(Circuits(qc,
                                                  initial_prog,
                                                  measurement_prog,
                                                  chosen_ansatz,
                                                  measurement_type,
                                                  analyze_exact(measurement.analysis,
                                                                measurement.pairs,
                                                                nh, nv,
                                                                U=U)
                                                  ))
        if opt_args is not None and "single" in opt_args:
            given_params = opt_args["single"]
        elif opt_args is not None and "single_flo" in opt_args:
            given_params = opt_args["single_flo"]
        else:
            given_params = theta
            
        if opt_algo == "mgd" or opt_algo == "bayes_mgd":
            given_params = [list(given_params)]
            extract_values_func = extract_values_mgd
        else:
            extract_values_func = extract_values
            
        mitigate_by_tflo(qubits, given_params, measurement_set,
                                exact_measurement_set,
                                num_layers=d,
                                num_params=num_params,
                                optimizer=optimizer,
                                optimizer_kwargs=optimizer_kwargs,
                                mitigation_kwargs=mitigation_kwargs,
                                run_args=tflo_run_args,
                                min_energy=min_energy,
                                n=nh*nv,
                                U=U,
                                filename=data_struct.tflo_filename,
                                extract_values_func=extract_values_func,
                                save_tflo_samples=save_tflo_samples,
                                tflo_samples_filename=tflo_samples_filename)



    with open(data_struct.metadata_filename, 'w') as outfile:
        json.dump(experiment_metadata, outfile, cls=NumpyEncoder, indent=4)
    with open(data_struct.ec_filename, 'w') as outfile:
        json.dump(error_correction_metadata, outfile, cls=NumpyEncoder, indent=4)

    print(f"Data collected can be found at {data_struct.experiment_folder}")
    return theta, min_energy, data_struct.experiment_folder


def main(args):
    print(cirq.__version__)
    settings = {}
    on_chip = True
    exact = False
    if args.simulation:
        on_chip = False
    if args.exact:
        on_chip = False
        exact = True

    if args.config:
        config_files = args.config
        for config_file in config_files:
            with open(config_file) as json_file:
                settings = json.load(json_file)
                settings.update({"settings_file": args.config})
            if args.qubits:
                print(f"Using qubits {args.qubits}")
                settings.update({"qubits": args.qubits})
            if args.nocc:
                print(f"Using occupation number {args.nocc}")
                settings.update({"nocc": int(args.nocc)})
                settings.update({"nocc_1": None})
                settings.update({"nocc_2": None})
    theta, min_energy, _ = vqe_experiment(on_chip=on_chip,
                                       exact=exact,
                                       **settings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vqe with given configuration file")
    parser.add_argument("-f", "--config", help="Configuration file", action="append")
    parser.add_argument("-s", "--simulation", help="Run simulated version, overwrites config settings",
                    action="store_true")
    parser.add_argument("-e", "--exact", help="Run exact version, overwrites config settings",
                    action="store_true")
    parser.add_argument("-q", "--qubits", help="Use a particular choice of qubits")
    parser.add_argument("-n", "--nocc", help="Use a specified occupation number")
    args = parser.parse_args()
    main(args)
