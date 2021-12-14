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
import time
from itertools import cycle
from typing import Callable, List
from enum import Enum
import copy

import numpy as np
from scipy.optimize import OptimizeResult
import uncertainties.unumpy as unp

from fhvqe.processing import NumpyEncoder
import fhvqe.experiment
from fhvqe.tools import (linear_least_squares,
                         quadratic_model,
                         quadratic_model_gradient,
                         quadratic_model_function,
                         randball,
                        )

module_logger = logging.getLogger("fhvqe.optimization")

SpsaSettings = collections.namedtuple("SpsaSettings", "tol alpha gamma a c A")
default_spsa_settings = SpsaSettings(1e-6, 0.602, 0.101, 0.2, 0.15, 1)

# Whether to evaluate energies at parameters and their negations, for error mitigation.
SYMMETRISE = True

def split_into_chunks(data, length):
    """Helper function to split data for when there's too much to send to the
       chip in one go."""
    return [data[x:x+length] for x in range(0, len(data), length)]

def spsa(fun, x0,
         args=(),
         repetitions=1,
         grad_evals=1, max_evals=100,
         prevk=0, split=True,
         opt_args=None,
         save_samples_on_last_iteration=False,
         save_file=False,
         save_filename="temp.json",
         num_layers = 0,
         num_params = 0,
         **kwargs):
    """SPSA algorithm.

    Args:
        fun -- function to be optimized on
        x0 -- initial parameters
        args -- objective function arguments
        grad_evals -- number of gradient evaluations for the numerical gradient (default 1)
        max_evals -- maximum number of energy evaluations (default 100)
        prevk -- value of k to which a previous SPSA evaluation reached
        split -- whether the circuit are being split or running as a batch (default True)
        save_samples_on_last_iteration -- whether to write the last measurement results to file
        opt_args -- spsa hyperparameter values
        kwargs -- other arguments, possibly from scipy minimize interface

    Returns:
        The minimum value reached, the parameters giving the minimum value
        and the dictionary detailing results
    """
    def _flatten_theta(theta):
        return [param for subparams in theta for param in subparams]
    theta = x0
    # spsa hyperparameters
    if opt_args is None:
        spsa_hyperparameters = default_spsa_settings
    else:
        spsa_hyperparameters = SpsaSettings(opt_args)

    values = []
    values_dict = []
    thetas = []

    maxIter = max_evals // (3 * grad_evals)
    
    for k in range(prevk, prevk + maxIter):
        ak = spsa_hyperparameters.a / pow(k + 1 + spsa_hyperparameters.A, spsa_hyperparameters.alpha)
        ck = spsa_hyperparameters.c / pow(k + 1, spsa_hyperparameters.gamma)

        grad = np.zeros((num_layers, num_params))
        min_energy = 0

        if split:
            for i in range(grad_evals):
                # generate  random perturbation vector
                delta = 2 * np.random.randint(2, size=(num_layers, num_params)) - 1

                # estimate the gradient
                theta += ck * delta
                x = _flatten_theta(theta)
                description = (1,
                               [(i, +1)])
                f_plus = fun(x, description, *args)
                values_dict.append(fhvqe.experiment.measured_values)
                theta -= 2 * ck * delta
                x = _flatten_theta(theta)
                description = (1,
                               [(i, -1)])
                f_minus = fun(x, description, *args)
                values_dict.append(fhvqe.experiment.measured_values)

                theta += ck * delta
                x = _flatten_theta(theta)
                description = (1,
                               [(i, 0)])
                energy = fun(x, description, *args)
                values_dict.append(fhvqe.experiment.measured_values)
                grad += (f_plus - f_minus) / (2 * ck * delta)
                min_energy += energy

            min_energy /= grad_evals
            grad = grad / grad_evals
        else:
            kwargs["save_samples"] = False
            if k == prevk + maxIter - 1 and save_samples_on_last_iteration:
                kwargs["save_samples"] = True
            
            deltas = []
            current_thetas = []
            descriptions = []
            if SYMMETRISE:
                for i in range(grad_evals):
                    delta = 2 * np.random.randint(2, size=(num_layers, num_params)) - 1
                    deltas.append(delta)
                    for j in range(repetitions):
                        for sgn in [+1, -1, 0]:
                            current_thetas += _flatten_theta(theta + sgn * ck * delta)
                        descriptions += [(2*i, sgn) for sgn in [+1, -1, 0]]

                    deltas.append(delta)
                    for j in range(repetitions):
                        for sgn in [+1, -1, 0]:
                            current_thetas += _flatten_theta(-theta - sgn * ck * delta)
                        descriptions += [(2*i+1, sgn) for sgn in [+1, -1, 0]]
            else:
                for i in range(grad_evals):
                    delta = 2 * np.random.randint(2, size=(num_layers, num_params)) - 1
                    deltas.append(delta)
                    for j in range(repetitions):
                        for sgn in [+1, -1, 0]:
                            current_thetas += _flatten_theta(theta + sgn * ck * delta)
                        descriptions += [(i, sgn) for sgn in [+1, -1, 0]]

            x = current_thetas
            if SYMMETRISE:
                description = (grad_evals * 6 * repetitions, descriptions)
            else:
                description = (grad_evals * 3 * repetitions, descriptions)
            E = fun(x, description, *args)
            if SYMMETRISE:
                for i in range(2 * grad_evals):
                    for sgn in [+1, -1, 0]:
                        fhvqe.experiment.measured_values["all_E"][i][sgn] /= repetitions
            else:
                for i in range(grad_evals):
                    for sgn in [+1, -1, 0]:
                        fhvqe.experiment.measured_values["all_E"][i][sgn] /= repetitions
            dE = np.mean((E[0::3] - E[1::3]).reshape(-1,repetitions), axis=1)
            min_energy = np.mean(E[2::3])
            print(f"Measured energies {E[2::3]}")
            fhvqe.experiment.measured_values["E"] = min_energy
            grad = np.mean((1 / (2 * ck)) * dE[:, np.newaxis, np.newaxis] / deltas, axis=0)
            values_dict.append(fhvqe.experiment.measured_values)
            m_d = fhvqe.experiment.measured_values
        if grad is not None:
            # In principle, this should be an approx bound on the maximum
            # possible difference between evals, but it seems too loose in practice.
            #max_possible_diff = ck * (len(qubits) // 2) * np.sum(np.abs(delta))
            if split:
                avg_diff = np.mean((2 * ck * delta) * grad)
            else:
                avg_diff = m_d["all_E"][0][1] - m_d["all_E"][0][-1]
            #print(f'Avg diff: {avg_diff}')
            # If the difference between evaluations is too big, there has been an
            # error in at least one of them, so ignore this update. Use 2 as an
            # arbitrary cutoff.
            if np.abs(avg_diff) > 4.0:
                print(f"Not updating params: avg difference {avg_diff}")
            else:
                theta -= ak * grad
        values.append(min_energy)
        thetas.append(np.copy(theta))

        print(k, ": current: ", min_energy,## " exact: {:.3f}".format(true_energy),
                 " params: ", theta, " grad: ", grad, " ak: ", ak)
#        if (k % 10 == 9):
#            end_time = time.perf_counter()
#            avg_time = (end_time - start_time) / 10
#            print(f'Average time over 10 iterations with {nmeas} shots: {avg_time} s')
#            start_time = end_time
        if (k % 10 == 1):
            with open("temp.json", 'w') as outfile:
                json.dump(values, outfile, cls=NumpyEncoder)

    if save_file:
        with open(save_filename, "a") as outfile:
            json.dump([theta, min_energy, values_dict, values, thetas, k],
                      outfile,
                      cls=NumpyEncoder,
                      indent=4)
    return OptimizeResult(fun=min_energy, x=theta, success=True, nit=k)


def mod_spsa(fun, x0, spsa_details=(), spsa_kwargs={},
             prevk=0, save_file=False, save_filename="temp.json", **kwargs):
    """Modified SPSA algorithm.

    Args:
        fun -- function to be optimized on
        x0 -- initial parameters
        args -- objective function arguments
        spsa_details -- tuple containing details on how to run spsa
            (number of measurements, number of gradient evaluations for the
             numerical gradient, maximum number of energy evaluations,
             measurement_set)
        spsa_kwargs -- various arguments relating to SPSA function
        prevk -- value of k to which a previous SPSA evaluation reached
        kwargs -- other arguments, possibly from scipy minimize interface

    Returns:
        The minimum value reached, the parameters giving the minimum value
        and the dictionary detailing results
    """
    theta = x0
    results = []
    k = 0
    spsa_details_list = list(spsa_details)
    if save_file:
        with open(save_filename, "w") as outfile:
            outfile.write("[")
    for idx, (nmeas, grad_evals, max_evals, measurement_set) in enumerate(spsa_details_list):
        save_samples_on_last_iteration = False
        if "samples_filename" in kwargs and idx == len(spsa_details_list) - 1:
            save_samples_on_last_iteration = True

        if "args" in kwargs:
            kwargs["args"] = kwargs["args"][:2] + (nmeas,) + kwargs["args"][3:]

        res = spsa(fun, theta, grad_evals=grad_evals, max_evals=max_evals,
                   prevk=k, save_samples_on_last_iteration=save_samples_on_last_iteration,
                   **kwargs, **spsa_kwargs)
        results.append(res)
        theta = res.x
        k = res.nit + 1
        if save_file and not idx == len(spsa_details_list) - 1:
            with open(save_filename, "a") as outfile:
                outfile.write(",\n")
    if save_file:
        with open(save_filename, "a") as outfile:
            outfile.write("]")
    return res


def no_optimization(objective, x0,
                    args=(),
                    split=True, opt_args=None,
                    save_file=False, save_filename="temp.json",
                    **kwargs):
    """Allows for single or parameter run of objective function.

    Args:
        objective -- function to be optimized on
        x0 -- initial parameters
        args -- objective function arguments
        split -- whether the circuit are being split or running as a batch (default True)
        opt_args -- optimizer arguments
        kwargs -- other arguments, possibly from scipy minimize interface

    Returns:
        The minimum value reached, the parameters giving the minimum value
        and the dictionary detailing results
    """
    theta=x0
    given_params = []
    parameters = []

    num_params = len(theta[0])
    num_layers = len(theta)
    
    print(f"No optimisation, save file is {save_filename}")

    if "parameters" in opt_args:
        parameter_files = opt_args["parameters"]
        for pfile in parameter_files:
            print(pfile)
            with open(pfile) as jfile:
                loaded_params = json.load(jfile)
            given_params.append([[params] for params in loaded_params])

    if "grid" in opt_args:
        num_layers = 1 # only supported for single layer atm
        min_val, max_val, step = tuple(opt_args["grid"])
        param_grid = np.mgrid[min_val:max_val:step]
        all_param_grid_lst = [param_grid] * num_params
        all_param_grid = np.meshgrid(*all_param_grid_lst)
        for i, grid in enumerate(all_param_grid):
            all_param_grid[i] = grid.flatten()
        parameters = [[list(el)] for el in zip(*all_param_grid)]
        given_params.append([params for params in parameters])


    if "single" in opt_args:
        neg_params = [[-x for x in y] for y in opt_args["single"]]
        given_params = [[ [opt_args["single"]], [neg_params]  ]]
        print(f"Given params: {given_params}")
        
    if "single_flo" in opt_args:
        flo_params = copy.deepcopy(opt_args["single_flo"])
        neg_params = [[-x for x in y] for y in opt_args["single_flo"]]
        neg_flo_params = [[-x for x in y] for y in opt_args["single_flo"]]
        for i in range(len(flo_params)):
            flo_params[i][0] = 0
            neg_flo_params[i][0] = 0
        given_params = [[ [opt_args["single_flo"]], [neg_params], [flo_params], [neg_flo_params]  ]]
        print(f"Given params: {given_params}")

    if "repeated_single" in opt_args:
        repetition = opt_args["repeated_single"]
        given_params = [[opt_args["single"]] * repetition]

    values = []
    values_dict = []
    thetas = []

    if split:
        #TODO: needs fixing...
        for parameters in given_params:
            for params in parameters:
                description = (1, [(1,0)])
                E = objective(params, description, *args)
                min_energy = np.mean(E)
                grad = None
                fhvqe.experiment.measured_values["E"] = min_energy
                fhvqe.experiment.measured_values["E_list"] = E
                values_dict.append(fhvqe.experiment.measured_values)
                values.append(min_energy)
                thetas.append(np.copy(params))
                print(f"{params}: {min_energy}")
    else:
        thetas = []
        for batched_params in given_params:
            description = (len(batched_params),
                           [(i,0) for i in range(len(batched_params))])
            E = objective(batched_params, description, *args)
            if "single_flo" in opt_args:
                min_energy = np.mean(E[:2])
            elif "parameters" in opt_args:
                min_energy = E
            else:
                min_energy = np.mean(E)
            grad = None
            fhvqe.experiment.measured_values["E"] = min_energy
            fhvqe.experiment.measured_values["E_list"] = E
            values_dict.append(fhvqe.experiment.measured_values)
            if isinstance(min_energy, (list, np.ndarray)):
                values += list(min_energy)
            else:
                values.append(min_energy)
            thetas += batched_params
            str_energies = " ".join([str(e) for e in E])
            print(f"==={str_energies} {time.time()}")

    print(f"Computed energies {values}, params {thetas}")
    if save_file:
        with open(save_filename, "w") as outfile:
            json.dump([[theta, min_energy, values_dict, values, thetas, 0]],
                      outfile,
                      cls=NumpyEncoder,
                      indent=4)
    return OptimizeResult(fun=min_energy, x=theta, success=True, nit=0)


def exact_parameters(objective, x0,
                    args=(),
                    split=True, opt_args=None,
                    save_file=False, save_filename="temp.json",
                    **kwargs):
    print(f"Exact params, args {args}, kwargs {kwargs}")
    

class MGDStatus(Enum):
    NOT_DONE = 0
    MAXEVAL_REACHED = 1
    MAXITER_REACHED = 2
    YTOL_REACHED = 3
    XTOL_REACHED = 4


def mgd(fun: Callable, x0: List,
        args=(),
        max_evals=1000, max_iter=30, y_tol=1e-4, x_tol=1e-4,
        opt_args=None,
        save_file=False, save_filename="temp.json",
        split = True,
        **kwargs):
    """MGD algorithm
    
    Args:
        fun -- The objective function to minimize. Assumed to return a `ufloat`.
        x0 -- Initial parameters
        args -- additional objectvive function arguments
        max_evals -- Maximum number of function evaluations
        max_iter -- Maximum number of MGD iterations
        y_tol -- y-tolerance to stop optimization. Defaults to 1e-5
        x_tol -- x-tolerance to stop optimization. Defaults to 1e-5
        opt_args -- The MGD hyper parameters. Default values are 
                    {"alpha": 0.602, "gamma": 0.3, "A": 100,
                     "delta": 0.2, "xi": 0.0673,
                     "eta": 0.6, "nshots": 100}
        kwargs -- Other arguments, possibly from the scipy minimize interface
    
    Returns:
        An `OptimizeResult` like for `scipy.optimize.minimize`.
    """
    # clever python hacking to merge opt_args with the default args
    if opt_args is None: opt_args = {}
    default_args = {"alpha": 0.602, "gamma": 0.3, "A": 100,
                    "delta": 0.2, "xi": 0.0673,
                    "eta": 0.6, "nshots": 100}
    opt_args = {**default_args, **opt_args}   # latter takes precedence
    alpha, gamma, A = opt_args["alpha"], opt_args["gamma"], opt_args["A"]
    delta, xi = opt_args["delta"], opt_args["xi"]
    eta, nshots = opt_args["eta"], opt_args["nshots"]
    
    # setup
    n = len(x0)
    dim_fit_params = (n+1) * (n+2) // 2
    fit_params = np.zeros(dim_fit_params)
    fit_params_inv_cov = np.full(dim_fit_params, 1e-7)
    fit_params_inv_cov[-n-2:-1] = 1e-5
    fit_params_inv_cov = np.diag(fit_params_inv_cov)
    n_evals = np.ceil(eta * dim_fit_params).astype("int")
    
    print(f"Running MGD: n_evals {n_evals}, nshots {nshots}, max_iter {max_iter}, max_evals {max_evals}")
    
    # these lists are needed for logging
    values_dict = []
    values = []
    thetas = []
    x, y = np.copy(x0), fun(x0, (1, [(1,0)]), *args)
    xmin, ymin = np.copy(x), np.copy(y)
    grad = np.zeros(n)
    fhvqe.experiment.measured_values["E"] = unp.nominal_values(y)
    values_dict.append(fhvqe.experiment.measured_values)
    # Assuming that y is actually only single value which will be true as long
    # as x0 wasn't a list of VQE parameters
    values.append(y[0].nominal_value)
    thetas.append(np.copy(x))
    
    # these are the arrays that mgd logs to
    Lx = np.reshape(x, (n, 1))
    Ly = np.array([y])
    Lx_near = np.reshape(x, (n, 1))
    Ly_near = np.array([y])
    
    # main loop
    iteration = 1
    const_iterations_limit = 2*dim_fit_params
    status = MGDStatus.NOT_DONE
    while status == MGDStatus.NOT_DONE:
        delta_k = delta / iteration ** xi  # sample radius
        gamma_k = gamma / (iteration + A) ** alpha # gradient descent step length
        
        # split == true disables circuit batching
        if split:
            for _ in range(n_evals):
                xk = x + delta_k * randball(n)
                yk = fun(xk, (1, [(1,0)]), *args)
                fhvqe.experiment.measured_values["E"] = unp.nominal_values(yk)
                values_dict.append(fhvqe.experiment.measured_values)
                Lx = np.column_stack([Lx, xk])
                Ly = np.append(Ly, yk)
                if SYMMETRISE: # evaluate at `-xk`, but pretend it is at `xk`
                    yk = fun(-xk, (1, [(1,-1)]), *args)
                    fhvqe.experiment.measured_values["E"] = unp.nominal_values(yk)
                    values_dict.append(fhvqe.experiment.measured_values)
                    Lx = np.column_stack([Lx, xk])
                    Ly = np.append(Ly, yk)
        else:
            descriptions = []
            current_thetas = []
            for k in range(n_evals):
                xk = x + delta_k * randball(n)
                Lx = np.column_stack([Lx, xk])
                current_thetas += list(xk)
                descriptions += [(k, 0)]
                if SYMMETRISE:
                    Lx = np.column_stack([Lx, xk]) # this is used for the fit
                    current_thetas += list(-xk)    # this is used for evaluation
                    descriptions += [(k, -1)]
            all_y = fun(current_thetas, ((1+SYMMETRISE) * n_evals, descriptions), *args)
            fhvqe.experiment.measured_values["E"] = unp.nominal_values(all_y)
            Ly = np.append(Ly, all_y)
            values_dict.append(fhvqe.experiment.measured_values)

        # collecting all data in the delta_k ball around x
        Lx_near = np.resize(Lx_near, (n, 0))   
        Ly_near = np.resize(Ly_near, (0,))    
        for (xk, yk) in zip(Lx.T, Ly):
            if np.linalg.norm(xk - x) < delta_k:
                Lx_near = np.column_stack([Lx_near, xk])
                Ly_near = np.append(Ly_near, yk)
                
        # bayes update of fit parameters
        fit_params, fit_params_inv_cov = linear_least_squares(
                                            quadratic_model,
                                            Lx_near,
                                            Ly_near,
                                            dim_fit_params)
        grad = quadratic_model_gradient(fit_params, x)
        x[:] -= gamma_k * grad[:]
        y = quadratic_model_function(fit_params, x, np.linalg.inv(fit_params_inv_cov))
        values.append(y.nominal_value)
        thetas.append(np.copy(x))
        
        # check if we converged or reached maxiter
        iteration += 1
        if iteration >= max_iter:
            print("Max iters reached")
            status = MGDStatus.MAXITER_REACHED
        if len(Ly) >= max_evals:
            print("Max evals reached")
            status = MGDStatus.MAXEVAL_REACHED
        if gamma_k * np.linalg.norm(grad) < x_tol:
            print("Tolerance reached")
            status = MGDStatus.XTOL_REACHED
        if y < ymin:
            ymin = y
            xmin[:] = x[:]
            const_iterations = 0
        elif y - ymin < y_tol:
            print(f"{const_iterations}")
            const_iterations += 1
            if const_iterations > const_iterations_limit:
                status = MGDStatus.YTOL_REACHED
        else:
            const_iterations = 0

        # more logging and debugging
        with np.printoptions(precision=3, suppress=True):
            if __debug__:
                updated_args = list(args)
                updated_args[2] = 100000   # use more shots to compute the final energy
                y_true = fun(x, (1, [(1,0)]), *updated_args)
                if SYMMETRISE:
                    y_true = 0.5 * (y_true + fun(-x, (1, [(1,-1)]), *updated_args))
                fhvqe.experiment.measured_values["E"] = unp.nominal_values(y_true)
                values_dict.append(fhvqe.experiment.measured_values)
                # Assuming that y_true is actually only single value which will be true as long
                # as x wasn't a list of VQE parameters
                values.append(y_true[0].nominal_value)
                thetas.append(np.copy(x))
                print(iteration, " :--------------------------------")
                print("fit_params: ", fit_params)
                print("current: {:.3f}".format(y),
                        " true: {:.3f}".format(y_true[0]),
                      " params: ", x,
                      " grad: ", grad,
                      " gamma_k: {:.4f}".format(gamma_k),
                      " delta_k: {:.4f}".format(delta_k))
            else: 
                # delete from here
                updated_args = list(args)
                updated_args[2] = 100000   # use more shots to compute the final energy
                y_true = fun(x, (1, [(1,0)]), *updated_args)
                if SYMMETRISE:
                    y_true = 0.5 * (y_true + fun(-x, (1, [(1,-1)]), *updated_args))
                fhvqe.experiment.measured_values["E"] = unp.nominal_values(y_true)
                values_dict.append(fhvqe.experiment.measured_values)
                # Assuming that y_true is actually only single value which will be true as long
                # as x wasn't a list of VQE parameters
                values.append(y_true[0].nominal_value)
                thetas.append(np.copy(x))
                # until here
                print(iteration, " :--------------------------------")
                print("current: {:.3f}".format(y),
                      " true: {:.3f}".format(y_true[0]), # delete this too
                      " params: ", x,
                      " grad: ", grad,
                      " gamma_k: {:.4f}".format(gamma_k),
                      " delta_k: {:.4f}".format(delta_k))
            
    if save_file:
        with open(save_filename, "a") as outfile:
            json.dump([[x, y.nominal_value, values_dict, values, thetas, iteration]], #unp.nominal_values(Ly), Lx.T, iteration]],
                      outfile,
                      cls=NumpyEncoder,
                      indent=4)
    
    return OptimizeResult(fun=y, jac=grad, x=x,
                          message=status.name,
                          nfev=iteration * n_evals, nit=iteration)

def bayes_mgd(fun: Callable, x0: List,
              args=(),
              max_evals=1000, max_iter=50, y_tol=1e-4, x_tol=1e-4,
              opt_args={},
              save_file=False, save_filename="temp.json",
              split = True,
              **kwargs):
    """BayesMGD algorithm
    
    Args:
        fun -- The objective function to minimize. Assumed to return a `ufloat`.
        x0 -- Initial parameters
        args -- additional objectvive function arguments
        max_evals -- Maximum number of function evaluations
        max_iter -- Maximum number of MGD iterations
        y_tol -- y-tolerance to stop optimization. Defaults to 1e-5
        x_tol -- x-tolerance to stop optimization. Defaults to 1e-5
        opt_args -- The MGD hyper parameters. Default values are 
                    {"alpha": 0.602, "gamma": 0.3, "A": 100,
                     "delta": 0.2, "xi": 0.0673,
                     "eta": 0.6, "l0": 1., "nshots": 100}
        kwargs -- Other arguments, possibly from the scipy minimize interface
    
    Returns:
        An `OptimizeResult` dict containing the minimum value reached, the
        parameters giving the minimum value and some more stuff.
    """
    # setting hyperparameters
    if opt_args is None: opt_args = {}
    default_args = {"alpha": 0.602, "gamma": 0.3, "A": 100,
                    "delta": 0.2, "xi": 0.0673,
                    "eta": 0.6, "l0": 1., "nshots": 100}
    opt_args = {**default_args, **opt_args}   # latter takes precedence
    alpha, gamma, A = opt_args["alpha"], opt_args["gamma"], opt_args["A"]
    delta, xi = opt_args["delta"], opt_args["xi"]
    eta, l0, nshots = opt_args["eta"], opt_args["l0"], opt_args["nshots"]
    
    # setup
    n = len(x0)
    dim_fit_params = (n+1) * (n+2) // 2
    fit_params = np.zeros(dim_fit_params)
    fit_params_inv_cov = np.full(dim_fit_params, 1e-7)
    fit_params_inv_cov[-n-2:-1] = 1e-5
    fit_params_inv_cov = np.diag(fit_params_inv_cov)
    n_evals = np.ceil(eta * dim_fit_params).astype("int")
    
    print(f"Running BayesMGD: n_evals {n_evals}, nshots {nshots}, max_iter {max_iter}, max_evals {max_evals}")
    
    # setup logging lists to save all data later to jsons
    values_dict = []
    values = []
    thetas = []

    x, y = np.copy(x0), fun(x0, (1, [(1,0)]), *args)
    xmin, ymin = np.copy(x), np.copy(y)
    grad = np.zeros(n)
    values_dict.append(fhvqe.experiment.measured_values)
    # Assuming that y is actually only single value which will be true as long
    # as x0 wasn't a list of VQE parameters
    values.append(y[0].nominal_value)
    thetas.append(np.copy(x))
    
    # these are the actual arrays used by BayesMGD and not for logging
    Lx = np.reshape(x, (n, 1))
    Ly = np.array([y])
    
    # main loop
    iteration = 1
    const_iterations_limit = 2*dim_fit_params
    status = MGDStatus.NOT_DONE
    while status == MGDStatus.NOT_DONE:
        delta_k = delta / iteration ** xi  # sample radius
        gamma_k = gamma / (iteration + A) ** alpha # gradient descent step length
        
        # split==true disables circuit batching
        if split:
            for _ in range(n_evals):
                xk = x + delta_k * randball(n)
                yk = fun(xk, (1, [(1,0)]), *args)
                fhvqe.experiment.measured_values["E"] = unp.nominal_values(yk)
                values_dict.append(fhvqe.experiment.measured_values)
                Lx = np.column_stack([Lx, xk])
                Ly = np.append(Ly, yk)
                if SYMMETRISE:
                    yk = fun(-xk, (1, [(1,-1)]), *args)
                    fhvqe.experiment.measured_values["E"] = unp.nominal_values(yk)
                    values_dict.append(fhvqe.experiment.measured_values)
                    Lx = np.column_stack([Lx, xk])
                    Ly = np.append(Ly, yk)
        else:
            descriptions = []
            current_thetas = []
            for k in range(n_evals):
                xk = x + delta_k * randball(n)
                Lx = np.column_stack([Lx, xk])
                current_thetas += list(xk)
                descriptions += [(k, 0)]
                if SYMMETRISE:
                    Lx = np.column_stack([Lx, xk]) # this is used for the fit
                    current_thetas += list(-xk)    # this is used for evaluation
                    descriptions += [(k, -1)]   
            chunk_size = 20  # arbitrary size that is small enough to avoid timeouts
            current_thetas_split = split_into_chunks(current_thetas, chunk_size * n)
            descriptions_split = split_into_chunks(descriptions, chunk_size)
            # collect `fhvqe.experiment.measured_values` after each 
            # batch in here
            all_y = []
            for block, description in zip(current_thetas_split, descriptions_split):
                print(f"Running block of size {len(description)}...")
                returned_y = fun(block, (len(description), description), *args)
                all_y = np.append(all_y, returned_y)
                fhvqe.experiment.measured_values["E"] = unp.nominal_values(returned_y)
                values_dict.append(fhvqe.experiment.measured_values)
            Ly = np.append(Ly, all_y)
            
                
        # Bayes update of fit parameters
        fit_params_cov = np.linalg.inv(fit_params_inv_cov)
        fit_params_inv_cov[:,:] = np.linalg.inv(fit_params_cov
                                        + gamma_k * np.linalg.norm(grad) / l0
                                          * np.identity(dim_fit_params))
        fit_params[:], fit_params_inv_cov[:,:] = linear_least_squares(
                                                        quadratic_model,
                                                        Lx[:,-(1+SYMMETRISE)*n_evals-1:-1],
                                                        Ly[-(1+SYMMETRISE)*n_evals-1:-1],
                                                        dim_fit_params,
                                                        fit_params,
                                                        fit_params_inv_cov)
        # gradient descent step
        grad = quadratic_model_gradient(fit_params, x)
        x[:] -= gamma_k * grad[:]
        y = quadratic_model_function(fit_params, x, np.linalg.inv(fit_params_inv_cov))

        # logging
        values.append(y.nominal_value)
        print(values)
        thetas.append(np.copy(x))
        
        # check if we converged or reached maxiter
        iteration += 1
        if iteration >= max_iter:
            status = MGDStatus.MAXITER_REACHED
        if len(Ly) >= max_evals:
            status = MGDStatus.MAXEVAL_REACHED
        if gamma_k * np.linalg.norm(grad) < x_tol:
            status = MGDStatus.XTOL_REACHED
        if y < ymin:
            ymin = y
            xmin[:] = x[:]
            const_iterations = 0
        elif y - ymin < y_tol:
            print(f"{const_iterations}")
            const_iterations += 1
            if const_iterations > const_iterations_limit:
                status = MGDStatus.YTOL_REACHED
        else:
            const_iterations = 0

        # more logging and debugging
        with np.printoptions(precision=3, suppress=True):
            if __debug__:
                fhvqe.experiment.measured_values["E"] = y.nominal_value
                values_dict.append(fhvqe.experiment.measured_values)
                # Assuming that y_true is actually only single value which will be true as long
                # as x wasn't a list of VQE parameters
                values.append(y.nominal_value)
                thetas.append(np.copy(x))
                print(iteration, " :--------------------------------")
                print("fit_params: ", fit_params)
                print("current: {:.3f}".format(y),
                      " params: ", x,
                      " grad: ", grad,
                      " gamma_k: {:.4f}".format(gamma_k),
                      " delta_k: {:.4f}".format(delta_k))
            else: 
                # delete from here
                updated_args = list(args)
                updated_args[2] = 100000   # use more shots to compute the final energy
                y_true = fun(x, (1, [(1,0)]), *updated_args)
                if SYMMETRISE:
                    y_true = 0.5 * (y_true + fun(-x, (1, [(1,-1)]), *updated_args))
                fhvqe.experiment.measured_values["E"] = unp.nominal_values(y_true)
                values_dict.append(fhvqe.experiment.measured_values)
                # Assuming that y_true is actually only single value which will be true as long
                # as x wasn't a list of VQE parameters
                values.append(y_true[0].nominal_value)
                thetas.append(np.copy(x))
                # until here
                print(iteration, " :--------------------------------")
                print("current: {:.3f}".format(y),
                      " true: {:.3f}".format(y_true[0]), # delete this too
                      " params: ", x,
                      " grad: ", grad,
                      " gamma_k: {:.4f}".format(gamma_k),
                      " delta_k: {:.4f}".format(delta_k))
            
    if save_file:
        with open(save_filename, "a") as outfile:
            json.dump([[x, y.nominal_value, values_dict, values, thetas, iteration]],
                      outfile,
                      cls=NumpyEncoder,
                      indent=4)
    
    return OptimizeResult(fun=y, jac=grad, x=x,
                          message=status.name,
                          nfev=iteration * n_evals, nit=iteration)

NAMED_OPT = { "spsa": spsa,
              "mod_spsa": mod_spsa,
              "mgd": mgd,
              "bayes_mgd": bayes_mgd,
              "none": no_optimization,
              "exact_parameters": exact_parameters,
              "bfgs": "BFGS",
              "lbfgs": "L-BFGS-B",
              "neldermead": "Nelder-Mead",
              "powell": "Powell"
            }
