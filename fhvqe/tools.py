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

import logging
from math import floor, log10
import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat

import openfermion as of
from openfermion.ops import FermionOperator
from typing import Callable

logger = logging.getLogger("fhvqe.tools")
def fermi_hubbard_JW(nh, nv, t, *args, **kwargs):
    """Creates Fermi-Hubbard openfermion model in Jordan-Wigner encoding.

    Args:
        nh -- number of horizontal sites
        nv -- number of vertical sites
        t -- hopping interaction parameter

    Returns:
        FermionOperator Fermi-Hubbard model
    """
    def _right_neighbor(xdim, ydim, site):
        if xdim == 1: return None
        if (site+1)%xdim == 0: return None
        return site+1

    def _bottom_neighbor(xdim, ydim, site):
        if ydim == 1: return None
        row = site // xdim
        if (row + 1) == ydim: return None # bottom row
        return (row + 2) * xdim - 1 - site % xdim

    def _hopping_term(i, j, coefficient):
        hopping_term = FermionOperator(((i, 1), (j, 0)), coefficient)
        hopping_term += FermionOperator(((j, 1), (i, 0)), coefficient.conjugate())
        return hopping_term

    n_sites = nh * nv
    hubbard_model = FermionOperator()
    for site in range(n_sites):
        right_neighbor = _right_neighbor(nh, nv, site)
        bottom_neighbor = _bottom_neighbor(nh, nv, site)
        if right_neighbor is not None:
            hubbard_model += _hopping_term(site, right_neighbor, -t)
        if bottom_neighbor is not None:
            hubbard_model += _hopping_term(site, bottom_neighbor, -t)
    return hubbard_model


def fermi_hubbard_rcs(nh, nv, t, *args, **kwargs):
    """Creates Fermi-Hubbard openfermion model in site encoding.

    Args:
        nh -- number of horizontal sites
        nv -- number of vertical sites
        t -- hopping interaction parameter

    Returns:
        FermionOperator Fermi-Hubbard model
    """
    def _right_neighbor(xdim, ydim, site):
        if xdim == 1: return None
        if (site+1)%xdim == 0: return None
        return site+1

    def _bottom_neighbor(xdim, ydim, site):
        if ydim == 1: return None
        if (site // xdim) + 1 == ydim: return None
        return site + xdim

    def _hopping_term(i, j, coefficient):
        hopping_term = FermionOperator(((i, 1), (j, 0)), coefficient)
        hopping_term += FermionOperator(((j, 1), (i, 0)), coefficient.conjugate())
        return hopping_term

    n_sites = nh * nv
    hubbard_model = FermionOperator()
    for site in range(n_sites):
        right_neighbor = _right_neighbor(nh, nv, site)
        bottom_neighbor = _bottom_neighbor(nh, nv, site)
        if right_neighbor is not None:
            hubbard_model += _hopping_term(site, right_neighbor, -t)
            hubbard_model += _hopping_term(site + n_sites, right_neighbor + n_sites, -t)
        if bottom_neighbor is not None:
            hubbard_model += _hopping_term(site, bottom_neighbor, -t)
            hubbard_model += _hopping_term(site + n_sites, bottom_neighbor + n_sites, -t)
    return hubbard_model


def get_givens(nhoriz, nvert, t, noccupied, hamiltonian=fermi_hubbard_JW):
    """Function to get givens rotations from openfermion and write to file.

    Args:
        nhoriz -- number of horizontal sites
        nvert -- number of vertical sites
        t -- hopping interaction parameter
        noccupied -- total number of fermions in the system
        hamiltonian -- Fermi-Hubbard Hamiltonian depending on the encoding (default fermi_hubbard_JW)

    Returns:
        List of qubit pairs and angles of givens rotations to be applied to be applied to those pairs.
    """
    # Set up Hubbard model and determine how to prepare the slater determinant
    h = hamiltonian(nhoriz, nvert, t, 0, periodic=0)# of.fermi_hubbard(nhoriz, nvert, t, 0, periodic=0)
    h_quad = of.get_quadratic_hamiltonian(h)
    D, U, _ = h_quad.diagonalizing_bogoliubov_transform()
    Q = U[range(noccupied)]
    slater_circuit = of.slater_determinant_preparation_circuit(Q)
    return slater_circuit


###############################
##
## Mappings
##
###############################


def map_ordered_to_rcs(nh, nv, num):
    """Mapping ordered encoding to (row, column, spin-type).

    Args:
        nhoriz -- number of horizontal sites
        nvert -- number of vertical sites
        num -- qubit in ordered encoding

    Returns:
        Tuple of qubit (row, column and spin-type)
    """
    spin = num % 2 # even numbers spin 0, odd number spin 1
    row = num // (2 * nh)
    col = ((num % (2 * nh)) - spin) // 2
    return (row, col, spin)


def map_rcs_to_ordered(nh, nv, row, col, spin):
    """Mapping (row, column, spin-type) to ordered encoding.

    Args:
        nhoriz -- number of horizontal sites
        nvert -- number of vertical sites
        row -- row location of the qubit in the lattice
        col -- column location of the qubit in the lattice
        spin -- spin-type: up or down

    Returns:
        Number of qubit in ordered encoding
    """
    return 2 * nh * row + 2 * col + spin


def map_rcs_to_site(nh, nv, row, col, spin):
    """Mapping (row, column, spin-type) to site encoding.

    Args:
        nhoriz -- number of horizontal sites
        nvert -- number of vertical sites
        row -- row location of the qubit in the lattice
        col -- column location of the qubit in the lattice
        spin -- spin-type: up or down

    Returns:
        Qubit site
    """
    return nh * row + col + spin * nh * nv


def map_rcs_to_JW(nh, nv, row, col, spin):
    """Mapping (row, column, spin-type) to Jordan-Wigner encoding.

    Args:
        nhoriz -- number of horizontal sites
        nvert -- number of vertical sites
        row -- row location of the qubit in the lattice
        col -- column location of the qubit in the lattice
        spin -- spin-type: up or down

    Returns:
        Number of Jordan-Wigner encoded qubit
    """
    col_adjust = col
    if (row % 2 == 1):
        col_adjust = nh - 1 - col
    return row * nh + col_adjust + nh * nv * spin


def map_rcs_to_snake(nh, nv, row, col, spin):
    """Mapping (row, column, spin-type) to snake encoding.

    Args:
        nhoriz -- number of horizontal sites
        nvert -- number of vertical sites
        row -- row location of the qubit in the lattice
        col -- column location of the qubit in the lattice
        spin -- spin-type: up or down

    Returns:
        Number of snake encoded qubit
    """
    return 2 * nh * row + col + nh * spin


def map_JW_to_rcs(nh, nv, site):
    """Mapping Jordan-Wigner encoding to (row, column, spin-type).

    Args:
        nhoriz -- number of horizontal sites
        nvert -- number of vertical sites
        site -- qubit in jordan-wigner string

    Returns:
        Tuple with qubit (row, column and spin-type)
    """
    row = site // nh
    if (row%2 == 0):
        col = site % nh
    else:
        col = nh - (site % nh) - 1
    spin = 0
    if row > nv:
        row = row - nv
        spin = 1
    return (row, col, spin)


def map_JW_to_site(nh, nv, site):
    """Mapping Jordan-Wigner encoding to site encoding.

    Args:
        nhoriz -- number of horizontal sites
        nvert -- number of vertical sites
        site -- qubit in jordan-wigner string

    Returns:
        Tuple with qubit (row, column and spin-type)
    """
    return map_rcs_to_site(nh, nv, *map_JW_to_rcs(nh, nv, site))


def map_site_to_JW(nh, nv, site):
    row = (site // nh) % nv
    if (row%2 == 0):
        return site
    else:
        col = site % nh
        return (site // nh) * nh + (nh - col - 1)


###############################
##
## General math tools
##
###############################


def round_sig(x, sig=2):
    """Rounds x to sig signfiicant digits.
    """
    if x == 0.0: return x
    return round(x, sig-int(floor(log10(abs(x))))-1)

def randball(n: int):
    """
    Uniform, random sample in the `n`-ball.
    
    Algorithm taken from https://arxiv.org/pdf/math/0503650.pdf
    """
    x = np.random.randn(n) / np.sqrt(2)
    y = -np.log(1 - np.random.rand())
    x /= np.sqrt(y + np.dot(x, x))
    return x

# Linear least squares utilities for model gradient descent
# #########################################################

def linear_least_squares(model_function: Callable,
                         x: np.array,
                         y: unp.uarray,
                         m: int,
                         prior_parameters=None,
                         prior_certainties=None):
    """Linear least squares with prior parameters
    
    Args: 
        model_function -- A function from ℝ^dim(x) -> ℝ^m that computes all
                          components of the model function
        x -- A `k x n` matrix with the independent variables as column vectors
        y -- The dependent data / measurements as a n-dimensional vector
        m -- The number of model parameters
        prior_parameters -- The prior on the fit parameters. Defaults to all zero
        prior_certainties -- The inverse covariance matrix of the `prior_parameters`.
                             Defaults to np.diag(np.full(m, 1e-10)).
                             
    Returns:
        `(posterior_parameters, posterior_certainties)` the ideal fit parameters
        and their inverse covariance matrix.
    """
    if prior_parameters is None:
        prior_parameters = np.zeros(m)
    if prior_certainties is None:
        prior_certainties = np.diag(np.full(m, 1e-5))
    
    k, n = x.shape
    X = np.empty((n, m))
    for (i, xx) in enumerate(x.T):
        model_function(X[i,:], xx)
    
    weights = np.diag(1 / unp.std_devs(y))**2
    values = unp.nominal_values(y)
    posterior_certainties = X.T @ weights @ X + prior_certainties
    Q = np.linalg.inv(posterior_certainties)
    posterior_parameters = Q @ (X.T @ weights @ values + prior_certainties @ prior_parameters)
    return posterior_parameters, posterior_certainties


def linear_model(X, x):
    """Model function of a linear model m∙x + b"""
    n = len(x)
    X[n] = 1.
    X[0:n] = x[:]


def linear_model_gradient(params, x):
    """Gradient of a linear model with `params = [m1, ⋯ , mn, b]` at `x`"""
    return x[0:-1]


def linear_model_function(params, x, params_cov=None):
    """Function value of a linear model with `params` at `x`"""
    y = np.dot(x, params[0:-1]) + params[-1]
    if params_cov is None:
        return y
    
    params_grad = np.copy(params)
    linear_model(params_grad, x)
    std = np.sqrt(np.dot(params_grad, params_cov @ params_grad))
    return ufloat(y, std)


def quadratic_model(X, x):
    """Model function of a quadratic model x∙Q∙x + m∙x + b"""
    k = len(x)
    idx = 0
    
    X[0:k] = x[:]**2 # Fill the Q_{ii} cofficients
    idx = k
    for i in range(1, k):   # Fill the Q_{i!=j} coeffiecients
        for j in range(0, i):
            X[idx] = 2 * x[i] * x[j]
            idx += 1
            
    X[idx:idx+k] = x[:] # Fill the m_i coefficients
    X[-1] = 1.          # Fill the b coefficient


def quadratic_model_gradient(params, x):
    """Gradient of a linear model with `params = [Q_11, Q12, ⋯ , m1, ⋯ , mn, b]` at `x`"""
    n = len(x)
    grad = np.zeros(n)
    grad[:] += params[-1-n:-1]  # add m
    grad[:] += 2. * params[0:n] * x # Add 2 * Q_[ii] * x_i
    idx = n
    for i in range(1, n):
        for j in range(0, i):
            grad[i] += 2 * params[idx] * x[j]
            grad[j] += 2 * params[idx] * x[i]
            idx += 1
    
    return grad


def quadratic_model_function(params, x, params_cov=None):
    """Function value of a quadratic model with `params` at `x`"""
    n = len(x)
    y = 0.
    y += np.dot(params[0:n], x**2)  # Q_ii * x_i^2
    y += np.dot(params[-1-n:-1], x) # m_i * x_i
    y += params[-1]                 # +b
    idx = n
    for i in range(1, n):
        for j in range(0, i):
            y += 2 * params[idx] * x[j] * x[j]
            idx += 1
    
    if params_cov is None:
        return y
    
    params_grad = np.copy(params)
    quadratic_model(params_grad, x)
    std = np.sqrt(np.dot(params_grad, params_cov @ params_grad))
    return ufloat(y, std)


###############################
##
## Random other tools
##
###############################
def color_qubits_in_grid(grid, qubits):
    to_color1 = []
    to_color2 = []
    total_len = len(qubits)
    for qubit in qubits[:total_len//2]:
        to_color1.append(f"{qubit.row, qubit.col}")
    for qubit in qubits[total_len//2:]:
        to_color2.append(f"{qubit.row, qubit.col}")
    s = grid[:]
    for color_q in to_color1:
        place = s.find(color_q)
        move = len(color_q)
        s = s[:place] + f"\x1b[31m{color_q}\x1b[0m" + s[place+move:]
    for color_q in to_color2:
        place = s.find(color_q)
        move = len(color_q)
        s = s[:place] + f"\x1b[35m{color_q}\x1b[0m" + s[place+move:]
    print(s)

