#   Copyright 2026 Jukka Aho
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
"""Exact ground-state spin correlations read in the state's own qubit layout.

The released 2x4 ``Ground state`` spin correlations and the ground-state
entries of the staggered panel were read with operators written for the
``snake`` layout from a state stored in OpenFermion's ``ordered`` layout
(mode ``2 * (nh * row + col) + spin``). The script that produced them was
not released; this one reads the operators with the layout of the state.
Passing ``layout=map_rcs_to_snake`` reproduces the released numbers.
"""

import numpy as np
import openfermion as of
import scipy.linalg

from fhvqe.tools import map_rcs_to_ordered, map_rcs_to_snake


def ground_state(nh, nv, t, U, n_up, n_down):
    """Energy, basis-state weights |psi_b|^2 and mode occupations occ[q, b]."""
    n_qubits = 2 * nh * nv
    occupation = np.array([of.get_sparse_operator(of.FermionOperator(((q, 1), (q, 0))),
                                                  n_qubits).diagonal().real for q in range(n_qubits)])
    basis = np.flatnonzero((occupation[0::2].sum(0) == n_up) & (occupation[1::2].sum(0) == n_down))
    hamiltonian = of.get_sparse_operator(of.fermi_hubbard(nh, nv, t, U, periodic=False), n_qubits)
    energies, vectors = scipy.linalg.eigh(hamiltonian[basis][:, basis].toarray().real, subset_by_index=[0, 0])
    weights = np.zeros(2 ** n_qubits)
    weights[basis] = vectors[:, 0] ** 2
    return energies[0], weights, occupation


def spin_profile(nh, nv, t, U, n_up, n_down, layout=map_rcs_to_ordered):
    """<m_1 m_j> - <m_1><m_j> with m = n_up - n_down along the figure's site order.

    The order is row by row with every odd row reversed: 11, 21, 22, 12, 13,
    23, 24, 14 for 2x4 (column, row) and sites 1..8 for 1x8.
    """
    energy, weights, occupation = ground_state(nh, nv, t, U, n_up, n_down)
    sites = [(row, col) for row in range(nv)
             for col in (range(nh) if row % 2 == 0 else range(nh - 1, -1, -1))]
    m = [occupation[layout(nh, nv, row, col, 0)] - occupation[layout(nh, nv, row, col, 1)]
         for (row, col) in sites]
    mean = [weights @ m_j for m_j in m]
    return np.array([weights @ (m[0] * m_j) - mean[0] * mean_j for m_j, mean_j in zip(m, mean)])


def staggered(profile):
    """Alternating sum of the profile over the sites after the first (Fig. 10)."""
    return sum((-1) ** k * value for k, value in enumerate(profile) if k > 0)


if __name__ == "__main__":
    print("2x4 U=4 (4,4) Ground state:", np.round(spin_profile(2, 4, 1, 4, 4, 4), 8))
    for n_up, n_down in ((4, 3), (4, 4)):
        values = [staggered(spin_profile(2, 4, 1, U, n_up, n_down)) for U in (0, 4, 8)]
        print(f"2x4 N_occ={n_up + n_down} (GS) staggered at U=0,4,8:", np.round(values, 7))
