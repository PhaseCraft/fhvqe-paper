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

import itertools
import logging

import cirq
import numpy as np
from cirq import CNOT, CZPowGate, H, X, rx, ry, rz
from numpy import pi
import random
import scipy

from fhvqe.settings import DEFAULT, GATESET
from fhvqe.tools import (get_givens, map_JW_to_site, map_rcs_to_site)


module_logger = logging.getLogger("fhvqe.circuit")


def base_sqrt_iswap_gate(qubits):
    """The underlying hardware-native gate we use."""
    yield cirq.ISWAP(qubits[0], qubits[1]) ** 0.5


def cphase_iswap(qubits, theta):
    """CPHASE gate, sqrtISWAP decomposition"""
    if theta > np.pi:
        theta = theta - 2*np.pi
    if theta < -np.pi:
        theta = theta + 2*np.pi
    phi = np.arcsin(np.sqrt(2) * np.sin(theta/4))
    xi = np.arctan(np.tan(phi) / np.sqrt(2))
    
    yield cirq.PhasedXZGate(x_exponent=xi/pi, z_exponent=1+0.5*theta/pi, axis_phase_exponent=-0.5*theta/pi).on(qubits[0])
    yield cirq.PhasedXZGate(x_exponent=-0.5, z_exponent=0.5*theta/pi, axis_phase_exponent=-0.5*theta/pi).on(qubits[1])
    yield from sqrt_iswap_gate(qubits)
    yield cirq.PhasedXZGate(x_exponent=-2*phi/pi, z_exponent=-1, axis_phase_exponent=1).on(qubits[0])
    yield from sqrt_iswap_gate(qubits)
    yield cirq.rx(xi).on(qubits[0])
    yield cirq.X(qubits[1]) ** 0.5
    
    
def givens_iswap(qubits, theta):
    """Givens gate, sqrtISWAP decomposition"""
    theta = theta/2
    yield from sqrt_iswap_gate(qubits)
    yield cirq.rz(pi-theta).on(qubits[0])
    yield cirq.rz(theta).on(qubits[1])
    yield from sqrt_iswap_gate(qubits)
    yield cirq.Z(qubits[0])
    yield cirq.PhasedXZGate(x_exponent=0, z_exponent=0, axis_phase_exponent=0).on(qubits[1]) # include identity gate to ensure that moments alternate between 1 and 2-qubit gates
    
    
def trans_iswap(qubits):
    """Measurement transformation gate, sqrtISWAP decomposition"""
    yield from givens_iswap(qubits, -pi/2)


def BHG_iswap(qubits, theta):
    """Bare hopping gate, sqrtISWAP decomposition"""
    theta = -theta/2
    yield cirq.Z(qubits[0]) ** -0.25
    yield cirq.Z(qubits[1]) ** 0.25
    yield from sqrt_iswap_gate(qubits)
    yield cirq.rz(-theta).on(qubits[0])
    yield cirq.rz(theta).on(qubits[1])
    yield cirq.Z(qubits[0])
    yield from sqrt_iswap_gate(qubits)
    yield cirq.Z(qubits[0])
    yield cirq.PhasedXZGate(x_exponent=0, z_exponent=0, axis_phase_exponent=0).on(qubits[1])
    yield cirq.Z(qubits[0]) ** 0.25
    yield cirq.Z(qubits[1]) ** -0.25
    

def FSWAP_iswap(qubits, angle):
    """Fermionic swap gate, sqrtISWAP decomposition"""
    yield from sqrt_iswap_gate(sorted(qubits))
    yield from sqrt_iswap_gate(sorted(qubits))
    yield cirq.Z(qubits[0]) ** -0.5
    yield cirq.Z(qubits[1]) ** -0.5


def prepH(pairs, qubits):
    """Changes basis for pairs of qubits in preparation for measurement."""
    def generate_prep():
        for pair in pairs:
            yield from horiz_basis_trans((qubits[pair[0]], qubits[pair[1]]))
    return generate_prep


def prepV(pairs, qubits):
    """Changes basis for pairs of qubits in preparation for measurement."""
    return prepH(pairs, qubits)


def prepV2wrap(nh, nv):
    """Inserts FSWAP gates in order to measure vertical terms on non-adjacent qubits"""
    def prepV2(pairs, qubits):
        def generate_prep():
            offset = 0
            n = nv * nh
            for i in range(offset, n, 2):
                if nh != 2 and (i+1) % nh == 0: continue
                yield FSWAP((qubits[i], qubits[i + 1]), 0.0)
                yield FSWAP((qubits[i + n], qubits[i + 1 + n]), 0.0)
            yield from prepH(pairs, qubits)()
        return generate_prep
    return prepV2


def pair_of_two_qubit_gates(qubits, params):
    """Implement a transformation given by a pair of sqrt_iswap gates with a pair of Z rotations
       in the middle, and a Z rotation on the first qubit on either side"""
    yield cirq.PhasedXZGate(x_exponent=0,
                            z_exponent=params[3]/pi,
                            axis_phase_exponent=0).on(qubits[0])
    yield from sqrt_iswap_gate(qubits)
    yield cirq.PhasedXZGate(x_exponent=0,
                            z_exponent=params[1]/pi,
                            axis_phase_exponent=0).on(qubits[0])
    yield cirq.PhasedXZGate(x_exponent=0,
                            z_exponent=params[2]/pi,
                            axis_phase_exponent=0).on(qubits[1])
    yield from sqrt_iswap_gate(qubits)
    yield cirq.PhasedXZGate(x_exponent=0,
                            z_exponent=params[0]/pi,
                            axis_phase_exponent=0).on(qubits[0])
                                
                                
def BHG_measurement(qubits, angle):
    """A transformation used when we combine a hopping gate together with a hopping term measurement"""
    return pair_of_two_qubit_gates(qubits, [angle, -pi/4-angle/2, pi/4-angle/2, 0.0])
    

##################
# Chosen gateset
##################

sqrt_iswap_gate = base_sqrt_iswap_gate

def onsite_gate(qubits, theta):
    return cphase_iswap(qubits, theta)
def givens_gate(qubits, theta):
    return givens_iswap(qubits, theta)
def BHG(qubits, theta):
    return BHG_iswap(qubits, theta)
def FSWAP(qubits, theta):
    return FSWAP_iswap(qubits, theta)
def horiz_basis_trans(qubits):
    return trans_iswap(qubits)
def BHGM(qubits, theta):
    return BHG_measurement(qubits, theta)


##################
# Specific state setup
#################

def initial_state_diff_spin_types(nh, nv, t, nocc1, nocc2, remap=None,
                  mapping=None):
    """Prepares ground state of non-interacting Hamiltonian.

    Assumes JW mapping.

    Args:
        nh -- number of horizontal sites
        nv -- number of vertical sites
        t -- hopping interaction parameter
        nocc -- total number of fermions in the system
        remap -- qubit remapping (default None)
        mapping -- logical layer of qubit mapping (default None):
            ordered:
              spin up:      spin down:
                0   2  4       1  3  5
                6   8 10       7  9 11
                12 14 16      13 15 17
            snake:
                spin up:   spin down:
                0  1  2      3   4  5
                6  7  8      9  10 11
               12 13 14     15  16 17
            jordan wigner:
              spin up:      spin down:
                0 1 2       9 10 11
                5 4 3      14 13 12
                6 7 8      15 16 17
            site:
              spin up:      spin down:
                0 1 2       9 10 11
                3 4 5      12 13 14
                6 7 8      15 16 17
    Returns:
     An iterator with gate instructions to generate the desired initial state.
    """
    slater_circuit = [get_givens(nh, nv, t, nocc1),get_givens(nh, nv, t, nocc2)]

    def generate_init_state():
        #init zero vector
        for q in range(0, nocc1):
            if mapping is not None:
                q = mapping(nh, nv, q)
            if remap:
                q = remap[q]
            yield X(q)
        for q in range(0, nocc2):
            if mapping is not None:
                q = mapping(nh, nv, q)
            q = q + nh * nv
            if remap:
                q = remap[q]
            yield X(q)
            
        for k in (0,1):
            for i in range(len(slater_circuit[k])):
                for j in range(len(slater_circuit[k][i])):
                    q1 = slater_circuit[k][i][j][0]
                    q2 = slater_circuit[k][i][j][1]
                    if mapping is not None:
                        q1 = mapping(nh, nv, slater_circuit[k][i][j][0])
                        q2 = mapping(nh, nv, slater_circuit[k][i][j][1])
                    if remap:
                        if k == 0:
                            q1 = remap[q1]
                            q2 = remap[q2]
                        else:
                            q1 = remap[q1 + nh*nv]
                            q2 = remap[q2 + nh*nv]
                    theta = 2 * slater_circuit[k][i][j][2]
                    yield from givens_gate([q1, q2], theta)
    return generate_init_state


def initial_state(nh, nv, t, nocc, remap=None,
                  mapping=map_JW_to_site):
    """Prepares ground state of non-interacting Hamiltonian.

    Returns:
     An iterator with gate instructions to generate the desired initial state.
    """
    return initial_state_diff_spin_types(nh, nv, t, nocc//2, nocc//2, remap=None,
                                         mapping=map_JW_to_site)

###############
# Generate ansatz
###############
def ansatz(ansatz_def, angles, qubits, print_gates=False):
    """Unpacks the given ansatz definition into gates.

    Args:
        ansatz_def -- iterator giving tuples (name, qubits, angle)
        angles -- parameters for gates in the ansatz, given in a list
        qubits -- qubits on which ansatz is applied

    Returns:
        An iterator with gate instructions to create the desired ansatz.
    """
    layers = ansatz_def(iter(angles), qubits=qubits)
    prog = cirq.Circuit()
    for layer in layers:
        prog.append(cirq.Circuit(ansatz_layers(layer, print_gates=print_gates)),
                    cirq.InsertStrategy.NEW_THEN_INLINE)
    return prog

def ansatz_layers(layer, print_gates=False):
    """Implement each gate in a given layer"""
    for gate_spec in layer:
        #print(gate_spec)
        if __debug__ and print_gates:
            module_logger.debug(gate_spec)
        yield gate_spec[0](gate_spec[1], gate_spec[2])


def ansatz_multistep(ansatz_def, multiangles, qubits):
    """Unpacks the given ansatz definition into layers of gates.

    Args:
        ansatz_def -- iterator giving tuples (name, qubits, angle)
        multiangles -- parameters for gates in the ansatz in a list of lists
                       the number of lists in multiangles corresponds to number
                       of ansatz layers, and each of the lists within give
                       parameters to be used in each of the layers
        qubits -- qubits on which ansatz is applied

    Returns:
        An iterator with gate instructions to create the desired ansatz.
    """
    for angles in multiangles:
        yield ansatz(ansatz_def, angles, qubits)


def ansatz_multilayer_circuit(ansatz_def, params, qubits):
    """Create a quantum circuit from an ansatz definition layer-by-layer."""
    prog = cirq.Circuit()
    for angles in params:
        prog.append(cirq.Circuit(ansatz(ansatz_def, angles, qubits)),
                    cirq.InsertStrategy.NEW_THEN_INLINE)
    return prog


def ansatz_multilayer_circuit_merge(ansatz_def, params, qubits):
    """Create a quantum circuit from an ansatz definition layer-by-layer,
       sometimes merging the last layer with measurement."""
    print_gates = False
    prog = cirq.Circuit()
    # first construct any undisturbed layers
    for angles in params[:-1]:
        prog.append(cirq.Circuit(ansatz(ansatz_def, angles, qubits)),
                    cirq.InsertStrategy.NEW_THEN_INLINE)
                    
    # in the last layer we have two options, we either want the standard
    # compile, or for some measurements we want to merge the hopping layer
    # with the measurement basis change
    temp_circuit = cirq.Circuit()
    temp_prog = []
    last_layer = cirq.Circuit()
    angles = params[-1]
    layers = ansatz_def(iter(angles), qubits=qubits)
    prev_layer = None
    for layer in layers:
        if prev_layer:
            for gate_spec in prev_layer:
#                print(gate_spec)
                if __debug__ and print_gates:
                    module_logger.debug(gate_spec)
                temp_prog.append(gate_spec[0](gate_spec[1], gate_spec[2]))
            temp_circuit.append(cirq.Circuit(temp_prog), cirq.InsertStrategy.NEW_THEN_INLINE)
            temp_prog = []
        prev_layer = layer
        
    prog.append(temp_circuit)

    prog_with_last_hop = prog.copy()
    if prev_layer:
        for gate_spec in prev_layer:
#            print(gate_spec)
            if __debug__ and print_gates:
                module_logger.debug(gate_spec)
            temp_prog.append(gate_spec[0](gate_spec[1], gate_spec[2]))
        prog_with_last_hop.append(cirq.Circuit(temp_prog),
                                  cirq.InsertStrategy.NEW_THEN_INLINE)
                                  
    return prog, prog_with_last_hop



###############
# Ansatz -- (Efficient) hamiltonian variational
###############

def horiz_layer(angle, qubits_t, nh, nv, offset):
    """Creates an odd/even horizontal layer of gates given an angle.

    Create a horizontal layer for the hamiltonian variational anstaz.
    This consists of the bare hopping gate only, with no swapping.

    Assumes JW ordering.

    Args:
        angle -- parameter for the given horizontal gate layer
        qubits_t -- list of physical qubits on which the gates are being appiled
        n -- total number of sites of specific spin type (nh * nv)
        offset -- odd (1) or even (0) layer of gates

    Returns:
        Iterator giving tuples with gate details (name, qubits, angles)
    """
    n = nv * nh
    for i in range(offset, n, 2):
        if nh != 2 and (i+1) % nh == 0: continue
        yield (BHG, (qubits_t[i], qubits_t[i + 1]), angle)
        yield (BHG, (qubits_t[i + n], qubits_t[i + 1 + n]), angle)


def vert_layer(angle, qubits_t, nh, nv, offset):
    """Creates an odd/even vertical layer of gates given an angle.

    Create a vertical layer for the efficient hamiltonian variational ansatz.
    This consists of the bare hopping gate only, assumes the necessary connectivity.
    The bare hopping gates occur on the leftmost and the rightmost qubits which are
    currently connected in the JW string.

    Assumes qubits are in JW ordering.

    Args:
        angle -- parameter for the given horizontal gate layer
        qubits_t -- list of physical qubits on which the gates are being appiled
        nh -- number of horizontal sites
        nv -- number of vertical sites
        offset -- odd (1) or even (0) layer of gates

    Returns:
        Iterator giving tuples with gate details (name, qubits, angles)
    """
    n = nv * nh
    for row in range(offset, nv-1, 2):
        site = row * nh - 1
        yield (BHG, (qubits_t[site], qubits_t[site+1]), angle)
        yield (BHG, (qubits_t[site+n], qubits_t[site+1+n]), angle)


def horiz_swap_layer(angle, qubits_t, nh, nv, offset, swap=False):
    """Creates an odd/even horizontal layer with fermionic swap incorporated.

    Create a horizontal layer for the efficient hamiltonian variational ansatz.
    This consists of the bare hopping gate mixed with fermionic swap for efficiency.

    Assumes qubits are in JW ordering.

    Args:
        angle -- parameter for the given horizontal gate layer
        qubits_t -- list of physical qubits on which the gates are being appiled
        n -- total number of sites of specific spin type (nh * nv)
        offset -- odd (1) or even (0) layer of gates

    Returns:
        Iterator giving tuples with gate details (name, qubits, angles)
    """
    n = nv * nh
    if swap:
        angle -= pi
    for i in range(offset, n, 2):
        if nh != 2 and (i+1) % nh == 0: continue
        ps_angle = -pi
        yield (BHG, (qubits_t[i], qubits_t[i + 1]), angle)
        yield (BHG, (qubits_t[i + n], qubits_t[i + 1 + n]), angle)
        if swap:
            yield (RZ, (qubits_t[i], ), ps_angle)
            yield (RZ, (qubits_t[i+1], ), ps_angle)
            yield (RZ, (qubits_t[i+n], ), ps_angle)
            yield (RZ, (qubits_t[i+n+1], ), ps_angle)


def vert_swap_layer(angles, qubits_t, nh, nv):
    """Creates a vertical hopping layer for efficient hamiltonian variational ansatz.

    This layer applies parts of both even and off vertical swap layer depending
    on which of the vertical pairs are found next to each other in the JW string.

    Assumes qubits are in JW ordering.

    Args:
        angles -- list of angles to be applied (first one is for the even, second for odd)
        qubits_t -- list of physical qubits on which the gates are being appiled
        nh -- number of horizontal sites
        nv -- number of vertical sites

    Returns:
        Iterator giving tuples with gate details (name, qubits, angles)
    """
    n = nv * nh
    for row in range(nv-1):
        site = (row + 1) * nh - 1
        yield (BHG, (qubits_t[site], qubits_t[site+1]), angles[row%2])
        yield (BHG, (qubits_t[site+n], qubits_t[site+1+n]), angles[row%2])


def onsite_layer(angle, qubits_t, n):
    """Creates an onsite layer of gates given an angle.

    Assumes appropriate connections between qubits exist.

    Args:
        angle -- parameter for the given horizontal gate layer
        qubits_t -- list of physical qubits on which the gates are being appiled
        n -- total number of sites of specific spin type (nh * nv)

    Returns:
        Iterator giving tuples with gate details (name, qubits, angles)
    """
    for i in range(n):
        yield (onsite_gate, (qubits_t[i], qubits_t[i+n]), angle)



def any_h_by_v(qubits_t, nh, nv, swap = True):
    """Creates efficent hamiltonian variational ansatz function for the given lattice size.

    This ansatz will create the extra swaps for a 1D horizontal lattice.

    Args:
        qubits_t -- list of physical qubits on which the gates are being appiled
        nh -- number of horizontal sites
        nv -- number of vertical sites
        swap -- whether swaps are integrated into hoppoing gates or not (default True)
                (CURRENTLY DISFUNCTIONAL)

    Returns:
        Iterator giving tuples with gate details (name, qubits, angles)
    """
    n = nh * nv
    def _nh_by_nv(angles, qubits=qubits_t):
        yield onsite_layer(next(angles), qubits, n)
        for i in range(nh):
            if (i == 0):
                angle = next(angles)
                if swap:
                    angle -= pi
                yield horiz_swap_layer(angle, qubits, nh, nv, 0, swap=swap)
                if not swap:
                    yield fswap_layer(qubits, nh, nv, 0)
                yield fswap_layer(qubits, nh, nv, 1)
            elif (i==nh-1):
                angle = next(angles)
                if swap:
                    angle -= pi
                yield fswap_layer(qubits, nh, nv, 0)
                if not swap:
                    yield fswap_layer(qubits, nh, nv, 1)
                yield horiz_swap_layer(angle, qubits, nh, nv, 1, swap=swap)
            else:
                yield fswap_layer(qubits, nh, nv, 0)
                yield fswap_layer(qubits, nh, nv, 0)
            yield vert_swap_layer(list(angles), qubits, nh, nv)
    return _nh_by_nv


def hopping_measurement_ansatz(qubits_t, nh, nv, offset):
    """Creates the transformations needed to measure the hopping terms."""
    n = nv * nh
    def _hopping_measurement_ansatz(angles, qubits=qubits_t):
        hopping_list = []
        angle = next(angles)
        for i in range(offset, n-1, 2):
            hopping_list.append((BHGM, (qubits_t[i], qubits_t[i + 1]), angle))
            hopping_list.append((BHGM, (qubits_t[i + n], qubits_t[i + 1 + n]), angle))
        yield hopping_list
    return _hopping_measurement_ansatz



def any_h_by_v_explicit(qubits_t, nh, nv, swap = True):
    """Creates hamiltonian variational ansatz function for the given lattice size.

    This ansatz will treats 2x1, 1x2, nx1, 1xn, and any other nxm separately.
    This is to ensure efficency for the smaller lattice sizes.

    Args:
        qubits_t -- list of physical qubits on which the gates are being appiled
        nh -- number of horizontal sites
        nv -- number of vertical sites
        swap -- whether swaps are integrate into hoppoing gates or not (default True)
                (CURRENTLY DISFUNCTIONAL)

    Returns:
        Iterator giving tuples with gate details (name, qubits, angles)
    """
    n = nh * nv
    def _2_by_1(angles, qubits = qubits_t):
        yield onsite_layer(next(angles), qubits, n)
        yield horiz_layer(next(angles), qubits, nh, nv, 0)
    def _1_by_2(angles, qubits = qubits_t):
        yield onsite_layer(next(angles), qubits, n)
        yield vert_layer(next(angles), qubits, nh, nv, 0)
    def _n_by_1(angles, qubits = qubits_t):
        yield onsite_layer(next(angles), qubits, n)
        yield horiz_layer(next(angles), qubits, nh, nv, 0)
        yield horiz_layer(next(angles), qubits, nh, nv, 1)
    def _1_by_n(angles, qubits = qubits_t):
        yield onsite_layer(next(angles), qubits, n)
        yield vert_layer(next(angles), qubits, nh, nv, 0)
        yield vert_layer(next(angles), qubits, nh, nv, 1)
    def _nh_by_nv(angles, qubits=qubits_t):
        yield onsite_layer(next(angles), qubits, n)
        for i in range(nh):
            if (i == 0):
                yield horiz_swap_layer(next(angles), qubits, nh, nv, 0, swap=swap)
                if not swap:
                    yield fswap_layer(qubits, nh, nv, 0)
                if nh > 2: yield fswap_layer(qubits, nh, nv, 1)
                vertical_angles = [next(angles)]
                if nv > 2: vertical_angles.append(next(angles))
            elif (i == nh-1):
                yield fswap_layer(qubits, nh, nv, 0)
                if not swap:
                    yield fswap_layer(qubits, nh, nv, 1)
                if nh > 2: yield horiz_swap_layer(next(angles), qubits, nh, nv, 0, swap=swap)
            else:
                yield fswap_layer(qubits, nh, nv, 0)
                yield fswap_layer(qubits, nh, nv, 1)
            yield vert_swap_layer(vertical_angles, qubits, nh, nv)
    if nh == 1:
        if nv == 2:
            return _1_by_2
        return _1_by_n
    if nv == 1:
        if nh == 2:
            return _2_by_1
        return _n_by_1
    return _nh_by_nv


def one_by_n(qubits_t, n):
    """Creates hamiltonian variational ansatz function for 1xn lattice.

    Args:
        qubits_t -- list of physical qubits on which the gates are being appiled
        n -- total number of sites of specific spin type (nh * nv)

    Returns:
        Iterator giving tuples with gate details (name, qubits, angles)
    """
    return any_h_by_v_explicit(qubits_t, n, 1)


def fswap_layer(qubits_t, nh, nv, offset):
    """Creates an odd/even horizontal fermionic swap layer.

    Assumes qubits are in JW ordering.

    Args:
        qubits_t -- list of physical qubits on which the gates are being appiled
        offset -- odd (1) or even (0) layer of gates
        nh -- number of horizontal sites
        nv -- number of vertical sites

    Returns:
        Iterator giving tuples with gate details (name, qubits, angles)
    """
    n = nv * nh
    for i in range(offset, n, 2):
        if nh != 2 and (i+1) % nh == 0: continue
        yield (FSWAP, (qubits_t[i], qubits_t[i + 1]), 0.0)
        yield (FSWAP, (qubits_t[i + n], qubits_t[i + 1 + n]), 0.0)


def fswap_zigzag_layer(qubits_t, nh, nv):
    """Creates a layer of FSWAP gates. Used in zig-zag ordering to enable
       onsite gates across non-neighbouring qubits.
    """
    n = nv * nh
    for i in range(1, n, 2):
        yield (FSWAP, (qubits_t[i], qubits_t[i - 1 + n]), 0.0)


def onsite_zigzag_layer(angle, qubits_t, n):
    """Creates an onsite layer of gates given an angle.

    Assumes appropriate connections between qubits exist.

    Args:
        angle -- parameter for the given horizontal gate layer
        qubits_t -- list of physical qubits on which the gates are being appiled
        n -- total number of sites of specific spin type (nh * nv)

    Returns:
        Iterator giving tuples with gate details (name, qubits, angles)
    """
    for i in range(0, 2*n, 2):
        yield (onsite_gate, (qubits_t[i], qubits_t[i+1]), angle)


def one_by_n_zigzag(qubits_t, nh, nv, swap = True):
    """A layer of the 1xn ansatz with a zigzag configuration,
       including fermionic swaps around the onsite part. """
    def _one_by_n_zigzag(angles, qubits = qubits_t):
        """
        """
        n = nh * nv
        yield fswap_zigzag_layer(qubits, nh, nv)
        yield onsite_zigzag_layer(next(angles), qubits, n)
        yield fswap_zigzag_layer(qubits, nh, nv)
        yield horiz_layer(next(angles), qubits, nh, nv, 0)
        yield horiz_layer(next(angles), qubits, nh, nv, 1)
    return _one_by_n_zigzag
    

def two_by_n_zigzag(qubits_t, nh, nv, swap = True):
    """A layer of the 2xn ansatz with a zigzag configuration,
       including fermionic swaps around the onsite part. """
    def _two_by_n_zigzag(angles, qubits=qubits_t):
        n = nh * nv
        yield fswap_zigzag_layer(qubits, nh, nv)
        yield onsite_zigzag_layer(next(angles), qubits, n)
        yield fswap_zigzag_layer(qubits, nh, nv)
        for i in range(nh):
            if (i == 0):
                yield horiz_swap_layer(next(angles), qubits, nh, nv, 0, swap=swap)
                if not swap:
                    yield fswap_layer(qubits, nh, nv, 0)
                if nh > 2: yield fswap_layer(qubits, nh, nv, 1)
                vertical_angles = [next(angles)]
                if nv > 2: vertical_angles.append(next(angles))
            elif (i == nh-1):
                yield fswap_layer(qubits, nh, nv, 0)
                if not swap:
                    yield fswap_layer(qubits, nh, nv, 1)
                if nh > 2: yield horiz_swap_layer(next(angles), qubits, nh, nv, 0, swap=swap)
            else:
                yield fswap_layer(qubits, nh, nv, 0)
                yield fswap_layer(qubits, nh, nv, 1)
            yield vert_swap_layer(vertical_angles, qubits, nh, nv)
    return _two_by_n_zigzag

###############
# Ansatz -- fixed definitions
###############


def one_by_two_square(angles, qubits=None):
    """Creates hamiltonian variational ansatz for 1x2 lattice assuming all-to-all connectivity.

    Single layer only. Returns an iterator. Apply multiple copies to have multilayer ansatz.

    Args:
        angles -- list of parameters used in the ansatz.
        qubits -- list of physical qubits on which the gates are being appiled

    Returns:
        Iterator giving tuples with gate details (name, qubits, angles)
    """
    onsite_angle, horizontal_angle_1 = angles
    yield  from [ (onsite_gate, (qubits[0], qubits[2]), onsite_angle),
                  (onsite_gate, (qubits[1], qubits[3]), onsite_angle),
                ]
    yield from  [ (BHG, (qubits[0], qubits[1]), horizontal_angle_1),
                  (BHG, (qubits[2], qubits[3]), horizontal_angle_1),
                ]


def one_by_three_square(angles, qubits=None):
    """Creates hamiltonian variational ansatz for 1x3 lattice assuming all-to-all connectivity.

    Single layer only. Returns an interator. Apply multiple copies to have multilayer ansatz.

    Args:
        angles -- list of parameters used in the ansatz.
        qubits -- list of physical qubits on which the gates are being appiled

    Returns:
        Iterator giving tuples with gate details (name, qubits, angles)
    """
    onsite_angle, horizontal_angle_1, horizontal_angle_2 = angles
    yield from [ (onsite_gate, (qubits[0], qubits[3]), onsite_angle),
                 (onsite_gate, (qubits[1], qubits[4]), onsite_angle),
                 (onsite_gate, (qubits[2], qubits[5]), onsite_angle)
               ]
    yield from [ (BHG, (qubits[0], qubits[1]), horizontal_angle_1),
                 (BHG, (qubits[3], qubits[4]), horizontal_angle_1),
               ]
    yield from [ (BHG, (qubits[1], qubits[2]), horizontal_angle_2),
                 (BHG, (qubits[4], qubits[5]), horizontal_angle_2),
               ]


def one_by_four_square(angles, qubits=None):
    """Creates hamiltonian variational ansatz for 1x4 lattice assuming all-to-all connectivity.

    Single layer only. Returns an iterator. Apply multiple copies to have multilayer ansatz.

    Args:
        angles -- list of parameters used in the ansatz.
        qubits -- list of physical qubits on which the gates are being appiled

    Returns:
        Iterator giving tuples with gate details (name, qubits, angles)
    """
    onsite_angle, horizontal_angle_1, horizontal_angle_2 = angles
    yield from [ (onsite_gate, (qubits[0], qubits[4]), onsite_angle),
                 (onsite_gate, (qubits[1], qubits[5]), onsite_angle),
                 (onsite_gate, (qubits[2], qubits[6]), onsite_angle),
                 (onsite_gate, (qubits[3], qubits[7]), onsite_angle)
               ]
    yield from [ (BHG, (qubits[0], qubits[1]), horizontal_angle_1),
                 (BHG, (qubits[2], qubits[3]), horizontal_angle_1),
                 (BHG, (qubits[4], qubits[5]), horizontal_angle_1),
                 (BHG, (qubits[6], qubits[7]), horizontal_angle_1)
               ]
    yield from [ (BHG, (qubits[1], qubits[2]), horizontal_angle_2),
                 (BHG, (qubits[5], qubits[6]), horizontal_angle_2),
               ]


def two_by_two_square(angles, qubits=None):
    """Creates efficient hamiltonian variational ansatz for 2x2 lattice assuming all-to-all connectivity.

    Single layer only. Returns an iterator. Apply multiple copies to have multilayer ansatz.

    Args:
        angles -- list of parameters used in the ansatz.
        qubits -- list of physical qubits on which the gates are being appiled

    Returns:
        Iterator giving tuples with gate details (name, qubits, angles)
    """
    o, h1, v1 = angles
    yield from [ (onsite_gate, (qubits[0], qubits[4]), o),
                 (onsite_gate, (qubits[1], qubits[5]), o),
                 (onsite_gate, (qubits[2], qubits[6]), o),
                 (onsite_gate, (qubits[3], qubits[7]), o)
               ]
    yield from [ (BHG, (qubits[0], qubits[1]), h1),
                 (BHG, (qubits[4], qubits[5]), h1),
                 (BHG, (qubits[2], qubits[3]), h1),
                 (BHG, (qubits[6], qubits[7]), h1),
                 (FSWAP, (qubits[0], qubits[1]), h1),
                 (FSWAP, (qubits[4], qubits[5]), h1),
                 (FSWAP, (qubits[2], qubits[3]), h1),
                 (FSWAP, (qubits[6], qubits[7]), h1),
                 (BHG, (qubits[1], qubits[3]), v1),
                 (BHG, (qubits[5], qubits[7]), v1),
               ]
    yield from [ (FSWAP, (qubits[0], qubits[1]), h1),
                 (FSWAP, (qubits[4], qubits[5]), h1),
                 (FSWAP, (qubits[2], qubits[3]), h1),
                 (FSWAP, (qubits[6], qubits[7]), h1),
                 (BHG, (qubits[1], qubits[3]), v1),
                 (BHG, (qubits[5], qubits[7]), v1),
               ]


def empty_ansatz(angles=None, qubits=None):
    """Creates empty ansatz.

    Useful for inspecition of the initial state creation / measurement preparation.

    Args:
        angles -- list of parameters used in the ansatz.
        qubits -- list of physical qubits on which the gates are being appiled

    Returns:
        Empty iterator.
    """
    yield from ()


def two_by_two_sycamore(angles, qubits=None):
    """Creates efficient hamiltonian variational ansatz for 2x2 lattice assuming sycamore connectivity.

    This ansatz assumes the qubits are physical found in a zig-zag like ordering, and require
    swaps two bring onsite terms together.
      4 5
    0 1 6 7
      2 3
    Single layer only. Returns an iterator. Apply multiple copies to have multilayer ansatz.

    Args:
        angles -- list of parameters used in the ansatz.
        qubits -- list of physical qubits on which the gates are being appiled

    Returns:
        Iterator giving tuples with gate details (name, qubits, angles)
    """
    o, h1, v1 = angles
    yield [ (BHG, (qubits[1], qubits[2]), v1),
            (BHG, (qubits[5], qubits[6]), v1)
          ]
    yield [ (BHG, (qubits[0], qubits[1]), h1),
            (BHG, (qubits[2], qubits[3]), h1),
            (BHG, (qubits[4], qubits[5]), h1),
            (BHG, (qubits[6], qubits[7]), h1),
            (FSWAP, (qubits[0], qubits[1]), h1),
            (FSWAP, (qubits[2], qubits[3]), h1),
            (FSWAP, (qubits[4], qubits[5]), h1),
            (FSWAP, (qubits[6], qubits[7]), h1)
          ]
    yield [ (BHG, (qubits[1], qubits[2]), v1),
            (BHG, (qubits[5], qubits[6]), v1)
          ]
    yield [ (SWAP_orig, (qubits[1], qubits[4]), v1),
            (SWAP_orig, (qubits[3], qubits[6]), v1)
          ]
    yield [ (onsite_gate, (qubits[0], qubits[1]), o),
            (onsite_gate, (qubits[2], qubits[3]), o),
            (onsite_gate, (qubits[4], qubits[5]), o),
            (onsite_gate, (qubits[6], qubits[7]), o)
          ]
    yield [ (SWAP_orig, (qubits[1], qubits[4]), v1),
            (SWAP_orig, (qubits[3], qubits[6]), v1)
          ]


def two_by_two_syc_line(angles, qubits=None):
    """Creates hamiltonian variational ansatz for 2x2 lattice assuming sycamore connectivity.

    This ansatz assumes the qubits are ordered in a line on top of each other for two spin types:
    0 1 2 3
    4 5 6 7
    Single layer only. Returns an iterator. Apply multiple copies to have multilayer ansatz.

    Args:
        angles -- list of parameters used in the ansatz.
        qubits -- list of physical qubits on which the gates are being appiled

    Returns:
        Iterator giving tuples with gate details (name, qubits, angles)
    """
    o, h1, v1 = angles
    yield [ (onsite_gate, (qubits[0], qubits[4]), o),
            (onsite_gate, (qubits[1], qubits[5]), o),
            (onsite_gate, (qubits[2], qubits[6]), o),
            (onsite_gate, (qubits[3], qubits[7]), o)
          ]
    yield [ (BHG, (qubits[1], qubits[2]), v1),
            (BHG, (qubits[5], qubits[6]), v1)
          ]
    yield [ (BHG, (qubits[0], qubits[1]), h1),
            (BHG, (qubits[2], qubits[3]), h1),
            (BHG, (qubits[4], qubits[5]), h1),
            (BHG, (qubits[6], qubits[7]), h1),
            (FSWAP, (qubits[0], qubits[1]), h1),
            (FSWAP, (qubits[2], qubits[3]), h1),
            (FSWAP, (qubits[4], qubits[5]), h1),
            (FSWAP, (qubits[6], qubits[7]), h1)
          ]
    yield [ (BHG, (qubits[1], qubits[2]), v1),
            (BHG, (qubits[5], qubits[6]), v1)
          ]


def two_by_four_line(angles, qubits=None):
    """Creates efficient hamiltonian variational ansatz for 2x4 lattice assuming NN connectivity.

    This ansatz assumes the qubits are ordered in a line on top of each other for two spin types:
    0 1 2  3  4  5  6  7
    8 9 10 11 12 13 14 15
    Single layer only. Returns an iterator. Apply multiple copies to have multilayer ansatz.

    Args:
        angles -- list of parameters used in the ansatz.
        qubits -- list of physical qubits on which the gates are being appiled

    Returns:
        Iterator giving tuples with gate details (name, qubits, angles)
    """
    o, h1, v1, v2 = angles
    yield [(onsite_gate, (qubits[i], qubits[2*4 + i]), o) for i in range(8)]
    yield [ (BHG, (qubits[i], qubits[i+1]), h1) for i in range(0,16,2) ]
    yield [ (BHG, (qubits[1], qubits[2]), v1),
            (BHG, (qubits[3], qubits[4]), v2),
            (BHG, (qubits[5], qubits[6]), v1),
            (BHG, (qubits[9], qubits[10]), v1),
            (BHG, (qubits[11], qubits[12]), v2),
            (BHG, (qubits[13], qubits[14]), v1),
          ]
    yield [ (FSWAP, (qubits[i], qubits[i+1]), 0.0) for i in range(0,16,2) ]
    yield [ (BHG, (qubits[1], qubits[2]), v1),
            (BHG, (qubits[3], qubits[4]), v2),
            (BHG, (qubits[5], qubits[6]), v1),
            (BHG, (qubits[9], qubits[10]), v1),
            (BHG, (qubits[11], qubits[12]), v2),
            (BHG, (qubits[13], qubits[14]), v1),
          ]


NAMED_ANSATZ = { "h_by_v_explicit": any_h_by_v_explicit,
                 "1x2_square": one_by_two_square,
                 "1x3_square": one_by_three_square,
                 "1x4_square": one_by_four_square,
                 "2x2_square": two_by_two_square,
                 "empty": empty_ansatz,
                 "two_by_two_syc": two_by_two_sycamore,
                 "two_by_two_line": two_by_two_syc_line,
                 "two_by_four_line": two_by_four_line,
                 "two_by_n_zigzag": two_by_n_zigzag,
                 "one_by_n_zigzag": one_by_n_zigzag,
                }
