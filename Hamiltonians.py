""" Author: Charlie Nation

Uses QuSpin (http://weinbe58.github.io/QuSpin/) to build spin-chain Hamiltonians.
"""

import numpy as np
# Quspin packages for exact diagonalization
from quspin.basis import spin_basis_1d  # Hilbert spaces
from quspin.operators import hamiltonian  # operators


class SpinChain:
    """ Spin-chain used in PRE 99 (5), 052139"""
    def __init__(self, n, params):
        self.N = n
        self.n_link = params[0]  # link qubit index
        self.j_sz = params[1]  # zz coupling between system and link qubits
        self.j_sx = params[2]  # xx coupling between system and link qubits
        self.j_bx = params[3]  # xx coupling in bath
        self.j_bz = params[4]  # zz coupling in bath
        self.b_sx = params[5]  # x field on system qubit
        self.b_sz = params[6]  # z field on system qubit
        self.b_bx = params[7]  # x field on bath qubit
        self.b_bz = params[8]  # z field on bath qubit

    def hamiltonian(self):
        """ Defines Hamiltonian used in used in PRE 99 (5), 052139"""

        # Define basis
        basis = spin_basis_1d(L=self.N, check_z_symm=False)
        # set parameters
        j_bx = self.j_bx / 4. * np.ones(self.N)
        j_bx[0] = 0
        b_0z = self.b_bz * np.ones(self.N)
        j_bz = self.j_bz * (np.ones(self.N))

        j_bz[0] = 0

        b_0z[0] = self.b_sz
        b_0x = self.b_bx * np.ones(self.N)
        b_0x[0] = self.b_sx

        # create lists generating operators
        # non-interacting part
        j_bzo = [[j_bz[i], i, i + 1] for i in range(1, self.N - 1)]  # OBC
        j_bxo = [[j_bx[i], i, i + 1] for i in range(1, self.N - 1)]  # OBC
        b_0xo = [[b_0x[i], i] for i in range(self.N)]
        b_0zo = [[b_0z[i], i] for i in range(self.N)]

        static_0 = [['-+', j_bxo], ['+-', j_bxo], ['zz', j_bzo], ['x', b_0xo], ['z', b_0zo]]
        dynamic_0 = []

        # Interaction operators
        j_szo = [[self.j_sz, 0, self.n_link]]  # OBC
        j_sxo = [[self.j_sx / 4., 0, self.n_link]]  # OBC

        static_i = [['zz', j_szo], ['+-', j_sxo], ['-+', j_sxo]]
        dynamic_i = []

        # build Hamiltonian
        h_0 = hamiltonian(static_0, dynamic_0, basis=basis, dtype=np.float64)
        h_i = hamiltonian(static_i, dynamic_i, basis=basis, dtype=np.float64)

        return h_0, h_i

    def choose_initial_state(self, init_state, eigstate=None):
        """ Choice of initial pure state |psi_0 > = |up>_S \otimes |choice of bath state>_B"""

        if init_state == 'Neel':  # anti-ferromagnetic state
            psi_0 = np.array([1., 0.])
            for p in range(self.N - 1):
                psi_0 = np.kron(psi_0, [(1. + (-1.) ** p) / 2., (1. - (-1.) ** p) / 2.])
        elif init_state == 'All down':  # All qubits (except systems) in |down> state
            psi_0 = np.array([1., 0.])
            for p in range(self.N - 1):
                psi_0 = np.kron(psi_0, [0., 1.])
        elif init_state == 'ES':  # |up> \otimes |bath_eigenstate>. Default = middle energy of bath

            if eigstate is None:
                eigstate = 2**(self.N - 2)  # Default

            h_0, _ = self.hamiltonian()
            obsz = self.observable(0, 'z')
            ancH = h_0 - (1000 + self.b_sz) * obsz
            _, v = ancH.eigh()
            psi_0 = v[:, eigstate]
        else:
            raise Exception('Unrecognised init_state')

        return psi_0

    def observable(self, index=0, operator='z'):
        """index gives spin index of local (up to two spins) observable
        operator gives pauli matrix type, 'z' for sigma_z,
        or e.g xx for sigma_x^index * sigma_x^(index+1)."""

        basis = spin_basis_1d(L=self.N, check_z_symm=False)
        if len(operator) == 1:
            o_list = np.zeros(self.N)
            o_list[index] = 1.
            obs_list = [[o_list[i], i] for i in range(self.N)]
            static_o = [[operator, obs_list]]
            obs = hamiltonian(static_o, [], basis=basis, dtype=np.float64)
        else:
            o_list = np.zeros(self.N)
            o_list[index] = 1.
            obs_list = [[o_list[i], i, i + 1] for i in range(self.N - 1)]
            static_o = [[operator, obs_list]]
            obs = hamiltonian(static_o, [], basis=basis, dtype=np.float64)

        return obs

    def save_groups(self, init):
        """Gives names for save files. Group and subgroup of hdf5 files."""
        parameter_group = f'N={self.N},N_link={self.n_link},' \
                          f'J_Sz={self.j_sz},J_Sx={self.j_sx},J_Bx={self.j_bx},' \
                          f'J_Bz={self.j_bz},B_Sx={self.b_sx},' \
                          f'B_Sz={self.b_sz},B_Bx={self.b_bx},B_Bz={self.b_bz},'
        init_group = f'init={init}'
        return parameter_group, init_group


class RandomMatrix:
    def __init__(self, n, g):
        self.N = n
        self.g = g

    def hamiltonian(self):
        h_0 = np.diag(np.linspace(0, 1, self.N))
        v = np.random.normal(0, 1, size=(self.N, self.N))
        v = np.triu(v)
        v = v + np.transpose(v)
        tr = sum(np.diag(np.dot(v, np.transpose(v)))) / self.N**2
        v = v / tr
        g = self.g / np.sqrt(self.N)
        h = h_0 + g * v
        return h

    def choose_initial_state(self, init=None):
        """very simple initial state for now. Element with index = init = 1. All others = 0.
            Note, this is an eigenstate of H_0."""
        if init is None:
            init = self.N // 2
        psi_0 = np.zeros(self.N)
        psi_0[init] = 1.
        return psi_0

    def observable(self, O_type):
        """Random matrix observables, O_odd and O_sym in , New. J. Phys. 20 (10), 103003
        and PRE 99 (5), 052139"""
        if O_type == 'odd':
            obs = np.zeros(self.N)
            obs[0:self.N:2] = np.ones(int(self.N / 2))
            obs = np.diag(obs)
            return obs

        elif O_type == 'sym':
            obs = - np.ones(self.N)
            obs[0:self.N:2] = np.ones(int(self.N / 2))
            obs = np.diag(obs)
            return obs

        else:
            raise Exception('Unrecognised O_type')

    def save_groups(self, init=None):
        """Gives names for save files. Group and subgroup of hdf5 files."""
        if init is None:
            init = self.N // 2
        parameter_group = f'N={self.N},g={self.g},'
        init_group = f'init={init}'
        return parameter_group, init_group
