""" Author: Charlie Nation

Various operations performed for verifying the FDT
Uses QuSpin for exact diagonalization of spin-chain Hamiltonians.
"""

import numpy as np
# Quspin packages for exact diagonalization
from quspin.operators import exp_op  # operators
from quspin.tools.measurements import obs_vs_time, ED_state_vs_time  # calculating dynamics
from scipy.optimize import curve_fit
from quspin.operators import hamiltonian, ishamiltonian


class HamilOperations:
    """Calculations that require diagonalization of H
    observables should be a dictionary of observables. e.g. dict(z=obs_z, x-obs_x, ...)"""
    def __init__(self, hamiltonian, observables, initial_state):
        self.H = hamiltonian
        self.Obs_dict = observables
        self.psi_0 = initial_state

        #  Checking if the operations will use the QuSpin package - to so so hamiltonian must be an
        #  instance of the QuSpin Hamiltonian class
        if ishamiltonian(hamiltonian):
            self.quspin = True
            self.Es, self.V = hamiltonian.eigh()
        else:
            self.quspin = False
            self.Es, self.V = np.linalg.eigh(hamiltonian)

    def time_evolve(self, start_time, end_time, time_staps, survival_prob=False):
        """Inputs:

            times = [start_time, end_time, time_steps].

            If survival_prob=True, this observable is added to the dictionary with
            name 's'."""

        unitary = exp_op(self.H, a=-1j, start=start_time, stop=end_time, num=time_staps, iterate=True)
        psi_t = unitary.dot(self.psi_0)
        t = unitary.grid
        obs_t = obs_vs_time(psi_t, t, self.Obs_dict)

        if survival_prob:
            psi_t = ED_state_vs_time(self.psi_0, self.Es, self.V, t, iterate=False)

            sp = np.zeros(len(t))
            for k in range(len(t)):
                a = np.dot(psi_t[:, k], self.psi_0)
                sp[k] = a * a.conj()
            obs_t['s'] = sp

        return obs_t

    def diag_ensemble(self):
        """Returns diagonal ensemble calculation of the time averaged observable expectation value and
        fluctuations"""
        de_fluctuations = dict()
        de_obs = dict()

        for key in self.Obs_dict.keys():

            obs = self.Obs_dict[key]
            c_init = np.square(np.dot(np.conjugate(np.transpose(self.V)), self.psi_0))
            obs_i = np.dot(np.transpose(self.V), np.dot(obs.toarray(), self.V))
            obs_i2 = np.square(obs_i)
            obs_i2 -= np.diag(np.diag(obs_i2))
            de_fluctuations[key] = np.dot(np.transpose(np.conjugate(c_init)), np.dot(obs_i2, c_init))
            de_obs[key] = np.dot(c_init, np.diag(obs_i))

        return de_obs, de_fluctuations

    def distributions(self):
        """Returns/Plots observable and initial state distributiions, including a fit to Lorentzian
        predicted by RMT model, and the associated Gamma value."""

    def level_spacing(self):
        """Returns/Plots the energy level spacing against Wigner-Dyson and Poissonian distributions"""


class TimeOperations:
    """Calculations that require the time dependence of an observable
    This could inherit from HamilOperations, though how to do this allowing for
    cases where obs_t is know, so the hamiltonian doesn't need to be passed?"""
    def __init__(self, t, obs_t):
        self.t = t
        self.obs_t = obs_t

    def decay_rate(self, fit_func='exp', guess=None):
        """If t, obs_t is supplied, then these are used, otherwise they are calculated from
        Processes.time_evolve"""

        if guess is None:
            guess = [0.1, 0.]

        if fit_func == 'exp':
            def fitting_function(x, a, b):
                return np.exp(- a * x) * (1 - b) + b
        elif fit_func == 'rabi':
            # guess here: c = np.sqrt(b_x ** 2 + b_z ** 2)
            #             d = 4 * ((b_z + e) ** 2 * b_x ** 2) / (b_x ** 2 + (b_z + e) ** 2) ** 2
            # b_x and b_z are system x and z fields respectively
            def fitting_function(x, a, b, c, d):
                return (((1 - d) + d * np.cos(2 * c * x)) - b) * np.exp(- a * x) + b
        else:
            raise Exception('Unrecognised fit_func')

        parameters, covariance = curve_fit(fitting_function, self.t, self.obs_t, guess)

        return parameters, fitting_function

    def time_fluctuatuions(self):
        """ Time_averaged fluctuations calculated from the decay to equilibrium, rather than the
        Diagonal ensemble. Not particularly useful for exact diagonalization, where DE result is
        obtainable, and more accurate, though may be useful for an approximate method such as
        tensor networks etc.

        If t, obs_t is supplied, then these are used, otherwise they are calculated from
        Processes.time_evolve

        Note that a more accurate method is to exclude the initial decay part from the time trace,
        as is done automatically when t and obs_t are not defined."""

        delta = 1

        return delta
