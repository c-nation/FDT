"""Author: Charlie Nation

Run to load/save and plot the fluctuation-dissipation theorem for a random matrix model
"""

import numpy as np
import FDT
import load_save_FDT as LSv
import Hamiltonians
import os
import matplotlib.pylab as plt
import seaborn as sns


def run_rm_instance(n, coupling, ave, start_t, end_t, t_steps, file_name, plot=False):
    """Outputs data required for FDT plots. If data does not exist, then it is calculated and saved, if
     it exists, this data is loaded"""
    # initialize Hamiltonian
    h_instance = Hamiltonians.RandomMatrix(n, coupling)

    # get save names
    parameter_group, init_group = h_instance.save_groups()
    print(init_group)
    parameter_group += f'averages={ave},start={start_t},end={end_t},steps={t_steps}/'
    pathname = os.getcwd()
    pathname = str.replace(pathname, '\\', '/')
    file_name = pathname + file_name
    # check if data exists (checker = True if exists)
    lsv_instance = LSv.LoadSave(file_name, parameter_group + init_group)
    checker = lsv_instance.check_hdf5()

    rmt_keys = ['odd_t', 'sym_t', 'odd_gamma', 'sym_gamma', 'odd_delta', 'sym_delta', 'odd_inf', 'sym_inf']
    times = np.linspace(start_t, end_t, t_steps)
    if not checker:

        # set up blank arrays to average over realisations of the random Hamiltonian
        obs_t_o = np.zeros([t_steps, ave])
        obs_t_s = np.zeros([t_steps, ave])
        gamma_o = np.zeros(ave)
        gamma_s = np.zeros(ave)
        o_inf_o = np.zeros(ave)
        o_inf_s = np.zeros(ave)
        delta_o = np.zeros(ave)
        delta_s = np.zeros(ave)
        for i in range(ave):
            print(f'Running average = {i + 1} out of {ave}')
            # do calculations
            h = h_instance.hamiltonian()
            o_odd = h_instance.observable('odd')
            o_sym = h_instance.observable('sym')
            init = h_instance.choose_initial_state()
            ops = FDT.HamilOperations(h, dict(odd=o_odd, sym=o_sym), init)
            obs_t = ops.time_evolve(start_t, end_t, t_steps)
            o_inf, delta = ops.diag_ensemble()
            fit_odd, _ = FDT.TimeOperations(times, obs_t['odd']).decay_rate(guess=[np.pi * coupling ** 2, 0.5])
            fit_sym, exp_func = FDT.TimeOperations(times, obs_t['sym']).decay_rate(guess=[np.pi * coupling ** 2, 0.])

            # set elements to average
            o_inf_s[i] = o_inf['sym']
            o_inf_o[i] = o_inf['odd']
            delta_s[i] = delta['sym']
            delta_o[i] = delta['odd']
            obs_t_o[:, i] = obs_t['odd']
            obs_t_s[:, i] = obs_t['sym']
            gamma_o[i] = fit_odd[0] / 2.
            gamma_s[i] = fit_sym[0] / 2.

        # take averages
        o_inf = dict(odd=np.mean(o_inf_o), sym=np.mean(o_inf_s))
        delta = dict(odd=np.mean(delta_o), sym=np.mean(delta_s))
        obs_t = dict(odd=np.mean(obs_t_o, 1), sym=np.mean(obs_t_s, 1))
        gamma = dict(odd=np.mean(gamma_o), sym=np.mean(gamma_s))

        # save
        save_dict = dict([('odd_t', obs_t['odd']), ('sym_t', obs_t['sym']),
                          ('odd_gamma', gamma['odd']), ('sym_gamma', gamma['sym']),
                          ('odd_delta', delta['odd']), ('sym_delta', delta['sym']),
                          ('odd_inf', o_inf['odd']), ('sym_inf', o_inf['sym'])])
        lsv_instance.save_hdf5(save_dict)

    else:
        save_dict = lsv_instance.load_hdf5(rmt_keys)

    if plot:
        fit_odd, _ = FDT.TimeOperations(times, save_dict['odd_t']).decay_rate(guess=[np.pi * coupling ** 2, 0.5])
        fit_sym, exp_func = FDT.TimeOperations(times, save_dict['sym_t']).decay_rate(guess=[np.pi * coupling ** 2, 0.])
        fig = plt.figure(figsize=(10., 5.))
        ax = fig.add_subplot(121)
        ax.plot(times, save_dict['odd_t'])
        ax.plot(times, exp_func(times, fit_odd[0], fit_odd[1]))
        ax.plot(times,  save_dict['odd_inf'] * np.ones(len(times)))
        ax.set_ylabel('$\\langle O_{odd}(t) \\rangle$')
        ax.set_xlabel('$t$')
        ax2 = fig.add_subplot(122)
        ax2.plot(times, save_dict['sym_t'], label='ED')
        ax2.plot(times, exp_func(times, fit_sym[0], fit_sym[1]), label='Fit to RMT')
        ax2.plot(times, save_dict['sym_inf'] * np.ones(len(times)), label='DE $\\overline{\\langle O(t) \\rangle}$')
        ax2.set_ylabel('$\\langle O_{sym}(t)\\rangle$')
        ax2.set_xlabel('$t$')
        ax2.legend()
        fig.tight_layout()

    return save_dict


def rm_fdt(ns, coupling, ave, start_t, end_t, t_steps, file_name):
    """Calls data and plots FDT. Will run ad save data by call to run_rm_instance if data isn't already saved"""

    def fdt(dos, gamma):
        return 1. / (4. * np.pi * dos * gamma)
    miny = 100
    maxy = 1e-3
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colours = iter(sns.color_palette('colorblind', len(ns) + 2))
    for n in ns:
        c = next(colours)
        data_dict = run_rm_instance(n, coupling, ave, start_t, end_t, t_steps, file_name)
        ax.loglog(fdt(n, data_dict['odd_gamma']), data_dict['odd_delta'], 's', c=c, label=f'$N={n}$')
        ax.loglog(fdt(n, data_dict['sym_gamma']), data_dict['sym_delta'], 'o', c=c)
        miny = min(miny, data_dict['odd_delta'])
        maxy = max(maxy, data_dict['odd_delta'])
    theory_line = np.linspace(0.1 * miny, 10. * maxy, 100)
    c = next(colours)
    ax.loglog(theory_line, theory_line, '--', c=c, label='RMT $O_{sym}$')
    c = next(colours)
    ax.loglog(theory_line, .25 * theory_line, '--', c=c, label='RMT $O_{odd}$')
    ax.legend()
    fig.tight_layout()


if __name__ == '__main__':
    # define parameters
    Ns = [500, 1000, 1500, 2000, 2500]
    g = 0.05

    averages = 1

    start = 0
    end = 1500
    steps = 300

    filename = '/Data/RMT_FDT.hdf5'

    rm_fdt(Ns, g, averages, start, end, steps, filename)

    plt.show()
