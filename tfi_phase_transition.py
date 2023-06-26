"""Tuning through the phase transition of the transverse field Ising model.

This example uses DMRG to find the ground state of the transverse field Ising model when tuning
through the phase transition by changing the field `g`.
It plots a few observables in the end.
"""
# Copyright 2018-2023 TeNPy Developers, GNU GPLv3

import numpy as np

from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
import matplotlib.pyplot as plt


def run(gs):
    print("running for gs = ", gs)
    L = 2
    model_params = dict(L=L, J=1., g=gs[0], bc_MPS='infinite', conserve=None)
    chi = 256
    dmrg_params = {
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-10,
            'trunc_cut': None
        },
        'update_env': 5,
        'start_env': 5,
        'max_E_err': 0.0001,
        'max_S_err': 0.0001,
        'max_sweeps': 1<<20,  # NOTE: this is not enough to fully converge at the critical point!
        'mixer': False
    }

    M = TFIChain(model_params)
    psi = MPS.from_product_state(M.lat.mps_sites(), (["up", "down"] * L)[:L], M.lat.bc_MPS)

    engine = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    np.set_printoptions(linewidth=120)

    corr_length = []
    fidelity = []
    Sz = []
    E = []
    S = []
    SxSx = []
    Sx = []
    old_psi = None
    for g in gs:
        print("-" * 80)
        print("g =", g)
        print("-" * 80)
        model_params['g'] = g
        M = TFIChain(model_params)
        engine.init_env(model=M)  # (re)initialize DMRG environment with new model
        # this uses the result from the previous DMRG as first initial guess
        E0, psi = engine.run()
        E.append(E0)
        # psi is modified by engine.run() and now represents the ground state for the current `g`.
        S.append(psi.entanglement_entropy()[0])
        corr_length.append(psi.correlation_length(tol_ev0=1.e-3))
        print("corr. length", corr_length[-1])
        Sz.append(psi.expectation_value('Sigmaz'))
        print("<Sigmaz>", Sz[-1])
        Sx.append(psi.expectation_value('Sigmax'))
        print("<Sigmax>", Sx[-1])
        SxSx.append(psi.correlation_function("Sigmax", "Sigmax", [0], 20)[0, :])
        print("<Sigmax_0 Sigmax_i>", SxSx[-1])
        if old_psi is not None:
            fidelity.append(np.abs(old_psi.overlap(psi)))
            print("fidelity", fidelity[-1])
        old_psi = psi.copy()
        dmrg_params['start_env'] = 0  # (some of) the parameters are read out again
    results = {
        'model_params': model_params,
        'dmrg_params': dmrg_params,
        'gs': gs,
        'corr_length': np.array(corr_length),
        'fidelity': np.array(fidelity),
        'Sz': np.array(Sz),
        'Sx': np.array(Sx),
        'E': np.array(E),
        'S': np.array(S),
        'SxSx': np.array(SxSx),
        'eval_transfermatrix': np.exp(-1. / np.array(corr_length)),
    }
    return results


def plot(results, filename):
    fig, axes = plt.subplots(2, 3, gridspec_kw=dict(wspace=0.3, hspace=0.3), figsize=(15, 10))
    gs = np.array(results['gs'])
    for ax, key in zip(axes.flatten(),
                       ['E', 'S', 'SxSx', 'Sx', 'eval_transfermatrix', 'fidelity']):
        if key == 'fidelity':
            ax.plot(0.5 * (gs[1:] + gs[:-1]), np.abs(results[key]), 'r.-')
        elif key == 'SxSx':
            cm = plt.cm.get_cmap("viridis")
            SxSx = results[key]
            max_j = SxSx.shape[1]
            for j in range(1, max_j):
                label = "j= {j:d}".format(j=j) if j in [1, max_j - 1] else None
                ax.plot(gs, SxSx[:, j], label=label, color=cm(j / max_j))
            ax.legend()
        else:
            ax.plot(gs, results[key], 'r.-')
        ax.set_xlabel('$g/J$')
        ax.set_ylabel(key)
        ax.set_xticks(np.arange(min(gs), max(gs)+0.1, 0.1))
    plt.savefig(filename)
    print("saved to " + filename)


def plot_separate(results, filename):
    gs = np.array(results['gs'])
    for key, ylabel in zip(['E', 'S', 'SxSx', 'Sx', 'eval_transfermatrix', 'corr_length', 'fidelity'], \
                          ['$E$', r'$\mathcal{S}_1$ (nats)', r'$\langle \sigma^z_{0} \sigma^z_{19} \rangle$', r'$\sigma^{z}_{0}$', r'$t_2/t_1$', r'$\xi$', r'$|\langle \psi(B) | \psi(B+\delta B) \rangle|$']):
        fig, ax = plt.subplots(1, 1, figsize=(16/2.54, 8/2.54))
        if key == 'fidelity':
            ax.plot(0.5 * (gs[1:] + gs[:-1]), np.abs(results[key]),)
        elif key == 'SxSx':
            cm = plt.cm.get_cmap("viridis")
            SxSx = results[key]
            max_j = SxSx.shape[1]
            for j in range(1, max_j):
                label = "j= {j:d}".format(j=j) if j in [1, max_j - 1] else None
                ax.plot(gs, SxSx[:, j], label=label, color=cm(j / max_j))
            ax.legend()
        else:
            ax.plot(gs, results[key], color='C0')
            
        ax.set_xlabel('$B/J_0$')
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(min(gs), max(gs)+0.1, 0.1))
        plt.tight_layout()
        plt.savefig(filename[:-4] + "_%s" % (key) + filename[-4:])
        print("saved to " + filename)
    

if __name__ == "__main__":
    filename = 'tfi_phase_transition_ferro.pkl'
    import pickle
    import os.path
    if not os.path.exists(filename):
        gs = sorted(set(np.linspace(0.5, 1.5, 121)).union(set(np.linspace(0.95, 1.05, 121))))[::-1]
        results = run(gs)
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
    else:
        print("just load the data")
        with open(filename, 'rb') as f:
            results = pickle.load(f)
    plot_separate(results, filename[:-4] + '.pdf')
    plot_separate(results, filename[:-4] + '.svg')
