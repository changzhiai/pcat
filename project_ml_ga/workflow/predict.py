"""Predict the relaxed energy via relaxing structures with ensemble calculator"""
from ase import units
from ase.io import read, write, Trajectory
import numpy as np
import torch
import sys
import glob
import toml
import argparse
from pathlib import Path
import logging
from PaiNN.data import AseDataset, collate_atomsdata
from PaiNN.model import PainnModel
from PaiNN.calculator import MLCalculator, EnsembleCalculator
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.db import connect
import numpy as np
import pandas as pd
import os
import json
import copy
import matplotlib
# matplotlib.use('TkAgg')

class EnergyObservor:
    def __init__(self, atoms):
        self.atoms = atoms
        print("Energy observor")
  
    def __call__(self, threshold=0):
        energy = self.atoms.get_potential_energy()
        forces = self.atoms.get_forces()
        print(f'energy: {energy}')
        if energy > threshold or energy<-10000:
            raise ValueError('energy is too large or too low')

def plot_fitting(dft_energies, nnp_energies, fig_name='prediction.png'):
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    rmse = mean_squared_error(dft_energies, nnp_energies, squared=False)
    m, b = np.polyfit(dft_energies, nnp_energies, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.array(dft_energies), m * np.array(dft_energies) + b)
    plt.plot(dft_energies, nnp_energies, 'ob',mfc='none')
    plt.xlabel("E_DFT (eV/atom)")
    plt.ylabel("E_NNP (eV/atom)")
    plt.title("fit using {} data points.".format(len(nnp_energies)))
    X = dft_energies
    Y = nnp_energies
    if np.size(X) and np.size(Y) != 0:
        e_range = max(np.append(X, Y)) - min(np.append(X, Y))
        rmin = min(np.append(X, Y)) - 0.05 * e_range
        rmax = max(np.append(X, Y)) + 0.05 * e_range
    else:
        rmin = -10
        rmax = 10
    linear_fit = np.arange(rmin - 10, rmax + 10, 1)
    ax.plot(linear_fit, linear_fit, 'r')
    ax.axis([rmin, rmax, rmin, rmax])
    ax.text(0.95,
            0.01,
            "RMSE = {:.3f} meV/atom".format(rmse*1000),
            verticalalignment='bottom',
            horizontalalignment='right',
            transform=ax.transAxes,
            fontsize=12)
    ticks = False
    if ticks:
        plt.xticks(np.arange(min(X), max(Y), 0.2))
        plt.yticks(np.arange(min(X), max(Y), 0.2))
    fig.savefig(fig_name)

def setup_seed(seed):
     torch.manual_seed(seed)
     if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="MD simulations drive by graph neural networks", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--init_traj",
        type=str,
        help="Path to start configurations",
    )
    parser.add_argument(
        "--start_indice",
        type=int,
        help="Indice of the start configuration",
    )
    parser.add_argument(
        "--load_model",
        type=str,
        help="Where to find the models",
    )
    parser.add_argument(
        "--time_step",
        type=float,
        default=0.5,
        help="Time step of MD simulation",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5000000,
        help="Maximum steps of MD",
    )
    parser.add_argument(
        "--min_steps",
        type=int,
        default=100000,
        help="Minimum steps of MD, raise error if not reached",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=350.0,
        help="Maximum time steps of MD",
    )
    parser.add_argument(
        "--fix_under",
        type=float,
        default=5.9,
        help="Fix atoms under the specified value",
    )
    parser.add_argument(
        "--dump_step",
        type=int,
        default=100,
        help="Fix atoms under the specified value",
    )
    parser.add_argument(
        "--print_step",
        type=int,
        default=1,
        help="Fix atoms under the specified value",
    )
    parser.add_argument(
        "--num_uncertain",
        type=int,
        default=1000,
        help="Stop MD when too many structures with large uncertainty are collected",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="Random seed for this run",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help="Set which device to use for running MD e.g. 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="arguments.toml",
        help="Path to config file. e.g. 'arguments.toml'"
    )

    return parser.parse_args(arg_list)

def update_namespace(ns, d):
    for k, v in d.items():
        ns.__dict__[k] = v

class CallsCounter:
    def __init__(self, func):
        self.calls = 0
        self.func = func
    def __call__(self, *args, **kwargs):
        self.calls += 1
        self.func(*args, **kwargs)

def generate_run_config(params):
    with open(os.path.join(params['run_dir'], "run.toml"), 'w') as f:
        toml.dump(params, f)
    return params

def read_df(args):
    df = pd.read_csv(args.save_csv)
    df = df.loc[df['converged']==True]
    dft_energies = df['dft_energies']
    nnp_energies = df['nnp_energies']
    fig_name = args.fig_name
    return dft_energies, nnp_energies, fig_name

def get_init_bad_structs(args):
    df = pd.read_csv(args.save_csv)
    df = df[(df['dft_energies']-df['nnp_energies']>0.02) | (df['converged']==False)]
    images = read(args.all_init_dft_dir, ':')
    lists = list(df.index.values)
    images_bad = [images[i] for i in lists]
    print(len(images_bad))
    print([images[i].get_chemical_formula() for i in lists])
    write('images_bad.traj', images_bad)
    return df

def main(read_csv=False):
    args = get_arguments()
    if args.cfg:
        with open(args.cfg, 'r') as f:
            params = toml.load(f)
    update_namespace(args, params)
    if read_csv:
        print('Reading csv')
        # get_init_bad_structs(args)
        return read_df(args)

    _ = generate_run_config(params)
    setup_seed(args.random_seed)
    # set logger
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)
    runHandler = logging.FileHandler(args.predict_log, mode='w')
    runHandler.setLevel(logging.DEBUG)
    runHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    errorHandler = logging.FileHandler(args.error_log, mode='w')
    errorHandler.setLevel(logging.WARNING)
    errorHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    logger.addHandler(runHandler)
    logger.addHandler(errorHandler)
    logger.addHandler(logging.StreamHandler())
    logger.warning = CallsCounter(logger.warning)
    logger.info = CallsCounter(logger.info)
    # load models
    models = []
    for each in args.load_models:
        if args.device == 'cpu':
            state_dict = torch.load(each, map_location=torch.device('cpu'))
        elif args.device == 'cuda':
            state_dict = torch.load(each)
        else:
            raise ValueError(f'set {args.device} error')
        model = PainnModel(
            num_interactions=state_dict["num_layer"],
            hidden_state_size=state_dict["node_size"],
            cutoff=state_dict["cutoff"],
        )
        model.to(args.device)
        model.load_state_dict(state_dict["model"])
        models.append(model)
    encalc = EnsembleCalculator(models)

    save_db = args.save_db
    if os.path.exists(save_db):
        os.remove(save_db)
    db = connect(save_db)
    if isinstance(args.init_traj, list):
        images_tot = []
        n = len(args.init_traj)
        for data in args.init_traj:
            images = read(data, ':')
            images_tot += images
        tot_traj = f'dft_{args.system}_adss_r{n}_init_total.traj'
        write(tot_traj, images_tot)
        args.init_traj = tot_traj
    images = read(args.all_init_dft_dir, ':')
    count = 0
    formulas = []
    dft_energies = []
    nnp_energies = []
    lens = []
    steps = []
    convergeds = []
    properties = []
    E_per_atoms = args.E_per_atoms
    for atoms in images:
        single_point=False
        l = len(atoms)
        lens.append(l)
        formula = atoms.get_chemical_formula(mode='hill')
        formulas.append(formula)
        mask = [atom.z < 2.0 for atom in atoms]
        atoms.set_constraint(FixAtoms(mask=mask))
        atoms.calc = encalc
        converged = True
        if not single_point:
            atoms_old = copy.deepcopy(atoms)
            try:
                opt = BFGS(atoms, trajectory=f'opti_{count}.traj', logfile='bfgs.log')
                eo = EnergyObservor(opt.atoms)
                opt.attach(eo)
                converged = opt.run(fmax=args.fmax, steps=args.max_steps)
            except:
                atoms = atoms_old
                converged = False
        convergeds.append(converged)
        logger.info('finishing: {}'.format(count))
        E_nnp = atoms.get_potential_energy()
        results = atoms.calc.results
        properties.append(results)
        # logger.info(results)
        energy_var = results['ensemble']['energy_var']
        step = opt.get_number_of_steps()
        steps.append(step)
        if E_per_atoms:
            E_nnp = E_nnp/len(atoms)
        nnp_energies.append(E_nnp)
        atoms.calc = None
        db.write(atoms, E_nnp=E_nnp, lens=l, step=step, converged=converged)
        count += 1
    if isinstance(args.final_traj, list):
        images_tot = []
        n = len(args.final_traj)
        for data in args.final_traj:
            images = read(data, ':')
            images_tot += images
        tot_traj = f'dft_{args.system}_adss_r{n}_final_total.traj'
        write(tot_traj, images_tot)
        args.final_traj = tot_traj
    images_dft = read(args.all_final_dft_dir, ':')
    for atoms in images_dft:
        e = atoms.get_potential_energy()
        if E_per_atoms:
            e = e/len(atoms)
        dft_energies.append(e)

    tuples = {
            'formula': formulas,
            'lens': lens,
            'dft_energies': dft_energies,
            'nnp_energies':nnp_energies,
            'converged': convergeds,
            'properties': properties,
            }
    df = pd.DataFrame(tuples)
    df.to_csv(args.save_csv)
    df = df.loc[df['converged']==True]
    return df['dft_energies'], df['nnp_energies'], args.fig_name

if __name__ == "__main__":
    dft_energies, nnp_energies, fig_name = main(read_csv=False)
    plot_fitting(dft_energies, nnp_energies, fig_name=fig_name)
    print('Predict done!')
