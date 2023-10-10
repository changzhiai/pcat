"""DFT calculation to get structures and energies"""
from ase.calculators.vasp import Vasp
from ase.io import read, write
import pickle
import os
import time
import sys
from ase.constraints import FixAtoms
from ase.db import connect
import argparse
import toml
import shutil
os.environ["ASE_VASP_VDW"] = '/home/energy/modules/software/VASP/vasp-potpaw-5.4'
os.environ["VASP_PP_PATH"] = '/home/energy/modules/software/VASP/vasp-potpaw-5.4'
os.environ["ASE_VASP_COMMAND"] = 'mpirun vasp_std'

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="GA simulations drive by graph neural networks", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="arguments.toml",
        help="Path to config file. e.g. 'arguments.toml'"
    )
    return parser.parse_args(arg_list)

class Params:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self.__dict__[key] = Params(**value)
            else:
                self.__dict__[key] = value

def get_next_iter_num(obj='OUTCAR', dir_name='./'):
    """Find the maximum of iteration i in OUTCAR_i, return next iteration number"""
    nums = [0] 
    list_of_files = os.listdir(dir_name)
    for file_name in list_of_files:
        splits = file_name.split('_')
        if len(splits)==2 and splits[0]==obj:
            num = int(splits[1])
            nums.append(num)
    max_iter = max(nums)
    next_iter = max_iter + 1
    return next_iter

def relax_structures(atoms, max_relax_times=3):
    """Relax structures and setup the maximun iteration times"""
    time.sleep(1.)
    next_iter_num = get_next_iter_num(obj='OUTCAR')
    relaxed = False
    i = 0
    while i < max_relax_times and not relaxed:
        nth = next_iter_num + i 
        try:
            e = atoms.get_potential_energy()
        except IndexError: # vasp error
            print(f'{nth} time vasp failed, trying to resume.')
            shutil.copy('CONTCAR', f'CONTCAR_{nth}')
            shutil.copy('OUTCAR', f'OUTCAR_{nth}')
        relaxed = atoms.get_calculator().read_relaxed()
        i += 1
    return atoms, e

def set_magmom(atoms):
    for atom in atoms:
        if atom.symbol == 'H':
            atom.magmom = 1
        elif atom.symbol == 'C':
            atom.magmom = 2
        elif atom.symbol == 'O':
            atom.magmom = 2
        elif atom.symbol == 'Pd':  # pure
            atom.magmom = 0
        elif atom.symbol == 'Sc': 
            atom.magmom = 1
        elif atom.symbol == 'Ti':
            atom.magmom = 2
        elif atom.symbol == 'V':
            atom.magmom = 3
        elif atom.symbol == 'Mn':
            atom.magmom = 5
        elif atom.symbol == 'Fe':
            atom.magmom = 4
        elif atom.symbol == 'Co':
            atom.magmom = 3
        elif atom.symbol == 'Ni':
            atom.magmom = 2
        elif atom.symbol == 'Cu':
            atom.magmom = 1
        elif atom.symbol == 'Zn':
            atom.magmom = 0
        elif atom.symbol == 'Y':
            atom.magmom = 1
        elif atom.symbol == 'Zr':
            atom.magmom = 2
        elif atom.symbol == 'Nb':
            atom.magmom = 3
        elif atom.symbol == 'Mo':
            atom.magmom = 4
        elif atom.symbol == 'Ru':
            atom.magmom = 4
        elif atom.symbol == 'Rh':
            atom.magmom = 3
        elif atom.symbol == 'Ag':
            atom.magmom = 1
        elif atom.symbol == 'Cr':
            atom.magmom = 4
        elif atom.symbol == 'Pt':
            atom.magmom = 2
        elif atom.symbol == 'Au':
            atom.magmom = 1
        elif atom.symbol == 'Cd':
            atom.magmom = 0
        elif atom.symbol == 'Ir':
            atom.magmom = 3
        elif atom.symbol == 'Tc':
            atom.magmom = 5
        elif atom.symbol == 'Hf':
            atom.magmom = 2
        elif atom.symbol == 'Ta':
            atom.magmom = 3
        elif atom.symbol == 'W':
            atom.magmom = 4
        elif atom.symbol == 'Re':
            atom.magmom = 5
        elif atom.symbol == 'Os':
            atom.magmom = 4
    return atoms

def generate_run_config(params):
    with open(os.path.join(params['run_dir'], "run.toml"), 'w') as f:
        toml.dump(params, f)
    return params

if __name__ == '__main__':
    args = get_arguments()
    if args.cfg:
        with open(args.cfg, 'r') as f:
            params = toml.load(f)
    dft = Params(**params) 
    
    _ = generate_run_config(params)
    atoms_start = read(dft.start_traj) # read structure
    atoms = atoms_start.copy()
    atoms.set_constraint(FixAtoms(mask=[atom.z<=dft.fix_z_max for atom in atoms]))
    atoms = set_magmom(atoms)
    relax = params['relax']
    calc = Vasp(**relax)
    atoms.set_pbc([True,True,True])
    atoms.set_calculator(calc)
    try:
        atoms, energy = relax_structures(atoms, max_relax_times=dft.max_relax_times)
        write(dft.final_traj, atoms)
    except:
        print('Relaxation failed')
        write(dft.final_traj, atoms)
    
    try:
        for f in dft.remove_files:
            os.remove(f)
    except:
        pass
    print('dft done !')
