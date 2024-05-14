"""Collect data from vasp calculation and then used for active learning"""
from contextlib import contextmanager
from ase.io import read, write
import numpy as np
import os
from ase.db import connect
import re
import shutil
import pandas as pd
from ase.neighborlist import NeighborList
import argparse
import toml
import time

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

def check_if_converged():
    search_term1 = 'reached required accuracy'
    search_term2 = 'aborting loop because EDIFF is reached'
    flag = 0
    for line in open('OUTCAR', 'r'): # check if converge
        if re.search(search_term1, line) or re.search(search_term2, line):
            flag = 1
    converged = None
    if flag == 0:
        converged=False
    else:
        converged=True
    return converged

@contextmanager
def cd(newdir, new=False):
    prevdir = os.getcwd()
    if new:
        try:
            os.makedirs(newdir)
        except OSError:
            pass
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

def collect_all_outcar_old(step):
    obj = 'OUTCAR'
    dir_name = './'
    images_in_dir = []
    try:
        list_of_files = os.listdir(dir_name)
        nums = []
        for file_name in list_of_files:
            splits = file_name.split('_')
            if len(splits)==2 and splits[0]==obj:
                try:
                    images = read('OUTCAR', f'::{step}')
                    images_in_dir += images
                    num = int(splits[1])
                    nums.append(num)
                except:
                    pass
        max_iter = max(nums)
        next_iter = max_iter + 1
    except:
        if os.path.exists(obj):
            next_iter = 1
        else:
            raise ValueError('Could not find OUTCAR*')
    return images_in_dir, next_iter

def check_distortion(atoms, cutoff=1.2):
    """Check if there are some distortion structures, such as, H2O, H2"""
    cutoff = cutoff # within 1.2 A
    nl = NeighborList(cutoffs=[cutoff / 2.] * len(atoms),
                            self_interaction=True,
                            bothways=True,
                            skin=0.)
    nl.update(atoms)
    for atom in atoms:
        if atom.symbol == 'O':
            n1 = atom.index
            indices, _ = nl.get_neighbors(n1)
            indices = list(set(indices))
            indices.remove(n1)
            if len(indices) != 0:
                syms = atoms[indices].get_chemical_symbols()
                num = syms.count('H')
                if num >= 2:
                    reason = 'H2O_exists'
                    return True, reason
        elif atom.symbol == 'H':
            n1 = atom.index
            indices, _ = nl.get_neighbors(n1)
            indices = list(set(indices))
            indices.remove(n1)
            if len(indices) != 0:
                syms = atoms[indices].get_chemical_symbols()
                num = syms.count('H')
                if num >= 1:
                    reason = 'H2_exists'
                    return True, reason
    return False, 'undistortion'

def get_all_spc_dft_images(collect, save=True):
    """Get the images (with step and undistortion) for train set"""
    init_spc_dft_images = read(collect.init_spc_dft_images, ':')
    all_spc_dft_images = init_spc_dft_images
    for i in range(collect.start_iteration, collect.iteration+1):
        collect_dir = f'{collect.root_dir}/iter_{i}/{collect.folder}'
        iteri_spc_dft_images = read(f'{collect_dir}/dft_{collect.system}_adss_r{i}_spc_undistor.traj', ':')
        all_spc_dft_images += iteri_spc_dft_images
    if save:
        write(collect.all_spc_dft_images, all_spc_dft_images)
    return all_spc_dft_images

def get_all_init_dft_images(collect, save=True):
    """Get all initial images"""
    init_init_dft_images = read(collect.init_init_dft_images, ':')
    all_init_dft_images = init_init_dft_images
    for i in range(collect.start_iteration, collect.iteration+1):
        collect_dir = f'{collect.root_dir}/iter_{i}/{collect.folder}'
        iteri_init_dft_images = read(f'{collect_dir}/dft_{collect.system}_adss_r{i}_init_tot.traj', ':')
        all_init_dft_images += iteri_init_dft_images
    if save:
        write(collect.all_init_dft_images, all_init_dft_images)
    return all_init_dft_images

def get_all_final_dft_images(collect, save=True):
    """Get all final images"""
    init_final_dft_images = read(collect.init_final_dft_images, ':')
    all_final_dft_images = init_final_dft_images
    for i in range(collect.start_iteration, collect.iteration+1):
        collect_dir = f'{collect.root_dir}/iter_{i}/{collect.folder}'
        iteri_final_dft_images = read(f'{collect_dir}/dft_{collect.system}_adss_r{i}_final_tot.traj', ':')
        all_final_dft_images += iteri_final_dft_images
    if save:
        write(collect.all_final_dft_images, all_final_dft_images)
    return all_final_dft_images

def generate_run_config(params):
    with open(os.path.join(params['run_dir'], "run.toml"), 'w') as f:
        toml.dump(params, f)
    return params

if __name__ == '__main__':
    args = get_arguments()
    if args.cfg:
        with open(args.cfg, 'r') as f:
            params = toml.load(f)
    collect = Params(**params)
    _ = generate_run_config(params)
    iter = f'r{collect.iteration}'
    step = collect.step
    images_dir = collect.ga_candidates_traj
    images = read(images_dir, ':')
    
    images_init_undistor = []
    images_init_distor = []
    images_spc = []
    images_final = []
    images_spc_undistor = []
    images_final_undistor = []
    images_spc_distor = []
    images_final_distor = []
    uniqueids, distortions, reasons, convergeds, paths = [], [], [], [], []
    for row_id, atoms in enumerate(images):
        # if row_id <= 0:
            # continue
        name = atoms.get_chemical_formula(mode='metal')
        job_id = str(row_id) + '_' + name
        with cd(f'{collect.dft_dir}/{job_id}/', new=False):
            uniqueid = name + '_' + str(row_id) + '_' + iter
            uniqueids.append(uniqueid)
            converged = check_if_converged()
            convergeds.append(converged)
            path = os.getcwd()
            path = path.replace('/home/energy/changai', '~')
            paths.append(path)
            atoms_final = read('OUTCAR')
            distortion, reason = check_distortion(atoms_final)
            distortions.append(distortion)
            reasons.append(reason)
            try:
                # write steps
                images_in_path = read('OUTCAR', f'::{step}')
                images_in_path_old, _ = collect_all_outcar_old(step)
                images_spc += images_in_path 
                images_spc += images_in_path_old
                if not distortion:
                    images_spc_undistor += images_in_path 
                    images_spc_undistor += images_in_path_old
                    images_init_undistor += [atoms]
                else:
                    images_spc_distor += images_in_path 
                    images_spc_distor += images_in_path_old
                    images_init_distor += [atoms]
                # write final to db
                db_opt = connect(collect.dft_final_db)
                db_opt.write(atoms_final, start_id=int(row_id), converged=converged,
                        uniqueid = uniqueid, path=path, distortion=distortion, reason=reason)
                images_final += [atoms_final]
                if not distortion:
                    images_final_undistor += [atoms_final]
                else:
                    images_final_distor += [atoms_final]
                print(f'saving: {uniqueid}')
            except:
                print(uniqueid, ' fail')
                print(path)     
            
            print(job_id)
            # break
    shutil.copy(collect.dft_final_db, f'{collect.dft_dir}/')
    tuples = {
            'uniqueids': uniqueids,
            'distortions': distortions,
            'reasons': reasons,
            'convergeds': convergeds,
            'paths': paths,
            }
    df = pd.DataFrame(tuples)
    df.to_csv(collect.csv)
    write(collect.images_spc, images_spc)
    write(collect.images_spc_undistor, images_spc_undistor)
    write(collect.images_spc_distor, images_spc_distor)
    write(collect.images_final, images_final)
    write(collect.images_final_undistor, images_final_undistor)
    write(collect.images_final_distor, images_final_distor)
    shutil.copy(images_dir, collect.images_init)
    write(collect.images_init_undistor, images_init_undistor)
    write(collect.images_init_distor, images_init_distor)
    time.sleep(10)
    _ = get_all_spc_dft_images(collect, save=True)
    _ = get_all_init_dft_images(collect, save=True)
    _ = get_all_final_dft_images(collect, save=True)
    time.sleep(10)
    print('Collect done')


