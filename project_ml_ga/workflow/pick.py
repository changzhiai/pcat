"""Pick candidate images from the results after GA. Compare new images with all old dft images and itself in case of repeated dft calculation"""
from ase.db import connect
from ase import Atom, Atoms
import os
import numpy as np
from multiprocessing import Pool
from ase.io import read, write
from ase.visualize import view
import argparse
import toml 

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="GA simulations drive by graph neural networks", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="arguments_predict.toml",
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

def get_sorted_dist_list(atoms, mic=False):
    """ Utility method used to calculate the sorted distance list
        describing the cluster in atoms. """
    numbers = atoms.numbers
    unique_types = set(numbers)
    pair_cor = dict()
    for n in unique_types:
        i_un = [i for i in range(len(atoms)) if atoms[i].number == n]
        d = []
        for i, n1 in enumerate(i_un):
            for n2 in i_un[i + 1:]:
                d.append(atoms.get_distance(n1, n2, mic))
        d.sort()
        pair_cor[n] = np.array(d)
    return pair_cor

def looks_like(a1, a2):
    """ Return Ture if structure a1 or a2 are similar. Otherwise, return False """
    n_top=len(a1)
    pair_cor_cum_diff=0.015,
    pair_cor_max=0.7
    dE=0.0000001
    mic=False
    if len(a1) != len(a2):
        return False
    if a1.get_chemical_formula() != a2.get_chemical_formula():
        return False
    # then we check the structure
    a1top = a1[-n_top:]
    a2top = a2[-n_top:]
    cum_diff, max_diff = compare_structure(a1top, a2top)
    return (cum_diff < pair_cor_cum_diff
            and max_diff < pair_cor_max)

def compare_structure(a1, a2):
    """ Private method for calculating the structural difference. """
    p1 = get_sorted_dist_list(a1, mic=False)
    p2 = get_sorted_dist_list(a2, mic=False)
    numbers = a1.numbers
    total_cum_diff = 0.
    max_diff = 0
    for n in p1.keys():
        cum_diff = 0.
        c1 = p1[n]
        c2 = p2[n]
        assert len(c1) == len(c2)
        if len(c1) == 0:
            continue
        t_size = np.sum(c1)
        d = np.abs(c1 - c2)
        cum_diff = np.sum(d)
        max_diff = np.max(d)
        ntype = float(sum([i == n for i in numbers]))
        total_cum_diff += cum_diff / t_size * ntype / float(len(numbers))
    return (total_cum_diff, max_diff)

def remove_repeated_struts_based_on_itself(images):
    """Remove all repeated atoms in one images, return unique structures"""
    unique_images = []
    unique_images.append(images[0])
    for atoms in images:
        flag = 1
        for atoms_unique in unique_images:
            if looks_like(atoms_unique, atoms):
                flag = 0
                # view([atoms_unique, atoms])
                break
        if flag == 1:
            unique_images.append(atoms)
    print('original images:', len(images))
    print('unique images:', len(unique_images))
    return unique_images

def compare(atoms_new):
        flag = 1
        for atoms_old in old_images:
            if looks_like(atoms_old, atoms_new):
                flag = 0
                break
        if flag == 1:
            return atoms_new
        else:
            return None

def remove_repeated_struts_based_on_old_images(new_images, old_images, cores=1):
    """Remove all repeated atoms in new images, which is compared to old images, return unique structures
    Parameters

    old images: as a reference and we do not change this images
    new_images: we remove repeated structures in new images that exist in old images
    """
    unique_traj = 'unique.traj'
    if os.path.exists(unique_traj):
        os.remove(unique_traj)
    print(f'new images: {len(new_images)}')
    if cores > 1:
        pool = Pool(cores) # Pool(os.cpu_count())
        with pool as p:
            images = p.map(compare, new_images)
    else:
        images = []
        for atoms in new_images:
            images.append(compare(atoms))
    images = [atoms for atoms in images if atoms != None]
    print('unique images in new images:', len(images))
    return images

def get_fittest_traj(pick, name='fittest_images.traj'):
    db = connect(pick.ga_db)
    fittest_images = []
    seen_dns = set()
    for row in db.select('relaxed=1'):
        atoms = row.toatoms()
        f_eff = row.raw_score
        dn = row.dominating_niche
        niches = row.data['niches']
        # Get the fittest individual with an effective fitness of 0 in each niche (without duplicates)
        if (f_eff == 0) and (dn not in seen_dns):
            seen_dns.add(dn)
            atoms = db.get_atoms(row.id, add_additional_information=True)
            niches = atoms.info['data']['niches'] # only the niches where this structures is the fittest
            scores = atoms.info['data']['raw_scores'] # print(scores[niches])
            fittest_images.append(atoms)
    write(name, fittest_images)
    return fittest_images

def get_last_generation(pick, name='last_gen_images.traj'):
    db = connect(pick.ga_db)
    keep = pick.pop_size # last 50 structures
    lens = len(db)
    start = lens - keep
    count = 0
    images = []
    for row in db.select():
        atoms = row.toatoms()
        if count >= start:
            atoms = db.get_atoms(row.id, add_additional_information=True)
            images.append(atoms)
            print(row.id)
        count += 1
    write(name, images)
    return images

def get_all_old_dft_final_images(pick):
    if pick.iteration == pick.start_iteration:
        all_old_dft_final_images = read(pick.init_final_dft_images, ':')
    else:
        all_old_dft_final_images = read(pick.all_old_dft_final_images, ':')
    return all_old_dft_final_images

def generate_run_config(params):
    with open(os.path.join(params['run_dir'], "run.toml"), 'w') as f:
        toml.dump(params, f)
    return params

if __name__ == '__main__':
    args = get_arguments()
    if args.cfg:
        with open(args.cfg, 'r') as f:
            params = toml.load(f)
    pick = Params(**params) 
        
    _ = generate_run_config(params)
    old_images = get_all_old_dft_final_images(pick)
    new_images = []
    if pick.pick_fittest:
        f_images = get_fittest_traj(pick, name=pick.fittest_images)
    else:
        f_images = []
    if pick.pick_last_gen:
        l_images = get_last_generation(pick, name=pick.last_gen_images)
    else:
        l_images = []
    # new_images = f_images + l_images 
    new_images = f_images
    images = remove_repeated_struts_based_on_old_images(new_images, old_images) 
    images = remove_repeated_struts_based_on_itself(images)
    write(pick.pick_candidates_traj, images)
    print('Pick done !')
