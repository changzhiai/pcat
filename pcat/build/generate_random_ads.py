# -*- coding: utf-8 -*-
"""

@author: changai
"""

from ase import Atom, Atoms
from ase.visualize import view
import numpy as np
from ase.io import write, read
import os
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.db import connect
from ase.constraints import UnitCellFilter
from ase.optimize import BFGS
import json
import sys
from ase.visualize import view
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.build import surface
import random
from scipy.spatial import Delaunay
from ase.neighborlist import NeighborList
from ase.geometry import find_mic
import matplotlib as mpl
# mpl.use('TkAgg')

def build_system(a = 4.138):
    bulk_PdH = Atoms(
        "Pd4H4",
        scaled_positions=[
            (0.0, 0.0, 0.0),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
            (0.5, 0.5, 0.5),
            (0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.0, 0.5),
        ],
        cell=[a, a, a],
        pbc=True,
    )
    slab = surface(bulk_PdH, (1, 1, 1), 4)
    slab.center(vacuum=10, axis=2)
    system = slab.copy()
    system.positions[:,2] = system.positions[:,2]-10.
    return system

def add_X(system):
    """Add customized vaccum layers, only for this system"""
    thickness = 2.389
    z_max_cell = system.cell[2, 2]
    top_metal_layer = system[
        [atom.index for atom in system if atom.z > 7 and atom.z < 8]
    ]
    for atom in top_metal_layer:
        atom.symbol = "X"
    while max(top_metal_layer.positions[:, 2]) + thickness < z_max_cell:
            top_metal_layer.positions[:, 2] = top_metal_layer.positions[:, 2] + thickness
            system += top_metal_layer
    return system

def rm_1X(atoms):
    """Remove the first X layer near to the surface"""
    del atoms[[atom.index for atom in atoms if atom.z > 9.555 and atom.z < 9.557]]
    return atoms

def get_top_sites(points):
    """Get ontop sites"""
    top_sites = list(points)
    return top_sites

def get_hollow_sites(tri_points):
    """Get ontop sites"""
    hollow_sites = []
    for tri_point in tri_points:
        average_site = np.average(tri_point, axis=0)
        hollow_sites.append(average_site)
    return hollow_sites

def remove_atoms_outside_cell(atoms):
    """For all cell"""
    a2 = atoms.copy()
    a2.wrap()
    da = atoms.positions - a2.positions
    indices = []
    for i in range(len(atoms)):
        # print(atoms[i].index, a2[i].index)
        if all(abs(atoms[i].position - a2[i].position) < 0.0001):
            indices.append(atoms[i].index)
    atoms = atoms[indices]
    return atoms

def get_extended_points(atoms):
    """Get extended triangle points, return a numpy array"""
    cutoff = 6.0
    nl = NeighborList(cutoffs=[cutoff / 2.] * len(atoms),
                            self_interaction=True,
                            bothways=True,
                            skin=0.)
    nl.update(atoms)
    all_atoms_pos = []
    for atom in atoms:
        indices, offsets = nl.get_neighbors(atom.index)
        for i, offset in zip(indices, offsets):
            pos = atoms.positions[i] + np.dot(offset, atoms.get_cell())
            all_atoms_pos.append(tuple(pos))
    unique_atoms_pos = set(all_atoms_pos)
    unique_atoms_pos = np.vstack([np.array(unique_atom_pos) for unique_atom_pos in unique_atoms_pos])
    return unique_atoms_pos

def get_sites(atoms):
    """Get adsorption sites for PdH(111)"""
    atoms = atoms[[atom.index for atom in atoms if atom.z > 6.5 and atom.symbol != 'H' and atom.symbol != 'X']]
    assert len(atoms) == 4
    extended_points = get_extended_points(atoms)
    points = extended_points
    tri = Delaunay(points[:,:2])
    simplices = tri.simplices
    tri_points = points[simplices]
    sites = {}
    sites['top'] = get_top_sites(points)
    sites['hollow'] = get_hollow_sites(tri_points)
    # sites['hollow'] = get_hollow_sites(extended_points)
    return sites

def add_full_adsorbates(atoms, sites, adsorbate='X'):
    """Add adsorbates on given sites"""
    distance = 2.0
    for site in sites:
        if adsorbate == 'X':
            ads = Atoms([Atom('X', (0, 0, 0))])
            ads.translate(site + (0., 0., distance))
        atoms.extend(ads)
    atoms = remove_atoms_outside_cell(atoms)
    return atoms

def remove_adsorbates_vertical_too_close(atoms):
    """Remove adsorbates as the distance of adsorbate and H is too close, 
    for example, their distance is within 0.9 such that H would react with adsorbates
    """
    cutoff = 0.9
    nl = NeighborList(cutoffs=[cutoff / 2.] * len(atoms),
                            self_interaction=True,
                            bothways=True,
                            skin=0.)
    nl.update(atoms)
    index_adsorbates = [atom.index for atom in atoms if atom.symbol == 'X']
    del_indices = []
    for index in index_adsorbates:
        indices, offsets = nl.get_neighbors(index)
        for i, offset in zip(indices, offsets):
            if atoms[i].symbol == 'H':
                del_indices.append(index)
                break
    atoms = atoms[[atom.index for atom in atoms if atom.index not in del_indices]]
    return atoms

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

def looks_like(a1, a2):
    """ Return if structure a1 or a2 are similar or not. """
    a1 = a1[[atom.index for atom in a1 if atom.symbol != 'X']]
    a2 = a2[[atom.index for atom in a2 if atom.symbol != 'X']]
    n_top=len(a1)
    pair_cor_cum_diff=0.015,
    pair_cor_max=0.7
    dE=0.0000001
    mic=False
    if len(a1) != len(a2):
        return False
    # first we check the formula
    if a1.get_chemical_formula() != a2.get_chemical_formula():
        return False
    # then we check the structure
    a1top = a1[-n_top:]
    a2top = a2[-n_top:]
    cum_diff, max_diff = compare_structure(a1top, a2top)
    return (cum_diff < pair_cor_cum_diff
            and max_diff < pair_cor_max)

def remove_similar_adsorbates(atoms):
    """Remove similar sites on the surface according to similar local environment"""
    cutoff = 4
    nl = NeighborList(cutoffs=[cutoff / 2.] * len(atoms),
                            self_interaction=True,
                            bothways=True,
                            skin=0.)
    nl.update(atoms)
    index_adsorbates = [atom.index for atom in atoms if atom.symbol == 'X']
    index_center = index_adsorbates[len(index_adsorbates)//6]
    unique_indices = [index_center]
    del_indices = []
    index_adsorbates.remove(index_center)
    for index in index_adsorbates:
        indices, offsets = nl.get_neighbors(index)
        atoms_temp = atoms.copy()
        for i, offset in zip(indices, offsets):
            atoms_temp.positions[i] = atoms.positions[i] + np.dot(offset, atoms.get_cell())
        atoms_part_comp = atoms_temp[indices]
        # view(atoms_part_comp)
        unique = False # flag
        for unique_index in unique_indices:
            indices_u, offsets_u = nl.get_neighbors(unique_index)
            atoms_temp = atoms.copy()
            for i, offset in zip(indices_u, offsets_u):
                atoms_temp.positions[i] = atoms.positions[i] + np.dot(offset, atoms.get_cell())
            atoms_part_u = atoms_temp[indices_u]
            result = looks_like(atoms_part_u, atoms_part_comp)
            if result == True:
                del_indices.append(index)
                unique = False
                break
            else:
                unique = True
        if unique == True:
            unique_indices.append(index)
    atoms = atoms[[atom.index for atom in atoms if atom.index not in del_indices]]
    print('del:', del_indices)
    return atoms

def top_pos(slab, i):
    """Get one top site position that can be used for adding adsorbates, only for this system"""
    top_site = slab[i].position
    return top_site


def hollow_pos(slab, i1, i2, i3):
    """Get one hollow site position that can be used for adding adsorbates, only for this system"""
    hollow_site = (
        1 / 3.0 * (slab[i1].x + slab[i2].x + slab[i3].x),
        1 / 3.0 * (slab[i1].y + slab[i2].y + slab[i3].y),
        1 / 3.0 * (slab[i1].z + slab[i2].z + slab[i3].z),
    )
    hollow_site = np.asarray(hollow_site)  # transfer type of variable to array
    return hollow_site

def add_ads(adsorbate, position, bg=False):
    """Add adsorbates on a specific site"""
    if adsorbate == "HOCO":
        ads = Atoms(
            [
                Atom("H", (0.649164000, -1.51784000, 0.929543000), tag=-4),
                Atom("C", (0.000000000, 0.000000000, 0.000000000), tag=-4),
                Atom("O", (-0.60412900, 1.093740000, 0.123684000), tag=-4),
                Atom("O", (0.164889000, -0.69080700, 1.150570000), tag=-4),
            ]
        )
        ads.translate(position + (0.0, 0.0, 2.5))
    elif adsorbate == "CO":
        ads = Atoms([Atom("C", (0.0, 0.0, 0.0), tag=-2), 
                      Atom("O", (0.0, 0.0, 1.14), tag=-2)])
        ads.translate(position + (0.0, 0.0, 2.0))
    elif adsorbate == "H":
        ads = Atoms([Atom("H", (0.0, 0.0, 0.0), tag=-1)])
        ads.translate(position + (0.0, 0.0, 2.0))
    elif adsorbate == "OH":
        ads = Atoms([Atom("O", (0.0, 0.0, 0.0), tag=-3), 
                      Atom("H", (0.0, 0.0, 0.97), tag=-3)])
        ads.translate(position + (0.0, 0.0, 2.0))
    if bg == True:
        for atom in ads:
            atom.symbol = "X"
    return ads

def generate_all_site_struts_to_db_for_CE(adsorbate='CO'):
    """Generate different sites CO adsorbed slab for CE"""
    # from pcat.build.diff_sites import get_sites, add_full_adsorbates, \
    #     remove_adsorbates_vertical_too_close, remove_similar_adsorbates
    db = connect("./random_50_2x2.db")
    db_all_sites = connect('all_sites_CO_on_cands.db')
    for row in db.select(calc='clease'):
        atoms = row.toatoms()
        system_ori = atoms.copy()
        atoms = atoms[[atom.index for atom in atoms if atom.symbol != 'X']]
        sites = get_sites(atoms)
        atoms = add_full_adsorbates(atoms, sites['top'], adsorbate='X')
        # atoms = add_full_adsorbates(atoms, sites['hollow'], adsorbate='X')
        atoms = remove_adsorbates_vertical_too_close(atoms)
        # view(atoms)
        atoms = remove_similar_adsorbates(atoms)
        # view(atoms)
        all_possible_sites = [atom.index for atom in atoms if atom.symbol == 'X']
        print('possible sites:', all_possible_sites)
        for site in all_possible_sites:
            system = system_ori.copy()
            for atom in system:
                if atom.z > 6.5 and atom.z < 7.5:
                    pos_X = atoms[site].position
                    pos_potential = atom.position
                    pos_pot = pos_potential + (0.0, 0.0, 2.0)
                    # print(pos_X, pos_pot)
                    if sum(abs(pos_X-pos_pot))<0.0001:
                        ads_bg = add_ads(adsorbate, pos_potential)
                        system.extend(ads_bg)
                    else:
                        ads = add_ads(adsorbate, pos_potential, bg=True)
                        system.extend(ads)
            system = rm_1X(system)
            # pos = atoms[site].position
            # ads_bg = add_ads(adsorbate, pos, bg=True)
            # slab.extend(ads_bg)
            db_all_sites.write(system)
            
def test_temp_all_sites(temp="./random_50_2x2.db"):
    db = connect(temp)
    row = db.get(id=2)
    atoms = row.toatoms()
    atoms = atoms[[atom.index for atom in atoms if atom.symbol != 'X']]
    write('template_2x2.traj', atoms)
    sites = get_sites(atoms)
    atoms = add_full_adsorbates(atoms, sites['top'], adsorbate='X')
    atoms = add_full_adsorbates(atoms, sites['hollow'], adsorbate='X')
    atoms = remove_adsorbates_vertical_too_close(atoms)
    # view(atoms)
    # atoms = remove_similar_adsorbates(atoms)
    # view(atoms)
    all_possible_sites = [atom.index for atom in atoms if atom.symbol == 'X']
    view(atoms)
    print('possible sites:', all_possible_sites)
    
def add_one_random_unique_ads(atoms, adsorbate=None):
    """Add adsorbate onto given slab"""
    sites = get_sites(atoms)
    atoms = add_full_adsorbates(atoms, sites['top'], adsorbate='X')
    atoms = add_full_adsorbates(atoms, sites['hollow'], adsorbate='X')
    atoms = remove_adsorbates_vertical_too_close(atoms) # vertical distance
    # view(atoms)
    atoms = remove_similar_adsorbates(atoms)
    # view(atoms)
    all_possible_sites = [atom.index for atom in atoms if atom.symbol == 'X']
    # view(atoms)
    print('possible sites:', all_possible_sites)
    site = random.choice(all_possible_sites)
    pos = atoms[site].position
    pos = pos - (0.0, 0.0, 2.0)
    if adsorbate == None:
        random.seed(random.randint(1,100000000))
        adsorbates = ['HOCO', 'CO', 'H', 'OH']
        adsorbate = random.choice(adsorbates)
        print(adsorbate)
    ads = add_ads(adsorbate, pos, bg=False)
    atoms.extend(ads)
    atoms = atoms[[atom.index for atom in atoms if atom.symbol != 'X']]
    # view(atoms)
    return atoms, adsorbate
    
def generate_random_init_structs_one_ads(tot=50, sub_ele='Ti', adsorbate=None, 
                            random_slab_Pd=True, 
                            random_slab_H=True, 
                            random_one_ads=True,):
    """Generate random clean slabs with random 1 adsorbates in unique sites"""
    temp = read('template_2x2.traj')
    # view(atoms)
    images = []
    for _ in range(tot):
        atoms = temp.copy()
        Pd_indices = [atom.index for atom in atoms if atom.symbol=='Pd' and atom.z>2.]
        H_indices = [atom.index for atom in atoms if atom.symbol=='H' and atom.z > 2.]
        atoms.set_constraint(FixAtoms(mask=[atom.z<=2. for atom in atoms]))
        if random_slab_Pd:
            random.seed(random.randint(1,100000000))
            Pd_times = random.choice(range(len(Pd_indices)))
            random.shuffle(Pd_indices)
            sub_Pd = random.sample(Pd_indices, Pd_times)
            if sub_Pd != []:
                for i in sub_Pd:
                    atoms[i].symbol = sub_ele
            # view(atoms)
        if random_slab_H:
            random.seed(random.randint(1,100000000))
            H_times = random.choice(range(len(H_indices)))
            random.shuffle(H_indices)
            sub_H = random.sample(H_indices, H_times)
            if sub_H != []:
                for i in sub_H:
                    atoms[i].symbol = 'X'
            atoms = atoms[[atom.index for atom in atoms if atom.symbol != 'X']]
        if random_one_ads:
            atoms, ads = add_one_random_unique_ads(atoms, adsorbate=adsorbate)
            atoms.info['ads'] = ads
            atoms.info['nums_Pd'] = len(Pd_indices)-len(sub_Pd)
            atoms.info['nums_H'] = len(H_indices)-len(sub_H)
            # view(atoms)
        images.append(atoms)
    # view(images)
    return images

def get_atoms_with_one_ads_data(images):
    """Get data including atoms with one adsobates"""
    adss, nums_Pds, nums_Hs = [], [], []
    for atoms in images:
        print(atoms.info)
        ads = atoms.info['ads']
        nums_Pd = atoms.info['nums_Pd']
        nums_H = atoms.info['nums_H']
        adss.append(ads)
        nums_Pds.append(nums_Pd)
        nums_Hs.append(nums_H)
    return zip(adss, nums_Pds, nums_Hs)

def atoms_too_close_after_addition(atoms, n_added, cutoff=1.7, mic=False): 
    """Check if there are atoms that are too close to each other after 
    adding some new atoms.

    Parameters
    ----------
    atoms : ase.Atoms object
        Accept any ase.Atoms object. No need to be built-in.
    n_added : int
        Number of newly added atoms.
    cutoff : float, default 1.5
        The cutoff radius. Two atoms are too close if the distance between
        them is less than this cutoff
    mic : bool, default False
        Whether to apply minimum image convention. Remember to set 
        mic=True for periodic systems.
    """
    newp, oldp = atoms.positions[-n_added:], atoms.positions[:-n_added]
    newps = np.repeat(newp, len(oldp), axis=0)
    oldps = np.tile(oldp, (n_added, 1))
    if mic:
        _, dists = find_mic(newps - oldps, atoms.cell, pbc=True)
    else:
        dists = np.linalg.norm(newps - oldps, axis=1)

    return any(dists < cutoff)

def add_n_random_ads(atoms, adsorbates=['CO', 'H', 'OH']):
    """Add n random adsorbates onto given slab"""
    sites = get_sites(atoms)
    atoms = add_full_adsorbates(atoms, sites['top'], adsorbate='X')
    atoms = add_full_adsorbates(atoms, sites['hollow'], adsorbate='X')
    all_possible_sites = [atom.index for atom in atoms if atom.symbol == 'X']
    view(atoms)
    print('possible sites:', all_possible_sites)
    random.seed(random.randint(1,100000000))
    ads_times = random.choice(range(len(all_possible_sites)))
    adsorbates_random = random.choices(adsorbates, k=ads_times)
    random.shuffle(all_possible_sites)
    sites_random = random.sample(all_possible_sites, ads_times)
    atoms_with_adsorbates = atoms.copy()
    for site, adsorbate in zip(sites_random, adsorbates_random):
        pos = atoms[site].position
        pos = pos - (0.0, 0.0, 2.0)
        ads = add_ads(adsorbate, pos, bg=False)
        atoms_with_adsorbates.extend(ads)
    atoms_with_adsorbates = atoms_with_adsorbates[[atom.index for atom in atoms_with_adsorbates if atom.symbol != 'X']]
    # view(atoms)
    return atoms_with_adsorbates, sites_random, adsorbates_random

def add_random_ads_one_by_one(atoms, pos, adsorbate, cutoff):
    """Add random adsorbates onto given slab one by one"""
    atoms_old = atoms.copy()
    pos = pos - (0.0, 0.0, 2.0)
    ads = add_ads(adsorbate, pos, bg=False)
    atoms.extend(ads)
    indices = [atom.index for atom in atoms]
    indices = indices[-len(ads):]
    if_too_close = atoms_too_close_after_addition(atoms, len(ads), cutoff=cutoff, mic=True)
    if if_too_close:
        atoms = atoms_old
        indices = []
    # atoms = atoms[[atom.index for atom in atoms if atom.symbol != 'X']]
    # view(atoms)
    return atoms, indices, if_too_close

def check_adsorbates(atoms):
    ads_indices = atoms.info['ads_indices']
    ads_symbols = atoms.info['ads_symbols']
    for ads_index, ads_symbol in zip(ads_indices, ads_symbols):
        print(str(atoms[ads_index].symbols), ads_symbol)
    print('-----')
    for ads_index, ads_symbol in zip(ads_indices, ads_symbols):
        print(str(atoms[ads_index].symbols), ads_symbol)
        assert str(atoms[ads_index].symbols)==ads_symbol
    return ads_indices, ads_symbols

def add_tags_for_metals(atoms):
    """Add tags for metals, it is very useful for the surface layer"""
    for atom in atoms:
        if atom.symbol=='Pd' and atom.z>6.5 and atom.z<7.5: # 1st layer
            atom.tag = 1
        elif atom.symbol=='Pd' and atom.z>4. and atom.z<5.:
            atom.tag = 2
        elif atom.symbol=='Pd' and atom.z>2. and atom.z<3.:
            atom.tag = 3
        elif atom.symbol=='Pd' and atom.z>-1 and atom.z<1.:
            atom.tag = 4
        elif atom.symbol=='H' and atom.z>=7. and atom.z<8.5:
            atom.tag = 5
        elif atom.symbol=='H' and atom.z>=4.6 and atom.z<7.:
            atom.tag = 6
        elif atom.symbol=='H' and atom.z>3. and atom.z<4.6:
            atom.tag = 7
        elif atom.symbol=='H' and atom.z>1. and atom.z<2.:
            atom.tag = 8
        else:
            atom.tag = 9
    return atoms

def generate_random_init_structs_n_ads(tot=50, sub_ele='Ti',
                            temp='',
                            adsorbates=['CO', 'H', 'OH'], 
                            random_slab_Pd=True, 
                            random_slab_H=True, 
                            random_n_ads=True,
                            cutoff=1.7):
    """Generate random clean slabs with random n adsorbates in unique sites"""
    # temp = read('template_2x2.traj')
    # view(atoms)
    images = []
    for _ in range(tot):
        atoms = temp.copy()
        del atoms[[atom.index for atom in atoms if atom.z>8.]] # remove 1st layer H
        Pd_indices = [atom.index for atom in atoms if atom.symbol=='Pd' and atom.z > 2.]
        H_indices = [atom.index for atom in atoms if atom.symbol=='H' and atom.z > 2.]
        atoms.set_constraint(FixAtoms(mask=[atom.z<=2. for atom in atoms]))
        atoms = add_tags_for_metals(atoms)
        # check_tags(atoms)
        # view(atoms)
        sub_Pd, sub_H = [], []
        if random_slab_Pd:
            random.seed(random.randint(1,100000000))
            Pd_times = random.choice(range(len(Pd_indices)))
            random.shuffle(Pd_indices)
            sub_Pd = random.sample(Pd_indices, Pd_times)
            if sub_Pd != []:
                for i in sub_Pd:
                    atoms[i].symbol = sub_ele
            # view(atoms)
        if random_slab_H:
            random.seed(random.randint(1,100000000))
            H_times = random.choice(range(len(H_indices)))
            random.shuffle(H_indices)
            sub_H = random.sample(H_indices, H_times)
            if sub_H != []:
                for i in sub_H:
                    atoms[i].symbol = 'X'
            atoms = atoms[[atom.index for atom in atoms if atom.symbol != 'X']]
            # check_tags(atoms)
            # view(atoms)
        if random_n_ads:
            sites = get_sites(atoms)
            atoms = add_full_adsorbates(atoms, sites['top'], adsorbate='X')
            atoms = add_full_adsorbates(atoms, sites['hollow'], adsorbate='X')
            all_possible_sites = [atom.index for atom in atoms if atom.symbol == 'X']
            atoms_ori = atoms.copy()
            atoms = atoms[[atom.index for atom in atoms if atom.symbol != 'X']]
            print('possible sites:', all_possible_sites)
            random.seed(random.randint(1,100000000))
            ads_times = random.choice(range(len(all_possible_sites))) # how many adss
            sitess, indicess, adss = [], [], []
            i = 0 
            # while i < ads_times:
            for _ in range(ads_times):
                random.seed(random.randint(1,100000000))
                adsorbate_random = random.choice(adsorbates) # one random adsorbate
                random.seed(random.randint(1,100000000))
                site_random = random.choice(all_possible_sites) # one random site
                pos = atoms_ori[site_random].position    
                all_possible_sites.remove(site_random)
                atoms, indices, if_too_close = add_random_ads_one_by_one(atoms, pos, adsorbate_random, cutoff)
                print(if_too_close)
                if if_too_close:
                    continue
                sitess.append(site_random)
                indicess.append(indices)
                adss.append(adsorbate_random)
                i += 1
            
            # atoms, sites, adss = add_n_random_ads(atoms, adsorbates=adsorbates)
            atoms.info['X_indices'] = sitess
            atoms.info['ads_indices'] = indicess
            atoms.info['ads_symbols'] = adss
            atoms.info['nums_Pd'] = len(Pd_indices)-len(sub_Pd)
            atoms.info['nums_H'] = len(H_indices)-len(sub_H)
            # view(atoms)
        # check_adsorbates(atoms)
        images.append(atoms)
    # view(images)
    return images

def generate_pure_compositions(sub_ele='Ti'):
    """Generate TiH, PdH, Ti, Pd"""
    temp = read('template_2x2.traj')
    atoms = temp.copy()
    images = []
    images.append(temp)
    images.append(atoms[[atom.index for atom in atoms if atom.symbol=='Pd']])
    Pd_indices = [atom.index for atom in atoms if atom.symbol=='Pd']
    for i in Pd_indices:
        atoms[i].symbol = sub_ele
    images.append(atoms)
    images.append(atoms[[atom.index for atom in atoms if atom.symbol==sub_ele]])
    write('../pure_compositions.traj', images)
    # view(images)
    return images

def get_atoms_with_n_ads_data(images):
    """Get data including atoms with n adsobates"""
    sites_list, indices_list, adss_list, nums_Pds, nums_Hs = [], [], [], [], []
    for atoms in images:
        print('\n')
        # check_adsorbates(atoms)
        print(atoms.info['ads_indices'])
        print(atoms.info['ads_symbols'])
        assert len(atoms.info['ads_indices']) == len(atoms.info['ads_symbols'])
        sites = atoms.info['X_indices']
        adss = atoms.info['ads_indices']
        indices = atoms.info['ads_symbols']
        nums_Pd = atoms.info['nums_Pd']
        nums_H = atoms.info['nums_H']
        sites_list.append(sites)
        indices_list.append(indices)
        adss_list.append(adss)
        nums_Pds.append(nums_Pd)
        nums_Hs.append(nums_H)
    return zip(sites_list, indices_list, adss_list, nums_Pds, nums_Hs)

def check_tags(atoms):
    """Check tags in atoms"""
    print(atoms.get_chemical_formula())
    for layer in [1, 2, 3, 4]:
        indices_existed_Ms = [atom.index for atom in atoms if atom.tag in [layer]] # ist bilayer H
        assert len(indices_existed_Ms)<=4
        indices_existed_Hs = [atom.index for atom in atoms if atom.tag in [layer+4]] # ist bilayer H
        assert len(indices_existed_Hs)<=4
    return True

def check_db(name):
    """Check atoms in database"""
    images = read(name, index=':')
    for atoms in images:
        # _, _ = check_adsorbates(atoms)
        _ = check_tags(atoms)
    print('done')
    
if __name__ == "__main__":
    temp = read('template_2x2.traj')
    # generate_all_site_structs_to_db(adsorbate='CO')
    # test_temp_all_sites()
    # images = generate_random_structs(tot = 50, sub_ele='Ti', adsorbate='CO')
    # images = generate_random_init_structs_one_ads(tot=50, sub_ele='Ti') # default is for all adsorbates
    images = generate_random_init_structs_n_ads(tot=10, sub_ele='Ni', temp=temp)
    # images = generate_pure_compositions(sub_ele='Ti')
    print(get_atoms_with_n_ads_data(images))
    # view(images)
    name = '../random50_adss.traj'
    write(name, images)
    
    check_db(name)
    print('done')