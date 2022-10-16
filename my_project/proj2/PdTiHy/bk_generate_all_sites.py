# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:37:29 2022

@author: changai
"""

from ase import Atom, Atoms
from ase.visualize import view
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
from contextlib import contextmanager
from multiprocessing import Pool
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from ase.build import surface
import random
from scipy.spatial import Delaunay
import numpy as np
from ase import Atom, Atoms
from ase.neighborlist import NeighborList
import matplotlib as mpl
# mpl.use('TkAgg')

# name = sys.argv[1]


@contextmanager
def cd(newdir):
    """Create directory and walk in"""
    prevdir = os.getcwd()
    try:
        os.makedirs(newdir)
    except OSError:
        pass
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

def build_system():
    a = 4.138
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
    assert len(atoms) == 16
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

def remove_adsorbates_too_close(atoms):
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
                Atom("H", (0.649164000, -1.51784000, 0.929543000)),
                Atom("C", (0.000000000, 0.000000000, 0.000000000)),
                Atom("O", (-0.60412900, 1.093740000, 0.123684000)),
                Atom("O", (0.164889000, -0.69080700, 1.150570000)),
            ]
        )
        ads.translate(position + (0.0, 0.0, 3.0))
    elif adsorbate == "CO":
        ads = Atoms([Atom("C", (0.0, 0.0, 0.0)), Atom("O", (0.0, 0.0, 1.14))])
        ads.translate(position + (0.0, 0.0, 2.0))
    elif adsorbate == "H":
        ads = Atoms([Atom("H", (0.0, 0.0, 0.0))])
        ads.translate(position + (0.0, 0.0, 1.5))
    elif adsorbate == "OH":
        ads = Atoms([Atom("O", (0.0, 0.0, 0.0)), Atom("H", (0.0, 0.0, 0.97))])
        ads.translate(position + (0.0, 0.0, 2.0))
    if bg == True:
        for atom in ads:
            atom.symbol = "X"
    return ads

def generate_all_site_sturts_to_db(adsorbate='CO'):
    """Generate different sites CO adsorbed slab"""
    # from pcat.build.diff_sites import get_sites, add_full_adsorbates, \
    #     remove_adsorbates_too_close, remove_similar_adsorbates
    db = connect("./candidates_initial_and_final_PdTiH_surf_r12.db")
    db_all_sites = connect('all_sites_CO_on_cands.db')
    for row in db.select(struct_type='initial'):
        atoms = row.toatoms()
        system_ori = atoms.copy()
        atoms = atoms[[atom.index for atom in atoms if atom.symbol != 'X']]
        slab_ori = atoms.copy()
        sites = get_sites(atoms)
        atoms = add_full_adsorbates(atoms, sites['top'], adsorbate='X')
        atoms = add_full_adsorbates(atoms, sites['hollow'], adsorbate='X')
        atoms = remove_adsorbates_too_close(atoms)
        # view(atoms)
        atoms = remove_similar_adsorbates(atoms)
        # view(atoms)
        all_possible_sites = [atom.index for atom in atoms if atom.symbol == 'X']
        print('possible sites:', all_possible_sites)
        for site in all_possible_sites:
            slab = slab_ori.copy()
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

def test_unique_sites(adsorbate='CO'):
    """Generate different sites CO adsorbed slab"""
    # from pcat.build.diff_sites import get_sites, add_full_adsorbates, \
    #     remove_adsorbates_too_close, remove_similar_adsorbates
    # db = connect("./dft_candidates_PdHx_r8.db")
    # db_all_sites = connect('all_t_p_sites_CO_on_cands.db')
    db_template = connect("/home/energy/changai/ce_PdxTiHy/random/mc/ce_ads/db/template.db")
    row1 = list(db_template.select(id=1))[0]
    atoms = row1.toatoms()
    system_ori = atoms.copy()
    atoms = atoms[[atom.index for atom in atoms if atom.symbol != 'X']]
    slab_ori = atoms.copy()
    sites = get_sites(atoms)
    atoms = add_full_adsorbates(atoms, sites['top'], adsorbate='X')
    # atoms = add_full_adsorbates(atoms, sites['hollow'], adsorbate='X')
    atoms = remove_adsorbates_too_close(atoms)
    # view(atoms)
    atoms = remove_similar_adsorbates(atoms)
    view(atoms)
    all_possible_sites = [atom.index for atom in atoms if atom.symbol == 'X']
    print('possible sites:', all_possible_sites)
    # for site in all_possible_sites:
    #     slab = slab_ori.copy()
    #     system = system_ori.copy()
    #     for atom in system:
    #         if atom.z > 6.5 and atom.z < 7.5:
    #             pos_X = atoms[site].position
    #             pos_potential = atom.position
    #             pos_pot = pos_potential + (0.0, 0.0, 2.0)
    #             # print(pos_X, pos_pot)
    #             if sum(abs(pos_X-pos_pot))<0.0001:
    #                 ads_bg = add_ads(adsorbate, pos_potential)
    #                 system.extend(ads_bg)
    #             else:
    #                 ads = add_ads(adsorbate, pos_potential, bg=True)
    #                 system.extend(ads)
    #     system = rm_1X(system)
    #     db_all_sites.write(system)

if __name__ == "__main__":
    print('start running') 
    generate_all_site_sturts_to_db(adsorbate='CO')
    print('done')
