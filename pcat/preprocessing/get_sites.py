from ase.io import write, read
from ase.db import connect
from ase.visualize import view
from scipy.spatial import Delaunay
import numpy as np
from ase import Atom, Atoms
from ase.neighborlist import NeighborList

# def remove_atoms_outside_cell(atoms):
#     """Only for cubic"""
#     cell = atoms.get_cell()
#     i = cell[0]
#     max_i = np.dot(i, i)
#     j = cell[1]
#     max_j = np.dot(j, j)
#     k = cell[2]
#     max_k = np.dot(k, k)
#     indices = []
#     for atom in atoms:
#         pos = atom.position
#         if (np.dot(pos, i) >=0 and np.dot(pos, i) <= max_i) and \
#            (np.dot(pos, j) >=0 and np.dot(pos, j) <= max_j) and \
#            (np.dot(pos, k) >=0 and np.dot(pos, k) <= max_k):
#             indices.append(atom.index)
#     atoms = atoms[indices]
#     return atoms

# def remove_atoms_outside_cell(atoms):
#     """Only for rhombus"""
#     cell = atoms.get_cell()
#     i = cell[0]
#     j = cell[1]
#     k = cell[2]
#     indices = []
#     for atom in atoms:
#         pos = atom.position
#         if (pos[0] >= pos[1]*j[0]/j[1] and pos[0] <= (pos[1]+i[0]*j[1]/j[0])*(j[0]/j[1])) and \
#            (pos[1] >= 0 and pos[1] <= j[1]) and \
#            (pos[2] >= 0 and pos[2] <= k[2]):
#             indices.append(atom.index)
#     atoms = atoms[indices]
#     return atoms

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
    print(del_indices)
    return atoms

if __name__ == '__main__':
    db_template = connect('template.db')
    row1 = list(db_template.select(id=1))[0]
    atoms = row1.toatoms()
    atoms = atoms[[atom.index for atom in atoms if atom.symbol != 'X']]
    sites = get_sites(atoms)
    atoms = add_full_adsorbates(atoms, sites['top'], adsorbate='X')
    atoms = add_full_adsorbates(atoms, sites['hollow'], adsorbate='X')
    atoms = remove_adsorbates_too_close(atoms)
    view(atoms)
    atoms = remove_similar_adsorbates(atoms)
    view(atoms)
