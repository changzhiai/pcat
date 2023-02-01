"""
Only for reference of genetic algorithm with cluser expansion


"""

# from acat.build.ordering import RandomOrderingGenerator as ROG
from ase.cluster import Icosahedron
from ase import Atoms
from ase.build import surface, bulk, cut
from ase.visualize import view
import numpy as np
from ase.io import write, read
from ase.geometry import get_duplicate_atoms
from ase.ga.data import DataConnection, PrepareDB
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import get_all_atom_types
from ase.ga.utilities import closest_distances_generator
from ase.constraints import FixAtoms
from ase.ga.offspring_creator import OperationSelector
from ase.ga.population import Population, RankFitnessPopulation
from ase.ga.convergence import GenerationRepetitionConvergence
from ase.optimize import BFGS
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.constraints import FixAtoms
from ase.ga.standard_comparators import SequentialComparator
from ase.ga.standard_comparators import InteratomicDistanceComparator
from ase.ga.standard_comparators import StringComparator
from ase.ga.utilities import get_nnmat_string
from ase.ga.ofp_comparator import OFPComparator
from ase.ga.slab_operators_hydride import RandomSlabPermutation
from ase.ga.slab_operators_hydride import CutSpliceSlabCrossover
from ase.ga.slab_operators_hydride import SymmetrySlabPermutation
from contextlib import contextmanager
from ase import Atoms
from ase.build import surface, bulk, root_surface_analysis, root_surface, add_vacuum, cut, make_supercell
from ase.geometry import get_duplicate_atoms
from ase.calculators.emt import EMT
from ase.db import connect
from ase.constraints import UnitCellFilter
from ase.optimize import BFGS
from clease.tools import update_db
from clease.settings import ClusterExpansionSettings
from clease.settings import Concentration
from clease.template_filters import SlabCellFilter
from clease.structgen import NewStructures
from clease.settings import ClusterExpansionSettings
from clease.settings import Concentration
from clease.template_filters import SlabCellFilter
from clease.calculator import get_ce_energy
from clease.calculator import attach_calculator
from clease.tools import reconfigure
from clease.tools import wrap_and_sort_by_position
from clease.corr_func import CorrFunction
from clease.settings import settings_from_json
import json
import sys
from multiprocessing import Pool
import os
import shutil
import string
import random
import subprocess
import time
import re
from ase.calculators.singlepoint import SinglePointCalculator as SPC

def remove_X(atoms):
    try:
        del atoms[[atom.index for atom in atoms if atom.symbol == 'X']]
    except:
        atoms = atoms
    return atoms

def remove_top_X(atoms, z=9.0):
    del atoms[[atom.index for atom in atoms if atom.symbol == 'X' and atom.z > z]]
    return atoms

def add_top_X(atoms, system, z=9.0):
    system = system.copy()
    for atom in system:
        if atom.z > z:
            atoms.append(atom)
    return atoms

def add_X(atoms):
    '''
    add X into vacuum layers
    '''
    thickness = 2.389
    z_max_cell = atoms.cell[2, 2]
    top_metal_layer = atoms[[atom.index for atom in atoms if atom.z > 7 and atom.z < 8]]
    for atom in top_metal_layer:
        atom.symbol = 'X'
    while max(top_metal_layer.positions[:,2]) + thickness <  z_max_cell:
        top_metal_layer.positions[:,2] = top_metal_layer.positions[:,2] + thickness
        atoms+=top_metal_layer
    return atoms

def remove_H(atoms):
    del atoms[[atom.index for atom in atoms if atom.symbol == 'H']]
    return atoms

def add_H(atoms, origin_with_H):
    [atoms.append(atom) for atom in origin_with_H if atom.symbol=='H'] # add H
    return atoms

def add_bulk_X(atoms, origin_with_X):
    [atoms.append(atom) for atom in origin_with_X if atom.symbol=='X'] # add H
    return atoms

def delete_random_elems(input_list, n):
    to_delete = set(random.sample(range(len(input_list)), n))
    return [x for i,x in enumerate(input_list) if not i in to_delete]

def random_H(atoms, include_X = False):
    '''
    substitue H using X
    '''
    H_origin_all = atoms[[atom.index for atom in atoms if atom.symbol=='H']]
    H_atoms_all = list(atoms[[atom.index for atom in atoms if atom.symbol=='H']])
    random_num = random.randint(0, 63)
    H_atoms_rest = delete_random_elems(H_atoms_all, random_num) # Randomly delete 3 elements from the list
    X_atoms = [atom for atom in H_atoms_all if atom not in H_atoms_rest]
    indices = [atom.index for atom in H_atoms_all if atom in X_atoms]
    syms =['X'] * len(indices)
    syms = np.array(syms, dtype=object)
    H_origin_all.symbols[indices] = syms
    atoms = remove_H(atoms)
    atoms = add_H(atoms, H_origin_all)
    if include_X == True:
        atoms = add_bulk_X(atoms, H_origin_all)
    return atoms

def formation_energy_ref_metals(atoms, energy):
    """
    Pure Pd: -1.951 eV/atom
    Pure Ti: -5.858 eV/atom
    H2 gas: -7.158 ev
    """
    try:
        Pds = len(atoms[[atom.index for atom in atoms if atom.symbol=='Pd']])
    except:
        Pds = 0
    try:
        Tis = len(atoms[[atom.index for atom in atoms if atom.symbol=='Ti']])
    except:
        Tis = 0
    try:
        Hs = len(atoms[[atom.index for atom in atoms if atom.symbol=='H']])
    except:
        Hs = 0
    form_e = (energy - Pds*(-1.951) - Tis*(-5.858) - 1./2*Hs*(-7.158))/(Pds+Tis+Hs)
    # print(Pds, Tis, Hs, form_e)
    return form_e

def formation_energy_ref_hydrides(atoms, energy):
    """
    reference PdH and TiH (vasp) for PdxTi(64-x)H64
    Pure PdH: -5.222 eV/PdH
    Pure TiH: -9.543 eV/TiH
    """
    try:
        Pds = len(atoms[[atom.index for atom in atoms if atom.symbol=='Pd']])
    except:
        Pds = 0
    try:
        Tis = len(atoms[[atom.index for atom in atoms if atom.symbol=='Ti']])
    except:
        Tis = 0
    try:
        Hs = len(atoms[[atom.index for atom in atoms if atom.symbol=='H']])
    except:
        Hs = 0
    form_e = (energy - Pds*(-5.222) - Tis*(-9.543))/(Pds+Tis+Hs)
    # print(Pds, Tis, Hs, form_e)
    return form_e

def get_ordered_init(db_name_withH, system):
    if os.path.exists(db_name_withH):
        os.remove(db_name_withH)
    system = system.copy()
    del system[[atom.index for atom in system if atom.symbol=='X']]
    alloy = system.copy()
    alloy_real = system.copy()
    del alloy[[atom.index for atom in alloy if atom.symbol=='H']] # delete H
    # pop_size = 5
    # rog = ROG(alloy, elements=["Pd", "Ti"], composition={"Pd": 0.5, "Ti": 0.5}, trajectory='orderings.traj')
    # rog = ROG(alloy, elements=["Pd", "Ti"], trajectory='orderings.traj')
    # rog.run(num_gen=pop_size)
    images = read("~/ce_PdxTiHy/random/ga/multiprocessing/orderings.traj", index=":")
    for image in images: # add H to all images
        image = add_H(image, system)
        image = random_H(image, include_X = True)
    atom_numbers = 32 * [46] + 32 * [79]
    db = PrepareDB(db_file_name=db_name_withH, # Instantiate the db
                  simulation_cell=alloy_real, # with H
                  stoichiometry=atom_numbers)
    for atoms in images:
        if 'data' not in atoms.info:
            atoms.info['data'] = {}
        # atoms = remove_X(atoms)
        db.add_unrelaxed_candidate(atoms, data=atoms.info['data'])

def get_comp(atoms):
    return atoms.get_chemical_formula()

def comp_settings(db):
    atom_numbers_to_optimize = db.get_atom_numbers_to_optimize()
    n_to_optimize = len(atom_numbers_to_optimize)
    print(n_to_optimize)
    compIDC = InteratomicDistanceComparator(n_top=n_to_optimize,
                                        pair_cor_cum_diff=0.015,
                                        pair_cor_max=0.7,
                                        dE=0.02,
                                        mic=False)
    compSC = StringComparator('nnmat_string')
    comp = SequentialComparator([compIDC,
                                compSC],
                                [0, 1])
    return comp

def get_clease_settings():
    conc = Concentration(basis_elements=[['Pd', 'Ti'], ['H', 'X'],
                                        ['X']  # This is a "ghost" sublattice for vacuum
                                        ])
    conc.set_conc_formula_unit(formulas=['Pd<x>Ti<1-x>', 'H<y>X<1-y>', 'X'], variable_range={'x': (0., 1.),'y': (0, 1.)})

    db_pri = connect('~/ce_PdxTiHy/random/ga/multiprocessing/PdxTi1_xH_clease.db')
    system_pri = db_pri.get("name=primitive_cell").toatoms()
    prim = system_pri
    concentration = conc
    size= [2, 2, 1]
    max_cluster_dia=[6.0, 5.0, 4.0]
    max_cluster_size=4
    supercell_factor=27
    db_name='PdxTi1_xHy_clease.db'
    settings = ClusterExpansionSettings(prim,
                                        concentration,
                                        size=size,
                                        max_cluster_dia=max_cluster_dia,
                                        max_cluster_size=max_cluster_size,
                                        supercell_factor=supercell_factor,
                                        db_name=db_name,
                                        )
    cell_filter = SlabCellFilter(prim.cell)
    settings.template_atoms.add_cell_filter(cell_filter)
    settings.include_background_atoms = True
    
    return settings

def get_last_n(num = 50):
    db_name = '../PdTiH_ga.db'
    lastn_name = 'lastn_PdTiH_ga.db'
    db = connect(db_name)
    tot_len = len(db)
    os.system('cp {0} {1}'.format(db_name, lastn_name))

    db_last = connect(lastn_name)
    del_list = []
    for row in db.select():
        if row.id <= tot_len - num:
            del_list.append(row.id)
    db_last.delete(del_list)
    return db_last

def continue_ga(db_name_withH, system):
    if os.path.exists(db_name_withH):
        os.remove(db_name_withH)
    system = system.copy()
    del system[[atom.index for atom in system if atom.symbol=='X']]
    alloy = system.copy()
    alloy_real = system.copy()
    del alloy[[atom.index for atom in alloy if atom.symbol=='H']] # delete H
    atom_numbers = 32 * [46] + 32 * [79]
    db = PrepareDB(db_file_name=db_name_withH, # Instantiate the db
                  simulation_cell=alloy_real, # with H
                  stoichiometry=atom_numbers)
    db_last = get_last_n(num = 50)
    for row in db_last.select():
        atoms = row.toatoms()
        if 'data' not in atoms.info:
            atoms.info['data'] = {}
        # atoms = remove_X(atoms)
        db.add_unrelaxed_candidate(atoms, data=atoms.info['data'])
    return db

def remain_last_n_to_db(num = 50):
    db_name = '../PdTiH_ga.db'
    lastn_name = 'PdTiH_ga.db'
    db = connect(db_name)
    tot_len = len(db)
    os.system('cp {0} {1}'.format(db_name, lastn_name))
    db_last = connect(lastn_name)
    del_list = []
    for row in db.select():
        if row.id <= tot_len - num -1000 and row.id > 1:
            del_list.append(row.id)
    db_last.delete(del_list)
    return db_last

def read_ga_db(db_name_withH, system):
    system = system.copy()
    del system[[atom.index for atom in system if atom.symbol=='X']]
    alloy = system.copy()
    alloy_real = system.copy()
    del alloy[[atom.index for atom in alloy if atom.symbol=='H']] # delete H
    atom_numbers = 32 * [46] + 32 * [79]
    db = PrepareDB(db_file_name=db_name_withH, # Instantiate the db
                  simulation_cell=alloy_real, # with H
                  stoichiometry=atom_numbers)
    db_last = connect('../candidates.db')
    for row in db_last.select():
        atoms = row.toatoms()
        # atoms = add_X(atoms)
        atoms = remove_top_X(atoms)
        # Ti_list = [atom.index for atom in atoms if atom.symbol == 'Ti']
        # atoms.symbols[Ti_list] = ['Ti']*len(Ti_list)
        if 'data' not in atoms.info:
            atoms.info['data'] = {}
        db.add_unrelaxed_candidate(atoms, data=atoms.info['data'])
    return db

if __name__ == '__main__':
    system = read('~/ce_PdxTiHy/random/ga/multiprocessing/template_from_clease.traj')
    k_value = 4
    multiprocessing = True
    db_name_withH = 'PdTiH_ga.db'
    pop_size = 200
    db = read_ga_db(db_name_withH)
    if multiprocessing:
        if 'SLURM_JOB_ID' in os.environ:
            # The script is submitted
            dname = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
            fname = '/scratch/changai/' + dname + '-' + db_name_withH
            shutil.copyfile(db_name_withH, fname)
            db = DataConnection(fname)
        else:
            db = DataConnection(db_name_withH)
    # else:
        # db = DataConnection(db_name_withH)
    
    # if True:
    # db = continue_ga(db_name_withH)
    # remain_last_n_to_db(num = 50)
    # db = read_ga_db(db_name_withH)
    # db = DataConnection(db_name_withH)
    # The population
    population = RankFitnessPopulation(data_connection=db,
                            population_size=pop_size,
                            comparator=comp_settings(db),
                            variable_function=get_comp,
                            logfile='log.txt')
    mutations = OperationSelector([5., 5, 2., 2, 3., 3.],
                                [RandomSlabPermutation(element_pools=['Pd', 'Ti'], num_muts=5),
                                RandomSlabPermutation(element_pools=['H', 'X'], num_muts=5),
                                SymmetrySlabPermutation(element_pools=['Pd', 'Ti'], num_muts=5),
                                SymmetrySlabPermutation(element_pools=['H', 'X'], num_muts=5),
                                CutSpliceSlabCrossover(element_pools=['Pd', 'Ti'], num_muts=5),
                                CutSpliceSlabCrossover(element_pools=['H', 'X'], num_muts=5),
                                ])
    with open('../../4body/eci_l1_PdTiH.json') as f:
        eci = json.load(f)
    settings = get_clease_settings()
    cf = CorrFunction(settings)
    # while db.get_number_of_unrelaxed_candidates() > 0:
    def relax_an_unrelaxed_candidate(a):
        a_copy = a.copy()
        a_copy = add_X(a_copy)
        a_copy = wrap_and_sort_by_position(a_copy)
        def run(i):
            eci_names = list(eci.keys())
            corr_func = cf.get_cf_by_names(a_copy, eci_names)
            clease_e = sum(eci[k]*corr_func[k] for k in eci_names)*256
            form_e = formation_energy_ref_metals(a_copy, clease_e)
            return clease_e, form_e

        clease_e, form_e = run(1)
        fitness = -form_e
        print('Relaxing starting candidate {0}'.format(a_copy.info['confid']))
        a.info['key_value_pairs']['raw_score'] = fitness
        a.info['key_value_pairs']['nnmat_string'] = get_nnmat_string(a_copy, 2, True)
        # a_copy.info['data']['clease_energies'] = clease_energies
        # a_copy.info['data']['form_energies'] = form_energies
        new_calc = SPC(a, energy=clease_e)
        a.set_calculator(new_calc) 
        db.add_relaxed_step(a) #insert optimized structures into database
    
    # pool = Pool(10)
    pool = Pool(os.cpu_count())
    print(db.get_all_unrelaxed_candidates())
    relaxed_candidates = pool.map(relax_an_unrelaxed_candidate, db.get_all_unrelaxed_candidates())  

    population.update()

    if multiprocessing:
        if 'SLURM_JOB_ID' in os.environ:
            shutil.copyfile(fname, db_name_withH)

    # Set convergence criteria
    cc = GenerationRepetitionConvergence(population, 5)

    gen_num = db.get_generation_number()
    max_gens = 20000 # maximum of generations
    for i in range(max_gens):
        # Check if converged
        if cc.converged():
            print('Converged')
            os.system('touch Converged')
            break
        
        def procreation(i):
            print('Creating and evaluating generation {0}'.format(gen_num + i))
            np.random.seed(random.randint(1,10000000))
            population.rng = np.random
            a1, a2 = population.get_two_candidates()
            # Do the operation
            op = mutations.get_operator()
            print('father:', a1)
            offspring = op.get_new_individual([a1, a2])
            a3, desc = offspring
            print(a3)
            nncomp = a3.get_chemical_formula(mode='hill')
            print('offspring: ' + nncomp)
            db.add_unrelaxed_candidate(a3, description=desc)
            a3_copy = a3.copy()
            a3_copy = add_X(a3_copy)
            a3_copy = wrap_and_sort_by_position(a3_copy)
            eci_names = list(eci.keys())
            corr_func = cf.get_cf_by_names(a3_copy, eci_names)
            clease_e = sum(eci[k]*corr_func[k] for k in eci_names)*256  
            form_e = formation_energy_ref_metals(a3_copy, clease_e)
            # fitness = - (mean_form_e - k_value*std_form_e)
            fitness = - form_e
            print('Relaxing starting candidate {0}'.format(a3_copy.info['confid']))
            a3.info['key_value_pairs']['raw_score'] = fitness
            a3.info['key_value_pairs']['nnmat_string'] = get_nnmat_string(a3_copy, 2, True)
            new_calc = SPC(a3, energy=clease_e)
            a3.set_calculator(new_calc) 
            db.add_relaxed_step(a3) # insert optimized structures into database
        
        # if multiprocessing:
        if True:
            # pool = Pool(10)
            pool = Pool(os.cpu_count())
            pool.map(procreation, range(10))
            pool.close()
            pool.join()
        else:
            procreation(1)
        # procreation()
        
        # update the population to allow new candidates to enter
        population.update()
        if multiprocessing:
            if 'SLURM_JOB_ID' in os.environ:
                shutil.copyfile(fname, db_name_withH)

