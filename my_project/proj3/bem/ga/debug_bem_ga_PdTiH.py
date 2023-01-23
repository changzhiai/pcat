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
from ase.constraints import FixAtoms
from ase.ga.standard_comparators import SequentialComparator
from ase.ga.standard_comparators import InteratomicDistanceComparator
from ase.ga.standard_comparators import StringComparator
from ase.ga.utilities import get_nnmat_string
from ase.ga.ofp_comparator import OFPComparator
from slab_operators_hydride import SymmetrySlabPermutation
from slab_operators_hydride import (RandomMetalPermutation,
                                           RandomMetalComposition,
                                           AdsorbateSubstitution, 
                                           AdsorbateAddition, 
                                           AdsorbateRemoval,
                                           AdsorbateSwapOccupied,
                                           AdsorbateMoveToUnoccupied,
                                           AdsorbateCutSpliceCrossover,
                                           InternalHydrogenAddition,
                                           InternalHydrogenRemoval,
                                           InternalHydrogenMoveToUnoccupied)
from acat.ga.adsorbate_operators import (AddAdsorbate, RemoveAdsorbate,
                                         MoveAdsorbate, ReplaceAdsorbate,
                                         SimpleCutSpliceCrossoverWithAdsorbates)
from acat.adsorption_sites import ClusterAdsorptionSites
from acat.ga.adsorbate_comparators import AdsorptionSitesComparator
from ase.ga.utilities import get_nnmat
from ase.ga.utilities import get_nnmat_string
from ase.db import connect
from ase.constraints import UnitCellFilter
from ase.optimize import BFGS
import json
import sys
from multiprocessing import Pool
import os
import shutil
import string
import random
import time
import re
from ase.calculators.singlepoint import SinglePointCalculator as SPC

def remove_X(atoms):
    try:
        del atoms[[atom.index for atom in atoms if atom.symbol == 'X']]
    except:
        atoms = atoms
    return atoms

def remove_top_X(atoms):
    del atoms[[atom.index for atom in atoms if atom.symbol == 'X' and atom.z > 9.]]
    return atoms

def add_top_X(atoms):
    system = read('/home/energy/changai/ce_PdxTiHy/random/ga/multiprocessing/template_from_clease.traj')
    for atom in system:
        if atom.z > 9.0:
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

def get_ordered_init(db_name):
    if os.path.exists(db_name):
        os.remove(db_name)
    system = read('/home/energy/changai/ce_PdxTiHy/random/ga/multiprocessing/template_from_clease.traj')
    del system[[atom.index for atom in system if atom.symbol=='X']]
    alloy = system.copy()
    alloy_real = system.copy()
    del alloy[[atom.index for atom in alloy if atom.symbol=='H']] # delete H
    # pop_size = 5
    # rog = ROG(alloy, elements=["Pd", "Ti"], composition={"Pd": 0.5, "Ti": 0.5}, trajectory='orderings.traj')
    # rog = ROG(alloy, elements=["Pd", "Ti"], trajectory='orderings.traj')
    # rog.run(num_gen=pop_size)
    images = read("/home/energy/changai/ce_PdxTiHy/random/ga/multiprocessing/orderings.traj", index=":")
    for image in images: # add H to all images
        image = add_H(image, system)
        image = random_H(image, include_X = True)
    atom_numbers = 32 * [46] + 32 * [79]
    db = PrepareDB(db_file_name=db_name, # Instantiate the db
                  simulation_cell=alloy_real, # with H
                  stoichiometry=atom_numbers)
    for atoms in images:
        if 'data' not in atoms.info:
            atoms.info['data'] = {}
        # atoms = remove_X(atoms)
        db.add_unrelaxed_candidate(atoms, data=atoms.info['data'])

def get_comparators(db):
    """Get comparators"""
    # atom_numbers_to_optimize = db.get_atom_numbers_to_optimize()
    # n_to_optimize = len(atom_numbers_to_optimize)
    # compIDC = InteratomicDistanceComparator(n_top=n_to_optimize,
    #                                     pair_cor_cum_diff=0.015,
    #                                     pair_cor_max=0.7,
    #                                     dE=0.02,
    #                                     mic=False)
    compSring = StringComparator('potential_energy')
    compSC = StringComparator('nnmat_string')
    # comp = SequentialComparator([compIDC, compSC], [0, 1])
    comp = SequentialComparator([compSring, compSC], [0, 1])
    return comp


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

def continue_ga(db_name):
    if os.path.exists(db_name):
        os.remove(db_name)
    system = read('/home/energy/changai/ce_PdxTiHy/random/ga/multiprocessing/template_from_clease.traj')
    del system[[atom.index for atom in system if atom.symbol=='X']]
    alloy = system.copy()
    alloy_real = system.copy()
    del alloy[[atom.index for atom in alloy if atom.symbol=='H']] # delete H
    atom_numbers = 32 * [46] + 32 * [79]
    db = PrepareDB(db_file_name=db_name, # Instantiate the db
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

def generate_initial_ga_db(db_name, pop_size, temp=None, init_images=None):
    if os.path.exists(db_name):
        os.system('cp {} backup.db'.format(db_name))
        os.remove(db_name)
    # temp = read('~/bem/PdTiH/ga/template_2x2.traj')
    # temp = read('./template_2x2.traj')
    db = PrepareDB(db_file_name=db_name, # Instantiate the db
                  cell=temp.cell, 
                  population_size=pop_size)
    # init_images = read('../random50_adss.traj', index=':')
    random.seed(888)
    for atoms in init_images:
        if 'data' not in atoms.info:
            atoms.info['data'] = {}
        atoms.info['data']['ads_indices'] = atoms.info['ads_indices']
        atoms.info['data']['ads_symbols'] = atoms.info['ads_symbols']
        atoms.calc = None
        db.add_unrelaxed_candidate(atoms, data=atoms.info['data'])
    return db

def copy_to_scratch(db_name):
    if 'SLURM_JOB_ID' in os.environ: # The script is submitted
        dname = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        fname = '/scratch/changai/' + dname + '-' + db_name
        shutil.copyfile(db_name, fname)
        db = DataConnection(fname)
    else:
        db = DataConnection(db_name)
    return db, fname

def vf(atoms):
    """Returns the descriptor that distinguishes candidates in the
    niched population."""
    return atoms.get_chemical_formula(mode='hill')

def relax(atoms, single_point=True):
    # t1 = time.time()
    atoms.calc = EMT()
    if not single_point:
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=0.1)
    Epot = atoms.get_potential_energy()
    atoms.calc = None
    # Calculate_mixing_energy()
    atoms.info['key_value_pairs']['potential_energy'] = Epot # for compator
    atoms.info['key_value_pairs']['nnmat_string'] = get_nnmat_string(atoms, 2, True) # for compator
    # atoms.info['data']['nnmat'] = get_nnmat(atoms)
    atoms.info['key_value_pairs']['raw_score'] = -Epot
    # t2 = time.time()
    # print('Relaxing time:', t2-t1)
    print('Relaxing offspring candidate {0}'.format(atoms.info['confid']))
    return atoms

def relax_an_unrelaxed_candidate(atoms):
    """Relax one unrelaxed candidate in population pool"""
    if 'data' not in atoms.info:
        atoms.info['data'] = {'tag': None}
    # nncomp = atoms.get_chemical_formula(mode='hill')
    # print('Relaxing ' + nncomp)
    return relax(atoms)

def relax_init_pop(cores, db):
    """Relax the initial structures in population pool"""
    if cores > 1:
        pool = Pool(cores)
        relaxed_candidates = pool.map(relax_an_unrelaxed_candidate, db.get_all_unrelaxed_candidates())  
        pool.close()
        pool.join()
        db.add_more_relaxed_candidates(relaxed_candidates)
    else:
        # while db.get_number_of_unrelaxed_candidates() > 0:
        #     t_start = time.time()
        #     atoms = db.get_an_unrelaxed_candidate()
        #     a = relax_an_unrelaxed_candidate(atoms)
        #     # print(a.info)
        #     db.add_relaxed_step(a) # insert optimized structures into database
        #     t_end = time.time()
        #     print('initial relaxing:', t_end-t_start)
        relaxed_candidates = []
        for atoms in db.get_all_unrelaxed_candidates():
            a = relax_an_unrelaxed_candidate(atoms)
            relaxed_candidates.append(a)
            check_adsorbates(a)
        db.add_more_relaxed_candidates(relaxed_candidates)
    population.update()
    if copy_to_scratch:
        if 'SLURM_JOB_ID' in os.environ:
            shutil.copyfile(fname, db_name)
    return db
            
def relax_offspring(n):
    """Relax the one offspring structure"""
    t1 = time.time()
    op = op_selector.get_operator()  # Do the operation
    print(op)
    while True:
        np.random.seed(random.randint(1,100000))
        population.rng = np.random
        a1, a2 = population.get_two_candidates()
        offspring = op.get_new_individual([a1, a2])
        a3, desc = offspring
        # view(a3)
        if a3 is not None:
            check_adsorbates(a3)
            break
    if 'data' not in a3.info:
        a3.info['data'] = {}
    a3 = relax(a3, single_point=True)
    t2 = time.time()
    print('offspring relaxing time:', t2-t1)
    return a3
        
def relax_generation(cores, db, gens_running):
    """Relax the all offspring structures of one generation"""
    t_start = time.time()
    if cores > 1: 
        pool = Pool(cores)
        relaxed_candidates = pool.map(relax_offspring, range(pop_size))
        for atoms in relaxed_candidates:
            atoms.info['key_value_pairs']['generation'] = gens_running
        pool.close()
        pool.join()
        db.add_more_relaxed_candidates(relaxed_candidates)
        population.update()
    else:
        relaxed_candidates = []
        for _ in range(pop_size):
            t1 = time.time()
            a = relax_offspring(1)
            check_adsorbates(a)
            a.info['key_value_pairs']['generation'] = gens_running
            # db.add_relaxed_step(a) # insert optimized structure into database
            # population.update() # !!! update after add one new atoms
            relaxed_candidates.append(a)
            t2 = time.time()
            print('offspring relaxing time:', t2-t1)
        db.add_more_relaxed_candidates(relaxed_candidates)
        population.update()
    t_end = time.time()
    print('generation timing:', t_end-t_start)
    if copy_to_scratch:
        if 'SLURM_JOB_ID' in os.environ:
            shutil.copyfile(fname, db_name)
    return db

def check_adss_num(atoms):
    ads_indices = atoms.info['data']['ads_indices']
    ads_symbols = atoms.info['data']['ads_symbols']
    symbols = atoms.symbols
    num_C = symbols.count('C')
    num_O = symbols.count('O')
    num_O = num_O - num_C
    num_CO = ads_symbols.count('CO')
    num_OH = ads_symbols.count('OH')
    print(num_C, num_CO)
    print(num_O, num_OH)
    try:
        assert num_C == num_CO
        assert num_O == num_OH
    except:
        view(atoms)
        assert False

def check_adsorbates(atoms):
    """Check if ads_indices corresponds ads_symbols"""
    ads_indices = atoms.info['data']['ads_indices']
    ads_symbols = atoms.info['data']['ads_symbols']
    for ads_index, ads_symbol in zip(ads_indices, ads_symbols):
        print(str(atoms[ads_index].symbols), ads_symbol)
    print('---check--')
    for ads_index, ads_symbol in zip(ads_indices, ads_symbols):
        print(str(atoms[ads_index].symbols), ads_symbol)
        for i in ads_index:
            s = atoms[i].symbol
            assert (s in ads_symbol)
        # assert str(atoms[ads_index].symbols)==ads_symbol
    check_adss_num(atoms)
    return ads_indices, ads_symbols

def check_tags(atoms):
    """Check tags in atoms"""
    for layer in [1, 2, 3, 4]:
        indices_existed_Ms = [atom.index for atom in atoms if atom.tag in [layer]] # ist bilayer H
        assert len(indices_existed_Ms)<=4
        indices_existed_Hs = [atom.index for atom in atoms if atom.tag in [layer+4]] # ist bilayer H
        assert len(indices_existed_Hs)<=4
    return True

def check_db(db_name):
    """Check atoms in database"""
    db = DataConnection(db_name)
    for atoms in db.get_all_relaxed_candidates():
        # _, _ = check_adsorbates(atoms)
        view(atoms)
        _ = check_tags(atoms)
    print('done')
    
def update_H_tags_after_dft_relax(atoms):
    """Update H tag because H atoms will segregate and thus the belonging layer of H will change """
    bilayers = [1, 2, 3, 4]
    all_Hs = [atom.index for atom in atoms if atom.symbol=='H' and atom.tag >=-1]
    for bilayer in bilayers:
        z_current = np.mean([atom.z for atom in atoms if atom.tag==bilayer])
        for index_H in all_Hs:
            atom = atoms[index_H]
            if atom.z >= z_current:
                old_tag = atom.tag
                atom.tag = bilayer + 4
                ads_indices = atoms.info['data']['ads_indices']
                ads_symbols = atoms.info['data']['ads_symbols']
                if bilayer == 1 and old_tag > atom.tag: # H on the surface,from deeper layer
                    ads_indices.append(index_H)
                    ads_symbols.append('H')
                    
                elif bilayer == 2 and old_tag < atom.tag:
                    i = ads_indices.index(index_H)
                    del ads_indices[i]
                    del ads_symbols[i]
                all_Hs.remove(index_H)
    _, _ = check_adsorbates(atoms)
    _ = check_tags(atoms)
                

if __name__ == '__main__':
    
    continue_ga = False
    copy_to_scratch = False
    k_value = 4
    cores = 1 # os.cpu_count()
    db_name = 'ga_PdTiH.db'
    pop_size = 10
    # check_db(db_name)
    temp = read('./template_2x2.traj')
    init_images = read('../random50_adss.traj', index=':')
    if continue_ga == False:
        db = generate_initial_ga_db(db_name, pop_size, temp, init_images)
    if copy_to_scratch:
        db, fname = copy_to_scratch(db_name)
    else:
        db = DataConnection(db_name)

    
    # op_selector = OperationSelector([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # 
    #                             [
    #                             RandomMetalPermutation(element_pools=['Pd', 'Ni'], num_muts=5),
    #                             RandomMetalComposition(element_pools=['Pd', 'Ni'], num_muts=5),
    #                             SymmetrySlabPermutation(element_pools=['Pd', 'Ni'], num_muts=1),
    #                             InternalHydrogenAddition(internal_H_pools=['H'], num_muts=5),
    #                             InternalHydrogenRemoval(internal_H_pools=['H'], num_muts=5),
    #                             InternalHydrogenMoveToUnoccupied(internal_H_pools=['H'], num_muts=5),
    #                             AdsorbateAddition(ads_pools=['CO', 'OH', 'H'], num_muts=2),
    #                             AdsorbateRemoval(ads_pools=['CO', 'OH', 'H'], num_muts=2),
    #                             AdsorbateSubstitution(ads_pools=['CO', 'OH', 'H'], num_muts=2),
    #                             AdsorbateSwapOccupied(ads_pools=['CO', 'OH', 'H'], num_muts=2),
    #                             AdsorbateMoveToUnoccupied(ads_pools=['CO', 'OH', 'H'], num_muts=1),
    #                             AdsorbateCutSpliceCrossover(ads_pools=['CO', 'OH', 'H'], num_muts=1),
    #                             ])
    
    op_selector = OperationSelector([1], 
                                [
                                # RandomMetalPermutation(element_pools=['Pd', 'Ni'], num_muts=5),
                                # RandomMetalComposition(element_pools=['Pd', 'Ni'], num_muts=5),
                                # SymmetrySlabPermutation(element_pools=['Pd', 'Ni'], num_muts=1),
                                # InternalHydrogenAddition(internal_H_pools=['H'], num_muts=5),
                                # InternalHydrogenRemoval(internal_H_pools=['H'], num_muts=5),
                                # InternalHydrogenMoveToUnoccupied(internal_H_pools=['H'], num_muts=1),
                                # AdsorbateAddition(ads_pools=['CO', 'OH', 'H'], num_muts=2),
                                # AdsorbateRemoval(ads_pools=['CO', 'OH', 'H'], num_muts=2),
                                # AdsorbateSubstitution(ads_pools=['CO', 'OH', 'H'], num_muts=2),
                                # AdsorbateSwapOccupied(ads_pools=['CO', 'OH', 'H'], num_muts=2),
                                # AdsorbateMoveToUnoccupied(ads_pools=['CO', 'OH', 'H'], num_muts=1),
                                AdsorbateCutSpliceCrossover(ads_pools=['CO', 'OH', 'H'], num_muts=1),
                                ])
  
    population = RankFitnessPopulation(data_connection=db,
                            population_size=pop_size,
                            comparator=get_comparators(db),
                            variable_function=vf,
                            exp_function=True,
                            logfile='log.txt')
    # population = Population(data_connection=db,
    #              population_size=pop_size,
    #              comparator=get_comparators(db),
    #              logfile='log.txt')
    
    # relax initial population
    db = relax_init_pop(cores, db)
    
    gen_num = db.get_generation_number()
    max_gens = 20000 # maximum of generations
    cc = GenerationRepetitionConvergence(population, 5) # Set convergence criteria
    for i in range(max_gens):
        if cc.converged():  # Check if converged
            print('Converged')
            os.system('touch Converged')
            break
        gens_running = gen_num + i
        print('\nCreating and evaluating generation {0}'.format(gens_running))
        db = relax_generation(cores, db, gens_running)
        # population.update()
        
   
