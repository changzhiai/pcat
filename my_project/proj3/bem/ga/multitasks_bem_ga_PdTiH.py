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
from acat.ga.multitasking import MultitaskPopulation
from acat.ga.multitasking import MultitaskGenerationRepetitionConvergence

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
from ase.neighborlist import NeighborList
from ase.units import kB


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

def get_gas_free_energy(ads, T=298.15, P=3534., geometry='nonlinear'):
    from ase.thermochemistry import IdealGasThermo
    import numpy as np
    energies = {}
    if ads == 'H2':
        atoms = Atoms(symbols='H2', pbc=True, cell=[15.0, 15.0, 15.0], positions=[(7.271282725527009, 7.5242228701978675, 7.67247819321021), (6.920516915473023, 7.176527003802101, 8.227171674789737)])
        energies['potential'] = -7.158
        energies['vib_energies'] = np.array([0.00937439716289498j, 0.008272872916201263j, 0.00032742164414834723j, (0.005878244084809664+0j), (0.012442148718103552+0j), (0.5487810154950278+0j)])
    elif ads == 'CO2':
        atoms = Atoms(symbols='CO2', pbc=True, cell=[15.0, 15.0, 15.0], positions=[(7.445530895066135, 7.445532525345406, 7.59940859119606), (8.27582705505936, 6.615291599705594, 7.599470868532947), (6.615292421874379, 8.275826246949087, 7.609470873270929)])
        energies['potential'] = -18.459
        energies['vib_energies'] = np.array([0.009326445700942517j, 0.0011444290930755579j, 0.00025577741354652754j, 0.00014501995032339908j, (0.005459458589350332+0j), (0.07802048032100792+0j), (0.07813913918325889+0j), (0.16321367651829186+0j), (0.2924408141103947+0j)])
    elif ads == 'H2O':
        atoms = Atoms(symbols='H2O', pbc=True, cell=[15.0, 15.0, 15.0], positions= [(7.407530500648641, 7.337146049188873, 8.826442364146274), (6.957262863279609, 8.116205530322027, 7.591395709428674), (7.42150681907166, 7.314099229488846, 7.871461662425041)])
        energies['vib_energies'] = np.array([0.020925276054351707j, 0.005409472594198771j, 0.0007208882189523696j, (0.003203591097425518+0j), (0.011879232687292142+0j), (0.021621631923307075+0j), (0.20087701898151597+0j), (0.46441623307420793+0j), (0.47841273786127075+0j)])
        energies['potential'] = -12.833
    elif ads == 'CO':
        atoms = Atoms(symbols='CO', pbc=True, cell=[[8.7782001495, 0.0, 0.0], [-4.3891000748, 7.602144329, 0.0], [0.0, 0.0, 27.1403007507]], positions=[(4.356747758338621, 2.5181998381431914, 14.909775815790633), (4.349119896661355, 2.51852500685681, 16.06160748820931)])
        energies['potential'] = -12.118
        energies['vib_energies'] = np.array([0.0047756513493744214j, 0.002608437221102516j, 7.154463200372838e-05j, (0.0014875269280506848+0j), (0.0018423337620198735+0j), (0.26328363577957453+0j)])

    thermo = IdealGasThermo(vib_energies=energies['vib_energies'],
                            potentialenergy=energies['potential'],
                            atoms=atoms,
                            geometry=geometry,  #linear or nonlinear
                            symmetrynumber=2, spin=0)
    energies['free_energy'] = thermo.get_gibbs_energy(temperature=T, pressure=P, verbose=False)
    # energies['entropy'] = thermo.get_entropy(temperature=T, pressure=P)
    # energies['TS'] = energies['entropy'] * 298.15
    # print('entropy:', energies['entropy'])
    # print('T*S:', energies['TS'])
    # print('gibbs free energy:',energies['free_energy'])
    return energies['free_energy']

def calculate_binding_free_e(atoms, Epot):
    """Get surface binding energy
    """
    adss = atoms.info['data']['ads_symbols']
    num_ads_H = adss.count('H')
    # num_ads_OH = adss.count('OH')
    # num_ads_CO = adss.count('CO')
    num_H = len([s for s in atoms.symbols if s == 'H']) - num_ads_H # H only in slab
    # num_C = len([s for s in atoms.symbols if s == 'C'])
    # num_O = len([s for s in atoms.symbols if s == 'O'])
    num_Pd = len([s for s in atoms.symbols if s == 'Pd'])
    # num_Ti = len([s for s in atoms.symbols if s == 'Ti'])
    A_surf = np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1]))
    mu_slab = {'PdH': -5.22219, 'Pd': -1.59002, 'Ti': -5.32613, 'TiH': -9.54306}
    T=298.15
    G_H2g = get_gas_free_energy(ads='H2', T=T, P=101325., geometry='linear')
    G_CO2g = get_gas_free_energy(ads='CO2', T=T, P=101325., geometry='linear')
    G_H2Og = get_gas_free_energy(ads='H2O', T=T, P=3534., geometry='nonlinear')
    G_COg = get_gas_free_energy(ads='CO', T=T, P=5562., geometry='linear')
    Gcor_H2g = 0.1 + 0 # overbinding and solvent correction
    Gcor_CO2g = 0.3 + 0
    Gcor_H2Og = 0
    Gcor_COg = 0
    G_H2g = G_H2g + Gcor_H2g
    G_CO2g = G_CO2g + Gcor_CO2g
    G_H2Og = G_H2Og + Gcor_H2Og
    G_COg = G_COg + Gcor_COg
    
    E_Pd_slab = -101.76139444 # 64 Pd atoms
    mu_Pd_bulk = -1.951 # per atom
    gamma_Pd = 1/(2*A_surf) * (E_Pd_slab - 64*mu_Pd_bulk)
    G_adss = 0
    for ads in adss:
        if ads == 'H':
            G_adss += 1/2. * G_H2g
            Gcor_H = 0.190 + 0.003 - 0.0000134*T + 0 + 0
            Epot += Gcor_H
        elif ads == 'OH':
            G_adss += G_H2Og - 0.5 * G_H2g
            Gcor_OH = 0.355 + 0.056 - 0.000345*T
            Epot += Gcor_OH
        elif ads == 'CO':
            G_adss += G_COg
            Gcor_CO = 0.186 + 0.080 - 0.000523*T + 0 - 0.10
            Epot += Gcor_CO
    gamma = 0
    if num_Pd >= num_H:
        gamma = 1/A_surf*(Epot - num_H*mu_slab['PdH']-(num_Pd-num_H)*mu_slab['Pd']-(64-num_Pd)*mu_slab['Ti']-G_adss)-gamma_Pd
    elif num_Pd < num_H:
        gamma = 1/A_surf*(Epot - num_Pd*mu_slab['PdH']-(num_H-num_Pd)*mu_slab['TiH']-(64-num_H)*mu_slab['Ti']-G_adss)-gamma_Pd
    return gamma
    
def calculate_gamma_old(atoms, Epot, T=298.15):
    """Get surface adsorption energy
    according to PdH, Pd3Ti and H2 gas
    """
    adss = atoms.info['data']['ads_symbols']
    num_ads_H = adss.count('H')
    num_H = len([s for s in atoms.symbols if s == 'H']) - num_ads_H # H only in slab
    num_Pd = len([s for s in atoms.symbols if s == 'Pd'])
    A_surf = np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1]))
    mu_slab = {'PdH': -5.22219, 'Pd': -1.59002, 'Ti': -5.32613, 'TiH': -9.54306}
    mu_bulk = {'PdH':-8.73007, 'Pd3Ti': -10.41178}
    G_H2g = get_gas_free_energy(ads='H2', T=T, P=101325., geometry='linear')
    G_CO2g = get_gas_free_energy(ads='CO2', T=T, P=101325., geometry='linear')
    G_H2Og = get_gas_free_energy(ads='H2O', T=T, P=3534., geometry='nonlinear')
    G_COg = get_gas_free_energy(ads='CO', T=T, P=5562., geometry='linear')
    Gcor_H2g = 0.1 + 0 # overbinding and solvent correction
    Gcor_CO2g = 0.3 + 0
    Gcor_H2Og = 0
    Gcor_COg = 0
    G_H2g = G_H2g + Gcor_H2g
    G_CO2g = G_CO2g + Gcor_CO2g
    G_H2Og = G_H2Og + Gcor_H2Og
    G_COg = G_COg + Gcor_COg
    
    E_Pd_slab = -101.76139444 # 64 Pd atoms
    mu_Pd_bulk = -1.951 # per atom
    gamma_Pd = 1/(2*A_surf) * (E_Pd_slab - 64*mu_Pd_bulk)
    G_adss = 0
    for ads in adss: # ads correction
        if ads == 'H':
            G_adss += 1/2. * G_H2g
            Gcor_H = 0.190 + 0.003 - 0.0000134*T + 0 + 0
            Epot += Gcor_H
        elif ads == 'OH':
            G_adss += G_H2Og - 0.5 * G_H2g
            Gcor_OH = 0.355 + 0.056 - 0.000345*T
            Epot += Gcor_OH
        elif ads == 'CO':
            G_adss += G_COg
            Gcor_CO = 0.186 + 0.080 - 0.000523*T + 0 - 0.10
            Epot += Gcor_CO
    gamma = 1/A_surf*(Epot - (67-4*num_Pd)*mu_bulk['PdH'] -
                      (1-num_Pd)*mu_bulk['Pd3Ti'] - 1/2*(num_H-4*num_Pd+3)*G_H2g - 64*mu_slab['PdH'] - G_adss)-gamma_Pd
    return gamma

def calculate_gamma(atoms, Epot, T=298.15, U=0, pH=0, P_H2=101325., P_CO2=101325., P_H2O=3534., P_CO=5562.):
    """Get surface adsorption energy
    Pure slab PdH: -5.22219 eV/atom
    Pure slab Pd: -1.59002 ev/atom
    Pure slab Ti: -5.32613 eV/atom
    
    Pure bulk Pd: -1.951 eV/atom
    Pure bulk Ti: -5.858 eV/atom
    H2 gas: -7.158 eV
    
    E_H2g = -7.158 # eV
    E_CO2g = -18.459
    E_H2Og = -12.833
    E_COg = -12.118
    
    # G_gas = E_pot + E_ZPE + C_Idel_gas - TS + Ecorr_overbing + E_solvent
    Gcor_H2g = 0.274 + 0.091 - 0.403 + 0.1 + 0
    Gcor_CO2g = 0.306 + 0.099 - 0.664 + 0.3 + 0
    Gcor_H2Og = 0.572 + 0.104 - 0.670
    Gcor_COg = 0.132 + 0.091 - 0.669
    G_H2g = E_H2g + Gcor_H2g
    G_CO2g = E_CO2g + Gcor_CO2g
    G_H2Og = E_H2Og + Gcor_H2Og
    G_COg = E_COg + Gcor_COg
    
    # G_gas = E_ZPE + C_harm - TS + Ecorr_overbing + E_solvent
    Gcor_H = 0.190 + 0.003 - 0.004 + 0 + 0
    Gcor_HOCO = 0.657 + 0.091 - 0.162 + 0.15 - 0.25
    Gcor_CO = 0.186 + 0.080 - 0.156 + 0 - 0.10
    Gcor_OH = 0.355 + 0.056 - 0.103
    """
    num_full = 16
    adss = atoms.info['data']['ads_symbols']
    num_ads_H = adss.count('H')
    # num_ads_OH = adss.count('OH')
    # num_ads_CO = adss.count('CO')
    num_H = len([s for s in atoms.symbols if s == 'H']) - num_ads_H # H only in slab
    d_num_H = num_H - num_full
    num_Pd = len([s for s in atoms.symbols if s == 'Pd'])
    d_num_Pd = num_Pd - num_full
    d_num_Ti = -d_num_Pd
    A_surf = np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1]))
    mu_slab = {'PdH': -5.22219, 'Pd': -1.59002, 'Ti': -5.32613, 'TiH': -9.54306}
    mu_bulk = {'PdH':-8.73007, 'Pd3Ti': -10.41178, 'Pd': -1.951}
    d_mu_equi = {'Pd': -2.24900, 'Ti': -7.28453, 'H': -3.61353}
    G_H2g = get_gas_free_energy(ads='H2', T=T, P=P_H2, geometry='linear')
    G_CO2g = get_gas_free_energy(ads='CO2', T=T, P=P_CO2, geometry='linear')
    G_H2Og = get_gas_free_energy(ads='H2O', T=T, P=P_H2O, geometry='nonlinear')
    G_COg = get_gas_free_energy(ads='CO', T=T, P=P_CO, geometry='linear')
    Gcor_H2g = 0.1 + 0 # overbinding and solvent correction
    Gcor_CO2g = 0.3 + 0
    Gcor_H2Og = 0
    Gcor_COg = 0
    G_H2g = G_H2g + Gcor_H2g
    G_CO2g = G_CO2g + Gcor_CO2g
    G_H2Og = G_H2Og + Gcor_H2Og
    G_COg = G_COg + Gcor_COg
    
    # E_Pd_slab = -101.76139444 # 64 Pd atoms
    # mu_Pd_bulk = -1.951 # per atom
    gamma_Pd = 1/(2*A_surf) * (num_full*mu_slab['Pd'] - num_full*mu_bulk['Pd'])
    G_adss = 0
    for ads in adss: # ads correction
        if ads == 'H':
            G_adss += 0.5 * G_H2g - U - kB * T * pH * np.log(10)
            Gcor_H = 0.190 + 0.003 - 0.0000134*T + 0 + 0
            Epot += Gcor_H
        elif ads == 'OH':
            G_adss += G_H2Og - 0.5 * G_H2g + U + kB * T * pH * np.log(10)  # !!! change into liquid H2O
            Gcor_OH = 0.355 + 0.056 - 0.000345*T
            Epot += Gcor_OH
        elif ads == 'CO':
            G_adss += G_COg
            Gcor_CO = 0.186 + 0.080 - 0.000523*T + 0 - 0.10
            Epot += Gcor_CO
    gamma = 1/A_surf*(Epot -d_num_Pd*d_mu_equi['Pd'] -
                      d_num_Ti*d_mu_equi['Ti'] - d_num_H*d_mu_equi['H'] - num_full*mu_slab['PdH'] - G_adss)-gamma_Pd
    return gamma

def relax(atoms, single_point=True):
    # t1 = time.time()
    atoms.calc = EMT()
    if not single_point:
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=0.1)
    Epot = atoms.get_potential_energy()
    # Calculate_mixing_energy()
    atoms.info['key_value_pairs']['potential_energy'] = Epot # for compator
    atoms.info['key_value_pairs']['nnmat_string'] = get_nnmat_string(atoms, 2, True) # for compator
    # atoms.info['data']['nnmat'] = get_nnmat(atoms)
    # atoms.info['key_value_pairs']['raw_score'] = -Epot
    Ts = np.arange(200., 300., 1.)
    fs = np.array([calculate_gamma(atoms, Epot, T=T) for T in Ts])
    atoms.info['data']['raw_scores'] = fs
    # t2 = time.time()
    # print('Relaxing time:', t2-t1)
    atoms.calc = None
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
        # db.add_more_relaxed_candidates(relaxed_candidates)
    else:
        relaxed_candidates = []
        images = db.get_all_unrelaxed_candidates()
        # while db.get_number_of_unrelaxed_candidates() > 0:
        for atoms in images:
            t_start = time.time()
            # atoms = db.get_an_unrelaxed_candidate()
            a = relax_an_unrelaxed_candidate(atoms)
            # print(a.info)
            # db.add_relaxed_step(a) # insert optimized structures into database
            relaxed_candidates.append(a)
            t_end = time.time()
            print('initial relaxing:', t_end-t_start)
    population.update(new_cand=relaxed_candidates)
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
            break
    # nncomp = a3.get_chemical_formula(mode='hill')
    # print('Relaxing: ' + nncomp)
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
        # db.add_more_relaxed_candidates(relaxed_candidates)
        # population.update()
        population.update(new_cand=relaxed_candidates)
    else:
        relaxed_candidates = []
        for _ in range(pop_size):
            t1 = time.time()
            a = relax_offspring(1)
            a.info['key_value_pairs']['generation'] = gens_running
            # db.add_relaxed_step(a) # insert optimized structure into database
            # population.update() # ! update after add one new atoms
            # population.update(new_cand=[a])
            relaxed_candidates.append(a)
            t2 = time.time()
            print('offspring relaxing time:', t2-t1)
    population.update(new_cand=relaxed_candidates)
    t_end = time.time()
    print('generation timing:', t_end-t_start)
    if copy_to_scratch:
        if 'SLURM_JOB_ID' in os.environ:
            shutil.copyfile(fname, db_name)
    return db

def check_adsorbates(atoms):
    """Check if ads_indices corresponds ads_symbols"""
    ads_indices = atoms.info['data']['ads_indices']
    ads_symbols = atoms.info['data']['ads_symbols']
    for ads_index, ads_symbol in zip(ads_indices, ads_symbols):
        print(str(atoms[ads_index].symbols), ads_symbol)
    print('-----')
    for ads_index, ads_symbol in zip(ads_indices, ads_symbols):
        print(str(atoms[ads_index].symbols), ads_symbol)
        for i in ads_index:
            s = atoms[i].symbol
            assert (s in ads_symbol)
        # assert str(atoms[ads_index].symbols)==ads_symbol
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


if __name__ == '__main__':
    
    continue_ga = False
    copy_to_scratch = False
    k_value = 4
    cores = 1 # os.cpu_count()
    db_name = 'ga_PdTiH.db'
    pop_size = 10
    # check_db(db_name)
    # assert False
    temp = read('./template_2x2.traj')
    init_images = read('../random50_adss.traj', index=':')
    if continue_ga == False:
        db = generate_initial_ga_db(db_name, pop_size, temp, init_images)
    if copy_to_scratch:
        db, fname = copy_to_scratch(db_name)
    else:
        db = DataConnection(db_name)
    
    op_selector = OperationSelector([1, 1, 1, 1, 1, 1, 1, 1, 1],
                                [
                                RandomMetalPermutation(element_pools=['Pd', 'Ni'], num_muts=5),
                                RandomMetalComposition(element_pools=['Pd', 'Ni'], num_muts=5),
                                SymmetrySlabPermutation(element_pools=['Pd', 'Ni'], num_muts=1),
                                InternalHydrogenAddition(internal_H_pools=['H'], num_muts=5),
                                InternalHydrogenRemoval(internal_H_pools=['H'], num_muts=5),
                                InternalHydrogenMoveToUnoccupied(internal_H_pools=['H'], num_muts=5),
                                # AdsorbateSubstitution(ads_pools=['CO', 'OH', 'H'], num_muts=1),
                                AdsorbateAddition(ads_pools=['CO', 'OH', 'H'], num_muts=1),
                                AdsorbateRemoval(ads_pools=['CO', 'OH', 'H'], num_muts=1),
                                AdsorbateSwapOccupied(ads_pools=['CO', 'OH', 'H'], num_muts=1), # the problem
                                # AdsorbateMoveToUnoccupied(ads_pools=['CO', 'OH', 'H'], num_muts=1),
                                # AdsorbateCutSpliceCrossover(ads_pools=['CO', 'OH', 'H'], num_muts=1),
                                ])
    
    # Define the tasks. In this case we use 10 different chemical potentials of CH4
    Ts = np.arange(200., 300., 1.)
    # Initialize the population and specify the number of tasks
    population = MultitaskPopulation(data_connection=db,
                              population_size=pop_size,
                              num_tasks=len(Ts),
                              comparator=get_comparators(db),
                              exp_function=True,
                              logfile='log.txt')
    
    # relax initial population
    db = relax_init_pop(cores, db)
    print('initial optimization done')
    gen_num = db.get_generation_number()
    max_gens = 20000 # maximum of generations
    # cc = GenerationRepetitionConvergence(population, 5) # Set convergence criteria
    cc = MultitaskGenerationRepetitionConvergence(population, 5)
    for i in range(max_gens):
        if cc.converged():  # Check if converged
            print('Converged')
            os.system('touch Converged')
            break
        gens_running = gen_num + i
        print('\nCreating and evaluating generation {0}'.format(gens_running))
        db = relax_generation(cores, db, gens_running)
        # population.update()
    
   
