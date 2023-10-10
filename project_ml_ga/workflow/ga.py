"""Genetic algorithim for slab with different compostion, different adsorbates, different adsorption sites and differnt coverages"""
from ase import Atoms
from ase.build import surface, bulk, cut
from ase.visualize import view
import numpy as np
from ase.thermochemistry import IdealGasThermo
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
from ase.calculators.emt import EMT
from ase.ga.standard_comparators import SequentialComparator
from ase.ga.standard_comparators import InteratomicDistanceComparator
from ase.ga.standard_comparators import StringComparator
from ase.ga.utilities import get_nnmat_string
from ase.ga.ofp_comparator import OFPComparator
from pcat.ga.operators_hydride import SymmetrySlabPermutation
from pcat.ga.operators_hydride import (RandomMetalPermutation,
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
from pcat.ga.comparators_hydride import DataComparator
from acat.ga.adsorbate_operators import (AddAdsorbate, RemoveAdsorbate,
                                         MoveAdsorbate, ReplaceAdsorbate,
                                         SimpleCutSpliceCrossoverWithAdsorbates)
from acat.adsorption_sites import ClusterAdsorptionSites
from acat.ga.adsorbate_comparators import AdsorptionSitesComparator
from acat.ga.multitasking import MultitaskPopulation
from acat.ga.graph_comparators import WLGraphComparator
from acat.utilities import neighbor_shell_list, get_adj_matrix
from pcat.build.generate_random_ads import generate_random_init_structs_n_ads
from ase.ga.utilities import get_nnmat
from ase.ga.utilities import get_nnmat_string
from ase.db import connect
from ase.constraints import UnitCellFilter
from ase.optimize import BFGS
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
import torch
from PaiNN.calculator import MLCalculator, EnsembleCalculator
from PaiNN.model import PainnModel
import toml
import argparse
import pandas as pd
from ase.units import kB
import copy
import logging
import types
from ase.geometry import find_mic

def basic_logger():
    logger = logging.basicConfig(
                    filename='basic.log',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    return logger

def log_newline(self, lines=1):
    self.h.setFormatter(self.blank_formatter)
    for i in range(lines):
        self.info('')
    self.h.setFormatter(self.formatter)

def create_file_logger(name='log.basic_ga', remove_basic_log=False):
    if remove_basic_log and os.path.exists(name):
        os.remove(name)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    blank_formatter = logging.Formatter('')
    fh = logging.FileHandler(name) # file handler 
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh) 
    logger.h = fh
    logger.formatter = formatter
    logger.blank_formatter = blank_formatter
    logger.newline = types.MethodType(log_newline, logger)
    return logger

def create_console_logger():
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    blank_formatter = logging.Formatter('')
    ch = logging.StreamHandler() # console handler
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    logger.h = ch
    logger.formatter = formatter
    logger.blank_formatter = blank_formatter
    logger.newline = types.MethodType(log_newline, logger)
    return logger

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

class EnergyObservor:
    def __init__(self, atoms):
        self.atoms = atoms
        print("Energy observor")

    def __call__(self, threshold=0):
        energy = self.atoms.get_potential_energy()
        forces = self.atoms.get_forces()
        # print(f'energy: {energy}')
        if energy > threshold or np.isnan(energy):
            print('energy is too large or None')
            print(f'energy: {energy}')
            raise ValueError('energy is too large or None')

def get_last_n(num = 50, db_name = '../PdTiH_ga.db', lastn_name = 'lastn_PdTiH_ga.db'):
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

def get_comparators(db):
    """Get comparators"""
    compSring = StringComparator('potential_energy') 
    compSC = DataComparator('nnmat_string')
    compGC = WLGraphComparator(dx=0.50)
    comp = SequentialComparator([compSring, compSC, compGC], [0, 1, 2])
    return comp

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
    energies = {}
    if ads == 'H2':
        atoms = Atoms(symbols='H2', pbc=True, cell=[15.0, 15.0, 15.0], positions=[(7.271282725527009, 7.5242228701978675, 7.67247819321021), 
            (6.920516915473023, 7.176527003802101, 8.227171674789737)])
        energies['potential'] = -7.158
        energies['vib_energies'] = np.array([0.00937439716289498j, 0.008272872916201263j, 0.00032742164414834723j, (0.005878244084809664+0j), 
            (0.012442148718103552+0j), (0.5487810154950278+0j)])
    elif ads == 'CO2':
        atoms = Atoms(symbols='CO2', pbc=True, cell=[15.0, 15.0, 15.0], positions=[(7.445530895066135, 7.445532525345406, 7.59940859119606), 
            (8.27582705505936, 6.615291599705594, 7.599470868532947), (6.615292421874379, 8.275826246949087, 7.609470873270929)])
        energies['potential'] = -18.459
        energies['vib_energies'] = np.array([0.009326445700942517j, 0.0011444290930755579j, 0.00025577741354652754j, 0.00014501995032339908j,
         (0.005459458589350332+0j), (0.07802048032100792+0j), (0.07813913918325889+0j), (0.16321367651829186+0j), (0.2924408141103947+0j)])
    elif ads == 'H2O':
        atoms = Atoms(symbols='H2O', pbc=True, cell=[15.0, 15.0, 15.0], positions= [(7.407530500648641, 7.337146049188873, 8.826442364146274), 
            (6.957262863279609, 8.116205530322027, 7.591395709428674), (7.42150681907166, 7.314099229488846, 7.871461662425041)])
        energies['vib_energies'] = np.array([0.020925276054351707j, 0.005409472594198771j, 0.0007208882189523696j, (0.003203591097425518+0j), 
            (0.011879232687292142+0j), (0.021621631923307075+0j), (0.20087701898151597+0j), (0.46441623307420793+0j), (0.47841273786127075+0j)])
        energies['potential'] = -12.833
    elif ads == 'CO':
        atoms = Atoms(symbols='CO', pbc=True, cell=[[8.7782001495, 0.0, 0.0], [-4.3891000748, 7.602144329, 0.0], [0.0, 0.0, 27.1403007507]], 
            positions=[(4.356747758338621, 2.5181998381431914, 14.909775815790633), (4.349119896661355, 2.51852500685681, 16.06160748820931)])
        energies['potential'] = -12.118
        energies['vib_energies'] = np.array([0.0047756513493744214j, 0.002608437221102516j, 7.154463200372838e-05j, 
            (0.0014875269280506848+0j), (0.0018423337620198735+0j), (0.26328363577957453+0j)])

    thermo = IdealGasThermo(vib_energies=energies['vib_energies'],
                            potentialenergy=energies['potential'],
                            atoms=atoms,
                            geometry=geometry,  #linear or nonlinear
                            symmetrynumber=2, spin=0)
    energies['free_energy'] = thermo.get_gibbs_energy(temperature=T, pressure=P, verbose=False)
    return energies['free_energy']

def calculate_binding_free_e(atoms, Epot):
    """Get surface binding energy
    """
    adss = atoms.info['data']['ads_symbols']
    num_ads_H = adss.count('H')
    num_H = len([s for s in atoms.symbols if s == 'H']) - num_ads_H # H only in slab
    num_Pd = len([s for s in atoms.symbols if s == 'Pd'])
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

def get_element_nums(atoms):
    "Calculate how many atoms for each element"
    num_full = 16
    adss = atoms.info['data']['ads_symbols']
    num_ads_OH = adss.count('OH')
    num_ads_CO = adss.count('CO')
    ave_z = np.mean([atom.z for atom in atoms if atom.tag==1])
    num_ads_H = len([atom for atom in atoms if atom.z >= ave_z and atom.symbol=='H'])-num_ads_OH
    num_H = len([s for s in atoms.symbols if s == 'H']) - num_ads_H - num_ads_OH # H only in slab
    d_num_H = num_H - num_full
    num_Pd = len([s for s in atoms.symbols if s == 'Pd'])
    d_num_Pd = num_Pd - num_full
    d_num_Ti = -d_num_Pd
    num_slab_H = num_H
    return d_num_Pd, d_num_Ti, d_num_H, num_ads_CO, num_ads_OH, num_ads_H, num_slab_H

def calculate_gamma(r): 
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
    Epot = r['Epot']
    d_mu_Pd=r['d_mu_Pd']
    d_mu_Ti=r['d_mu_Ti']
    d_mu_H=r['d_mu_H']
    T=r['T']
    U=r['U']
    pH=r['pH']
    P_H2=r['P_H2']
    P_CO2=r['P_CO2']
    P_H2O=r['P_H2O']
    P_CO=r['P_CO']
    num_full = 16
    d_num_Pd = r['d_num_Pd']
    d_num_Ti = r['d_num_Ti']
    d_num_H = r['d_num_H']
    num_ads_CO = r['num_ads_CO'].astype(int)
    num_ads_OH = r['num_ads_OH'].astype(int)
    num_ads_H = r['num_ads_H'].astype(int)
    num_slab_H = r['num_slab_H'].astype(int)
    mu_slab = {'PdH': -5.22219, 'Pd': -1.59002, 'Ti': -5.32613, 'TiH': -9.54306}
    mu_bulk = {'PdH':-8.73007, 'Pd3Ti': -10.41178, 'Pd': -1.951}
    # d_mu_equi = {'Pd': -2.24900, 'Ti': -7.28453, 'H': -3.61353}
    d_mu_equi = {'Pd': d_mu_Pd, 'Ti': d_mu_Ti, 'H': d_mu_H}
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
    gamma_Pd = 1/2 * (num_full*mu_slab['Pd'] - num_full*mu_bulk['Pd'])
    G_adss = 0
    for _ in range(num_ads_OH):
        G_adss += G_H2Og - 0.5 * G_H2g + U + kB * T * pH * np.log(10)
        Gcor_OH = 0.355 + 0.056 - 0.000345*T
        Epot += Gcor_OH
    for _ in range(num_ads_CO):
        G_adss += G_COg
        Gcor_CO = 0.186 + 0.080 - 0.000523*T + 0 - 0.10
        Epot += Gcor_CO
    for _ in range(num_ads_H):
        G_adss += 0.5 * G_H2g - U - kB * T * pH * np.log(10)
        Gcor_H = 0.190 + 0.003 - 0.0000134*T + 0 + 0
        Epot += Gcor_H
    for _ in range(num_slab_H): # assume H comes from proton
        Gcor_slab_H_from_proton = - U - kB * T * pH * np.log(10)
        Epot -= Gcor_slab_H_from_proton

    gamma = Epot -d_num_Pd*d_mu_equi['Pd'] - d_num_Ti*d_mu_equi['Ti'] - d_num_H*d_mu_equi['H'] - num_full*mu_slab['PdH'] - G_adss - gamma_Pd
    return gamma

def single_emt_calculator():
    """EMT calculator for test"""
    calc = EMT()
    return calc

def single_painn_calculator(ga):
    """Single calculator for painn"""
    state_dict = torch.load(ga.load_model)
    model = PainnModel(
        num_interactions=state_dict["num_layer"],
        hidden_state_size=state_dict["node_size"],
        cutoff=state_dict["cutoff"],
    )
    model.to(ga.device)
    model.load_state_dict(state_dict["model"])
    calc = MLCalculator(model)
    return calc

def ensemble_painn_calculator(ga):
    """Ensemble calculator for painn"""
    models = []
    for each in ga.load_models:
        state_dict = torch.load(each, map_location=torch.device(ga.device))
        model = PainnModel(
            num_interactions=state_dict["num_layer"],
            hidden_state_size=state_dict["node_size"],
            cutoff=state_dict["cutoff"],
        )
        model.to(ga.device)
        model.load_state_dict(state_dict["model"])
        models.append(model)
    encalc = EnsembleCalculator(models)
    return encalc

def relax_single_calc(atoms, single_point=True):
    """Relax one atoms"""
    calc = single_painn_calculator(ga)
    mask = [atom.z < 2.0 for atom in atoms]
    atoms.set_constraint(FixAtoms(mask=mask))
    atoms.calc = calc
    converged = True
    if not single_point:
        opt = BFGS(atoms, logfile=None)
        converged = opt.run(fmax=ga.fmax, steps=200)
    Epot = atoms.get_potential_energy()
    atoms.info['key_value_pairs']['potential_energy'] = Epot # for compator
    atoms.info['key_value_pairs']['energy_var'] = ens['energy_var'] 
    atoms.info['data']['nnmat_string'] = get_nnmat_string(atoms, 3, True) # decimal is 2
    atoms.info['data']['atoms_string'] = ''.join(atoms.get_chemical_symbols())
    Ts = np.arange(200., 300., 1.)
    fs = np.array([calculate_gamma(atoms, Epot, T=T) for T in Ts])
    atoms.info['data']['raw_scores'] = fs
    atoms.calc = None
    return atoms

def relax_ensemble_calc(atoms, single_point=False, early_return=False):
    """Relax one atoms ensemble"""
    calc = ensemble_painn_calculator(ga)
    mask = [atom.z < 2.0 for atom in atoms]
    atoms.set_constraint(FixAtoms(mask=mask))
    atoms.calc = calc
    converged = True
    step = 0
    if not single_point:
        atoms_old = copy.deepcopy(atoms)
        try:
            opt = BFGS(atoms, logfile='bfgs.log')
            eo = EnergyObservor(opt.atoms)
            opt.attach(eo)
            converged = opt.run(fmax=ga.fmax, steps=ga.max_steps)
            step = opt.get_number_of_steps()
        except:
            atoms = atoms_old
            converged = False

    Epot = atoms.get_potential_energy()
    Epot = round(Epot, 4)
    ens = atoms.calc.get_ensemble()
    if early_return:
        return atoms
    atoms.info['key_value_pairs']['potential_energy'] = Epot # for compator
    atoms.info['key_value_pairs']['energy_var'] = ens['energy_var']
    atoms.info['key_value_pairs']['step'] = step
    atoms.info['key_value_pairs']['converged'] = converged
    atoms.info['data']['nnmat_string'] = get_nnmat_string(atoms, 3, True) # decimal is 2
    atoms.info['data']['atoms_string'] = ''.join(atoms.get_chemical_symbols())
    std = np.sqrt(ens['energy_var']) # standard variance
    A_surf = np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1]))
    gammas = []
    scores = []
    d_num_Pd, d_num_Ti, d_num_H, num_ads_CO, num_ads_OH, num_ads_H, num_slab_H = get_element_nums(atoms)
    niches['Epot'] = Epot
    niches['d_num_Pd'] = d_num_Pd
    niches['d_num_Ti'] = d_num_Ti
    niches['d_num_H'] = d_num_H
    niches['num_ads_CO'] = num_ads_CO
    niches['num_ads_OH'] = num_ads_OH
    niches['num_ads_H'] = num_ads_H
    niches['num_slab_H'] = num_slab_H
    # print(niches)
    gammas = niches.apply(calculate_gamma, axis=1).values
    # print('gammas:', gammas)
    scores = (niches['kappa'].values * std - gammas) / A_surf
    # print('scores:', scores)
    atoms.info['data']['gammas'] = gammas / A_surf
    atoms.info['data']['raw_scores'] = scores
    atoms.calc = None
    return atoms

def write_db_distortion(atoms, reason, db_name_dist='distortion_init.db'):
    db_distortion = connect(db_name_dist)
    db_distortion.write(atoms, reason=reason, data=atoms.info)

def relax(atoms, ensemble_calc=True):
    logger.info(f"relaxing atoms {atoms.info['confid']}")
    atoms_old = copy.deepcopy(atoms)
    if ensemble_calc:
        atoms = relax_ensemble_calc(atoms, single_point=False)
    else:
        atoms = relax_single_calc(atoms, single_point=True)
    distortion, reason = check_distortion(atoms, cutoff=1.2)
    if distortion:
        atoms = None
        write_db_distortion(atoms_old, reason)
        logger.debug(f"distortion - {distortion} - {reason}")
    return atoms

def relax_offspring(n):
    """Relax the one offspring structure"""
    t1 = time.time()
    while True:
        op = op_selector.get_operator()  # Do the operation
        print('\n', op)
        np.random.seed(random.randint(1,100000))
        population.rng = np.random
        a1, a2 = population.get_two_candidates()
        try:
            offspring = op.get_new_individual([a1, a2])
            a3, desc = offspring
            if a3 is not None:
                a3 = relax(a3)
                if a3 is not None:
                    break
        except:
            ims = [a1, a2]
            write('debug.traj', ims)
            # assert False
    if 'data' not in a3.info:
        a3.info['data'] = {}
    t2 = time.time()
    print('offspring relaxing time:', t2-t1)
    return a3
 
def relax_init_atoms(atoms, ensemble_calc=True, early_return=False):
    atoms = copy.deepcopy(atoms)
    atoms_old = copy.deepcopy(atoms)
    if ensemble_calc:
        atoms = relax_ensemble_calc(atoms, single_point=False, early_return=early_return)
    else:
        atoms = relax_single_calc(atoms, single_point=True)
    distortion, reason = check_distortion(atoms, cutoff=1.2)
    if distortion:
        atoms = None
        write_db_distortion(atoms_old, reason)
        logger.debug(f"distortion - {distortion} - {reason}")
    return atoms

def generate_1st_init_ga_db(db_name, pop_size, temp=None, init_images=None):
    if os.path.exists(db_name):
        os.system('cp {} backup.db'.format(db_name))
        os.remove(db_name)
    db = PrepareDB(db_file_name=db_name, # Instantiate the db
                  cell=temp.cell, 
                  population_size=pop_size)
    count = 0
    for atoms in init_images:
        if 'data' not in atoms.info:
            atoms.info['data'] = {}
        atoms.info['data']['ads_indices'] = atoms.info['ads_indices']
        atoms.info['data']['ads_symbols'] = atoms.info['ads_symbols']
        a = relax_init_atoms(atoms, early_return=True)
        if a is not None:
            db.add_unrelaxed_candidate(atoms, data=atoms.info['data'])
            count += 1
            if count == pop_size:
                break
    while count<pop_size: # add more
        images = generate_random_init_structs_n_ads(tot=1, sub_ele='Ti', temp=temp)
        atoms = images[0]
        if 'data' not in atoms.info:
            atoms.info['data'] = {'tag': None}
        atoms.info['data']['ads_indices'] = atoms.info['ads_indices']
        atoms.info['data']['ads_symbols'] = atoms.info['ads_symbols']
        a = relax_init_atoms(atoms, early_return=True)
        if a is not None:
            db.add_unrelaxed_candidate(atoms, data=atoms.info['data'])
            count += 1
    return db

def generate_ith_init_ga_db(db_name, pop_size, temp=None, init_images=None):
    if os.path.exists(db_name):
        os.system('cp {} backup.db'.format(db_name))
        os.remove(db_name)
    db = PrepareDB(db_file_name=db_name, # Instantiate the db
                  cell=temp.cell, 
                  population_size=pop_size)
    count = 0
    for atoms in init_images:
        if 'data' not in atoms.info:
            if 'ads_indices' in atoms.info and 'ads_symbols' in atoms.info:
                atoms.info['data'] = {'tag': None}
                atoms.info['data']['ads_indices'] = atoms.info['ads_indices']
                atoms.info['data']['ads_symbols'] = atoms.info['ads_symbols']
            else:
                raise ValueError('Check the structures and data should be included !')

        a = relax_init_atoms(atoms, early_return=True)
        if a is not None:
            db.add_unrelaxed_candidate(atoms, data=atoms.info['data'])
            count += 1
    db_last = connect(ga.last_db_dir) # ga db in last iteration
    keep = ga.pop_size # last 50 structures
    lens = len(db_last)
    start = lens - keep
    while count<pop_size: # add more
        i = 0
        for row in db_last.select():
            if i == start:
                atoms = db_last.get_atoms(row.id, add_additional_information=True)
                a = relax_init_atoms(atoms, early_return=True)
                if a is not None:
                    db.add_unrelaxed_candidate(atoms, data=atoms.info['data'])
                    count += 1
                    print(row.id)
                start -= 1
            i += 1
    return db
 
def relax_an_unrelaxed_candidate(atoms):
    """Relax one unrelaxed candidate in population pool"""
    logger.info(f"relaxing atoms {atoms.info['confid']}")
    if 'data' not in atoms.info:
        atoms.info['data'] = {'tag': None}
    return relax_init_atoms(atoms)

def relax_init_pop(cores, db):
    """Relax the initial structures in population pool"""
    logger.info(f"relaxing generation 0")
    relaxed_candidates = []
    if cores > 1:
        pool = Pool(cores)
        relaxed_candidates = pool.map(relax_an_unrelaxed_candidate, db.get_all_unrelaxed_candidates())  
        pool.close()
        pool.join()
    else:
        images = db.get_all_unrelaxed_candidates()
        print(len(images))
        for atoms in images:
            t_start = time.time()
            a = relax_an_unrelaxed_candidate(atoms)
            if a == None:
                assert False
            relaxed_candidates.append(a)
            t_end = time.time()
            print('initial relaxing:', t_end-t_start)
    if len(relaxed_candidates) > pop_size:
        relaxed_candidates = relaxed_candidates[:pop_size]
    assert len(relaxed_candidates)==pop_size
    population.update(new_cand=relaxed_candidates)
    if copy_to_scratch:
        if 'SLURM_JOB_ID' in os.environ:
            shutil.copyfile(fname, db_name)
    return db
            
def relax_generation(cores, db, gens_running):
    """Relax the all offspring structures of one generation"""
    relaxed_candidates = []
    t_start = time.time()
    if cores > 1: 
        pool = Pool(cores)
        relaxed_generation = pool.map(relax_offspring, range(pop_size))
        pool.close()
        pool.join()
        for a in relaxed_generation:
            a.info['key_value_pairs']['generation'] = gens_running
            relaxed_candidates.append(a)
    else:
        count = pop_size
        while count > 0:
            t1 = time.time()
            a = relax_offspring(1)
            if a is not None:
                a.info['key_value_pairs']['generation'] = gens_running
                relaxed_candidates.append(a)
                count -= 1
            t2 = time.time()
            print('offspring relaxing time:', t2-t1)
    assert len(relaxed_candidates)==pop_size
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
        assert str(atoms[ads_index].symbols)==ads_symbol
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

def get_adsorbates_from_slab(atoms, debug=False):
    """Get adsorbate information from atoms, including indices and symbols"""
    atoms = copy.deepcopy(atoms)
    ads_indices = atoms.info['data']['ads_indices']
    ads_symbols = atoms.info['data']['ads_symbols']
    assert len(ads_indices)==len(ads_symbols)
    if debug:
        check_adsorbates(atoms, ads_indices, ads_symbols)
    return ads_indices, ads_symbols

def check_adsorbates_too_close(atoms, newp, oldp, cutoff, mic=True): 
    """Check if there are atoms that are too close to each other.
    mic : bool, default False
        Whether to apply minimum image convention. Remember to set 
        mic=True for periodic systems.
    """
    newps = np.repeat(newp, len(oldp), axis=0)
    oldps = np.tile(oldp, (len(newp), 1))
    if mic:
        _, dists = find_mic(newps - oldps, atoms.cell, pbc=True)
    else:
        dists = np.linalg.norm(newps - oldps, axis=1)
    return any(dists < cutoff)

def check_too_close_after_optimized(atoms, cutoff=1.7):
    """Check if adsorbates is too close"""
    atoms_old = copy.deepcopy(atoms)
    ads_indices, ads_symbols = get_adsorbates_from_slab(atoms)
    if_too_close = False
    for ads_index, ads_symbol in zip(ads_indices, ads_symbols):
        atoms = copy.deepcopy(atoms_old)
        newp = atoms[ads_index].positions
        del atoms[ads_index]
        oldp = atoms.positions
        if_too_close = check_adsorbates_too_close(atoms, newp, oldp, cutoff, mic=True)
        if if_too_close:
            break
    return if_too_close

def check_distortion(atoms, cutoff=1.2):
    """Check if there are some distortion structures, such as, H2O, H2"""
    cutoff = cutoff # within 1.2 A
    nl = NeighborList(cutoffs=[cutoff / 2.] * len(atoms),
                            self_interaction=True,
                            bothways=True,
                            skin=0.)
    nl.update(atoms)
    if True:
        if atoms.info['key_value_pairs']['converged'] == False:
            reason = 'unconverged'
            return True, reason
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
        if atom.tag == 1:
            if atom.z < 6.2:
                reason = 'layer1_collapes'
                return True, reason
            elif atom.z > 7.5:
                reason = 'layer1_pop_too_much'
                return True, reason
            indices = [a.index for a in atoms if a.symbol=='C' or a.symbol=='O']
            if indices != []:
                C_Os = atoms[indices]
                for a in C_Os:
                    if a.z < atom.z:
                        reason = 'C_or_O_collapes'
                        return True, reason
        elif atom.tag == 3:
            if atom.z < 2.:
                reason = 'layer3_collapes'
                return True, reason
        elif atom.tag == -2:
            if atom.z < 6.:
                reason = 'CO_sink'
                return True, reason

    if_too_close = check_too_close_after_optimized(atoms, cutoff=tc_cutoff)
    if if_too_close:
        reason = 'ads_too_close'
        return True, reason
    indices = [atom.index for atom in atoms if atom.tag==1] # check metal distance only 1st layer
    for i1 in indices:
        for i2 in indices:
            if i1 != i2:
                d = atoms.get_distance(i1, i2, mic=True)
                if d < 2.2: # distance between metals
                    reason = 'metals_1st_too_close'
                    return True, reason
    if 'key_value_pairs' in atoms.info:
        if atoms.info['key_value_pairs']['potential_energy'] < -1000.:
            reason = 'energy_too_low'
            return True, reason
    return False, 'undistortion'

def clean_before_starting(ga):
    db_name_dist=ga.db_name_dist
    pop_log = ga.pop_log
    debug_traj = ga.debug_traj
    converged_file = ga.converged_file
    remove_list = [db_name_dist, pop_log, debug_traj, converged_file, ga.converged_file]
    for item in remove_list:
        if os.path.exists(item):
            os.remove(item)

class Params:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self.__dict__[key] = Params(**value)
            else:
                self.__dict__[key] = value

def get_fittest_traj(ga, name='fittest_images.traj'):
    db = connect(ga.db_name)
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

def get_last_generation(ga, name='last_gen_images.traj'):
    db = connect(ga.db_name)
    keep = ga.pop_size # last 50 structures
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

def generate_run_config(params):
    with open(os.path.join(params['run_dir'], "run.toml"), 'w') as f:
        toml.dump(params, f)
    return params

def generate_summary_table(ga):
    ids, convergeds, formulas, adsorbates, indices, num_adss = [], [], [], [], [], []
    db = connect(ga.db_name)
    for row in db.select(relaxed=1):
        print(row.id)
        atoms = row.toatoms(add_additional_information=True)
        formula = atoms.get_chemical_formula(mode='metal')
        adsorbate = atoms.info['data']['ads_indices']
        index = atoms.info['data']['ads_symbols']
        converged = atoms.info['key_value_pairs']['converged']
        num_ads = len(adsorbate)
        ids.append(row.id)
        formulas.append(formula)
        adsorbates.append(adsorbate)
        indices.append(index)
        num_adss.append(num_ads)
        convergeds.append(converged)
    tuples = {
            'ids': ids,
            'formulas': formulas,
            'adsorbates': adsorbates,
            'indices': indices,
            'num_adss': num_adss,
            'convergeds': convergeds,
            }
    df = pd.DataFrame(tuples)
    df.to_csv('summary.csv')
    return df

if __name__ == '__main__':
    args = get_arguments()
    if args.cfg:
        with open(args.cfg, 'r') as f:
            params = toml.load(f)
    ga = Params(**params)
    _ = generate_run_config(params)
    if True:
        continue_ga = ga.continue_ga
        copy_to_scratch = ga.copy_to_scratch
        tc_cutoff = ga.too_close_cutoff
        print(tc_cutoff)
        if ga.device == 'cpu':
            cores = ga.cpu.cores
        elif ga.device == 'cuda':
            cores = 1
        else:
            raise ValueError('Please setup device')
        print(f'cores: {cores}')
        db_name = ga.db_name
        pop_size = ga.pop_size
        temp = read(ga.temp_traj)
        if continue_ga:
            remove_log = False
        else:
            remove_log = True
        logger = create_file_logger(name=ga.basic_ga_log, remove_basic_log=remove_log)
        # Define the tasks. 
        niches = pd.read_pickle(ga.task_pkl)
        
        initialize = not continue_ga
        if initialize: # only use for each iteration
            clean_before_starting(ga)
            if ga.iteration == ga.start_iteration:
                init_images = read(ga.init_gen_traj, index=':')
                db = generate_1st_init_ga_db(db_name, pop_size, temp, init_images)
            else:
                init_images = read(ga.last_gen_traj, index=':')
                db = generate_ith_init_ga_db(db_name, pop_size, temp, init_images)
            shutil.copyfile(db_name, 'init_'+db_name)
        else: # continue ga
            if ga.use_init_db:
                shutil.copyfile('init_'+db_name, db_name)
        if copy_to_scratch:
            db, fname = copy_to_scratch(db_name)
        else:
            db = DataConnection(db_name)

        op_selector = OperationSelector([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [
                                    RandomMetalPermutation(element_pools=['Pd', 'Ti'], num_muts=1),
                                    RandomMetalComposition(element_pools=['Pd', 'Ti'], num_muts=1),
                                    SymmetrySlabPermutation(element_pools=['Pd', 'Ti'], num_muts=1),
                                    InternalHydrogenAddition(internal_H_pools=['H'], num_muts=1),
                                    InternalHydrogenRemoval(internal_H_pools=['H'], num_muts=1),
                                    InternalHydrogenMoveToUnoccupied(internal_H_pools=['H'], num_muts=1),
                                    AdsorbateAddition(ads_pools=['CO', 'OH', 'H'], num_muts=1, tc_cutoff=tc_cutoff),
                                    AdsorbateRemoval(ads_pools=['CO', 'OH', 'H'], num_muts=1, tc_cutoff=tc_cutoff),
                                    AdsorbateSubstitution(ads_pools=['CO', 'OH', 'H'], num_muts=1, tc_cutoff=tc_cutoff),
                                    AdsorbateSwapOccupied(ads_pools=['CO', 'OH', 'H'], num_muts=1, tc_cutoff=tc_cutoff),
                                    AdsorbateMoveToUnoccupied(ads_pools=['CO', 'OH', 'H'], num_muts=1, tc_cutoff=tc_cutoff),
                                    AdsorbateCutSpliceCrossover(ads_pools=['CO', 'OH', 'H'], num_muts=1, tc_cutoff=tc_cutoff),
                                    ])
        
        # Initialize the population and specify the number of tasks
        population = MultitaskPopulation(data_connection=db,
                                  population_size=pop_size,
                                  num_tasks=niches.shape[0],
                                  comparator=get_comparators(db),
                                  exp_function=True,
                                  logfile=ga.pop_log)
        
        # relax initial population
        if not continue_ga:
            db = relax_init_pop(cores, db)
            print('initial optimization done')
        else:
            print('continuing ga')
        gen_num = db.get_generation_number()
        max_gens = ga.max_gens # maximum of generations
        cc = GenerationRepetitionConvergence(population, ga.repetition_convergence_times)
        for i in range(max_gens):
            gens_running = gen_num + i
            if cc.converged():  # Check if converged
                logger.info('Converged!')
                # if os.path.exists(ga.converged_file):
                    # os.remove(ga.converged_file)
                fp = open(ga.converged_file, 'w')
                fp.close()
                break
            logger.newline()
            logger.info(f"relaxing generation {i}")
            db = relax_generation(cores, db, gens_running)
        print('GA done!')
    
    if True:
        selected_images = []
        f_images = get_fittest_traj(ga, name=ga.fittest_images)
        l_images = get_last_generation(ga, name=ga.last_gen_images)
        selected_images = f_images + l_images
        print(f'fittest images: {len(f_images)}; last gen images: {len(l_images)}' )
        write(ga.cand_images, selected_images)
        generate_summary_table(ga) 
        print('GA data generating done!')
