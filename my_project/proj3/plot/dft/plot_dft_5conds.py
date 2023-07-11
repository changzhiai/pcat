# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 10:32:51 2023

@author: changai
"""

from ase import Atoms
from ase.visualize import view
import numpy as np
from ase.thermochemistry import IdealGasThermo
from ase.io import write, read
import pandas as pd
import copy
from ase.units import kB
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from cores_conds import (plot_surf_free_vs_U, 
                         plot_surf_free_vs_U_matrix, 
                         plot_surf_free_vs_U_contour,
                         plot_surf_free_vs_U_matrix_all)
import pickle
import matplotlib
matplotlib.use('Agg')

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

def get_element_nums(atoms):
    "Calculate how many atoms for each element"
    num_full = 16
    if 'data' not in atoms.info:
        atoms.info['data'] = {}
    symbols = atoms.get_chemical_symbols()
    num_ads_OH = symbols.count('O') - symbols.count('C')
    num_ads_CO = symbols.count('C')
    ave_z = np.mean([atom.z for atom in atoms if atom.tag==1])
    num_ads_H = len([atom for atom in atoms if atom.z >= ave_z and atom.symbol=='H'])-num_ads_OH
    num_H = len([s for s in atoms.symbols if s == 'H']) - num_ads_H - num_ads_OH # H only in slab
    d_num_H = num_H - num_full
    num_Pd = len([s for s in atoms.symbols if s == 'Pd'])
    d_num_Pd = num_Pd - num_full
    d_num_Ti = -d_num_Pd
    num_slab_H = num_H
    return d_num_Pd, d_num_Ti, d_num_H, num_ads_CO, num_ads_OH, num_ads_H, num_slab_H

def calc_dft_gamma(atoms):
    """Relax one atoms ensemble"""
    Epot = atoms.get_potential_energy()
    Epot = round(Epot, 4)
    std = 0 # standard variance
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

def generate_tasks(save_to_files=False):
    """Get reaction conditions"""
    d_mu_Pd = np.array([0, -0.25, -0.5, -0.75, -1.0])+(-2.24900)
    d_mu_Ti = np.array([0, -0.25, -0.5, -0.75, -1.0])+(-7.28453)
    d_mu_H = np.array([-3.61353])
    T = np.array([10, 25, 40, 65, 80])+273.15
    U = np.array([0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8])
    pH = np.array([0,])
    P_H2 = np.array([101325.])
    P_CO2 = np.array([101325.])
    P_H2O = np.array([3534.])
    # P_CO = np.logspace(-6, 0, 5)*101325.
    P_CO = np.array([0.101325, 10.1325, 1013.25, 5562, 101325])
    kappa = np.array([0, 1])
    keys = ['d_mu_Pd', 'd_mu_Ti', 'd_mu_H', 'T', 'U', 'pH', 'P_H2', 'P_CO2', 'P_H2O', 'P_CO', 'kappa']
    words = [d_mu_Pd, d_mu_Ti, d_mu_H, T, U, pH, P_H2, P_CO2, P_H2O, P_CO, kappa]
    df = {}
    for i in range(len(keys)):
        df[keys[i]] = words[i]  
    if save_to_files:
        cond_grid = np.array(np.meshgrid(d_mu_Pd, d_mu_Ti, d_mu_H, T, U, pH, P_H2, P_CO2, P_H2O, P_CO, kappa)).T.reshape(-1, 11)
        df = pd.DataFrame(cond_grid, columns=['d_mu_Pd', 'd_mu_Ti', 'd_mu_H', 'T', 'U', 'pH', 'P_H2', 'P_CO2', 'P_H2O', 'P_CO', 'kappa'])
        df.to_pickle('em_tasks.pkl')
        df.to_csv('em_tasks.csv')
    return df

def generate_csv(fittest_images, raw_niches, save_to_csv=False):
    """Generate CSV for each images"""
    dfs = []
    for i, atoms in enumerate(fittest_images):
        # niches = atoms.info['data']['niches']
        scores = atoms.info['data']['raw_scores']
        # dn = atoms.info['key_value_pairs']['dominating_niche']
        # rs = atoms.info['key_value_pairs']['raw_score']
        # print(atoms.info['data']['raw_scores'])
        # print(i, ':', scores[niches])
        # print(set(raw_niches.iloc[niches]['U']))
        data = copy.deepcopy(raw_niches)
        data['raw_scores'] = scores # add a raw scores colomn
        csv = str(i) + '_' + atoms.get_chemical_formula(mode='metal') + '.csv'
        if save_to_csv:
            data.to_csv(csv)
        dfs.append(data)
    return dfs

def plot_matrix_all_conds(images, iterations):
    '''T->Pco->{matrix} return all candidates and ids at one iteration'''
    raw_niches = copy.deepcopy(niches)
    dfs = generate_csv(images, raw_niches, save_to_csv=False)  
    d_mu_Pd = sorted(set(raw_niches['d_mu_Pd']))[:]
    d_mu_Ti = sorted(set(raw_niches['d_mu_Ti']), reverse=True)[:]
    P_CO = sorted(set(raw_niches['P_CO']))[:]
    T = sorted(set(raw_niches['T']))[:]
    print({'d_mu_Pd': d_mu_Pd, 'd_mu_Ti': d_mu_Ti, 'P_CO': P_CO, 'T': T})
    all_cands, all_ids = plot_surf_free_vs_U_matrix_all(dfs, **{'df_raw': raw_niches,
                                             'images':images, 
                                             'iter':iterations,
                                             'gray_above': True, 
                                             'd_mu_Pd': d_mu_Pd,
                                             'd_mu_Ti': d_mu_Ti,
                                             'P_CO': P_CO,
                                             'T': T,
                                             })
    return all_cands, all_ids
    

def plot_SFE_at_One_Temp_and_Pco(images, iterations):
    """Fix temperature=298.15 K and partial pressure of CO=5562.0 Pa"""
    raw_niches = copy.deepcopy(niches)
    dfs = generate_csv(images, raw_niches, save_to_csv=False)
    d_mu_Pd = sorted(set(raw_niches['d_mu_Pd']))[:]
    d_mu_Ti = sorted(set(raw_niches['d_mu_Ti']), reverse=True)[:]
    P_CO = sorted(set(raw_niches['P_CO']))[3:4] # 5562.0 Pa
    T = sorted(set(raw_niches['T']))[1:2] # 298.15 K
    print({'d_mu_Pd': d_mu_Pd, 'd_mu_Ti': d_mu_Ti, 'P_CO': P_CO, 'T': T})
    _, _ = plot_surf_free_vs_U(dfs, **{'df_raw': raw_niches,
                                               'images':images, 
                                               'iter':iterations,
                                               'gray_above': True,
                                               'd_mu_Pd': d_mu_Pd,
                                               'd_mu_Ti': d_mu_Ti,
                                               'P_CO': P_CO,
                                               'T': T,
                                               })
    cands, ids = plot_surf_free_vs_U_matrix(dfs, **{'df_raw': raw_niches,
                                              'images':images, 
                                              'iter':iterations,
                                              'gray_above': True,
                                              'd_mu_Pd': d_mu_Pd,
                                              'd_mu_Ti': d_mu_Ti,
                                              'P_CO': P_CO,
                                              'T': T,
                                              })
    minuss, idss = plot_surf_free_vs_U_contour(dfs, **{'df_raw': raw_niches,
                                              'images':images, 
                                              'iter':iterations,
                                              'gray_above': None, # no plot
                                              'd_mu_Pd': d_mu_Pd,
                                              'd_mu_Ti': d_mu_Ti,
                                              'P_CO': P_CO,
                                              'T': T,
                                              })  
    # minuss, idss = [], []
    return cands, ids, minuss, idss 


def write_list(n_list, iteration, name='all_ids.pkl'):
    with open(f'./figures/matrix_all/iter_{iteration}/{name}', 'wb') as fp:
        pickle.dump(n_list, fp)
        print('Done writing list into a binary file')

def read_list(iteration, name='all_ids.pkl'):
    with open(f'./figures/matrix_all/iter_{iteration}/{name}', 'rb') as fp:
        n_list = pickle.load(fp)
        return n_list

def flatten_list(n_list):
    import itertools
    n_list = list(itertools.chain(*n_list))
    n_list = list(set(n_list))
    return n_list

if __name__ == '__main__':
    # niches = pd.read_pickle('em_tasks.pkl')
    # niches = generate_tasks(save_to_files=True)
    niches = pd.read_csv('./data/em_tasks.csv')
    iter = 26
    for i in range(1,iter):
        print(f'iter{i}')
        if False:
            images = read(f'./data/dft_PdTiH_adss_r0_to_r{i}_final_tot.traj', ':')
            for atoms in images:
                atoms = calc_dft_gamma(atoms)
            write(f'dft_iter_{i}.traj', images)
        else:
            print('reading images...')
            images = read(f'dft_iter_{i}.traj', ':')
        if False:
            cands, ids, minuss, idss = plot_SFE_at_One_Temp_and_Pco(images, i)
            print(cands, ids, minuss, idss)
        if False: # generate matrix_all
            all_cands, all_ids = plot_matrix_all_conds(images, i)
            print(all_cands, all_ids)
            write_list(all_ids, i, name='all_ids.pkl')
            unique_ids = flatten_list(all_ids)
            write_list(unique_ids, i, name='all_unique_ids.pkl')
            print(unique_ids, len(unique_ids))
        plt.clf()   
        plt.close()
        
        
    

    
    