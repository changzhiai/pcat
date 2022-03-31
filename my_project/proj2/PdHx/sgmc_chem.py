# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:58:03 2022

@author: changai
"""

import numpy as np
from pcat.lib.io import pd_read_excel
import matplotlib.pyplot as plt
import pandas as pd
from ase.db import connect

kB = 8.617e-5  # Boltzmann constant in eV/K

def Hcons(atoms):
    try:
        sites_H = [a.index for a in atoms if a.symbol == 'H']
        nums_H = len(sites_H)
        cons_H = len(sites_H) / 64.
    except:
        sites_H = []
        nums_H = 0
        cons_H = 0
        
    return cons_H, nums_H
        

def db2xls(db_name='./data/sgmc_results.db'):
    """Convert database to excel"""
    db = connect(db_name)
    cons_H = []
    nums_H = []
    # form_energies = []
    ids = []
    energies = []
    Ts = []
    chem_Hs = []
    pH2s = []
    # for temp in [700, 600, 500, 400, 300, 200, 100, 0]:
    for row in db.select():
        atoms = row.toatoms()
        con_H, num_H = Hcons(atoms)
        cons_H.append(con_H)
        nums_H.append(num_H)
        # form_energies.append(row.form_energy)
        mu_H = row.data.chem_pot_H
        chem_Hs.append(mu_H)
        ids.append(row.id)
        energies.append(row.energy)
        temp = row.data.temperature
        Ts.append(temp)
        pH2 = P_H2_s(mu_H, temp)
        pH2s.append(pH2)
        # print(row.id)
    tuples = {'cons_H': cons_H,
              'nums_H': nums_H,
            # 'form_energies': form_energies,
            '$\mu$_H': chem_Hs,
            'ids': ids,
            'Energies': energies,
            'temp_H2': Ts,
            'pH2': pH2s,
             }
    df = pd.DataFrame(tuples)
    df.to_excel(xls_name, sheet_name_convex_hull, float_format='%.3f')
    
def P_H2(mu_H, T):
    """Get presure of H2"""
    pH2 = np.exp((2*mu_H + 0.0012*T - 0.3547)/(kB*T))
    return pH2


def T_H2(mu_H, P):
    """Get Temperature of H2"""
    temp_H2 = (2*mu_H - 0.3547) / (kB*np.log(P) - 0.0012)
    return temp_H2



def mu_H2(T, pH2):
    """Get chemical potential using standard method"""
    mu_0_H2 = -7.096 - T * 0.00135
    mu_H2 = mu_0_H2 + kB*T*np.log(pH2/1)
    mu_H = mu_H2 / 2.
    return mu_H

def P_H2_s(mu_H, T):
    """Get presure of H2 using standard method"""
    pH2 = np.exp((2*mu_H + 0.00135*T + 7.096)/(kB*T))
    return pH2

def T_H2_s(mu_H, P):
    """Get Temperature of H2 using standard method"""
    temp_H2 = (2*mu_H + 7.096) / (kB*np.log(P) - 0.00135)
    return temp_H2

def get_real_quan(df):
    row_E_Pd = df[df['cons_H'] == 0]
    E_Pd = (row_E_Pd['Energies'].values)[0]

    nums_Hs = []
    mu_Hs = []
    pH2s = []
    temp_H2s = []
    for i, row in df.iterrows():
        dft_energy = row['Energies']
        nums_H = row['nums_H']
        if nums_H != 0:
            # print(nums_H)
            mu_H = (dft_energy - E_Pd) / nums_H
        else:
            mu_H = -4.2 #-3.548

        pH2 = P_H2_s(mu_H, T)
        # temp_H2 = T_H2_s(mu_H, P)

        mu_Hs.append(mu_H)
        pH2s.append(pH2)
    #     # temp_H2s.append(temp_H2)

    df['$\mu$_H'] = mu_Hs
    df['pH2'] = pH2s
    # df['temp_H2']=temp_H2s
    return df


def get_quantities(xls_name, sheet, T=500, P=100):
    """Get chemical potential using energy difference"""
    df = pd_read_excel(filename=xls_name, sheet=sheet)
    df = df.loc[df['temp_H2']==T]
    # df = df.loc[(df['temp_H2']==T) | (df['cons_H']==0) | (df['cons_H']==1)]
    
    df = df.sort_values(by=['$\mu$_H'], ascending=False)
    
    # df = get_real_quan(df)
    print(df)
    return df


def plot_chem_vs_cons(xls_name, sheet, T):
    """Plot chemical potential vs. concentration of H"""
    df = get_quantities(xls_name, sheet, T=T)

    mu_Hs = df['$\mu$_H']
    cons_Hs = df['cons_H']
    # plt.figure()
    plt.plot(cons_Hs[:-1], mu_Hs[:-1], 'o')
    plt.plot(cons_Hs[:-1], mu_Hs[:-1], )
    # plt.plot(cons_Hs, mu_Hs, 'o')
    # plt.plot(cons_Hs, mu_Hs, )
    # plt.plot(cons_Hs[:-1], mu_Hs[:-1])
    # plt.plot( mu_Hs[:-1], cons_Hs[:-1])
    # plt.xlabel('Concentration of H')
    # plt.ylabel('H chemical potential')
    # plt.show()


def plot_pressure_vs_cons(xls_name, sheet, T):
    """Plot partial pressure vs. concentration"""
    df = get_quantities(xls_name, sheet, T=T)
    # df = get_real_quan(df)
    # print(df)

    pH2s = df['pH2']
    cons_Hs = df['cons_H']
    # plt.figure()
    # plt.plot(cons_Hs[:-1], np.log(pH2s[:-1]))
    plt.semilogy(cons_Hs, pH2s)
    # plt.semilogy(cons_Hs[:-1], pH2s[:-1])
    # plt.xlabel('Concentration of H')
    # plt.ylabel('Pressure')
    # plt.show()


def plot_temp_vs_cons(xls_name, sheet, P):
    """Plot temperature vs. concentration"""
    df = get_quantities(xls_name, sheet, P=P)

    cons_Hs = df['cons_H']
    temp_H2 = df['temp_H2']
    plt.figure()
    plt.plot(cons_Hs[:-1], temp_H2[:-1])
    plt.xlabel('Concentration of H')
    plt.ylabel('Temperature (K)')
    plt.show()


if __name__ == '__main__':
    # system = 'candidates_PdHx'  # candidates surface of CE
    # system = 'results_last1'  # candidates surface of CE
    system = 'sgmc_results'  # candidates surface of CE
    # system = 'sgmc_results_r2'  # candidates surface of CE
    # system = 'sgmc_results_r3'  # candidates surface of CE
    # system = 'sgmc_results_r3re'  # candidates surface of CE
    # system = 'sgmc_results_r4'  # candidates surface of CE
    # system = 'sgmc_results_r4re'  # candidates surface of CE
    # system = 'sgmc_results_r4_Hchem'
    # system = 'sgmc_results_r5_re'
    # system = 'sgmc_results_r6'

    fig_dir = './figures/'
    data_dir = './data'
    db_name = f'./{data_dir}/{system}.db'
    xls_name = f'./{data_dir}/{system}.xlsx'

    sheet_name_convex_hull = 'convex_hull'
    
    if True:
        db2xls(db_name=db_name)
    if True:
        # get_quantities(xls_name, sheet=sheet_name_convex_hull)
        # plot_chem_vs_cons(xls_name, sheet=sheet_name_convex_hull, T=500)
        if True:
            for T in [300]:
            # for T in [100, 200, 300, 400, 500, 600, 700, 800]:
                plot_chem_vs_cons(xls_name, sheet=sheet_name_convex_hull, T=T)
            plt.xlabel('Concentration of H')
            plt.ylabel('H chemical potential')
            plt.show()
            
        # plot_pressure_vs_cons(xls_name, sheet=sheet_name_convex_hull, T=300)
        # for T in [300, 400, 500, 600, 700]:
        for T in [100, 200, 300, 400, 500, 600, 700, 800]:
            plot_pressure_vs_cons(xls_name, sheet=sheet_name_convex_hull, T=T)
        plt.xlabel('Concentration of H')
        plt.ylabel('Pressure')
        plt.show()
        
        # plot_temp_vs_cons(xls_name, sheet=sheet_name_convex_hull, P=100)
    
    
