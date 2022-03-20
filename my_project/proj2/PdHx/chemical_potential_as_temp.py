# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 12:54:42 2022

@author: changai
"""

import numpy as np
from pcat.lib.io import pd_read_excel
import matplotlib.pyplot as plt
import pandas as pd
from ase.db import connect

kB = 8.617e-5  # Boltzmann constant in eV/K

def db2xls(db_name='./data/results_temp.db'):
    """Convert database to excel"""
    db = connect(db_name)
    cons_H = []
    form_energies = []
    ids = []
    energies = []
    Ts = []
    for temp in [700, 600, 500, 400, 300, 200, 100, 0]:
        for row in db.select(temp=temp):
            cons_H.append(row.con_H)
            form_energies.append(row.form_energy)
            ids.append(row.uni_id)
            energies.append(row.energy)
            Ts.append(row.temp)
    tuples = {'cons_H': cons_H,
      'form_energies': form_energies,
      'ids': ids,
      'Energies': energies,
      'temp_H2': Ts,
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



def get_quantities(xls_name, sheet, T=500, P=100):
    """Get chemical potential using energy difference"""
    df = pd_read_excel(filename=xls_name, sheet=sheet)
    df = df.loc[(df['temp_H2']==T) | (df['cons_H']==0) | (df['cons_H']==1)]
    
    df = df.sort_values(by=['cons_H'], ascending=False)

    row_E_Pd = df[df['cons_H'] == 0]
    E_Pd = (row_E_Pd['Energies'].values)[0]

    nums_Hs = []
    mu_Hs = []
    pH2s = []
    temp_H2s = []
    for i, row in df.iterrows():
        dft_energy = row['Energies']
        splits = row['ids'].split('_')
        nums_H = 64 - int(splits[1])
        if nums_H != 0:
            mu_H = (dft_energy - E_Pd) / nums_H
        else:
            mu_H = -4.2 #-3.548

        pH2 = P_H2_s(mu_H, T)
        # temp_H2 = T_H2_s(mu_H, P)

        nums_Hs.append(nums_H)
        mu_Hs.append(mu_H)
        pH2s.append(pH2)
        # temp_H2s.append(temp_H2)
    df['num_H'] = nums_Hs
    df['$\mu$_H'] = mu_Hs
    df['pH2'] = pH2s
    # df['temp_H2']=temp_H2s
    print(df)
    return df


def plot_chem_vs_cons(xls_name, sheet, T):
    """Plot chemical potential vs. concentration of H"""
    df = get_quantities(xls_name, sheet, T=T)

    mu_Hs = df['$\mu$_H']
    cons_Hs = df['cons_H']
    # plt.figure()
    # plt.plot(cons_Hs, mu_Hs)
    plt.plot(cons_Hs[:-1], mu_Hs[:-1])
    # plt.plot( mu_Hs[:-1], cons_Hs[:-1])
    # plt.xlabel('Concentration of H')
    # plt.ylabel('H chemical potential')
    # plt.show()


def plot_pressure_vs_cons(xls_name, sheet, T):
    """Plot partial pressure vs. concentration"""
    df = get_quantities(xls_name, sheet, T=T)

    pH2s = df['pH2']
    cons_Hs = df['cons_H']
    # plt.figure()
    # plt.plot(cons_Hs[:-1], np.log(pH2s[:-1]))
    # plt.semilogy(cons_Hs, pH2s)
    plt.semilogy(cons_Hs[:-1], pH2s[:-1])
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
    system = 'results_temp'  # candidates surface of CE

    fig_dir = './figures/'
    data_dir = './data'
    db_name = f'./{data_dir}/{system}.db'
    xls_name = f'./{data_dir}/{system}.xlsx'

    sheet_name_convex_hull = 'convex_hull'
    
    if False:
        db2xls(db_name='./data/results_temp.db')
    if True:
        # get_quantities(xls_name, sheet=sheet_name_convex_hull)
        plot_chem_vs_cons(xls_name, sheet=sheet_name_convex_hull, T=500)
        for T in [300, 400, 500, 600, 700]:
            plot_chem_vs_cons(xls_name, sheet=sheet_name_convex_hull, T=T)
        plt.xlabel('Concentration of H')
        plt.ylabel('H chemical potential')
        plt.show()
            
        # plot_pressure_vs_cons(xls_name, sheet=sheet_name_convex_hull, T=300)
        for T in [300, 400, 500, 600, 700]:
            plot_pressure_vs_cons(xls_name, sheet=sheet_name_convex_hull, T=T)
        plt.xlabel('Concentration of H')
        plt.ylabel('Pressure')
        plt.show()
        
        # plot_temp_vs_cons(xls_name, sheet=sheet_name_convex_hull, P=100)
    
    
