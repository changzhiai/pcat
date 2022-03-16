# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 12:04:10 2022

@author: changai
"""

import numpy as np
from pcat.lib.io import pd_read_excel
import matplotlib.pyplot as plt

kB = 8.617e-5  # Boltzmann constant in eV/K


def P_H2(mu_H, T):
    """Get presure of H2"""
    pH2 = np.exp((2*mu_H + 0.0012*T - 0.3547)/(kB*T))
    return pH2


def T_H2(mu_H, P):
    """Get Temperature of H2"""
    temp_H2 = (2*mu_H - 0.3547) / (kB*np.log(P) - 0.0012)
    return temp_H2


def get_quantities(xls_name, sheet, T=298.15, P=1):
    """Get chemical potential using energy difference"""
    df = pd_read_excel(filename=xls_name, sheet=sheet)
    df = df.sort_values(by=['cons_H'], ascending=False)

    row_E_Pd = df[df['cons_H'] == 0]
    E_Pd = (row_E_Pd['Energies'].values)[0]

    nums_Hs = []
    mu_Hs = []
    pH2s = []
    for i, row in df.iterrows():
        cons_H = row['cons_H']
        dft_energy = row['Energies']
        splits = row['ids'].split('_')
        nums_H = 64 - int(splits[1])
        if nums_H != 0:
            mu_H = (dft_energy - E_Pd) / nums_H
        else:
            mu_H = 0

        pH2 = P_H2(mu_H, T)
        temp_H2 = T_H2(mu_H, P)

        nums_Hs.append(nums_H)
        mu_Hs.append(mu_H)
        pH2s.append(pH2)
    df['num_H'] = nums_Hs
    df['$\mu$_H'] = mu_Hs
    df['pH2'] = pH2s
    df['temp_H2']=temp_H2
    return df


def plot_chem_vs_cons(xls_name, sheet, T):
    """Plot chemical potential vs. concentration of H"""
    df = get_quantities(xls_name, sheet, T=T)

    mu_Hs = df['$\mu$_H']
    cons_Hs = df['cons_H']
    plt.figure()
    # plt.plot(cons_Hs, mu_Hs)
    plt.plot(cons_Hs[:-1], mu_Hs[:-1])
    # plt.plot( mu_Hs[:-1], cons_Hs[:-1])
    plt.xlabel('Concentration of H')
    plt.ylabel('H chemical potential')
    plt.show()


def plot_pressure_vs_cons(xls_name, sheet, T):
    """Plot partial pressure vs. concentration"""
    df = get_quantities(xls_name, sheet, T=T)

    pH2s = df['pH2']
    cons_Hs = df['cons_H']
    plt.figure()
    plt.plot(cons_Hs[:-1], np.log(pH2s[:-1]))
    plt.xlabel('Concentration of H')
    plt.ylabel('Pressure')
    plt.show()


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
    system = 'candidates_PdHx'  # candidates surface of CE

    fig_dir = './figures/'
    data_dir = './data'
    db_name = f'./{data_dir}/{system}.db'
    xls_name = f'./{data_dir}/{system}.xlsx'

    sheet_name_convex_hull = 'convex_hull'

    # get_quantities(xls_name, sheet=sheet_name_convex_hull)
    plot_chem_vs_cons(xls_name, sheet=sheet_name_convex_hull, T=1)
    plot_pressure_vs_cons(xls_name, sheet=sheet_name_convex_hull, T=298.15)
    # plot_pressure_vs_cons(xls_name, sheet=sheet_name_convex_hull, T=700)
    plot_temp_vs_cons(xls_name, sheet=sheet_name_convex_hull, P=1)
