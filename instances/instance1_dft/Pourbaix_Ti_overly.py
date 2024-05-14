# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 16:56:18 2022

@author: changai
"""

"""
Elementary steps:
    
CO2(g) + * + H+ + e- -> HOCO*   (dG1)
dG1 = G_HOCO* - G_CO2g - G* - 1/2 G_H2g
Gb_HOCO = dG1

HOCO* + H+ + e- -> CO* + H2O(g) (dG2)
dG2 = G_CO* + G_H2Og - G_HOCO* - 1/2 G_H2g
Gb_CO_ref_e = dG2 - dG1 # reference electrochemical initial step

CO* -> CO(g) + *                  (dG3)
dG3 = G_COg + G* - G_CO*
Gb_CO = -dG3

H+ + e- + * -> H*                 (dG4)
dG4 = G_H - G* - 1/2 G_H2g
Gb_H = dG4

H2O(g) + * -> OH* + H+ + e-       (dG5)
dG5 = G_OH + 1/2 G_H2g - G* - G_H2O(g)
Gb_OH = dG5

Pd45Ti9H54 -> Pd54H54 - 9*Pd2+ + 9*Ti3+ + 9*e-         (dG6)
dG6 = G_Pd54H54 - 9*G_Pd2+ + 9*G_Ti3+ - 9*eU - G_Pd45Ti9H54

for remove one Ti atom on the overlayer surface:
Pd45Ti9H54 -> Pd45Ti8H54 + Ti3+ + 3*e-         (dG6)

for remove first Ti bilayer on the overlayer surface:
Pd45Ti9H54 -> Pd45H45 + 9*Ti3+ + 27*e- + 9*H+ + 9*e-    (dG6)

G = E_dft + E_zpe + C_p - TS

delta_G(U, pH) = delta_G(U=0, pH=0) - ne * U + n * k_B * T * pH * ln(10)
Here e is -1, reference Andrew`s paper
μ( H+(aq)) + μ(e-) = 1/2μH2(g) - eUSHE + kBT ln(10) pH from Heine`s theis

Note:   alpha = np.log(10) * self.kT
        if name == 'e-':
            energy = -U
        elif name == 'H+(aq)':
            energy = -pH * alpha
        , which is from ase
    
"""

from pcat.lib.io import pd_read_excel
import numpy as np
import pcat.utils.constants as const
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({'mathtext.default':  'regular', 'figure.dpi': 150})

def pourbaix_diagram(U, pH):
    """Get pourbaix diagram as a function of applied potential and pH value"""
    # df = pd_read_excel(filename=xls_name, sheet=sheet_name_dGs)
    # df = df.iloc[10]
    # surface = df['Surface']
    # dG1 = df['dG1']
    # dG2 = df['dG2']
    # dG3 = df['dG3']
    # dG4 = df['dG4']
    # dG5 = df['dG5']
    G_H2g = -7.096

    
    # for Ti in configuration of overlayer
    E_Pd54H54 = -285.3708828
    E_Pd45Ti9H54 = -330.0370939 # Ti overlayer
    E_Pd45Ti8H54 = -322.46471945 # remove 1 Ti
    E_Pd45H45 = -236.30774029 # remove first bilayer
    
    E_Pd_bulk = -1.950920655
    E_Ti_bulk = -7.244883085
    G_Pd2plus = E_Pd_bulk + 2 * 0.915 + 0.0592 * np.log(10**(-6))
    G_Ti3plus = E_Ti_bulk + 3 * 0.915 + 0.0592 * np.log(10**(-6))
    G_Ti2plus = E_Ti_bulk + 2 * 0.915 + 0.0592 * np.log(10**(-6))
    dG1 = 0.588
    dG2 = -0.720
    dG3 = -0.255
    dG4 = 0.138
    dG5 = -0.280
    
    # for pure PdH
    # dG1 = 0.820
    # dG2 = -0.603
    # dG3 = 0.093
    # dG4 = 0.501
    # dG5 = 1.581
    
    # for Ti in configuration of overlayer
    
    
    kB = const.kB
    T = 297.15
    
    N, M = 200, 200
    
    # pH = 0
    U_model = np.linspace(min(U), max(U), N)
    if type(pH) == int or type(pH) == float:
        pH_model = [pH]
    else:
        pH_model = np.linspace(min(pH), max(pH), M)
    Us = []
    pHs = []
    Gb_HOCOs = []
    Gb_CO_ref_es = []
    Gb_COs = []
    Gb_Hs = []
    Gb_OHs = []
    G_Ti_overly_to_Pds = []
    G_rm_one_Tis = []
    G_rm_first_bilayer_Tis = []
    bare_PdHs = [] 
    colors = []
    Us_acc = np.zeros(8) # accumulate U in order to calculate average U
    pHs_acc = np.zeros(8)
    count = np.zeros(8)
    for i, ph in enumerate(pH_model):
        for j, u in enumerate(U_model):
            bare_PdH = 0
            Gb_HOCO = dG1 + u + kB * T * ph * np.log(10)
            Gb_CO_ref_e = dG2 - dG1
            # Gb_CO = dG2 - dG1
            Gb_CO = -dG3
            Gb_H = dG4 + u + kB * T * ph * np.log(10)
            Gb_OH = dG5 - u - kB * T * ph * np.log(10)
            
            G_Ti_overly_to_Pd = (E_Pd54H54 - 9*G_Pd2plus + 9*G_Ti3plus - 9*u - E_Pd45Ti9H54) / 9.0 # replace Ti overlayer to Pd overlayer
            G_rm_one_Ti = E_Pd45Ti8H54 + G_Ti3plus - 3*u - E_Pd45Ti9H54
            G_rm_first_bilayer_Ti = (E_Pd45H45 + 9*G_Ti3plus - 36*u + 9*0.5*G_H2g - E_Pd45Ti9H54 + 9 * kB * T * ph * np.log(10))/ 9.0
            # G_rm_first_bilayer_Ti = 0
            
            # G_Ti_overly_to_Pd = (E_Pd54H54 - 9*G_Pd2plus + 9*G_Ti2plus - E_Pd45Ti9H54) / 9.0 # replace Ti overlayer to Pd overlayer
            # G_Ti_overly_to_Pd = 0
            # G_rm_one_Ti = E_Pd45Ti8H54 + G_Ti2plus - 2*u - E_Pd45Ti9H54
            # G_rm_first_bilayer_Ti = (E_Pd45H45 + 9*G_Ti2plus - 27*u + 9*0.5*G_H2g - E_Pd45Ti9H54 - 9 * kB * T * ph * np.log(10))/ 9.0
            # G_rm_first_bilayer_Ti = 0
            
            # Pd45Ti9H54 -> Pd45H45 + 9*Ti3+ + 27*e- + 9*H+ + 9*e-
            # print(u, Gb_HOCO)
            # plt.scatter(u, Gb_HOCO)
            Us.append(u)
            pHs.append(ph)
            
            # Gb_HOCO = Gb_HOCO/4
            # Gb_CO = Gb_CO/2
            # Gb_H = Gb_H
            # Gb_OH = Gb_OH/2
            
            Gb_HOCOs.append(Gb_HOCO)
            # Gb_CO_ref_es.append(Gb_CO_ref_e)
            Gb_COs.append(Gb_CO)
            Gb_Hs.append(Gb_H)
            Gb_OHs.append(Gb_OH)
            G_Ti_overly_to_Pds.append(G_Ti_overly_to_Pd)
            G_rm_one_Tis.append(G_rm_one_Ti)
            G_rm_first_bilayer_Tis.append(G_rm_first_bilayer_Ti)
            bare_PdHs.append(bare_PdH)
            
            min_dot = min(Gb_HOCO, Gb_CO, Gb_H, Gb_OH, G_Ti_overly_to_Pd, G_rm_one_Ti, G_rm_first_bilayer_Ti, bare_PdH)
            
            if min_dot == Gb_HOCO:
                color = 'blue'
                Us_acc[0] += u
                pHs_acc[0] += ph
                count[0] += 1
            elif  min_dot == Gb_CO:
                color = 'orange'
                Us_acc[1] += u
                pHs_acc[1] += ph
                count[1] += 1
            elif  min_dot == Gb_H:
                color = 'green'
                Us_acc[2] += u
                pHs_acc[2] += ph
                count[2] += 1
            elif  min_dot == Gb_OH:
                color = 'brown'
                Us_acc[3] += u
                pHs_acc[3] += ph
                count[3] += 1
            elif min_dot == G_Ti_overly_to_Pd:
                color = 'red'
                Us_acc[4] += u
                pHs_acc[4] += ph
                count[4] += 1
            elif min_dot == G_rm_one_Ti:
                color = 'olive'
                Us_acc[5] += u
                pHs_acc[5] += ph
                count[5] += 1
            elif min_dot == G_rm_first_bilayer_Ti:
                color = 'gray'
                Us_acc[6] += u
                pHs_acc[6] += ph
                count[6] += 1
            if min_dot == bare_PdH:
                color = 'yellow'
                Us_acc[7] += u
                pHs_acc[7] += ph
                count[7] += 1
            
            
            colors.append(color)
            
    if type(pH) == int or type(pH) == float:
        plt.plot(Us, Gb_HOCOs, label='Gb_HOCOs', color='blue')
        # plt.plot(Us, Gb_CO_ref_es, label='Gb_CO_ref_es')
        plt.plot(Us, Gb_COs, label='Gb_COs', color = 'orange')
        plt.plot(Us, Gb_Hs, label='Gb_Hs', color = 'green')
        plt.plot(Us, Gb_OHs, label='Gb_OHs', color = 'brown')
        plt.plot(Us, G_Ti_overly_to_Pds, label='G_Ti_overly_to_Pd', color = 'red')
        plt.plot(Us, G_rm_one_Tis, label='G_rm_one_Ti', color = 'olive')
        plt.plot(Us, G_rm_first_bilayer_Tis, label='G_rm_first_bilayer_Ti', color = 'gray')
        plt.plot(Us, bare_PdHs, label='bare_PdH', color = 'yellow')
        plt.xlabel('$U_{SHE}$')
        plt.ylabel('$\Delta G$ (eV/per adsorbate)')
        plt.legend(fontsize=6)
        plt.show()
    else:
        plt.scatter(pHs, Us, c=colors, marker='o', zorder=2)
        plt.xlabel('pH')
        plt.ylabel('$U_{SHE}$ (V)')
        for i, txt in enumerate(['Gb_HOCOs', 'Gb_COs', 'Gb_Hs', 'Gb_OHs', 'G_Ti_overly_to_Pd', 'G_rm_one_Ti', 'G_rm_first_bilayer_Ti', 'bare_PdH']):
            x = pHs_acc[i]/count[i]
            y = Us_acc[i]/count[i] 
            plt.text(x, y, txt, horizontalalignment='center')
        
    tuples = {'Us': Us,
              'pHs': pHs,
              'Gb_HOCOs': Gb_HOCOs,
              # 'Gb_CO_ref_es': Gb_CO_ref_es,
              'Gb_COs': Gb_COs,
              'Gb_Hs': Gb_Hs,
              'Gb_OHs': Gb_OHs,
              'Colors': colors,
              }
    df_Pourbaix = pd.DataFrame(tuples)
    # df_bonds.to_excel(xls_name, sheet_name_origin, float_format='%.3f')
    
        
if __name__ == '__main__':
    
    system_name = 'collect_vasp_PdHy_v3'
    
    xls_name = f'../data/{system_name}.xlsx'
    fig_dir = '../figures'
    
    sheet_name_dGs = 'dGs'
    
    U = [-2, 3]
    pH = 0
    pourbaix_diagram(U, pH)
    
    pH = [0, 14]
    pourbaix_diagram(U, pH)
    
    