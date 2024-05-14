# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 12:07:44 2022

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

Pd45Nb9H54 -> Pd54H54 - 9*Pd2+ + 9*Nb3+ + 9*e-         (dG6)
dG6 = G_Pd64H64 - 9*G_Pd2+ + 9*G_Nb3+ - 9*eU - G_Pd55Nb9H64

for remove one Nb atom on the overlayer surface:
Pd45Nb9H54 -> Pd45Nb8H54 + Nb3+ + 3*e-         (dG6)

for remove first Nb bilayer on the overlayer surface:
Pd45Nb9H54 -> Pd45H45 + 9*Nb3+ + 27*e- + 9*H+ + 9*e-    (dG6)

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
import matplotlib
plt.rcParams.update({'mathtext.default':  'regular', 'figure.dpi': 150})
# matplotlib.use('TkAgg')

def pourbaix_diagram(U, pH, system):
# def pourbaix_diagram(U, pH, Nb_overly=False, Ti_overly=False, Ti_paral=False, Pd_pure=False):
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
    G_H2Og = -12.827 # eV
    
    E_Pd54H54 = -285.3708828 # pure PdH
    E_Pd53H54 = -282.78668443 # remove 1 Pd
    E_Pd53H53 = -279.36308240 # remove 1 Pd and 1 H
    
    E_Pd45Nb9H54 = -336.2919741 # Nb overlayer
    E_Pd45Nb8H54 = -327.95832335 # remove 1 Nb
    E_Pd45Nb8H53 = -323.66792350 # remove 1 Nb and 1 H
    E_Pd45H45 = -236.30774029 # remove first bilayer
    
    E_Pd45Ti9H54 = -330.0370939 # Ti overlayer
    E_Pd45Ti8H54 = -322.46471945 # remove 1 Ti
    E_Pd45Ti8H53 = -318.73737847 # remove 1 Ti and 1 H
    E_Pd45H45 = -236.30774029 # remove first bilayer
    
    E_Pd50Ti4H54 = -305.3263439 # Ti parallelogram
    E_Pd50Ti3H54 = -297.69180749 # remove 1 Ti
    E_Pd50Ti3H53 = -293.96607660 # remove 1 Ti and 1 H
    E_Pd50H50 = -262.23018405 # remove first bilayer
    
    E_Pd_bulk = -1.950920655
    E_Nb_bulk = -7.244883085
    E_Ti_bulk = -5.85829807
    G_Pd2plus = E_Pd_bulk + 2 * 0.915 + 0.0592 * np.log(10**(-6))
    G_Nb3plus = E_Nb_bulk + 3 * (-0.8) + 0.0592 * np.log(10**(-6))
    G_Nb_OH_4 = E_Nb_bulk + 4 * G_H2Og - 2 * G_H2g + 5*(-0.537) + 0.0592 * np.log(10**(-6))
    # G_Ti2plus = E_Ti_bulk + 2 * (-1.63) + 0.0592 * np.log(10**(-6))
    # G_Ti2plus = E_Ti_bulk + 2 * (-1.628) + 0.0592 * np.log(10**(-6))
    # G_Ti3plus = E_Ti_bulk + 3 * (-1.26) + 0.0592 * np.log(10**(-6))
    G_Ti2plus = E_Ti_bulk + 2 * (-1.60) + 0.0592 * np.log(10**(-6))
    G_Ti3plus = E_Ti_bulk + 3 * (-1.37) + 0.0592 * np.log(10**(-6))
    G_Ti_OH_2 = E_Ti_bulk + 2 * G_H2Og - G_H2g + 4*(-1) + 0.0592 * np.log(10**(-6))
    
    if system == 'Nb_overly':
        dG1 = 0.624
        dG2 = -0.725
        dG3 = -0.224
        dG4 = 0.467
        dG5 = -0.132
    elif system == 'Ti_overly':
        dG1 = 0.588
        dG2 = -0.720
        dG3 = -0.255
        dG4 = 0.138
        dG5 = -0.280
    elif system == 'Ti_paral':
        dG1 = 0.509
        dG2 = -0.674
        dG3 = -0.288
        dG4 = 0.407
        dG5 = -0.222
    elif system == 'Pd_pure':
        dG1 = 0.820
        dG2 = -0.603
        dG3 = 0.093
        dG4 = 0.501
        dG5 = 1.581
    
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
    G_Nb_overly_to_Pds = []
    G_rm_one_Ms = []
    G_rm_first_bilayer_Ms = []
    G_rm_one_M2s = []
    G_rm_first_bilayer_M2s = []
    G_rm_one_M3s = []
    G_rm_first_bilayer_M3s = []
    G_rm_one_M_OH_2s = []
    G_rm_one_M_OH_4s = []
    G_rm_one_M2_Hs = []
    G_rm_one_M3_Hs = []
    bare_PdHs = [] 
    colors = []
    Us_acc = np.zeros(20) # accumulate U in order to calculate average U
    pHs_acc = np.zeros(20)
    count = np.zeros(20)
    color_list = ['blue', 'orange', 'green', 'brown', 'red', 'olive', 'gray', 'yellow', 'purple', 'pink']
    for i, ph in enumerate(pH_model):
        for j, u in enumerate(U_model):
            bare_PdH = 0
            bare_PdHs.append(bare_PdH)
            
            Gb_HOCO = dG1 + u + kB * T * ph * np.log(10)
            Gb_HOCOs.append(Gb_HOCO)
            
            Gb_CO_ref_e = dG2 - dG1
            Gb_CO = -dG3
            Gb_COs.append(Gb_CO) # or Gb_CO_ref_e
            # Gb_COs.append(Gb_CO_ref_e)
            
            Gb_H = dG4 + u + kB * T * ph * np.log(10)
            Gb_Hs.append(Gb_H)
            
            Gb_OH = dG5 - u - kB * T * ph * np.log(10)
            Gb_OHs.append(Gb_OH)
            
            
            if system == 'Nb_overly':
                # G_Nb_overly_to_Pd = (E_Pd54H54 - 9*G_Pd2plus + 9*G_Nb3plus - 9*u - E_Pd45Nb9H54) / 9.0 # replace Nb overlayer to Pd overlayer
                # G_Nb_overly_to_Pds.append(G_Nb_overly_to_Pd)
                
                # only remove one Nb to be Nb3+ on Pd45Nb9H54 surface
                G_rm_one_M3 = E_Pd45Nb8H54 + G_Nb3plus - 3*u - E_Pd45Nb9H54
                G_rm_one_M3s.append(G_rm_one_M3)
                
                G_rm_one_M3_H = E_Pd45Nb8H53 + G_Nb3plus + 0.5*G_H2g - 4*u - kB * T * ph * np.log(10) - E_Pd45Nb9H54 
                G_rm_one_M3_Hs.append(G_rm_one_M3_H)
                
                # remove first bilayer including overlayer Nb and H
                G_rm_first_bilayer_M3 = (E_Pd45H45 + 9*G_Nb3plus - 36*u + 9*0.5*G_H2g - E_Pd45Nb9H54 - 9 * kB * T * ph * np.log(10))/ 9.0
                G_rm_first_bilayer_M3s.append(G_rm_first_bilayer_M3)
                # G_rm_first_bilayer_M = (E_Pd45H45 + 9*G_Pd2plus - 27*u + 9*0.5*G_H2g - E_Pd54H54 - 9 * kB * T * ph * np.log(10))/ 9.0
                
                G_rm_one_M_OH_4 = E_Pd45Nb8H54 + G_Nb_OH_4 + 2*G_H2g - 5 * u - 4 * G_H2Og - E_Pd45Nb9H54
                G_rm_one_M_OH_4s.append(G_rm_one_M_OH_4)
                
                # G_values = [Gb_HOCO, Gb_CO, Gb_H, Gb_OH, G_rm_first_bilayer_M3, bare_PdH]
                G_values = [Gb_HOCO, Gb_CO, Gb_H, Gb_OH, G_rm_one_M3_H, bare_PdH]
                # G_values = [Gb_HOCO, Gb_CO, Gb_H, Gb_OH, G_rm_one_M3, bare_PdH]
                # G_values = [Gb_HOCO, Gb_CO, Gb_H, Gb_OH, G_rm_one_M3_H, G_rm_one_M_OH_4, bare_PdH]
                # G_values = [Gb_HOCO, Gb_CO, Gb_H, Gb_OH, G_rm_one_M3, G_rm_one_M3_H, bare_PdH]
                # G_values = [Gb_HOCO, Gb_CO, Gb_H, Gb_OH, G_rm_one_M_OH_4, bare_PdH]
            
            elif system == 'Ti_overly':
                # remove only one Ti to be Ti2+
                G_rm_one_M2 = E_Pd45Ti8H54 + G_Ti2plus - 2*u - E_Pd45Ti9H54
                G_rm_one_M2s.append(G_rm_one_M2)
                # remove only one Ti to Ti3+
                G_rm_one_M3 = E_Pd45Ti8H54 + G_Ti3plus - 3*u - E_Pd45Ti9H54
                G_rm_one_M3s.append(G_rm_one_M3)
                
                G_rm_one_M2_H = E_Pd45Ti8H53 + G_Ti2plus + 0.5*G_H2g - 3*u - kB * T * ph * np.log(10) - E_Pd45Ti9H54 
                G_rm_one_M2_Hs.append(G_rm_one_M2_H)
                
                G_rm_one_M3_H = E_Pd45Ti8H53 + G_Ti3plus + 0.5*G_H2g - 4*u - kB * T * ph * np.log(10) - E_Pd45Ti9H54 
                G_rm_one_M3_Hs.append(G_rm_one_M3_H)
                
                # remove first bilayer to Ti2+ including overlayer Ti and H
                G_rm_first_bilayer_M2 = (E_Pd45H45 + 9*G_Ti2plus - 27*u + 9*0.5*G_H2g - E_Pd45Ti9H54 - 9 * kB * T * ph * np.log(10))/ 9.0
                G_rm_first_bilayer_M2s.append(G_rm_first_bilayer_M2)
                # remove first bilayer to Ti3+ including overlayer Ti and H
                G_rm_first_bilayer_M3 = (E_Pd45H45 + 9*G_Ti3plus - 36*u + 9*0.5*G_H2g - E_Pd45Ti9H54 - 9 * kB * T * ph * np.log(10))/ 9.0
                G_rm_first_bilayer_M3s.append(G_rm_first_bilayer_M3)
                
                # G_Ti_OH_2 = 2*G_H2Og + E_Ti_bulk - G_H2g + 2* (-1) 
                G_rm_one_M_OH_2 = E_Pd45Ti8H54 + G_Ti_OH_2 + G_H2g - 4*u - 2*G_H2Og - E_Pd45Ti9H54 - 2 * kB * T * ph * np.log(10)
                G_rm_one_M_OH_2s.append(G_rm_one_M_OH_2)
                
                G_values = [Gb_HOCO, Gb_CO, Gb_H, Gb_OH, G_rm_one_M2_H, G_rm_one_M3_H, bare_PdH]
                # G_values = [Gb_HOCO, Gb_CO, Gb_H, Gb_OH, G_rm_one_M2, G_rm_one_M3, bare_PdH]
                # G_values = [Gb_HOCO, Gb_CO, Gb_H, Gb_OH, G_rm_one_M_OH_2, bare_PdH]
                # G_values = [Gb_HOCO, Gb_CO, Gb_H, Gb_OH, G_rm_one_M2_H, bare_PdH]
            
            elif system == 'Ti_paral':
                # remove only one Ti to be Ti2+
                G_rm_one_M2 = E_Pd50Ti3H54 + G_Ti2plus - 2*u - E_Pd50Ti4H54
                G_rm_one_M2s.append(G_rm_one_M2)
                # remove only one Ti to Ti3+
                G_rm_one_M3 = E_Pd50Ti3H54 + G_Ti3plus - 3*u - E_Pd50Ti4H54
                G_rm_one_M3s.append(G_rm_one_M3)
                
                G_rm_one_M2_H = E_Pd50Ti3H53 + G_Ti2plus + 0.5*G_H2g - 3*u - kB * T * ph * np.log(10) - E_Pd50Ti4H54 
                G_rm_one_M2_Hs.append(G_rm_one_M2_H)
                
                G_rm_one_M3_H = E_Pd50Ti3H53 + G_Ti3plus + 0.5*G_H2g - 4*u - kB * T * ph * np.log(10) - E_Pd50Ti4H54 
                G_rm_one_M3_Hs.append(G_rm_one_M3_H)
                
                # remove first bilayer to Ti2+ including overlayer Ti and H
                G_rm_first_bilayer_M2 = (E_Pd50H50 + 4*G_Ti2plus - 12*u + 4*0.5*G_H2g - E_Pd50Ti4H54 - 4 * kB * T * ph * np.log(10))/ 4.0
                G_rm_first_bilayer_M2s.append(G_rm_first_bilayer_M2)
                # remove first bilayer to Ti3+ including overlayer Ti and H
                G_rm_first_bilayer_M3 = (E_Pd50H50 + 4*G_Ti3plus - 16*u + 4*0.5*G_H2g - E_Pd50Ti4H54 - 4 * kB * T * ph * np.log(10))/ 4.0
                G_rm_first_bilayer_M3s.append(G_rm_first_bilayer_M3)
                
                G_rm_one_M_OH_2 = E_Pd50Ti3H53 + G_Ti_OH_2 + G_H2g - 4*u - 2*G_H2Og - E_Pd50Ti4H54 - 2 * kB * T * ph * np.log(10)
                G_rm_one_M_OH_2s.append(G_rm_one_M_OH_2)
                
                G_values = [Gb_HOCO, Gb_CO, Gb_H, Gb_OH, G_rm_one_M2_H, G_rm_one_M3_H, bare_PdH]
                # G_values = [Gb_HOCO, Gb_CO, Gb_H, Gb_OH, G_rm_one_M2, G_rm_one_M3, bare_PdH]
            
            elif system == 'Pd_pure':
                # only remove one Nb to be Pd2+ on E_Pd54H54 surface
                G_rm_one_M2 = E_Pd53H54 + G_Pd2plus - 2*u - E_Pd54H54
                G_rm_one_M2s.append(G_rm_one_M2)
                
                G_rm_one_M2_H = E_Pd53H53 + G_Pd2plus + 0.5*G_H2g  - 3*u - kB * T * ph * np.log(10) - E_Pd54H54
                G_rm_one_M2_Hs.append(G_rm_one_M2_H)
    
                # remove first bilayer Pd2+ including overlayer Pd and H
                G_rm_first_bilayer_M2 = (E_Pd45H45 + 9*G_Pd2plus - 27*u + 9*0.5*G_H2g - E_Pd54H54 - 9 * kB * T * ph * np.log(10))/ 9.0
                G_rm_first_bilayer_M2s.append(G_rm_first_bilayer_M2)
                # G_rm_first_bilayer_M = (E_Pd45H45 + 9*G_Pd2plus - 27*u + 9*0.5*G_H2g - E_Pd54H54 - 9 * kB * T * ph * np.log(10))/ 9.0
                
                G_values = [Gb_HOCO, Gb_CO, Gb_H, Gb_OH, G_rm_one_M2_H, bare_PdH]
                # G_values = [Gb_HOCO, Gb_CO, Gb_H, Gb_OH, G_rm_one_M2, bare_PdH]
                
                
            # G_values = [Gb_HOCO, Gb_CO, Gb_H, Gb_OH, G_rm_first_bilayer_M, bare_PdH]
            # G_values = [Gb_HOCO, Gb_CO, Gb_H, Gb_OH, G_rm_one_M, bare_PdH]
            # G_values = [Gb_HOCO, Gb_CO, Gb_H, Gb_OH, G_rm_first_bilayer_M2s, G_rm_first_bilayer_M3s, bare_PdH]
            min_dot = min(G_values)
            for i, G_value in enumerate(G_values):
                if min_dot == G_value:
                    color = color_list[i]
                    Us_acc[i] += u
                    pHs_acc[i] += ph
                    count[i] += 1
            Us.append(u)
            pHs.append(ph)
            colors.append(color)
            
    if system == 'Nb_overly':
        tuples = {'Gb_HOCOs': Gb_HOCOs,
                  # 'Gb_CO_ref_es': Gb_CO_ref_es,
                  'Gb_COs': Gb_COs,
                  'Gb_Hs': Gb_Hs,
                  'Gb_OHs': Gb_OHs,
                  # 'G_rm_one_M2': G_rm_one_M2s,
                   # 'G_rm_one_M3': G_rm_one_M3s,
                   # 'G_rm_first_bilayer_M3': G_rm_first_bilayer_M3s,
                    # 'G_rm_one_M_OH_4': G_rm_one_M_OH_4s,
                  'G_rm_one_M3_H': G_rm_one_M3_Hs,
                  'bare_PdH': bare_PdHs
                  }
    elif system == 'Ti_overly':
        tuples = {'Gb_HOCOs': Gb_HOCOs,
                  # 'Gb_CO_ref_es': Gb_CO_ref_es,
                  'Gb_COs': Gb_COs,
                  'Gb_Hs': Gb_Hs,
                  'Gb_OHs': Gb_OHs,
                   # 'G_rm_one_M2': G_rm_one_M2s,
                   # 'G_rm_one_M3': G_rm_one_M3s,
                  # 'G_rm_first_bilayer_M2': G_rm_first_bilayer_M2s,
                  # 'G_rm_first_bilayer_M3': G_rm_first_bilayer_M3s,
                   # 'G_rm_one_M_OH_2': G_rm_one_M_OH_2s,
                   'G_rm_one_M2_H': G_rm_one_M2_Hs,
                   'G_rm_one_M3_H': G_rm_one_M3_Hs,
                  'bare_PdH': bare_PdHs
                  }
    elif system == 'Ti_paral':
        tuples = {'Gb_HOCOs': Gb_HOCOs,
                  # 'Gb_CO_ref_es': Gb_CO_ref_es,
                  'Gb_COs': Gb_COs,
                  'Gb_Hs': Gb_Hs,
                  'Gb_OHs': Gb_OHs,
                  # 'G_rm_one_M2': G_rm_one_M2s,
                  # 'G_rm_one_M3': G_rm_one_M3s,
                  # 'G_rm_first_bilayer_M2': G_rm_first_bilayer_M2s,
                  # 'G_rm_first_bilayer_M3': G_rm_first_bilayer_M3s,
                   'G_rm_one_M2_H': G_rm_one_M2_Hs,
                   'G_rm_one_M3_H': G_rm_one_M3_Hs,
                  'bare_PdH': bare_PdHs
                  }
    elif system == 'Pd_pure':
        tuples = {'Gb_HOCOs': Gb_HOCOs,
                  # 'Gb_CO_ref_es': Gb_CO_ref_es,
                  'Gb_COs': Gb_COs,
                  'Gb_Hs': Gb_Hs,
                  'Gb_OHs': Gb_OHs,
                  'G_rm_one_M2_H': G_rm_one_M2_Hs,
                  # 'G_rm_one_M2': G_rm_one_M2s,
                   # 'G_rm_one_M3': G_rm_one_M3s,
                    # 'G_rm_first_bilayer_M2': G_rm_first_bilayer_M2s,
                   # 'G_rm_first_bilayer_M3': G_rm_first_bilayer_M3s,
                   # 'G_rm_one_M2': G_rm_one_M2s,
                  'bare_PdH': bare_PdHs
                  }
    df = pd.DataFrame(tuples)
    if type(pH) == int or type(pH) == float:
        # plot linear
        for i, column in enumerate(df.columns):
            plt.plot(Us, df[column].values, label=column, color=color_list[i])
        plt.xlabel('$U_{SHE}$')
        plt.ylabel('$\Delta G$ (eV/per adsorbate)')
        plt.legend(fontsize=8)
        # plt.show()
    else:
        # plot 2D
        plt.scatter(pHs, Us, c=colors, marker='o', zorder=2, s=2)
        plt.xlabel('pH')
        plt.ylabel('$U_{SHE}$ (V)')
        for i, txt in enumerate(df.columns):
            x = pHs_acc[i]/count[i]
            y = Us_acc[i]/count[i] 
            plt.text(x, y, txt, horizontalalignment='center')

        
if __name__ == '__main__':
    
    system_name = 'collect_vasp_PdHy_v3'
    
    xls_name = f'../data/{system_name}.xlsx'
    fig_dir = '../figures'
    
    sheet_name_dGs = 'dGs'

    for system in ['Nb_overly', 'Ti_overly', 'Ti_paral', 'Pd_pure']:
        U = [-2, 3]
        pH = 0
        pourbaix_diagram(U, pH, system)
        plt.title(str(system))
        plt.show()
        
        pH = [0, 14]
        pourbaix_diagram(U, pH, system)
        plt.title(str(system))
        plt.show()