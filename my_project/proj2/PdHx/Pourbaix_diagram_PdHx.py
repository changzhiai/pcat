# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:06:23 2022

@author: changai
"""

"""
Elementary steps:

Pd64H64 -> Pd64Hx + (64-x) * (H+ + e-)  [dG_Hx]
dG_Hx = E_Pd64Hx + (64-x) * (1/2. * G_H2) - E_Pd64H64


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
    df = pd_read_excel(filename=xls_name, sheet=sheet_name_origin)
    df = df[df['Adsorbate']=='surface']
    df['num_H'] = round(df['Cons_H']*64).astype(int)
    # df['dG_Hx'] = [0] * len(df['num_H'])
    # print(df)
    
    E_Pd64H64 = df[df['num_H']==64]['Energy'].values[0]
    G_H2g = -7.096
    nums_row = len(df.index)
    dG_Hxs = []
    # for i in list(reversed(range(48, 64))):
    for i in range(nums_row):
        row = df.iloc[[i]]
        # E_Pd64Hx = df[df['num_H']==i]['Energy'].values[0]
        E_Pd64Hx = row['Energy'].values[0]
        num_H = row['num_H'].values[0]
        dG_Hx = E_Pd64Hx + (64-num_H)*(1/2.)*G_H2g - E_Pd64H64
        print(dG_Hx)
        # row['dG_Hx'].values[0] = dG_Hx
        dG_Hxs.append(dG_Hx)

    df['dG_Hx'] = dG_Hxs
    
    
    kB = const.kB
    T = 297.15
    
    N, M = 200, 200
    U_model = np.linspace(min(U), max(U), N)
    if type(pH) == int or type(pH) == float:
        pH_model = [pH]
    else:
        pH_model = np.linspace(min(pH), max(pH), M)


    formulas = []
    Us = []
    pHs = []
    dG_Hx_dps = []
    num_Hs = []
    for i in range(nums_row):
        row = df.iloc[[i]]
        dG_Hx = row['dG_Hx'].values[0]
        num_H = row['num_H'].values[0]
        formula = row['Surface'].values[0]

        # plt.figure()
        for i, ph in enumerate(pH_model):
            for j, u in enumerate(U_model):
                dG_Hx_dp = dG_Hx - num_H*u - num_H * kB * T * ph * np.log(10)
                # dG_Hx_dp = dG_Hx_dp/num_H
                formulas.append(formula)
                Us.append(u)
                pHs.append(ph)
                dG_Hx_dps.append(dG_Hx_dp)
                num_Hs.append(num_H)
    
    tuples = {'formulas': formulas,
              'Us': Us,
              'pHs': pHs,
              'dG_Hx_dps': dG_Hx_dps,
              'num_Hs': num_Hs
              }
    df_plot = pd.DataFrame(tuples)
    
    colors = {}
    formulas_set = df_plot['formulas'].unique()   
    if type(pH) == int or type(pH) == float:
        for formu in formulas_set:
            df_sub = df_plot[df_plot['formulas']==formu]
            Us_sub = df_sub['Us']
            dG_Hx_dps_sub = df_sub['dG_Hx_dps']
            line, = plt.plot(Us_sub, dG_Hx_dps_sub, label=f'{formu}')
            c = line.get_color()
            colors[formu] = c
    
    df_min = pd.DataFrame()
    for j, u in enumerate(U_model):
        df_sub = df_plot[df_plot['Us']==u]
        row_min = df_sub[df_sub['dG_Hx_dps']==df_sub['dG_Hx_dps'].min()]
        df_min = df_min.append(row_min, ignore_index=True)
    
    # plt.figure()
    Us_sub = df_min['Us']
    dG_Hx_dps_sub = df_min['dG_Hx_dps']
    nums_row = len(df_min.index)
    print(df_min['formulas'].unique())
    for i in range(nums_row):
        row = df_min.iloc[[i]]
        formu = row['formulas'].values[0]
        color = colors[formu]
        plt.scatter(row['Us'], row['dG_Hx_dps'], c=color, linewidth=6)
    # plt.plot(Us_sub, dG_Hx_dps_sub, label=f'min', linewidth=6)   
    plt.xlabel('$U_{SHE}$')
    plt.ylabel('$\Delta G$ (eV/)')
    plt.legend()
    plt.show()
    
    

        # else:
        #     plt.scatter(pHs, Us, c=colors, marker='o', zorder=2)
        #     plt.xlabel('pH')
        #     plt.ylabel('$U_{SHE}$ (V)')
        #     for i, txt in enumerate(['Gb_HOCOs', 'Gb_COs', 'Gb_Hs', 'Gb_OHs']):
        #         x = pHs_acc[i]/count[i]
        #         y = Us_acc[i]/count[i] 
        #         plt.text(x, y, txt, horizontalalignment='center')
        
    # tuples = {'Us': Us,
    #           'pHs': pHs,
    #           'Gb_HOCOs': Gb_HOCOs,
    #           'Gb_CO_ref_es': Gb_CO_ref_es,
    #           'Gb_COs': Gb_COs,
    #           'Gb_Hs': Gb_Hs,
    #           'Gb_OHs': Gb_OHs,
    #           'Colors': colors,
    #           }
    # df_FE = pd.DataFrame(tuples)
    
        
if __name__ == '__main__':
    
    system_name = 'collect_vasp_coverage_H'
    
    xls_name = f'../data/{system_name}.xlsx'
    fig_dir = '../figures'
    
    sheet_name_origin = 'Origin'
    sheet_name_dGs = 'dGs'
    
    U = [-2, 2]
    pH = 0
    pourbaix_diagram(U, pH)
    
    # pH = [0, 14]
    # pourbaix_diagram(U, pH)