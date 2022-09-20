# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:06:23 2022

@author: changai
"""

"""
Elementary steps:

Pd64H64 -> Pd64Hx + (64-x) * (H+ + e-)  [dG_Hx]
dG_Hx = E_Pd64Hx + (64-x) * (1/2. * G_H2) - E_Pd64H64  - (64-x)*u - (64-x) * kB * T * ph * np.log(10)


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
import matplotlib as mpl
from cycler import cycler
plt.rcParams.update({'mathtext.default':  'regular', 'figure.dpi': 300})


def to_xls(xls_name_Pourbaix):
    """Write data to excel"""
    kB = const.kB
    
    df = pd_read_excel(filename=xls_name, sheet=sheet_name_origin)
    df = df[df['Adsorbate']=='surface']
    df = df.sort_values(by=['Cons_H'])
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
        dG_Hxs.append(dG_Hx)

    df['dG_Hx'] = dG_Hxs
    
    U_model = np.linspace(min(U), max(U), N)
    if type(pH) == int or type(pH) == float:
        pH_model = [pH]
    else:
        pH_model = np.linspace(min(pH), max(pH), M)
    print('numbers of surface: ', nums_row)
    print('numbers of potential: ', len(U_model))
    print('numbers of pH: ', len(pH_model))
    NUM_COLORS = nums_row # 12
    cm = plt.get_cmap(plt.cm.jet)
    cs = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    mpl.rcParams['axes.prop_cycle'] = cycler(color=cs)
    from matplotlib.axes._axes import _log as matplotlib_axes_logger
    matplotlib_axes_logger.setLevel('ERROR')
    formulas_set = df['Surface'].unique()
    colors = {}
    hexs = []
    for i, formu in enumerate(formulas_set):
        hex_temp = mpl.colors.to_hex(cs[i])
        colors[formu] = hex_temp
        hexs.append(hex_temp)
        # colors[formu] = cs[i]
    tuples = {'formulas': formulas_set,
              'colors': hexs,
              }
    df_color = pd.DataFrame(tuples) # color list
    
    df_min = pd.DataFrame()
    df_all = pd.DataFrame()
    for i, ph in enumerate(pH_model):
        for j, u in enumerate(U_model):
            formulas_temp = []
            Us_temp = []
            pHs_temp = []
            dG_Hx_dps_temp = []
            num_Hs_temp = []
            color_list = []
            for k in range(nums_row):
                row = df.iloc[[k]]
                dG_Hx = row['dG_Hx'].values[0]
                num_H = row['num_H'].values[0]
                formula = row['Surface'].values[0]
                dG_Hx_dp = dG_Hx - (64-num_H)*u - (64-num_H) * kB * T * ph * np.log(10)
                # dG_Hx_dp = dG_Hx_dp/num_H
                color = colors[formula]
                
                formulas_temp.append(formula)
                Us_temp.append(u)
                pHs_temp.append(ph)
                dG_Hx_dps_temp.append(dG_Hx_dp)
                num_Hs_temp.append(num_H)
                color_list.append(color)
                
            tuples = {'formulas': formulas_temp,
                      'Us': Us_temp,
                      'pHs': pHs_temp,
                      'dG_Hx_dps': dG_Hx_dps_temp,
                      'num_Hs': num_Hs_temp,
                      'color': color_list,
                      }
            df_temp = pd.DataFrame(tuples)
            df_all = df_all.append(df_temp, ignore_index=True)
            row_min = df_temp[df_temp['dG_Hx_dps']==df_temp['dG_Hx_dps'].min()]
            df_min = df_min.append(row_min, ignore_index=True)
    
    df_color.to_excel(xls_name_Pourbaix, sheet_name_color_list, float_format='%.5f')
    
    with pd.ExcelWriter(xls_name_Pourbaix, engine='openpyxl', mode='a') as writer:
        df_all.to_excel(writer, sheet_name=sheet_name_all, float_format='%.5f')
    with pd.ExcelWriter(xls_name_Pourbaix, engine='openpyxl', mode='a') as writer:
        df_min.to_excel(writer, sheet_name=sheet_name_min, float_format='%.5f')

def plot_1d(xls_name_Pourbaix):
    """Plot 1d Pourbaix diagram: dG vs. U"""
    df_colors = pd_read_excel(filename=xls_name_Pourbaix, sheet=sheet_name_color_list)
    colors = df_colors.set_index('formulas')['colors'].to_dict()
    
    df_all = pd_read_excel(filename=xls_name_Pourbaix, sheet=sheet_name_all)
    df_min = pd_read_excel(filename=xls_name_Pourbaix, sheet=sheet_name_min)
    print('total rows: ', len(df_all))
    formulas_set = df_all['formulas'].unique()
    # fig = plt.figure()
    for formu in formulas_set:
        df_sub = df_all[df_all['formulas']==formu]
        Us_sub = df_sub['Us']
        dG_Hx_dps_sub = df_sub['dG_Hx_dps']
        # color = eval(colors[formu])
        color = colors[formu]
        SHE = True
        if SHE: # SHE
            line, = plt.plot(Us_sub, dG_Hx_dps_sub, c=color, label=f'{formu}')
        else: # RHE
            line, = plt.plot(Us_sub-kB * T * pH * np.log(10), dG_Hx_dps_sub, c=color, label=f'{formu}')
        # c = line.get_color()
        # colors[formu] = c
        print(formu, colors[formu])
    
    nums_row = len(df_min.index)
    for i in range(nums_row):
        row = df_min.iloc[[i]]
        formu = row['formulas'].values[0]
        # color = eval(colors[formu])
        color = colors[formu]
        # plt.scatter(row['Us'], row['dG_Hx_dps'], c=color, linewidth=1)
        plt.axvspan(row['Us'].values[0], row['Us'].values[0]+.03, facecolor=color, alpha=1)
    
    if SHE: # SHE
        plt.xlabel('$U_{SHE}$')
    else: # RHE
        plt.xlabel('$U_{RHE}$')
    plt.ylabel('$\Delta G$ (eV/)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.xlim(U)
    # plt.axvline(x = -0.5, color = 'w', label = 'axvline - full height')
    plt.show()
    print(df_min['formulas'].unique())


def plot_2d_contour(xls_name_Pourbaix):
    """Plot 2d Pourbaix diagram: dG vs. U and pH"""
    # df_colors = pd_read_excel(filename=xls_name_Pourbaix, sheet=sheet_name_color_list)
    # colors = df_colors.set_index('formulas')['colors'].to_dict()
    df_min = pd_read_excel(filename=xls_name_Pourbaix, sheet=sheet_name_min)

    df=df_min.pivot('Us', 'pHs', 'dG_Hx_dps')
    X=df.columns.values
    print(X)
    Y=df.index.values
    Z=df.values
    x,y=np.meshgrid(X, Y)

    cm = plt.get_cmap(plt.cm.jet)
    # NUM_COLORS=12
    # cs = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    mins = min(df_min['dG_Hx_dps'])
    maxs = max(df_min['dG_Hx_dps'])
    contours = np.linspace(mins, maxs, 12) 
    plt.contourf(x, y, Z, contours, cmap=cm)
    plt.xlabel('pH')
    plt.ylabel('$U_{SHE}$ (V)')
    plt.tight_layout(pad=0.)
    plt.xlim(pH)
    plt.ylim(U)
    plt.show()
    print(df_min['formulas'].unique())

def plot_2d(xls_name_Pourbaix):
    """Plot 2d Pourbaix diagram: dG vs. U and pH"""
    df_colors = pd_read_excel(filename=xls_name_Pourbaix, sheet=sheet_name_color_list)
    colors = df_colors.set_index('formulas')['colors'].to_dict()
    df_min = pd_read_excel(filename=xls_name_Pourbaix, sheet=sheet_name_min)
    
    nums_row = len(df_min.index)
    print(df_min['formulas'].unique())
    for i in range(nums_row):
        row = df_min.iloc[[i]]
        formu = row['formulas'].values[0]
        # color = eval(colors[formu])
        color = colors[formu]
        plt.scatter(row['pHs'], row['Us'], c=color, marker='o', s=15)

    plt.xlabel('pH')
    plt.ylabel('$U_{SHE}$ (V)')
    plt.tight_layout(pad=0.)
    plt.xlim(pH)
    plt.ylim(U)
    plt.show()

def plot_2d_scatter(xls_name_Pourbaix):
    """Plot 2d Pourbaix diagram: dG vs. U and pH"""
    df_min = pd_read_excel(filename=xls_name_Pourbaix, sheet=sheet_name_min)
    
    print(df_min['formulas'].unique())
    pHs = df_min['pHs'].values
    Us = df_min['Us'].values
    colors = df_min['color'].values
    plt.scatter(pHs, Us, c=colors, marker='o', s=2)
    
    df_colors = pd_read_excel(filename=xls_name_Pourbaix, sheet=sheet_name_color_list)
    colors = df_colors.set_index('formulas')['colors'].to_dict()
    formus = df_min['formulas'].unique()
    # for formu in formus:
    #     color = colors[formu]
    #     plt.scatter(0, -1, c=color, label=formu)
    for formu, color in colors.items():
        if formu in formus:
            plt.scatter(-0.1, 0, c=color, marker='s', s=10, label=formu)
    plt.legend(loc='upper right')
    plt.xlabel('pH')
    plt.ylabel('$U_{SHE}$ (V)')
    plt.tight_layout(pad=0.)
    plt.xlim(pH)
    plt.ylim(U)
    plt.show()

def pourbaix_diagram(U, pH, gen=True):
    """Get pourbaix diagram as a function of applied potential and pH value
    
    gen: Bool
        True means generating excel files
        False means no generating excel files and only read it
    """
    if type(pH) == int or type(pH) == float:
        xls_name_Pourbaix_1d = f'./data/{system_name}_write_Pourbaix_diagram_1d.xlsx' # write
        if gen:
            print('\nGenerate 1d data')
            to_xls(xls_name_Pourbaix_1d)
        print('\nPlotting 1d')
        plot_1d(xls_name_Pourbaix_1d)
    else:
        xls_name_Pourbaix_2d = f'./data/{system_name}_write_Pourbaix_diagram_2d.xlsx' # write
        if gen:
            print('\nGenerate 2d data')
            to_xls(xls_name_Pourbaix_2d)
        print('\nPlotting 2d')
        # plot_2d(xls_name_Pourbaix_2d)
        # plot_2d_contour(xls_name_Pourbaix_2d)
        plot_2d_scatter(xls_name_Pourbaix_2d)
        

if __name__ == '__main__':
    T = 297.15
    N, M = 200, 200
    kB = const.kB
    
    # system_name = 'collect_vasp_coverage_H'
    system_name = 'collect_vasp_candidates_PdHx_all_sites_stdout'
    
    # xls_name = f'./data/{system_name}_read_pourbaix.xlsx' # read
    xls_name = f'./data/{system_name}_read_pourbaix_diagram.xlsx' # read
    fig_dir = './figures'
    
    sheet_name_origin = 'Origin'
    # sheet_name_origin = 'Origin2' # has Pd64H65
    sheet_name_dGs = 'dGs'
    sheet_name_color_list = 'color_list'
    sheet_name_all = 'all'
    sheet_name_min = 'min'
    
    U = [-1, 1]
    
    pH = 7.3
    pourbaix_diagram(U, pH, gen=True)
    
    # pH = [0, 14]
    # pourbaix_diagram(U, pH, gen=False)
    
    