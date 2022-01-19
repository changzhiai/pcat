# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 13:55:28 2022

@author: changai
"""
# import sys
# sys.path.append("../../../")

from pcat.preprocessing.db2xls import db2xls
from ase.db import connect
from pcat.lib.io import pd_read_excel
from pcat.free_energy import CO2RRFED
from pcat.scaling_relation import ScalingRelation
import os
from pcat.selectivity import Selectivity
from pcat.activity import Activity
from ase.visualize import view
import matplotlib.pyplot as plt
from pcat.pourbaix import PourbaixDiagram

def concatenate_db(db_name1, db_name2, db_tot):
    """Contatenate two database into the total one"""
    
    if os.path.exists(db_tot):
        assert False
    db1 = connect(db_name1)
    db2 = connect(db_name2)
    db_tot = connect(db_tot)
    for row in db1.select():
        db_tot.write(row)
    for row in db2.select():
        db_tot.write(row)

def views(formula, all_sites=False):
    """View specific structures
    
    Parameters
    
    formula: str
        surface name
    all_sites: str
        view all structures of all sites 
    
    Global variables
        
    db_name: str
        database name given atoms
    xls_name: str
        excel used
    sheet_name_origin: str
        sheet of excel used
    sheet_name_stable: str
        sheet of excel used
    
    Input: 
        surface formula  
    Output:
        all structures based on this surface
    """
    if all_sites==True:
        sheet = sheet_name_origin
    else:
        sheet = sheet_name_stable
    df = pd_read_excel(xls_name, sheet)
    df_sub = df.loc[df['Surface'] == formula]
    db = connect(db_name)
    # db_name_temp = 'temp.db'
    # if os.path.exists(db_name_temp):
    #     os.remove(db_name_temp)
    # db_temp = connect(db_name_temp)
    atoms_list = []
    for index, row in df_sub.iterrows():
        origin_id = row['Origin_id']
        site = row['Site']
        adsorbate = row['Adsorbate']
        for r in db.select():
            unique_id = r.uniqueid
            r_origin_id = unique_id.split('_')[4]
            r_site = unique_id.split('_')[2]
            r_adsorbate = unique_id.split('_')[3]
            if str(origin_id)==r_origin_id and site==r_site and adsorbate==r_adsorbate:
                # db_temp.write(r)
                # view(r.toatoms())
                atoms_list.append(r.toatoms())
    view(atoms_list)

def plot_BE_as_Hcons(xls_name, sheet_cons):
    """Plot binding energy as a function of hydrogen concentrations"""
    df = pd_read_excel(xls_name, sheet_cons)
    cons_H = df['Cons_H'].values
    E_HOCO = df['E(*HOCO)'].values
    E_CO = df['E(*CO)'].values
    E_H = df['E(*H)'].values
    E_OH = df['E(*OH)'].values
    plt.figure()
    plt.plot(cons_H, E_HOCO, c='blue', label='*HOCO')
    plt.plot(cons_H, E_CO, c='orange', label='*CO')
    plt.plot(cons_H, E_H, c='green', label='*H')
    plt.plot(cons_H, E_OH, c='brown', label='OH')
    plt.xlabel('Concentration of H', fontsize=12)
    plt.ylabel('Binding energy (eV)', fontsize=12)
    plt.legend()
    plt.show()
    # df.plot(kind='scatter',x='Cons_H',y='E(*HOCO)',ax=ax)
   
    
def view_db(db_name):
    """View all structures in the database"""
    db = connect(db_name)
    atoms_all = []
    for row in db.select():
        atoms_all.append(row.toatoms())
    view(atoms_all)
    
def plot_pourbaix_diagram(xls_name, sheet_name_dGs):
    """Plot pourbaix diagram"""
    df = pd_read_excel(filename=xls_name, sheet=sheet_name_dGs)
    df = df.iloc[10]
    U = [-2, 3]
    pH = 0
    pourbaix = PourbaixDiagram(df)
    pourbaix.plot(U, pH)
    
    pH = [0, 14]
    pourbaix.plot(U, pH)
    

def plot_free_enegy(xls_name, sheet_free_energy, fig_dir):
    """
    Plot free energy
    """
    df = pd_read_excel(xls_name, sheet_free_energy)
    step_names = ['* + CO$_{2}$', 'HOCO*', 'CO*', '* + CO']  #reload step name for CO2RR
    df.set_axis(step_names, axis='columns', inplace=True)
    name_fig_FE = f'{fig_dir}/{system_name}_{sheet_free_energy}.jpg'
    fig = plt.figure(figsize=(8, 6), dpi = 300)
    ax = fig.add_subplot(111)
    CO2RR_FED = CO2RRFED(df, fig_name=name_fig_FE)
    CO2RR_FED.plot(ax=ax, save=False, title='')
    plt.legend(loc = "lower left", bbox_to_anchor=(0.00, -0.50, 0.8, 1.02), ncol=5, borderaxespad=0)
    plt.show()
    fig.savefig(name_fig_FE, dpi=300, bbox_inches='tight')
    

def plot_scaling_relations(xls_name, sheet_binding_energy, fig_dir):
    """
    Plot scaling relation by binding energy
    """

    df = pd_read_excel(xls_name, sheet_binding_energy)
    col1 = [2, 2, 2, 3, 3, 5] # column in excel
    col2 = [3, 5, 4, 5, 4, 4] # column in excel
    
    fig = plt.figure(figsize=(18, 16), dpi = 300)
    name_fig_BE = f'{fig_dir}/{system_name}_{sheet_binding_energy}.jpg'
    M  = 3
    i = 0
    for m1 in range(M-1):
        for m2 in range(M):
            ax = plt.subplot(M, M, m1*M + m2 + 1)
            descriper1 = df.columns[col1[i]-2]
            descriper2 = df.columns[col2[i]-2]
            sr = ScalingRelation(df, descriper1, descriper2, fig_name=name_fig_BE)
            sr.plot(ax = ax, save=False,color_ditc=True, title='', xlabel=descriper1, ylabel=descriper2, dot_color='red', line_color='red')
            i+=1
    plt.show()
    fig.savefig(name_fig_BE, dpi=300, bbox_inches='tight')

def plot_stability():
    """Plot convex hull by formation energy"""
    ''
    
def plot_selectivity(xls_name, sheet_selectivity, fig_dir):
    """Plot selectivity of CO2RR and HER"""
    
    df = pd_read_excel(filename=xls_name, sheet=sheet_selectivity)
    # df.set_axis(['Single'], axis='columns', inplace=True)
    name_fig_select = f'{fig_dir}/{system_name}_{sheet_selectivity}.jpg'
    
    selectivity = Selectivity(df, fig_name=name_fig_select)
    selectivity.plot(save=True, title='',xlabel='Different surfaces', tune_tex_pos=1.5, legend=False)
    
def plot_activity(xls_name, sheet_binding_energy, fig_dir):
    """Plot activity of CO2RR"""
    df = pd_read_excel(filename=xls_name, sheet=sheet_binding_energy)
    df.drop(['Pd16Ti48H8', 'Pd16Ti48H24'], inplace=True)
    name_fig_act = f'{fig_dir}/{system_name}_activity.jpg'
    activity = Activity(df, descriper1 = 'E(*CO)', descriper2 = 'E(*HOCO)', fig_name=name_fig_act,
                        U0=-0.3, 
                        T0=297.15, 
                        pCO2g = 1., 
                        pCOg=0.005562, 
                        pH2Og = 1., 
                        cHp0 = 10.**(-0.),
                        Gact=0.2, 
                        p_factor = 3.6 * 10**4)
    # activity.verify_BE2FE()
    activity.plot(save=True, )

if __name__ == '__main__':
    if False:
        db_tot = '../data/collect_vasp_PdHy_and_Pd16Hy_and_Pd32Hy_and_Pd48Hy_and_Pd51Hy.db'
        concatenate_db('../data/collect_vasp_PdHy_and_Pd16Hy_and_Pd32Hy_and_Pd48Hy.db', '../data/collect_vasp_Pd51Hy.db', db_tot)
    
    
    system_name = 'collect_vasp_PdHy_v3'
    # system_name = 'collect_vasp_Pd32Hy'
    # system_name = 'collect_vasp_Pd48Hy'
    # system_name = 'collect_vasp_Pd16Hy'
    # system_name = 'collect_vasp_Pd51Hy'
    # system_name = 'collect_vasp_PdHy_and_Pd32Hy'
    # system_name = 'collect_vasp_PdHy_and_Pd16Hy_and_Pd32Hy'
    # system_name = 'collect_vasp_PdHy_and_Pd16Hy_and_Pd32Hy_and_Pd48Hy'
    # system_name = 'collect_vasp_PdHy_and_Pd16Hy_and_Pd32Hy_and_Pd48Hy_and_Pd51Hy'
    ref_eles=['Pd', 'Ti']
    db_name = f'../data/{system_name}.db' # the only one needed
    xls_name = f'../data/{system_name}.xlsx'
    fig_dir = '../figures'
    
    sheet_name_origin='Origin'
    sheet_name_stable='Ori_Stable'
    sheet_free_energy = 'CO2RR_FE'
    sheet_binding_energy = 'CO2RR_BE'
    sheet_cons = 'Cons_BE'
    sheet_name_allFE ='All_FE'
    sheet_selectivity = 'Selectivity'
    sheet_name_dGs = 'dGs'
    
    db = connect(db_name)
    if False: # database to excel
        db2xls(system_name, xls_name, db, ref_eles, sheet_name_origin, sheet_name_stable, sheet_free_energy, sheet_binding_energy, sheet_cons, sheet_name_allFE, sheet_selectivity, sheet_name_dGs)
    
    if False: # plot
        plot_free_enegy(xls_name, sheet_free_energy, fig_dir)
        plot_scaling_relations(xls_name, sheet_binding_energy, fig_dir)
        plot_selectivity(xls_name, sheet_selectivity, fig_dir)
        plot_activity(xls_name, sheet_binding_energy, fig_dir)
        
    
    # views(formula='Pd51Ti13H59', all_sites=True)
    # view(db_name)
    # plot_BE_as_Hcons(xls_name, sheet_cons)
    # plot_pourbaix_diagram(xls_name, sheet_name_dGs)