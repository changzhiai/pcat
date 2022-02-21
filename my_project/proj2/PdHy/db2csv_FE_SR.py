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
import numpy as np

def concatenate_db(db_name1, db_name2, db_tot):
    """Contatenate two database into the total one"""
    
    if os.path.exists(db_tot):
        assert False
    db_tot = connect(db_tot)
    if os.path.exists(db_name1) and os.path.exists(db_name2):
        db1 = connect(db_name1)
        db2 = connect(db_name2)
        for row in db1.select():
            db_tot.write(row)
        for row in db2.select():
            db_tot.write(row)
        
def concatenate_all_db(db_tot: str, db_names: list ):
    """Contatenate all database into the total one"""
    
    if os.path.exists(db_tot):
        assert False
    db_tot = connect(db_tot)
    for db_name in db_names:
        if os.path.exists(db_name):
            db = connect(db_name)
            for row in db.select():
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
    
        
def view_db(db_name):
    """View all structures in the database"""
    db = connect(db_name)
    atoms_all = []
    for row in db.select():
        atoms_all.append(row.toatoms())
    view(atoms_all)
    
def view_ads(ads, all_sites=False):
    """View one kind of adsorbate`s all structures in the database
    
    ads: str
        'HOCO' or 'CO' or 'H' or 'OH'
        
    Global variables
        
    db_name: str
        database name given atoms
    """
    db = connect(db_name)
    
    if all_sites==True:
        sheet = sheet_name_origin
    else:
        sheet = sheet_name_stable
    df = pd_read_excel(xls_name, sheet)
    df = df.sort_values(by=['Cons_H'])
    df_sub = df.loc[df['Adsorbate'] == ads]
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
                atoms_list.append(r.toatoms())
    view(atoms_list)
    
def get_each_layer_cons(atoms, obj):
    """Get concentration in each layer"""
    if obj == 'Pd':
        obj_lys = [(-1, 1), (2, 3), (4, 5), (6.5, 7.5)]
    elif obj == 'H':
        obj_lys = [(1, 2), (3, 4), (5, 6.5), (7.5, 8.5)]
    obj_cons = np.zeros(4)
    for atom in atoms:
        for i, obj_ly in enumerate(obj_lys):
            min_z = min(obj_ly)
            max_z = max(obj_ly)
            if atom.symbol == obj and atom.z > min_z and atom.z < max_z:
                obj_cons[i] += 1
    obj_cons= obj_cons/16
    # print(obj_cons)
    return obj_cons

def plot_cons_as_layers(obj='H'):
    """Plot concentrations as a function of layers
    
    obj = 'H' or 'Pd'
    """
    db = connect(db_name)
    for row in db.select():
        # atoms_all.append(row.toatoms())
        atoms = row.toatoms()
        obj_cons = get_each_layer_cons(atoms, obj=obj)
        con_tot = sum(obj_cons)/4
        formula = row.formula
        i_X = formula.find('X')
        formula = formula[:i_X] + formula[i_X+4:] # remove Xxxx
        xs = ['layer 1', 'layer 2', 'layer 3', 'layer 4']
        plt.figure()
        plt.bar(xs, obj_cons, color='blue')
        plt.ylim(0, 1.18)
        if obj == 'H':
            plt.title(formula + ', H cons:' + str(round(con_tot, 3)))
            plt.ylabel('Concentration of H')
        elif obj == 'Pd':
            plt.title(formula + ', Pd cons:' + str(round(con_tot, 3)))
            plt.ylabel('Concentration of Pd')

def plot_BE_as_Hcons(xls_name, sheet_cons):
    """Plot binding energy as a function of hydrogen concentrations"""
    df = pd_read_excel(xls_name, sheet_cons)
    df = df.sort_values(by=['Cons_H'])
    cons_H = df['Cons_H'].values
    E_HOCO = df['E(*HOCO)'].values
    E_CO = df['E(*CO)'].values
    E_H = df['E(*H)'].values
    E_OH = df['E(*OH)'].values
    plt.figure()
    plt.plot(cons_H, E_HOCO, c='blue', label='*HOCO')
    plt.plot(cons_H, E_CO, c='orange', label='*CO')
    plt.plot(cons_H, E_H, c='green', label='*H')
    plt.plot(cons_H, E_OH, c='brown', label='*OH')
    plt.xlabel('Concentration of H', fontsize=12)
    plt.ylabel('Binding energy (eV)', fontsize=12)
    plt.legend()
    plt.show()
    # df.plot(kind='scatter',x='Cons_H',y='E(*HOCO)',ax=ax)
    
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
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=True, ncol=5)
    # plt.legend(loc = "lower left", bbox_to_anchor=(0.00, -0.50, 0.8, 1.02), ncol=5, borderaxespad=0)
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
            sr.plot(ax = ax, save=False, 
                    color_dict=True, 
                    title='', 
                    xlabel=descriper1, 
                    ylabel=descriper2, 
                    dot_color='red', 
                    line_color='red',
                    annotate=True,)
            i+=1
    plt.show()
    fig.savefig(name_fig_BE, dpi=300, bbox_inches='tight')

def plot_stability():
    """Plot convex hull by formation energy"""
    ''
    
def plot_chemical_potential(xls_name, sheet_name_origin):
    """Plot chemical potential using DFT energy"""
    df = pd_read_excel(filename=xls_name, sheet=sheet_name_origin)
    df = df.loc[df['Adsorbate'] == 'surface']
    df = df.sort_values(by=['Cons_H'], ascending=False)
    cons_H = df['Cons_H'].to_numpy()
    dft_energy = df['Energy'].to_numpy()
    dyf = [0.0] * len(cons_H)
    for i in range(len(dft_energy)-1):
        dyf[i] = (dft_energy[i+1] - dft_energy[i])/((cons_H[i+1]-cons_H[i])*64)
    # set last element by backwards difference
    dyf[-1] = (dft_energy[-1] - dft_energy[-2])/((cons_H[-1] - cons_H[-2])*64)
    plt.figure()
    plt.plot(cons_H, dyf)
    plt.show()
    
def plot_selectivity(xls_name, sheet_selectivity, fig_dir):
    """Plot selectivity of CO2RR and HER"""
    
    df = pd_read_excel(filename=xls_name, sheet=sheet_selectivity)
    # df.set_axis(['Single'], axis='columns', inplace=True)
    name_fig_select = f'{fig_dir}/{system_name}_{sheet_selectivity}.jpg'
    
    selectivity = Selectivity(df, fig_name=name_fig_select)
    selectivity.plot(save=True, title='', xlabel='Different surfaces', tune_tex_pos=1.5, legend=False)
    
def plot_activity(xls_name, sheet_binding_energy, fig_dir):
    """Plot activity of CO2RR"""
    df = pd_read_excel(filename=xls_name, sheet=sheet_binding_energy)
    # df.drop(['Pd16Ti48H8', 'Pd16Ti48H24'], inplace=True)
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
    activity.plot(save=True)
    # activity.plot(save=True, xlim=[-2.5, 2.5], ylim=[-2.5, 2.5])

def del_partial_db(db):
    """Delet uncomplete database"""
    # db = connect(db_name)
    del_ids = [142, 141, 140, 139, 138]
    del_rows = []
    for row in db.select():
        for del_id in del_ids:
            if row.start_id == del_id:
                del_rows.append(row.id)
    db.delete(del_rows)
    print(del_rows)
    return db

if __name__ == '__main__':
    if False:
        db_tot = '../data/collect_vasp_PdHy_and_insert.db'
        concatenate_db('../data/collect_vasp_PdHy_v3.db', '../data/collect_vasp_insert_PdHy.db', db_tot)
    
    # system_name = 'collect_vasp_test_m'
    # system_name = 'collect_vasp_Pd0Hy'
    # system_name = 'collect_vasp_layers_H'
    # system_name = 'collect_vasp_PdHy_and_insert'
    # system_name = 'collect_ce_candidates_v0'
    # system_name = 'collect_ce_candidates'
    # system_name = 'collect_ce_candidates_single_ads'
    # system_name = 'collect_ce_candidates_single_ads_72s'
    # system_name = 'collect_ce_Pd32Hy'
    # system_name = 'cand_init_Pd64Hy'
    # system_name = 'collect_ce_candidates_predict_vasp'
    
    
    # system_name = 'collect_vasp_PdHy_v3'
    # system_name = 'collect_vasp_Pd32Hy'
    # system_name = 'collect_vasp_Pd13Hy'
    # system_name = 'collect_vasp_Pd48Hy'
    # system_name = 'collect_vasp_Pd16Hy'
    # system_name = 'collect_vasp_Pd51Hy'
    # system_name = 'collect_vasp_PdHy_and_Pd32Hy'
    # system_name = 'collect_vasp_PdHy_and_Pd16Hy_and_Pd32Hy'
    system_name = 'collect_vasp_PdHy_and_Pd16Hy_and_Pd32Hy_and_Pd48Hy'
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
        # db = del_partial_db(db)
        db2xls(system_name, xls_name, db, ref_eles, sheet_name_origin, sheet_name_stable, sheet_free_energy, sheet_binding_energy, sheet_cons, sheet_name_allFE, sheet_selectivity, sheet_name_dGs)
    
    if True: # plot
        plot_free_enegy(xls_name, sheet_free_energy, fig_dir)
        plot_scaling_relations(xls_name, sheet_binding_energy, fig_dir)
        plot_selectivity(xls_name, sheet_selectivity, fig_dir)
        plot_activity(xls_name, sheet_binding_energy, fig_dir)
        
    # plot_free_enegy(xls_name, sheet_free_energy, fig_dir)
    # plot_scaling_relations(xls_name, sheet_binding_energy, fig_dir)
    # plot_activity(xls_name, sheet_binding_energy, fig_dir)
    # views(formula='Pd51Ti13H59', all_sites=True)
    # view(db_name)
    # plot_BE_as_Hcons(xls_name, sheet_cons)
    # plot_pourbaix_diagram(xls_name, sheet_name_dGs)
    # plot_chemical_potential(xls_name, sheet_name_origin)
    # view_ads('CO')
    # view_db(db_name)
    
    # plot_cons_as_layers(obj='Pd')