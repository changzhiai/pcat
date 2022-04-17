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
import pandas as pd

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
    
def view_ads(ads, all_sites=False, save=False):
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
    if save==True:
        db_surface_name = '{}_vasp.db'.format(ads)
        if os.path.exists(db_surface_name):
            assert False
        db_surface = connect(db_surface_name)
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
                if save==True:
                    db_surface.write(r)
    view(atoms_list)
    
def get_each_layer_cons(db_name, atoms, obj):
    """Get concentration in each layer"""
    if obj == 'Pd':
        obj_lys = [(-1, 1), (2, 3), (4, 5), (6.5, 7.5)]
    elif obj == 'H':
        # obj_lys = [(1, 2), (3, 4), (5, 6.5), (7.5, 8.5)]
        obj_lys = [(1, 2), (3, 4.6), (4.6, 7), (6.9999, 8.5)]
    obj_cons = np.zeros(4)
    for atom in atoms:
        for i, obj_ly in enumerate(obj_lys):
            min_z = min(obj_ly)
            max_z = max(obj_ly)
            if atom.symbol == obj and atom.z > min_z and atom.z < max_z:
                obj_cons[i] += 1
    # if adsorbate is H for vasp calculation, remove one
    if db_name == './data/H_vasp.db' and obj == 'H':
        obj_cons[3] -= 1
    obj_cons= obj_cons/16
    # print(obj_cons)
    return obj_cons

def plot_cons_as_layers(db='', obj='H', removeX=False):
    """Plot concentrations as a function of layers
    for CE bare surface without adsorbate (initial structures of DFT)
    
    obj = 'H' or 'Pd'
    """
    if db == '':
        db = connect(db_name)
    else:
        db = db
    M = 6
    N = 8
    m1 = 0
    m2 = 0
    fig = plt.figure(figsize=(16,16))
    for row in db.select():
        # atoms_all.append(row.toatoms())
        ax = plt.subplot(N, M, m1*M + m2 + 1)
        atoms = row.toatoms()
        atoms =  atoms[[atom.index for atom in atoms if atom.z < 8.5]] # remove adsorbate
        obj_cons = get_each_layer_cons(db_name, atoms, obj=obj)
        con_tot = sum(obj_cons)/4
        formula = row.formula
        if removeX == True:
            i_X = formula.find('X')
            formula = formula[:i_X] + formula[i_X+4:] # remove Xxxx
        xs = ['ly1', 'ly2', 'ly3', 'ly4']
        # xs = ['layer 1', 'layer 2', 'layer 3', 'layer 4']
        # plt.figure()
        plt.bar(xs, obj_cons, color='blue')
        plt.ylim(0, 1.18)
        if obj == 'H':
            title = formula + ', H:' + str(round(con_tot, 3))
            plt.text(0.05, 0.92, title, fontsize=8, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, color='black', fontweight='bold')
            # plt.title(formula + ', H:' + str(round(con_tot, 3)), fontsize=8)
            if m2==0:
                plt.ylabel('Concentration of H', fontsize=10)
        elif obj == 'Pd':
            plt.title(formula + ', Pd cons:' + str(round(con_tot, 3)))
            plt.ylabel('Concentration of Pd')
        
        m2 += 1
        if m2 == M:
            m2 = 0
            m1 += 1
        # name_fig_cons_as_lys=f'{fig_dir}/{system_name}_cons_as_lys.jpg'
        # fig.savefig(name_fig_cons_as_lys, dpi=300, bbox_inches='tight')
        
def plot_cons_as_layers_with_ads(obj='H'):
    """Plot concentrations as a function of layers
    for optimized DFT suface and surface with intermediates
    
    obj = 'H' or 'Pd'
    """
    db = connect(db_name)
    sheet = sheet_name_stable
    df = pd_read_excel(xls_name, sheet)
    df = df.sort_values(by=['Cons_H'], ascending=False)
    uniqueids = df['Origin_id'].astype(int).unique()
    M = 5
    # N = 30
    N = len(uniqueids)
    m1 = 0
    m2 = 0
    fig = plt.figure(figsize=(16,16*4))
    for id in uniqueids:
        df_sub = df.loc[df['Origin_id'].astype(int) == id]
        df_hoco = df_sub[df_sub['Adsorbate']=='HOCO']
        df_surf = {'Id': df_hoco.Id.values[0],
                   'Origin_id': df_hoco.Origin_id.values[0],
                   'Surface': df_hoco.Surface.values[0],
                   'Cons_Pd': df_hoco.Cons_Pd.values[0],
                   'Cons_H': df_hoco.Cons_H.values[0],
                   'Site': 'top1',
                   'Adsorbate': 'surface',
                   }
        df_sub = df_sub.append(df_surf, ignore_index = True)
        custom_dict = {'surface':0, 'HOCO': 1, 'CO': 2, 'H': 3, 'OH':4} 
        df_sub = df_sub.sort_values(by=['Adsorbate'], key=lambda x: x.map(custom_dict))
        
        atoms_list = []
        for i,row in df_sub.iterrows():
            origin_id = row['Origin_id']
            site = row['Site']
            ads = row['Adsorbate']
            
            for r in db.select():
                unique_id = r.uniqueid
                r_origin_id = unique_id.split('_')[4]
                r_site = unique_id.split('_')[2]
                r_adsorbate = unique_id.split('_')[3]
                if str(origin_id)==r_origin_id and site==r_site and ads==r_adsorbate:
                    ax = plt.subplot(N, M, m1*M + m2 + 1)
                    atoms = r.toatoms()
                    if ads == 'surface':
                        atoms = atoms
                        title = 'Bare slab' 
                    elif ads == 'HOCO':
                        atoms = atoms[0:-4]
                        title = 'HOCO*'
                    elif ads == 'CO':
                        atoms = atoms[0:-2]
                        title = 'CO*'
                    elif ads == 'H':
                        atoms = atoms[0:-1]
                        title = 'H*'
                    elif ads == 'OH':
                        atoms = atoms[0:-2]
                        title = 'OH*'
                    
                    # atoms_list.append(r.toatoms())
                    obj_cons = get_each_layer_cons(db_name, atoms, obj=obj)
                    con_tot = sum(obj_cons)/4
                    formula = f'{row.Surface}_{ads}'
                    xs = ['ly1', 'ly2', 'ly3', 'ly4']
                    # xs = ['layer 1', 'layer 2', 'layer 3', 'layer 4']
                    # plt.figure()
                    plt.bar(xs, obj_cons, color='blue')
                    plt.ylim(0, 1.18)
                    if obj == 'H':
                        title = formula + ', H:' + str(round(con_tot, 3))
                        plt.text(0.05, 0.92, title, fontsize=8, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, color='black', fontweight='bold')
                        # plt.title(formula + ', H:' + str(round(con_tot, 3)), fontsize=8)
                        if m2==0:
                            plt.ylabel('Concentration of H', fontsize=10)
                    elif obj == 'Pd':
                        plt.title(formula + ', Pd cons:' + str(round(con_tot, 3)))
                        plt.ylabel('Concentration of Pd')
                    
                    m2 += 1
                    if m2 == M:
                        m2 = 0
                        m1 += 1
    plt.show()
    name_fig_cons_as_lys=f'{fig_dir}/{system_name}_cons_as_lys.jpg'
    fig.savefig(name_fig_cons_as_lys, dpi=300, bbox_inches='tight')
    # view(atoms_list)

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
    # df.drop(['Pd16Ti48H8', 'Pd16Ti48H24'], inplace=True)
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
    # df.drop(['Pd16Ti48H8', 'Pd16Ti48H24'], inplace=True)
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
    plt.figure()
    plt.plot(cons_H, dft_energy)
    plt.xlabel('Concentration of H')
    plt.ylabel('DFT energy (eV)')
    plt.show()
    dyf = [0.0] * len(cons_H)
    for i in range(len(dft_energy)-1):
        dyf[i] = (dft_energy[i+1] - dft_energy[i])/((cons_H[i+1]-cons_H[i])*64)
    # set last element by backwards difference
    dyf[-1] = (dft_energy[-1] - dft_energy[-2])/((cons_H[-1] - cons_H[-2])*64)
    plt.figure()
    ids = []
    for i, y in enumerate(dyf):
        if y < -5:
            ids.append(i)
    cons_H = np.delete(cons_H, ids)
    dyf = np.delete(dyf, ids)
    plt.plot(cons_H, dyf)
    plt.xlabel('Concentration of H')
    plt.ylabel('$\Delta \mu$ (eV)')
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
    # activity.plot(save=True, xlim=[-1., 0], ylim=[-0.2, 1])

def del_partial_db(db):
    """Delet uncomplete database"""
    # db = connect(db_name)
    del_ids = [142, 141, 140, 139, 138]
    # del_ids = np.arange(311)
    del_rows = []
    for row in db.select():
        for del_id in del_ids:
            if row.start_id == del_id:
                del_rows.append(row.id)
    db.delete(del_rows)
    print(del_rows)
    return db

def sort_db(db):
    """Sort database according to H concentration"""
    db_sort = connect('./data/{}_sort.db'.format(system_name))
    ids = []
    num_Hs = []
    for row in db.select():
        atoms = row.toatoms()
        try:
            num_H = len(atoms[[atom.index for atom in atoms if atom.symbol=='H']])
        except:
            num_H = 0
        ids.append(row.id)
        num_Hs.append(num_H)
    tuples = {'Id': ids,
          'Numbers_H': num_Hs,
          }
    df = pd.DataFrame(tuples)
    df = df.sort_values(by=['Numbers_H'])
    for index, row in df.iterrows():
        id = row['Id']
        for r in db.select(id=int(id)):
            print(id)
            db_sort.write(r)
    print('sort success')

def get_ads_db(ads, save=True):
    """Get database of slab with one specific adsorbate from slab with all adsorbates
    
    get bare slab, or HOCO, CO, OH, H
    """
    db = connect(db_name)
    
    if ads == 'surface':
        sheet = sheet_name_origin
    else:
        sheet = sheet_name_stable
    df = pd_read_excel(xls_name, sheet)
    df = df.sort_values(by=['Cons_H'])
    df_sub = df.loc[df['Adsorbate'] == ads]
    atoms_list = []
    if save==True:
        db_surface_name = '{}_vasp_temporary.db'.format(ads)
        if os.path.exists(db_surface_name):
            os.remove(db_surface_name)
        db_surface = connect(db_surface_name)
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
                if save==True:
                    db_surface.write(r)
    if save==True:
        return db_surface, atoms_list
    else:
        return atoms_list
    
def plot_ce_H_distribution():
    """Plot all H distributions"""
    db = connect('./data/candidates_PdHx_sort.db')
    plot_cons_as_layers(db=db, obj='H', removeX=True) # plot slab for CE
    db = connect('./data/surface_vasp.db')
    plot_cons_as_layers(db=db, obj='H', removeX=False) # plot slab for vasp
    
    for system in ['HOCO', 'CO', 'OH', 'H']:
        db_name=f'./data/{system}_vasp.db'
        if os.path.exists(db_name):
            plot_cons_as_layers(db=connect(db_name), obj='H', removeX=False)
            plt.show()

def plot_H_distribution(save=False):
    """Plot all H distributions"""
    # db_surf, _ = get_ads_db(ads='surface')
    # plot_cons_as_layers(db=db_surf, obj='H', removeX=True) # plot slab for CE
    
    for system in ['surface', 'HOCO', 'CO', 'OH', 'H']: # plot for vasp
        db_ads, _ = get_ads_db(ads=system)
        fig = plt.figure()
        plot_cons_as_layers(db=db_ads, obj='H', removeX=False)
        # fig.add_subplot(111, frameon=False) # hide frame
        # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False) # hide 
        st = plt.suptitle(str(system))
        st.set_y(0.90)
        # fig.subplots_adjust(top=0.85)
        plt.show()
        if save==True:
            fig.savefig(fig_dir+'/{}.png'.format(system))
    
    

if __name__ == '__main__':
    if False:
        db_tot = '../data/collect_vasp_PdHy_and_insert.db'
        concatenate_db('../data/collect_vasp_PdHy_v3.db', '../data/collect_vasp_insert_PdHy.db', db_tot)
    
    # system_name = 'collect_ce_init_PdHx_r2_sort'
    # system_name = 'collect_ce_init_PdHx_r3'
    # system_name = 'collect_ce_init_PdHx_r4'
    # system_name = 'collect_ce_init_PdHx_r7'
    
    # system_name = 'collect_vasp_candidates_PdHx_r2_sort'
    # system_name = 'collect_vasp_candidates_PdHx_r3'
    # system_name = 'collect_vasp_candidates_PdHx_r4'
    system_name = 'collect_vasp_candidates_PdHx_r7'
    
    
    
    # system_name = 'candidates_PdHx_sort' # candidates surface of CE
    # system_name = 'surface_vasp' # vasp 
    
    # system_name = 'HOCO_vasp' # vasp HOCO
    # system_name = 'CO_vasp' # vasp CO
    # system_name = 'collect_ce_init_PdHx_r2'
    
    # system_name = 'collect_vasp_candidates_PdHx' # 9 times


    ref_eles=['Pd', 'Ti']
    db_name = f'./data/{system_name}.db' # the only one needed
    xls_name = f'./data/{system_name}.xlsx'
    fig_dir = './figures'
    
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
    
    if True:
        plot_BE_as_Hcons(xls_name, sheet_cons)
        plot_cons_as_layers_with_ads(obj='H')
        plot_H_distribution(save=False)
        
    # plot_free_enegy(xls_name, sheet_free_energy, fig_dir)
    # plot_scaling_relations(xls_name, sheet_binding_energy, fig_dir)
    # plot_activity(xls_name, sheet_binding_energy, fig_dir)
    # views(formula='Pd51Ti13H59', all_sites=True)
    # view(db_name)
    # plot_BE_as_Hcons(xls_name, sheet_cons)
    # plot_pourbaix_diagram(xls_name, sheet_name_dGs)
    # plot_chemical_potential(xls_name, sheet_name_origin)
    
    # view_ads('H', all_sites=False, save=True)
    # view_db(db_name)
    # view_ads('surface', all_sites=True)
    
    # plot_cons_as_layers(obj='Pd')
    
    
    # plot_cons_as_layers(obj='H')
    # sort_db(db)
    # plot_cons_as_layers(db_name=db_name, obj='H', removeX=False)
    # plot_cons_as_layers_with_ads(obj='H')
    # sort_db(db)
    # print(len(db_name))
    # plot_H_distribution()