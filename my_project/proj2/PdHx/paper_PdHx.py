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
import seaborn as sns
# from matplotlib.patches import Rectangle

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
                
def remove_specific_rows(db_name, condition):
    """Remove some rows according to specific condition"""
    db = connect(db_name)
    del_rows = []
    for row in db.select(condition):
        del_rows.append(row)
    db.delete(del_rows)
    return db

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
            if str(origin_id)==r_origin_id and str(site)==r_site and adsorbate==r_adsorbate:
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
            if str(origin_id)==r_origin_id and str(site)==r_site and adsorbate==r_adsorbate:
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
    # if db_name == './data/H_vasp.db' and obj == 'H':
    #     obj_cons[3] -= 1
    obj_cons= obj_cons/16
    # print(obj_cons)
    return obj_cons

def plot_cons_as_layers(db='', obj='H', removeX=False, minusH=True):
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
    # fig = plt.figure(figsize=(16,16))
    for row in db.select():
        # atoms_all.append(row.toatoms())
        ax = plt.subplot(N, M, m1*M + m2 + 1)
        atoms = row.toatoms()
        atoms =  atoms[[atom.index for atom in atoms if atom.z < 8.5]] # remove adsorbate
        obj_cons = get_each_layer_cons(db_name, atoms, obj=obj)
        if minusH == True:
            obj_cons[3] -= 1/16.
        con_tot = sum(obj_cons)/4
        formula = row.formula
        if removeX == True:
            i_X = formula.find('X')
            formula = formula[:i_X] + formula[i_X+4:] # remove Xxxx
        xs = ['ly1', 'ly2', 'ly3', 'ly4']
        # xs = ['layer 1', 'layer 2', 'layer 3', 'layer 4']
        plt.bar(xs, obj_cons, color='blue')
        plt.ylim(0, 1.18)
        if obj == 'H':
            title = formula + ', H:' + str(round(con_tot, 3))
            plt.text(0.05, 0.92, title, fontsize=8, horizontalalignment='left', 
                     verticalalignment='center', transform=ax.transAxes, color='black', fontweight='bold')
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
        
def plot_layers_as_strutures(db='', obj='H', removeX=False, minusH=False):
    """Plot concentrations as a function of layers
    for CE bare surface without adsorbate (initial structures of DFT)
    
    obj = 'H' or 'Pd'
    """
    if db == '':
        db = connect(db_name)
    else:
        db = db
    # fig = plt.figure(figsize=(16,16))
    fig = plt.figure(dpi=300)
    ly1s = []
    ly2s = []
    ly3s = []
    ly4s = []
    con_tots = []
    for row in db.select():
        # atoms_all.append(row.toatoms())
        atoms = row.toatoms()
        atoms =  atoms[[atom.index for atom in atoms if atom.z < 8.5]] # remove adsorbate
        obj_cons = get_each_layer_cons(db_name, atoms, obj=obj)
        if minusH == True:
            obj_cons[3] -= 1/16.
        con_tot = sum(obj_cons)/4
        formula = row.formula
        if removeX == True:
            i_X = formula.find('X')
            formula = formula[:i_X] + formula[i_X+4:] # remove Xxxx
        # xs = ['ly1', 'ly2', 'ly3', 'ly4']
        # xs = ['layer 1', 'layer 2', 'layer 3', 'layer 4']
        ly1s.append(obj_cons[0]/4.)
        ly2s.append(obj_cons[1]/4.)
        ly3s.append(obj_cons[2]/4.)
        ly4s.append(obj_cons[3]/4.)
        con_tots.append(con_tot)
    
    tuples = {'ly1s': ly1s,
              'ly2s': ly2s,
              'ly3s': ly3s,
              'ly4s': ly4s,
              'con_tots': con_tots,
             }
    df = pd.DataFrame(tuples)
    df = df.sort_values(by=['con_tots'])
    # print(df)
    colors = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'); 
    if False:
        with pd.ExcelWriter(xls_name, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name='layers_as_structures', index=False, float_format='%.8f')
    a,  = plt.plot(df['con_tots'], df['ly4s'], '-o', label='1st layer', color=colors[0]) # top layer
    b,  = plt.plot(df['con_tots'], df['ly3s'], '-o', label='2nd layer', color=colors[1])
    c,  = plt.plot(df['con_tots'], df['ly2s'], '-o', label='3rd layer', color=colors[2])
    d,  = plt.plot(df['con_tots'], df['ly1s'], '-o', label='4th layer', color=colors[3]) # bottom layer
    # a,  = plt.plot(df['con_tots'], df['ly4s'], '-o', label='1st layer (top)') # top layer
    # b,  = plt.plot(df['con_tots'], df['ly3s'], '-o', label='2nd layer')
    # c,  = plt.plot(df['con_tots'], df['ly2s'], '-o', label='3rd layer')
    # d,  = plt.plot(df['con_tots'], df['ly1s'], '-o', label='4th layer') # bottom layer
    plt.xlabel('Concentration of H', fontsize=10)
    plt.ylabel('H concentration of each layer', fontsize=10)
    plt.legend()
    print(a.get_color(), b.get_color(), c.get_color(), d.get_color())
    # plt.show()
    name_fig_cons_as_lys=f'{fig_dir}/{system_name}_cons_as_lys.jpg'
    if False:
        fig.savefig(name_fig_cons_as_lys, dpi=300, bbox_inches='tight')
        
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
        
        # atoms_list = []
        for i,row in df_sub.iterrows():
            origin_id = row['Origin_id']
            site = row['Site']
            ads = row['Adsorbate']
            
            for r in db.select():
                unique_id = r.uniqueid
                r_origin_id = unique_id.split('_')[4]
                r_site = unique_id.split('_')[2]
                r_adsorbate = unique_id.split('_')[3]
                if str(origin_id)==r_origin_id and str(site)==r_site and ads==r_adsorbate:
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
                        plt.text(0.05, 0.92, title, fontsize=8, horizontalalignment='left', 
                                 verticalalignment='center', transform=ax.transAxes, color='black', fontweight='bold')
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
    plt.figure(dpi=300)
    plt.plot(cons_H, E_HOCO, '-o', c='blue', label='*HOCO')
    plt.plot(cons_H, E_CO, '-o', c='orange', label='*CO')
    plt.plot(cons_H, E_H, '-o', c='green', label='*H')
    plt.plot(cons_H, E_OH, '-o', c='brown', label='*OH')
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
    

def plot_scaling_relations_old(xls_name, sheet_binding_energy, fig_dir):
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
            offsets={}
            if i==0: # first subplot
                offsets={'Pd64H13': [-0.15, -0.25], 'Pd64':[-0.04, -0.15], 
                         'Pd64H4':[0.02, 0.05]}
            elif i==1:
                offsets={'Pd64H4':[0.0, 0.08], 'Pd64H62':[-0.1, 0.15], 
                         'Pd64H53':[-0.2, 0.05]}
            elif i==2:
                offsets={'Pd64':[-0.04, -0.15], 'Pd64H2':[-0.05, 0.1], 
                         'Pd64H4':[-0.1, 0.2], 'Pd64H10':[-0.1, 0.], 
                         'Pd64H39':[-0.1, 0.10]}
            elif i==3:
                offsets={'Pd64':[-0.07, -0.2], 'Pd64H2':[0.05, 0.10], 
                         'Pd64H4':[-0.15, 0.22], 'Pd64H8':[0.1, 0.05], 
                         'Pd64H10':[-0.1, 0.1], 'Pd64H39':[-0.1, 0.10],
                         'Pd64H64':[-0.1, 0.08]}
            elif i==4:
                offsets={'Pd64':[-0.08, -0.15], 'Pd64H2':[-0.14, 0.1], 
                         'Pd64H4':[-0.1, 0.2], 'Pd64H8':[0.0, -0.15], 
                         'Pd64H10':[-0.02, -0.1], 'Pd64H31':[-0.2, -0.1], }
            elif i==5:
                offsets={'Pd64':[-0.06, -0.12], 'Pd64H2':[-0.05, 0.1], 
                         'Pd64H4':[-0.1, 0.2], 'Pd64H8':[0., -0.1], 
                         'Pd64H10':[-0.13, 0.1], 'Pd64H31':[-0.12, -0.15],
                         'Pd64H62':[-0.12, -0.15],}
            sr = ScalingRelation(df, descriper1, descriper2, fig_name=name_fig_BE)
            sr.plot(ax = ax, save=False, 
                    color_dict=True, 
                    title='', 
                    xlabel=descriper1, 
                    ylabel=descriper2, 
                    dot_color='red', 
                    line_color='red',
                    offsets=offsets,
                    annotate=True,)
            i+=1
    plt.show()
    fig.savefig(name_fig_BE, dpi=300, bbox_inches='tight')
    
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
    for m1 in range(M):
        for m2 in range(M-1):
            ax = plt.subplot(M, M, m1*M + m2 + 1)
            descriper1 = df.columns[col1[i]-2]
            descriper2 = df.columns[col2[i]-2]
            offsets={}
            if i==0: # first subplot
                offsets={'Pd64H13': [-0.15, -0.25], 'Pd64':[-0.04, -0.15], 
                         'Pd64H4':[0.02, 0.05]}
            elif i==1:
                offsets={'Pd64H4':[0.0, 0.08], 'Pd64H62':[-0.1, 0.15], 
                         'Pd64H53':[-0.2, 0.05]}
            elif i==2:
                offsets={'Pd64':[-0.04, -0.15], 'Pd64H2':[-0.05, 0.1], 
                         'Pd64H4':[-0.1, 0.2], 'Pd64H10':[-0.1, 0.], 
                         'Pd64H39':[-0.1, 0.10]}
            elif i==3:
                offsets={'Pd64':[-0.07, -0.2], 'Pd64H2':[0.05, 0.10], 
                         'Pd64H4':[-0.15, 0.22], 'Pd64H8':[0.1, 0.05], 
                         'Pd64H10':[-0.1, 0.1], 'Pd64H39':[-0.1, 0.10],
                         'Pd64H64':[-0.1, 0.08]}
            elif i==4:
                offsets={'Pd64':[-0.08, -0.15], 'Pd64H2':[-0.14, 0.1], 
                         'Pd64H4':[-0.1, 0.2], 'Pd64H8':[0.0, -0.15], 
                         'Pd64H10':[-0.02, -0.1], 'Pd64H31':[-0.2, -0.1], }
            elif i==5:
                offsets={'Pd64':[-0.06, -0.12], 'Pd64H2':[-0.05, 0.1], 
                         'Pd64H4':[-0.1, 0.2], 'Pd64H8':[0., -0.1], 
                         'Pd64H10':[-0.13, 0.1], 'Pd64H31':[-0.12, -0.15],
                         'Pd64H62':[-0.12, -0.15],}
            sr = ScalingRelation(df, descriper1, descriper2, fig_name=name_fig_BE)
            sr.plot(ax = ax, save=False, 
                    color_dict=True, 
                    title='', 
                    xlabel=descriper1, 
                    ylabel=descriper2, 
                    dot_color='red', 
                    line_color='blue',
                    offsets=offsets,
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
                        U0=-0.5, 
                        T0=297.15, 
                        pCO2g = 1., 
                        pCOg=0.005562, 
                        pH2Og = 1., 
                        cHp0 = 10.**(-0.),
                        Gact=0.2, 
                        p_factor = 3.6 * 10**4,
                        )
    # activity.verify_BE2FE()
    tune_tex_pos={'Pd64H13':[-0.05, -0.1], 'Pd64':[-0.05, -0.02]}
    ColorDict = {'Pd64H64': 'red', 'Pd64H39': 'red', 'Pd64H63': 'red',}
    # activity.plot(save=True, tune_tex_pos=tune_tex_pos, scaling=True, ColorDict=ColorDict)
    activity.plot(save=True, text=False, tune_tex_pos=tune_tex_pos, scaling=True, ColorDict=ColorDict, **{'fontsize':14})
    # activity.plot(save=True, xlim=[-2.5, 2.5], ylim=[-2.5, 2.5])
    # activity.plot(save=True, xlim=[-1., 0], ylim=[-0.2, 1])
    # activity.plot(save=False, xlim=[-2.5, 2.5], ylim=[-2.5, 2.5])

def del_partial_db(db):
    """Delet uncomplete database"""
    # db = connect(db_name)
    # del_ids = [142, 141, 140, 139, 138]
    del_ids = [16]
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
            if str(origin_id)==r_origin_id and str(site)==r_site and adsorbate==r_adsorbate:
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

def plot_bar_H_distribution(save=False):
    """Plot all H distributions in bar chart"""
    # db_surf, _ = get_ads_db(ads='surface')
    # plot_cons_as_layers(db=db_surf, obj='H', removeX=True) # plot slab for CE

    for system in ['surface', 'HOCO', 'CO', 'OH', 'H']: # plot for vasp
        db_ads, _ = get_ads_db(ads=system)
        fig = plt.figure()
        if system != 'H':
            plot_cons_as_layers(db=db_ads, obj='H', removeX=False)
        else:
            plot_cons_as_layers(db=db_ads, obj='H', removeX=False, minusH=True)
        # fig.add_subplot(111, frameon=False) # hide frame
        # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False) # hide 
        st = plt.suptitle(str(system))
        st.set_y(0.90)
        # fig.subplots_adjust(top=0.85)
        plt.show()
        if save==True:
            fig.savefig(fig_dir+'/{}.png'.format(system))

def plot_line_H_distribution(save=False):
    """Plot all H distributions in line plot"""
    for system in ['surface', 'HOCO', 'CO', 'OH', 'H']: # plot for vasp
        db_ads, _ = get_ads_db(ads=system)
        fig = plt.figure(dpi=300)
        if system != 'H':
            plot_layers_as_strutures(db=db_ads, obj='H', removeX=False)
        else:
            plot_layers_as_strutures(db=db_ads, obj='H', removeX=False, minusH=True)
        # st = plt.suptitle(str(system))
        # st.set_y(0.92)
        # fig.subplots_adjust(top=0.85)
        plt.show()
        if save==True:
            fig.savefig(fig_dir+'/{}.png'.format(system))

def binding_energy_distribution(ads='CO'):
    """Analysis binding energy distribution"""
    df = pd_read_excel(xls_name, sheet_name_origin)
    df_sub = df.loc[df['Adsorbate'] == ads]
    # df_sub = df_sub.set_index('Origin_id')
    BE = df_sub['BE']
    # print(BE)
    sns.set_context("paper"); fontsize = 14
    sns.displot(data=BE, kde=True, facet_kws=dict(despine=False))   
    plt.xlabel('$\Delta E({})$'.format(str('*'+ads)), fontsize=fontsize)
    plt.ylabel('Frequency', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(f'{len(df_sub.index)} datapoints', fontsize=fontsize)
    plt.gcf().set_size_inches(8, 6)
    plt.show()

def plot_count_nn(ads='CO'):
    """Count how many different types of atoms neibour near adsorbate"""
    df = pd_read_excel(xls_name, sheet_name_origin)
    df_sub = df.loc[df['Adsorbate'] == ads]
    num_Pd_nn = df_sub['num_Pd_nn']
    # num_Ti_nn = df_sub[f'num_{ref_eles[1]}_nn']
    num_H_nn = df_sub['num_H_nn']
    BE = df_sub['BE']
    colors = ('green', 'blue', 'orange', 'red', 'magenta'); fontsize = 14
    alpha = 0.5; width = 0.05
    plt.figure()
    plt.bar(BE, num_Pd_nn, alpha=alpha, color=colors[0], width=width, label='Pd')
    # plt.bar(BE, num_Ti_nn, alpha=alpha, color=colors[1], width=width, label='Ti')
    plt.bar(BE, num_H_nn, alpha=alpha, color=colors[2], width=width, label='H')
    plt.xlabel('$\Delta E({})$'.format(str('*'+ads)), fontsize=fontsize)
    plt.ylabel('Counts', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.title(f'{len(df_sub.index)} datapoints', fontsize=fontsize)
    # plt.gcf().set_size_inches(8, 6)
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    if ads == 'CO':
        bestx1 = -0.8
        bestx2 = 0.
        x = np.arange(bestx1, bestx2, 0.01)
        # ax.add_patch(Rectangle((bestx1, ymax-0.2), bestx2, ymax, facecolor = 'red', fill=True, lw=0))
        plt.fill_between(x, ymin, ymax, 
                color='black', alpha=0.2, transform=ax.get_xaxis_transform(), zorder=-1)
    elif ads == 'HOCO':
        bestx1 = -0.8
        bestx2 = 0.4
        x = np.arange(bestx1, bestx2, 0.01)
        # ax.add_patch(Rectangle((bestx1, ymax-0.2), bestx2, ymax, facecolor = 'red', fill=True, lw=0))
        plt.fill_between(x, ymin, ymax, 
                color='black', alpha=0.2, transform=ax.get_xaxis_transform(), zorder=-1)
    plt.show()
    
def plot_count_nn_stack(ads='CO'):
    """Count how many different types of atoms neibour near adsorbate"""
    df = pd_read_excel(xls_name, sheet_name_origin)
    df_sub = df.loc[df['Adsorbate'] == ads]
    num_Pd_nn = df_sub['num_Pd_nn']
    num_Ti_nn = df_sub[f'num_{ref_eles[1]}_nn']
    num_H_nn = df_sub['num_H_nn']
    BE = df_sub['BE']
    colors = ('green', 'blue', 'orange', 'red', 'magenta'); fontsize = 14
    alpha = 1; width = 0.01
    plt.bar(BE, num_Pd_nn, alpha=alpha, color=colors[0], width=width, label='Pd')
    # plt.bar(BE, num_Ti_nn, bottom=num_Pd_nn, alpha=alpha, color=colors[1], width=width, label='Ti')
    plt.bar(BE, num_H_nn, bottom=num_Pd_nn+num_Ti_nn, alpha=alpha, color=colors[2], width=width, label='H')
    plt.xlabel('$\Delta E({})$'.format(str('*'+ads)), fontsize=fontsize)
    plt.ylabel('Counts', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.title(f'{len(df_sub.index)} datapoints', fontsize=fontsize)
    plt.gcf().set_size_inches(8, 6)
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    if ads == 'CO':
        bestx1 = -0.8
        bestx2 = 0.
        x = np.arange(bestx1, bestx2, 0.01)
        plt.fill_between(x, ymin, ymax, 
                color='black', alpha=0.2, transform=ax.get_xaxis_transform(), zorder=-1)
    elif ads == 'HOCO':
        bestx1 = -0.8
        bestx2 = 0.4
        x = np.arange(bestx1, bestx2, 0.01)
        plt.fill_between(x, ymin, ymax, 
                color='black', alpha=0.2, transform=ax.get_xaxis_transform(), zorder=-1)
    plt.show()

def plot_count_nn_hist(ads='CO'):
    """Plot histogram of statistics of atoms"""
    df = pd_read_excel(xls_name, sheet_name_origin)
    df_sub = df.loc[df['Adsorbate'] == ads]
    hist_Pd_nn = []
    hist_Ti_nn = []
    hist_H_nn = []
    plt.figure()
    fig, ax = plt.subplots()
    for i, row in df_sub.iterrows():
        num_Pd_nn = row['num_Pd_nn']
        num_Ti_nn = row[f'num_{ref_eles[1]}_nn']
        num_H_nn = row['num_H_nn']
        BE = row['BE']
        if int(num_Pd_nn) != 0:
            for _ in range(int(num_Pd_nn)):
                hist_Pd_nn.append(BE)
        if int(num_Ti_nn) != 0:
            for _ in range(int(num_Ti_nn)):
                hist_Ti_nn.append(BE)
        if int(num_H_nn) != 0:
            for _ in range(int(num_H_nn)):
                hist_H_nn.append(BE)
    if ads == 'HOCO':
        start, stop, spacing = -1.5, 1.5, 0.075
    elif ads == 'CO':
        start, stop, spacing = -2.5, 0.5, 0.075
    elif ads == 'H':
        start, stop, spacing = -1.0, 1.0, 0.075
    elif ads == 'OH':
        start, stop, spacing = -0.25, 2.75, 0.075
    # start, stop, spacing = -2, 2.5, 0.075
    bins = np.arange(start, stop, spacing)
    colors = ('green', 'blue', 'orange', 'red', 'magenta'); fontsize = 16
    zorders=[1, 2, 3]
    plt.hist(hist_Pd_nn, bins, facecolor=colors[0], ec='black', alpha=0.75, histtype='stepfilled', zorder=zorders[0], label='Pd')
    # plt.hist(hist_Ti_nn, bins, facecolor=colors[1], ec='black', alpha=0.75, histtype='stepfilled', zorder=zorders[1], label='Ti')
    plt.hist(hist_H_nn, bins, facecolor=colors[2], ec='black', alpha=0.75, histtype='stepfilled', zorder=zorders[2], label='H')
    print(hist_H_nn)
    from scipy.stats import norm
    if ads == 'CO':
        data = [x for x in hist_H_nn if x>-1]
    # elif ads == 'H':
    #     data = [x for x in hist_H_nn if x>0]
    else:
        data=hist_H_nn
    mu, std = norm.fit(data) 
    xmin, xmax = plt.xlim()
    # xmin, xmax = min(data), max(data)
    # bin_width = (xmax - xmin) / len(bins)
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    if ads == 'HOCO':
        N=7
    elif ads == 'CO':
        N=4
    elif ads == 'H':
        N=7
    elif ads == 'OH':
        N=8
    plt.plot(x, p*N, colors[2], linewidth=2)
    print("Fit Values: {:.2f} and {:.2f}".format(mu, std))
    
    plt.hist(hist_Pd_nn + hist_Ti_nn + hist_H_nn, bins, facecolor='grey', ec='black', alpha=0.75, histtype='stepfilled', zorder=0, label='total')
    plt.xlim([start, stop])
    plt.xlabel('$\Delta E({})$'.format(str('*'+ads)), fontsize=fontsize)
    plt.ylabel('Frequency', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # plt.title(f'{len(df_sub.index)} datapoints', fontsize=fontsize)
    plt.legend()
    plt.gcf().set_size_inches(8, 6)
    ymin, ymax = ax.get_ylim()
    if ads == 'CO':
        bestx1 = -0.8
        bestx2 = 0.
        x = np.arange(bestx1, bestx2, 0.01)
        plt.fill_between(x, ymin, ymax, 
                color='blue', alpha=0.2, transform=ax.get_xaxis_transform(), zorder=-1)
    elif ads == 'HOCO':
        bestx1 = -0.8
        bestx2 = 0.4
        x = np.arange(bestx1, bestx2, 0.01)
        plt.fill_between(x, ymin, ymax, 
                color='blue', alpha=0.2, transform=ax.get_xaxis_transform(), zorder=-1)
    elif ads == 'H':
        plt.ylim([0, 25.])
    plt.show()

def write_paper_db(db):
    db_new_name = 'PdHx_all_sites.db'
    if os.path.exists(db_new_name):
        os.remove(db_new_name)
    db_new = connect(db_new_name)
    for row in db.select():
        atoms = row.toatoms()
        converged = row.converged
        uniqueid = (row.uniqueid).split('_')
        slab = uniqueid[1].replace('X', '')
        sites = uniqueid[2]
        ads = uniqueid[3]
        if ads == 'surface':
            db_new.write(atoms, converged=converged, slab=slab)
        else:
            db_new.write(atoms, converged=converged, slab=slab, ads_site=int(sites), adsorbate=ads)
        

if __name__ == '__main__':
    # if False:
    #     db_tot = '../data/collect_vasp_PdHy_and_insert.db'
    #     concatenate_db('../data/collect_vasp_PdHy_v3.db', '../data/collect_vasp_insert_PdHy.db', db_tot)
        
    # system_name = 'collect_ce_init_PdHx_r2_sort'
    # system_name = 'collect_ce_init_PdHx_r3'
    # system_name = 'collect_ce_init_PdHx_r4'
    # system_name = 'collect_ce_init_PdHx_r7'
    
    # system_name = 'collect_vasp_candidates_PdHx'
    # system_name = 'collect_vasp_candidates_PdHx_r2_sort'
    # system_name = 'collect_vasp_candidates_PdHx_r3'
    # system_name = 'collect_vasp_candidates_PdHx_r4'
    # system_name = 'collect_vasp_candidates_PdHx_r7'
    # system_name = 'collect_vasp_candidates_PdHx_r8'
    # system_name = 'collect_vasp_extra_H'
    # system_name = 'collect_vasp_candidates_PdHx_CO_r7'
    # system_name = 'collect_vasp_candidates_PdHx_all_sites'
    system_name = 'collect_vasp_candidates_PdHx_all_sites_stdout'
    
    # system_name = 'candidates_PdHx_sort' # candidates surface of CE
    # system_name = 'surface_vasp' # vasp 
    
    # system_name = 'HOCO_vasp' # vasp HOCO
    # system_name = 'CO_vasp' # vasp CO
    # system_name = 'collect_ce_init_PdHx_r2'
    # system_name = 'collect_vasp_candidates_PdHx' # 9 times
    # system_name = 'collect_vasp_coverage_H'
    # system_name = 'dft_PdHx_lowest'


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
    # write_paper_db(db)
    # assert False
    if 1:
        if False: # database to excel
            # db = del_partial_db(db)
            db2xls(system_name, xls_name, db, ref_eles, sheet_name_origin, sheet_name_stable, 
                   sheet_free_energy, sheet_binding_energy, sheet_cons, sheet_name_allFE, sheet_selectivity, sheet_name_dGs,
                   cutoff=2.8)
            print('Data done')
        
        if True: # plot
            # plot_free_enegy(xls_name, sheet_free_energy, fig_dir)
            # plot_scaling_relations(xls_name, sheet_binding_energy, fig_dir)
            # plot_selectivity(xls_name, sheet_selectivity, fig_dir)
            plot_activity(xls_name, sheet_binding_energy, fig_dir)
        
        if False:
            plot_BE_as_Hcons(xls_name, sheet_cons)
            # plot_cons_as_layers_with_ads(obj='H')
            # plot_bar_H_distribution(save=False)
            plot_line_H_distribution(save=False) # candidates distribution
        
        if False:
            plot_layers_as_strutures(db=db, obj='H', removeX=False) # dft_PdHx_lowest; change db to the lowest
            # binding_energy_distribution(ads='CO')
            # plot_count_nn(ads='CO')
            # plot_count_nn_stack(ads='CO')
            # plot_count_nn_hist(ads='CO')
        
        if False: # statistical distribution
            for adsorbate in ['HOCO', 'CO', 'H', 'OH']:
            # for adsorbate in ['OH']:
                # binding_energy_distribution(ads=adsorbate)
                # plot_count_nn(ads=adsorbate)
                # plot_count_nn_stack(ads=adsorbate)
                plot_count_nn_hist(ads=adsorbate)
    
        if False:
            # view(db_name)
            # view_ads('H', all_sites=False, save=True)
            view_db(db_name)    
            # view_ads('surface', all_sites=True)
            # views(formula='Pd51Ti13H59', all_sites=True)
    else: # test
        # plot_line_H_distribution(save=False)
        # db_ads, _ = get_ads_db(ads='surface')
        # plot_layers_as_strutures(db=db_ads, obj='H', removeX=False)
        # plot_layers_as_strutures(db=db, obj='H', removeX=False) # dft_PdHx_lowest when database is the lowest db
        
        # plot_free_enegy(xls_name, sheet_free_energy, fig_dir)
        # plot_scaling_relations(xls_name, sheet_binding_energy, fig_dir)
        plot_activity(xls_name, sheet_binding_energy, fig_dir)
        # plot_BE_as_Hcons(xls_name, sheet_cons)
        # plot_pourbaix_diagram(xls_name, sheet_name_dGs)
        # plot_chemical_potential(xls_name, sheet_name_origin)
    
        # plot_cons_as_layers(obj='Pd')
        # plot_cons_as_layers(obj='H')
        # sort_db(db)
        # plot_cons_as_layers(db_name=db_name, obj='H', removeX=False)
        # plot_cons_as_layers_with_ads(obj='H')
        # sort_db(db)
        # plot_H_distribution()
        # for adsorbate in ['HOCO', 'CO', 'H', 'OH']:
        #     plot_count_nn_hist(ads=adsorbate)
        # print(len(db_name))