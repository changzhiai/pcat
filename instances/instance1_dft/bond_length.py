# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 16:32:44 2022

@author: changai
"""
from ase.db import connect
from ase.visualize import view
# from ase.geometry import get_distances
import numpy as np
import pandas as pd

def view_db(db_name):
    """View all structures in the database"""
    db = connect(db_name)
    atoms_all = []
    for row in db.select():
        atoms_all.append(row.toatoms())
    view(atoms_all)
    
def get_first_layer(atoms):
    """Verify the first layer`s index"""
    first_layer_index = np.arange(45, 54)
    del_list = []
    for atom in atoms:
        if atom.index not in first_layer_index:
            del_list.append(atom.index)
    del atoms[del_list]
    return atoms

def get_distance(atoms):
    """Get distance
    
    C index: 109
    O index: 110
    """
    
    dists = []
    for ads_index in [109, 110]:
        dis_min = 100
        for surf_index in np.arange(45, 54):
            dis = atoms.get_distance(ads_index, surf_index)
            dis_min = min(dis, dis_min)
        dists.append(dis_min) 
    return dists
    

def view_HOCO(db_name):
    """View HOCO in configuration of overlayer"""
    db = connect(db_name)
    atoms_ads = []
    for row in db.select():
        confs = (row.configurations).split('_')
        if len(confs) == 4 and confs[3] == 'HOCO' and confs[0] == 'paral':
        # if len(confs) == 4 and confs[3] == 'HOCO' and confs[0] == 'overlayer':
            atoms = row.toatoms()
            # atoms = get_first_layer(atoms)
            atoms_ads.append(atoms)
    view(atoms_ads)
    
def view_ids(ids):
    db = connect(db_name)
    structs = []
    for row_id in ids:
        row = list(db.select(id=row_id))[0]
        structs.append(row.toatoms())
    view(structs)
        

def get_bond_length(db_name, configuration, sheet_name_bonds, sheet_name_stable):
    """Get C-M and C-O bond lengths for each structure for HOCO*
    """
    db = connect(db_name)
    
    names = []
    eles = []
    sites = []
    ads = []
    bonds_C_M = []
    bonds_O_M = []
    energies = []
    ids = []
    for row in db.select():
        # confs = (row.configurations).split('_')
        confs = (row.name).split('_')
        if len(confs) == 4 and confs[3] == 'HOCO' and confs[0] == configuration:
        # if len(confs) == 4 and confs[3] == 'HOCO' and confs[0] == 'overlayer':
            atoms = row.toatoms()
            # atoms = get_first_layer(atoms)
            dists = get_distance(atoms)
            names.append(confs[0])
            eles.append(confs[1])
            sites.append(confs[2])
            ads.append(confs[3])
            bonds_C_M.append(dists[0])
            bonds_O_M.append(dists[1])
            energies.append(row.energy)
            ids.append(row.id)
    
    tuples = {'Name': names,
              'Dopant': eles,
              'Site': sites,
              'Adsorbate': ads,
              'C-M': bonds_C_M,
              'O-M': bonds_O_M,
              'Energy': energies,
              'DB_id': ids,
              }
    df_bonds = pd.DataFrame(tuples)
    # df_bonds.to_excel(xls_name, sheet_name_bonds, float_format='%.3f')
    with pd.ExcelWriter(xls_name, engine='openpyxl', mode='a') as writer:
        df_bonds.to_excel(writer, sheet_name=sheet_name_bonds, float_format='%.3f')
    
    df_new  = pd.DataFrame()
    for ele in sorted(set(eles), key=eles.index):
        df_sub = df_bonds.loc[df_bonds['Dopant']==ele]
        HOCO_min = df_sub[df_sub.Energy == df_sub.Energy.min()]
        # E_HOCO_min = HOCO_min.Energy.values[0]
        df_new = df_new.append(HOCO_min, ignore_index=True)
        
    with pd.ExcelWriter(xls_name, engine='openpyxl', mode='a') as writer:
        df_new.to_excel(writer, sheet_name=sheet_name_stable, float_format='%.3f')
    
    # view_ids(df_new.DB_id)
    # single bond by Fe, Co, Ni, Cu, Ru, Rh and Ag


if __name__ == '__main__':
    db_name = 'bonds/doped_PdH.db'
    
    system_name = 'bonds'
    xls_name = f'bonds/{system_name}.xlsx'
    fig_dir = 'bonds'
    
    # configurations = ['dimer']
    # configurations = ['single', 'triangle', 'paral', 'island', 'overlayer']
    configurations = ['single', 'dimer', 'triangle', 'paral', 'island', 'overlayer']
    df = pd.DataFrame(configurations)
    df.to_excel(xls_name, 'Blank')
    for configuration in configurations:
        sheet_name_bonds = 'Origin_{}'.format(configuration)
        sheet_name_stable = 'Stable_{}'.format(configuration)
        
        # view_HOCO(db_name)
        get_bond_length(db_name, configuration, sheet_name_bonds, sheet_name_stable)
    