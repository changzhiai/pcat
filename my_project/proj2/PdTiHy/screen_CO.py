# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 11:40:54 2022

@author: changai
"""

from pcat.preprocessing.db2xls_CO_filter import db2xls
from pcat.lib.io import pd_read_excel
from ase.db import connect
import pandas as pd
from ase import Atom, Atoms
from ase.visualize import view

def get_CO_binding_energies(xls_name, sheet_binding_e, save_to_csv = False):
    df = pd_read_excel(filename=xls_name, sheet=sheet_name_stable)
    surf = df['Surface']
    site = df['Site']
    be = df['BE']
    df_sub = df[(df['BE']>-0.8) & (df['BE']<0.7)]
    # print(df)
    print(df_sub)
    save_to_xls = False
    if save_to_xls:
        with pd.ExcelWriter(xls_name, engine='openpyxl', mode='a') as writer:
            df_sub.to_excel(writer, sheet_name='CO_candidates', index=False, float_format='%.3f')
    # save_to_csv = False
    if save_to_csv:   
        df_sub.to_csv('CO_candidates.csv')
    return df_sub

def add_ads(adsorbate, position, bg=False):
    """Add adsorbates on a specific site"""
    if adsorbate == "HOCO":
        ads = Atoms(
            [
                Atom("H", (0.649164000, -1.51784000, 0.929543000), tag=-4),
                Atom("C", (0.000000000, 0.000000000, 0.000000000), tag=-4),
                Atom("O", (-0.60412900, 1.093740000, 0.123684000), tag=-4),
                Atom("O", (0.164889000, -0.69080700, 1.150570000), tag=-4),
            ]
        )
        ads.translate(position + (0.0, 0.0, 2.5))
    elif adsorbate == "CO":
        ads = Atoms([Atom("C", (0.0, 0.0, 0.0), tag=-2), 
                      Atom("O", (0.0, 0.0, 1.14), tag=-2)])
        ads.translate(position + (0.0, 0.0, 2.0))
    elif adsorbate == "H":
        ads = Atoms([Atom("H", (0.0, 0.0, 0.0), tag=-1)])
        ads.translate(position + (0.0, 0.0, 2.0))
    elif adsorbate == "OH":
        ads = Atoms([Atom("O", (0.0, 0.0, 0.0), tag=-3), 
                      Atom("H", (0.0, 0.0, 0.97), tag=-3)])
        ads.translate(position + (0.0, 0.0, 2.0))
    if bg == True:
        for atom in ads:
            atom.symbol = "X"
    return ads

def generate_all_sites_HOCO(cands_id):
    db_CO = connect('./data/all_sites_CO_on_cands.db')
    db_HOCO = connect('all_sites_HOCO_on_cands.db')
    for row in db_CO.select():
        uniqueid = (row.uniqueid).split('_')
        row_id = uniqueid[0]
        name = uniqueid[1]
        site = uniqueid[2]
        adsorbate = 'HOCO'
        ori_id = uniqueid[4] # initial id
        unique_id = str(row_id) + '_' + name + '_' + site + '_' + adsorbate + '_' + str(ori_id)
        
        if int(ori_id) in cands_id:
            atoms = row.toatoms()
            pos = [atom for atom in atoms if atom.symbol=='C'][0].position
            pos = pos - (0.0, 0.0, 2.0)
            atoms = atoms[[atom.index for atom in atoms if atom.symbol!='C' and atom.symbol!='O']]
            ads = add_ads(adsorbate, pos, bg=False)
            atoms.extend(ads)
            # view(atoms)
            db_HOCO.write(atoms, uniqueid=unique_id, slab_id=row.slab_id)
            print(atoms)
    
    
    
if __name__ == '__main__':

    system_name = 'collect_vasp_candidates_PdTiH_all_sites'
    
    metal_obj = 'Ti'
    ref_eles=['Pd', 'Ti']
    db_name = f'./data/{system_name}.db' # the only one needed
    xls_name = f'./data/{system_name}_CO.xlsx'
    fig_dir = '../figures'
    

    sheet_binding_e = 'Binding_energy'
    sheet_name_stable = 'most_stable_b'
    sheet_binding_energy = 'be'
    db = connect(db_name)
    if False: # database to excel
        Ti_energy_ref_eles={'Pd':-1.951, metal_obj:-5.858, 'H': -7.158*0.5}
        db2xls(system_name, xls_name, db,  ref_eles, sheet_binding_e, sheet_name_stable, sheet_binding_energy)
        print('data done')
    
    if True:
        df_sub = get_CO_binding_energies(xls_name, sheet_binding_e, save_to_csv = False)
        cands_id = list(df_sub.Origin_id.values)
        generate_all_sites_HOCO(cands_id)

    
