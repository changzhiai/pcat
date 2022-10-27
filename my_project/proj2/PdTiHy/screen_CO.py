# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 11:40:54 2022

@author: changai
"""

from pcat.preprocessing.db2xls import db2xls_CO_filter
from pcat.lib.io import pd_read_excel
from ase.db import connect


def get_CO_binding_energies(xls_name, sheet_binding_e):
    df = pd_read_excel(filename=xls_name, sheet=sheet_binding_e)
    x = df['Cons_Pd']
    y = df['Cons_H']
    z = df['Form_e']
    id = df['Id']




if __name__ == '__main__':


    system_name = 'PdTiH_surf_r'
    
    metal_obj = 'Ti'
    ref_eles=['Pd', 'Ti']
    db_name = f'./data/{system_name}.db' # the only one needed
    xls_name = f'./data/{system_name}_CO.xlsx'
    fig_dir = '../figures'
    

    sheet_binding_e = 'Binding_energy'
    
    db = connect(db_name)
    if False: # database to excel
        Ti_energy_ref_eles={'Pd':-1.951, metal_obj:-5.858, 'H': -7.158*0.5}
        db2xls_CO_filter(system_name, xls_name, sheet_binding_e, Ti_energy_ref_eles)
    
    if True:
        get_CO_binding_energies(xls_name, sheet_binding_e)
    
