# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:20:36 2022

@author: changai
"""


from ase.db import connect
from ase.visualize import view
# from ase.geometry import get_distances
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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

def get_overlayer_bond_length(db_name, configuration, sheet_name_bonds, sheet_name_stable):
    """Get C-M and C-O bond lengths for each structure for HOCO*
    """
    db = connect(db_name)
    
    eles = []
    overly_lens = []
    bulk_lens = []
    for row in db.select():
        # confs = (row.configurations).split('_')
        confs = (row.configurations).split('_')
        
        if len(confs) == 3 and confs[2] == 'surface' and confs[0] == configuration:
        # if len(confs) == 4 and confs[3] == 'HOCO' and confs[0] == 'overlayer':
            ele = confs[1]
            # print(ele)
            eles.append(ele)
            atoms = row.toatoms()
            # view(atoms)
            # assert False
            # 45-53
            atoms = get_first_layer(atoms)
            # view(atoms)
            
            # dists = atoms.get_all_distances()
            dist = atoms.get_distance(0, 1)
            # print(dist)
            overly_lens.append(dist)
            
            kw = get_bulk_bond_length()
            
            for e in kw:
                if ele.lower() == e:
                    bulk_lens.append(kw[e])
                    print(e)
                    print(kw[e])
    
    tuples = {'Element': eles,
              'Overly_length': overly_lens,
              'Bulk_length': bulk_lens,
              }
    df_bonds = pd.DataFrame(tuples)
    df_bonds['Bulk_length'][df_bonds.Element == 'Mn'] = 2.481
    df_bonds.to_excel(xls_name, 'Bulk', float_format='%.3f')
    diff = df_bonds['Overly_length'] - df_bonds['Bulk_length']
    

    # plt.figure()
    # plt.scatter(df_bonds['Element'], diff.values)
    # plt.plot(df_bonds['Element'], diff.values)
    # plt.xlabel('Doped elements')
    # plt.ylabel('Bond length difference of overlayer and bulk')
    # plt.show()
    
    from plotpackage.lib.io import read_excel, read_csv
    import numpy as np
    filename = './sites.xlsx'
    min_col = 1+14 #1st column in excel
    max_col = 7+14 #5th column in excel
    sheet = 'stability' #Sheet1 by defaut
    min_row = 2 #1st column in excel
    max_row = 24 #9st column in excel

    colorList = ['k', 'lime', 'r', 'b', 'darkcyan', 'cyan', 'olive', 'magenta', 'pink', 'gray', 'orange', 'purple', 'g']
    typeNames, observationName, X = read_excel(filename, sheet, min_col, max_col, min_row, max_row) #load excel data
    
    
    tuples = {'Elements': eles,
              'Bond_length_diff': diff,
              'FE_Single': X[:,0],
              'FE_Dimer': X[:,1],
              'FE_Triangle': X[:,2],
              'FE_Parall.': X[:,3],
              'FE_Island': X[:,4],
              'FE_Overlayer': X[:,5],
              }
    df = pd.DataFrame(tuples)
    df = df.set_index(['Elements'])
    # df.drop(['Y', 'Cd', 'Mn'], inplace=True)
    # print(X[:,5])
    
    # plt.scatter(X[:,5], diff)
    # plt.xlabel('Formation energy')
    # plt.ylabel('Bond length difference of overlayer and bulk')
    # plt.show()
    # fig = plt.figure(figsize=(8, 6), dpi = 300)
    # x = np.arange(0,len(observationName),1)
    
    # marker = ['o', '^', '<', '>', 'v', 's', 'd', '.', ',', 'x', '+']
    text = ['Single', 'Dimer', 'Triangle', 'Parall.', 'Island', 'Overlayer']
    for j in range(len(text)):    
        # plt.plot(x, X[:,i], 's', color=colorList[i])  #plot dots
        # x_axis = X[:,j]
        x_axis = df['FE_'+typeNames[j]]
        # y_axis = diff
        y_axis = df['Bond_length_diff']
        fig = plt.figure()
        plt.scatter(x_axis, y_axis)
        for i, name in enumerate(df.index):
            plt.annotate(name, (x_axis[i], y_axis[i]+0.005), fontsize=14, horizontalalignment='center', verticalalignment='bottom')
        
        # rmse = mean_squared_error(x_axis, y_axis, squared=False)
        # print(rmse)
        
        r2 = r2_score(x_axis, y_axis)
        m, b = np.polyfit(x_axis, y_axis, 1)
        handleFit = plt.plot(np.array(x_axis), m * np.array(x_axis) + b)
        # fig.text(0.5, 0.17, 'RMSE = {:.3f} meV/atom'.format(rmse*1000), fontsize=12)
        plt.legend(handles = handleFit, labels = ['$R^2$ = {}\ny = {} + {} * x '.format(round(r2, 2), round(b, 2), round(m, 2))],
                                                            loc="lower right", handlelength=0, fontsize=14)
        
        plt.title(typeNames[j])
        plt.xlabel('Formation energy')
        plt.ylabel('Bond length difference of overlayer and bulk')
        plt.show()
        
    x_axis = df['FE_Single'] - df['FE_Overlayer'] 
    # x_axis = df['FE_Overlayer'] - df['FE_Single']
    y_axis = df['Bond_length_diff']
    fig = plt.figure()
    plt.scatter(x_axis, y_axis)
    for i, name in enumerate(df.index):
        plt.annotate(name, (x_axis[i], y_axis[i]+0.005), fontsize=14, horizontalalignment='center', verticalalignment='bottom')
    
    # rmse = mean_squared_error(x_axis, y_axis, squared=False)
    r2 = r2_score(x_axis.values, y_axis.values)
    # print(rmse)
    m, b = np.polyfit(x_axis, y_axis, 1)
    handleFit = plt.plot(np.array(x_axis), m * np.array(x_axis) + b)
    # fig.text(0.5, 0.17, 'RMSE = {:.3f} meV/atom'.format(rmse*1000), fontsize=12)
    plt.legend(handles = handleFit, labels = ['$R^2$ = {}\ny = {} + {} * x '.format(round(r2, 2), round(b, 2), round(m, 2))],
                                                        loc="lower right", handlelength=0, fontsize=14)
    
    plt.title('FE_single_minus_overlayer')
    plt.xlabel('Formation energy')
    plt.ylabel('Bond length difference of overlayer and bulk')
    plt.show()

    # typeNames = ['Overlayer']    
    # plt.legend(typeNames, framealpha=0.5, fontsize=12, bbox_to_anchor=(0.15, 0, 0.8, 1.02), edgecolor='grey')
    # plt.axhline(y=0, color='r', linestyle='--')
    
    # plt.xlim([-0.3, 21.3])
    # plt.ylim([-3., 2])
    # plt.xlabel('Doping elements', fontsize=16)
    # plt.ylabel('Formation energy (eV/atom)', fontsize=16)
    # ax = fig.gca()
    # ax.set_xticks(x)
    # ax.set_xticklabels(observationName)
    
    # ax.tick_params(labelsize=13.5) #tick label font size
    # for axis in ['top','bottom','left','right']:
    #     ax.spines[axis].set_linewidth(1.2) #linewith of frame
    
    plt.show()
    # fig.savefig(figName1, dpi=300, bbox_inches='tight')

def get_bulk_bond_length(db_name='./metal.db'):
    db = connect(db_name)
    kw = {}
    for row in db.select():
        confs = (row.name).split('_')
        ele = confs[1]
        print(ele)
        atoms = row.toatoms()
        dists = atoms.get_all_distances()
        nums = dists.ravel().tolist()
        nums = [num for num in nums if num != 0]
        print(min(nums))
        kw[ele] = min(nums)
        
    return kw


if __name__ == '__main__':
    # db_name = 'bonds/doped_PdH.db'
    db_name = 'final_doped_PdH.db'
    
    system_name = 'bonds'
    xls_name = f'bonds/{system_name}.xlsx'
    fig_dir = 'bonds'
    # configurations = ['dimer']
    # configurations = ['single', 'triangle', 'paral', 'island', 'overlayer']
    # configurations = ['single', 'dimer', 'triangle', 'paral', 'island', 'overlayer']
    configurations = ['overlayer']
    # df = pd.DataFrame(configurations)
    # df.to_excel(xls_name, 'Blank')
    for configuration in configurations:
        sheet_name_bonds = 'Origin_{}'.format(configuration)
        sheet_name_stable = 'Stable_{}'.format(configuration)
        
        # view_HOCO(db_name)
        get_overlayer_bond_length(db_name, configuration, sheet_name_bonds, sheet_name_stable)
        print('====================================')
        # get_bulk_bond_length()
    