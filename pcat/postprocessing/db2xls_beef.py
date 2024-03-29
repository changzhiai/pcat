import pandas as pd
import pcat.utils.constants as const
from pcat.convex_hull import con_ele
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

def db2xls(system_name, 
           xls_name, 
           db,
           beef_molecule,
           ref_eles, # such as, ref_eles=['Pd', 'Ti']
           sheet_name_origin, 
           sheet_name_stable, 
           sheet_free_energy, 
           sheet_binding_energy,
           sheet_cons,
           sheet_name_allFE, 
           sheet_selectivity,
           sheet_name_dGs):
    """
    convert database into excel
    
    totally produce FIVE sheets
    """
    """
    E_H2g = -7.158 # eV
    E_CO2g = -18.459
    E_H2Og = -12.833
    E_COg = -12.118
    
    # G_gas = E_pot + E_ZPE + C_Idel_gas - TS + Ecorr_overbing + E_solvent
    Gcor_H2g = 0.274 + 0.091 - 0.403 + 0.1 + 0
    Gcor_CO2g = 0.306 + 0.099 - 0.664 + 0.3 + 0
    Gcor_H2Og = 0.572 + 0.104 - 0.670
    Gcor_COg = 0.132 + 0.091 - 0.669
    G_H2g = E_H2g + Gcor_H2g
    G_CO2g = E_CO2g + Gcor_CO2g
    G_H2Og = E_H2Og + Gcor_H2Og
    G_COg = E_COg + Gcor_COg
    
    # G_gas = E_ZPE + C_harm - TS + Ecorr_overbing + E_solvent
    Gcor_H = 0.190 + 0.003 - 0.004 + 0 + 0
    Gcor_HOCO = 0.657 + 0.091 - 0.162 + 0.15 - 0.25
    Gcor_CO = 0.186 + 0.080 - 0.156 + 0 - 0.10
    Gcor_OH = 0.355 + 0.056 - 0.103
    """
    E_H2g = const.E_H2g
    E_CO2g = const.E_CO2g
    E_H2Og = const.E_H2Og
    E_COg = const.E_COg
    
    E_H2g_ens = (beef_molecule.get(formula='H2')).data.contributions
    E_CO2g_ens = (beef_molecule.get(formula='CO2')).data.contributions
    E_H2Og_ens = (beef_molecule.get(formula='H2O')).data.contributions
    E_COg_ens = (beef_molecule.get(formula='CO')).data.contributions
    
    G_H2g = const.G_H2g
    G_CO2g = const.G_CO2g
    G_H2Og = const.G_H2Og
    G_COg = const.G_COg
    
    Gcor_H = const.Gcor_H
    Gcor_HOCO = const.Gcor_HOCO
    Gcor_CO = const.Gcor_CO
    Gcor_OH = const.Gcor_OH
    
    # import pdb; pdb.set_trace()
    ids = []
    formulas = []
    sites = []
    adsors = []
    energies = []
    ori_ids = []
    cons_Pd = []
    cons_H = []
    ensEs = []
    for row in db.select():
        ensE = row.data.contributions
        uniqueid = row.uniqueid
        items = uniqueid.split('_')
        id = items[0]
        formula = items[1]
        if 'X' in formula:
            i_X = formula.find('X')
            formula = formula[:i_X] + formula[i_X+4:] # remove Xxxx
        site = items[2]
        adsor = items[3]
        ori_id = items[4]
        
        if adsor == 'surface':
            con_Pd = con_ele(row.toatoms(), ele='Pd', ref_eles=ref_eles) 
            con_H = con_ele(row.toatoms(), ele='H', ref_eles=ref_eles)
        else:
            con_Pd = None
            con_H = None
        
        ids.append(id)
        formulas.append(formula)
        sites.append(site)
        adsors.append(adsor)
        energies.append(row.energy)
        ori_ids.append(ori_id)
        cons_Pd.append(con_Pd)
        cons_H.append(con_H)
        ensEs.append(ensE)
    
    tuples = {'Id': ids,
              'Origin_id': ori_ids,
              'Surface': formulas,
              'Cons_Pd': cons_Pd,
              'Cons_H': cons_H,
              'Site': sites,
              'Adsorbate': adsors,
              'Energy': energies,
              'Ens_e': ensEs,
             }
    df = pd.DataFrame(tuples)
    
    """
    Save the original data and sorted by adsorbate
    """
    uniqueids = df['Origin_id'].astype(int).unique()
    df_sort  = pd.DataFrame()
    custom_dict = {'surface':0, 'HOCO': 1, 'CO': 2, 'H': 3, 'OH':4} 
    for id in uniqueids:
        df_sub = df.loc[df['Origin_id'].astype(int) == id]
        df_sub = df_sub.sort_values(by=['Adsorbate'], key=lambda x: x.map(custom_dict))
        
        Surface = df_sub.loc[df_sub['Adsorbate'] == 'surface']
        E_Surface = Surface.Ens_e.values[0]
        
        con_Pd = Surface.Cons_Pd.values[0]
        con_H = Surface.Cons_H.values[0]
        df_sub.Cons_Pd = con_Pd
        df_sub.Cons_H = con_H
        
        Binding_energy = []
        for i,row in df_sub.iterrows():
            ads = row['Adsorbate']
            if ads == 'surface':
                Binding_energy.append(0)
            elif ads == 'HOCO':
                E_HOCO = row.Ens_e
                Eb_HOCO = E_HOCO - E_Surface - E_CO2g - 0.5 * E_H2g
                Binding_energy.append(Eb_HOCO)
            elif ads == 'CO':
                E_CO = row.Ens_e
                Eb_CO = E_CO - E_Surface - E_COg
                Binding_energy.append(Eb_CO)
            elif ads == 'H':
                E_H = row.Ens_e
                Eb_H = E_H - E_Surface - 0.5 * E_H2g
                Binding_energy.append(Eb_H)
            elif ads == 'OH':
                E_OH = row.Ens_e
                Eb_OH = E_OH - E_Surface - E_H2Og + 0.5 * E_H2g
                Binding_energy.append(Eb_OH)
                
        df_sub['BE'] = Binding_energy
        df_sort = df_sort.append(df_sub, ignore_index=True)
    df_sort.to_excel(xls_name, sheet_name_origin, float_format='%.3f')
    
    surfaces = []
    step_ini = []
    step_HOCO = []
    step_CO = []
    step_final = []
    Eb_HOCOs = []
    Eb_COs = []
    Eb_Hs = []
    Eb_OHs = []
    Eb_HOCOs_ens = []
    Eb_COs_ens = []
    Eb_Hs_ens = []
    Eb_OHs_ens = []
    selectivities = []
    selectivities_ens = []
    G_HOCOs = []
    G_COs = []
    G_Hs = []
    G_OHs = []
    cons_Pd = []
    cons_H = []
    dG1s = []
    dG2s = []
    dG3s = []
    dG4s = []
    dG5s = []
    FE_final = G_COg + G_H2Og - G_CO2g - G_H2g
    df_new  = pd.DataFrame()
    for id in uniqueids:
        # print(id)
        df_sub = df_sort.loc[df_sort['Origin_id'].astype(int) == id]
        del df_sub['BE']
        Surface = df_sub.loc[df_sub['Adsorbate'] == 'surface']
        E_Surface = Surface.Energy.values[0]
        E_Surface_ens = Surface.Ens_e.values[0]
        # print(df_sub)
        HOCOs = df_sub.loc[df_sub['Adsorbate'] == 'HOCO']
        HOCO = HOCOs[HOCOs.Energy == HOCOs.Energy.min()].head(1)
        E_HOCO = HOCO.Energy.values[0]
        E_HOCO_ens = HOCO.Ens_e.values[0]
        Eb_HOCO = E_HOCO - E_Surface - E_CO2g - 0.5 * E_H2g
        Eb_HOCO_ens = E_HOCO_ens - E_Surface_ens - E_CO2g_ens - 0.5 * E_H2g_ens
        G_HOCO = E_HOCO + Gcor_HOCO - E_Surface - G_CO2g - 0.5 * G_H2g
        
        COs = df_sub.loc[df_sub['Adsorbate'] == 'CO']
        CO = COs[COs.Energy == COs.Energy.min()].head(1)
        E_CO = CO.Energy.values[0]
        E_CO_ens = CO.Ens_e.values[0]
        Eb_CO = E_CO - E_Surface - E_COg
        Eb_CO_ens = E_CO_ens - E_Surface_ens - E_COg_ens
        G_CO = E_CO + Gcor_CO + G_H2Og - E_Surface - G_H2g - G_CO2g
        
        # print(G_CO-Eb_CO)
        # >> 0.579
        # import pdb; pdb.set_trace()
        
        Hs = df_sub.loc[df_sub['Adsorbate'] == 'H']
        H = Hs[Hs.Energy == Hs.Energy.min()].head(1)
        E_H = H.Energy.values[0]
        E_H_ens = H.Ens_e.values[0]
        Eb_H = E_H - E_Surface - 0.5 * E_H2g
        Eb_H_ens = E_H_ens - E_Surface_ens - 0.5 * E_H2g_ens
        G_H = E_H + Gcor_H - E_Surface - 0.5 * G_H2g
        
        OHs = df_sub.loc[df_sub['Adsorbate'] == 'OH']
        OH = OHs[OHs.Energy == OHs.Energy.min()].head(1)
        E_OH = OH.Energy.values[0]
        E_OH_ens = OH.Ens_e.values[0]
        Eb_OH = E_OH - E_Surface - E_H2Og + 0.5 * E_H2g
        Eb_OH_ens = E_OH_ens - E_Surface_ens - E_H2Og_ens + 0.5 * E_H2g_ens
        G_OH = E_OH + Gcor_OH - E_Surface - G_H2Og + 0.5 * G_H2g
        
        Binding_energy = [Eb_HOCO, Eb_CO, Eb_H, Eb_OH]
        Binding_energy_ens = [Eb_HOCO_ens, Eb_CO_ens, Eb_H_ens, Eb_OH_ens]
        Free_energy = [G_HOCO, G_CO, G_H, G_OH] # free energy according to reaction equations
        df_stack = pd.concat([HOCO, CO, H, OH], axis=0)
        # print(HOCO)
        # print(Free_energy)
        df_stack['BE'] = Binding_energy
        df_stack['BE_ens'] = Binding_energy_ens
        df_stack['FE'] = Free_energy
        df_new = df_new.append(df_stack, ignore_index=True)
        
        surfaces.append(Surface.Surface.values[0])
        step_ini.append(0)
        step_HOCO.append(G_HOCO)
        step_CO.append(G_CO)
        step_final.append(FE_final)
        
        Eb_HOCOs.append(Eb_HOCO)
        Eb_COs.append(Eb_CO)
        Eb_Hs.append(Eb_H)
        Eb_OHs.append(Eb_OH)
        
        Eb_HOCOs_ens.append(Eb_HOCO_ens)
        Eb_COs_ens.append(Eb_CO_ens)
        Eb_Hs_ens.append(Eb_H_ens)
        Eb_OHs_ens.append(Eb_OH_ens)
        
        G_HOCOs.append(G_HOCO)
        G_COs.append(G_CO)
        G_Hs.append(G_H)
        G_OHs.append(G_OH)
        
        selectivities.append(G_HOCO - G_H)
        selectivities_ens.append(Eb_HOCO_ens - Eb_H_ens)
        
        cons_Pd.append(Surface.Cons_Pd.values[0])
        cons_H.append(Surface.Cons_H.values[0])
        
        # calculate each equation:
        dG1 = E_HOCO + Gcor_HOCO - E_Surface - G_CO2g - 0.5 * G_H2g
        dG2 = E_CO + Gcor_CO + G_H2Og - E_HOCO - Gcor_HOCO - 0.5 * G_H2g
        dG3 = G_COg + E_Surface - E_CO - Gcor_CO
        dG4 = E_H + Gcor_H - E_Surface - 0.5 * G_H2g
        dG5 = E_OH + Gcor_OH - E_Surface - G_H2Og + 0.5 * G_H2g
        dG1s.append(dG1)
        dG2s.append(dG2)
        dG3s.append(dG3)
        dG4s.append(dG4)
        dG5s.append(dG5)
        
    """
    Save the most stable site into excel
    """
    with pd.ExcelWriter(xls_name, engine='openpyxl', mode='a') as writer:
        df_new.to_excel(writer, sheet_name=sheet_name_stable, float_format='%.3f')
    
    """
    Save free energy sheet to excel
    """
    tuples = {'Surface': surfaces,
              '* + CO2': step_ini,
              '*HOCO': step_HOCO,
              '*CO': step_CO,
              '* + CO': step_final,
              }
    df_FE = pd.DataFrame(tuples)
    with pd.ExcelWriter(xls_name, engine='openpyxl', mode='a') as writer:
        df_FE.to_excel(writer, sheet_name=sheet_free_energy, index=False, float_format='%.3f')
        
    
    """
    Save binding energy sheet to excel
    """
    tuples = {'Surface': surfaces,
              'E(*HOCO)': Eb_HOCOs,
              'E(*CO)': Eb_COs,
              'E(*H)': Eb_Hs,
              'E(*OH)': Eb_OHs,
              'E(*HOCO)_ens': Eb_HOCOs_ens,
              'E(*CO)_ens': Eb_COs_ens,
              'E(*H)_ens': Eb_Hs_ens,
              'E(*OH)_ens': Eb_OHs_ens,
              }
    df_BE = pd.DataFrame(tuples)
    with pd.ExcelWriter(xls_name, engine='openpyxl', mode='a') as writer:
        df_BE.to_excel(writer, sheet_name=sheet_binding_energy, index=False, float_format='%.3f')
    
    
    """
    Save concentration sheet to excel
    """
    tuples = {'Surface': surfaces,
              'Cons_Pd': cons_Pd,
              'Cons_H': cons_H,
              'E(*HOCO)': Eb_HOCOs,
              'E(*CO)': Eb_COs,
              'E(*H)': Eb_Hs,
              'E(*OH)': Eb_OHs,
              'E(*HOCO)_ens': Eb_HOCOs_ens,
              'E(*CO)_ens': Eb_COs_ens,
              'E(*H)_ens': Eb_Hs_ens,
              'E(*OH)_ens': Eb_OHs_ens,
              }
    df_cons = pd.DataFrame(tuples)
    with pd.ExcelWriter(xls_name, engine='openpyxl', mode='a') as writer:
        df_cons.to_excel(writer, sheet_name=sheet_cons, index=False, float_format='%.3f')
    
    
    """
    Save all intermediate`s free energy sheet to excel for check
    """
    tuples = {'Surface': surfaces,
              'G(*HOCO)': G_HOCOs,
              'G(*CO)': G_COs,
              'G(*H)': G_Hs,
              'G(*OH)': G_OHs,
              }
    df_all_FE = pd.DataFrame(tuples)
    with pd.ExcelWriter(xls_name, engine='openpyxl', mode='a') as writer:
        df_all_FE.to_excel(writer, sheet_name=sheet_name_allFE, index=False, float_format='%.3f')
    
    
    """
    Save selectivity sheet to excel
    """
    tuples = {'Surface': surfaces,
              'G_HOCO-G_H': selectivities,
              'E_HOCO-E_H': selectivities_ens,
              'E(*HOCO)_ens': Eb_HOCOs_ens,
              'E(*CO)_ens': Eb_COs_ens,
              'E(*H)_ens': Eb_Hs_ens,
              'E(*OH)_ens': Eb_OHs_ens,
              'E(*HOCO)': Eb_HOCOs,
              'E(*CO)': Eb_COs,
              'E(*H)': Eb_Hs,
              'E(*OH)': Eb_OHs,
              }
    df_select = pd.DataFrame(tuples)
    with pd.ExcelWriter(xls_name, engine='openpyxl', mode='a') as writer:
        df_select.to_excel(writer, sheet_name=sheet_selectivity, index=False, float_format='%.3f')
        
    """
    Save each reaction equation`s free energy difference sheet to excel for check
    """
    tuples = {'Surface': surfaces,
              'dG1': dG1s,
              'dG2': dG2s,
              'dG3': dG3s,
              'dG4': dG4s,
              'dG5': dG5s,
              }
    df_dGs = pd.DataFrame(tuples)
    with pd.ExcelWriter(xls_name, engine='openpyxl', mode='a') as writer:
        df_dGs.to_excel(writer, sheet_name=sheet_name_dGs, index=False, float_format='%.3f')
        
        
        