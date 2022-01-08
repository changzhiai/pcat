# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 16:40:52 2022

@author: changai
"""


def formation_energy_np():
    """Old version to analyze data by numpy"""
    from pcat.lib.io import read_excel
    from pcat.formation_energy import FormationEnergyPlot
    
    type_names, obser_names, X = read_excel(filename='./data/data.xlsx', 
                                            sheet='formation_energy', 
                                            min_col=1, 
                                            max_col=7, 
                                            min_row=1, 
                                            max_row=15)
    
    type_names = ['Single', 'Dimer', 'Triangle', 'Parall.', 'Island', 'Overlayer']
    form_energy = FormationEnergyPlot(type_names, obser_names, X, fig_name='./figures/Formation_energy_example.jpg')
    form_energy.plot(save=True, title='')
    
def formation_energy_pd():
    """New version to analyze data by pandas"""
    from pcat.lib.io import pd_read_excel
    from pcat.formation_energy import FormationEnergyPlot
    
    df = pd_read_excel(filename='./data/data.xlsx', sheet='formation_energy')
    type_names = ['Single', 'Dimer', 'Triangle', 'Parall.', 'Island', 'Overlayer']
    form_energy = FormationEnergyPlot(type_names, df.index, df.values, fig_name='./figures/Formation_energy_example.jpg')
    form_energy.plot(save=True, title='')
    
def formation_energy():
    """New version to analyze data by pandas"""
    from pcat.lib.io import pd_read_excel
    from pcat.formation_energy import FormationEnergy
    
    df = pd_read_excel(filename='./data/data.xlsx', sheet='formation_energy')
    df.set_axis(['Single', 'Dimer', 'Triangle', 'Parall.', 'Island', 'Overlayer'], axis='columns', inplace=True)
    form_energy = FormationEnergy(df, fig_name='./figures/Formation_energy_example.jpg')
    form_energy.plot(save=True, title='')

if __name__ == '__main__':
    formation_energy()