# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 17:03:32 2022

@author: changai
"""

def selectivity_np():
    """Old version to analyze data by numpy"""
    from pcat.lib.io import read_excel
    from pcat.selectivity import SelectivityPlot
    
    type_names, obser_names, X = read_excel(filename='./data/data.xlsx', 
                                            sheet='selectivity', 
                                            min_col=1, 
                                            max_col=7, 
                                            min_row=1, 
                                            max_row=15)
    
    type_names = ['Single', 'Dimer', 'Triangle', 'Parall.', 'Island', 'Overlayer']
    selectivity = SelectivityPlot(type_names, obser_names, X, fig_name='./figures/Selectivity_example.jpg')
    selectivity.plot(save=True, title='')
    
def selectivity_pd():
    """New version to analyze data by pandas"""
    from pcat.lib.io import pd_read_excel
    from pcat.selectivity import SelectivityPlot
    
    df = pd_read_excel(filename='./data/data.xlsx', sheet='selectivity')
    type_names = ['Single', 'Dimer', 'Triangle', 'Parall.', 'Island', 'Overlayer']
    selectivity = SelectivityPlot(type_names, df.index, df.values, fig_name='./figures/Selectivity_example.jpg')
    selectivity.plot(save=True, title='')
    
def selectivity():
    """New version to analyze data by pandas"""
    from pcat.lib.io import pd_read_excel
    from pcat.selectivity import Selectivity
    
    df = pd_read_excel(filename='./data/data.xlsx', sheet='selectivity')
    df.set_axis(['Single', 'Dimer', 'Triangle', 'Parall.', 'Island', 'Overlayer'], axis='columns', inplace=True)
    selectivity = Selectivity(df, fig_name='./figures/Selectivity_example.jpg')
    selectivity.plot(save=True, title='')

if __name__ == '__main__':
    selectivity()