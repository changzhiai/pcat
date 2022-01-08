# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 15:26:18 2022

@author: changai
"""



def scaling_relation_np():
    """Old version to analyze data by numpy"""
    from pcat.lib.io import read_excel
    from pcat.scaling_relation import ScalingRelationPlot
    
    step_names, obser_names, X = read_excel(filename='./data/data.xlsx', 
                                            sheet='binding_energy', 
                                            min_col=1, 
                                            max_col=5, 
                                            min_row=1, 
                                            max_row=15)
    col1 = 2 # column in excel
    col2 = 3 # column in excel
    xcol = col1 - 2 
    ycol = col2 - 2
    descriper1 = (X[:, xcol]).astype(float)
    descriper2 = (X[:, ycol]).astype(float)
    sr = ScalingRelationPlot(descriper1, descriper2, obser_names, fig_name='./figures/CO2RR_SR_example.jpg')
    sr.plot(save=True, title='', xlabel=step_names[xcol], ylabel=step_names[ycol], dot_color='black', line_color='r')

def scaling_relation_pd():
    """New version to analyze data by pandas"""
    from pcat.lib.io import pd_read_excel
    from pcat.scaling_relation import ScalingRelationPlot
    
    df = pd_read_excel(filename='./data/data.xlsx', sheet='binding_energy')
    descriper1 = '*HOCO'
    descriper2 = '*CO'
    sr = ScalingRelationPlot(df[descriper1], df[descriper2], df.index, fig_name='./figures/CO2RR_SR_example.jpg')
    sr.plot(save=True, title='', xlabel=descriper1, ylabel=descriper2, dot_color='black', line_color='r')
    
def scaling_relation():
    """New version to analyze data by pandas"""
    from pcat.lib.io import pd_read_excel
    from pcat.scaling_relation import ScalingRelation
    
    df = pd_read_excel(filename='./data/data.xlsx', sheet='binding_energy')
    descriper1 = '*HOCO'
    descriper2 = '*CO'
    sr = ScalingRelation(df, descriper1, descriper2, fig_name='./figures/CO2RR_SR_example.jpg')
    sr.plot(save=True, title='', xlabel=descriper1, ylabel=descriper2, dot_color='black', line_color='r')
    
if __name__ == '__main__':
    scaling_relation()