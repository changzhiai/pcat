# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 15:21:50 2022

@author: changai
"""



def free_energy_np():
    """Old version to analyze data by numpy"""
    from pcat.lib.io import read_excel
    from pcat.free_energy import CO2RRFEDplot, CO2RRFED_with_TS
    from pcat.utils.styles import StepNamesCO2RR
    
    # step_names, obser_names, X = read_csv(filename, , min_col, max_col) # load csv data
    step_names, obser_names, X = read_excel(filename='./data/data.xlsx', 
                                            sheet='free_energy', 
                                            min_col=1, 
                                            max_col=5, 
                                            min_row=1, 
                                            max_row=15) 
    CO2RR_FED = CO2RRFEDplot(step_names, obser_names, X, fig_name='./figures/CO2RR_FED_example.jpg')
    CO2RR_FED.plot(save=True, title='')
    
def free_energy_pd():
    """New version to analyze data by pandas"""
    from pcat.lib.io import pd_read_excel
    from pcat.free_energy import CO2RRFEDplot

    
    df = pd_read_excel(filename='./data/data.xlsx', sheet='free_energy')
    CO2RR_FED = CO2RRFEDplot(df.columns, df.index,df.values, fig_name='./figures/CO2RR_FED_example.jpg')
    CO2RR_FED.plot(save=True, title='')
    
def free_energy():
    """New version to analyze data by pandas"""
    from pcat.lib.io import pd_read_excel
    from pcat.free_energy import CO2RRFED
    
    df = pd_read_excel(filename='./data/data1.xlsx', sheet='free_energy')
    CO2RR_FED = CO2RRFED(df, fig_name='./figures/CO2RR_FED_example.jpg')
    CO2RR_FED.plot(save=True, title='')

def free_energy_TS():
    """New version to analyze data by pandas"""
    from pcat.lib.io import pd_read_excel
    from pcat.free_energy import CO2RRFED, CO2RRFED_TS
    
    df = pd_read_excel(filename='./data/data_TS.xlsx', sheet='ex1')
    CO2RR_FED = CO2RRFED_TS(df, fig_name='./figures/CO2RR_FED_example1.jpg')
    CO2RR_FED.plot(save=False, title='')

if __name__ == '__main__':
    free_energy_TS()


