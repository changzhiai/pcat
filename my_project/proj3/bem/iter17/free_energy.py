# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:30:44 2023

@author: changai
"""

def free_energy_pd():
    """New version to analyze data by pandas"""
    from pcat.lib.io import pd_read_excel
    from pcat.free_energy import CO2RRFEDplot

    
    df = pd_read_excel(filename='./data/ads.xlsx', sheet='free_energy')
    
    ColorDict = {'image0': 'C0', 'image1': 'C1', 'image2': 'C2', 'image3': 'C3', 'image4': 'C4', }
    CO2RR_FED = CO2RRFEDplot(df.columns, df.index,df.values, fig_name='./CO2RR_FED.jpg', ColorDict=ColorDict)
    CO2RR_FED.plot(save=True, title='')
    
if __name__ == '__main__':
    free_energy_pd()
