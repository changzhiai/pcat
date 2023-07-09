# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:30:44 2023

@author: changai
"""

def free_energy_pd():
    """New version to analyze data by pandas"""
    from pcat.lib.io import pd_read_excel
    from pcat.free_energy import CO2RRFEDplot

    
    df = pd_read_excel(filename='./data/iter25.xlsx', sheet='free_energy')
    
    ColorDict = {'image0': 'C0', 'image1': 'C1', 'image2': 'C2', 'image3': 'C3', 'image4': 'C4', }
    stepsNames = ['* + CO$_{2}$', 'HOCO*', 'CO*', '* + CO']
    CO2RR_FED = CO2RRFEDplot(stepsNames, df.index,df.values, fig_name='./CO2RR_FED.jpg', ColorDict=ColorDict)
    CO2RR_FED.plot(save=True, title='')
    
def plot_activity():
    from pcat.lib.io import pd_read_excel
    from pcat.activity import Activity
    """Plot activity of CO2RR"""
    df = pd_read_excel(filename='./data/iter25.xlsx', sheet='binding_energy')
    
    # df.drop(['Pd16Ti48H8', 'Pd16Ti48H24'], inplace=True)
    name_fig_act = './data/iter25_activity.jpg'
    activity = Activity(df, descriper1 = 'E(*CO)', descriper2 = 'E(*HOCO)', fig_name=name_fig_act,
                        U0=-0.5, 
                        T0=297.15, 
                        pCO2g = 1., 
                        pCOg=0.005562, 
                        pH2Og = 1., 
                        cHp0 = 10.**(-0.),
                        Gact=0.2, 
                        p_factor = 3.6 * 10**4)
    # activity.verify_BE2FE()
    activity.plot(save=True, text=True)
    # activity.plot(save=True,)
    # activity.plot(save=True, xlim=[-2.5, 2.5], ylim=[-2.5, 2.5])
    # activity.plot(save=True, xlim=[-1., 0], ylim=[-0., 1])
    # activity.plot(save=True, xlim=[-2.5, 1.0], ylim=[-2.5, 1])


if __name__ == '__main__':
    free_energy_pd()
    plot_activity()
