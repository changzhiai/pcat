# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 21:05:33 2022

@author: changai
"""
import matplotlib.pyplot as plt
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG, format='(%(asctime)s) %(message)s')
logging.getLogger('matplotlib.font_manager').disabled = True

class FormationEnergyPlot:
    """Class for CO2RR selectivity
    
    Parameters:
        
    type_names: str
        names of columns (steps)
    obser_names: str
        names of rows (observations)
    X: float
        data for CO2RR formation energy plot
    ColorDict: dict
        used to costom color of curve
    DefaultColor: list
        default color of curve
    """
    def __init__(self, type_names, obser_names, X, fig_name):
        self.type_names = type_names
        self.obser_names = obser_names
        self.X = X
        self.fig_name = fig_name
        # self.ColorDict = ColorDict
        # self.DefaultColor=cm.rainbow(np.linspace(0,1,len(self.obser_names)))
        
    
    def plot(self, title='', save = False, ymin=None, ymax=None):
        """Plot formation energy"""
        
        color_list = ['k', 'lime', 'r', 'b', 'darkcyan', 'cyan', 'olive', 'magenta', 'pink', 'gray', 'orange', 'purple', 'g']
        
        fig = plt.figure(figsize=(8, 6), dpi = 300)
        
        x = np.arange(0,len(self.obser_names),1)
        for i in range(len(self.type_names)):    
            plt.plot(x, self.X[:,i], 's', color=color_list[i])  # plot dots
        
        plt.legend(self.type_names, framealpha=0.5, fontsize=12, bbox_to_anchor=(0.15, 0, 0.8, 1.02), edgecolor='grey')
        plt.axhline(y=0, color='r', linestyle='--')
        
        # plt.xlim([-0.3, 21.3])
        plt.ylim([-3., 2])
        plt.xlabel('Doping elements', fontsize=16)
        plt.ylabel('Formation energy (eV/atom)', fontsize=16)
        ax = fig.gca()
        ax.set_xticks(x)
        ax.set_xticklabels(self.obser_names)
        ax.tick_params(labelsize=13.5) # tick label font size
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.2) # linewith of frame
        plt.show()
        
        if save:
            fig.savefig(self.fig_name, dpi=300, bbox_inches='tight')

class FormationEnergy(FormationEnergyPlot):
    """New version of CO2RR formation energy using panda, and thus less varalbe would be used"""
    def __init__(self, df, fig_name):
        self.type_names = df.columns
        self.obser_names = df.index
        self.X = df.values
        self.fig_name = fig_name
        super().__init__(self.type_names, self.obser_names, self.X, self.fig_name)
        logging.debug(f'loaded formation energy table: \n {df}')