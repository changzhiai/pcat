# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 21:05:13 2022

@author: changai
"""
import matplotlib.pyplot as plt
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
logging.getLogger('matplotlib.font_manager').disabled = True

class SelectivityPlot:
    """Class for CO2RR selectivity
    
    Parameters:
        
    step_names: str
        names of columns (steps)
    obser_names: str
        names of rows (observations)
    X: float
        data for CO2RR selectivity plot
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
        
    
    def plot(self, title='', save = False, ymin=None, ymax=None,xlabel='Doping elements', tune_tex_pos=1.5, legend=True):
        """Plot formation energy"""
        
        color_list = ['k', 'lime', 'r', 'b', 'darkcyan', 'cyan', 'olive', 'magenta', 'pink', 'gray', 'orange', 'purple', 'g']
        
        fig = plt.figure(figsize=(8, 6), dpi = 300)
        x = np.arange(0,len(self.obser_names),1)
        
        for i in range(len(self.type_names)):    
            plt.plot(x, self.X[:,i], 'o', color=color_list[i])  #plot dots
          
        if legend == True:
            plt.legend(self.type_names, framealpha=0.5, fontsize=12,  edgecolor='grey')
            
        plt.axhline(y=0, color='r', linestyle='--')
        # plt.xlim([-0.3, 23.5])
        plt.ylim([-2.5, 2.5])
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel('$\Delta G_{HOCO*}$-$\Delta G_{H*}$', fontsize=16)
        ax = fig.gca()
        ax.set_xticks(x)
        ax.set_xticklabels(self.obser_names)
        
        
        ax.tick_params(labelsize=13.2) # tick label font size
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.2) # linewith of frame
        
        max_x = len(self.obser_names)
        plt.arrow(x=max_x, y=0, dx=0, dy=2, width=.4, head_width=0.8, head_length=0.3, color='grey') 
        plt.annotate('HER', xy = (max_x-tune_tex_pos, 1), rotation=90, fontsize=14, color='grey')
        
        plt.arrow(x=max_x, y=0, dx=0, dy=-2, width=.4, head_width=0.8, head_length=0.3, color='grey') 
        plt.annotate('CO$_2$RR', xy = (max_x-tune_tex_pos, -1.5), rotation=90, fontsize=14, color='grey')
        plt.show()
        
        if save:
            fig.savefig(self.fig_name, dpi=300, bbox_inches='tight')
            
class Selectivity(SelectivityPlot):
    """New version of CO2RR selectivity using panda, and thus less varalbe would be used"""
    def __init__(self, df, fig_name):
        self.type_names = df.columns
        self.obser_names = df.index
        self.X = df.values
        self.fig_name = fig_name
        super().__init__(self.type_names, self.obser_names, self.X, self.fig_name)
        logging.debug(f'loaded selectivity table: \n {df}')