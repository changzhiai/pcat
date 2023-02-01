# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 00:31:10 2021

@author: changai
"""

from pcat.lib.FED import FED
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from pcat.utils.styles import ColorDict
from typing import List
import logging
logging.basicConfig(level=logging.DEBUG, format='\n(%(asctime)s) \n%(message)s')
logging.getLogger('matplotlib.font_manager').disabled = True
logging.disable()

class CO2RRFEDplot:
    """Class for CO2RR free energy diagram without TS data
    
    Parameters:
        
    step_names: list, str
        names of columns (steps)
    obser_names: list, str
        names of rows (observations)
    X: float
        data for CO2RR free energy plot
    fig_name: str
        figure name
    ColorDict: dict
        used to costom color of curve
    DefaultColor: list
        default color of curve
    """
    def __init__(self, step_names: List[str], 
                 obser_names: List[str], 
                 X: np.ndarray, 
                 fig_name: str,
                 info=True) -> None:
        self.step_names = step_names
        self.obser_names = obser_names
        self.X = X
        self.fig_name = fig_name
        # self.axFree = None
        # self.figFree = None
        self.ColorDict = ColorDict
        self.DefaultColor=cm.rainbow(np.linspace(0,1,len(self.obser_names)))
        if info == True:
            logging.debug(f'loaded step_names:\n {self.step_names}')
            logging.debug(f'loaded obser_names:\n {self.obser_names}')
            logging.debug(f'loaded data:\n {self.X}')
        self.diagram = FED()
        count = 0
        for i, specis in enumerate(self.obser_names):
            for step in range(len(self.step_names)):
        # for specis in range(len(self.obser_names)):
        #     for step in range(len(self.step_names)):
                count += 1
                if step == 0:
                    self.diagram.pos_number = 0
                try:
                    self.diagram.add_level(self.X[i][step], color = self.ColorDict[specis])
                except:
                    self.diagram.add_level(self.X[i][step], color = self.DefaultColor[i])
        
                if count % (len(self.step_names)) != 0:
                    try:
                        self.diagram.add_link(count-1, count, color = self.ColorDict[specis])
                    except:
                        self.diagram.add_link(count-1, count, color = self.DefaultColor[i])
    
    def add_link(self, start_id=None, end_id=None, color='k', linestyle='--', linewidth=1):
        """Add dashed link line between levels"""
        if start_id != None and end_id != None:  # pos starts from 0
            self.diagram.add_link(start_id, end_id, color, linestyle, linewidth)

    def remove_link(self, start_id=None, end_id=None):
        """Remove dashed link line between levels"""
        if start_id != None and end_id != None:
            self.diagram.remove_link(start_id, end_id)
    
    def plot(self, ax: plt.Axes = None, title='', save = False, legend=True, legendSize=14, text='', ratio=1.6181, ymin=None, ymax=None):
        """Plot free energy diagram without energy barrier"""
        if not ax:
            figFree = plt.figure(figsize=(8, 6), dpi = 300)
            axFree = figFree.add_subplot(111)
        else:
            axFree = ax
            # self.fig = ax.figure
        pos = self.diagram.plot(xtickslabel = self.step_names, stepLens=len(self.step_names), ax=axFree, ratio=ratio, ymin=ymin, ymax=ymax) # this is the default ylabel
        # axFree.set_zorder(ax2.get_zorder()+1)
        if ymin != None and ymax != None:
            plt.ylim(ymin, ymax)
        
        # add legend
        for i, specis in enumerate(self.obser_names):
            try:
                plt.hlines(0.1, pos[0], pos[0], color=self.ColorDict[specis], label= specis)
            except:
                plt.hlines(0.1, pos[0], pos[0], color=self.DefaultColor[i], label= specis)
        if legend == True:
            plt.legend(fontsize=legendSize)
        plt.title(title, fontsize=14)
        plt.text(0.04, 0.93, text, horizontalalignment='left', verticalalignment='center', transform=axFree.transAxes, fontsize=14, fontweight='bold')        
        axFree.yaxis.set_label_coords(-0.1, 0.6)
        axFree.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # save figure
        if save == True: 
            plt.show()
            figFree.savefig(self.fig_name, dpi=300, bbox_inches='tight')
        # print('initial x pos:', pos[0])
        # return figFree
        return pos[0], pos[-1]
        
class CO2RRFED(CO2RRFEDplot):
    """New version of CO2RR free energy diagram using panda, and thus less varalbe would be used"""
    def __init__(self, df, fig_name):
        self.step_names = df.columns
        self.obser_names = df.index
        self.X = df.values
        self.fig_name = fig_name
        super().__init__(self.step_names, self.obser_names, self.X, self.fig_name, info=False)
        logging.debug(f'Loaded free energy table: \n {df}')

class HERFEDplot:
    """Class for HER free energy diagram without TS data
    
    Parameters:
        
    step_names: str
        names of columns (steps)
    obser_names: str
        names of rows (observations)
    X: float
        data for HER free energy plot
    fig_name: str
        figure name
    """
    def __init__(self, step_names, obser_names, X, fig_name, info=True):
        # plot parameters
        self.step_names = step_names
        self.obser_names = obser_names
        self.X = X
        self.fig_name = fig_name
        self.ColorDict = ColorDict
        self.DefaultColor=cm.rainbow(np.linspace(0,1,len(self.obser_names)))
        self.diagram = FED()
        if info == True:
            logging.debug(f'loaded step_names:\n {self.step_names}')
            logging.debug(f'loaded obser_names:\n {self.obser_names}')
            logging.debug(f'loaded data:\n {self.X}')
        count = 0
        for i, specis in enumerate(self.obser_names):
            for step in range(len(self.step_names)):
                count += 1
                if step == 0:
                    self.diagram.pos_number = 0
                try:
                    self.diagram.add_level(self.X[i][step], color = self.ColorDict[specis])
                except:
                    self.diagram.add_level(self.X[i][step], color = self.DefaultColor[i])
        
                if count % (len(self.step_names)) != 0:
                    try:
                        self.diagram.add_link(count-1, count, color = self.ColorDict[specis])
                    except:
                        self.diagram.add_link(count-1, count, color = self.DefaultColor[i])
        
    def plot(self, ax: plt.Axes = None, title='', save=False, legend=True, legendSize=14,text='', ratio=1.6181, **kwargs):
        if not ax:
            figFree = plt.figure(figsize=(8, 6), dpi=300)
            axFree = figFree.add_subplot(111)
        else:
            axFree = ax
            
        pos = self.diagram.plot(xtickslabel = self.step_names, stepLens=len(self.step_names), ax=axFree, ratio=ratio) 
        
        # add legend    
        for i, specis in enumerate(self.obser_names):
            try:
                plt.hlines(0.1, pos[0], pos[0], color=self.ColorDict[specis], label=specis)
            except:
                plt.hlines(0.1, pos[0], pos[0], color=self.DefaultColor[i], label=specis)
        if legend == True:
            plt.legend(fontsize=legendSize)
        plt.title(title, fontsize=14)
        plt.text(0.04, 0.93, text, horizontalalignment='left', verticalalignment='center', transform=axFree.transAxes, fontsize=14, fontweight='bold')        
        axFree.yaxis.set_label_coords(-0.1, 0.5)
        axFree.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        if save:
            plt.show()
            figFree.savefig(self.fig_name, dpi=300, bbox_inches='tight')
        # print('initial x pos:', pos[0])
        # return figFree
        return pos[0], pos[-1]

class HERFED(HERFEDplot):
    """New version of HER free energy diagram using panda, and thus less varalbe would be used"""
    def __init__(self, df, fig_name):
        self.step_names = df.columns
        self.obser_names = df.index
        self.X = df.values
        self.fig_name = fig_name
        super().__init__(self.step_names, self.obser_names, self.X, self.fig_name, info=False)
        logging.debug(f'Loaded free energy table: \n {df}')
        
class CO2RRFED_with_TS:
    """Class for CO2RR free energy diagram with TS data
    
    Parameters:
        
    step_names: str
        names of columns (steps)
    obser_names: str
        names of rows (observations)
    X: float
        data for CO2RR free energy plot
    fig_name: str
        figure name
    """
    def __init__(self, step_names, obser_names, X, fig_name, info=True):
        # plot parameters
        self.step_names = step_names
        self.obser_names = obser_names
        self.X = X
        self.fig_name = fig_name
        
        self.axFree = None
        self.figFree = None

        # print('auto loaded step_names: ', self.step_names)
        # print('auto loaded obser_names: ', self.obser_names)
        # print('auto loaded data: \n', self.X)
        
        if info == True:
            logging.debug(f'loaded step_names:\n {self.step_names}')
            logging.debug(f'loaded obser_names:\n {self.obser_names}')
            logging.debug(f'loaded data:\n {self.X}')
        
        delNames = []
        for i in range(int(len(self.step_names))):
            if i % 2 != 0:
                delNames.append(self.step_names[i])
        self.realSteps = list(set(self.step_names).difference(set(delNames)))
        
        self.ColorDict = ['k', 'lime', 'r', 'b', 'darkcyan', 'cyan', 'olive', 'magenta', 'pink', 'gray', 'orange', 'purple', 'g']
        # ColorDict = ['gray', 'brown', 'orange', 'olive', 'green', 'cyan', 'blue', 'purple', 'pink', 'red']
        # ColorDict = ['k', 'g', 'r', 'b', 'c', 'm', 'y', 'brown', 'pink', 'gray', 'orange', 'purple', 'olive']
        # self.realSteps = ['* + CO2', '*HOCO', '*CO', '* + CO']  #reload step name for CO2RR
        # realSteps = ['* + $H^+$', '*H', '* + 1/2$H_2$',]  #reload step name for HER
        # self.obser_names = ["Pure", "Ni", "Co", "V", "Cr", "Mn", "Fe", "Pt"]  #reload specis name

        print('reload:', self.realSteps)
        print('reload:', self.obser_names, '\n')
        
        self.diagram = FED()
        count = 0
        for specis in range(len(self.obser_names)):
            for step in range(len(self.realSteps)):
                count += 1
                if step == 0:
                    self.diagram.pos_number = 0
                
                energy_col = 2 * step        
                self.diagram.add_level(self.X[specis][energy_col], color = self.ColorDict[specis])
        
                if count % (len(self.realSteps)) != 0:
                    if self.X[specis][energy_col+1] == 0: # plot general link line if TS energy is equle to 0
                        self.diagram.add_link(count-1, count, color = self.ColorDict[specis])
                        # print(count-1)
                    else: # plot ts barrier
                        self.diagram.add_barrier(start_level_id=count-1, barrier=self.X[specis][energy_col+1]+self.X[specis][energy_col], end_level_id=count, color = self.ColorDict[specis]) #add energy TS barriers
    
    def add_link(self, start_id=None, end_id=None, color='k', linestyle='--', linewidth=1):
        """Add dashed link line between levels"""
        if start_id != None and end_id != None:  # pos starts from 0
            self.diagram.add_link(start_id, end_id, color, linestyle, linewidth)
    
    def remove_link(self, start_id=None, end_id=None):
        """Remove dashed link line between levels"""
        if start_id != None and end_id != None:
            self.diagram.remove_link(start_id, end_id)
            
    def add_ts(self, start_id, barrier, end_id, color='k', linestyle='--', linewidth=1):
        """Add transition state energy barrier curve between levels"""
        if start_id != None and end_id != None:
            specis = int(start_id) // len(self.realSteps)
            energy_col = 2 * int(start_id) % len(self.realSteps)
            real_barrier = self.X[specis][energy_col] + barrier
            self.diagram.add_barrier(start_id, real_barrier, end_id, color, linestyle, linewidth)
    
    def remove_ts(self, start_id=None, end_id=None):
        """Remove transition state energy barrier curve between levels"""
        if start_id != None and end_id != None:
            self.diagram.remove_barrier(start_id, end_id)
        
    def plot(self, save=False, title=''):
        """Plot free energy diagram with energy barrier"""
        figFree = plt.figure(figsize=(8,6), dpi = 300)
        axFree = figFree.add_subplot(111)                 
        # diagram.add_barrier(start_level_id=0, barrier=2, end_level_id=1) # add energy barriers
        # diagram.plot(xtickslabel = self.step_names, stepLens=len(self.step_names), ax=axFree) # this is the default ylabel
        pos = self.diagram.plot(xtickslabel = self.realSteps, stepLens=len(self.realSteps), ax=axFree) # this is the default ylabel
        # add legend
        for specis in range(len(self.obser_names)):
            plt.hlines(0.1, pos[0], pos[0], color=self.ColorDict[specis], label= self.obser_names[specis])
        plt.legend(fontsize=12)
        plt.title(title, fontsize=14)
        # xmin = min([min(axFree.lines[i].get_xdata()) for i in range(len(axFree.lines))])
        # xmax = max([max(axFree.lines[i].get_xdata()) for i in range(len(axFree.lines))])
        # axFree.set_xlim([xmin, xmax])
        
        plt.show()
        if save == True:
            figFree.savefig(self.fig_name)

class CO2RRFED_TS(CO2RRFED_with_TS):
    """New version of CO2RR free energy diagram using panda, and thus less varalbe would be used"""
    def __init__(self, df, fig_name):
        self.step_names = df.columns
        self.obser_names = df.index
        self.X = df.values
        self.fig_name = fig_name
        super().__init__(self.step_names, self.obser_names, self.X, self.fig_name, info=False)
        logging.debug(f'Loaded free energy table: \n {df}')