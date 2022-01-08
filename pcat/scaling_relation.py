# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 01:06:31 2021

@author: changai
"""
# import sys
# sys.path.append("../../../")

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from pcat.utils.styles import ColorDict
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')
logging.getLogger('matplotlib.font_manager').disabled = True

class ScalingRelationPlot:
    """Class for scaling relations
    
    Parameters:
        
    descriper1: list, float
        data for x axis
    descriper2: list, float
        data for y axis
    obser_names: list, str
        observation name
    fig_name: str
        figure name
    DefaultColor: list
        default color of curve
    """
    def __init__(self, descriper1, descriper2, obser_names, fig_name):
        # plot parameters
        self.descriper1 = descriper1
        self.descriper2 = descriper2
        self.obser_names = obser_names
        self.fig_name = fig_name
        self.ColorDict = ColorDict
        
    def plot(self, ax: plt.Axes = None, 
             dot_color='black', 
             line_color='red', 
             save = False, 
             xlabel='*HOCO', 
             ylabel='*CO', 
             title='', 
             text='',
             color_ditc=False):
        
        # plot data points
        if not ax:
            fig = plt.figure(figsize=(8, 6), dpi = 300)
            ax = fig.add_subplot(111)
        # Otherwise register the axes and figure the user passed.
        else:
            self.ax = ax
            # self.fig = ax.figure

        # fig = plt.figure(figsize=(8, 6), dpi = 300)
        # plt.plot(self.descriper1, self.descriper2, 's', color='black')  #plot dots
        
        for i, name in enumerate(self.obser_names):
            if color_ditc==False:
                self.ColorDict[name] = dot_color
            try:
                plt.plot(self.descriper1[i], self.descriper2[i], 's', color=self.ColorDict[name])  # plot dots
                plt.annotate(name, (self.descriper1[i], self.descriper2[i]+0.005), color=self.ColorDict[name], fontsize=14, horizontalalignment='center', verticalalignment='bottom')
            except:
                plt.plot(self.descriper1[i], self.descriper2[i], 's', color=dot_color)  # plot dots
                plt.annotate(name, (self.descriper1[i], self.descriper2[i]+0.005), color=dot_color, fontsize=14, horizontalalignment='center', verticalalignment='bottom')

        # add element tags
        if False: #detached
            if isinstance(dot_color, dict)==True:
                for i, name in enumerate(self.obser_names):
                    plt.plot(self.descriper1[i], self.descriper2[i], 's', color=dot_color[name]) 
                    plt.annotate(name, (self.descriper1[i], self.descriper2[i]+0.005), color=dot_color[name], fontsize=14, horizontalalignment='center', verticalalignment='bottom')
            else:
                plt.plot(self.descriper1, self.descriper2, 's', color=dot_color)
                for i, name in enumerate(self.obser_names):
                    plt.annotate(name, (self.descriper1[i], self.descriper2[i]+0.005), color=dot_color, fontsize=14, horizontalalignment='center', verticalalignment='bottom')
                    

        # plt.plot(self.descriper1, self.descriper2, 's', color=dot_color)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        ax.yaxis.set_label_coords(-0.12, 0.5)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.margins(y=0.08)
        plt.title(title, fontsize=14)
        plt.text(0.05, 0.93, text, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=14, fontweight='bold')        
        
        # get current axis object and change format
        # ax = fig.gca()
        ax.tick_params(labelsize=12) #tick label font size
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.2) #linewith of frame
        
        # linear fiting and plot linear line
        m, b = np.polyfit(self.descriper1, self.descriper2, 1)
        handleFit = plt.plot(self.descriper1, m * self.descriper1 + b, linewidth=2, color=line_color)
        
        # add r2 tag
        from sklearn.metrics import r2_score
        model = np.array([m, b])
        predict = np.poly1d(model)
        r2 = r2_score(self.descriper2, predict(self.descriper1))
        r2 = np.round(r2, 2)
        m = np.round(m, 2)
        b = np.round(b, 2)
        # plt.text(0.85, 0.3, 'R2 = {}'.format(r2), fontsize=14)
        plt.legend(handles = handleFit, labelcolor=line_color, labels = ['$R^2$ = {}\ny = {} + {} * x '.format(r2, b, m)], loc="lower right", handlelength=0, fontsize=14)
        logging.debug(f'r2: {r2}')
        # print('r2:', r2)
        
        # save figure
        if save == True: 
            plt.show()
            fig.savefig(self.fig_name)
        # return fig
        
class ScalingRelation(ScalingRelationPlot):
    """New version of CO2RR scaling relation using panda, and thus less varalbe would be used"""
    def __init__(self, df,descriper1, descriper2, fig_name):
        self.descriper1 = df[descriper1]
        self.descriper2 = df[descriper2]
        self.obser_names = df.index
        self.fig_name = fig_name
        super().__init__(self.descriper1, self.descriper2, self.obser_names, self.fig_name)
        logging.debug(f'loaded scaling relation table: \n {df}')