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

def subscript_chemical_formula(formula):
    """Return a formuala with subscript numbers"""
    if_exist_a_num = any(i.isdigit() for i in formula)
    if if_exist_a_num:
        subscript = formula.split('+')[0]
        sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        formula_sub = subscript.translate(sub) + formula.replace(formula.split('+')[0], '')
    else:
        formula_sub = formula
    return formula_sub

def subscript_chemical_formulas(formulas):
    """Return formualas with subscript numbers"""
    formulas_sub = []
    for formala in formulas:
        formula_sub = subscript_chemical_formula(formala)
        formulas_sub.append(formula_sub)
    return formulas_sub
    
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
        
    
    def plot(self, title='', save = False, ymin=None, ymax=None,xlabel='Doping elements', tune_tex_pos=1.5, legend=True, tune_ano_pos=None, **kwargs):
        """Plot formation energy"""
        
        color_list = ['k', 'lime', 'r', 'b', 'darkcyan', 'cyan', 'olive', 'magenta', 'pink', 'gray', 'orange', 'purple', 'g']
        
        fig = plt.figure(figsize=(8, 6), dpi=300)
        x = np.arange(0,len(self.obser_names),1)
        
        if 'fontsize' in kwargs:
            fontsize = kwargs['fontsize']
        else:
            fontsize = 12
        for j in range(len(self.type_names)):  
            y = self.X[:,j]
            plt.plot(x, y, 'o', color=color_list[j])  # plot dots
            if tune_ano_pos!=None:
                for obser_name in tune_ano_pos.keys():
                    i = list(self.obser_names).index(obser_name)
                    x_bias = tune_ano_pos[obser_name][0]
                    y_bias = tune_ano_pos[obser_name][1]
                    defultx = -1.
                    defulty = 0.4
                    # plt.text(x[i]+x_bias, y[i]+defulty+y_bias, obser_name, fontsize=12, rotation = 90, \
                    #          horizontalalignment='center', verticalalignment='bottom', \
                    #          arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),color='black',zorder=10)
                    if 'subscritpt' in kwargs and kwargs['subscritpt']==True:
                        obser_name = subscript_chemical_formula(obser_name)
                    plt.annotate(obser_name,xy=(x[i], y[i]), xycoords='data', rotation=90,
                    xytext=(x[i]+defultx+x_bias, y[i]+defulty+y_bias), textcoords='data', fontsize=fontsize,
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3",color='black'))
        print(kwargs)
        if legend:
            plt.legend(self.type_names, framealpha=0.5, fontsize=12,  edgecolor='grey')
            
        plt.axhline(y=0, color='r', linestyle='--')
        # plt.xlim([-0.3, 23.5])
        plt.ylim([-2.5, 2.5])
        plt.xlabel(xlabel, fontsize=16, labelpad=15)
        plt.ylabel('$\Delta G_{HOCO*}$-$\Delta G_{H*}$', fontsize=16)
        ax = fig.gca()
        if False:
            ax.set_xticks(x)
            ax.set_xticklabels(self.obser_names)
        else:
            ax.set_xticks([])
    
        ax.tick_params(labelsize=13.2) # tick label font size
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.2) # linewith of frame
        
        max_x = len(self.obser_names)
        if 'head_width' in kwargs:
            head_width = kwargs['head_width']
        else:
            head_width = 2.8
        if 'width' in kwargs:
            width = kwargs['width']
        else:
            width = 1.4
        plt.arrow(x=max_x, y=0, dx=0, dy=2, width=width, head_width=head_width, head_length=0.3, color='grey') 
        plt.annotate('HER', xy = (max_x-tune_tex_pos, 1), rotation=90, fontsize=14, color='grey')
        
        plt.arrow(x=max_x, y=0, dx=0, dy=-2, width=width, head_width=head_width, head_length=0.3, color='grey') 
        plt.annotate('CO$_2$RR', xy = (max_x-tune_tex_pos, -1.5), rotation=90, fontsize=14, color='grey')
        
        plt.show()
        
        if save:
            fig.savefig(self.fig_name, dpi=300, bbox_inches='tight')
        return fig, ax
            
class Selectivity(SelectivityPlot):
    """New version of CO2RR selectivity using panda, and thus less varalbe would be used"""
    def __init__(self, df, fig_name):
        self.type_names = df.columns
        self.obser_names = df.index
        self.X = df.values
        self.fig_name = fig_name
        super().__init__(self.type_names, self.obser_names, self.X, self.fig_name)
        logging.debug(f'loaded selectivity table: \n {df}')