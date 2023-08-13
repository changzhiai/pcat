# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 17:08:44 2022

@author: changai
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
from pcat.lib.kinetic_model import CO2toCO
from matplotlib import rc
import pcat.utils.constants as const
from pcat.lib.io import pd_read_excel
from pcat.utils.styles import ColorDict
import pandas as pd
import logging
logging.basicConfig(level=logging.DEBUG, format='\n(%(asctime)s) \n%(message)s')
logging.getLogger('matplotlib.font_manager').disabled = True

# rc('font', **{'family':'sans-serif','sans-serif':['Helvetica'], 'size':8})

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

class Activity:
    """Activity calculation of CO2 to CO in acid environment
    
    BEEF-vdw correction
    
    Parameters
    
    descriper1: str
        one column for x axis
    descriper1: str
        one column for y axis
    fig_name: str
        figure name to be saved
    U0: float
        applied potential (v)
    T0: float
        applied temperature (K)
    pCO2g: float
        partial pressure for CO2 gas
    pCOg: float
        partial pressure for CO gas
    pH2Og: float
        partial pressure for H2O gas
    Gact*: float
        activative free energy
    cHp0: float
        concentration of H proton affected by pH, default pH is 0
    UHE0 or URHE0: float
        potential shifted by pH or concentraion of H proton
    nu_e: float
        pre-exponential factor for k1 and k2 (depatched)
    ne_c: float 
        common pre-exponential factor for k3
    A_prior: float
        pre-exponential factor used here for k1 and k2
        
    """
    
    
    def __init__(self, df, 
                 descriper1 = 'E(*CO)',
                 descriper2 = 'E(*HOCO)', 
                 fig_name = 'activity.jpg',
                 U0=-0.3, 
                 T0=297.15, 
                 pCO2g = 1., 
                 pCOg=0.005562, 
                 pH2Og = 1., 
                 cHp0 = 10.**(-0.), 
                 Gact=0.2, 
                 p_factor = 3.6 * 10**4):
        
        global G_H2g
        global G_CO2g
        global G_H2Og
        global G_COg
        global kB
        global hplanck
        global ddG_HOCO
        global ddG_CO
        
        G_H2g = const.G_H2g
        G_CO2g = const.G_CO2g
        G_H2Og = const.G_H2Og
        G_COg = const.G_COg
        
        kB = const.kB # Boltzmann constant in eV/K
        hplanck = const.hplanck # eV s
        
        ddG_HOCO = const.ddG_HOCO
        ddG_CO =  const.ddG_CO
        """
        ddG_HOCO explanation:
            ddG_HOCO is the correction for the whole first equations, not for only *HOCO
            CO2(g) + * + 1/2 H2(g) -> HOCO*
            
            ddG_HOCO = Gcor_HOCO - Gcor_CO2g - 0 - 0.5 * Gcor_H2g 
            >> 0.41400000000000003
        
        ddG_CO explanation:
            ddG_CO is the correction for the whole CO binding equations, not for the second equations
            * + CO(g) -> CO*
            
            ddG_CO = Gcor_CO - 0 - Gcor_COg
            >> 0.45600000000000007
        """
        
        self.df = df
        self.descriper1 = descriper1
        self.descriper2 = descriper2   
        self.fig_name = fig_name
        
        global UHER0, URHE0
        global A_act1, A_act2
        global G_1act_cap, G_2act_cap
        global A_prior
        UHER0 = URHE0 = kB * T0 * np.log(cHp0)   # introduced to shift the plotted potential window to the relevant range w
        Gact1 = Gact2 = Gact # activative free energy 0.475
        A_act1 = np.exp( - Gact1 / ( kB * T0 ) ) # 
        A_act2 = np.exp( - Gact2 / ( kB * T0 ) ) # electrochemical prefactor, fitting
        G_1act_cap = -Gact1
        G_2act_cap = -Gact2
        A_prior = p_factor
        
        # U0 = applied_p # applied potential vs. she
        self.T0 = T0 # 297.15, higher 370
        self.U = U0 + UHER0
        self.pCO2g = pCO2g
        self.pCOg =  pCOg
        self.pH2Og = pH2Og
        self.cHp = cHp0 #1.
        self.nu_e = kB * T0 / hplanck
        self.nu_c = 1.e13
        
    def get_K1(self, Eb_HOCO, U, T):
        """ K1 using HOCO binding
        
        CO2(g) + * + 1/2 H2(g) -> HOCO*
        dG1 = G_HOCO* - G_CO2g - G* - 1/2 G_H2g = Eb_HOCO + ddG_HOCO
        """
        beta = 1. / (kB * T) 
        dG = Eb_HOCO + ddG_HOCO
        K1 = exp( - (dG + 1.0 * U ) * beta )
        return K1
    
    def get_K2(self, Eb_HOCO, Eb_CO, U,  T):
        """ K2 using HOCO and CO binding.
        
        HOCO* + 1/2 H2(g) -> CO* + H2O(g)
        dG2 = G_CO* + G_H2Og - G_HOCO* - 1/2 G_H2g
            = (G_CO* - G* -G_COg + G* + G_COg)
              + G_H2Og 
              - (G_HOCO* - G_CO2g - G* - 1/2 G_H2g + G_CO2g + G* + 1/2 G_H2g)
              - 1/2 G_H2g
            = Gb_CO + G* + G_COg
              + G_H2Og 
              - (Gb_HOCO + G_CO2g + G* + 1/2 G_H2g)
              - 1/2 G_H2g
            = Gb_CO - Gb_HOCO + G_COg + G_H2Og - G_CO2g - G_H2g
            = Eb_CO + ddG_CO - Eb_HOCO - ddG_HOCO - G_CO2g - G_H2g + G_H2Og + G_COg
            
        """
        beta = 1. / (kB * T) 
        dG = Eb_CO + ddG_CO - Eb_HOCO - ddG_HOCO - G_CO2g - G_H2g + G_H2Og + G_COg
        K2 =  exp( - ( dG + 1.0 * U ) * beta ) 
        return K2
    
    def get_K3(self, Eb_CO, U, T):
        """ K3 asumming scaling.
        CO* -> CO(g) + *
        dG3 = G_COg + G* - G_CO* = - Gb_CO
        """
        beta = 1. / (kB * T) 
        dG = - (Eb_CO + ddG_CO)
        K3 = exp( - dG * beta )
        return K3
    
    def verify_BE2FE(self):
        """Verify if it is right from binding energy to free energy of three steps
        Eb_HOCO, ddG_HOCO, Eb_CO, ddG_CO, G_CO2g, G_H2g, G_H2Og, G_COg"""
        Eb_CO = (self.df[self.descriper1]).values
        Eb_HOCO = (self.df[self.descriper2]).values
        obser_names = (self.df.index).values
        G1 = Eb_HOCO + ddG_HOCO
        G2 = Eb_CO + ddG_CO - Eb_HOCO - ddG_HOCO - G_CO2g - G_H2g + G_H2Og + G_COg
        G3 = - (Eb_CO + ddG_CO)
        tuples = {'Surface': obser_names,
                  'G_step1': G1,
                  'G_step2': G1+G2,
                  'G_step3': G1+G2+G3,
                  }
        df_all_FE = pd.DataFrame(tuples)
        logging.debug(f'Transform binding energy to free energy of three steps:\n{df_all_FE}')
    
    def get_k1(self, nu, Eb_HOCO, U, T, tc):
        """ k1 using HOCO binding (vs CO2 and H2)
        """
        beta = 1. / (kB * T) 
        dG_rhe = Eb_HOCO + ddG_HOCO # vs. RHE
        Urev_rhe = -dG_rhe
        # dG_she = dG_rhe 
        # Urev_she = -dG_she + UHER0
        k1 = A_prior * exp( - max( ( U - Urev_rhe ) * tc, G_1act_cap) * beta ) 
        # k1 = nu * A_act1 * exp( - max( ( U - Urev_rhe ) * tc, G_1act_cap) * beta ) 
        return k1
    
    def get_k2(self, nu, Eb_HOCO, Eb_CO, U, T, tc):
        """ k2 using HOCO and CO energies.
        """    
        beta = 1. / (kB * T)  
        dG_rhe = Eb_CO + ddG_CO - Eb_HOCO - ddG_HOCO - G_CO2g - G_H2g + G_H2Og + G_COg
        Urev_rhe = -dG_rhe
        # dG_she = dG_rhe
        # Urev_she = - dG_she + URHE0
        k2 = A_prior * exp( - max(( U - Urev_rhe ) * tc, G_2act_cap) * beta ) 
        # k2 = nu * A_act2 * exp( - max(( U - Urev_rhe ) * tc, G_2act_cap) * beta ) 
        return k2
    
    def get_k3(self, nu, Eb_CO, U, T, tc):
        """ k3 assuming scaling.
        """
        beta = 1. / (kB * T) 
        dE = - Eb_CO
        dE = max(dE,0)
        k3 = nu * exp( - dE * beta )
        return k3
        
        
    def get_rates(self, nu_e, nu_c, Eb_HOCO, Eb_CO, U, T, tc1, tc2, tc0):
        """ Returns rate constants and equilibirum constants,
        """
        K1 = self.get_K1(Eb_HOCO, U, T=T)
        K2 = self.get_K2(Eb_HOCO, Eb_CO, U, T=T)
        K3 = self.get_K3(Eb_CO, U, T=T)
        k1 = self.get_k1(nu_e, Eb_HOCO, U, T=T, tc=tc1)
        k2 = self.get_k2(nu_e, Eb_HOCO, Eb_CO, U, T=T, tc=tc2)
        k3 = self.get_k3(nu_c, Eb_CO, U, T=T, tc=tc0)
        return k1, K1, k2, K2, k3, K3
    
    def plot_scaling_rev(self, Eb_CO_model, Eb_HOCO_model, xlim, ylim, text=True, tune_tex_pos=None, scaling=False, ColorDict=ColorDict, **kwargs):
        """Plot scaling relation but slope is inverse in order to correspond to previous scaling relation
        tune_tex_pos = {'Pd64H31': [0.1, 0.2], 'Pd64H39': [0.3, 0.4]}"""
        
        Eb_CO_d = (self.df[self.descriper1]).values
        Eb_HOCO_d = (self.df[self.descriper2]).values
        obser_names = (self.df.index).values
        if 'fontsize' in kwargs:
            fontsize = kwargs['fontsize']
        else:
            fontsize = 10
        # fontsize = 10
        # default_bias = 0.05
        default_bias = 0.02
        # xlim = [-1, 1]
        # ylim = [-2, 1]
        for i,obser_name in enumerate(obser_names):
            try:
                color=ColorDict[obser_name]
                if Eb_CO_d[i] >= min(xlim) and Eb_CO_d[i] <= max(xlim) and Eb_HOCO_d[i] >= min(ylim) and Eb_HOCO_d[i] <= max(ylim):
                    plt.plot(Eb_CO_d[i], Eb_HOCO_d[i], 'o', color=color) 
                    # plt.text would fail xlim and ylim 
                    if tune_tex_pos==None:
                        if 'subscritpt' in kwargs and kwargs['subscritpt']==True:
                            obser_name = subscript_chemical_formula(obser_name)
                        plt.text(Eb_CO_d[i], Eb_HOCO_d[i]+default_bias, obser_name, fontsize=fontsize, \
                                 horizontalalignment='center', verticalalignment='bottom', color=color,zorder=10)
                    else:
                        names = tune_tex_pos.keys()
                        if obser_name in names:
                            pos_tune = tune_tex_pos[obser_name]
                            tune_x = pos_tune[0]
                            tune_y = pos_tune[1]
                        else:
                            tune_x, tune_y = 0, 0
                        if tune_x == 0 and tune_y == 0:
                            plt.text(Eb_CO_d[i]+tune_x, Eb_HOCO_d[i]+default_bias+tune_y, obser_name, fontsize=fontsize, \
                                      horizontalalignment='center', verticalalignment='bottom', color=color,zorder=10)
                        else:
                            if 'subscritpt' in kwargs and kwargs['subscritpt']==True:
                                obser_name = subscript_chemical_formula(obser_name)
                            plt.annotate(obser_name,xy=(Eb_CO_d[i], Eb_HOCO_d[i]), xycoords='data',
                            xytext=(Eb_CO_d[i]+tune_x, Eb_HOCO_d[i]+default_bias+tune_y), textcoords='data', fontsize=fontsize,
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3",color=color), color=color)
            except:
                if Eb_CO_d[i] >= min(xlim) and Eb_CO_d[i] <= max(xlim) and Eb_HOCO_d[i] >= min(ylim) and Eb_HOCO_d[i] <= max(ylim):
                    plt.plot(Eb_CO_d[i], Eb_HOCO_d[i], 'o', color='white')
                    if text:
                        if tune_tex_pos==None:
                            plt.text(Eb_CO_d[i], Eb_HOCO_d[i]+default_bias, obser_name, fontsize=fontsize, \
                                     horizontalalignment='center', verticalalignment='bottom', color='white')
                        else:
                            names = tune_tex_pos.keys()
                            if obser_name in names:
                                pos_tune = tune_tex_pos[obser_name]
                                tune_x = pos_tune[0]
                                tune_y = pos_tune[1]
                            else:
                                tune_x, tune_y = 0, 0
                            if 'subscritpt' in kwargs and kwargs['subscritpt']==True:
                                obser_name = subscript_chemical_formula(obser_name)
                            plt.text(Eb_CO_d[i]+tune_x, Eb_HOCO_d[i]+default_bias+tune_y, obser_name, fontsize=12, \
                                     horizontalalignment='center', verticalalignment='bottom', color='white')
        if scaling:
            m, b = np.polyfit(Eb_HOCO_d, Eb_CO_d, 1)
            plt.axline(( Eb_CO_d[0], Eb_CO_d[0]/m-b/m), slope=1/m, color='white')
            # plt.plot(self.descriper2, m * self.descriper2 + b, linewidth=2, color=linecolor)   
        
    
    def plot(self, save=True, Eb_CO_d=None, Eb_HOCO_d=None, TOF_to_j=47.96, title='', subtitle='', \
             xlim=None, ylim=None, text=True, tune_tex_pos=None, ColorDict=ColorDict, **kwargs):
        """
        Set range, for example, Eb_CO_d=[-2,0.3], Eb_HOCO_d=[-1,1.3]
        """
        """Old version
        global T0
        global tc0, tc1, tc2
        global A_act1, A_act2
        global G_1act_cap, G_2act_cap
        global A_prior
        global UHER0, URHE0
        
        U0 = applied_p # applied potential vs. she
        T0 = temp # 297.15, higher 370
        UHER0 = URHE0 = kB * T0 * np.log(cHp0)   # introduced to shift the plotted potential window to the relevant range w
        U = U0 + UHER0
        
        pCO2g = pCO2g
        pCOg =  pCOg
        pH2Og = pH2Og
        cHp = cHp0 #1.
        
        nu_e = kB * T0 / hplanck
        nu_c = 1.e13
        Gact1 = Gact2 = Gact # activative free energy 0.475
        A_act1 = np.exp( - Gact1 / ( kB * T0 ) ) # 
        A_act2 = np.exp( - Gact2 / ( kB * T0 ) ) # electrochemical prefactor, fitting
        G_1act_cap = -Gact1
        G_2act_cap = -Gact2
        
        tc0 = tc1 = tc2 = 0.5 # transfer coefficiency
        A_prior = p_factor
        
        Parameters：
        
        xlim: list, float
            a range of x axis
        ylim: list, float
            a range of y axis
        """
        
        T0 = self.T0
        U = self.U
        pCO2g = self.pCO2g
        pCOg = self.pCOg
        pH2Og = self.pH2Og
        cHp = self.cHp
        nu_e = self.nu_e
        nu_c = self.nu_c
        
        tc0 = tc1 = tc2 = 0.5 # transfer coefficiency
        TOF_to_j = TOF_to_j
        
        N, M = 20*4, 20*4
        R = np.empty([M,N])
        Thetas = np.empty([M,N,3])
        
        if Eb_CO_d == None:
            Eb_CO_d = (self.df[self.descriper1]).values
        if Eb_HOCO_d == None:
            Eb_HOCO_d = (self.df[self.descriper2]).values
            
        Eb_CO_model = np.linspace(min(Eb_CO_d)-0.2, max(Eb_CO_d)+0.2, N)
        Eb_HOCO_model = np.linspace(min(Eb_HOCO_d)-0.1, max(Eb_HOCO_d)+0.1, M)
        
        jmax = 10.0e3 # exptl current plateau's at 10 mA/cm2 
        jmin = 0.1
        for j, Eb_CO in enumerate(Eb_CO_model):
            for i, Eb_HOCO in enumerate(Eb_HOCO_model):
                k1, K1, k2, K2, k3, K3 = self.get_rates(nu_e, nu_c, Eb_HOCO, Eb_CO, U, T0, tc1, tc2, tc0)
                rm = CO2toCO(pCO2g, pCOg, pH2Og, cHp, k1, K1, k2, K2, k3, K3)
                # rm = CO2toCO(pCO2g, pCOg, pH2Og, cOHm, k1, K1, k2, K2, k3, K3, T0)
                thetas, rates = rm.solve()
                # print(rates)
                rate = min(jmax, rates[0])
                rate = max(jmin, rate)
                R[i,j] = np.log10(rate*TOF_to_j) # TOF to current
                Thetas[i,j,:] = thetas
        
        fig = plt.figure(figsize=(9, 6), dpi = 150)
        ax = plt.gca()
        contours = np.linspace(np.log10(jmin*TOF_to_j), np.log10(jmax*TOF_to_j), 11) 
        plt.contourf(Eb_CO_model, Eb_HOCO_model, R, contours, cmap=plt.cm.jet) # plot countour
        bar = plt.colorbar(ticks=np.arange(min(contours), max(contours), 0.5))
        bar.ax.tick_params(labelsize=10)
        bar.set_label(r'log$_{10}$(j/$\mu$Acm$^{-2}$)', fontsize=14,)
        
        
        if xlim == None:
            xlim = [min(Eb_CO_model), max(Eb_CO_model)]
        if ylim == None:
            ylim = [min(Eb_HOCO_model), max(Eb_HOCO_model)]
        self.plot_scaling_rev(Eb_CO_model, Eb_HOCO_model, xlim, ylim,text=text, tune_tex_pos=tune_tex_pos, ColorDict=ColorDict, **kwargs)
        
        plt.tick_params(labelsize=12) # tick label font size
        plt.title(title, fontsize=14,)
        plt.text(0.05, 0.93, subtitle, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='white', fontweight='bold')        
        plt.xlabel(r'$E_{\mathrm{*CO}}$ (eV)', fontsize=14,)
        plt.ylabel(r'$E_{\mathrm{*HOCO}}$ (eV)', fontsize=14,)
        
        """add figure index in front of figure"""
        # import string
        # ax.text(-0.17, 0.97, string.ascii_lowercase[index], transform=ax.transAxes, size=20, weight='bold')
        
        plt.xlim(xlim)
        plt.ylim(ylim)
        fig.tight_layout()
        plt.show()
        
        if save == True:
            fig.savefig(self.fig_name, dpi=300, bbox_inches='tight')    
        