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
import pcat.utils.constants as cons
from pcat.lib.io import pd_read_excel
# rc('font', **{'family':'sans-serif','sans-serif':['Helvetica'], 'size':8})

class Activity:
    """Activity calculation of CO2 to CO in acid environment
    
    BEEF-vdw correction
    
    Gact*: float
        activative free energy
    cHp0: float
        concentration of H proton affected by pH, default pH is 0
    UHE0 or URHE0: float
        potential shifted by pH or concentraion of H proton
    nu_e: float
        pre-exponential factor
    ne_c: float 
        common pre-exponential factor
    A_prior: float
        pre-exponential factor used here
        
    """
    
    
    def __init__(self, df, descriper1 = '*HOCO', descriper2 = '*CO', 
                 U0=-0.3, T0=297.15, pCO2g = 1., pCOg=0.005562, pH2Og = 1., cHp0 = 10.**(-0.), Gact=0.2, p_factor = 3.6 * 10**4):
        
        global G_H2g
        global G_CO2g
        global G_H2Og
        global G_COg
        global kB
        global hplanck
        global ddG_HOCO
        global ddG_CO
        
        G_H2g = cons.G_H2g
        G_CO2g = cons.G_CO2g
        G_H2Og = cons.G_H2Og
        G_COg = cons.G_COg
        
        kB = cons.kB # Boltzmann constant in eV/K
        hplanck = cons.hplanck # eV s
        
        ddG_HOCO = cons.ddG_HOCO
        ddG_CO =  cons.ddG_CO
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
        
        # global tc0, tc1, tc2
        global A_act1, A_act2
        global G_1act_cap, G_2act_cap
        global A_prior
        global UHER0, URHE0
        
        # U0 = applied_p # applied potential vs. she
        self.T0 = T0 # 297.15, higher 370
        self.U = U0 + UHER0
        self.pCO2g = pCO2g
        self.pCOg =  pCOg
        self.pH2Og = pH2Og
        self.cHp = cHp0 #1.
        self.nu_e = kB * T0 / hplanck
        self.nu_c = 1.e13
        UHER0 = URHE0 = kB * T0 * np.log(cHp0)   # introduced to shift the plotted potential window to the relevant range w
        Gact1 = Gact2 = Gact # activative free energy 0.475
        A_act1 = np.exp( - Gact1 / ( kB * T0 ) ) # 
        A_act2 = np.exp( - Gact2 / ( kB * T0 ) ) # electrochemical prefactor, fitting
        G_1act_cap = -Gact1
        G_2act_cap = -Gact2
        A_prior = p_factor
        
        
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
    
    def plot_scaling_rev(self, fig, ax, contours, Eb_CO_e, Eb_HOCO_e, subtitle=''):
        EHOCO_d = (self.df[self.descriper1]).values
        ECO_d = (self.df[self.descriper2]).values
        obser_names = (self.df.index).values
        
        for i,obser_name in enumerate(obser_names):
            plt.plot(ECO_d[i], EHOCO_d[i], 'o', color='white') 
            plt.text(ECO_d[i], EHOCO_d[i]+0.05, obser_name, fontsize=12, horizontalalignment='center', verticalalignment='bottom', color='white')
        
        
        m, b = np.polyfit(EHOCO_d, ECO_d, 1)
        plt.axline(( ECO_d[0], ECO_d[0]/m-b/m), slope=1/m, color='white')
        # plt.plot(self.descriper1, m * self.descriper1 + b, linewidth=2, color=linecolor)
        
        plt.xlim([min(ECO_d)-0.2, max(ECO_d)+0.2])
        plt.ylim([min(EHOCO_d)-0.1, max(EHOCO_d)+0.1])
        ax.tick_params(labelsize=12) #tick label font size
        # plt.title(text[index], fontsize=14,)
        plt.text(0.05, 0.93, subtitle, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=14, color='white', fontweight='bold')        
        plt.xlabel(r'$E_{\mathrm{CO}}$ (eV)', fontsize=14,)
        plt.ylabel(r'$E_{\mathrm{HOCO}}$ (eV)', fontsize=14,)
        bar = plt.colorbar(ticks=np.arange(min(contours), max(contours), 0.5))
        bar.ax.tick_params(labelsize=10)
        bar.set_label(r'log$_{10}$(j/$\mu$Acm$^{-2}$)', fontsize=14,)
            
        # import string
        # ax.text(-0.17, 0.97, string.ascii_lowercase[index], transform=ax.transAxes, size=20, weight='bold')
        # import pdb; pdb.set_trace()
        
    
    def plot(self, ):
        # global T0
        # global tc0, tc1, tc2
        # global A_act1, A_act2
        # global G_1act_cap, G_2act_cap
        # global A_prior
        # global UHER0, URHE0
        
        # U0 = applied_p # applied potential vs. she
        # T0 = temp # 297.15, higher 370
        # UHER0 = URHE0 = kB * T0 * np.log(cHp0)   # introduced to shift the plotted potential window to the relevant range w
        # U = U0 + UHER0
        
        # pCO2g = pCO2g
        # pCOg =  pCOg
        # pH2Og = pH2Og
        # cHp = cHp0 #1.
        
        # nu_e = kB * T0 / hplanck
        # nu_c = 1.e13
        # Gact1 = Gact2 = Gact # activative free energy 0.475
        # A_act1 = np.exp( - Gact1 / ( kB * T0 ) ) # 
        # A_act2 = np.exp( - Gact2 / ( kB * T0 ) ) # electrochemical prefactor, fitting
        # G_1act_cap = -Gact1
        # G_2act_cap = -Gact2
        
        # tc0 = tc1 = tc2 = 0.5 # transfer coefficiency
        # A_prior = p_factor
        
        T0 = self.T0
        U = self.U
        pCO2g = self.pCO2g
        pCOg = self.pCOg
        pH2Og = self.pH2Og
        cHp = self.cHp
        nu_e = self.nu_e
        nu_c = self.nu_c
        
        tc0 = tc1 = tc2 = 0.5 # transfer coefficiency
        
        N, M = 20*4, 20*4
        R = np.empty([M,N])
        Thetas = np.empty([M,N,3])
        # Eb_HOCO_e = np.linspace(-0.8, 1.45, M)
        # Eb_CO_e = np.linspace(-2.2, 0.6, N)
        
        EHOCO_d = (self.df[self.descriper1]).values
        ECO_d = (self.df[self.descriper2]).values
        Eb_CO_e = np.linspace(min(ECO_d)-0.2, max(ECO_d)+0.2, N)
        Eb_HOCO_e = np.linspace(min(EHOCO_d)-0.1, max(EHOCO_d)+0.1, M)
        # Eb_CO_e = np.linspace(-1.8, 1., N)
        # Eb_HOCO_e = np.linspace(-1.2, 1.8, M)
        
        jmax = 10.0e3 # exptl current plateau's at 10 mA/cm2 
        jmin = 0.1
        for j, Eb_CO in enumerate(Eb_CO_e):
            for i, Eb_HOCO in enumerate(Eb_HOCO_e):
                k1, K1, k2, K2, k3, K3 = self.get_rates(nu_e, nu_c, Eb_HOCO, Eb_CO, U, T0, tc1, tc2, tc0)
                rm = CO2toCO(pCO2g, pCOg, pH2Og, cHp, k1, K1, k2, K2, k3, K3)
                # rm = CO2toCO(pCO2g, pCOg, pH2Og, cOHm, k1, K1, k2, K2, k3, K3, T0)
                thetas, rates = rm.solve()
                # print(rates)
                rate = min(jmax, rates[0])
                rate = max(jmin, rate)
                R[i,j] = np.log10(rate*47.96) # TOF to current
                Thetas[i,j,:] = thetas
        
        
        fig = plt.figure(figsize=(8, 6), dpi = 100)
        ax = plt.subplot(111)
        contours = np.linspace(np.log10(jmin*47.96), np.log10(jmax*47.96), 11) 
        plt.contourf(Eb_CO_e, Eb_HOCO_e, R, contours, cmap=plt.cm.jet) # plot countour
        self.plot_scaling_rev(fig, ax, contours, Eb_CO_e, Eb_HOCO_e, )
        
        # # fig.tight_layout()
        # plt.savefig('./paper1/Rate_vs_HOCO_CO.png', dpi=300, bbox_inches='tight')    
        # plt.show()