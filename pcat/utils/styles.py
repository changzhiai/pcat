# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 12:32:08 2021

@author: changai
"""

# ColorDict = {'Pd64H64': 'black', 
#               # 'Pd32Ti32H64': 'black', 
             
#               'Pd16Ti48H64': 'black', 
#               # 'Pd16Ti48H40': 'black',
             
#               'Pd48Ti16H64': 'black',
#               'Pd48Ti16H60': 'black',
#               'Pd48Ti16H59': 'black',
             
#               'Pd51Ti13H59': 'black',
#               }

ColorDict = {'Pd64H64': 'red', 'Pd64H39': 'red', 'Pd64H63': 'red',}

ColorDict_proj1 = {'Pd64H64': 'black','PdH': 'black', 'Pure': 'black', 'Ti': 'red', 'Pd': 'black', 'Sc': 'blue', 'V': 'orange', 'Cr': 'wheat', 'Mn': 'green', \
                          'Fe': 'lightgray', 'Co': 'deepskyblue', 'Ni': 'pink', 'Cu': 'purple', 'Zn': 'olive', 'Y': 'cyan', 'Zr': 'lime', \
                          'Nb': 'yellow', 'Mo': 'navy', 'Ru': 'magenta', 'Rh': 'brown', 'Ag': 'lightseagreen', 'Cd': 'steelblue', 'Hf': 'slateblue', \
                          'Ta': 'violet', 'W': 'deeppink', 'Re': 'palevioletred'}

    
ColorList = ['k', 'lime', 'r', 'b', 'darkcyan', 'cyan', 'olive', 'magenta', 'pink', 'gray', 'orange', 'purple', 'g', 'crimson', 'brown', \
                  'teal', 'thistle', 'y', 'tan', 'navy', 'wheat', 'gold', 'lightcoral', 'silver', 'violet', 'turquoise', 'seagreen', 'tan', \
                  'k', 'lime', 'r', 'b', 'darkcyan', 'cyan', 'olive', 'magenta', 'pink', 'gray', 'orange', 'purple', 'g', 'pink', 'brown',\
                  'k', 'lime', 'r', 'b', 'darkcyan', 'cyan', 'olive', 'magenta', 'pink', 'gray', 'orange', 'purple', 'g', 'pink', 'brown']
    
StepNamesCO2RR = ['* + CO$_{2}$', 'HOCO*', 'CO*', '* + CO']  # reload step name for CO2RR
StepNamesHER = ['* + $H^+$', '*H', '* + 1/2$H_2$',]  # reload step name for HER
ObserNames = ["Pure", "Ni", "Co", "V", "Cr", "Mn", "Fe", "Pt"]  # reload specis name