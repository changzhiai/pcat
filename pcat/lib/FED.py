# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 14:24:21 2021

@author: changai
@reference Energy profile diagram - Giacomo Marchioro

"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import interp1d


class FED:
    """Class for free energy diagram libraray
    
    Parameters:
        
    ratio: float
        ratio of figure. The default value is gold ratio
    ax: object
        matplotlib figure handlers for stack of figures
    """
    def __init__(self, aspect='equal'):
        # plot parameters
        # self.ratio = 1.6181 #1.5
        self.dimension = 'auto'
        self.space = 'auto'
        self.offset = 'auto'
        self.offset_ratio = 0.02
        self.color_bottom_text = 'blue'
        self.aspect = aspect
        # data
        self.pos_number = 0
        self.energies = []
        self.positions = []
        self.bottom_texts = []     
        self.top_texts = []
        self.colors = []
        self.right_texts = []
        self.left_texts = []        
        self.links = []
        self.barriers = []
        self.ts_energies = []
        self.all_energies = []
        
        # self.fig = None
        self.ax = None
        self.label = []   
        
        # fixed y range
        # self.ymin = None
        # self.ymax = None

    def add_level(self, energy, bottom_text='', position=None, color='k',
                  top_text='', right_text='', left_text='', label=''):
        """Add horizonal level line to the figure"""
        if position is None:
            position = self.pos_number + 1
            self.pos_number += 1
        elif isinstance(position, (int, float)):
            pass
        elif position == 'last' or position == 'l':
            position = self.pos_number
        else:
            raise ValueError(
                "Position must be None or 'last' (abrv. 'l') or in case an integer or float specifing the position. It was: %s" % position)
        if top_text == 'Energy':
            top_text = energy

        link = []
        barrier = []
        
        self.energies.append(energy)
        self.positions.append(position)
        self.bottom_texts.append(bottom_text)
        self.top_texts.append(top_text)
        self.colors.append(color)
        self.right_texts.append(right_text)
        self.left_texts.append(left_text)
        self.label.append(label)
        
        self.links.append(link)
        self.barriers.append(barrier)

    def add_link(self, start_level_id, end_level_id, color='k', ls='--', linewidth=1, ):
        """Add dashed link line between levels"""
        self.links[start_level_id].append((end_level_id, ls, linewidth, color))
        #print(self.links)
    
    def remove_link(self, start_level_id, end_level_id):
        """Remove dashed link line between levels"""
        for i, link in enumerate(self.links):
            for j in link: 
                if start_level_id == i and end_level_id == j[0]:
                    self.links[i].remove(j)
        
    def add_barrier(self, start_level_id, barrier, end_level_id, color='k', ls='--', linewidth=1, ):
        """Add energy barrier curve between levels"""
        self.ts_energies.append(barrier)       
        self.barriers[start_level_id].append((end_level_id, barrier, ls, linewidth, color))
        #print(self.barrier)
        
    def remove_barrier(self, start_level_id, end_level_id):
        """Remove energy barrier curve between levels"""
        print(self.barriers)
        for i, barrier in enumerate(self.barriers):
            for j in barrier:        
                if start_level_id == i and end_level_id == j[0]:
                    self.barriers[i].remove(j)
                    print('run this line')
        print('\n after removing', self.barriers)

    def plot(self, show_IDs=False, 
             xlabel = "Reaction coordinate", 
             ylabel="Free energy (eV)", 
             xtickslabel='write xticks', 
             stepLens=4, 
             ax: plt.Axes = None, title='', 
             ratio=1.6181, 
             ymin=None, 
             ymax=None):
        """Plot energy diagram"""
        self.ratio = ratio
        self.ymin = ymin
        self.ymax = ymax
        # Create a figure and axis if the user didn't specify them.
        if not ax:
            fig = plt.figure(figsize=(8,6), dpi = 300)
            ax = fig.add_subplot(111, aspect=self.aspect)
        # Otherwise register the axes and figure the user passed.
        else:
            self.ax = ax
            # self.fig = ax.figure
            # Constrain the target axis to have the proper aspect ratio
            self.ax.set_aspect(self.aspect)

        ax.set_xlabel(xlabel, fontsize=14) #xlabel frontsize
        ax.set_ylabel(ylabel, fontsize=14) #ylabel frontsize
        
        ax.tick_params(axis="x", labelsize=12) #xtick frontsize
        ax.tick_params(axis="y", labelsize=12) #ytick frontsize
        #ax.tick_params(labelsize=8)
        
        ax.axes.get_xaxis().set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        plt.title(title, fontsize=14)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.2) #linewith of frame


        self.__auto_adjust()

        data = list(zip(self.energies,  # 0
                        self.positions,  # 1  [1,2,3,4,1,2,3,4,1,2...]
                        self.bottom_texts,  # 2
                        self.top_texts,  # 3
                        self.colors,  # 4
                        self.right_texts,  # 5
                        self.left_texts,
                        self.label))  # 6
        
        pos = []
        for level in data:
            start = level[1]*(self.dimension+self.space)
            pos.append(start + self.dimension/2.)
            #print(pos)
            ax.hlines(level[0], start, start + self.dimension, color=level[4], linewidth=2)
            # ax.text(start+self.dimension/2.,  # X
            #         level[0]+self.offset,  # Y
            #         level[3],  # self.top_texts
            #         horizontalalignment='center',
            #         verticalalignment='bottom')

            # ax.text(start + self.dimension,  # X
            #         level[0],  # Y
            #         level[5],  # self.bottom_text
            #         horizontalalignment='left',
            #         verticalalignment='center',
            #         color=self.color_bottom_text)

            # ax.text(start,  # X
            #         level[0],  # Y
            #         level[6],  # self.bottom_text
            #         horizontalalignment='right',
            #         verticalalignment='center',
            #         color=self.color_bottom_text)

            # ax.text(start + self.dimension/2.,  # X
            #         level[0] - self.offset*2,  # Y
            #         level[2],  # self.bottom_text
            #         horizontalalignment='center',
            #         verticalalignment='top',
            #         color=self.color_bottom_text)

        # ax.set_xticks(np.arange(0.1,2.91,0.7))
        ax.set_xticks(pos[0:stepLens])
        ax.set_xticklabels(xtickslabel)
        
        # for showing the ID allowing the user to identify the level
        if show_IDs:     
            for ind, level in enumerate(data):
                start = level[1]*(self.dimension+self.space)
                ax.text(start, level[0]+self.offset, str(ind),
                        horizontalalignment='right', color='red')
        
        # add connection line
        for idx, link in enumerate(self.links):
            # here we connect the levels with the links
            # x1, x2   y1, y2
            for i in link:
                # i is a tuple: (end_level_id,ls,linewidth,color of connection line)
                start = self.positions[idx]*(self.dimension+self.space)
                x1 = start + self.dimension
                x2 = self.positions[i[0]]*(self.dimension+self.space)
                y1 = self.energies[idx]
                y2 = self.energies[i[0]]
                line = Line2D([x1, x2], [y1, y2],
                              ls=i[1],
                              linewidth=i[2],
                              color=i[3])
                ax.add_line(line)

        # add connection barriers
        # diagram.add_barrier(start_level_id=1, barrier=1, end_level_id=2) #example
        for idx, barrier in enumerate(self.barriers):
            # here we connect the levels with the barriers lines
            # x1, x2  xb, yb  y1, y2
            for i in barrier:
                # i is a tuple: (end_level_id, barrier, ls,linewidth,color of connection line)
                start = self.positions[idx]*(self.dimension+self.space)
                x1 = start + self.dimension
                x2 = self.positions[i[0]]*(self.dimension+self.space)
                xb = 1/2. * (x1 + x2)            
                y1 = self.energies[idx]
                y2 = self.energies[i[0]]
                yb = i[1]
                
                x = [x1, xb, x2]
                y = [y1, yb, y2]
                f = interp1d(x, y, kind='quadratic')  #Interpolate a 1-D function
                x_interpol = np.linspace(x1, x2, 1000)
                y_interpol = f(x_interpol)
                line = plt.plot(x_interpol, y_interpol,
                              ls=i[2],
                              linewidth=i[3],
                              color=i[4])
                # ax.add_line(line)
        return pos #return x ticks values


    def __auto_adjust(self):
        '''
        Method of ED class
        This method use the ratio to set the best dimension and space between
        the levels.

        Affects
        -------
        self.dimension
        self.space
        self.offset

        '''
        # Max range between the energy
        self.all_energies = self.energies + self.ts_energies
        if self.ymin == None and self.ymax == None:
            Energy_variation = abs(max(self.all_energies) - min(self.all_energies))
        else:
            Energy_variation = self.ymax - self.ymin
        
        if self.dimension == 'auto' or self.space == 'auto':
            # Unique positions of the levels
            unique_positions = float(len(set(self.positions)))
            space_for_level = Energy_variation*self.ratio/unique_positions
            self.dimension = space_for_level*0.6
            self.space = space_for_level*0.4

        if self.offset == 'auto':
            self.offset = Energy_variation*self.offset_ratio


if __name__ == '__main__':
    ''
