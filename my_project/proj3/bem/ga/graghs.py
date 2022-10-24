# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:58:55 2022

@author: changai
"""
import networkx as nx
import numpy as np
from itertools import product, combinations
from ase.data import (covalent_radii, 
                      atomic_numbers, 
                      atomic_masses)
from ase.io import read
from ase.visualize import view


def neighbor_shell_list(atoms, dx=0.3, neighbor_number=1, 
                        different_species=False, mic=False,
                        radius=None, span=False):
    """Make dict of neighboring shell atoms for both periodic and 
    non-periodic systems. Possible to return neighbors from defined 
    neighbor shell e.g. 1st, 2nd, 3rd by changing the neighbor number.
    Essentially returns a unit disk (or ring) graph.

    Parameters
    ----------
    atoms : ase.Atoms object
        Accept any ase.Atoms object. No need to be built-in.

    dx : float, default 0.3
        Buffer to calculate nearest neighbor pairs.

    neighbor_number : int, default 1
        Neighbor shell number.

    different_species : bool, default False
        Whether each neighbor pair are different species.

    mic : bool, default False
        Whether to apply minimum image convention. Remember to set 
        mic=True for periodic systems.

    radius : float, default None 
        The radius of each shell. Works exactly as a conventional 
        neighbor list when specified. If not specified, use covalent 
        radii instead.

    span : bool, default False
        Whether to include all neighbors spanned within the shell.
        Returns a unit disk graph if True, otherwise returns a unit
        ring graph.

    """

    natoms = len(atoms)
    if natoms == 1:
        return {0: []}
    cell = atoms.cell
    positions = atoms.positions
    nums = set(atoms.numbers)
    pairs = product(nums, nums)
    if not radius:
        cr_dict = {(i, j): (covalent_radii[i] + covalent_radii[j]) for i, j in pairs}
    
    ds = atoms.get_all_distances(mic=mic)
    conn = {k: [] for k in range(natoms)}
    for atomi in atoms:
        for atomj in atoms:
            i, j = atomi.index, atomj.index
            if i != j:
                if not (different_species & (atomi.symbol == atomj.symbol)):
                    d = ds[i,j]
                    crij = 2 * radius if radius else cr_dict[(atomi.number, atomj.number)] 

                    if neighbor_number == 1 or span:
                        d_max1 = 0.
                    else:
                        d_max1 = (neighbor_number - 1) * crij + dx

                    d_max2 = neighbor_number * crij + dx

                    if d > d_max1 and d < d_max2:
                        conn[atomi.index].append(atomj.index)

    return conn




def get_adj_matrix(neighborlist):
    """Returns an adjacency matrix from a neighborlist object.

    Parameters
    ----------
    neighborlist : dict
        A neighborlist (dictionary) that contains keys of each 
        atom index and values of their neighbor atom indices.

    """ 

    conn_mat = []
    index = range(len(neighborlist.keys()))
    # Create binary matrix denoting connections.
    for index1 in index:
        conn_x = []
        for index2 in index:
            if index2 in neighborlist[index1]:
                conn_x.append(1)
            else:
                conn_x.append(0)
        conn_mat.append(conn_x)

    return np.asarray(conn_mat)


def get_connectivity(atoms, tol=.5):                                      
    """Get the adjacency matrix."""
    nblist = neighbor_shell_list(atoms, dx=tol, neighbor_number=1, 
                                         different_species=True, mic=True) 
    return get_adj_matrix(nblist)


def get_graph(atoms, return_adj_matrix=False):                             
    """Get the graph representation of the slab.

    Parameters
    ----------
    return_adj_matrix : bool, default False
        Whether to return adjacency matrix instead of the networkx.Graph 
        object.

    """

    cm = get_connectivity(atoms)
    if return_adj_matrix:
        return cm
    
    G = nx.Graph()                               
    symbols = atoms.symbols                               
    G.add_nodes_from([(i, {'symbol': symbols[i]}) 
                      for i in range(len(symbols))])
    rows, cols = np.where(cm == 1)
    edges = zip(rows.tolist(), cols.tolist())
    G.add_edges_from(edges)

    return G

def draw_graph(G, savefig='graph.png', layout='spring', *args, **kwargs):               
    """Draw the graph using matplotlib.pyplot.

    Parameters
    ----------
    G : networkx.Graph object
        The graph object

    savefig : str, default 'graph.png'
        The name of the figure to be saved.

    layout : str, default 'spring'
        The graph layout supported by networkx. E.g. 'spring',
        'graphviz', 'random', etc.

    """

    import matplotlib.pyplot as plt
    labels = nx.get_node_attributes(G, 'symbol')
    
    # Get unique groups
    groups = sorted(set(labels.values()))
    mapping = {x: "C{}".format(i) for i, x in enumerate(groups)}
    nodes = G.nodes()
    colors = [mapping[G.nodes[n]['symbol']] for n in nodes]

    # Drawing nodes, edges and labels separately
    if layout in ['graphviz', 'pygraphviz']:
        layout_to_call = getattr(nx.drawing.nx_agraph, layout + '_layout')
    else:
        layout_to_call = getattr(nx, layout + '_layout')
    pos = layout_to_call(G)
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.5)
    nx.draw_networkx_nodes(G, pos, 
                           nodelist=nodes, 
                           node_color=colors, 
                           node_size=500)
    nx.draw_networkx_labels(G, pos, labels, 
                            font_size=10, 
                            font_color='w')
    plt.axis('off')
    plt.savefig(savefig, *args, **kwargs)
    plt.show()
    plt.clf()
    return pos

def plot_gragh(G, pos):
    import matplotlib.pyplot as plt
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    plt.show()
    nodes = G.nodes()
    for node in nodes:
        if G.degree(node) == 1:
            print('vertex element {}: {}'.format(G.nodes[node]['symbol'], node))
            adj = list(G.adj[node].keys())[0]
            print('vertex adjacency {}: {}'.format(G.nodes[adj]['symbol'], adj))
            
def plot_index_gragh(G, pos, savefig='graph_both.png', *args, **kwargs):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,6))
    subax1 = plt.subplot(121)
    labels = nx.get_node_attributes(G, 'symbol')
    groups = sorted(set(labels.values()))
    mapping = {x: "C{}".format(i) for i, x in enumerate(groups)}
    nodes = G.nodes()
    colors = [mapping[G.nodes[n]['symbol']] for n in nodes]
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=1.5)
    nx.draw_networkx_nodes(G, pos, 
                           nodelist=nodes, 
                           node_color=colors, 
                           node_size=500)
    nx.draw_networkx_labels(G, pos, labels, 
                            font_size=10, 
                            font_color='w')
    plt.axis('off')
    
    subax2 = plt.subplot(122)
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    plt.show()
    nodes = G.nodes()
    for node in nodes:
        if G.degree(node) == 1:
            print('vertex element {}: {}'.format(G.nodes[node]['symbol'], node))
            adj = list(G.adj[node].keys())[0]
            print('vertex adjacency {}: {}'.format(G.nodes[adj]['symbol'], adj))
    plt.savefig(savefig, *args, **kwargs)

if __name__ == '__main__':
    
    atoms = read('../random50_adss.traj', index=2)
    view(atoms)
    G = get_graph(atoms)
    pos = draw_graph(G)
    # plot_gragh(G, pos)
    plot_index_gragh(G, pos)
    