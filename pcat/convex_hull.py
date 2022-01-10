# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 21:05:46 2022

@author: changai
"""

from contextlib import contextmanager
import numpy as np
import os
from ase.db import connect
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import pickle
import matplotlib.pyplot as plt 
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['font.size'] = 8
mpl.use('TkAgg')

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    # try:
    #     os.makedirs(newdir)
    # except OSError:
    #     pass
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

def num_ele(atoms, ele):
    """Numbers calculation of object element"""
    num_ele = len(atoms[[atom.index for atom in atoms if atom.symbol==ele]])
    return num_ele

def con_ele(atoms, ele, ref_eles=['Pd', 'Ti']):
    """Concentration calculation of element
    totally have three elements, such as, ele=='H', ref_eles=['Pd', 'Ti']
    
    con = num_ele / num_ref_eles
    
    """
    num_obj_ele = num_ele(atoms, ele)
    num_ref_eles = 0
    for ref_ele in set(ref_eles):
        try:
            num_ref_ele = num_ele(atoms, ref_ele)
        except:
            num_ref_ele = 0
        num_ref_eles += num_ref_ele
    con_ele = num_obj_ele / num_ref_eles
    return con_ele

def formation_energy_ref_metals(atoms, energy_tot, energy_ref_eles):
    """Formation energy calculation references pure metal and H2 gas
    
    For exmaple: energy_ref_eles={'Pd':-1.951, 'Ti':-5.858, 'H': -7.158*0.5}
    Pure Pd: -1.951 eV/atom
    Pure Ti: -5.858 eV/atom
    H2 gas: -7.158 eV
    """
    energies_ref_eles = 0
    num_eles_tot = 0
    for ele, energy_ref_ele in energy_ref_eles.items():
        num_ref_ele = num_ele(atoms, ele)
        energies_ref_eles += num_ref_ele * energy_ref_ele
        num_eles_tot += num_ref_ele
        
    form_e_ref_metals = energy_tot - energies_ref_eles
    form_e_ref_metals_per_atom = form_e_ref_metals / num_eles_tot
    return form_e_ref_metals_per_atom

def formation_energy_ref_hydrides(atoms, energy_tot, energy_ref_hyds):
    """Formation energy calculation references metal hydrides
    
    For example: energy_ref_eles={'Pd':-1.951, 'Ti':-5.858, 'H': -7.158*0.5}
    Reference PdH and TiH (vasp) for PdxTi(64-x)H64
    Pure PdH: -5.222 eV/PdH
    Pure TiH: -9.543 eV/TiH
    """
    energies_ref_hyds = 0
    num_hyds_tot = 0
    for hyd, energy_ref_hyd in energy_ref_hyds.items():
        ele_metal = hyd.split('_')[0]
        num_metal_ref_hyd = num_ele(atoms, ele_metal)
        energies_ref_hyds += num_metal_ref_hyd * energy_ref_hyd
        num_hyds_tot += num_metal_ref_hyd
    
    form_e_ref_hyds = energy_tot - energies_ref_hyds
    form_e_ref_hyds_per_atom = form_e_ref_hyds / (2*num_hyds_tot)
    return form_e_ref_hyds_per_atom


def get_candidates(pts, db_name='results_last.db', db_cand_name='candidates.db'):
    """Get candidates on vertex of convex hull
    
    Parameters
    
    db_name: str
        database name of structures of the lowest energy in each concentration
    db_cand_name: str
        database name of structures of the energies on vertices of convex hull
    """
    points = pts[:,0:3].astype(np.float32)
    hull = ConvexHull(points)
    vertices = pts[hull.vertices]
    ids = vertices[:,3]
    # print(vertices[1,:])
    db = connect(db_name)
    if os.path.exists(db_cand_name):
        os.remove(db_cand_name)
    db_cand = connect(db_cand_name)
    for uni_id in ids:
        row = db.get(uni_id=uni_id)
        db_cand.write(row)

def plot_convex_hull_ref_metals_3d_line(pts):
    """Convex hull in the style of 3d line"""
    hull = ConvexHull(pts)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pts.T[0], pts.T[1], pts.T[2], "bx")
    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-")
    ax.set_xlabel('Concentration of Pd')
    ax.set_ylabel('Concentration of H')
    ax.set_zlabel('Formation energies (eV/atom)')
    plt.title(str(pts.shape[0]) + ' data points')
    plt.show()
    # save to html format
    import base64
    from io import BytesIO
    temp = BytesIO()
    fig.savefig(temp, format="png")
    fig_encode_bs64 = base64.b64encode(temp.getvalue()).decode('utf-8')
    html_string = """
    <h2>This is a convex hull html</h2>
    <img src = 'data:image/png;base64,{}'/>
    """.format(fig_encode_bs64)
    with open("convex_hull.html", "w") as f:
        f.write(html_string) 

def plot_convex_hull_ref_metals_3d_plane(pts):
    """Convex hull in the style of 3d plane"""
    hull = ConvexHull(pts)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # ax.plot(pts.T[0], pts.T[1], pts.T[2], "bx")
    X = []
    Y = []
    Z = []
    for s in hull.simplices:
        # print(s)
        # print(pts[s, 0])
        # print(pts[s, 1])
        # print(pts[s, 2])
        X.append(pts[s, 0])
        Y.append(pts[s, 1])
        Z.append(pts[s, 2])
    ax.plot_surface(np.array(X), np.array(Y), np.array(Z), rstride=1, cstride=1,
                    cmap='viridis', edgecolor='r')
    ax.set_xlabel('Concentration of Pd')
    ax.set_ylabel('Concentration of H')
    ax.set_zlabel('Formation energies (eV/atom)')
    plt.title(str(pts.shape[0]) + ' data points')
    plt.show()  
    # plt.savefig('plot_convex_hull_ref_metals_3d_plane.png')

def plot_convex_hull_ref_metals_3d_poly(pts):
    """Convex hull in the style of 3d polygon"""
    hull = ConvexHull(pts)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for simplex in hull.simplices:
        x = pts[simplex[0]]
        y = pts[simplex[1]]
        z = pts[simplex[2]]
        verts = [x,y,z]
        ax.add_collection3d(Poly3DCollection([verts],facecolors='w',
                                            edgecolors='b',
                                            linewidths=1,
                                            alpha=1.0)) 
    ax.set_xlabel('Concentration of Pd')
    ax.set_ylabel('Concentration of H')
    ax.set_zlabel('Formation energies (eV/atom)')
    plt.title(str(pts.shape[0]) + ' data points')
    plt.show()  
    # plt.savefig('plot_convex_hull_ref_metals_3d_plane.png')


def plot_2d_contour(pts, vertices=True):
    """Projection 3d convex hull to 2d contour"""
    ax = plt.figure()
    # scat = plt.contourf(pts[:,0], pts[:,1], pts[:,2], cmap=plt.cm.jet)
    # scat = plt.scatter(pts[:,0], pts[:,1], c=pts[:,2], marker='o', cmap="viridis")
    scat = plt.scatter(pts[:,0], pts[:,1], c=pts[:,2], marker='o', cmap=plt.cm.jet)
    bar = plt.colorbar(scat)
    if vertices == True:
        hull = ConvexHull(pts)
        vertices = pts[hull.vertices]
        plt.scatter(vertices[:,0], vertices[:,1], c='r', marker='.', zorder=2)
        for s in hull.simplices:
            s = np.append(s, s[0])  # Here we cycle back to the first coordinate
            plt.plot(pts[s, 0], pts[s, 1], "r--", alpha=0.3, zorder=1)
    bar.set_label(r'Formation energy (eV/atom)', fontsize=12,)
    plt.title(str(pts.shape[0]) + ' data points')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Concentration of Pd')
    plt.ylabel('Concentration of H')
    plt.show()
    ax.savefig('2d_contour.png')

def plot_basical_convex_hull(vertices, ax=None):
    """Plot 2d convex hull"""
    # ax.plot(hull.points[:,0], hull.points[:,1], 'x')
    vertices = vertices[vertices[:,0].argsort()] 
    ax.plot(vertices[:,0], vertices[:,1], 'x')
    ax.plot(vertices[:,0], vertices[:,1])

def basical_convex_hull(arr, ax, fix_ele, metal_obj):
    """Plot 2d convex hull according to given axis"""
    if fix_ele == 'H':
        col1 = arr[:,0].astype(np.float32)
        col2 = arr[:,2].astype(np.float32)
    elif fix_ele == metal_obj:
        col1 = arr[:,1].astype(np.float32)
        col2 = arr[:,2].astype(np.float32)
    points = np.column_stack((col1, col2))
    hull = ConvexHull(points=points)
    # convex_hull_plot_2d(hull, ax=ax)
    vertices = points[hull.vertices] # get convex hull vertices
    plot_basical_convex_hull(vertices, ax=ax)

def get_y_range(pts):
    """Get the minimum and maximum value of y axis"""
    y_max = -10
    y_min = 10
    for k, v in pts.items():
        y_values = v[:,2].astype(np.float32)
        if min(y_values) < y_min:
            y_min = min(y_values)
        if max(y_values) > y_max:
            y_max = max(y_values)
    return y_min, y_max

def convex_hull_stacking_subplots(pts, fix_ele='H', metal_obj='Ti'):
    """Plot stacking convex hull in 2d"""
    fig = plt.figure(figsize=(16,16))
    M = 8
    N = 9
    m1 = 0
    m2 = 0
    y_min, y_max = get_y_range(pts)
    for k, v in pts.items():
        ax = plt.subplot(N, M, m1*M + m2 + 1)
        basical_convex_hull(v, ax, fix_ele, metal_obj)
        if fix_ele == 'H':
            plt.title('$Pd_x{}_{}$'.format(metal_obj, '64-x') + '$H_{}$'.format({64-int(k)}), x=0.5, y=0.7)
        if fix_ele == metal_obj:
            plt.title('$Pd_{}{}_{}Hy$'.format({64-int(k)}, metal_obj, {k}), x=0.5, y=0.7)
        plt.xlim(0,1)
        plt.ylim(y_min,y_max)
        m2 += 1
        if m2 == M:
            m2 = 0
            m1 += 1
    fig.add_subplot(111, frameon=False) # hide frame
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False) # hide ticks
    if fix_ele == 'H':
        plt.xlabel('Concentration of Pd')
    elif fix_ele == metal_obj:
        plt.xlabel('Concentration of H')
    plt.ylabel('Formation energy (eV)')
    fig.tight_layout()
    plt.show()
    fig.savefig('stacking_subplots_{}'.format(fix_ele))

def dct_to_array(pts_dict):
    """Dictionay format to 3d array"""
    pts = np.zeros((0,5))
    for k, v in pts_dict.items():
        pts = np.concatenate((pts, v), axis=0)
    pts = pts[:,0:3].astype(np.float32)
    return pts

def get_chem_pot_H_continuous_cons(pts, num_metal_obj=0):
    """
    specific at PdHx

    X H form
    """
    alloy_Hx = pts[num_metal_obj]  # PdHx
    Etot = alloy_Hx[:,4].astype(np.float32)
    Hcons = alloy_Hx[:,1].astype(np.float32)
    # print(alloy_Hx)
    print(Etot)
    print(Hcons)
    dyf = [0.0] * len(Hcons)
    for i in range(len(Etot)-1):
        # dyf[i] = (Etot[i+1] - Etot[i])/(Hcons[i+1]-Hcons[i])
        print(Etot[i])
        dyf[i] = (Etot[i+1] - Etot[i])/(-1)
    #set last element by backwards difference
    # dyf[-1] = (Etot[-1] - Etot[-2])/(Hcons[-1] - Hcons[-2])
    dyf[-1] = (Etot[-1] - Etot[-2])/(-1)
    print(dyf)
    plt.figure()
    plt.xlabel('Concentration of H')
    plt.ylabel('$\Delta \mu_H $ (eV)')
    plt.plot(Hcons, dyf)
    plt.show()

def get_chem_pot_H_vertices(pts, fix_ele, metal_obj, db_name, num_metal_obj=0, save_cand=True, db_candidate='PdHx_vertices.db'):
    """
    specific at PdHx

    X H form ids pot_E
    """
    pts = pts[num_metal_obj]  # PdHx
    if fix_ele == 'H':
        col1 = pts[:,0].astype(np.float32)
        col2 = pts[:,2].astype(np.float32)
    elif fix_ele == metal_obj:
        col1 = pts[:,1].astype(np.float32)
        col2 = pts[:,2].astype(np.float32)
    points = np.column_stack((col1, col2))
    hull = ConvexHull(points)
    vertices = pts[hull.vertices]
    vertices = vertices[vertices[:,1].argsort()]
    Etot = vertices[:,4].astype(np.float32)
    Hcons = vertices[:,1].astype(np.float32)
    if save_cand == True:
        ids = vertices[:,3]
        db = connect('results_last.db')
        if os.path.exists(db_candidate):
            os.remove(db_candidate)
        db_cand = connect(db_candidate)
        for uni_id in ids:
            row = db.get(uni_id=uni_id)
            db_cand.write(row)
    # print(Etot)
    # print(Hcons)
    dyf = [0.0] * len(Hcons)
    for i in range(len(Etot)-1):
        dyf[i] = (Etot[i+1] - Etot[i])/((Hcons[i+1]-Hcons[i])*64)
        # print(Etot[i])
    #set last element by backwards difference
    dyf[-1] = (Etot[-1] - Etot[-2])/((Hcons[-1] - Hcons[-2])*64)
    # print(dyf)
    plt.figure()
    plt.xlabel('Concentration of H')
    plt.ylabel('$\Delta \mu_H $ (eV)')
    plt.plot(Hcons, dyf)
    plt.show()

def get_chem_pot_H_2d_contour(pts):
    """
    X H form
    """
    Pdxcons = []
    Hycons = []
    chem_pot_H = []
    for k, v in pts.items():
        alloy_Hx = v  # PdTiHx
        # print(alloy_Hx)
        Etot = alloy_Hx[:,4].astype(np.float32)
        Hcons = alloy_Hx[:,1].astype(np.float32)
        Pdcons = alloy_Hx[:,0].astype(np.float32)
        dyf = [0.0] * len(Hcons)
        for i in range(len(Etot)-1):
            dyf[i] = (Etot[i+1] - Etot[i])/(-1)
        #set last element by backwards difference
        dyf[-1] = (Etot[-1] - Etot[-2])/(-1)
        Pdxcons.append(Pdcons.tolist())
        Hycons.append(Hcons.tolist())
        chem_pot_H.append(dyf)
    # print(Pdxcons)
    # print(Hycons)
    # print(chem_pot_H)
    plt.figure()
    scat = plt.scatter(Pdxcons, Hycons, c=chem_pot_H, marker='o', cmap=plt.cm.jet)
    bar = plt.colorbar(scat)
    bar.set_label(r'$\Delta \mu_H $ (eV/atom)', fontsize=12,)
    # plt.title(str(len(chem_pot_H)) + ' data points')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Concentration of Pd')
    plt.ylabel('Concentration of H')
    plt.show()

def get_chem_pot_H_vertices_2d_contour(pts, fix_ele='Ti', metal_obj='Ti'):
    """
    X H form ids E_pot
    """
    Pdxcons = []
    Hycons = []
    chem_pot_H = []
    for k, v in pts.items():
        alloy_Hx = v  # PdTiHx
        if fix_ele == 'H':
            col1 = alloy_Hx[:,0].astype(np.float32)
            col2 = alloy_Hx[:,2].astype(np.float32)
        elif fix_ele == metal_obj:
            col1 = alloy_Hx[:,1].astype(np.float32)
            col2 = alloy_Hx[:,2].astype(np.float32)
        points = np.column_stack((col1, col2))
        hull = ConvexHull(points)
        vertices = alloy_Hx[hull.vertices]
        vertices = vertices[vertices[:,1].argsort()]
        Etot = vertices[:,4].astype(np.float32)
        Hcons = vertices[:,1].astype(np.float32)
        Pdcons = vertices[:,0].astype(np.float32)
        dyf = [0.0] * len(Hcons)
        for i in range(len(Etot)-1):
            dyf[i] = (Etot[i+1] - Etot[i])/((Hcons[i+1]-Hcons[i])*64)
        #set last element by backwards difference
        dyf[-1] = (Etot[-1] - Etot[-2])/((Hcons[-1] - Hcons[-2])*64)
        Pdxcons.append(Pdcons.tolist())
        Hycons.append(Hcons.tolist())
        chem_pot_H.append(dyf)
    
    plt.figure()
    Pdxcons = list(np.concatenate(Pdxcons).flat)
    Hycons = list(np.concatenate(Hycons).flat)
    chem_pot_H = list(np.concatenate(chem_pot_H).flat)
    scat = plt.scatter(Pdxcons, Hycons, c=chem_pot_H, marker='o', cmap=plt.cm.jet)
    bar = plt.colorbar(scat)
    bar.set_label(r'$\Delta \mu_H $ (eV/atom)', fontsize=12,)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Concentration of Pd')
    plt.ylabel('Concentration of H')
    plt.show()

def collect_array_all_data(energy_ref_eles, db_name='results.db', data_name='data_tot.npy', eles=['Pd', 'Ti', 'H']):
    """get all data in array format, otherwise get the late one
    
    Parameters: 
    
    db_name: str
         OUTPUT for all data including structures. This would need huge storage.
    data_name: str
         OUTPUT for needed all numpy array data. This would need less storage.
    """
    if 1: # the first run, get all data in results.db
        if os.path.exists(db_name):
            os.remove(db_name)
        db_tot = connect(db_name)
        metal_obj = eles[1]
        cons_Pd = []
        cons_H = []
        form_energies = []
        ids = []
        clease_es = []
        for num_H in range(0, 65):
            for num_metal_obj in range(0, 65):
                with cd('H{0}'.format(str(num_H))):
                    with cd(metal_obj+'{0}'.format(str(num_metal_obj))):
                        db = connect('result.db')
                        for row in db.select():
                           atoms = row.toatoms()
                           uni_id = 'H' + str(num_H) + '_' + metal_obj + str(num_metal_obj) + '_' +str(row.id)
                           ids.append(uni_id)
                           con_Pd = con_ele(atoms, eles[0], ref_eles=[eles[0], eles[1]])
                           con_H = con_ele(atoms, eles[2], ref_eles=[eles[0], eles[1]])
                           cons_Pd.append(con_Pd)
                           cons_H.append(con_H)
                           clease_e = row.energy
                           form_energy = formation_energy_ref_metals(atoms, clease_e, energy_ref_eles)
                           form_energies.append(form_energy)
                           clease_es.append(clease_e)
                           db_tot.write(atoms, con_Pd=con_Pd, con_H=con_H, form_energy=form_energy, uni_id=uni_id,)
        zipped = zip(cons_Pd, cons_H, form_energies, ids)
        pts = np.array(list(zipped))
        np.save(data_name, pts) 
        pts = pts[:,0:3].astype(np.float32)
    else:
        pts = np.load(data_name, mmap_mode='r')
        pts = pts[:,0:3].astype(np.float32)
    # plot_convex_hull_ref_metals_3d_line(pts=pts)
    # plot_convex_hull_ref_metals_3d_plane(pts=pts)

def collec_array_last_data(energy_ref_eles, db_name='results_last.db', data_name='data_last.npy', eles=['Pd', 'Ti', 'H']):
    """get the last one in numpy array format.
    
    Not recommended
    
    Parameters: 
    
    db_name: str
         OUTPUT for last data including structures. This would need larger storage.
    data_name: str
         OUTPUT for needed last numpy array data. This would need less storage.
    """
    if 1:
        if os.path.exists(db_name): # get the last one (the lowest energies in each concentration)
            os.remove(db_name)
        db_last1 = connect(db_name)
        metal_obj = eles[1]
        cons_Pd = []
        cons_H = []
        form_energies = []
        ids = []
        clease_es = []
        for num_H in range(0, 65):
            for num_metal_obj in range(0, 65):
                with cd('H{0}'.format(str(num_H))):
                    with cd(metal_obj+'{0}'.format(str(num_metal_obj))):
                       db = connect('result.db')
                       lens = len(db)
                       row = list(db.select(id=lens))[0] # the last row
                       atoms = row.toatoms()
                       uni_id = 'H' + str(num_H) + '_' + metal_obj + str(num_metal_obj) + '_' +str(row.id)
                       ids.append(uni_id)
                       con_Pd = con_ele(atoms, eles[0], ref_eles=[eles[0], eles[1]])
                       con_H = con_ele(atoms, eles[2], ref_eles=[eles[0], eles[1]])
                       cons_Pd.append(con_Pd)
                       cons_H.append(con_H)
                       clease_e = row.energy
                       form_energy = formation_energy_ref_metals(atoms, clease_e, energy_ref_eles)
                       form_energies.append(form_energy)
                       clease_es.append(clease_e)
                       db_last1.write(atoms, con_Pd=con_Pd, con_H=con_H, form_energy=form_energy, uni_id=uni_id,)
        zipped = zip(cons_Pd, cons_H, form_energies, ids)
        pts = np.array(list(zipped))
        np.save(data_name, pts) 
        pts = pts[:,0:3].astype(np.float32)
    else:
        pts = np.load(data_name, mmap_mode='r')
        # get_candidates(pts, db_name='results_last1.db')
        pts = pts[:,0:3].astype(np.float32)
    # plot_convex_hull_ref_metals_3d_line(pts=pts)
    # plot_convex_hull_ref_metals_3d_plane(pts=pts)
    # plot_2d_contour(pts=pts)

def collec_dict_all_data(energy_ref_eles, db_name='results.db', data_name='data_tot_dict.pkl', eles=['Pd', 'Ti', 'H']):
    """get the all data in dictionary format
    
    Parameters: 
    
    db_name: str
         OUTPUT for last data including structures. This would need larger storage.
    data_name: str
         OUTPUT for needed last dict data saved pickle format. This would need less storage.
    """
    if 1: # the first run, get all data in results.db
        # if os.path.exists('results.db'):
            # os.remove('results.db')
        # db_tot = connect('results.db')
        metal_obj = eles[1]
        cons_Pd = []
        cons_H = []
        form_energies = []
        ids = []
        clease_es = []
        pts = dict()
        for num_H in range(0, 65):
            for num_metal_obj in range(0, 65):
                with cd('H{0}'.format(str(num_H))):
                    with cd(metal_obj+'{0}'.format(str(num_metal_obj))):
                        db = connect(db_name)
                        for row in db.select():
                           atoms = row.toatoms()
                           uni_id = 'H' + str(num_H) + '_' + metal_obj + str(num_metal_obj) + '_' +str(row.id)
                           ids.append(uni_id)
                           con_Pd = con_ele(atoms, eles[0], ref_eles=[eles[0], eles[1]])
                           con_H = con_ele(atoms, eles[2], ref_eles=[eles[0], eles[1]])
                           cons_Pd.append(con_Pd)
                           cons_H.append(con_H)
                           clease_e = row.energy
                           form_energy = formation_energy_ref_metals(atoms, clease_e, energy_ref_eles)
                           form_energies.append(form_energy)
                           clease_es.append(clease_e)
                           # db_tot.write(atoms, con_Pd=con_Pd, con_H=con_H, form_energy=form_energy, uni_id=uni_id,)
            zipped = zip(cons_Pd, cons_H, form_energies, ids)
            pts[num_H] = np.array(list(zipped))
        with open(data_name, 'wb') as outfile:
            pickle.dump(pts, outfile, protocol=pickle.HIGHEST_PROTOCOL) 
    else:
        with open(data_name, 'rb') as f:
            pts = pickle.load(f)
    # plot_convex_hull_ref_metals_3d_line(pts=dct_to_array(pts))
    # plot_convex_hull_ref_metals_3d_plane(pts=dct_to_array(pts))

def collec_dict_last_data(energy_ref_eles, 
                          db_name='results_last.db', 
                          data_name='data_last_dict.pkl', 
                          eles=['Pd', 'Ti', 'H']):
    """get the last data in dictionary format
    
    Parameters: 
    
    db_name: str
         OUTPUT for last data including structures. This would need larger storage.
    data_name: str
         OUTPUT for needed last dict data saved pickle format. This would need less storage.
    """
    if 1:
        # if os.path.exists('results_last1_dic.db'): # get the last one (the lowest energies in each concentration)
            # os.remove('results_last1.db')
        # db_last1 = connect('results_last1.db')
        metal_obj = eles[1]
        pts = dict()
        for num_H in range(0, 65):
            cons_Pd = []
            cons_H = []
            form_energies = []
            ids = []
            clease_es = []
            for num_metal_obj in range(0, 65):
                with cd('H{0}'.format(str(num_H))):
                    with cd(metal_obj+'{0}'.format(str(num_metal_obj))):
                       db = connect('result.db')
                       lens = len(db)
                       row = list(db.select(id=lens))[0] # the last row
                       atoms = row.toatoms()
                       uni_id = 'H' + str(num_H) + '_' + metal_obj + str(num_metal_obj) + '_' +str(row.id)
                       ids.append(uni_id)
                       con_Pd = con_ele(atoms, eles[0], ref_eles=[eles[0], eles[1]])
                       con_H = con_ele(atoms, eles[2], ref_eles=[eles[0], eles[1]])
                       cons_Pd.append(con_Pd)
                       cons_H.append(con_H)
                       clease_e = row.energy
                       form_energy = formation_energy_ref_metals(atoms, clease_e, energy_ref_eles)
                       form_energies.append(form_energy)
                       clease_es.append(clease_e)
                       # db_last1.write(atoms, con_Pd=con_Pd, con_H=con_H, form_energy=form_energy, uni_id=uni_id,)
            zipped = zip(cons_Pd, cons_H, form_energies, ids, clease_es)
            pts[num_H] = np.array(list(zipped))
        with open(data_name, 'wb') as outfile:
            pickle.dump(pts, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(data_name, 'rb') as f:
            pts = pickle.load(f)
    # plot_convex_hull_ref_metals_3d_line(pts=dct_to_array(pts)) # input: dict to array
    # plot_convex_hull_ref_metals_3d_plane(pts=dct_to_array(pts))
    # plot_2d_contour(pts=dct_to_array(pts))
    # convex_hull_stacking_subplots(pts, fix_ele='H')
    
    
def collec_dict_last_data_reverse(energy_ref_eles, 
                                  db_name='results_last.db', 
                                  data_name_rev='data_last_dict_reverse.pkl', 
                                  eles=['Pd', 'Ti', 'H']):
    """get the last one in dictionary but in reverse fix_ele sequence
    
    
    Parameters: 
    
    db_name: str
         OUTPUT for last data including structures. This would need larger storage.
    data_name: str
         OUTPUT for needed last dict data saved pickle format. This would need less storage.
    """
    if 1:
        # if os.path.exists('results_last1_dic.db'): # get the last one (the lowest energies in each concentration)
            # os.remove('results_last1.db')
        # db_last1 = connect('results_last1.db')
        metal_obj = eles[1]
        pts = dict()
        for num_metal_obj in range(0, 65): # reverse metal sequence
            cons_Pd = []
            cons_H = []
            form_energies = []
            ids = []
            clease_es = []
            for num_H in range(0, 65):
                with cd('H{0}'.format(str(num_H))):
                    with cd(metal_obj+'{0}'.format(str(num_metal_obj))):
                       db = connect('result.db')
                       lens = len(db)
                       row = list(db.select(id=lens))[0] # the last row
                       atoms = row.toatoms()
                       uni_id = 'H' + str(num_H) + '_' + metal_obj + str(num_metal_obj) + '_' +str(row.id)
                       ids.append(uni_id)
                       con_Pd = con_ele(atoms, eles[0], ref_eles=[eles[0], eles[1]])
                       con_H = con_ele(atoms, eles[2], ref_eles=[eles[0], eles[1]])
                       cons_Pd.append(con_Pd)
                       cons_H.append(con_H)
                       clease_e = row.energy
                       form_energy = formation_energy_ref_metals(atoms, clease_e, energy_ref_eles)
                       form_energies.append(form_energy)
                       clease_es.append(clease_e)
                       # db_last1.write(atoms, con_Pd=con_Pd, con_H=con_H, form_energy=form_energy, uni_id=uni_id,)
            zipped = zip(cons_Pd, cons_H, form_energies, ids, clease_es)
            pts[num_metal_obj] = np.array(list(zipped))
        with open(data_name, 'wb') as outfile:
            pickle.dump(pts, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(data_name, 'rb') as f:
            pts = pickle.load(f)
    
    # plot_convex_hull_ref_metals_3d_line(pts=dct_to_array(pts)) # input: dict to array
    # plot_convex_hull_ref_metals_3d_plane(pts=dct_to_array(pts))
    # plot_convex_hull_ref_metals_3d_poly(pts=dct_to_array(pts))
    # convex_hull_stacking_subplots(pts, fix_ele='Ti') 
    # get_chem_pot_H(pts)
    # get_chem_pot_H_2d_contour(pts)
    # get_chem_pot_H_vertices(pts, num_metal_obj=0, fix_ele='Ti')
    # get_chem_pot_H_vertices_2d_contour(pts, fix_ele='Ti')
    
def plot_convex_hull_stacking_subplots(data_name, data_name_rev, fix_ele='H', metal_obj='Ti'):
    """Plot stacking convex hull in 2d
    
    Parameters
        
    pts: np arrary or dict
        points to plot 
    fix_ele: str
        element to fix, the x axis would be another element
        for example, if fix_ele is H, x axis is concentration of Pd from 0 to 1
    """
    if fix_ele == 'H':
        with open(data_name, 'rb') as f:
            pts = pickle.load(f)
    elif fix_ele == metal_obj:
        with open(data_name_rev, 'rb') as f:
            pts = pickle.load(f)
    convex_hull_stacking_subplots(pts, fix_ele=fix_ele, metal_obj=metal_obj) 
    
    
    
if __name__ == '__main__':
    
    db_name='results_last.db'
    data_name='data_last_dict.pkl'
    data_name_rev='data_last_dict_reverse.pkl'
    metal_obj = 'Sc'
    eles=['Pd', metal_obj, 'H']
    Ti_energy_ref_eles={'Pd':-1.951, metal_obj:-5.858, 'H': -7.158*0.5}
    Sc_energy_ref_eles={'Pd':-1.951, metal_obj:-3.626, 'H': -7.158*0.5}
    # collec_dict_last_data(Sc_energy_ref_eles,
    #                       db_name=db_name, 
    #                       data_name=data_name, 
    #                       eles=eles)
    # collec_dict_last_data_reverse(Sc_energy_ref_eles,
    #                               db_name=db_name, 
    #                               data_name=data_name_rev, 
    #                               eles=eles)
    with open(data_name_rev, 'rb') as f:
        pts = pickle.load(f)
    
    plot_convex_hull_ref_metals_3d_line(pts=dct_to_array(pts))
    plot_convex_hull_ref_metals_3d_plane(pts=dct_to_array(pts))
    plot_convex_hull_ref_metals_3d_poly(pts=dct_to_array(pts))
    plot_2d_contour(pts=dct_to_array(pts))
    
    
    plot_convex_hull_stacking_subplots(data_name, data_name_rev, fix_ele='H', metal_obj=metal_obj)
    plot_convex_hull_stacking_subplots(data_name, data_name_rev, fix_ele=metal_obj, metal_obj=metal_obj)
    # get_chem_pot_H_vertices(pts, num_metal_obj=0, fix_ele='Ti')
    # get_chem_pot_H_vertices_2d_contour(pts, fix_ele='Ti')
    
    get_chem_pot_H_vertices(pts, 
                            fix_ele=metal_obj, 
                            db_name=db_name, 
                            metal_obj=metal_obj, 
                            num_metal_obj=0, 
                            save_cand=False, 
                            db_candidate='PdHx_vertices.db') # PdHy
    
