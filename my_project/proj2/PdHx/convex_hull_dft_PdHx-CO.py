# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:41:48 2022

@author: changai
"""

from contextlib import contextmanager
from ase.io import read, write
import numpy as np
import os
import subprocess
import time
from ase.db import connect
from ase.visualize import view
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import pickle
import matplotlib.pyplot as plt 
import matplotlib as mpl
import pandas as pd
from pcat.lib.io import pd_read_excel
import imageio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['font.size'] = 8
# mpl.use('TkAgg')

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    # try:
        # os.makedirs(newdir)
    # except OSError:
        # pass
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

def cons_Pdx(atoms):
    try:
        Pds = len(atoms[[atom.index for atom in atoms if atom.symbol=='Pd']])
    except:
        Pds = 0
    try:
        Tis = len(atoms[[atom.index for atom in atoms if atom.symbol=='Ti']])
    except:
        Tis = 0
    try:
        Hs = len(atoms[[atom.index for atom in atoms if atom.symbol=='H']])
    except:
        Hs = 0
    con_Pds = Pds/(Pds+Tis)
    return con_Pds

def cons_Hy(atoms):
    try:
        Pds = len(atoms[[atom.index for atom in atoms if atom.symbol=='Pd']])
    except:
        Pds = 0
    try:
        Tis = len(atoms[[atom.index for atom in atoms if atom.symbol=='Ti']])
    except:
        Tis = 0
    try:
        Hs = len(atoms[[atom.index for atom in atoms if atom.symbol=='H']])
    except:
        Hs = 0
    con_Hs = Hs/(Pds+Tis)
    return con_Hs

def formation_energy_ref_metals(atoms, energy):
    """
    Pure Pd: -1.951 eV/atom
    Pure Ti: -5.858 eV/atom
    H2 gas: -7.158 eV
    """
    try:
        Pds = len(atoms[[atom.index for atom in atoms if atom.symbol=='Pd']])
    except:
        Pds = 0
    try:
        Tis = len(atoms[[atom.index for atom in atoms if atom.symbol=='Ti']])
    except:
        Tis = 0
    try:
        Hs = len(atoms[[atom.index for atom in atoms if atom.symbol=='H']])
    except:
        Hs = 0
    form_e = (energy - Pds*(-1.951) - Tis*(-5.858) - 1./2*Hs*(-7.158))/(Pds+Tis+Hs)
    # print(Pds, Tis, Hs, form_e)
    return form_e

def formation_energy_ref_hydr_and_metals(atoms, energy):
    """
    Pure PdH111: -5.22219 eV/atom
    Pure Pd111: -1.59002 eV
    
    Mixing energy = Pd64Hx(111) - xPdH(111) - (64-x)Pd(111)
    """
    try:
        Pds = len(atoms[[atom.index for atom in atoms if atom.symbol=='Pd']])
    except:
        Pds = 0
    try:
        Hs = len(atoms[[atom.index for atom in atoms if atom.symbol=='H']])
    except:
        Hs = 0
    form_e = (energy - Hs*(-5.22219) - (64-Hs)*(-1.59002))/(Pds+Hs)
    # print(Pds, Tis, Hs, form_e)
    return form_e

def formation_energy_CO_ref_hydr_and_metals(atoms, energy):
    """
    Pure PdH111-CO: -5.41717 eV/atom
    Pure Pd111-CO: -1.79977 eV
    
    Mixing energy = Pd64Hx(111)_CO - xPdH(111)_CO - (64-x)Pd(111)_CO
    """
    try:
        Pds = len(atoms[[atom.index for atom in atoms if atom.symbol=='Pd']])
    except:
        Pds = 0
    try:
        Hs = len(atoms[[atom.index for atom in atoms if atom.symbol=='H']])
    except:
        Hs = 0
    form_e = (energy - Hs*(-5.41717) - (64-Hs)*(-1.79977))/(Pds+Hs)
    # print(Pds, Tis, Hs, form_e)
    return form_e

def formation_energy_ref_hydrides(atoms, energy):
    """
    reference PdH and TiH (vasp) for PdxTi(64-x)H64
    Pure PdH: -5.222 eV/PdH
    Pure TiH: -9.543 eV/TiH
    """
    try:
        Pds = len(atoms[[atom.index for atom in atoms if atom.symbol=='Pd']])
    except:
        Pds = 0
    try:
        Tis = len(atoms[[atom.index for atom in atoms if atom.symbol=='Ti']])
    except:
        Tis = 0
    try:
        Hs = len(atoms[[atom.index for atom in atoms if atom.symbol=='H']])
    except:
        Hs = 0
    form_e = (energy - Pds*(-5.222) - Tis*(-9.543))/(Pds+Tis+Hs)
    # print(Pds, Tis, Hs, form_e)
    return form_e

def get_candidates(pts, db_name='results_last1.db', db_cand_name='candidates.db'):
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

def convex_hull_ref_metals_3d_line(pts):
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
    import base64
    from io import BytesIO
    temp = BytesIO()
    fig.savefig(temp, format="png")
    fig_encode_bs64 = base64.b64encode(temp.getvalue()).decode('utf-8')
    html_string = """
    <h2>This is a test html</h2>
    <img src = 'data:image/png;base64,{}'/>
    """.format(fig_encode_bs64)
    with open("test.html", "w") as f:
        f.write(html_string) 

def convex_hull_ref_metals_3d_plane(pts):
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
    # plt.savefig('convex_hull_ref_metals_3d_plane.png')

def convex_hull_ref_metals_3d_poly(pts):
    hull = ConvexHull(pts)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # ax.plot(pts.T[0], pts.T[1], pts.T[2], "bx")
    # X = []
    # Y = []
    # Z = []
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
    # plt.savefig('convex_hull_ref_metals_3d_plane.png')


def plot_2d_contour(pts, vertices=True):
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
    # ax.plot(hull.points[:,0], hull.points[:,1], 'x')
    vertices = vertices[vertices[:,0].argsort()]
    ax.plot(vertices[:,0], vertices[:,1], 'x')
    ax.plot(vertices[:,0], vertices[:,1])

def basical_convex_hull(arr, ax, varable):
    if varable == 'H':
        col1 = arr[:,0].astype(np.float32)
        col2 = arr[:,2].astype(np.float32)
    elif varable == 'Ti':
        col1 = arr[:,1].astype(np.float32)
        col2 = arr[:,2].astype(np.float32)
    points = np.column_stack((col1, col2))
    hull = ConvexHull(points=points)
    # convex_hull_plot_2d(hull, ax=ax)
    vertices = points[hull.vertices] # get convex hull vertices
    plot_basical_convex_hull(vertices, ax=ax)

def convex_hull_stacking_subplots(pts, varable='H'):
    fig = plt.figure(figsize=(16,16))
    M = 8
    N = 9
    m1 = 0
    m2 = 0
    for k, v in pts.items():
        ax = plt.subplot(N, M, m1*M + m2 + 1)
        basical_convex_hull(v, ax, varable)
        if varable == 'H':
            plt.title('$\mathregular{Pd_{x}Ti_{64-x}}$'+ '$H_{}$'.format({64-int(k)}), x=0.5, y=0.7)
        if varable == 'Ti':
            plt.title('$Pd_{}Ti_{}Hy$'.format({64-int(k)},{k}), x=0.5, y=0.7)
        plt.xlim(0,1)
        plt.ylim(-0.4,0.4)
        m2 += 1
        if m2 == M:
            m2 = 0
            m1 += 1
    fig.add_subplot(111, frameon=False) # hide frame
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False) # hide ticks
    if varable == 'H':
        plt.xlabel('Concentration of Pd')
    elif varable == 'Ti':
        plt.xlabel('Concentration of H')
    plt.ylabel('Formation energy (eV)')
    fig.tight_layout()
    plt.show()
    fig.savefig('stacking_subplots_{}'.format(varable))

def dct_to_array(pts_dict):
    pts = np.zeros((0,4))
    for k, v in pts_dict.items():
        pts = np.concatenate((pts, v), axis=0)
    pts = pts[:,0:3].astype(np.float32)
    return pts

def views(pts_rev, num_Pd, num_H, save=False, db_target='target.db'):
    """Views one structure in specific 2d concentration
    
    pts: dict
        each key corresponding to each concentration of Pd
        for example:
            key: x, value: | x | cons_H | form_energies | ids | clease_es |
                           | x | ...... | ............. | ... | ......... |
    note: when num_metal_obj=0, pts should be pts_rev
    """
    alloy_Hx = pts_rev[64-num_Pd]
    Hcons = alloy_Hx[:,1].astype(np.float32)
    ids = alloy_Hx[:,3]
    for Hcon, index in zip(Hcons, ids):
        if Hcon == num_H/64.:
            db = connect(db_name)
            row = list(db.select(uni_id=index))[0]
            if save==True:
                db_targ = connect(db_target)
                ori_id = 'i_'+str(row.id)
                db_targ.write(row, ori_id=ori_id)
            view(row.toatoms())

def get_chem_pot_H_continuous_cons(pts, num_Ti=0):
    """
    specific at PdHx

    X H form
    """
    alloy_Hx = pts[num_Ti]  # PdHx
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

def get_chem_pot_H_vertices(pts, num_Ti=0, varable='Ti', save_cand=True, db_candidate='PdHx_vertices.db'):
    """
    specific at PdHx

    X H form ids pot_E
    """
    pts = pts[num_Ti]  # PdHx
    if varable == 'H':
        col1 = pts[:,0].astype(np.float32)
        col2 = pts[:,2].astype(np.float32)
    elif varable == 'Ti':
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
        db = connect('results_last1.db')
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

def get_chem_pot_H_vertices_2d_contour(pts, varable='Ti'):
    """
    X H form ids E_pot
    """
    Pdxcons = []
    Hycons = []
    chem_pot_H = []
    for k, v in pts.items():
        alloy_Hx = v  # PdTiHx
        if varable == 'H':
            col1 = alloy_Hx[:,0].astype(np.float32)
            col2 = alloy_Hx[:,2].astype(np.float32)
        elif varable == 'Ti':
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
    
def plot_simulated_annealing():
    """Plot simulated annealing"""
    M = 8
    N = 8
    m1 = 0
    m2 = 0
    fig = plt.figure(figsize=(16,16))
    for num_H in range(0, 64):
        # cons_H = []
        form_energies = []
        ids = []
        ax = plt.subplot(N, M, m1*M + m2 + 1)
        with cd('H{0}'.format(str(num_H))):
            db = connect('result.db')
            for row in db.select():
               # atoms = row.toatoms()
               uni_id = 'H_' + str(num_H) + '_' +str(row.id)
               ids.append(uni_id)
               clease_e = row.energy
               form_energies.append(clease_e)
        plt.plot(form_energies)
        # plt.title(f'H: {str(64-num_H)}')
        plt.text(0.45, 0.90, f'H: {str(64-num_H)}', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, color='black', fontweight='bold')
        m2 += 1
        if m2 == M:
            m2 = 0
            m1 += 1
    plt.show()
    fig.savefig('simulated_anealing.png', dpi=300, bbox_inches='tight')

def get_db_and_excel():
    """Get database"""
    if os.path.exists('results.db'):
        os.remove('results.db')
    db_tot = connect('results.db')
    cons_H = []
    form_energies = []
    ids = []
    for num_H in range(0, 65):
        with cd('H{0}'.format(str(num_H))):
            db = connect('result.db')
            for row in db.select():
               atoms = row.toatoms()
               uni_id = 'H_' + str(num_H) + '_' +str(row.id)
               ids.append(uni_id)
               con_H = cons_Hy(atoms)
               cons_H.append(con_H)
               clease_e = row.energy
               form_energy = formation_energy_ref_metals(atoms, clease_e)
               form_energies.append(form_energy)
               db_tot.write(atoms, con_H=con_H, form_energy=form_energy, uni_id=uni_id,)

def db2xls_dft(db_name):
    """Convert database to excel"""
    db = connect(db_name)
    cons_H = []
    form_energies = []
    ids = []
    energies = []
    for row in db.select(calculator='vasp'):
        atoms= row.toatoms()
        con_H = cons_Hy(atoms)
        cons_H.append(con_H)
        dft_e = row.energy
        # form_energy = formation_energy_ref_metals(atoms, dft_e)
        form_energy = formation_energy_CO_ref_hydr_and_metals(atoms, dft_e)
        form_energies.append(form_energy)
        uni_id = str(row.id) + '_' + row.name
        ids.append(uni_id)
        energies.append(row.energy)
    tuples = {'cons_H': cons_H,
      'form_energies': form_energies,
      'ids': ids,
      'Energies': energies,
    }
    df = pd.DataFrame(tuples)
    df.to_excel(xls_name, sheet_name_convex_hull, float_format='%.3f')

def plot_convex_hull_PdHx_dft(db_name, cand=False, round=1):
    """Plot convex hull"""
    df = pd_read_excel(filename=xls_name, sheet=sheet_name_convex_hull)
    cons_H = df['cons_H']
    form_energies = df['form_energies']
    ids = df['ids']
    fig, ax = plt.subplots(dpi=300)
    # fig = plt.figure(dpi=300)
    # ax = plt.axes()
    plt.plot(cons_H, form_energies, 'x')
    points = np.column_stack((cons_H, form_energies, ids))
    hull = ConvexHull(points=points[:, 0:2])
    vertices = points[hull.vertices] # get convex hull vertices
    cand_ids = vertices[:,2]
    if cand==True:
        get_PdHx_candidates_dft(cand_ids, db_name=db_name)
    plot_basical_convex_hull(vertices, ax=ax)
    plt.xlabel('Concentration of H')
    plt.ylabel('Mixing energies (eV/atom)')
    plt.xticks(np.arange(-0.00, 1.01, step=0.2))
    plt.yticks(np.arange(-0.065, 0.01, step=0.01))
    # plt.ylim([0., 0.5])
    plt.title(f'Convex hull {round} of PdHx-CO ({len(ids)} DFT data points)')
    # fig.savefig(f'./{fig_dir}/convex_hull.png', dpi=300, bbox_inches='tight')
    return fig, ax

def get_PdHx_candidates_dft(cand_ids, db_name):
    """Get and save candidates"""
    db_cand_name='dft_candidates_PdHx_CO_r7.db'
    db = connect(db_name)
    if os.path.exists(db_cand_name):
        os.remove(db_cand_name)
    db_cand = connect(db_cand_name)
    for uni_id in cand_ids:
        splits = uni_id.split('_')
        id = splits[0]
        row_vasp = db.get(id=id)
        db_cand.write(row_vasp, ori_vasp_id=int(id), calc='vasp')
        print(row_vasp.formula, ' ',)
        
        row_ce = db.get(final_struct_id=id)
        db_cand.write(row_ce, ori_vasp_id=int(id), calc='clease')
                

def plot_chem_pot_H_PdHx_discrete():
    """
    specific at PdHx for CE

    X H form ids pot_E
    """
    df = pd_read_excel(filename=xls_name, sheet=sheet_name_convex_hull)
    df = df.sort_values(by=['cons_H'], ascending=False)
    Hcons = df['cons_H'].values
    Etot = df['Energies'].values
    plt.figure()
    plt.plot(Hcons, Etot)
    plt.xlabel('Concentration of H')
    plt.ylabel('CE energy (eV)')
    plt.show()
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

def plot_animate(i):
    global system, fig_dir, data_dir, db_name, xls_name, sheet_name_convex_hull
    system = f'PdHx_top1_CO_r{i}'
    fig_dir = './figures/'
    data_dir = './data'
    db_name = f'./{data_dir}/{system}.db'
    xls_name = f'./{data_dir}/{system}.xlsx'
    sheet_name_convex_hull = 'convex_hull'
    # fig, ax = plt.subplots(figsize=(8, 6))
    fig, ax = plot_convex_hull_PdHx_dft(db_name, cand=False, round=i)
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image

if __name__ == '__main__':
    
    for i in [1, 2, 3, 4, 5, 6, 7]:
    # for i in [7]:
        system = f'PdHx_top1_CO_r{i}' # round 1, ce and dft
        # system = 'PdHx_train_r1' # round 1, ce and dft
        # system = 'PdHx_train_r5' # round 5, ce and dft
    
        fig_dir = './figures/'
        data_dir = './data'
        db_name = f'./{data_dir}/{system}.db'
        xls_name = f'./{data_dir}/{system}.xlsx'
        
        sheet_name_convex_hull = 'convex_hull'
        # plot_simulated_annealing()
        # get_db_and_excel()
        if False:
            db2xls_dft(db_name)
        if True:   
            plot_convex_hull_PdHx_dft(db_name, cand=True, round=i)
            # plot_chem_pot_H_PdHx_discrete()
        
    # kwargs_write = {'fps':1.0, 'quantizer':'nq'}
    # imageio.mimsave('./convex_hull_PdHx_CO.gif', [plot_animate(i) for i in [1, 2, 3, 4, 5, 6,]], fps=1)
