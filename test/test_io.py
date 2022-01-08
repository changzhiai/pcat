# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:45:24 2022

@author: changai
"""
from pcat.lib.io import read_excel

def test_io():
    step_names, obser_names, X = read_excel(filename='examples/data/data.xlsx', 
                                            sheet='free_energy', 
                                            min_col=1, 
                                            max_col=5, 
                                            min_row=1, 
                                            max_row=15)
    assert len(step_names)==4
    assert len(obser_names)==14
    assert X.shape==(14,4)


