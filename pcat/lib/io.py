# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 21:29:01 2021

@author: changai
"""
import numpy as np
import pandas as pd
import xlrd


def read_excel(filename='./excel_data.xlsx', sheet='Sheet1', min_col=1, max_col=5, min_row=1, max_row=9):
    """Read excel data and return step_names, obser_names and X. This function is more flexible.
    
    Parameters:
        
    filename: str
        specify filename
    sheet: str
        specify sheet in excel
    min_col: int
        column number to start
    max_col: int
        column number to end
    min_row: int
        row number to start
    max_row: int
        row number to end
    """
    row_of_tag = min_row-1 # value 0 means 1st raw in excel
    col_of_tag = min_col-1 # value 0 means 1st(A) coloum in excel
    doc = xlrd.open_workbook(filename).sheet_by_name(sheet)
    step_names = doc.row_values(rowx=row_of_tag, start_colx=min_col, end_colx=max_col)
    obser_names = doc.col_values(col_of_tag, min_row, max_row)

    X = np.empty((len(obser_names),len(step_names)))
    for i in range(len(step_names)):
        X[:,i] = np.array(doc.col_values(i+1+col_of_tag ,1+row_of_tag,len(obser_names)+1+row_of_tag)).T #raw data 
    return step_names, obser_names, X

def pd_read_excel(filename='./excel_data.xlsx', sheet='Sheet1', columns=[], **kwargs):
    """Read excel data via panda and return dataframe object. Automatically read all.
    
    Parameters:
        
    filename: str
        specify filename
    sheet: str
        specify sheet in excel
    columns: list
        specify columns you want to use
    **kwargs: key-value pairs
        specify other key-value arguments
        for example:
            usecols = ['Surface', '*HOCO'] # selected cols
            nrow = 5 # selected first n rows
            skiprows = 0 # line number to skip at the start of file
    """
    if columns == []:
        df = pd.read_excel(filename, sheet_name=sheet, index_col=0, header=0, **kwargs)
    else:
        df = pd.read_excel(filename, sheet_name=sheet, index_col=0, header=0, usecols=columns)
    return df


def read_csv(filename = './csv_data.csv', min_col=1, max_col=5):
    """Read csv data and return step_names, obser_names and X"""
    df = pd.read_csv(filename)
    raw_data = df.values
    
    cols = range(min_col, max_col)
    step_names = np.asarray(df.columns[cols])
    obser_names = raw_data[:, 0]
    
    X = raw_data[:, cols]
    return step_names, obser_names, X


def read_txt(filename, min_col=1, max_col=5):
    """Read txt data for no blank lines case and return step_names, obser_names and X"""
    file = open(filename, 'r') 
    step_names = file.readline().split()
    
    raw_data = np.loadtxt(filename, usecols=range(min_col-1, max_col), dtype=str, skiprows=1)
    obser_names = raw_data[:, min_col-1]
    
    X = raw_data[:, min_col:].astype(float)
    return step_names, obser_names, X


def read_two_column_file(filename):
    """Read txt data with two column for no blank lines case"""
    file = open(filename, 'r') 
    lines = file.readlines()
    x = []
    y = []
    for line in lines:
        p = line.split()
        x.append(float(p[0]))
        y.append(float(p[1]))
    return x, y


def read_two_column_PDOS(filename, l, threshold):
    """Read two column txt with blank lines every x lines, like PDOS"""
    x = []
    y = []
    file = open(filename, 'r') 
    lines = file.readlines()
    for line in lines[threshold*l+l:threshold*(l+1)+l]:
        p = line.split()
        x.append(float(p[0]))
        y.append(float(p[1]))
    return x, y