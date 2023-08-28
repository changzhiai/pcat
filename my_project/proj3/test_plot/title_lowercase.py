# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 15:12:11 2023

@author: changai
"""

def lowercase_titles(filename, ):
    """lowercase latex bibliography titles"""
    file = open(filename, 'r') 
    lines = file.readlines()
    lines_new = []
    for line in lines:
        l = line.split('=')
        if l[0] == '  title ':
            l[1] = l[1][2:].capitalize()
            line_new = l[0] + ' = {' + l[1]
            line = line_new
        lines_new.append(line)
    return lines_new


if __name__ == '__main__':
    lines_new = lowercase_titles(filename='bib.txt')
    with open('bib_new.txt', 'w') as f:
        for line in lines_new:
            f.write(line)