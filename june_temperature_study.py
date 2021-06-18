# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:59:21 2021
sum_liposome_files.py

May 11, added normalization factor to convert to cross sectio
in inv. cm.

This code opens up the circularly averaged data files saved by the
beamline at 12ID and sums the 40 individual sets. It then save the data
to native numpy format so that it can be called without summing it a
second time
@author: lluri
"""

import numpy as np
from matplotlib import pyplot as plt
from liposome_sum_code import find_file, sum_files_glitch, subtract_sets, shorten


dhead = 'C:\\Users\\lluri\\Dropbox\\Runs\\2021\\'
ddir = dhead + 'June 2021 Liposomes\\SAXS\\Averaged\\'
tnam = 'water2'
tfn = find_file(ddir,tnam)
nfiles = 10
water = shorten(sum_files_glitch(ddir,tfn,nfiles),4)
n1 = np.mean(water['I'][(water['q']>3)*(water['q']<5)])
plt.figure('DPPC_data')
plt.clf()
for t in range(370,470,5):
    tnam = 'DPPC_50_B_{0:3d}'.format(t)
    tfn = find_file(ddir,tnam)

    dppc = shorten(sum_files_glitch(ddir,tfn,nfiles),4)
    n2 = np.mean(dppc['I'][(dppc['q']>3)*(dppc['q']<5)])
    tdif = subtract_sets(dppc,water,SF=0.99*n2/n1)   
    plt.plot(tdif['q'],tdif['I'],label='{0:4.1f}C'.format(t/10.0))
    plt.xlabel('q (inv. nm)')
    plt.ylabel('Scattering Intensity (arb)')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('pure DPPC 50 nm vs. temperature (increasing T)')
    plt.legend()
    plt.ylim(0.0001,50)
    plt.xlim(.1,8)
