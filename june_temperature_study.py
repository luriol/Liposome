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
nshort = 4 # number of times to bin (shorten) the data to reduce noise
qrmin = 2.3 # minimum of normalization range
qrmax = 3.0 # max of normalization range
SFM = 0.95 # scale factor multiplier for water subtraction
water = shorten(sum_files_glitch(ddir,tfn,nfiles),nshort)
n1 = np.mean(water['I'][(water['q']>qrmin)*(water['q']<qrmax)])
plt.figure('DPPC_data_bg_subtracted')
plt.clf()
# plt.figure('DPPC_data_sample_norm')
# plt.clf()
for t in range(370,460,5):
    tnam = 'DPPC_50_B_{0:3d}'.format(t)
    tfn = find_file(ddir,tnam)

    dppc = shorten(sum_files_glitch(ddir,tfn,nfiles),nshort)
    n2 = np.mean(dppc['I'][(dppc['q']>qrmin)*(dppc['q']<qrmax)])
    #tdif = subtract_sets(dppc,water,SF=SFM*n2/n1)  
    tc = 420
    dt = 100
    lw = np.abs((t-tc)/dt)
    linewidth = lw*3
    plt.figure('DPPC_data_bg_subtracted')
    #plt.plot(tdif['q'],tdif['I'],label='{0:4.1f}C'.format(t/10.0))
    plt.plot(dppc['q'],dppc['I']/dppc['I'][0],label='{0:4.1f}C'.format(t/10.0))
    plt.xlabel('q (inv. nm)')
    plt.ylabel('Scattering Intensity (arb)')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('pure DPPC 50 nm vs. temperature (increasing T)')
    
    plt.legend()
    plt.ylim(.00001)
    plt.xlim(.01,8)
