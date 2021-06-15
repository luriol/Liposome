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
from liposome_sum_code import find_file, sum_files_glitch


names = ['DPPC_30_RT_1','DPPC_50_RT_1','DPPC_100_RT_1',
15         'water2','DPPC_200_RT_1','water1']



ddir = 'C:\\Users\\lluri\\Dropbox\\Runs\\2021\\June 2021 Liposomes\\Data\\SAXS\\Averaged\\'
nfiles = 30
data_sets = {}
all_sets = [names]

# normalization factor from water diffuse, converts data
# to cm^-1
norm = 1.849e-01 
for this_set in all_sets:
    print('summing {0}'.format(this_set))
    for tnam in this_set:
        print('summing {0}'.format(tnam))
        tfn = find_file(ddir,tnam)
        data_sets[tnam] = sum_files_glitch(ddir,tfn,nfiles)
        data_sets[tnam]['I'] *=norm
        data_sets[tnam]['dI'] *= norm
savename = 'liposome_data_june.npy'
with open(savename,'wb') as fdir:
    np.save(fdir,data_sets,allow_pickle=True)
#%%
with open(savename,'rb') as fdir2:
    mydata = np.load(fdir2,allow_pickle=True)[()]
    # note the strange [()] syntax, required to recover dictionary
    # from zero dimensional array
# now plot all the datasets on top of one another
plt.figure('all_data')
plt.clf()
for tsetname in mydata.keys():
    tset = mydata[tsetname]
    q = tset['q']
    I = tset['I']
    dI = tset['dI']
    plt.plot(q,I,'-',label=tsetname)
plt.xlabel('q (inv. nm)')
plt.ylabel('Scattering Intensity (arb)')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.ylim(0.008,50)
plt.xlim(.08,8)
plt.savefig('all_plots_june.jpg')
# plot out all data sets with water subtracted
from liposome_sum_code import subtract_sets
plt.figure('all_data_bg_june')
plt.clf()
water = mydata['water1']
for tsetname in mydata.keys():
    if tsetname not in ['water_c','aircap_c']:
        tset = mydata[tsetname]
        tdif = subtract_sets(tset,water)
        q = tdif['q']
        I = tdif['I']
        dI = tdif['dI']
        plt.plot(q,I,'-',label=tsetname)
plt.xlabel('q (inv. nm)')
plt.ylabel('Scattering Intensity - water (arb)')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.ylim(0.0001,1.5)
plt.xlim(.08,8)
plt.savefig('all_plots_bg_june.jpg')

    
    
    