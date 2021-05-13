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
from liposome_sum_code import find_file, sum_files




names_200_nm = ['egg4_chol1_200_a','egg2_chol1_200_a','egg3_chol2_200_a',
                'egg1_chol1_200_a','egg2_chol3_200_a']
names_50_nm = ['eggPC_50_d','egg4_chol1_50_a',
               'egg2_chol3_50_b','egg1_chol1_50_a','egg3_chol2_50_a',
               'egg2_chol1_50_a']
names_water = ['water_c']
names_air = ['aircap_c']
# note, there appears to be a spurious file
# Segg2_chol3_50_b_00029_00001.dat.  This needs to be deleted, or else
# causes a problem when trying to sum

ddir = 'C:\\Users\\lluri\\Dropbox\\Runs\\2021\\March 2021 Liposomes\\Data\\SAXS\\Averaged\\'
nfiles = 40
data_sets = {}
all_sets = [names_200_nm,names_50_nm,names_water,names_air]

# normalization factor from water diffuse, converts data
# to cm^-1
norm = 1.849e-01 
for this_set in all_sets:
    print('summing {0}'.format(this_set))
    for tnam in this_set:
        print('summing {0}'.format(tnam))
        tfn = find_file(ddir,tnam)
        data_sets[tnam] = sum_files(ddir,tfn,nfiles)
        data_sets[tnam]['I'] *=norm
        data_sets[tnam]['dI'] *= norm
savename = 'liposome_data.npy'
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
    plt.plot(q,I,'-')
plt.xlabel('q (inv. nm)')
plt.ylabel('Scattering Intensity (arb)')
plt.yscale('log')
plt.xscale('log')
plt.ylim(0.008,50)
plt.xlim(.08,7)
plt.savefig('all_plots.jpg')
# plot out all data sets with water subtracted
from liposome_sum_code import subtract_sets
plt.figure('all_data_bg')
plt.clf()
water = mydata['water_c']
for tsetname in mydata.keys():
    if tsetname not in ['water_c','aircap_c']:
        tset = mydata[tsetname]
        tdif = subtract_sets(tset,water)
        q = tdif['q']
        I = tdif['I']
        dI = tdif['dI']
        plt.plot(q,I,'-')
plt.xlabel('q (inv. nm)')
plt.ylabel('Scattering Intensity - water (arb)')
plt.yscale('log')
plt.xscale('log')
plt.ylim(0.0001,1.5)
plt.xlim(.08,7)
plt.savefig('all_plots_bg.jpg')

    
    
    