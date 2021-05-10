# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:59:21 2021
sum_liposome_files.py

This code opens up the circularly averaged data files saved by the
beamline at 12ID and sums the 40 individual sets. It then save the data
to native numpy format so that it can be called without summing it a
second time
@author: lluri
"""
from os import listdir
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt



def find_file(ddir,searchstring):
    '''

    Parameters
    ----------
    ddir : string
        location of data directory.
    searchstring : string
        unique filename identifier.

    Returns
    -------
    string
        full file name identifer with file number stripped from end.

    '''
    names = listdir(ddir)
    tname = [name for name in names if searchstring in name][0]
    return tname[0:-10]

def cmbwe(yin1,ein1,yin2,ein2):
    '''

    Parameters
    ----------
    yin1 : numpy array
        values of first data set.
    ein1 : numpy array
        error values of first data set.
    yin2 : numpy array
        values of second data set.
    ein2 : numpy array
        error values of second data set.

    Returns
    -------
    yout : numpy array
        optimally weighted average of two data sets.
    eout : numpy array
        uncertainy in yout.

    '''
    # Now a little extra code to avoid nan's and divide by zero
    # errors.  First find all non "Nan" values and non 0 error
    # values and put in array wgood.  Then define arrays yout and
    # eout as sums of the input values.  These will be overwritten
    # except for the parts that are Nans.  However, one array might be
    # a nan, and the other not, so add them together to guarentee a
    # nan value in the output array.
    wgood = np.argwhere(np.invert(np.isnan(yin1*yin2))*(ein1*ein2))
    yout = (yin1+yin2)/2
    eout = (ein1+ein2)/2
    yout[wgood] = np.sqrt(yin1[wgood]**2/ein1[wgood]**2 +
                yin2[wgood]**2/ein2[wgood]**2)
    yout[wgood] /= np.sqrt(1/ein1[wgood]**2 + 1/ein2[wgood]**2)
    eout[wgood] = np.sqrt(ein1[wgood]**2*ein2[wgood]**2/
                (ein1[wgood]**2+ein2[wgood]**2))
    return yout,eout

def sum_files(ddir,fbase,nfiles):
    '''

    Parameters
    ----------
    ddir : string
        location of data directory.
    fbase : string
        base part of file name.
    nfiles : int
        number of runs for each file name.

    Returns
    -------
    dictionary of q, I, dI values.

    '''
    for fnum in range(nfiles):
        this_data = ddir+fbase+'_{0:05d}'.format(fnum+1)+'.dat'
        with open(this_data) as fdir:
            data = pd.read_csv(fdir,header=0,skiprows=12,delimiter='\t',names=['q','I','dI'])
        if fnum == 0:
            qout = data['q'].to_numpy()*10 # convert from inv. ang. to inv. nm
            iout = data['I'].to_numpy()
            diout = data['dI'].to_numpy()
        else:
            tival = data['I'].to_numpy()
            tdi = data['dI'].to_numpy()
            iout,diout = cmbwe(iout,diout,tival,tdi)
    yout = {'q':qout, 'I':iout, 'dI':diout}
    return yout


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
for this_set in all_sets:
    print('summing {0}'.format(this_set))
    for tnam in this_set:
        print('summing {0}'.format(tnam))
        tfn = find_file(ddir,tnam)
        data_sets[tnam] = sum_files(ddir,tfn,nfiles)
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
plt.ylim(0.08,50)
plt.xlim(.08,7)
plt.savefig('all_plots.jpg')
# plot out all data sets with water subtracted
from liposome_saxs_funs import subtract_sets
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
plt.ylim(0.001,1.5)
plt.xlim(.08,7)
plt.savefig('all_plots_bg.jpg')

    
    
    