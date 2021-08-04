# -*- coding: utf-8 -*-
"""
Created on Tue May 11 08:36:24 2021
Routines for handling liposome data files, reading and summing
@author: lluri
"""
from os import listdir
import pandas as pd
import numpy as np

def find_file(ddir,searchstring):
    '''

    This is a utility to locate a group of files with the 
    same name but different file numbers.  It takes the name 
    of the data directory to search "ddir" 
    and the string to find "searchstring" and returns 
    all files in that directory which have the 
    searchstring in their name.
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
    print('summing {0}'.format(fbase))
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

def cmb_set(s1,s2):
    yin1 = s1['I']
    ein1 = s1['dI']
    yin2 = s2['I']
    ein2 = s2['dI']
    yout,sout = cmbwe(yin1,ein1,yin2,ein2)
    return {'q':s1['q'],'I':yout,'dI':sout}

def deglitch(ds1,ds2,cut):
    '''
    This routine is used to remove glitches in the data 
    resulting (typically) from cosmic ray hits on the 
    detector during the measurement.  ds1 and ds2 are 
    two data dictionaries containing q, I and dI.  
    "cut" is a threshold used to compare the data.  
    If the first data set exceeds the second by more 
    than the fractional difference specified by the 
    cut value, then those points in ds1 are replaced 
    by the corresponding points from ds2.  
    This routine is used in the routine sum_files_glitch 
    to avoid summing glitchy points when combining two files.
    '''
    i1 = ds1['I']
    i2 = ds2['I']
    e1 = ds1['dI']
    e2 = ds2['dI']
    y = 2*(i1-i2)/(i1+i2)
    rr = y > cut
    for inum, bval in enumerate(rr):
        if bval:
            i1[inum]=i2[inum]
            e1[inum]=e2[inum]       
    return {'q':ds1['q'],'I':i1,'dI':e1}

def sum_files_glitch(ddir,fbase,nfiles):
    cut = 0.1
    da = []
    for fnum in range(nfiles):
        this_data = ddir+fbase+'_{0:05d}'.format(fnum+1)+'.dat'
        with open(this_data) as fdir:
            data = pd.read_csv(fdir,header=0,skiprows=12,delimiter='\t',names=['q','I','dI'])
            qout = data['q'].to_numpy()*10 # convert from inv. ang. to inv. nm
            iout = data['I'].to_numpy()
            diout = data['dI'].to_numpy()
            dstruc = {'q':qout,'I':iout,'dI':diout}
        da.append(dstruc)
    for fnum in range(nfiles):
        if fnum == 0:
            sum = deglitch(da[2],da[0],cut)
        else:
            sum = cmb_set(sum,deglitch(da[fnum],da[fnum-1],cut))
    return(sum)


def half(dset):
    '''
    This routine takes a dataset 
    (e.g. dictionary of q, I and dI) 
    and merges all pairs of adjacent points in q, 
    so that there are half as many q points with 
    correspondingly smaller error bars.  
    '''
    q = dset['q']
    I = dset['I']
    dI = dset['dI']
    wgi = np.where(~np.isnan(I))[0]
    q = q[wgi]
    I = I[wgi]
    dI = dI[wgi]
    nlen = int(2*np.floor(len(q)/2))
    q = q[0:nlen]
    q.shape
    I = I[0:nlen]
    dI = dI[0:nlen]
    eve = np.arange(0,nlen,2).astype(int)
    odd = np.arange(1,nlen,2).astype(int)
    qout = (q[eve]+q[odd])/2
    #Iout = (I[eve]+I[odd])/2
    #dIout = np.sqrt(dI[eve]**2 + dI[odd]**2)/2
    Iout,dIout = cmbwe(I[eve],dI[eve],I[odd],dI[odd])
    return {'q':qout, 'I':Iout, 'dI':dIout}

def shorten(dset,nhalf):
    '''
    This routine applies half to the dataset dset 
    nhalf times, to reduce the number of point 
    by $2^{(-nhalf)}$.
    '''
    for nit in range(nhalf):
        dset = half(dset)
    return dset
    
def subtract_sets(set1,set2,SF=1):
    '''
    Subtracts the intensity in set2 from the intensity 
    in set1 after scaling set1 by 1/SF.  
    The errors are combined appropriately 
    to yeild a (larger) final error.
    '''
    # SF is adjustable scale factor
    q = set2['q']
    I2 = set2['I']
    dI2 = set2['dI']
    I1 = set1['I']
    dI1 = set1['dI']
    I = I1/SF-I2
    dI = np.sqrt(dI1**2/SF**2+dI2**2)
    return {'q':q, 'I':I, 'dI':dI}

