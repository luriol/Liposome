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

def merge_data(x,y):
    """
    routine to merge identical x-data points so that data is single
    valued
    
    returns, merged data and error bars, assuming multiple data points
    at the same value combine with reduced errors
    """
    s = 0*x+1 #define error bars to all initially be equal to 1
    xout = [x[0]]
    yout = [y[0]]
    sout = [s[0]]
    for tx,ty,ts in zip(x[1:],y[1:],s[1:]):
        if xout[-1] == tx:
            yout[-1],sout[-1] = cmbwe(yout[-1],sout[-1],ty,ts)
        else:
            yout.append(ty)
            sout.append(ts)
            xout.append(tx)
    xout = np.array(xout)
    yout = np.array(yout)
    sout = np.array(sout)   
    return xout,yout,sout

def half(dset):
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
    
def subtract_sets(set1,set2,SF=1):
    # SF is adjustable scale factor
    q = set2['q']
    I2 = set2['I']
    dI2 = set2['dI']
    I1 = set1['I']
    dI1 = set1['dI']
    I = I1/SF-I2
    dI = np.sqrt(dI1**2+dI2**2)
    return {'q':q, 'I':I, 'dI':dI}




