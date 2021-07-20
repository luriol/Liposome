# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:09:27 2021

Code to run a fit of liposome data file
This is a copy of run_liposome_fit, but with the option to select
a single file to run replaced by running all the files in a loop

@author: Laurence Lurio

"""

from liposome_saxs_funs import  liposome_model
from liposome_sum_code import shorten
import numpy as np
from lmfit import Parameters
import os
import time
from liposome_plotting import plot_final_results
from matplotlib.backends.backend_pdf import PdfPages

#%%
# read in saved data and open
dataname =  'liposome_data.npy'
with open(dataname,'rb') as fdir:
    data = np.load(fdir,allow_pickle=True)[()]
#%%
# create auxilary array of data that is associated with individual
# datasets
aux_data = {'egg4_chol1_200_a':{'cfrac':0.2,'R':200/2,'dR':60/2},
            'egg2_chol1_200_a':{'cfrac':0.33,'R':200/2,'dR':60/2},
            'egg3_chol2_200_a':{'cfrac':0.4,'R':200/2,'dR':60/2},
            'egg1_chol1_200_a':{'cfrac':0.5,'R':200/2,'dR':60/2},
            'egg2_chol3_200_a':{'cfrac':0.6,'R':200/2,'dR':60/2},
            'eggPC_50_d':{'cfrac':0,'R':62/2,'dR':15/2},
            'egg4_chol1_50_a':{'cfrac':.2,'R':62/2,'dR':15/2},
            'egg2_chol3_50_b':{'cfrac':.6,'R':62/2,'dR':15/2},
            'egg1_chol1_50_a':{'cfrac':.5,'R':62/2,'dR':15/2},
            'egg3_chol2_50_a':{'cfrac':.4,'R':62/2,'dR':15/2},
            'egg2_chol1_50_a':{'cfrac':.33,'R':62/2,'dR':15/2},
            }
#%%
# Define list of just 50 (62) nm diameter 
D50 = ['eggPC_50_d','egg4_chol1_50_a','egg2_chol3_50_b',
      'egg1_chol1_50_a','egg3_chol2_50_a','egg2_chol1_50_a']
#%% Set up fit directory and names
if not os.path.exists('Results'):
    os.mkdir('Results')
T1 = time.time()
timestring = str(int(T1))[-6:]
figname = 'Results/liposome_fit_results_'+timestring+'.pdf'
resultname = 'Results/liposome_fit_results_'+timestring+'.npz'
pp = PdfPages(figname)
multi_results_out = {}
#%% Run through data and fit
select_all = 0  # flag to select all datasets or just a subset
nshort = 4      # number of times to average data set by 1/2
# Truncate the data range so as not to fit below qmin and above qmax
qmin = 0.1
qmax = 5
eps = 1e-3
water = shorten(data['water_c'],nshort)
air = shorten(data['aircap_c'],nshort)
#%%
#Initialize the parameters for the fit.
#
par = Parameters()
par.add('bg1sf',value=1.00,vary=True,min=.25,max=4) 
par.add('bg2sf',value=0,vary=True,min=-.5,max=.5) 
par.add('bg',value=0,vary=False,min= -.001,max=.001) 
par.add('lbg',value=0.000,vary=False,min=0,max=1e-4) 
par.add('qbg',value=0.000,vary=False,min=0,max=1e-4) 
par.add('W',value=4.37,vary=True,min=3,max=6) 
par.add('d_H',value=.7,vary=False,min=.3,max=.9) 
par.add('d_M',value=.1,vary=True,min=.05,max=.15) 
par.add('A_H',value=107,vary=False,min=50,max=200)
par.add('A_T',value=-150,vary=True,min=-250,max=0)
par.add('A_M',value=-334,vary=False,min=-335,max=0) 
par.add('sig',value=.3,vary=True,min=.2,max=.75)  
par.add('I',value=1,vary=True,min=.1,max=10) 
par.add('R0',value=50,vary=False,min=10,max=300) 
par.add('Rsig',value=15,vary=False,min=.25*10,max=.25*300) 
par.add('W_asym',value=0,vary=False,min=-2,max=2) 
par.add('A_T_asym',value=0,vary=False,min=-2,max=2) 
#%%
# loop through fit
ngood = 0
for dind, dsetname in enumerate(data.keys()):
    if select_all:
        selection = 1 # select all sets
        nsets = len(data.keys)-2
    else:
        selection = (dsetname in D50)
        nsets = len(D50)
    if (dsetname not in ['water_c','aircap_c'] and selection):
        ngood += 1
        print('dataset {0:20s}\t (set number {1}) {2} out of {3}'.format(dsetname,
        dind,ngood,nsets))
        fitset = shorten(data[dsetname],nshort)
        par['R0'].value = aux_data[dsetname]['R']
        par['Rsig'].value = aux_data[dsetname]['dR']
        q = fitset['q']
        I = fitset['I']
        w = np.zeros(len(fitset['dI']))            
        rr = (q>qmin)*(q<qmax)
        sys_err = 0 # estimate of systematica normalization error
        w[rr] = 1/np.sqrt(fitset['dI'][rr]**2 + water['dI'][rr]**2 +
                          (sys_err*fitset['I'][rr])**2 +(sys_err*water['I'][rr])**2)
        w[1^rr] = 0 # "eps" is small but nonzero
        # choose fit method, least squares = 1, differential evolution =0
        fit_method = 0
        if (fit_method):
            print('running least squares fit')
            result = liposome_model.fit(I,par,q=q,bgfun1 = water['I'],
                        bgfun2=air['I'],weights=w)
        else:
            psize = 16
            print('running differential evolution fit ')
            result = liposome_model.fit(I,par,q=q,bgfun1 = water['I'],
                        bgfun2=air['I'],weights=w,
                        method='differential_evolution',max_nfev=200000,
                        fit_kws={'popsize':psize}
                        )
            # append additional variables to result values
            result.values['cfrac'] = aux_data[dsetname]['cfrac']
            result.values['psize'] = psize
            result.values['qmin'] = qmin
            result.values['qmax'] = qmax
            print('Finished fit #2 redchi = {0:7.2f}'.format(result.redchi))
            multi_results_out[dsetname] = {'result':result,
                'cfrac':aux_data[dsetname]['cfrac'],'dsetname':dsetname}
        #make_plots(pp,result,dsetname)
        print('Fit to dataset {0:s} complete'.format(dsetname))
timestring = str(int(time.time()))[-6:]
multi_name = 'liposome_multi_fit_results'+timestring+'.npy'
with open(resultname,'wb') as fd:
    np.save(fd,multi_results_out,allow_pickle=True)
plot_final_results(pp,resultname)
pp.close()
T2 = time.time()
print('total elapsed time {0:4.1f} s'.format(T2-T1))

