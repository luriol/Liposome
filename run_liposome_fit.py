# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:09:27 2021

Code to run a fit of liposome data file
This is a copy of run_liposome_fit, but with the option to select
a single file to run replaced by running all the files in a loop

@author: Laurence Lurio

"""

from matplotlib import pyplot as plt
from liposome_saxs_funs import  profile, liposome_model
from liposome_sum_code import shorten
import numpy as np
from lmfit import Parameters
from copy import copy
import os
import time
from liposome_plotting import make_plots, plot_final_results
from matplotlib.backends.backend_pdf import PdfPages

#%%
# read in saved data and open
dataname =  'liposome_data.npy'
with open(dataname,'rb') as fdir:
    data = np.load(fdir,allow_pickle=True)[()]
#%%
# create auxilary array of data that is associated with individual
# datasets
aux_data = {'egg4_chol1_200_a':{'cfrac':0.2,'R':200,'dR':60},
            'egg2_chol1_200_a':{'cfrac':0.33,'R':200,'dR':60},
            'egg3_chol2_200_a':{'cfrac':0.4,'R':200,'dR':60},
            'egg1_chol1_200_a':{'cfrac':0.5,'R':200,'dR':60},
            'egg2_chol3_200_a':{'cfrac':0.6,'R':200,'dR':60},
            'eggPC_50_d':{'cfrac':0,'R':62,'dR':15},
            'egg4_chol1_50_a':{'cfrac':.2,'R':62,'dR':15},
            'egg2_chol3_50_b':{'cfrac':.6,'R':62,'dR':15},
            'egg1_chol1_50_a':{'cfrac':.5,'R':62,'dR':15},
            'egg3_chol2_50_a':{'cfrac':.4,'R':62,'dR':15},
            'egg2_chol1_50_a':{'cfrac':.33,'R':62,'dR':15},
            }
#%%
if not os.path.exists('Results'):
    os.mkdir('Results')
timestring = str(int(time.time()))[-6:]
figname = 'Results/liposome_fit_results_'+timestring+'.pdf'
resultname = 'Results/liposome_fit_results_'+timestring+'.npz'
pp = PdfPages(figname)
multi_results_out = {}
for dind, dsetname in enumerate(data.keys()):
    selection = 1 # select all sets
    #selection = (dsetname in  ['eggPC_50_d','egg2_chol3_50_b','egg4_chol1_200_a','egg1_chol1_200_a'])
    if (dsetname not in ['water_c','aircap_c'] and selection):
        print('{0:20s}\t{1}'.format(dsetname,dind))
        #%%
        # shorten datasets and backgrounds
        nshort = 4
        water = shorten(data['water_c'],nshort)
        air = shorten(data['aircap_c'],nshort)
        fitset = shorten(data[dsetname],nshort)
        #%%
        #Initialize the parameters for the fit.
        #
        par = Parameters()
        par.add('bg1sf',value=1.00,vary=True,min=.25,max=4) 
        par.add('bg2sf',value=0,vary=False,min=-.1,max=.1) 
        par.add('bg',value=0,vary=False,min= -.001,max=.001) 
        par.add('lbg',value=0.000,vary=False,min=0,max=1e-4) 
        par.add('qbg',value=0.000,vary=False,min=0,max=1e-4) 
        par.add('W',value=4.37,vary=True,min=3,max=6) 
        par.add('d_H',value=.7,vary=False,min=.3,max=.9) 
        par.add('d_M',value=0,vary=True,min=-.001,max=.15) 
        par.add('A_H',value=107,vary=False,min=50,max=200)
        par.add('A_T',value=-150,vary=True,min=-250,max=0)
        par.add('A_M',value=-334,vary=False,min=-335,max=0) 
        par.add('sig',value=.3,vary=True,min=.2,max=.5)  
        par.add('I',value=1,vary=True,min=.1,max=10) 
        par.add('R0',value=aux_data[dsetname]['R'],vary=False,min=40,max=300) 
        par.add('Rsig',value=aux_data[dsetname]['dR'],vary=False,min=.25*40,max=.25*300) 
        par.add('W_asym',value=0,vary=True,min=-0.01,max=2) 
        par.add('A_T_asym',value=0,vary=True,min=-1,max=1) 
        #%%
        # extract intensity from data structure for fitting
        q = fitset['q']
        I = fitset['I']
        bgfun1 = water['I']
        bgfun2 = air['I']
        w = np.zeros(len(fitset['dI']))            
        # Truncate the data range so as not to fit below qmin and above qmax
        qmin = 0.3
        qmax = 6
        rr = (q>qmin)*(q<qmax)
        w[rr] = 1/np.sqrt(fitset['dI'][rr]**2 + water['dI'][rr]**2 +
                          (0.001*fitset['I'][rr])**2)
        #w[rr] = 1/fitset['dI'][rr]**2 
        w[1^rr] = 0
        #%% plot the fit results
        # choose fit method, least squares = 1, differential evolution =0
        # least squares is faster, but less accurate
        fit_method = 0
        if (fit_method):
            print('running least squares fit')
            result = liposome_model.fit(I,par,q=q,bgfun1 = bgfun1,
                        bgfun2=bgfun2,weights=w)
        else:
            psize = 64
            print('running differential evolution fit ')
            result = liposome_model.fit(I,par,q=q,bgfun1 = bgfun1,
                        bgfun2=bgfun2,weights=w,
                        method='differential_evolution',max_nfev=200000,
                        fit_kws={'popsize':psize}
                        )
            # append cholesterol fraction to result values
            result.values['cfrac'] = aux_data[dsetname]['cfrac']
            result.values['psize'] = psize
            result.values['qmin'] = qmin
            result.values['qmax'] = qmax
            print('Finished fit #2 redchi = {0:7.2f}'.format(result.redchi))
            multi_results_out[dsetname] = {'result':result,
                'cfrac':aux_data[dsetname]['cfrac'],'dsetname':dsetname}
        make_plots(pp,result,dsetname)
        print('Fit to dataset {0:s} complete'.format(dsetname))
timestring = str(int(time.time()))[-6:]
multi_name = 'liposome_multi_fit_results'+timestring+'.npy'
with open(resultname,'wb') as fd:
    np.save(fd,multi_results_out,allow_pickle=True)
plot_final_results(pp,resultname)
pp.close()

