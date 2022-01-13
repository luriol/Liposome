# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:09:27 2021

Code to run a fit of liposome data file
This is a copy of run_liposome_fit, but with the option to select
a single file to run replaced by running all the files in a loop

@author: Laurence Lurio

"""
import numpy as np
from lmfit import Parameters
import time
from liposome_fit_funs import fit_liposome, plot_final_results
#%% (A)
# read in saved data and open
dataname =  'Data/liposome_data.npy'
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
fit_pars = {}
fit_pars['aux_data'] = aux_data
D50 = ['eggPC_50_d','egg4_chol1_50_a','egg2_chol3_50_b',
      'egg1_chol1_50_a','egg3_chol2_50_a','egg2_chol1_50_a']
fit_pars['selected_data'] = D50[0:2]
fit_pars['backgrounds'] = ['water_c','aircap_c']
#%% Set up  names (B)
T1 = time.time()
timestring = str(int(T1))[-6:]
fit_pars['figname'] = 'Results/liposome_fit_results_'+timestring+'.pdf'
fit_pars['resultname'] = 'Results/liposome_fit_results_'+timestring+'.npz'
#%% (C) Define parameters for fit
fit_pars['nshort'] = 4      
fit_pars['qmin'] = 0.1
fit_pars['qmax'] = 5
fit_pars['nfev'] =200000
fit_pars['psize'] = 16
linear_regression = 1
differential_evolution = 0
fit_pars['fit_method'] = differential_evolution
fit_pars['sample variable'] = 'Cholesterol'
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
#%% Now run the actual fit
result = fit_liposome(data,par,fit_pars)  
#%% Plot the fit results
plot_final_results(fit_pars)


