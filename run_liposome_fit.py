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
dataname =  'liposome_data_june_A.npy'
with open(dataname,'rb') as fdir:
    data = np.load(fdir,allow_pickle=True)[()]

#%%
# Define list of just 50 (62) nm diameter 
fit_pars = {}
aux_data = {}
D50 = ['DPPC_50_C_450','DPPC_50_C_455']
for thiskey in D50:
    adr = {'R':60,'dR':60*.25,'cfrac':0} 
    # Radius, polydispersity and cholesterol fraction are the same for all data
    # sets (approximately)
    aux_data[thiskey] = adr
fit_pars['aux_data'] = aux_data
fit_pars['selected_data'] = D50[0:2]
fit_pars['backgrounds'] = ['water_00008','air_new']
#%% Set up  names (B)
T1 = time.time()
timestring = str(int(T1))[-6:]
fit_pars['figname'] = 'Results/liposome_fit_results_'+timestring+'.pdf'
fit_pars['resultname'] = 'Results/liposome_fit_results_'+timestring+'.npz'
#%% (C) Define parameters for fit
fit_pars['nshort'] = 3      
fit_pars['qmin'] = 0.02
fit_pars['qmax'] = 4
fit_pars['nfev'] =200000
fit_pars['psize'] = 4
# Choose the variable for comparing different samples in 
# the final fit results plot.
fit_pars['sample variable'] = 'Temperature'
#fit_pars['sample variable'] = 'Cholesterol'
linear_regression = 1
differential_evolution = 0
fit_pars['fit_method'] = differential_evolution
#%%
#Initialize the parameters for the fit.
#
par = Parameters() 
par.add('bg1sf',value=1,vary=True,min=.5,max=1.5) 
par.add('bg2sf',value=0,vary=False,min=-.5,max=.5) 
par.add('bg',value=0,vary=False,min= -.001,max=.001) 
par.add('lbg',value=0.000,vary=False,min=0,max=1e-4) 
par.add('qbg',value=0.000,vary=False,min=0,max=1e-4) 
par.add('W',value=4.37,vary=True,min=3,max=6) 
par.add('d_H',value=.7,vary=False,min=.3,max=.9) 
par.add('d_M',value=.1,vary=True,min=.05,max=.15) 
par.add('A_H',value=107,vary=False,min=50,max=200)
par.add('A_T',value=-150,vary=True,min=-250,max=0)
par.add('A_M',value=-334,vary=False,min=-335,max=0) 
par.add('sig',value=.3,vary=True,min=.1,max=1)  
par.add('I',value=1,vary=True,min=.1,max=10) 
par.add('R0',value=50,vary=True,min=10,max=300) 
par.add('Rsig',value=15,vary=False,min=.25*10,max=.25*300) 
par.add('W_asym',value=0,vary=True,min=0,max=10) 
par.add('A_T_asym',value=0,vary=True,min=-2,max=2) 
#%% Now run the actual fit
fit_liposome(data,par,fit_pars)  
#%% Plot the fit results
plot_final_results(fit_pars)


