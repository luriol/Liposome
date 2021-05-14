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
from datetime import datetime
import os
import time
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
multi_results = {}
for dind, dsetname in enumerate(data.keys()):
    if dsetname not in ['water_c','aircap_c']:
        print('{0:20s}\t{1}'.format(dsetname,dind))
        #%%
        # shorten datasets and backgrounds
        nshort = 4
        water = shorten(data['water_c'],nshort)
        air = shorten(data['aircap_c'],nshort)
        fitset = shorten(data[dsetname],nshort)
        #%%
        ''' Initialize the parameters for the fit.
        '''
        par = Parameters()
        par.add('bg1sf',value=1.05,vary=True,min=.5,max=1.5) # optional linear background, not used
        par.add('bg2sf',value=.05,vary=True,min=-1,max=1) # optional quadratic background, not used
        par.add('bg',value=0.002,vary=True,min=0,max=.01)
        par.add('lbg',value=0.000,vary=False,min=-.1,max=.1)
        par.add('qbg',value=0.000,vary=False,min=-1,max=1)
        par.add('W',value=4.37,vary=True,min=3,max=6)
        par.add('d_H',value=.609,vary=False,min=.3,max=.9)
        par.add('d_M',value=.1,vary=True,min=0.05,max=.15)
        par.add('A_H',value=70,vary=False,min=50,max=90)  # since scale factor is arbitrary, fix one value
        par.add('A_T',value=-84.7,vary=True,min=-100,max=-50)
        par.add('A_M',value=-333,vary=False,min=-334,max=-100) # fix methyl amplitude at -water value (e.g. assume 0)
        par.add('sig',value=.3,vary=True,min=.2,max=.5)  
        par.add('I',value=1,vary=True,min=.1,max=10)
        par.add('R0',value=aux_data[dsetname]['R'],vary=False,min=40,max=70)
        par.add('Rsig',value=aux_data[dsetname]['dR'],vary=False,min=.25*40,max=.25*70)
        #%%
        # extract intensity from data structure for fitting
        q = fitset['q']
        I = fitset['I']
        bgfun1 = water['I']
        bgfun2 = air['I']
        w = 1/np.sqrt(fitset['dI']**2 + water['dI']**2)
        # Truncate the data range so as not to fit below qmin and above qmax
        qmin = .1
        qmax = 6
        w[q<qmin] = 0
        w[q>qmax] = 0
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
            print('running differential evolution fit')
            result = liposome_model.fit(I,par,q=q,bgfun1 = bgfun1,
                        bgfun2=bgfun2,weights=w,
                        method='differential_evolution',
                        fit_kws={'maxfev':20000,'popsize':psize}
                        )
        fit_results = result.fit_report()
        print(fit_results)
        #%%
        # Create background subtracted data 
        fitpar = result.params
        bg1sf = fitpar['bg1sf'].value
        bg2sf = fitpar['bg2sf'].value
        bg = fitpar['bg'].value
        lbg = np.abs(fitpar['lbg'].value)
        qbg = np.abs(fitpar['qbg'].value)
        bgdata = I -bg  - lbg*q - qbg*q**2-bgfun1*bg1sf-bgfun2*bg2sf
        # Create background subtracted fit
        bgpar = copy(fitpar)
        bgpar['bg'].value = 0
        bgpar['lbg'].value = 0
        bgpar['qbg'].value = 0
        nullbg1 = copy(bgfun1)*0
        nullbg2 = copy(bgfun2)*0
        bgfit = liposome_model.eval(bgpar,bgfun1 = nullbg1,bgfun2=nullbg2,q=q)
        #%%
        # plot background subtracted data - fit
        plt.figure('Fit_Minus_Background')
        plt.clf()
        plt.errorbar(q,bgdata,fitset['dI'],fmt='ks',label='data')
        plt.plot(q,bgfit,'-r',label='fit')
        plt.yscale('log')
        plt.ylim(0.000001,.2)
        plt.title(dsetname)
        plt.xlabel('q (nm${-1}$)')
        plt.ylabel(r'$\frac{1}{V}\frac{d\Sigma}{d\Omega}$ (cm $^{-1}$)')
        plt.legend()
        plt.show()
        #%%
        #plot the profile in real space corresponding to the fit
        plt.figure('Real_Space_Fit')
        plt.clf()
        P3 = profile([])
        P3.load_par(result.params)
        P3.draw_rho(offset = 333.3, ymin = 0, ymax = 450, color='blue')
        plt.title(dsetname)
        plt.show()
        #%%
        # Archive data and fit
        print('Archiving data')
        # create a unique directory name to save data and fit 
        timestring = str(int(time.time()))[-6:]
        if not os.path.exists('Results'):
            os.makedirs('Results')
        if not os.path.exists('Results'):
            os.makedirs('Results\\{0:s}'.format(dsetname))
        dname = 'Results\\{0:s}\\{1:s}\\'.format(dsetname,timestring)
        os.makedirs(dname)
        plt.figure('Fit_Minus_Background')
        plt.savefig(dname+'Fit_minus_background')
        plt.figure('Real_Space_Fit')
        plt.savefig(dname +'Real_Space_Fit')
        with open(dname +'Fit_Results.txt','w') as fd:
            print(fit_results,file=fd)
        outdata = {'water':water,'air':air,
                       'result':result,'Profile':P3,'qmin':qmin,
                       'qmax':qmax,'fitset':fitset,
                       'dsetname':dsetname,
                       'aux_data':aux_data[dsetname]}
        with open(dname +'Fit_Data.npy','wb') as fd:
            np.save(fd,outdata,allow_pickle=True)
        #%%
        multi_results[dsetname] = outdata
        print('Fit to dataset {0:s} complete'.format(dsetname))
timestring = str(int(time.time()))[-6:]
multi_name = 'liposome_multi_fit_results'+timestring+'.npy'
with open('Results\\'+multi_name,'wb') as fd:
    np.save(fd,multi_results,allow_pickle=True)
#%%
# collect summary results
sigs = np.array([])
dsigs = np.array([])
Ws = np.array([])
dWs = np.array([])
chi2 = np.array([])
Cf = np.array([])
#Cf = np.array([0.2, 0.33, 0.4, 0.5, 0.6, 0, 0.2, 0.6, 0.5, 0.4, 0.33])
for tout in multi_results:
    tdic = multi_results[tout]
    result=tdic['result']
    params = result.params   
    chi2 = np.append(chi2,result.redchi)
    sigs = np.append(sigs,params['sig'].value)
    Ws = np.append(Ws,params['W'].value)
    dsigs = np.append(dsigs,params['sig'].stderr)
    dWs = np.append(dWs,params['W'].stderr)
    Cf  = np.append(Cf,tdic['aux_data']['cfrac'])
plt.figure('Ws')
plt.clf()
gg = chi2<2
plt.errorbar(Cf[gg],Ws[gg],dWs[gg],fmt='ks')
plt.xlabel('cholesterol fraction')
plt.ylabel('bilayer thickness')
plt.savefig('Results\\'+'thickness.png')
plt.figure('sigs')
plt.clf()
plt.errorbar(Cf[gg],sigs[gg],dsigs[gg],fmt='ks')
plt.xlabel('cholesterol fraction')
plt.ylabel('bilayer roughness')
plt.savefig('Results\\'+'roughness.png')
plt.figure('chi2')
plt.clf()
plt.plot(Cf,chi2,'ks')
plt.xlabel('cholesterol fraction')
plt.ylabel('fit chi-squared (reduced) ')
plt.savefig('Results\\'+'redchi.png')
