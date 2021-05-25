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

from matplotlib.backends.backend_pdf import PdfPages

def make_plots(pp,result,dsetname):
    #
    # First figure, printout of fit report
    #
    fit_results = result.fit_report()
    fig = plt.figure('results')
    fig.clf()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.text(.02,.02,fit_results,fontsize='x-small')
    ax.text(.75,.9,dsetname,fontsize='large')
    ax.axis('off')
    fig.show()
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    #
    # Second figure background subtracted data
    #
    vals = result.values
    q = result.userkws['q']
    bgfun1 = result.userkws['bgfun1']
    bgfun2 = result.userkws['bgfun2']
    bg = (vals['bg'] + q*vals['lbg'] +
        q**2*vals['qbg'] + 
        bgfun1*vals['bg1sf'] +
        bgfun2*vals['bg2sf'])
    bgdata = result.data-bg
    fit = result.best_fit - bg
    plt.figure('Fit_minus_bg')
    plt.clf()
    plt.errorbar(q,bgdata,1/result.weights,fmt='-k',label='data')
    plt.plot(q,fit,'-r',label='fit')
    plt.yscale('log')
    plt.ylim(1e-6,1)
    plt.title(dsetname) 
    plt.xlabel('q (nm${-1}$)')
    plt.ylabel(r'$\frac{1}{V}\frac{d\Sigma}{d\Omega}$ (cm $^{-1}$)')
    plt.legend()
    plt.show()
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    # #%%
    # Plot residuals
    plt.figure('Fractonal Residuals')
    plt.clf()
    plt.errorbar(q,(result.data-result.best_fit)/result.data,
        1/result.weights/result.data,fmt='ks',label='data')
    plt.yscale('linear')
    plt.ylim(-.025,.025)
    plt.title(dsetname)
    plt.xlabel('q (nm${-1}$)')
    plt.ylabel('residuals')
    plt.legend()
    plt.show()
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    # #%%
    #plot the profile in real space corresponding to the fit
    plt.figure('Real_Space_Fit')
    plt.clf()
    P3 = profile([])
    P3.load_par(result.params)
    P3.draw_rho(offset = 333.3, ymin = 0, ymax = 450, color='blue')
    plt.title(dsetname)
    plt.show()
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    #%%



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
timestring = str(int(time.time()))[-6:]
figname = 'Results/liposome_fit_results_'+timestring+'.pdf'
resultname = 'Results/liposome_fit_results_'+timestring+'.npz'
pp = PdfPages(figname)
multi_results_out = {}
for dind, dsetname in enumerate(data.keys()):
    selection = 1 # select all sets
    #selection = (dsetname in  ['eggPC_50_d'])
    if (dsetname not in ['water_c','aircap_c'] and selection):
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
        par.add('bg1sf',value=1.05,vary=True,min=.25,max=4) # scale factor for water background
        par.add('bg2sf',value=0,vary=False,min=-.1,max=.1) # scale factor for air background (not used)
        par.add('bg',value=0,vary=False,min= -.001,max=.001) # optional constant background (not used)
        par.add('lbg',value=0.000,vary=False,min=-.1,max=.1) # optional linear background (not used)
        par.add('qbg',value=0.000,vary=False,min=-1,max=1) # optional quadratic background (not used)
        par.add('W',value=4.37,vary=True,min=3,max=6) # overall bilayer width
        par.add('d_H',value=.609,vary=False,min=.3,max=.9) # head group thickness. 
        par.add('d_M',value=.1,vary=True,min=0.05,max=.15) # Methyl group thickness
        # note there is no parameter for tail group thickness.  This is calculated by subtracting 
        # the other widths from the total bilayer width.
        par.add('A_H',value=70,vary=False,min=50,max=90)  # The amplitude of the head group
        # The fit is only sensative to the relative amplitudes of the layers, thus
        # one amplitude factor is arbitrary, and should not be varied.  The 
        # head group amplitude is thus fixed
        par.add('A_T',value=-84.7,vary=True,min=-100,max=-50) # Tail group amplitude
        par.add('A_M',value=-333,vary=False,min=-334,max=-100) # The methyl group  amplitude 
        #is set to be -333, which is negative the electron density (in e-/nm^3)
        #of water.  This is because water is the reference point for amplitudes
        # (the bilayer is submerged in water).  An amplitude of -333 is equivilent
        # to vaccum, so it is an overestimate, as the hydrogens in methyl have
        # a small but nonzero electron density.  However, this is a reasonable 
        # approximation, and since the roughness is typically large enough to no
        # see this dip to high accuracy, it is good enough to just vary the thicknes
        # of the methyl group. 
        par.add('sig',value=.3,vary=True,min=.2,max=.5)  # bilayer roughness.  Right now all layers
        # are fixed to hae the same roughness value
        par.add('I',value=1,vary=True,min=.1,max=10) # Amplitude scale factor
        par.add('R0',value=aux_data[dsetname]['R'],vary=False,min=40,max=70) # Bilayer radius
        par.add('Rsig',value=aux_data[dsetname]['dR'],vary=False,min=.25*40,max=.25*70) #Bilayer variance in radius
        # this is the polydispersity factor from DLS times R
        par.add('W_asym',value=0,vary=False,min=-1,max=1) # Asymmetry factor for 
        # bilayer, inner leaflet is larger than outer leaflet by approximately the  ratio
        # 4W/pi
        par.add('A_T_asym',value=0,vary=False,min=-1,max=1) # Asymmetry factor for amplitude of
        #tail group.  The inner tailgroup is higher electron density than outer by a ration of approximately
        # A_T_asym/4
        #%%
        # extract intensity from data structure for fitting
        q = fitset['q']
        I = fitset['I']
        bgfun1 = water['I']
        bgfun2 = air['I']
        w = np.zeros(len(fitset['dI']))            
        # Truncate the data range so as not to fit below qmin and above qmax
        qmin = 0.5
        qmax = 5.5
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
            psize = 4
            print('running differential evolution fit ')
            result = liposome_model.fit(I,par,q=q,bgfun1 = bgfun1,
                        bgfun2=bgfun2,weights=w,
                        method='differential_evolution',max_nfev=200000,
                        fit_kws={'popsize':psize}
                        )
            # append cholesterol fraction to result values
            result.values['cfrac'] = aux_data[dsetname]['cfrac']
            print('Finished fit #2 redchi = {0:7.2f}'.format(result.redchi))
            multi_results_out[dsetname] = {'result':result,
                'cfrac':aux_data[dsetname]['cfrac'],'dsetname':dsetname}
        make_plots(pp,result,dsetname)
        print('Fit to dataset {0:s} complete'.format(dsetname))
timestring = str(int(time.time()))[-6:]
multi_name = 'liposome_multi_fit_results'+timestring+'.npy'
with open(resultname,'wb') as fd:
    np.save(fd,multi_results_out,allow_pickle=True)
#%%
# reload mult-results file and restore
with open(resultname,'rb') as fd:
    multi_results_in = np.load(fd,allow_pickle=True)[()]
#%%
# collect summary results
sigs = np.array([])
dsigs = np.array([])
Ws = np.array([])
dWs = np.array([])
chi2 = np.array([])
Cf = np.array([])
W_asym = np.array([])
dW_asym = np.array([])
A_T_asym = np.array([])
dA_T_asym = np.array([])
for tout in multi_results_in:
    tres = multi_results_in[tout]['result']
    params = tres.params  
    chi2 = np.append(chi2,result.redchi)
    sigs = np.append(sigs,params['sig'].value)
    Ws = np.append(Ws,params['W'].value)
    dsigs = np.append(dsigs,params['sig'].stderr)
    dWs = np.append(dWs,params['W'].stderr)
    Cf  = np.append(Cf,multi_results_in[tout]['cfrac'])
    W_asym = np.append(W_asym,params['W_asym'].value)
    dW_asym = np.append(dW_asym,params['W_asym'].stderr)
    A_T_asym = np.append(A_T_asym,params['A_T_asym'].value)
    dA_T_asym = np.append(dA_T_asym,params['A_T_asym'].stderr)
plt.figure('Ws')
plt.clf()
chi2lim = 5
gg = chi2<chi2lim
plt.errorbar(Cf[gg],Ws[gg],dWs[gg],fmt='ks')
plt.xlabel('cholesterol fraction')
plt.ylabel('bilayer thickness')
plt.savefig(pp, format='pdf',bbox_inches='tight')
#plt.savefig('Results\\'+timestring+'thickness.png')
plt.figure('sigs')
plt.clf()
plt.errorbar(Cf[gg],sigs[gg],dsigs[gg],fmt='ks')
plt.xlabel('cholesterol fraction')
plt.ylabel('bilayer roughness')
plt.savefig(pp, format='pdf',bbox_inches='tight')
#plt.savefig('Results\\'+timestring+'roughness.png')

plt.figure('chi2')
plt.clf()
plt.plot(Cf[gg],chi2[gg],'ks')
plt.xlabel('cholesterol fraction')
plt.ylabel('fit chi-squared (reduced) ')
plt.savefig(pp, format='pdf',bbox_inches='tight')
#plt.savefig('Results\\'+timestring+'redchi.png')

plt.figure('asymmetry')
plt.clf()
plt.errorbar(Cf[gg],W_asym[gg],dW_asym[gg],fmt='ks',label='width asymmetry')
plt.errorbar(Cf[gg],A_T_asym[gg],dA_T_asym[gg],fmt='ro',label='Amplitude asymmetry')
plt.xlabel('cholesterol fraction')
plt.ylabel('fit asymmetry parameter ')
plt.legend()
plt.savefig(pp, format='pdf',bbox_inches='tight')
#plt.savefig('Results\\'+timestring+'asymmetry.png')
pp.close()
