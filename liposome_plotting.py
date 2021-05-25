# -*- coding: utf-8 -*-
"""
Created on Tue May 25 09:48:33 2021

@author: lluri
"""
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from liposome_saxs_funs import  profile, liposome_model
import numpy as np

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

def plot_final_results(pp,resultname):
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
    plt.figure('real_space_plots')
    plt.clf()
    col = ['black','red','green','blue','cyan','magenta']
    lsty = ['solid','dashed','dashdot','dotted']
    ncol = len(col)
    for nout, tout in enumerate(multi_results_in):
        tres = multi_results_in[tout]['result']
        tsetname = multi_results_in[tout]['dsetname']
        cfrac = multi_results_in[tout]['cfrac']
        params = tres.params  
        chi2 = np.append(chi2,tres.redchi)
        sigs = np.append(sigs,params['sig'].value)
        Ws = np.append(Ws,params['W'].value)
        dsigs = np.append(dsigs,params['sig'].stderr)
        dWs = np.append(dWs,params['W'].stderr)
        Cf  = np.append(Cf,cfrac)
        W_asym = np.append(W_asym,params['W_asym'].value)
        dW_asym = np.append(dW_asym,params['W_asym'].stderr)
        A_T_asym = np.append(A_T_asym,params['A_T_asym'].value)
        dA_T_asym = np.append(dA_T_asym,params['A_T_asym'].stderr)
        P3 = profile([])
        P3.load_par(params)
        off = cfrac*200
        P3.draw_rho(offset = 333.3+off, ymin = 300, 
                    ymax = 525, color=col[nout%ncol],
                    linestyle = lsty[int(nout/ncol)],label=tsetname)
    plt.xlabel('distance [nm]')
    plt.ylabel('electron density $e^-/nm^3$')
    plt.legend()
    plt.savefig(pp, format='pdf',bbox_inches='tight')
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

    
