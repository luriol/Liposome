# -*- coding: utf-8 -*-
"""
Created on Tue May 25 09:48:33 2021

@author: lluri
"""
from matplotlib import pyplot as plt
from liposome_saxs_funs import  profile
from liposome_model import liposome_model
from liposome_sum_code import shorten
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
def fit_liposome(data,par,fit_pars):
    #%% (D) loop through each data set and fit
    multi_results_out = {}
    ngood = 0
    water = shorten(data[fit_pars['backgrounds'][0]],
                    fit_pars['nshort'])
    air = shorten(data[fit_pars['backgrounds'][1]],
                  fit_pars['nshort'])
    for dind, dsetname in enumerate(data.keys()):
        selection = (dsetname in fit_pars['selected_data'])
        if (dsetname not in fit_pars['backgrounds'] and 
            selection):
            ngood += 1
            fitset = shorten(data[dsetname],fit_pars['nshort'])
            par['R0'].value = fit_pars['aux_data'][dsetname]['R']
            par['Rsig'].value = fit_pars['aux_data'][dsetname]['dR']
            q = fitset['q']
            I = fitset['I']
            w = np.zeros(len(fitset['dI']))            
            rr = (q>fit_pars['qmin'])*(q<fit_pars['qmax'])
            sys_err = 0 # estimate of systematica normalization error
            w[rr] = 1/np.sqrt(fitset['dI'][rr]**2 + water['dI'][rr]**2 +
                              (sys_err*fitset['I'][rr])**2 +(sys_err*water['I'][rr])**2)
            w[1^rr] = 0 # "eps" is small but nonzero
            # choose fit method, least squares = 1, differential evolution =0
            if (fit_pars['fit_method']):
                print('running least squares fit')
                result = liposome_model.fit(I,par,q=q,bgfun1 = water['I'],
                            bgfun2=air['I'],weights=w)
            else:
                print('running differential evolution fit ')
                result = liposome_model.fit(I,par,q=q,bgfun1 = water['I'],
                            bgfun2=air['I'],weights=w,
                            method='differential_evolution',max_nfev=fit_pars['nfev'],
                            fit_kws={'popsize':fit_pars['psize']}
                            )
                # append additional variables to result values
                print('Finished fit #2 redchi = {0:7.2f}'.format(result.redchi))
                multi_results_out[dsetname] = {'result':result,
                    'cfrac':fit_pars['aux_data'][dsetname]['cfrac'],'dsetname':dsetname,
                    'fit_pars':fit_pars}
            print('Fit to dataset {0:s} complete'.format(dsetname))
    with open(fit_pars['resultname'],'wb') as fd:
        np.save(fd,multi_results_out,allow_pickle=True)
    return multi_results_out
def make_plots(pp,result,fit_pars,dsetname):
    #
    # First figure, printout of fit report
    #
    if not os.path.exists('Results'):
        os.mkdir('Results')
    fit_results = result.fit_report()
    fig = plt.figure('results_{0}'.format(dsetname))
    fig.clf()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.text(.02,.02,fit_results,fontsize='x-small')
    ax.text(.75,.9,dsetname,fontsize='large')
    ax.axis('off')
    fig.show()
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    #
    # Second figure plot data and background
    #
    vals = result.values
    q = result.userkws['q']
    bgfun1 = result.userkws['bgfun1']
    bgfun2 = result.userkws['bgfun2']
    bg = (vals['bg'] + q*vals['lbg'] +
        q**2*vals['qbg'] + 
        bgfun1*vals['bg1sf'] +
        bgfun2*vals['bg2sf'])
    data = result.data
    fit = result.best_fit
    plt.figure('Fit_with_bg_{0}'.format(dsetname))
    plt.clf()
    w = result.weights>0
    plt.errorbar(q[w],data[w],1/result.weights[w],fmt='-k',label='data')
    plt.plot(q[w],fit[w],'-r',label='fit')
    #plt.plot(q[w], bgfun1[w]*vals['bg1sf'],'-b',label='water')
    #plt.plot(q[w], bgfun2[w]*vals['bg2sf'],'-g',label='glass')
    bg = (vals['bg'] + q*vals['lbg'] +
     q**2*vals['qbg'] + 
     bgfun1*vals['bg1sf'] +
     bgfun2*vals['bg2sf'])
    plt.plot(q[w], bg[w],'--k',label='total bg')
    plt.yscale('linear')
    plt.xscale('linear')
    #plt.ylim(.0325,.07)
    plt.title(dsetname) 
    plt.xlabel('q (nm$^{-1}$)')
    plt.ylabel(r'$\frac{1}{V}\frac{d\Sigma}{d\Omega}$ (cm $^{-1}$)')
    plt.legend()
    plt.yscale('log')
    #plt.show()
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    #
    # Third figure background subtracted data
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
    qfull = np.linspace(0.05,7,10000)
    # create data set of fit over finer q-range without background.
    fit = liposome_model.eval(result.params,q=qfull,bgfun1 = qfull*0,bgfun2=qfull*0)
    #fit = result.best_fit - bg
    plt.figure('Fit_minus_bg_{0}'.format(dsetname))
    plt.clf()
    #plt.errorbar(q[w],bgdata[w],1/result.weights[w],fmt='-k',label='data')
    plt.plot(q[w],bgdata[w],'ko',label='data',markersize=2)
    #plt.plot(q[w],fit[w],'-r',label='fit')
    plt.plot(qfull,fit,'-r',label='fit')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(1e-6,100)
    plt.title(dsetname) 
    plt.xlabel('q (nm$^{-1}$)')
    plt.ylabel(r'$\frac{1}{V}\frac{d\Sigma}{d\Omega}$ (cm $^{-1}$)')
    plt.legend()
    #plt.show()
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    # #%%
    # Fourth figure, plot residuals
    plt.figure('Fractonal Residuals_{0}'.format(dsetname))
    plt.clf()
    plt.errorbar(q[w],(result.data[w]-result.best_fit[w])/result.data[w],
        1/result.weights[w]/result.data[w],fmt='ks',label='data')
    plt.yscale('linear')
    plt.ylim(-.025,.025)
    plt.title(dsetname)
    plt.xlabel('q (nm$^{-1}$)')
    plt.ylabel('residuals')
    plt.legend()
    #plt.show()
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    # #%%
    # Fifth figure plot the profile in real space corresponding to the fit
    plt.figure('Real_Space_Fit_{0}'.format(dsetname))
    plt.clf()
    P3 = profile([])
    P3.load_par(result.params)
    P3.draw_rho(offset = 333.3, ymin = 0, ymax = 450, color='blue')
    plt.title(dsetname)
    #plt.show()
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    #%%
def key_plot(Cf,y,dy,Rs,chi2,rlim,chi2lim):
    gg1 = (chi2<chi2lim)*(Rs<rlim)
    gg2 = (chi2>chi2lim)*(Rs<rlim)
    gg3 = (chi2<chi2lim)*(Rs>rlim)
    gg4 = (chi2>chi2lim)*(Rs>rlim)
    if (sum(gg1)>0):
        plt.errorbar(Cf[gg1],y[gg1],dy[gg1],fmt='ks',
                label='R < {0:d} chi2 < {1:d}'.format(rlim,chi2lim))
    if (sum(gg2)>0):
        plt.errorbar(Cf[gg2],y[gg2],dy[gg2],fmt='ko',
                label='R < {0:d} chi2 > {1:d}'.format(rlim,chi2lim))    
    if (sum(gg3)>0):
        plt.errorbar(Cf[gg3],y[gg3],dy[gg3],fmt='rs',
                label='R > {0:d} chi2 < {1:d}'.format(rlim,chi2lim))  
    if (sum(gg4)>0):
        plt.errorbar(Cf[gg4],y[gg4],dy[gg4],fmt='ro',
                label='R > {0:d} chi2 > {1:d}'.format(rlim,chi2lim))
def plot_final_results(fit_pars):
    # reload mult-results file and restore
    with open(fit_pars['resultname'],'rb') as fd:
        multi_results_in = np.load(fd,allow_pickle=True)[()]
    pp = PdfPages(fit_pars['figname'])
    #%%
    # collect summary results
    Rs = np.array([])
    sigs = np.array([])
    dsigs = np.array([])
    Ws = np.array([])
    dWs = np.array([])
    chi2 = np.array([])
    Cf = np.array([])
    Ts = np.array([])
    W_asym = np.array([])
    dW_asym = np.array([])
    A_T_asym = np.array([])
    dA_T_asym = np.array([])

    col = ['black','red','green','blue','cyan','magenta']
    lsty = ['solid','dashed','dashdot','dotted']
    ncol = len(col)
    for nout, tout in enumerate(multi_results_in):
        tres = multi_results_in[tout]['result']
        tsetname = multi_results_in[tout]['dsetname']
        make_plots(pp,tres,fit_pars,tsetname)
    plt.figure('real_space_plots_overview')
    plt.clf()
    for nout, tout in enumerate(multi_results_in):
        tres = multi_results_in[tout]['result']
        tsetname = multi_results_in[tout]['dsetname']  
        params = tres.params  
        chi2 = np.append(chi2,tres.redchi)
        sigs = np.append(sigs,params['sig'].value)
        Rs = np.append(Rs,params['R0'].value)
        Ws = np.append(Ws,params['W'].value)
        dsigs = np.append(dsigs,params['sig'].stderr)
        dWs = np.append(dWs,params['W'].stderr)
        W_asym = np.append(W_asym,params['W_asym'].value)
        dW_asym = np.append(dW_asym,params['W_asym'].stderr)
        A_T_asym = np.append(A_T_asym,params['A_T_asym'].value)
        dA_T_asym = np.append(dA_T_asym,params['A_T_asym'].stderr)
        if fit_pars['sample variable'] == 'Temperature':
            T = float(tsetname[10:13])/10.0
            Ts = np.append(Ts,T)
            print(T)
            print(tsetname)
        elif fit_pars['sample variable'] == 'Cholesterol':
            cfrac = multi_results_in[tout]['cfrac']
            Cf  = np.append(Cf,cfrac)
        P3 = profile([])
        P3.load_par(params)
        if fit_pars['sample variable'] == 'Cholesterol':
            off = cfrac*200
        elif fit_pars['sample variable'] == 'Temperature':
            off = (T-36)*200*.25/4.0
        P3.draw_rho(offset = 333.3+off, ymin = 300, 
                    ymax = 525, color=col[nout%ncol],
                    linestyle = lsty[int(nout/ncol)],label=tsetname)
    plt.xlabel('distance [nm]')
    plt.ylabel('electron density $e^-/$nm$^3$')
    plt.legend(loc=(1.04,0))
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    if fit_pars['sample variable'] == 'Cholesterol':
        xvals = Cf
        xlabel = 'chlesterol fraction'
    elif fit_pars['sample variable'] == 'Temperature':
        xvals = Ts
        xlabel = 'Temperature (C)'
        # throw away radius values, as they are all the same
        Rs = Rs*0+50
    plt.figure('Ws')
    plt.clf()
    chi2lim = 2
    rlim = 100
    key_plot(xvals,Ws,dWs,Rs,chi2,rlim,chi2lim)
    plt.xlabel(xlabel)
    plt.ylabel('bilayer thickness')
    plt.legend()
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    
    #plt.savefig('Results\\'+timestring+'thickness.png')
    plt.figure('sigs')
    plt.clf()
    key_plot(xvals,sigs,dsigs,Rs,chi2,rlim,chi2lim)
    plt.xlabel(xlabel)
    plt.ylabel('bilayer roughness')
    plt.legend()
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    #plt.savefig('Results\\'+timestring+'roughness.png')
    
    plt.figure('chi2')
    plt.clf()
    gg1 = Rs<rlim
    gg2 = Rs>rlim
    if sum(gg1)>1:
        plt.plot(xvals[gg1],chi2[gg1],'ks',label='R0 < {0:4.1f}'.format(rlim))
    if sum(gg2)>1:
        plt.plot(xvals[gg2],chi2[gg2],'rs',label='R0 > {0:4.1f}'.format(rlim))
    plt.xlabel(xlabel)
    plt.ylabel('fit chi-squared (reduced) ')
    plt.legend()
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    #plt.savefig('Results\\'+timestring+'redchi.png')
    
    plt.figure('asymmetry')
    plt.clf()
    key_plot(xvals,W_asym,dW_asym,Rs,chi2,rlim,chi2lim)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('fit asymmetry parameter ')
    plt.legend(loc=(1.04,0))
    plt.savefig(pp, format='pdf',bbox_inches='tight')
    pp.close()

    
