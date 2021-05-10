# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:09:27 2021

@author: Laurence Lurio

T

"""

from matplotlib import pyplot as plt
from liposome_saxs_funs import slab, profile, liposome_model
from liposome_saxs_funs import sum_files, subtract_sets,half, get_SF
from liposome_saxs_funs import norm_model, trunc_set, get_SF_tail
import numpy as np
from lmfit import Parameters
#%%
from os import listdir

# read in data
Datadir = 'C:\\Users\\lluri\\Dropbox\\Runs\\2021\\March 2021 Liposomes\\Data\\SAXS\\Averaged\\'

def find_file(Datadir,searchstring):
    names = listdir(Datadir)
    tname = [name for name in names if searchstring in name][0]
    return tname[0:-10]

#%%
nfiles = 40 # 40 data files per set
name = 'aircap_c'
air = sum_files(Datadir,find_file(Datadir,name),nfiles)
air = trunc_set(air,22)
name = 'Swater_c'
water = sum_files(Datadir,find_file(Datadir,name),nfiles)
water = trunc_set(water,22)
SF1 = get_SF(water,air)
water = subtract_sets(water,air,SF=SF1)
plt.figure('water')
plt.clf()
plt.plot(water['q'],water['I'],'ks')
plt.yscale('log')
plt.xscale('linear')
plt.xlabel('q [$nm^{-1}$]')
plt.ylabel('Scattering Intensity')
plt.show
wf = water['I'][(water['q']>2)*(water['q']<3)]
wnorm = np.mean(wf)
print('wnorm = {0:4.2e}'.format(wnorm))
name = 'egg1_chol1'
name = 'egg4_chol1_50_a'
eggPC = sum_files(datadir,find_file(Datadir,name),nfiles)
eggPC = trunc_set(eggPC,22)
SF2 = get_SF(eggPC,air)
eggPC = subtract_sets(eggPC,air,SF=SF2)
SF3 = get_SF_tail(eggPC,water)
eggPCs = subtract_sets(eggPC,water,SF=SF3)
eggPCs = half(half(half(eggPCs)))
eggPCs['I'] /= wnorm
eggPCs['dI'] /= wnorm

#%%
plt.figure('qspace')
plt.plot(eggPCs['q'],eggPCs['I'],'ks')
plt.xlim([.05,6])
plt.ylim([1e-6,2])
plt.yscale('log')
plt.xscale('linear')
plt.xlabel('q [$nm^{-1}$]')
plt.ylabel('Scattering Intensity')
plt.show
#%%

''' Initialize the parameters for the fit.
'''
par = Parameters()
par.add('W',value=4.37,vary=True,min=3,max=6)
par.add('d_H',value=.609,vary=True,min=.5,max=.75)
par.add('d_M',value=.1,vary=True,min=0.05,max=.15)
par.add('A_H',value=70,vary=False,min=50,max=90)  # since scale factor is arbitrary, fix one value
par.add('A_T',value=-84.7,vary=True,min=-100,max=-50)
par.add('A_M',value=-333,vary=False) # fix methyl amplitude at -water value (e.g. assume 0)
par.add('sig',value=.3,vary=False)  
par.add('I',value=10,vary=True)
par.add('R0',value=50,vary=False)
par.add('Rsig',value=10,vary=False)



# extract intensity from data structure for fitting
q = eggPCs['q']
I = eggPCs['I']
w = 1/(abs(I)+.0001)
w[q>3] = 0
# run the actual fit
result = liposome_model.fit(I,par,q=q,weights=w)
#%%
# plot the fit results
plt.figure(3)
qq = np.linspace(np.log(.03),np.log(10),1000)
qq = np.exp(qq)
yfit = liposome_model.eval(result.params,q=qq)
plt.plot(q,I,'ks')
plt.plot(qq,yfit,'-g')
plt.yscale('log')
plt.xscale('log')
plt.xlim(.05,10)
plt.ylim(1e-5,5)
plt.xlabel('q (invers  nm)')
plt.ylabel('scattering intensity')
plt.legend(['data','fit'])
plt.title('eggPC')
plt.show()
#plot the profile in real space corresponding to the fit
plt.figure(2)
P3 = profile([])
P3.load_par(result.params)
P3.draw_rho(offset = 333.3, ymin = 0, ymax = 450, color='cyan')
#print out the fit results
print(result.fit_report())
