# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 12:18:45 2021

@author: lluri
"""

import pint
from mendeleev import O, H, C, P
import numpy as np
import matplotlib.pyplot as plt
import xraydb
ur = pint.UnitRegistry()
ur.setup_matplotlib(True)
#%% Calculate the thermal diffuse scattering from water
rhoW = 1.0035*ur('gram/cc')
A = O.atomic_weight + 2*H.atomic_weight
Z = O.atomic_number + 2*H.atomic_number
rhoe = rhoW*Z/(A*ur('amu'))
print('water electron density = {0:5.2e~P}'.format(rhoe.to('m^-3')))
T = (273.15+23)*ur('kelvin')
v = 1493.5932864269*ur('m/s')
rho = 997.54049129117*ur('kg/m^3')
#Find the compressibility from sound velocity and density
kappa = v**2*rho
# Calculate scattering cross section
sig = rhoe**2*ur('k')*T*ur('r_e')**2/kappa
print('scattering cross section = {0:7.3e~P}'.format(sig.to('cm^-1')))
#%%
#Now open up exerimental data
savename = 'liposome_data.npy'
with open(savename,'rb') as fdir2:
    mydata = np.load(fdir2,allow_pickle=True)[()]
water = mydata['water_c']
air = mydata['aircap_c']
q = water['q']
I = water['I']-air['I']
dI = (water['dI']**2+air['dI']**2)**.5
# find the average water scattering between q = 1 and q = 4 to
# get the mean value of the thermal diffuse scattering
wave = np.mean(I[(q>1)*(q<4)])
print('average of water diffuse {0:7.3e}'.format(wave))
# find the normalization constant to convert data to absolute
# scattering units based on the water cross section
norm = sig.to('cm^-1').magnitude/wave
print('normalization factor {0:7.3e}'.format(norm))
plt.figure('normalized')
plt.clf()
q = water['q']
I = (water['I']-air['I'])*norm
dI = (water['dI']**2+air['dI']**2)**.5*norm
plt.plot(q,I,'-k',label='experimental data')
plt.xlabel('q (inv. nm)')
plt.ylabel(r'Scattering Intensity water - air (cm$^{-1}$)')
plt.xlim(.1,8)
plt.ylim(.0,.025)
plt.yscale('linear')
plt.xscale('linear')
#%%
# Now calculate the contribution from compton scattering
sig_comp = rhoe*ur('r_e')**2
sig_comp = sig_comp.to('cm^-1').magnitude
s = q/10/4/np.pi
Ne = (xraydb.f0('O2-',s*0)-xraydb.f0('O2-',s))/xraydb.f0('O2-',s*0)
sig_comp_O = sig_comp*Ne
plt.plot(q,sig_comp_O,'--r',label="Just compton")
plt.plot(q,sig_comp_O+sig.to('cm^-1').magnitude,'--g',label="thermal diffuse + compton")
plt.plot(q,sig_comp_O*0+sig.to('cm^-1').magnitude,'--m',label='thermal diffuse')
#Now calculate expected normalization
plt.legend()
plt.savefig('water_norm.jpg')
# find effective number of electrons contributing to compton scattering
# for hydrogen and carbon.  Assume CH2
NeC = xraydb.f0('C',s*0)-xraydb.f0('C',s)
NeC += 2*(xraydb.f0('H',s*0)-xraydb.f0('H',s))
NeC /= 2*xraydb.f0('H',s*0)+xraydb.f0('C',s*0)
plt.figure('compton')
plt.clf()
plt.plot(q,NeC/Ne)
plt.ylabel('ratio of hydro carbon to water compton cross section')
plt.xlabel('q (inv. nm)')
plt.savefig('compton')
