# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 12:18:45 2021
Last updated May 13, 2021
Code to calculate the diffuse scattering from water
and use this to find normalization constant for scattering

@author: Laurence Lurio
"""

import pint
from mendeleev import O, H
import numpy as np
import matplotlib.pyplot as plt
import xraydb
from liposome_sum_code import sum_files
ur = pint.UnitRegistry()
ur.setup_matplotlib(True)
#%% Calculate the thermal diffuse scattering from water
rhoW = 1.0035*ur('gram/cc')
A = O.atomic_weight + 2*H.atomic_weight
Z = O.atomic_number + 2*H.atomic_number
rhoe = rhoW*Z/(A*ur('amu'))
print('water electron density = {0:5.1f~P}'.format(rhoe.to('nm^-3')))
T = (273.15+23)*ur('kelvin')
v = 1493.5932864269*ur('m/s')
rho = 997.54049129117*ur('kg/m^3')
#Find the compressibility from sound velocity and density
chi = 1/((v**2*rho).to('GPa'))
kappa_symbol = '\u03ba'
chi_symbol = '\u03c7'
print('water compressibility {0} = {1:4.2f~P}'.format(chi_symbol,chi))
# Calculate scattering cross section
sig = rhoe**2*ur('k')*T*ur('r_e')**2*chi
print('water cross section = {0:7.3e~P}'.format(sig.to('cm^-1')))
#%% calculate water S(Q) for comparison with theory
S = (rhoe/Z)*chi*ur('k')*T
print('water structure factor {0:7.3e~P}'.format(S.to('dimensionless')))
#%%
#Now sum  exerimental data for water and empty capillary
ddir = 'C:\\Users\\lluri\\Dropbox\\Runs\\2021\\March 2021 Liposomes\\Data\\SAXS\\Averaged\\'
nfiles = 40
tfn = 'Swater_c_00035'
water =  sum_files(ddir,tfn,nfiles)
tfn = 'Saircap_c_00022'
air =  sum_files(ddir,tfn,nfiles)
#%%
# find the average water scattering between q = 1 and q = 4 to
# get the mean value of the thermal diffuse scattering
range = (water['q']>1)*(water['q']<4)
wave = np.mean((water['I'][range]-air['I'][range]))
print('average of water diffuse {0:7.3f}'.format(wave))
#%%
# find the normalization constant to convert data to absolute
# scattering units based on the water cross section
norm = sig.to('cm^-1').magnitude/wave
print('normalization factor {0:7.3e}'.format(norm))
#%%
# we can also estimate the theoretical normalization
Cj = 200*ur('microA/V')/(100*ur('kHz/V'))/(1*ur('e'))
Ephot = 12*ur('keV')
Epair = 3.65*ur('eV')
Npairs = Ephot/Epair
Cj /= Npairs
print('Calculated Cj = {0:7.3e~P}'.format(Cj.to('dimensionless')))
dOmega = (0.146/2269)**2
Lambda = 0.15*ur('cm') # sample thickness
#%% Estimate the normalization constant from the data
I = (water['I']-air['I'])
q = water['q']
dI = np.sqrt(water['dI']**2+air['dI']**2)
rr = (q>.1)*(q<6)
I = I[rr]
dI = dI[rr]
q = q[rr]
plt.figure('photon flux')
plt.clf()
plt.plot(q,I**2/dI**2)
plt.yscale('log')
plt.figure('raw counts')
plt.clf()
plt.plot(q,I)
plt.yscale('log')
# plot gives 17,000 photons at q=3.  Now, this number is total photons
# in that bin, so photons per pixel is around 10^3 less
Iraw = 17000.0
npring = 1000
ppix = Iraw/npring
I0 = 4e11
# raw counts per pixel around q = 3 inv. nm. is 0.09 
N = .09
#so N = Cj*ppix/I0, Cj = N*I0/ppix
Cj = (N*I0/ppix)*ur('dimensionless')
print('Cj  from I^2/dI^2 {0:7.3e~P}'.format(Cj))
#%%
# put in alternate value of Cj from comparint I to dI
k = 1/(Lambda*dOmega*Cj)
print('photons/ct = {0:7.3e}'.format(Cj.to_base_units()))
Jt = 288644.200
It = (Jt*Cj).to('dimensionless')
print('Estimated transmitted flux {0:7.3e~P}'.format(It))
print('theoretical conversion factor {0:7.3e~P}'.format(k.to('cm^-1')))
#%%
# Plot the normalized scattering from water
plt.figure('normalized')
plt.clf()
plt.plot(water['q'],(water['I']-air['I'])*norm,'-k',label='experimental data')
plt.xlabel('q (nm$^{-1}$)')
plt.ylabel(r'$\frac{1}{V}\frac{d\Sigma}{d\Omega}$ water - air (cm$^{-1}$)')
plt.xlim(.1,8)
plt.ylim(.0,.025)
plt.yscale('linear')
plt.xscale('linear')
#%%
# Now calculate the contribution from compton scattering
sig_comp = ((rhoe*ur('r_e')**2).to('cm^-1')).magnitude
# convert from q to "s" which is the parameter used by the 
# atomic scattering factor code
s = water['q']/10/4/np.pi
# calculate the fraction of the electrons which scatter inelastically
# this is the f0 factor at s=0 - the f0 factor at s divided by
# the f0 factor at s=0
#
# we assume the Oxygen is electronegative, so we take it as -2 
# ionization state and ignore the electrons for hydrogen
fe = (xraydb.f0('O2-',s*0)-xraydb.f0('O2-',s))/xraydb.f0('O2-',s*0)
sig_comp = sig_comp*fe
plt.plot(water['q'],sig_comp,'--r',label="Just compton")
plt.plot(water['q'],sig_comp+sig.to('cm^-1').magnitude,'--g',label="thermal diffuse + compton")
plt.plot(water['q'],sig_comp*0+sig.to('cm^-1').magnitude,'--m',label='thermal diffuse')
#Now calculate expected normalization
plt.legend()
plt.savefig('water_norm.jpg')

