# -*- coding: utf-8 -*-
"""
File to calculate normalization values for liposome data

@author: lluri
"""

import pint
from mendeleev import O, H, C, P
import numpy as np
import matplotlib.pyplot as plt
import xraydb
import pint
ur = pint.UnitRegistry()

MDPPC = 734.04 # molecular weight of DPPC
eggS = [(14,0.2,0),(16,32.7,0),(16,1.1,1),(18,12.3,0),
        (18,32.0,1),(18,17.1,2),(20,0.2,2),(20,0.3,3),(20,2.7,4),
        (22,0.6,6)] #saturated lipid percents
M = 0
for tup in eggS:
    tm = MDPPC + (tup[0]-16)*14 - tup[2]*2
    M += tup[1]*tm/100
print('Mass of Egg PS {0:7.3f}'.format(M))  

W_l = M*ur('amu')
S_l =  1/(0.48*ur('nm^2'))
fact = (8*W_l*np.pi*S_l).to('kg/m^2')
print('{0:5.2e~P}'.format(fact))
rho_m = 10*ur('mg/ml')
R = 62*ur('nm')
N = rho_m*ur('r_e^2')/fact/R**2
print('{0:7.3e~P}'.format(N.to('cm^-1')))
