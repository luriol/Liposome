# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:02:09 2021

@author: lluri
"""

import numpy as np
from lmfit.models import Model
from liposome_saxs_funs import slab, profile

# (A) define model
def saxsfit(q,bgfun1,bgfun2,bg=0,bg1sf=1,bg2sf=0,W=4,sig=.3,d_H=0.5,d_M=0.5,I=1,A_H=107,
            A_T=-90,A_M=-334,R0=200,Rsig=50,lbg=0,qbg=0,W_asym=0,A_T_asym=0):
    Wout = W*(1-2*np.arctan(W_asym)/np.pi)
    Win = W*(1+2*np.arctan(W_asym)/np.pi)
    # calculate offset due to asymmetry and use this to keep the profile centered on zero
    dr = (W-Wout)/2
    A_T_out = (A_T+334)*(1-2*np.arctan(A_T_asym)/np.pi) -334
    A_T_in = (A_T+334)*(1+2*np.arctan(A_T_asym)/np.pi)  -334
    s1 = slab(A_H,-Win/2+dr,sig,'water to inner head')
    s2 = slab(A_T_in,-Win/2+d_H + dr,sig,'inner head to inner tail')
    s3 = slab(A_M,-d_M/2 + dr ,sig,'inner tail methyl')
    s4 = slab(-A_M,d_M/2 + dr,sig,'methyl to outer tail')
    s5 = slab(-A_T_out,Wout/2-d_H + dr,sig,'outer tail to outer head')
    s6 = slab(-A_H,Wout/2+ dr,sig,'outer head to water')
    P = profile([s1,s2,s3,s4,s5,s6])
    #P.list_slabs()
    F = P.make_F_res(q,R0,Rsig)
    # assume concentration = 10 mg/ml liposome concentration
    # see documentation
    norm = 3.2e-12*I*62**2/(R0**2+Rsig**2)
    F = (F*norm+bgfun1*bg1sf+bgfun2*bg2sf + np.abs(bg)+np.abs(lbg)*q + 
    +np.abs(qbg)*q**2)
    return F
# (B) routine to load model parameters into slab parameters
def load_par(self,par):
        W = par['W'].value
        sig = par['sig'].value
        d_H = par['d_H'].value
        d_M = par['d_M'].value
        A_H = par['A_H'].value
        A_T = par['A_T'].value
        A_M = par['A_M'].value
        W_asym = par['W_asym'].value
        A_T_asym = par['A_T_asym'].value
        Wout = W*(1-2*np.arctan(W_asym)/np.pi)
        Win = W*(1+2*np.arctan(W_asym)/np.pi)
        dr = (W-Wout)/2
        A_T_out = A_T*(.5-np.arctan(A_T_asym)/np.pi)
        A_T_in = A_T*(.5+np.arctan(A_T_asym)/np.pi)
        self.slabs=[]
        self.add_slab(slab(A_H,-Win/2+dr,sig,'inner head'))
        self.add_slab(slab(A_T_in,-Win/2+d_H+dr,sig,'inner tail'))
        self.add_slab(slab(A_M,-d_M/2+dr,sig,'methyl'))
        self.add_slab(slab(-A_M,d_M/2+dr,sig,'inner head'))
        self.add_slab(slab(-A_T_out,Wout/2-d_H+dr,sig,'inner tail'))
        self.add_slab(slab(-A_H,Wout/2+dr,sig,'inner tail'))
# (C) convert liposome model into lmfit fitting model    
liposome_model = Model(saxsfit,independent_vars=['q','bgfun1','bgfun2'])


 




