# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:09:27 2021

@author: lluri
"""
import numpy as np
from numpy import sin, cos
from scipy.special import erf
from matplotlib import pyplot as plt
from lmfit.models import Model

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

liposome_model = Model(saxsfit,independent_vars=['q','bgfun1','bgfun2'])

class slab:
    def __init__(self,amplitude,center,sigma,name):
        self.amplitude = amplitude
        self.sigma = sigma
        self.center = center
        self.name = name
    def drho(self,x):
        xp = x-self.center
        arg = (xp/self.sigma)**2/2
        drho = self.amplitude*np.exp(-arg)/np.sqrt(2*np.pi)/self.sigma
        return drho
    def f_of_q(self,q,R):
        y = q*(R+self.center)
        eps = q*self.sigma
        A = self.amplitude
        f =  4*np.pi*A/q**3
        f *= np.sin(y)*(1-eps**2)-y*np.cos(y)
        f *= np.exp(-eps**2/2)
        return f
    def rho(self,x):
        xp = x-self.center
        arg = (xp/self.sigma)/np.sqrt(2)
        rho = (erf(arg)/2+1/2)*self.amplitude
        return rho     
    def draw_drho(self):
        x1 = self.center-4*self.sigma
        x2 = x1+8*self.sigma
        xr = np.linspace(x1,x2,1000)
        yr = self.drho(xr)
        plt.plot(xr,yr)
        plt.xlabel('z [nm]')
        plt.ylabel('$d\\rho/dz$ (e$^-$/nm$^4$)')
        plt.title('density profile derivative {0:s}'.format(self.name))
    def draw_rho(self):
        x1 = self.center-4*self.sigma
        x2 = x1+8*self.sigma
        xr = np.linspace(x1,x2,1000)
        yr = self.rho(xr)
        plt.plot(xr,yr)
        plt.xlabel('z [nm]')
        plt.ylabel('$\\rho$ (e$^-$/nm$^3$)')
        plt.title('density profile  {0:s}'.format(self.name))
        
class profile:
    def __init__(self,slabs):
        self.slabs = slabs
        self.order_slabs()
    def add_slab(self,slab):
        self.slabs.append(slab)
    def rho(self,x):
        y = x*0;
        for slab in self.slabs:
            y+= slab.rho(x)
        return y
    def list_slabs(self):
        for i, slab in enumerate(self.slabs):
            print('{0:d}: {1:<15s} {2:7.2f} {3:7.2f} {4:7.2f}'.format(i,
                slab.name,slab.amplitude,
                slab.sigma,slab.center))
    def order_slabs(self):
         self.slabs = sorted(self.slabs, key=lambda Slab: Slab.center)
    def make_x(self):
        s1 = self.slabs[0]
        s2 = self.slabs[-1]
        x1 = s1.center-4*s1.sigma
        x2 = s2.center+4*s2.sigma
        x = np.linspace(x1,x2,10000)
        return x
    def make_F(self,q,R):
        f = q*0
        for slab in self.slabs:
            f += slab.f_of_q(q,R)
        F = np.abs(f)**2
        return F
    def draw_rho(self,offset=0,ymin =-9999,ymax=-9999,
                 color='black',label='',linestyle='solid'):
        x = self.make_x()
        y = self.rho(x)+offset
        if ymin == -9999:
            ymin = min(y)
        if ymax == -9999:
            ymax = max(y)
        plt.plot(x ,y, color = color, label = label,linestyle=linestyle)
        plt.xlabel('z [nm]')
        plt.ylabel('$\\rho$ (e$^-$/nm$^3$)')
        plt.title('density profile  ')
        plt.ylim(ymin,ymax)
    def make_F_res(self,q,R0,sig):
        phi = q*sig
        y = 0
        for slabi in self.slabs:
            eps_i = q*slabi.sigma
            A_i = slabi.amplitude*np.exp(-eps_i**2/2)
            w_i = q*(R0+slabi.center)
            d_i = 1-eps_i**2
            EP = np.exp(-2*phi**2)
            for slabj in self.slabs:            
                eps_j = q*slabj.sigma                
                A_j = slabj.amplitude*np.exp(-eps_j**2/2)                
                w_j = q*(R0+slabj.center)                  
                d_j = 1-eps_j**2
                Cx = cos(w_i)*cos(w_j)
                Sx = sin(w_i)*sin(w_j)
                Cp = cos(w_i+w_j)
                Sp = sin(w_i+w_j)
                C1 = Cx*d_i*d_j 
                C1 += cos(w_i)*sin(w_j)*2*w_j*d_i
                C1 += Sx*w_i*w_j
                C2 = (Cp*(w_i*w_j-d_i*d_j)-
                    2*Sp*w_j*d_i)
                C3 = Cp
                C4 = -Cp*d_i-Sp*w_i
                C5 = Sx             
                I1 = 1
                I2 = .5*(1+EP)
                I3 = (phi**2/2)*(1+(1-4*phi**2)*EP)
                I4 = 2*phi**2*EP
                I5 = phi**2
                arg = C1*I1+C2*I2+C3*I3+C4*I4+C5*I5
                arg *= 16*np.pi**2*A_i*A_j/q**6
                y += arg
        return y
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


 



