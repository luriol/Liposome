# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:09:27 2021

@author: lluri
"""
import numpy as np
from numpy import sin, cos, exp
from scipy.special import erf
from matplotlib import pyplot as plt
from lmfit.models import Model
from lmfit import Parameters 
import pandas as pd

def saxsfit(q,W=4,sig=.3,d_H=0.5,d_M=0.5,I=10,A_H=70,A_T=-90,A_M=-50,R0=200,Rsig=50):
    s1 = slab(A_H,-W/2,sig,'inner head')
    s2 = slab(A_T,-W/2+d_H,sig,'inner tail')
    s3 = slab(A_M,-d_M/2,sig,'methyl')
    s4 = slab(-A_M,d_M/2,sig,'inner head')
    s5 = slab(-A_T,W/2-d_H,sig,'inner tail')
    s6 = slab(-A_H,W/2,sig,'inner tail')
    P = profile([s1,s2,s3,s4,s5,s6])
    F = P.make_F_res(q,R0,Rsig)
    # assume I in mg/ml liposome concentration
    norm = 4.043e-12*I/W/R0
    F = F*norm
    return F

liposome_model = Model(saxsfit)

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
        rho = (erf(arg)/2+1)*self.amplitude
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
            print('{0:d}: {1:s}'.format(i,slab.name))
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
    def draw_rho(self,offset=0,ymin =-9999,ymax=-9999,color='black'):
        x = self.make_x()
        y = self.rho(x)+offset
        if ymin == -9999:
            ymin = min(y)
        if ymax == -9999:
            ymax = max(y)
        plt.plot(x ,y, color = color)
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
        self.slabs=[]
        self.add_slab(slab(A_H,-W/2,sig,'inner head'))
        self.add_slab(slab(A_T,-W/2+d_H,sig,'inner tail'))
        self.add_slab(slab(A_M,-d_M/2,sig,'methyl'))
        self.add_slab(slab(-A_M,d_M/2,sig,'inner head'))
        self.add_slab(slab(-A_T,W/2-d_H,sig,'inner tail'))
        self.add_slab(slab(-A_H,W/2,sig,'inner tail'))
 



def merge_data(x,y):
    """
    routine to merge identical x-data points so that data is single
    valued
    
    returns, merged data and error bars, assuming multiple data points
    at the same value combine with reduced errors
    """
    s = 0*x+1 #define error bars to all initially be equal to 1
    xout = [x[0]]
    yout = [y[0]]
    sout = [s[0]]
    for tx,ty,ts in zip(x[1:],y[1:],s[1:]):
        if xout[-1] == tx:
            yout[-1],sout[-1] = cmbwe(yout[-1],sout[-1],ty,ts)
        else:
            yout.append(ty)
            sout.append(ts)
            xout.append(tx)
    xout = np.array(xout)
    yout = np.array(yout)
    sout = np.array(sout)   
    return xout,yout,sout

def half(dset):
    q = dset['q']
    I = dset['I']
    dI = dset['dI']
    wgi = np.where(~np.isnan(I))[0]
    q = q[wgi]
    I = I[wgi]
    dI = dI[wgi]
    nlen = int(2*np.floor(len(q)/2))
    q = q[0:nlen]
    q.shape
    I = I[0:nlen]
    dI = dI[0:nlen]
    eve = np.arange(0,nlen,2).astype(int)
    odd = np.arange(1,nlen,2).astype(int)
    qout = (q[eve]+q[odd])/2
    #Iout = (I[eve]+I[odd])/2
    #dIout = np.sqrt(dI[eve]**2 + dI[odd]**2)/2
    Iout,dIout = cmbwe(I[eve],dI[eve],I[odd],dI[odd])
    return {'q':qout, 'I':Iout, 'dI':dIout}
    
def subtract_sets(set1,set2,SF=1):
    # SF is adjustable scale factor
    q = set2['q']
    I2 = set2['I']
    dI2 = set2['dI']
    I1 = set1['I']
    dI1 = set1['dI']
    I = I1/SF-I2
    dI = np.sqrt(dI1**2+dI2**2)
    return {'q':q, 'I':I, 'dI':dI}


def norm_fun(q,air,con,scale,lin,quad):
    y = scale*air+con+lin*q+quad*q**2
    return y

norm_model = Model(norm_fun,independent_vars=['q','air'],
                   param_names = ['con','scale','lin','quad'])               
def trunc_set(set,start):
    q = set['q'][start:]
    I = set['I'][start:]
    dI = set['dI'][start:]
    return {'q':q, 'I':I, 'dI':dI}

def get_SF(data,ref):
    norm_par = Parameters()
    norm_par.add('con',value=.1,vary=True,min=.05,max=.2)
    norm_par.add('scale',value=10,vary=True,min=.9,max=1000)
    norm_par.add('lin',value=0,vary=True,min=-10,max=10)
    norm_par.add('quad',value=0,vary=True,min=-100,max=100)
    q = ref['q'][0:12]
    Ia = ref['I'][0:12]
    Iw = data['I'][0:12]
    wt = 1/Ia
    result = norm_model.fit(Iw,norm_par,q=q,air=Ia,weights=wt)
    return result.params['scale'].value

def get_SF_tail(set1,set2):
    I2 = set2['I']
    I1 = set1['I']
    R = I1/I2
    NR = len(R)
    R = R[int(.75*NR):]
    R = np.sort(R)
    R = R[0:5]
    R = np.mean(R)
    return R

                       
        
