# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 14:03:49 2021

@author: lluri
"""
from liposome_plotting import  plot_final_results
from matplotlib.backends.backend_pdf import PdfPages
resultname = 'Results/liposome_fit_results_264150.npz'
resultname = 'Results/liposome_fit_results_773369.npz'
figname = 'temp_results_2.pdf'
pp = PdfPages(figname)
plot_final_results(pp,resultname)
pp.close()
