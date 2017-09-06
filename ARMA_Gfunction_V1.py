# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 22:37:23 2017
@author: Magne

###
use scipy.optimize.minimize to calculate the mle equation and get the 
parameters of ARMA(1,1)-Garch(1,1) model
total : 8 parameters 
Import data: common factor parameters and history fiting data

FUNCTIONS:
pdf = a matrix of pdf at each element of u1 of the distribution  


INPUTS:
xt is time series data, gt_series = xt_(t-1) 
e1: residual
sigmat: conditional variance
u1:  i.d.d. variance, obeyskewed-t distribution 
nu, lambda: skewness-t distritbuion parameters


OUTPUTS:
res: 
rt: fitting data using estimated parameters 
fitval:

COMMENTS:

"""

import math 
import numpy as np
import scipy as sp
from scipy import  stats
from scipy.optimize import minimize
from scipy.special import gamma
import pandas as pd
import statsmodels.api as sm 
import types
import csv
import matplotlib.pyplot as plt
import logging
import sys

def Skewtdis_max_likelyhood(self, paras): 

	# expression of Maximum Likelihood Estimate
	# nu and lambda: skewed-T disribution parameters	    
	# supposed xt=0 when t=0

	global data_xt, data_gt
	global intial_residual

	n = len(g_t)
	# print("The length of xt is %d" %n)

	nu, lambda0 = paras[0], paras[1]
	a0, a1, a2 = paras[2], paras[3], paras[4] 
	para_GARCH1 = parameters[5], parameters[6], parameters[7] 
	print(para_GARCH1[:])  ## test

	# when t=0
	e1[0], sigmat[0] = intial_residual[0], intial_residual[1]
	e1, u1, sigmat = np.zeros(n), np.zeros(n), np.zero(n)
	e1[0] = self.xt[0] - (a0 + a1 * self.gt_series[0])     
	sigmat[0] = self.Get_Sigmat(para_GARCH1, 0, 0.01) 
	u1[0] = e1[0] / sigmat[0]  # used for Likelihood

	# u1 when i>1
	for i in range(1, n):
		e1[i] = self.xt[i] - (a0 + a1 * self.gt_series[i] + a2 * e1[i-1])
		sigmat[i] = self.Get_Sigmat(para_GARCH1, e1[i-1], sigmat[i-1])
		u1[i] = e1[i] / sigmat[i]      
		
	#calculate the mle based on skewed-t distribution
	#since min(-lnlikelihood)=max(lnlikelihood),a minus is added
	like = []
	like = self.Skew_tdis_pdf(u1, nu, lambda0)
	lnlike = sum(np.log(like)) - sum(np.log(sigmat))
	f = -lnlike 
	return f


class Arma11_Garch11_method():

	pdf = []
	xt = []
	gt_series = [] 
	intial_residual = [0, 0.1]  # the initial e1[0] and sigmat[0] at t=0

	def __init__(self, parameters, parasbound, filename):
		self.intial_fit_paras = parameters
		self.intial_fit_paras_bounds = parasbound
		self.data_filename = filename


	def Get_Sigmat(self, paras, et_lag1, sigmat_lag1):  
		c0, c1, c2 = para[0], para[1], para[2]   
		# alpha_0  alpha_1  beta_1
		sigma_t2 = c0 + c1 * et_lag1*et_lag1 + c2 * sigmat_lag1*sigmat_lag1
		sigma_t = math.sqrt(sigma_t2)
		return sigma_t

	#returns the pdf at u1 of Hansen's (1994) 'skewed t' distribution 
	def Skewt_dis_pdf(self, u1, nu, lambda0):

		Gauss_u1 = u1
		T, n = len(Gauss_u1), len(Gauss_u1)
		pdf1, pdf2, pdf = np.zeros(n), np.zeros(n), np.zeros(n)
		nulist = nu * np.ones(T)  # initializing
		lambda_list = lambda0 * np.ones(T)
		c = gamma((nulist+1)/2) / (np.sqrt(math.pi*(nulist-2))*gamma(nulist/2))
		a = 4.0 * lambda_list * c* ((nulist-2)/(nulist-1))
		b = np.sqrt(1 + 3 * lambda_list * lambda_list - a * a)
		avb=-a/b

		pdf1 = b*c*np.power(1 + 1/(nulist-2)* np.power((b*Gauss_u1+a)/(1-lambda_list), 2), -(nulist+1)/2)
		pdf2 = b*c*np.power(1 + 1/(nulist-2)* np.power((b*Gauss_u1+a)/(1+lambda_list), 2), -(nulist+1)/2)
		pdf = np.zeros(n)
		for i in range(0, n):
		    pdf[i] = pdf1[i]*int(Gauss_u1[i]<avb[i]) + pdf2[i] *int(Gauss_u1[i] >= avb[i])
		return pdf


	def AG_data_process(self):

		# read under th current folder, csv name, it will read the 
		csv_file = open(self.data_filename,'r')
		reader = csv.DictReader(csv_file)
		x_str = [row['closeIndex'] for row in reader]
		# print (type(x_str))
		csv_file.close()

		n = len(x_str)    # n time scale
		x0 = np.zeros(n)
		for i, items in enumerate(x_str):
			x0[i] = float(items)

		IndexData = x0
		# first log
		IndexData_1 = np.diff(IndexData)
		# second difference
		IndexData_2 = np.diff(IndexData_1)
		# IndexData_2 = IndexData_1  # second difference

		self.xt = IndexData_2 


	def sent_xt_data(self):
		return self.xt


	def Arma_Garch_fitting(self):

		# I: see the stability of series
		n = len(self.xt)
		result = sm.tsa.adfuller(self.xt, regresults=True)
		print(result)

		# II: correlation test
		# numbers of coefficient to be test
		m = 10  
		## autocorrelation coefficient and p-value
		acf, q, p = sm.tsa.acf(self.xt, nlags=m, qstat=True)  
		out = np.c_[range(1,11), acf[1:], q, p]
		output=pd.DataFrame(out, columns=['lag', "AC", "Q", "P-value"])
		output = output.set_index('lag')
		print(output)

		# III: PAC test 
		#####################################################
		# please add PAC test funtion here , and import the package at first
		####################################################

		# get the optimal parameters	    
		self.gt_series = [0].extend(self.xt[0:-1])

		#ARMR(1,1) parameters
		self.intial_fit_paras
		a0_0, a1_0, a2_0 = self.intial_fit_paras[2], self.intial_fit_paras[3], \
		self.intial_fit_paras[4] 
		a = [a0_0, a1_0, a2_0] 

		# Garch(1,1) parameters
		GARCH1_1_0, GARCH1_2_0, GARCH1_3_0 = self.intial_fit_paras[5], \
		self.intial_fit_paras[6], self.intial_fit_paras[7]
		para_GARCH10 = [GARCH1_1_0, GARCH1_2_0, GARCH1_3_0]

		# skewed t-distributio parameters 
		nu, lamda0 = self.intial_fit_paras[0], self.intial_fit_paras[1]
		para0 = [nu, lamda0, a, para_GARCH10]

		# initialization:
		bnds = self.intial_fit_paras_bounds
		cons = ({'type': 'ineq', 'fun': lambda x: - x[6] - x[7] + 0.9999})
		fun = self.Skewtdis_max_likelyhood()
		res = minimize(fun, self.intial_fit_paras, bounds=bnds, constraints=cons)
		para = res.x
		print(res.x)
		print(res.message)

		# get fitting parameters
		e1, u1 = np.zeros(n), np.zeros(n) 
		sigmat, sigma1t  = np.zeros(n), np.zeros(n)
		rt = np.zeros(n)

		nu, lamda0 = para[0], para[1]
		a0, a1, a2 = para[2], para[3], para[4]
		para_GARCH1 = para[5], para[6], para[7]
		fitval = np.zeros(n)
		e1_0, sigma1_0 = self.intial_residual[0], self.intial_residual[1]


		# get estimated time series: fatcval
		e1[0] = self.xt[0]-(a0 + a1*self.gt_series[0])
		fitval[0] = a0 + a1*self.gt_series[0]
		sigmat[0] = self.Get_Sigmat(para_GARCH1, e1_0, sigma1_0)
		u1[0] = e1[0] / sigmat[0]

		for i in range(1,n):
			fitval[i] = (a0 + a1 * self.gt_series[i] + a2 * e1[i-1]) 
			e1[i] = self.xt[i] - fitval[i]
			sigmat[i] = self.Get_Sigmat(para_GARCH1, e1[i-1], sigma1[i-1])
			u1[i] = e1[i] / sigmat[i]

		# rt[0] = a0 + a1 * self.gt_series[0]
		# sigma1t[0] = self.Get_Sigmat(para_GARCH1, 0, np.std(self.xt, ddof = 1))
		# for i in range(1,n):
		#	rt[i] = a0 + a1 * self.gt_series[i] + a2 * e1[i-1]
		#	sigma1t[i] = self.Get_Sigmat(para_GARCH1, e1[i-1], [i-1])


if __name__ == '__main__':
	ini_paras = (2.01, -0.99, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1)
	bounds = ((2, 10000000000), (-1, 1), (-100, 100), (-100, 100), (-100, 100), \
		(0, 100), (0.1, 1), (0.1, 1))
	filename = 'data.csv' 
	AG_member = Arma11_Garch11_method(ini_paras, bounds, filename)
	try:
		AG_member.Arma_Garch_fitting()
	except Exception as e:
		logging.warning(e)
