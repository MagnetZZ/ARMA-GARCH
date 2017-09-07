# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 22:37:23 2017
@author: Magne
------------------------------------------------------------------------
#parameters of ARMA(1,1)-Garch(1,1) model
#use scipy.optimize.minimize(defult algorithm) to solve maximum likelihood estimation 

# ARMR(1,1) parameters
a0_0, a1_0, a2_0 = self.intial_fit_paras[2], self.intial_fit_paras[3], \
self.intial_fit_paras[4] 

# Garch(1,1) parameters
GARCH1_1_0, GARCH1_2_0, GARCH1_3_0 = self.intial_fit_paras[5], \
self.intial_fit_paras[6], self.intial_fit_paras[7]

# skewed t-distributio parameters 
nu, lamda0 = self.intial_fit_paras[0], self.intial_fit_paras[1]
------------------------------------------------------------------------
#FUNCTIONS:
Get_Sigmat : interation function for GArch(1,1) parameter sigma(t)
pdf : a matrix of pdf at each element of u1 of the distribution  
Skewtdis_max_likelyhood : max likelyhood function 
-----------------------------------------------------------------------
#INPUTS:
Import data: common factor parameters and history fiting data

total : 8 parameters 
xt is time series data, gt = xt(t-1) 
e1: residual
sigmat: conditional variance
u1:  i.d.d. variance, obeyskewed-t distribution 
nu, lambda: skewness-t distritbuion parameters
------------------------------------------------------------------------
OUTPUTS:
res : fitting results for  
fitval : estimated value using fitting para, compared to the orign data
-----------------------------------------------------------------------
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

def Get_Sigmat(paras, et_lag1, sigmat_lag1):  
	c0, c1, c2 = paras[0], paras[1], paras[2]   
	#print('c0 %f, c1:%f, c2:%f'%(c0,c1,c2))
	# alpha_0  alpha_1  beta_1
	sigma_t2 = c0 + c1 * et_lag1*et_lag1 + c2 * sigmat_lag1*sigmat_lag1
	sigma_t = math.sqrt(sigma_t2)
	return sigma_t

#returns the pdf at u1 of Hansen's (1994) 'skewed t' distribution 
def Skewt_dis_pdf(u1, nu, lambda0):

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


def Skewtdis_max_likelyhood(paras): 

	# expression of Maximum Likelihood Estimate
	# nu and lambda: skewed-T disribution parameters	    
	# supposed xt=0 when t=0

	global x_t, g_t
	global intial_residual

	global steps

	n = len(g_t)
	# print("The length of xt is %d" %n)

	nu, lambda0 = paras[0], paras[1]
	a0, a1, a2 = paras[2], paras[3], paras[4] 
	para_GARCH1 = [paras[5], paras[6], paras[7]]
	#print(para_GARCH1[:])  ## test

	# when t=0
	e1, u1, sigmat = np.zeros(n), np.zeros(n), np.zeros(n)
	#print('xt0 %f, gt0:%f '%(x_t[0],g_t[0]))
	e1[0] = x_t[0] - (a0 + a1 * g_t[0])     
	sigmat[0] = Get_Sigmat(para_GARCH1, intial_residual[0], intial_residual[1]) 
	u1[0] = e1[0] / sigmat[0]  # used for Likelihood

	steps = steps+1
	print('=========================================================')
	print('steps: %f'%steps)

	# u1 when i>1
	for i in range(1, n):
		e1[i] = x_t[i] - (a0 + a1 * g_t[i] + a2 * e1[i-1])
		sigmat[i] = Get_Sigmat(para_GARCH1, e1[i-1], sigmat[i-1])
		u1[i] = e1[i] / sigmat[i]
		#print('sigmat %f'%(sigmat[i]))
		
	#calculate the mle based on skewed-t distribution
	#since min(-lnlikelihood)=max(lnlikelihood),a minus is added
	like = []
	like = Skewt_dis_pdf(u1, nu, lambda0)
	lnlike = sum(np.log(like)) - sum(np.log(sigmat))
	f = -lnlike 
	print('f %f'%(f))
	print('=========================================================')
	return f


class Arma11_Garch11_method():

	xt = []
	gt = [] 

	def __init__(self, parameters, parasbound, filename):
		self.intial_fit_paras = parameters
		self.intial_fit_paras_bounds = parasbound
		self.data_filename = filename


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
		#print(len(self.xt))

		# get the optimal parameters	    
		self.gt = [0]
		self.gt.extend(self.xt[0:-1])

		# I: see the stability of series
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
		# raw_input()
		# III: PAC test 
		#####################################################
		# please add PAC test funtion here , and import the package at first
		####################################################




	def sent_xtgt_data(self):
		return [self.xt, self.gt] 


	def Arma_Garch_fitting(self):


		n = len(self.xt)

		bnds = self.intial_fit_paras_bounds
		cons = ({'type': 'ineq', 'fun': lambda x: - x[6] - x[7] + 0.9999})
		#fun = Skewtdis_max_likelyhood(self.intial_fit_paras)
		fun = Skewtdis_max_likelyhood
		print(fun)
		#print(type(fun))

		res = minimize(fun, self.intial_fit_paras, bounds=bnds, constraints=cons)
		#raw_input()
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
		e1_0, sigma1_0 = intial_residual[0], intial_residual[1]


		('pass-------')
		# get estimated time series: fatcval
		e1[0] = self.xt[0]-(a0 + a1*self.gt[0])
		fitval[0] = a0 + a1*self.gt[0]
		sigmat[0] = Get_Sigmat(para_GARCH1, e1_0, sigma1_0)
		u1[0] = e1[0] / sigmat[0]

		for i in range(1,n):
			fitval[i] = (a0 + a1 * self.gt[i] + a2 * e1[i-1]) 
			e1[i] = self.xt[i] - fitval[i]
			sigmat[i] = Get_Sigmat(para_GARCH1, e1[i-1], sigma1t[i-1])
			u1[i] = e1[i] / sigmat[i]

		# rt[0] = a0 + a1 * self.gt[0]
		# sigma1t[0] = self.Get_Sigmat(para_GARCH1, 0, np.std(self.xt, ddof = 1))
		# for i in range(1,n):
		#	rt[i] = a0 + a1 * self.gt[i] + a2 * e1[i-1]
		#	sigma1t[i] = self.Get_Sigmat(para_GARCH1, e1[i-1], [i-1])


if __name__ == '__main__':
	ini_paras = (2.01, -0.99, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1)
	bounds = ((2, 10000000000), (-1, 1), (-100, 100), (-100, 100), (-100, 100), \
		(0, 100), (0.1, 1), (0.1, 1))
	filename = 'SHXZ1.csv' 
	intial_residual = [0, 0.1]   # the initial e1[0] and sigmat[0] at t=0
	AG_member = Arma11_Garch11_method(ini_paras, bounds, filename)
	AG_member.AG_data_process()
	[x_t, g_t] = AG_member.sent_xtgt_data()
	steps = 0

	try:
		AG_member.Arma_Garch_fitting()
	except Exception as e:
		logging.warning(e)
