# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 22:37:23 2017
@author: Magne
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
########

def skewtdismle(paras):  
	    """
    expression of Maximum Likelihood Estimate
    nu and lambda: skewed-T disribution parameters
    x is time series data, g = x(t-1) 
    e1: residual  
    u1: gaussian noise distribution obeying i.d.d.
    sigma1: conditional variance(sigma) 	    
    """

    g = [0]
    g.extend(self.x[0:-1])
    n = len(self.x)
    # print("The length of x is %d" %n)
    
    nu, self.lambda = parameters[0], parameters[1]
    a0, a1, a2 = parameters[2], parameters[3], parameters[4]
    c0, c1, c2 = parameters[5], parameters[6], parameters[7]  
    para_GARCH1 = [c0, c1, c2]
    print(para_GARCH1[:])  ## test

    e1_0, sigma1_0 = self.intial_fit_paras[0:1]
    # generate u1 when t=0 (supposed x=0 when t=0)：
    e1 = np.zeros(n)
    u1 = np.zeros(n)
    sigma1 = np.zeros(n)
    e1[0] = x[0] - (a0 + a1 * g[0])     
    sigma1[0] = self.__sigmat__gmat__(para_GARCH1, e1_0, sigma1_0) 
    u1[0] = e1[0] / sigma1[0]  # used for Likelihood
    e1_0 = e1[0]
    sigma1_0 = sigma1[0]
    # u1 when i>1
    for i in range(1, n):
        e1[i] = x[i] - (a0 + a1 * g[i] + a2 * e1[i-1])
        # regression equation
        sigma1[i] = self.__sigmat__(para_GARCH1, e1_0, sigma1_0)
        u1[i] = e1[i] / sigma1[i]
        # r0=x(i,1)        
        e1_0 = e1[i] 
        sigma1_0 = sigma1[i]    
    # calculate the mle based on skewed-t distribution
    like = 1
    like = skewtdis_pdf(u1, nu, lambda0)
    lnlike = sum(np.log(like)) - sum(np.log(sigma1))
    f = -lnlike 
    #since min(-lnlikelihood)=max(lnlikelihood)  
    #in orger to get minimum value using fmincon, a minus is added
    return f





class Arma_Garch():

    def __init__(self, paras, parasbound, filename):
		self.intial_fit_paras = paras
		self.intial_fit_paras_bounds = parasbound
		self.data_filename = filename

	def __sigmat__(para, et_lag1, sigmat_lag1):  ##########################################################################
	    c0, c1, c2 = para[0], para[1], para[2]   
	    # alfa_0  alfa_1  beta_1
	    sigma_t2 = c0 + c1 * et_lag1*et_lag1 + c2 * sigmat_lag1*sigmat_lag1
	    sigma_t = math.sqrt(sigma_t2)
	    return sigma_t
	#########
	def __skewtdis_pdf__(self):
	# def __skewtdis_pdf__(self, u1, nu, lambda):	
	    """
	    returns the pdf at u1 of Hansen's (1994) 'skewed t' distribution 
	    pdf = skewtdis_pdf(u1,nu,lambda)
	    INPUTS:
	    u1 = gaussian noise distribution obeying i.d.d.
	    nulist = a matrix or scalar degrees of freedom parameter 
	    lambda_list = vector of skewness-t distritbuion result 
	    # 
	    OUTPUTS:
	    pdf = a matrix of pdf at each element of u1 of the distribution      
	    """
	    Gauss_u1 = self.u1
	    T, n = len(Gauss_u1), len(Gauss_u1)
	    pdf1, pdf2, pdf = np.zeros(n), np.zeros(n), np.zeros(n)
	    nulist = self.nu * np.ones(T)  # initializing
	    lambda_list = self.lambda * np.ones(T)
	    c = gamma((nulist+1)/2) / (np.sqrt(math.pi*(nulist-2))*gamma(nulist/2))
	    a = 4.0 * lambda_list * c* ((nulist-2)/(nulist-1))
	    b = np.sqrt(1 + 3 * lambda_list * lambda_list - a * a)
	    avb=-a/b
	
	    pdf1 = b*c*np.power(1 + 1/(nulist-2)* np.power((b*Gauss_u1+a)/(1-lambda_list), 2), -(nulist+1)/2)
	    pdf2 = b*c*np.power(1 + 1/(nulist-2)* np.power((b*Gauss_u1+a)/(1+lambda_list), 2), -(nulist+1)/2)
	    self.pdf = np.zeros(n)
	    for i in range(0,n):
	        self.pdf[i] = pdf1[i]*int(Gauss_u1[i]<avb[i]) + pdf2[i] *int(Gauss_u1[i] >= avb[i])
	##############

		"""
		### main programe ###
		# use scipy.optimize.minimize to calculate the mle equation and get the 
		# parameters of ARMA(1,1)-Garch(1,1) model
		# total : 8+x parameters (x is the parameters pass into the program) 
		"""
		# Import data: common factor parameters and history fiting data
		# 矩阵？让他们把参数按列传进来
		# 函数传进来，读列

	# if __name__ == '__main__':   ###############################################################################
    def __Arma_Garch_fitting__():

	    csv_file = open('SHSZ.csv','r')  #under th current folder 
	    reader = csv.DictReader(csv_file)
	    x_str = [row['ClPr'] for row in reader][0:1943]
	    n = len(x_str)
	    x0 = np.zeros(n) #n time scale
	    for i, items in enumerate(x_str):
	        x0[i] = float(items)
	    csv_file.close()
	    IndexData = x0
	    IndexData_1 = np.diff(IndexData)  # first log;
	    IndexData_2 = np.diff(IndexData_1)  # second difference
	    # IndexData_2 = IndexData_1  # second difference

	    # first, see the stability of series
	    plt.figure(1, figsize=(18,45))
	    ax1 = plt.subplot(311) 
	    ax2 = plt.subplot(312) 
	    ax3 = plt.subplot(313)
	    plt.sca(ax1) 
	    plt.plot(IndexData)
	    plt.sca(ax2) 
	    plt.plot(IndexData_1) 
	    plt.sca(ax3) 
	    plt.plot(IndexData_2)
	    plt.savefig("data.pdf")
	    plt.show()
	    #using second diff of Pareto-efficiency 
	    x = IndexData_2 
	    n = len(x)
	    result = sm.tsa.adfuller(x, regresults=True)
	    print(result)

	    # second, correlation test
	    m = 10  # numbers of coefficient to be test
	    acf, q, p = sm.tsa.acf(x, nlags=m, qstat=True)  ## autocorrelation coefficient and p-value
	    out = np.c_[range(1,11), acf[1:], q, p]
	    output=pd.DataFrame(out, columns=['lag', "AC", "Q", "P-value"])
	    output = output.set_index('lag')
	    print(output)

	    # get the optimal parameters
	    g = [0]
	    g.extend(x[0:-1])
	    #ARMR(1,1) parameters
	    a0_0, a1_0, a2_0 = 0.1, 0.1, 0.1 
	    a = [a0_0, a1_0, a2_0] 
	    # Garch(1,1) parameters
	    GARCH1_1_0, GARCH1_2_0, GARCH1_3_0 = 0.1, 0.2, 0.1
	    para_GARCH10 = [GARCH1_1_0, GARCH1_2_0, GARCH1_3_0]
	    # t-distributio parameters 
	    nu, lamda0 = 2.01, -0.99
	    para0 = [nu, lamda0, a, para_GARCH10]
	    bnds = ((2, 10000000000), (-1, 1), (-100, 100), (-100, 100), (-100, 100), \
	    (0, 100), (0.1, 1), (0.1, 1)) 
	    cons = ({'type': 'ineq', 'fun': lambda x: - x[6] - x[7] + 0.9999})
	    # (lower, upper) bounds for parameters
	    fun = skewtdismle
	    # --test: --
	    #fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
	    #cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2})
	    #bnds = ((0, None), (0, None))
	    #res = minimize(fun, (0.1, 0.1), method='SLSQP', bounds=bnds, constraints=cons)
	    res = minimize(fun, (2.01, -0.99, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1), \
	                   bounds=bnds, constraints=cons)
	    # initialization:
	    para = res.x
	    print(res.x)
	    print(res.message)
	    
	    # begin to get the estimated value 
	    e1 = np.zeros(n)
	    u1 = np.zeros(n)
	    sigma1 = np.zeros(n)
	    sigma1t = np.zeros(n)
	    rt = np.zeros(n)
	    #
	    nu = para[0]
	    lamda0 = para[1]
	    a0, a1, a2 = para[2], para[3], para[4]
	    c0, c1, c2 = para[5], para[6], para[7]
	    para_GARCH1 = [c0, c1, c2]
	    fitval = np.zeros(n)
	    e1_0, sigma1_0 = 0, 0.01
	    # 0 = mean(x) 
	    e1[0] = x[0]-(a0 + a1*g[0])  ## residual error
	    fitval[0] =  a0 + a1*g[0]
	    sigma1[0] = self.__sigmat__(para_GARCH1, e1_0, sigma1_0)
	    u1[0] = e1[0] / sigma1[0]
	    # r0=x(i,1)
	    e1_0 = e1[0]
	    sigma1_0 = sigma1[0]
	    #
	    for i in range(1,n):
	        fitval[i] = (a0 + a1 * g[i] + a2 * e1[i-1]) 
	        e1[i] = x[i] - fitval[i]
	    # regression equation
	        sigma1[i] = self.__sigmat__(para_GARCH1, e1_0, sigma1_0)    
	        u1[i] = e1[i] / sigma1[i]
	    #r0=x[i]
	        e1_0 = e1[i] 
	        sigma1_0 = sigma1[i]
	    rt[0] = a0 + a1 * g[0]
	    sigma1t[0] = self.__sigmat__(para_GARCH1, 0, np.std(x, ddof = 1))
	    for i in range(1,n):
	        rt[i] = a0 + a1 * g[i] + a2 * e1[i-1]
	        sigma1t[i] = self.__sigmat__(para_GARCH1, e1[i-1], sigma1[i-1])


if __name__ == '__main__':
	intial_fit_paras = (2.01, -0.99, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1)
	intial_fit_paras_bounds = ((2, 10000000000), (-1, 1), (-100, 100), (-100, 100), (-100, 100), \
	    (0, 100), (0.1, 1), (0.1, 1))
	data_filename = 'SHSZ.csv' 
	AG_member = Arma_Garch(intial_fit_paras, intial_fit_paras_bounds, data_filename)
	try:
		AG_member.Arma_Garch_fitting()
    except Exception as e:
        logging.warning(e)

