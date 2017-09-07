#### ARMA-GARCH fitting module for python

Created on Thu Jul 13 22:37:23 2017

@author: Magne

+ parameters of ARMA(1,1)-Garch(1,1) model
+ use scipy.optimize.minimize(defult algorithm) to solve maximum likelihood estimation

###### ARMR(1,1) initial
a0_0, a1_0, a2_0 = self.intial_fit_paras[2], self.intial_fit_paras[3], \
self.intial_fit_paras[4]

###### Garch(1,1) initial
GARCH1_1_0, GARCH1_2_0, GARCH1_3_0 = self.intial_fit_paras[5], \
self.intial_fit_paras[6], self.intial_fit_paras[7]

###### skewed t-distributio parameters

nu, lamda0 = self.intial_fit_paras[0], self.intial_fit_paras[1]

##### FUNCTIONS:
+ Get_Sigmat : interation function for GArch(1,1) parameter sigma(t)
+ pdf : a matrix of pdf at each element of u1 of the distribution
+ Skewtdis_max_likelyhood : max likelyhood function

###### VARIANCE:
+ total : 8 parameters
+ xt is time series data, gt = xt(t-1)
+ e1: residual
+ sigmat: conditional variance
+ u1:  i.d.d. variance, obeyskewed-t distribution
+ nu, lambda: skewness-t distritbuion parameters

##### OUTPUTS:
+ res : fitting results for  [nu, sigma, a0, a1, a2, c0, c1, c2]
+ fitval : estimated value using fitting para, compared to the orign data
