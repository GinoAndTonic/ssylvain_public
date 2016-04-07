from __future__ import division
import shelve
import psutil
import subprocess
import os
import datetime as DT
import matplotlib.dates as mdates
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
import pylab as plab
import sys
print sys.argv
from xlrd import open_workbook
import xlwt
from irp_h import *
from kalman import *
from extract_parameters import *
from math import exp
from scipy.optimize import minimize, fmin_slsqp
from scipy.linalg import expm
from scipy import integrate
from estim_constraints import *
from scipy.stats import norm
from import_data import *
from estimation_rolling import *
plt.close("all")
plt.close()
plt.ion()
# plt.rc('text', usetex=True)     #for TeX interpretation of title and legends
__author__ = 'ssylvain'

for t in range(1000):
    start_time = time.time()
    print start_time

########################################################################################################################
tgap = 1 # the frequency with which we run the estimation
rolling = 1 # 0 if using expanding window, 1 if using rolling window

np.set_printoptions(precision=32, suppress=True)
# globals
global a, Kp, lmda,lmda2,  Phi, sigma11, sigma22, sigma22_2, sigma33, sigma33_2, sigma44, Sigma, thetap, Nfeval, figures, cum_log_likelihood,\
    Nfeval_vec, cum_log_likelihood_vec, doc, Nfeval_inner


# PRIMITIVES:
figures = []
allow_missing_data = 0
estim_freq = 'weekly'  # estimation frequency: 'daily', 'weekly', 'monthly','quarterly'
fix_Phi = 1     # "1" if you want to fix the volatility of observed yields using covar of historical data
                # "0" if you want to jointly estimate it with other model parameters
setdiag_Kp = 1  # "1" if you want to Kp to be diagonal so the state variables are assumed independent
                # "0" if you want to Kp to be unrestricted

# options for initializing the error variance: 'steady_state' or 'unconditional' or 'identity' matrix
initV = 'unconditional'
num_states = 4      # number of states
if ((initV == 'steady_state') | (initV == 'unconditional')):
    stationarity_assumption = 'yes'
else:
    stationarity_assumption = 'no'

US_ilbmaturities = np.array([2, 3,  5, 6, 8, 9, 10])
US_nominalmaturities = np.array([2, 3,  5, 6, 8, 9, 10])
US_num_maturities = len(US_ilbmaturities) + len(US_nominalmaturities)
US_maturities = np.hstack((US_nominalmaturities, US_ilbmaturities))

# frequency of data for estimation:
if estim_freq == 'daily':
    dt = 1.0/252  # daily increment
elif estim_freq == 'weekly':
    dt = 1.0/52     # weekly increment
elif estim_freq == 'monthly':
    dt = 1.0/12     # monthly increment
elif estim_freq == 'quarterly':
    dt = 1.0/4      # quarterly increment

# import importUS_Data
sdate = np.array('2004-01-01', dtype=np.datetime64)
edate = np.array(time.strftime("%Y-%m-%d"), dtype=np.datetime64)  # in format : '2015-02-11'
# yield data is updated daily. no need to re-import data every time we run code
output_filename=r'S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python\outputfiles\raw_data_US.out'
if os.path.isfile(r'S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python\outputfiles\raw_data_US.out'):
    if time.strftime("%Y-%m-%d",(time.gmtime(os.path.getmtime(output_filename)))) < edate: # last modified date is not current date
        data_US_NB, US_NB_dates, data_US_ILB, US_ILB_dates = ImportData().importUS_Data(US_ilbmaturities, US_nominalmaturities)
    else: #simple load existing data
        my_shelf = shelve.open(output_filename)
        for key in my_shelf:
            globals()[key]=my_shelf[key]
        my_shelf.close()
else:
    data_US_NB, US_NB_dates, data_US_ILB, US_ILB_dates = ImportData().importUS_Data(US_ilbmaturities, US_nominalmaturities)

fulldata, fulldates = ImportData().extract_subset(data_US_NB, US_NB_dates, data_US_ILB, US_ILB_dates,sdate, edate, allow_missing_data, estim_freq)
yld_dates = np.union1d(fulldates['US_NB'], fulldates['US_ILB'])
yld_dates_mod = np.array([yld_dates.astype(DT.datetime)[tt].year*10000+yld_dates.astype(DT.datetime)[tt].month*100+yld_dates.astype(DT.datetime)[tt].day for tt in np.arange(0,yld_dates.shape[0])])

if os.path.isfile(r'S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python\txt_files\prob_def_'+estim_freq+'.txt')==False:  # if files do not exist, creat new ones with appropriate headers
    prev_estim_date = yld_dates_mod[(1/dt)*2-1-1]
    tinc = tgap
else:
    # read date column from existing estimation
    prev_estim_date = np.loadtxt(r'S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python\txt_files\prob_def_'+estim_freq+'.txt', delimiter=',', skiprows=2)[-1,0]



if prev_estim_date < yld_dates_mod[-1]: # this is so that we run the estimation only for new data
    tinc = np.searchsorted(yld_dates_mod, prev_estim_date)-(1/dt)*2+1+tgap
    if rolling==1:
        sdate = yld_dates[0+tinc]
    edate = yld_dates[(1/dt)*2-1+tinc]  # 2yr rolling calc
    tinc += tgap
    while True:
        data, dates = ImportData().extract_subset(data_US_NB, US_NB_dates, data_US_ILB, US_ILB_dates,sdate, edate, allow_missing_data, estim_freq)

        Rolling(data, dates, US_ilbmaturities, US_nominalmaturities, dt, US_num_maturities, estim_freq, num_states, fix_Phi, setdiag_Kp, initV, stationarity_assumption)
        try:
            if rolling==1:
                sdate = yld_dates[0+tinc]
            edate = yld_dates[(1/dt)*2-1+tinc]  # 2yr rolling calc
            tinc += tgap
        except:
            break
end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))