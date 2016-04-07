from __future__ import division
import shelve
import psutil
import subprocess
import os
import datetime as DT
import matplotlib.dates as mdates
import time
import numpy as np
# import numpy.ma as ma
import matplotlib.pyplot as plt
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
# from scipy.optimize import rosen, differential_evolution
from scipy.linalg import expm
from scipy import integrate
from estim_constraints import *
from scipy.stats import norm
#from openopt import NLP
# from  matplotlib.pyplot import ion
# ion()
plt.close("all")
plt.close()
plt.ion()
# plt.rc('text', usetex=True)     #for TeX interpretation of title and legends
__author__ = 'ssylvain'
start_time = time.time()
########################################################################################################################


np.set_printoptions(precision=32, suppress=True)
# globals
global a, Kp, lmda, Phi, sigma11, sigma22, sigma33, sigma44, Sigma, thetap, Nfeval, figures, cum_log_likelihood,\
    Nfeval_vec, cum_log_likelihood_vec, doc, Nfeval_inner


# PRIMITIVES:
figures = []
allow_missing_data = 0
estim_freq = 'weekly'  # estimation frequency: 'daily', 'weekly', 'monthly','quarterly'
fix_Phi = 1     # "1" if you want to fix the volatility of observed yields using covar of historical data
                # "0" if you want to jointly estimate it with other model parameters

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

# start and end dates for estimation
sdate = np.array('2010-01-01', dtype=np.datetime64)
edate = np.array(time.strftime("%Y-%m-%d"), dtype=np.datetime64)  # in format : '2015-02-11'

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
data_US_NB, US_NB_dates, data_US_ILB, US_ILB_dates = ([] for i in range(4))
execfile("importUS_Data.py")

# storing copies of the data in dictionaries:
data = {'US_NB': data_US_NB, 'US_ILB': data_US_ILB}
dates = {'US_NB': US_NB_dates, 'US_ILB': US_ILB_dates}

# creating ILBs and Nominal Bond objects
USilbs = np.array([InfLinkBonds(m, 'USA') for m in US_ilbmaturities])
USnominals = np.array([NominalBonds(m, 'USA') for m in US_nominalmaturities])


if allow_missing_data == 1:
    for vv in ['US_NB', 'US_ILB']:
        loc = np.array((dates[vv] >= sdate)&(dates[vv] <= edate))[:, 0]   # valid US NB dates

        dates[vv] = dates[vv][loc]

        data[vv] = data[vv][loc, :]

    # taking the union of dates and padding data where appropriate
    tempdates = np.union1d(dates['US_NB'], dates['US_ILB'])
    for vv in ['US_NB', 'US_ILB' ]:
        tempdata = np.empty([tempdates.size, data[vv].shape[1]]) * np.nan     # matrix with nan
        temploc = np.in1d(tempdates, dates[vv])
        tempdata[temploc, :] = data[vv]
        dates[vv] = tempdates
        data[vv] = tempdata
        # masking missing variables so they are not used in calculations (THIS DOES NOT WORK AS I THOUGHT)
        #data[vv] = ma.masked_array(data[vv], mask=np.array(data[vv] == np.nan))
else:
    for vv in ['US_NB', 'US_ILB' ]:
        loc2 = np.array((dates[vv] >= sdate)&(dates[vv] <= edate))[:, 0]    # valid US NB dates

        dates[vv] = dates[vv][loc2]
        data[vv] = data[vv][loc2, :]

        loc3 = np.array(np.sum(data[vv] != np.nan, axis=1) == data[vv].shape[1])
        dates[vv] = dates[vv][loc3]
        data[vv] = data[vv][loc3, :]

    # keep only dates in common:
    loc_ilb2 = np.in1d(dates['US_ILB'], dates['US_NB'])  # intersecting dates
    dates['US_ILB'] = dates['US_ILB'][loc_ilb2]
    data['US_ILB'] = data['US_ILB'][loc_ilb2, :]

    loc_nb2 = np.in1d(dates['US_NB'], dates['US_ILB'])  # intersecting dates
    dates['US_NB'] = dates['US_NB'][loc_nb2]
    data['US_NB'] = data['US_NB'][loc_nb2, :]


# extracting dates at the right interval
if estim_freq == 'daily':
    timeloop = np.arange(0, dates['US_NB'].shape[0], 1)  # daily increment
elif estim_freq == 'weekly':
    timeloop = np.arange(0, dates['US_NB'].shape[0], 7)  # weekly increment
elif estim_freq == 'monhtly':
    timeloop = np.arange(0, dates['US_NB'].shape[0], 30)  # monthly increment
elif estim_freq == 'quarterly':
    timeloop = np.arange(0, dates['US_NB'].shape[0], 120)  # quarterly increment

for vv in ['US_NB', 'US_ILB']:
    dates[vv] = dates[vv][timeloop]
    data[vv] = data[vv][timeloop, :]

# storing yields time series in bond objects:
for m in range(US_ilbmaturities.size):
    USilbs[m].setZeroYieldsTS = data['US_ILB'][:, m]/100
    USilbs[m].setZeroYieldsDates = dates['US_ILB']


for m in range(US_nominalmaturities.size):
    USnominals[m].setZeroYieldsTS = data['US_NB'][:, m]/100
    USnominals[m].setZeroYieldsDates = dates['US_NB']


Y = np.mat(np.hstack((data['US_NB'], data['US_ILB'])))/100

# initializing parameters

# using numbers from Christensen, Diebold, Rudebusch (2010)
a = 0.677
Kp = np.array([1.305, 0, 0, -1.613, 1.559, 0.828, -1.044, 0, 0, 0, 0.884, 0, -1.531, -0.364, 0, 1.645])     # reshape(num_states, num_states)
Kp = np.reshape(np.identity(num_states),(num_states**2))
lmda = 0.5319
sigmas = np.array([0.0047, 0.00756, 0.02926, 0.00413])
thetap = np.array([0.06317, -0.1991, -0.00969, 0.03455])
if num_states == 6:
    #Kp = np.array([1.305, 0, 0, -1.613, 0, 0, 1.559, 0.828, -1.044, 0, 0, 0, 0, 0, 0.884, 0, 0, 0, -1.531, -0.364, 0, 1.645, 0, 0,
    #                 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])     # reshape(num_states, num_states)
    Kp = np.reshape(np.identity(num_states),(num_states**2))
    lmda2 = 0.5319
    sigmas = np.array([0.0047, 0.00756, 0.00756, 0.02926, 0.02926, 0.00413])
    thetap = np.array([0.06317, -0.1991, -0.1991, -0.00969, -0.00969, 0.03455])
Phi_prmtr = ((np.diag(np.cov(Y.T)) - 1e-16 + 1)**2-1)**0.5  # estimate Phi
if fix_Phi == 0:
    prmtr_0 = np.random.rand(1 + num_states**2 + 1 + num_states + US_num_maturities + num_states)
    if num_states == 6:
        prmtr_0 = np.random.rand(1 + num_states**2 + 1 + 1 + num_states + US_num_maturities + num_states)
    # repopulate initial parameter vector:
    prmtr_0 = np.array([a])
    prmtr_0 = np.append(prmtr_0, Kp)
    prmtr_0 = np.append(prmtr_0, ((lmda - 1e-16 + 1)**2-1)**0.5)  #  to impose non-negativity
    if num_states == 6:
            prmtr_0 = np.append(prmtr_0, ((lmda2 - 1e-16 + 1)**2-1)**0.5)  #  to impose non-negativity
    prmtr_0 = np.append(prmtr_0, Phi_prmtr)  #  to impose non-negativity
    prmtr_0 = np.append(prmtr_0, ((sigmas - 1e-16 + 1)**2-1)**0.5) #  to impose non-negativity
    prmtr_0 = np.append(prmtr_0, thetap)
else:
    prmtr_0 = np.random.rand(1 + num_states**2 + 1 + num_states + num_states)
    if num_states == 6:
        prmtr_0 = np.random.rand(1 + num_states**2 + 1 + 1 + num_states + num_states)
    Phi_prmtr = ((np.diag(np.cov(Y.T)) - 1e-16 + 1)**2-1)**0.5  #  to impose non-negativity  # estimate Phi
    # populate initial parameter vector:
    prmtr_0 = np.array([a])
    prmtr_0 = np.append(prmtr_0, Kp)
    prmtr_0 = np.append(prmtr_0, ((lmda - 1e-16 + 1)**2-1)**0.5)  #  to impose non-negativity
    if num_states ==6:
        prmtr_0 = np.append(prmtr_0, ((lmda2 - 1e-16 + 1)**2-1)**0.5)  #  to impose non-negativity
    # prmtr_0 = np.append(prmtr_0, ((np.diag(Phi) - 1e-16 + 1)**2-1)**0.5)  #  to impose non-negativity
    prmtr_0 = np.append(prmtr_0, ((sigmas - 1e-16 + 1)**2-1)**0.5) #  to impose non-negativity
    prmtr_0 = np.append(prmtr_0, thetap)


def prmtr_ext(prmtr):   #function to insert parameters for Phi
    if fix_Phi == 0:
        prmtr_ext = prmtr
    else:
        prmtr_ext = prmtr
        prmtr_ext = np.append(prmtr_ext[:-num_states-num_states], Phi_prmtr)
        prmtr_ext = np.append(prmtr_ext, prmtr[-num_states-num_states:])
    return prmtr_ext

# this is to display the results in the console
Nfeval = 1
Nfeval_inner = 1
Nfeval_vec = np.array(Nfeval)
cum_log_likelihood_vec = np.array(None)
str_form = '{0:9f}  '
str_form2 = np.array('Iter')
str_form3 = '{0:9s}  '
for iv in range(prmtr_0.size):
    str_form = str_form + '  {' + str(iv+1) + ': 3.10f}'
    str_form2 = np.vstack((str_form2, 'prmtr('+str(iv+1)+')'))
    str_form3 = str_form3 + '  {' + str(iv+1) + ':14s}'
str_form = str_form + '  {' + str(prmtr_0.size + 1) + ': 6.10f}'    # leave more space for likelihood
str_form2 = np.reshape(np.vstack((str_form2, '  cumloglik')), prmtr_0.size + 2, 0)
str_form3 = str_form3 + '  {' + str(prmtr_0.size + 1) + ':16s}'
figures['fig3_name'] = '\\optim_iter'


cons = ineq_cons(num_states, US_num_maturities, fix_Phi, setdiag_Kp, Phi_prmtr)
ieconsls = ineq_cons_ls(num_states, US_num_maturities, fix_Phi, setdiag_Kp, Phi_prmtr)
bnds_lvl = prmtr_bounds_lvl(num_states, US_num_maturities)
bnds_exp = prmtr_bounds_exp(num_states, US_num_maturities)

doc = []
adobe = r'C:\Program Files (x86)\Adobe\Reader 11.0\Reader\AcroRd32.exe'
mspaint = 'mspaint.exe'
fh = open("NUL","w")
subprocess.Popen("TASKKILL /F /IM mspaint.exe ", stdout=fh, stderr=fh)
fh.close()
# print psutil.Process().pid

# refining the initial guess
for ref in range(50):
    if num_states == 4:
        a_ref, Kp_ref, lmda_ref, Phi_ref, sigma11_ref, sigma22_ref, sigma33_ref, sigma44_ref, Sigma_ref, thetap_ref = extract_vars(prmtr_ext(prmtr_0), num_states, US_num_maturities)
    elif num_states == 6:
        a_ref, Kp_ref, lmda_ref, lmda2_ref, Phi_ref, sigma11_ref, sigma22_ref, sigma22_2_ref, sigma33_ref, sigma33_2_ref, sigma44_ref, Sigma_ref, thetap_ref = extract_vars(prmtr_ext(prmtr_0), num_states, US_num_maturities)
    try:
        A0_ref, A1_ref, U0_ref, U1_ref, Q_ref = extract_mats(prmtr_ext(prmtr_0), num_states, US_num_maturities, US_nominalmaturities, US_ilbmaturities, dt)
        kalman1 = Kalman(Y, A0_ref, A1_ref, U0_ref, U1_ref, Q_ref, Phi_ref, initV)     # default uses X0, V0 = unconditional mean and error variance
        Ytt_ref, Yttl_ref, Xtt_ref, Xttl_ref, Vtt_ref, Vttl_ref, Gain_t_ref, eta_t_ref, cum_log_likelihood_ref = kalman1.filter()
    except:
        print 'initial guess is no good'
        break
    prmtr_0[-num_states:] = np.array(np.mean(Xtt_ref[5:,:],0).T)[:,0]


# objective function:
def neg_cum_log_likelihood(prmtr):
    global Nfeval_inner, Nfeval_vec, cum_log_likelihood, cum_log_likelihood_vec, figures
    # try:
    if num_states == 4:
        a_out, Kp_out, lmda_out, Phi_out, sigma11_out, sigma22_out, sigma33_out, sigma44_out, Sigma_out, thetap_out = extract_vars(prmtr_ext(prmtr), num_states, US_num_maturities)
    elif num_states == 6:
        a_out, Kp_out, lmda_out, lmda2_out, Phi_out, sigma11_out, sigma22_out, sigma22_2_out, sigma33_out, sigma33_2_out, sigma44_out, Sigma_out, thetap_out = extract_vars(prmtr_ext(prmtr), num_states, US_num_maturities)
    if (min(np.real(np.linalg.eig(np.array(Kp_out))[0])) < 0) & (stationarity_assumption == 'yes'):
        cum_log_likelihood = -np.inf
    else:
        try:
            A0_out, A1_out, U0_out, U1_out, Q_out = extract_mats(prmtr_ext(prmtr), num_states, US_num_maturities, US_nominalmaturities, US_ilbmaturities, dt)
            kalman1 = Kalman(Y, A0_out, A1_out, U0_out, U1_out, Q_out, Phi_out, initV)     # default uses X0, V0 = unconditional mean and error variance
            Ytt, Yttl, Xtt, Xttl, Vtt, Vttl, Gain_t, eta_t, cum_log_likelihood = kalman1.filter()
            kalman1.plot(Ytt, Yttl, Xtt, Xttl, Vtt, Vttl, Gain_t, eta_t, figures, US_ilbmaturities, US_nominalmaturities)
            plt.close("all")
        except:
            cum_log_likelihood = -np.inf
    # except:
    #    cum_log_likelihood = -np.inf
    out_n = np.hstack((Nfeval_inner, prmtr, cum_log_likelihood))
    print str_form.format(*tuple(out_n))    # use * to unpack tuple
    if Nfeval_inner == 1:
        cum_log_likelihood_vec = np.array(cum_log_likelihood)
    else:
        cum_log_likelihood_vec = np.vstack((cum_log_likelihood_vec, cum_log_likelihood))
    Nfeval_inner += 1
    # if Nfeval_inner == 43:
    #    print 'stop'
    Nfeval_vec = np.vstack((Nfeval_vec, Nfeval_inner))
    return (-1) * cum_log_likelihood    # we will minimize the negative of the likelihood


def callbackF(prmtr):   # this is to plot and display results at each iteration
    global Nfeval, Nfeval_inner, Nfeval_vec, filename, figures, cum_log_likelihood, cum_log_likelihood_vec, doc
    cumlik = cum_log_likelihood     # -neg_cum_log_likelihood(prmtr)
    if Nfeval == 1:
        print 'inner iterations ends__________________________________________'        # cum_log_likelihood_vec = np.array(cumlik)
    else:
        cum_log_likelihood_vec = np.vstack((cum_log_likelihood_vec, cumlik))
        fh = open("NUL","w")
        subprocess.Popen("TASKKILL /F /IM mspaint.exe ", stdout=fh, stderr=fh)
        fh.close()
    # out_n = np.hstack((Nfeval, prmtr, cumlik))
    # print str_form.format(*tuple(out_n))    # use * to unpack tuple
    # print 'inner iterations begins__________________________________________'
    plt.close()
    fig, ax = plt.subplots(2, sharex=False)
    figures['fig3'] = fig
    figures['ax3'] = ax
    figures['ax3'][0].scatter(Nfeval_vec[:-1,:], cum_log_likelihood_vec)
    # plt.title('Iteration Details')
    figures['ax3'][0].set_title('Log Likelihood')
    # figures['ax3'][1].set_title('Latest Parameters')
    figures['ax3'][0].set_xlabel('iteration')
    figures['ax3'][1].bar(range(prmtr.size), prmtr)
    # figures['ax3'][1].set_xlabel('Latest Parameters')
    plt.draw()
    # plt.show()
    filename = r"S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python" + \
        str(figures['fig3_name']) + ".png"
    figures['fig3'].savefig(filename, format="png")

    plt.close()
    # plt.figure(figures['YvYtt'])
    filename = r"S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python" + \
        str(figures['YvYtt_name']) + ".png"
    # plt.draw()
    figures['YvYtt'].savefig(filename, format="png")

    plt.close()
    # plt.figure(figures['Y_min_Ytt'])
    filename = r"S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python" + \
        str(figures['Y_min_Ytt_name']) + ".png"
    # plt.draw()
    figures['Y_min_Ytt'].savefig(filename, format="png")

    plt.close()
    # plt.figure(figures['XttvThetap'])
    filename = r"S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python" + \
        str(figures['XttvThetap_name']) + ".png"
    figures['XttvThetap'].savefig(filename, format="png")

    doc = subprocess.Popen([mspaint, r"S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python\optim_iter.png"])
    doc = subprocess.Popen([mspaint, r"S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python\YvYtt.png"])
    doc = subprocess.Popen([mspaint, r"S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python\XttvThetap.png"])
    doc = subprocess.Popen([mspaint, r"S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python\Y_min_Ytt.png"])
    Nfeval += 1
    Nfeval_inner += 1
    Nfeval_vec = np.vstack((Nfeval_vec, Nfeval_inner))


print '\n\n\n'
print str_form3.format(*tuple(str_form2))
print 'inner iterations begins__________________________________________'
if stationarity_assumption == 'no':
    optim_output = minimize(neg_cum_log_likelihood, prmtr_0,  method='SLSQP', options={'disp': True}, callback=callbackF)
elif stationarity_assumption == 'yes':
    # optim_output = fmin_slsqp(neg_cum_log_likelihood, prmtr_0, ieqcons=ieconsls, iprint=1, callback=callbackF, full_output=1)
    # optim_output = minimize(neg_cum_log_likelihood, prmtr_0, jac=False, constraints=cons,  method='SLSQP', options={'disp': 1}, callback=callbackF)
    # SLSQP does not work. it takes steps that are too large
    # optim_output = minimize(neg_cum_log_likelihood, prmtr_0, jac=False, method='SLSQP', options={'disp': 1}, callback=callbackF)
    # optim_output = minimize(neg_cum_log_likelihood, prmtr_0, jac=False, constraints=cons,  method='COBYLA', options={'disp': 1}, callback=callbackF)
    # optim_output = minimize(neg_cum_log_likelihood, prmtr_0, jac=False,  method='BFGS', options={'disp': 1}, callback=callbackF)
    # optim_output = minimize(neg_cum_log_likelihood, prmtr_0, jac=False,  method='Newton-CG', options={'disp': 1}, callback=callbackF)
    optim_output = minimize(neg_cum_log_likelihood, prmtr_0, jac=False, method='L-BFGS-B', options={'disp': 1}, callback=callbackF)
    # optim_output = fmin_l_bfgs_b(neg_cum_log_likelihood, prmtr_0,iprint=1, callback=callbackF, approx_grad=1)
    # optim_output = minimize(neg_cum_log_likelihood, prmtr_0, method='Powell', options={'disp': 1}, callback=callbackF)
    # optim_output = minimize(neg_cum_log_likelihood, prmtr_0, jac=False, method='CG', options={'disp': 1}, callback=callbackF)
    # optim_output = minimize(neg_cum_log_likelihood, prmtr_0, jac=False, method='Newton-CG', options={'disp': 1}, callback=callbackF)
    # optim_output = minimize(neg_cum_log_likelihood, prmtr_0, jac=False, method='Nelder-Mead', options={'disp': 1}, callback=callbackF)
    # optim_output = minimize(neg_cum_log_likelihood, prmtr_0, jac=False, method='TNC', options={'disp': 1}, callback=callbackF)
    # optim_output = differential_evolution(neg_cum_log_likelihood, bounds=bnds_lvl,  callback=callbackF)
    # optim_output = minimize(neg_cum_log_likelihood, prmtr_0, jac=False, method='dogleg', options={'disp': 1}, callback=callbackF)
    # trust-ncg does not work need to provide analytical gradient:
    # optim_output = minimize(neg_cum_log_likelihood, prmtr_0, jac=False, method='trust-ncg', options={'disp': 1}, callback=callbackF)


print 'stop here'

if num_states == 4:
    a_new, Kp_new, lmda_new, Phi_new, sigma11_new, sigma22_new, sigma33_new, sigma44_new, Sigma_new, thetap_new = extract_vars(prmtr_ext(optim_output['x']), num_states, US_num_maturities)
elif num_states == 6:
    a_new, Kp_new, lmda_new, lmda2_new, Phi_new, sigma11_new, sigma22_new, sigma22_2_new, sigma33_new, sigma33_2_new, sigma44_new, Sigma_new, thetap_new = extract_vars(prmtr_ext(optim_output['x']), num_states, US_num_maturities)
A0_new, A1_new, U0_new, U1_new, Q_new = extract_mats(prmtr_ext(optim_output['x']), num_states, US_num_maturities, US_nominalmaturities, US_ilbmaturities, dt)
kalman2 = Kalman(Y, A0_new, A1_new, U0_new, U1_new, Q_new, Phi_new, initV)
Ytt_new, Yttl_new, Xtt_new, Xttl_new, Vtt_new, Vttl_new, Gain_t_new, eta_t_new, cum_log_likelihood_new = kalman2.filter()
rho_n = np.mat(np.array([1, 1, 0, 0])).T
rho_r = np.mat(np.array([0, a_new, 0, 1])).T


# Computing Expected Inflation:
horizons = np.array(np.arange(101)).T               #in years
mttau = np.empty((Y.shape[0]+1, num_states+1, horizons.size))
vttau = np.empty((Y.shape[0]+1, (num_states+1)**2, horizons.size))

# v_ode_out = integrate.odeint(v0teqns,np.zeros(((num_states+1)**2,1))[:, 0], horizons, (Kp_new, rho_n, rho_r, Sigma_new, thetap_new) )
v_ode_out = integrate.ode(v0teqns).set_integrator('dopri5', verbosity=1)
v_ode_out.set_initial_value(np.zeros(((num_states+1)**2,1))[:, 0], horizons[0]).set_f_params(Kp_new, rho_n, rho_r, Sigma_new, thetap_new)
t_out, v_out = np.array([0]), np.zeros(((num_states+1)**2,1)).T
# for tinc in np.arange(1,11):
#    v_ode_out.integrate(v_ode_out.t+tinc)
while v_ode_out.successful() and v_ode_out.t < horizons[-1]:
    v_ode_out.integrate(v_ode_out.t+1)
    t_out = np.vstack((t_out, v_ode_out.t))
    v_out = np.vstack((v_out, v_ode_out.y))
plt.close()
fig, ax = plt.subplots(2, sharex=True)
figures['v_ode_out'] = fig
figures['ax_v_ode_out'] = ax
figures['v_ode_out_name'] = '\\v_ode_out'
figures['ax_v_ode_out'][0].plot(np.hstack((v_out[:,0:3],v_out[:,5:8],v_out[:,10:13],v_out[:,15:18])) )
ax[0].set_title('v_{t,t+tau}[0:3,5:8,10:13,15:18]')
figures['ax_v_ode_out'][1].plot(np.hstack((v_out[:,[4]], v_out[:,[9]], v_out[:,[14]])) )
ax[1].set_title('v_{t,t+tau}[4,9,14]')
plt.draw()


for tt in np.arange(0,Y.shape[0]+1):
    m_ode_out = integrate.ode(m0teqns).set_integrator('dopri5', verbosity=1)
    m_ode_out.set_initial_value(np.array(np.hstack((np.vstack((thetap_new.T, Xtt_new)),np.zeros((Xtt_new.shape[0]+1,1))))[tt,:])[0,:],\
                                horizons[0]).set_f_params(Kp_new, rho_n, rho_r, Sigma_new, thetap_new)
    t_out2, m_out = np.array([0]), np.array(np.hstack((np.vstack((thetap_new.T, Xtt_new)),np.zeros((Xtt_new.shape[0]+1,1)))))[tt,:]
    while m_ode_out.successful() and m_ode_out.t < horizons[-1]:
        m_ode_out.integrate(m_ode_out.t+1)
        t_out2 = np.vstack((t_out2, m_ode_out.t))
        m_out = np.vstack((m_out, m_ode_out.y))
    mttau[tt,:,:] = m_out.T
    vttau[tt,:,:] = v_out.T
mttau_nn =  np.mat(mttau[:,-1,:])
vttau_nn =  np.mat(vttau[:,-1,:])
plt.close()
fig, ax = plt.subplots(3, sharex=True)
figures['m_ode_out'] = fig
figures['ax_m_ode_out'] = ax
figures['m_ode_out_name'] = '\\m_ode_out'
figures['ax_m_ode_out'][0].plot(mttau[0,:,0:11].T )
figures['ax_m_ode_out'][0].set_title('m_{0,tau}')
figures['ax_m_ode_out'][1].plot(mttau[29,:,0:11].T )
figures['ax_m_ode_out'][1].set_title('m_{t,t+tau} for t=30')
figures['ax_m_ode_out'][2].plot(mttau[-1,:,0:11].T )
figures['ax_m_ode_out'][2].set_title('m_{T,T+tau}')
plt.draw()


# %now we can compute the expected inflation
exp_inf = -mttau_nn+0.5*vttau_nn
exp_inf = np.array(np.tile(-1.0/horizons.T,(Xtt_new.shape[0]+1,1)))*np.array(exp_inf)
exp_inf[:, 0] = np.array(  ((rho_n-rho_r).T) * (np.mat(thetap_new)) )[0,0]
exp_inf = np.mat(exp_inf)
plt.close()
fig, ax = plt.subplots()
figures['exp_inf'] = fig
figures['ax_exp_inf'] = ax
figures['exp_inf_name'] = '\\exp_inf'
figures['ax_exp_inf'].plot(dates['US_ILB'].astype(DT.datetime), np.hstack((exp_inf[1:,2],exp_inf[1:,5],exp_inf[1:,10])), label=[""] )
handles, labels = figures['ax_exp_inf'].get_legend_handles_labels()
figures['ax_exp_inf'].legend(handles, ["horizon: 2yr", "horizon: 5yr", "horizon: 10yr"] ,loc=2)
figures['ax_exp_inf'].set_title('Inflation Expectations (model-implied)')
plt.draw()
plt.close()
fig, ax = plt.subplots()
figures['exp_inf_2'] = fig
figures['ax_exp_inf_2'] = ax
figures['exp_inf_2_name'] = '\\exp_inf_2'
figures['ax_exp_inf_2'].plot(exp_inf[:,0:11] )
plt.draw()
plt.close()
fig, ax = plt.subplots()
figures['exp_inf_3'] = fig
figures['ax_exp_inf_3'] = ax
figures['exp_inf_3_name'] = '\\exp_inf_3'
figures['ax_exp_inf_3'].plot(np.hstack((exp_inf[:,1],exp_inf[:,2])) )
plt.draw()

# %now we can compute the break-evens
bk_mats = np.unique(np.vstack((US_nominalmaturities, US_ilbmaturities)))
# bk_evens =[USnominals[np.in1d(US_nominalmaturities, m)][0].getZeroYieldsTS - USilbs[np.in1d(US_ilbmaturities, m)][0].getZeroYieldsTS for m in bk_mats]
bk_evens = (np.array([Y[:,np.in1d(US_nominalmaturities, m)] - (Y[:,US_nominalmaturities.size:])[:,np.in1d(US_ilbmaturities, m)] for m in bk_mats])[:,:,0]).T
plt.close()
fig, ax = plt.subplots()
figures['bk_evens'] = fig
figures['ax_bk_evens'] = ax
figures['bk_evens_name'] = '\\bk_evens'
figures['ax_bk_evens'].plot(dates['US_ILB'].astype(DT.datetime), np.hstack((bk_evens[:,np.in1d(bk_mats, 2)],bk_evens[:,np.in1d(bk_mats, 5)],bk_evens[:,np.in1d(bk_mats, 10)])), label=[""] )
handles, labels = figures['ax_bk_evens'].get_legend_handles_labels()
figures['ax_bk_evens'].legend(handles, ["2yr", "5yr", "10yr"] ,loc=2)
figures['ax_bk_evens'].set_title('Break-Evens')
plt.draw()
plt.close()
fig, ax = plt.subplots()
figures['bk_evens_2'] = fig
figures['ax_bk_evens_2'] = ax
figures['bk_evens_2_name'] = '\\bk_evens_2'
figures['ax_bk_evens_2'].plot(bk_evens[:,:] )
plt.draw()


# %now we can compute the IRPs
irps = bk_evens - exp_inf[1:,bk_mats]
plt.close()
fig, ax = plt.subplots()
figures['irps'] = fig
figures['ax_irps'] = ax
figures['irps_name'] = '\\irps'
figures['ax_irps'].plot(dates['US_ILB'].astype(DT.datetime), np.hstack((irps[:,np.in1d(bk_mats, 2)],irps[:,np.in1d(bk_mats, 5)],irps[:,np.in1d(bk_mats, 10)])), label=[""] )
handles, labels = figures['ax_irps'].get_legend_handles_labels()
figures['ax_irps'].legend(handles, ["2yr", "5yr", "10yr"] ,loc=2)
figures['ax_irps'].set_title('Inflation Risk Premia (IRP)')
plt.draw()
plt.close()
fig, ax = plt.subplots()
figures['irps_2'] = fig
figures['ax_irps_2'] = ax
figures['irps_2_name'] = '\\irps_2'
figures['ax_irps_2'].plot(irps[:,:] )
plt.draw()


# lastly we compute the deflation probabilities
prob_def = -np.array(mttau_nn)/(np.array(vttau_nn)**0.5)
prob_def = norm.cdf(prob_def)
prob_def[:, 0] = 0
prob_def = np.mat(prob_def)
plt.close()
fig, ax = plt.subplots()
figures['prob_def'] = fig
figures['ax_prob_def'] = ax
figures['prob_def_name'] = '\\prob_def'
figures['ax_prob_def'].plot(dates['US_ILB'].astype(DT.datetime),np.hstack((prob_def[1:,2],prob_def[1:,5],prob_def[1:,10])), label=[""] )
handles, labels = figures['ax_prob_def'].get_legend_handles_labels()
figures['ax_prob_def'].legend(handles, ["horizon: 2yr", "horizon: 5yr", "horizon: 10yr"] ,loc=2)
figures['ax_prob_def'].set_title("Probability of Deflation (model-implied)")
plt.draw()
plt.close()
fig, ax = plt.subplots()
figures['prob_def_2'] = fig
figures['ax_prob_def_2'] = ax
figures['prob_def_2_name'] = '\\prob_def_2'
figures['ax_prob_def_2'].plot(prob_def[:,0:10] )
plt.draw()
plt.close()
fig, ax = plt.subplots()
figures['prob_def_3'] = fig
figures['ax_prob_def_3'] = ax
figures['prob_def_3_name'] = '\\prob_def_3'
figures['ax_prob_def_3'].plot(np.hstack((prob_def[:,1],prob_def[:,2])) )
plt.draw()


# recording output
datadate = np.array([dates['US_ILB'].astype(DT.datetime)[tt,0].year*10000+dates['US_ILB'].astype(DT.datetime)[tt,0].month*100+dates['US_ILB'].astype(DT.datetime)[tt,0].day for tt in np.arange(0,dates['US_ILB'].shape[0])])
datadate = np.hstack((0,datadate))
np.savetxt(r'S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python\prob_def_'+estim_freq+'_'+str(datadate[1])+'_'+time.strftime("%Y%m%d")+'.csv',\
            np.array(np.hstack((np.mat(datadate).T,prob_def))), delimiter=',')
np.savetxt(r'S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python\irps_'+estim_freq+'_'+str(datadate[1])+'_'+time.strftime("%Y%m%d")+'.csv',\
            np.array(np.hstack((np.mat(datadate[1:]).T,irps))), delimiter=',')
np.savetxt(r'S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python\bk_evens_'+estim_freq+'_'+str(datadate[1])+'_'+time.strftime("%Y%m%d")+'.csv',\
            np.array(np.hstack((np.mat(datadate[1:]).T,bk_evens))), delimiter=',')
np.savetxt(r'S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python\exp_inf_'+estim_freq+'_'+str(datadate[1])+'_'+time.strftime("%Y%m%d")+'.csv',\
            np.array(np.hstack((np.mat(datadate).T,exp_inf))), delimiter=',')
np.savetxt(r'S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python\Y_'+estim_freq+'_'+str(datadate[1])+'_'+time.strftime("%Y%m%d")+'.csv',\
            np.array(np.hstack((np.mat(datadate[1:]).T,Y))), delimiter=',')
np.savetxt(r'S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python\Y_'+estim_freq+'_'+str(datadate[1])+'_'+time.strftime("%Y%m%d")+'.csv',\
            np.array(np.hstack((np.mat(datadate[1:]).T,Ytt_new))), delimiter=',')
np.savetxt(r'S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python\Y_'+estim_freq+'_'+str(datadate[1])+'_'+time.strftime("%Y%m%d")+'.csv',\
            np.array(np.hstack((np.mat(datadate[1:]).T,Xtt_new))), delimiter=',')

output_filename=r'S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python\estimation_output_'+estim_freq+'_'+str(datadate[1])+'_'+time.strftime("%Y%m%d")+'.out'
my_shelf = shelve.open(output_filename,'n') # 'n' for new
for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()

end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))