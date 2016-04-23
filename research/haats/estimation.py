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
print(sys.argv)
from irp_obj import *
from kalman import *
from extract_parameters import *
from math import exp
from scipy.optimize import minimize, fmin_slsqp
from scipy.linalg import expm
from scipy import integrate
from estim_constraints import *
from scipy.stats import norm
from io import open


class Estimation:

    def __init__(self):
       class_name = self.__class__.__name__
        # print(class_name, "constructed")

    def __del__(self):
        class_name = self.__class__.__name__
        # print(class_name, "destroyed")


class Rolling(Estimation):

    def __init__(self, data, US_ilbmaturities, US_nominalmaturities, \
            estim_freq='daily', num_states=4, fix_Phi=1, setdiag_Kp=1, initV='unconditional', stationarity_assumption='yes'):
        Estimation.__init__(self)

        US_num_maturities = len(US_ilbmaturities) + len(US_nominalmaturities)

        # frequency of data for estimation:
        if estim_freq == 'daily':
            dt = 1.0/252  # daily increment
        elif estim_freq == 'weekly':
            dt = 1.0/52     # weekly increment
        elif estim_freq == 'monthly':
            dt = 1.0/12     # monthly increment
        elif estim_freq == 'quarterly':
            dt = 1.0/4      # quarterly increment

        if ((initV == 'steady_state') | (initV == 'unconditional')):
            stationarity_assumption = 'yes'
        else:
            stationarity_assumption = 'no'

        # creating ILBs and Nominal Bond objects
        USilbs = np.array([InfLinkBonds(m, 'USA') for m in US_ilbmaturities])
        USnominals = np.array([NominalBonds(m, 'USA') for m in US_nominalmaturities])

        # storing yields time series in bond objects:
        # in Python, attributes are public: no need for get, set methods
        for m in range(US_ilbmaturities.size):
            USilbs[m].setZeroYieldsTS(data['US_ILB'][[m]]/100)
            USilbs[m].setZeroYieldsDates(data['US_ILB'].index)

        for m in range(US_nominalmaturities.size):
            USnominals[m].setZeroYieldsTS(data['US_NB'][[m]]/100)
            USnominals[m].setZeroYieldsDates(data['US_NB'].index)

        # Stacking yields data:
        Y = np.mat(np.hstack((data['US_NB'].values, data['US_ILB'].values)))/100

        # See haats_documentation.pdf and haats_documentation.lyx
        # initializing parameters
        # using numbers from Christensen, Diebold, Rudebusch (2010)
        a = 0.677
        # Kp = np.array([1.305, 0, 0, -1.613, 1.559, 0.828, -1.044, 0, 0, 0, 0.884, 0, -1.531, -0.364, 0, 1.645])     # reshape(num_states, num_states)
        Kp = np.reshape(np.identity(num_states),(num_states**2))
        lmda = 0.5319
        sigmas = np.array([0.0047, 0.00756, 0.02926, 0.00413])
        thetap = np.array([0.06317, -0.1991, -0.00969, 0.03455])
        if num_states == 6:
            Kp = np.reshape(np.identity(num_states),(num_states**2))
            lmda2 = 0.5319/2
            sigmas = np.array([0.0047, 0.00756, 0.00756, 0.02926, 0.02926, 0.00413])
            thetap = np.array([0.06317, -0.1991, -0.1991, -0.00969, -0.00969, 0.03455])
        Phi_prmtr = np.log(np.diag(np.cov(Y.T)))   # estimate Phi
        if fix_Phi == 0:
            prmtr_0 = np.random.rand(1 + num_states**2 + 1 + num_states + US_num_maturities + num_states)
            if num_states == 6:
                prmtr_0 = np.random.rand(1 + num_states**2 + 1 + 1 + num_states + US_num_maturities + num_states)
            # repopulate initial parameter vector:
            prmtr_0 = np.array([a])
            if setdiag_Kp==1:
                prmtr_0 = np.append(prmtr_0, np.diag(np.reshape(Kp,(num_states,num_states))))
            else:
                prmtr_0 = np.append(prmtr_0, Kp)
            prmtr_0 = np.append(prmtr_0, np.log(lmda ))  #  to impose non-negativity
            if num_states == 6:
                    prmtr_0 = np.append(prmtr_0, np.log(lmda2 ))  #  to impose non-negativity
            prmtr_0 = np.append(prmtr_0, Phi_prmtr)  #  to impose non-negativity
            prmtr_0 = np.append(prmtr_0, np.log(sigmas )) #  to impose non-negativity
            prmtr_0 = np.append(prmtr_0, thetap)
        else:
            prmtr_0 = np.random.rand(1 + num_states**2 + 1 + num_states + num_states)
            if num_states == 6:
                prmtr_0 = np.random.rand(1 + num_states**2 + 1 + 1 + num_states + num_states)
            Phi_prmtr = np.log(np.diag(np.cov(Y.T)))  #  to impose non-negativity  # estimate Phi
            # populate initial parameter vector:
            prmtr_0 = np.array([a])
            if setdiag_Kp==1:
                prmtr_0 = np.append(prmtr_0, np.diag(np.reshape(Kp,(num_states,num_states))))
            else:
                prmtr_0 = np.append(prmtr_0, Kp)
            prmtr_0 = np.append(prmtr_0, np.log(lmda ))  #  to impose non-negativity
            if num_states ==6:
                prmtr_0 = np.append(prmtr_0, np.log(lmda2))  #  to impose non-negativity
            prmtr_0 = np.append(prmtr_0, np.log(sigmas )) #  to impose non-negativity
            prmtr_0 = np.append(prmtr_0, thetap)

        def prmtr_ext(prmtr):   #function to insert parameters for Phi
            return prmtr_ext0(prmtr, num_states, Phi_prmtr, fix_Phi, setdiag_Kp)

        # this is to display the results in the console
        Nfeval = 1
        self.Nfeval_inner = 1
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

        cons = ineq_cons(num_states, US_num_maturities, fix_Phi, setdiag_Kp, Phi_prmtr)
        ieconsls = ineq_cons_ls(num_states, US_num_maturities, fix_Phi, setdiag_Kp, Phi_prmtr)
        bnds_lvl = prmtr_bounds_lvl(num_states, US_num_maturities)
        bnds_exp = prmtr_bounds_exp(num_states, US_num_maturities)

        # refining the initial guess
        if 0:
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
                    print('initial guess is no good')
                    break
                prmtr_0[-num_states:] = np.array(np.mean(Xtt_ref[5:,:],0).T)[:,0]

        # objective function:
        def neg_cum_log_likelihood(prmtr):
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
                    plt.close("all")
                except:
                    cum_log_likelihood = -np.inf
            out_n = np.hstack((self.Nfeval_inner, prmtr, cum_log_likelihood))
            print(str_form.format(*tuple(out_n)))    # use * to unpack tuple
            self.Nfeval_inner += 1
            return (-1) * cum_log_likelihood    # we will minimize the negative of the likelihood

        print('\n\n\n')
        print(str_form3.format(*tuple(str_form2)))
        print('new iterations begins__________________________________________')
        if stationarity_assumption == 'no':
            optim_output = minimize(neg_cum_log_likelihood, prmtr_0,  method='Powell', options={'disp': 1})
        elif stationarity_assumption == 'yes':
            optim_output = minimize(neg_cum_log_likelihood, prmtr_0, method='Powell', options={'disp': 1})

        prmtr_new = optim_output['x']

        if num_states == 4:
            a_new, Kp_new, lmda_new, Phi_new, sigma11_new, sigma22_new, sigma33_new, sigma44_new, Sigma_new, thetap_new = extract_vars(prmtr_ext(optim_output['x']), num_states, US_num_maturities)
        elif num_states == 6:
            a_new, Kp_new, lmda_new, lmda2_new, Phi_new, sigma11_new, sigma22_new, sigma22_2_new, sigma33_new, sigma33_2_new, sigma44_new, Sigma_new, thetap_new = extract_vars(prmtr_ext(optim_output['x']), num_states, US_num_maturities)
        A0_new, A1_new, U0_new, U1_new, Q_new = extract_mats(prmtr_ext(optim_output['x']), num_states, US_num_maturities, US_nominalmaturities, US_ilbmaturities, dt)
        kalman2 = Kalman(Y, A0_new, A1_new, U0_new, U1_new, Q_new, Phi_new, initV)
        Ytt_new, Yttl_new, Xtt_new, Xttl_new, Vtt_new, Vttl_new, Gain_t_new, eta_t_new, cum_log_likelihood_new = kalman2.filter()

        yields_forecast = kalman2.forecast(Xtt_new, 90*(estim_freq=='daily')+12*(estim_freq=='weekly')+3*(estim_freq=='monthly'))
        for m in range(US_nominalmaturities.size):
            USnominals[m].yields_forecast = yields_forecast[:, m, :]
            USnominals[m].forecast_e, USnominals[m].forecast_se, USnominals[m].forecast_rmse = kalman2.rmse(USnominals[m].yields_forecast, m)

        for m in range(US_ilbmaturities.size):
            USilbs[m].yields_forecast = yields_forecast[:, US_nominalmaturities.size + m - 1, :]
            USilbs[m].forecast_e, USilbs[m].forecast_se, USilbs[m].forecast_rmse = kalman2.rmse(USilbs[m].yields_forecast, m)

        if num_states == 4:
            rho_n = np.mat(np.array([1, 1, 0, 0])).T
            rho_r = np.mat(np.array([0, a_new, 0, 1])).T
        elif num_states == 6:
            rho_n = np.mat(np.array([1, 1, 1, 0, 0, 0])).T
            rho_r = np.mat(np.array([0, a_new, a_new, 0, 0, 1])).T

        # keep only the last 10 observations to make the code run faster
        #Y, Ytt_new, Yttl_new, Xtt_new, Xttl_new, Vtt_new, Vttl_new, Gain_t_new, eta_t_new =\
        #    Y[-10:, :], Ytt_new[-10:, :], Yttl_new[-10:, :], Xtt_new[-10:, :], Xttl_new[-10:, :], Vtt_new[-10:, :], Vttl_new[-10:, :], Gain_t_new[-10:, :], eta_t_new[-10:, :]

        # Computing Expected Inflation:
        horizons = np.array(np.arange(100).T)               #in years
        mttau = np.empty((Y.shape[0], num_states+1, horizons.size))
        vttau = np.empty((Y.shape[0], (num_states+1)**2, horizons.size))

        v_ode_out = integrate.ode(v0teqns).set_integrator('dopri5', verbosity=1)
        v_ode_out.set_initial_value(np.zeros(((num_states+1)**2,1))[:, 0], horizons[0]).set_f_params(Kp_new, rho_n, rho_r, Sigma_new, thetap_new)
        t_out, v_out = np.array([0]), np.zeros(((num_states+1)**2,1)).T
        while v_ode_out.successful() and v_ode_out.t < horizons[-1]:
            v_ode_out.integrate(v_ode_out.t+1)
            t_out = np.vstack((t_out, v_ode_out.t))
            v_out = np.vstack((v_out, v_ode_out.y))

        for tt in np.arange(0,Y.shape[0]):
            m_ode_out = integrate.ode(m0teqns).set_integrator('dopri5', verbosity=1)
            m_ode_out.set_initial_value(np.array(np.hstack((Xtt_new,np.zeros((Xtt_new.shape[0],1))))[tt,:])[0,:],\
                                        horizons[0]).set_f_params(Kp_new, rho_n, rho_r, Sigma_new, thetap_new)
            t_out2, m_out = np.array([0]), np.array(np.hstack((Xtt_new,np.zeros((Xtt_new.shape[0],1)))))[tt,:]
            while m_ode_out.successful() and m_ode_out.t < horizons[-1]:
                m_ode_out.integrate(m_ode_out.t+1)
                t_out2 = np.vstack((t_out2, m_ode_out.t))
                m_out = np.vstack((m_out, m_ode_out.y))
            mttau[tt,:,:] = m_out.T
            vttau[tt,:,:] = v_out.T
        mttau_nn =  np.mat(mttau[:,-1,:])
        vttau_nn =  np.mat(vttau[:,-1,:])

        # %now we can compute the expected inflation
        exp_inf = -mttau_nn+0.5*vttau_nn
        exp_inf = np.array(np.tile(-1.0/horizons.T,(Xtt_new.shape[0],1)))*np.array(exp_inf)
        exp_inf[:, 0] = np.array(  ((rho_n-rho_r).T) * (np.mat(thetap_new)) )[0,0]
        exp_inf = np.mat(exp_inf)

        # %now we can compute the break-evens
        bk_mats = np.unique(np.vstack((US_nominalmaturities, US_ilbmaturities)))
        bk_evens = (np.array([Y[:,np.in1d(US_nominalmaturities, m)] - (Y[:,US_nominalmaturities.size:])[:,np.in1d(US_ilbmaturities, m)] for m in bk_mats])[:,:,0]).T

        # %now we can compute the IRPs
        irps = bk_evens - exp_inf[:,bk_mats-1]

        # lastly we compute the deflation probabilities
        prob_def = -np.array(mttau_nn)/(np.array(vttau_nn)**0.5)
        prob_def = norm.cdf(prob_def)
        prob_def[:, 0] = 0
        prob_def = np.mat(prob_def)

        # recording output
        datadate = np.array([np.union1d(dates['US_NB'], dates['US_ILB']).astype(DT.datetime)[tt].year*10000+np.union1d(dates['US_NB'], dates['US_ILB']).astype(DT.datetime)[tt].month*100+np.union1d(dates['US_NB'], dates['US_ILB']).astype(DT.datetime)[tt].day for tt in np.arange(0,np.union1d(dates['US_NB'], dates['US_ILB']).shape[0])])
        datadate = np.hstack((0, datadate))

        if num_states ==4:
            lmda2_new=0
        # TODO: NEED TO FIX THIS CODE BELOW FOR SAVING THE OUTPUT AS TXT FILES. IT IS CURRENTLY NOT STARTING NEW LINE AFTER THE HEADERS.
        # creating txt files to record latest observation of each time series of key results
        my_matrix = np.array(['prob_def','irps','bk_evens','exp_inf','Y','Ytt_new','Yttl_new','Xtt_new','Xttl_new','Vtt_new','Vttl_new','prmtr_new','Kp_new','thetap_new','Sigma_new','a_new','lmda_new','lmda2_new','Phi_new','Q_new','U0_new','U1_new','A0_new','A1_new','V0','X0'])
        if os.path.isfile(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files\prob_def_'+estim_freq+'.txt')==False:  # if files do not exist, creat new ones with appropriate headers
            header_prob_def, header_exp_inf = (tuple(['dates'])+tuple([',horizon: '+str(vv)+'yr' for vv in horizons])+tuple(['\n']) for vvv in range(2))
            header_Y, header_Ytt_new,  header_Yttl_new = (tuple(['dates'])+tuple([',Nominal: '+str(US_nominalmaturities[vv])+'yr' for vv in range(US_nominalmaturities.size)])\
                                                 + tuple([',ILB: '+str(US_ilbmaturities[vv])+'yr' for vv in range(US_ilbmaturities.size)])+tuple(['\n']) for vvv in range(3))
            header_irps, header_bk_evens = (tuple(['dates'])+tuple([','+str(vv)+'yr' for vv in bk_mats])+tuple(['\n']) for vvv in range(2))
            if num_states==4:
                header_Xtt_new,  header_Xttl_new, header_thetap_new, header_X0  = (tuple(['dates,Ln,S,C,Lr'])+tuple(['\n']) for vvv in range(4))
            if num_states==6:
                header_Xtt_new,  header_Xttl_new, header_thetap_new, header_X0  = (tuple(['dates,Ln,S1,S2,C1,C2,Lr'])+tuple(['\n']) for vvv in range(4))
            header_Vtt_new,  header_Vttl_new, header_V0 = (((tuple(['dates'])+tuple(np.reshape([[',V('+str(vv)+';'+str(vv2)+')'  for vv in range(num_states)] for vv2 in range(num_states)],(num_states**2)))+tuple(['\n']) )) for vvv in range(3))
            header_prmtr_new = tuple(['dates'])+tuple([',prmtr('+str(vv)+')' for vv in range(prmtr_new.size)])+tuple(['\n'])
            header_Kp_new = tuple(['dates'])+tuple(np.reshape([[',Kp('+str(vv)+';'+str(vv2)+')'  for vv in range(num_states)] for vv2 in range(num_states)],(num_states**2)))+tuple(['\n'])
            header_Sigma_new = tuple(['dates'])+tuple(np.reshape([[',Sigma('+str(vv)+';'+str(vv2)+')'  for vv in range(num_states)] for vv2 in range(num_states)],(num_states**2)))+tuple(['\n'])
            header_a_new, header_lmda_new, header_lmda2_new  = tuple(['dates,a'])+tuple(['\n']), tuple(['dates,lmda'])+tuple(['\n']), tuple(['dates,lmda2'])+tuple(['\n'])
            header_Phi_new = tuple(['dates'])+tuple(np.reshape([[',Sigma('+str(vv)+';'+str(vv2)+')'  for vv in range(US_num_maturities)] for vv2 in range(US_num_maturities)],(US_num_maturities**2)))+tuple(['\n'])
            header_Q_new = tuple(['dates'])+tuple(np.reshape([[',Q('+str(vv)+';'+str(vv2)+')'  for vv in range(num_states)] for vv2 in range(num_states)],(num_states**2)))+tuple(['\n'])
            header_U0_new = tuple(['dates'])+tuple([',U0('+str(vv)+')' for vv in range(num_states)])+tuple(['\n'])
            header_U1_new = tuple(['dates'])+tuple(np.reshape([[',U1('+str(vv)+';'+str(vv2)+')'  for vv in range(num_states)] for vv2 in range(num_states)],(num_states**2)))+tuple(['\n'])
            header_A0_new = tuple(['dates'])+tuple([',A0('+str(vv)+')' for vv in range(US_num_maturities)])+tuple(['\n'])
            header_A1_new = tuple(['dates'])+tuple(np.reshape([[',A1('+str(vv2)+';'+str(vv)+')'  for vv in range(num_states )] for vv2 in range(US_num_maturities)],(num_states*US_num_maturities)))+tuple(['\n'])
            header_US_nominal_yields_forecast, header_US_ilb_yields_forecast = tuple(['dates'])+tuple([','+str(vv)+' period' for vv in range(yields_forecast.shape[2])])+tuple(['\n']), tuple(['dates'])+tuple([','+str(vv)+' period' for vv in range(yields_forecast.shape[2])])+tuple(['\n'])
            header_US_nominal_forecast_e, header_US_ilb_forecast_e = tuple(['dates'])+tuple([','+str(vv)+' period' for vv in range(yields_forecast.shape[2])])+tuple(['\n']), tuple(['dates'])+tuple([','+str(vv)+' period' for vv in range(yields_forecast.shape[2])])+tuple(['\n'])
            header_US_nominal_forecast_se, header_US_ilb_forecast_se = tuple(['dates'])+tuple([','+str(vv)+' period' for vv in range(yields_forecast.shape[2])])+tuple(['\n']), tuple(['dates'])+tuple([','+str(vv)+' period' for vv in range(yields_forecast.shape[2])])+tuple(['\n'])
            header_US_nominal_forecast_rmse, header_US_ilb_forecast_rmse = (tuple(['dates'])+tuple([',rmse'])+tuple(['\n']) for vvv in range(2))
            for f in np.arange(my_matrix.size):
                try: #python 2
                    f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files'+'\\'+my_matrix[f] +'_'+estim_freq+'.txt', 'w' )
                    f_handle.write(np.array(eval('header_'+my_matrix[f])))
                except: #python 3
                    f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files'+'\\'+my_matrix[f] +'_'+estim_freq+'.txt', 'wb' )
                    np.savetxt(f_handle, np.array(eval('header_'+my_matrix[f])), fmt='%s', delimiter=',')
                f_handle.close()
            for nn in ['nominal','ilb']:
                for m in range(eval('US_'+nn+'maturities.size')):
                    for ss in ['yields_forecast','forecast_e' ,'forecast_se','forecast_rmse' ]:
                        try: #python 2
                            f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files\US'+nn+'s_'+ss+'_'+str(eval('US_'+nn+'maturities[m]'))+'yrMat_'+estim_freq+'.txt','w')
                            f_handle.write(np.array(eval('header_US_'+nn+'_'+ss)))
                        except: #python 3
                            f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files\US'+nn+'s_'+ss+'_'+str(eval('US_'+nn+'maturities[m]'))+'yrMat_'+estim_freq+'.txt','wb')
                            np.savetxt(f_handle, np.array(eval('header_US_'+nn+'_'+ss)), fmt='%s', delimiter=',')
                        f_handle.close()
        # append latest observation of each time series of key results:
        my_matrix = np.array(['prob_def','irps','bk_evens','exp_inf','Y','Ytt_new','Yttl_new','Xtt_new','Xttl_new','Vtt_new','Vttl_new'])
        for f in np.arange(my_matrix.size):
            try: #python 2
                f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files'+'\\'+my_matrix[f] +'_'+estim_freq+'.txt' , 'a')
                np.savetxt(f_handle, np.mat(np.hstack((datadate[-1], np.reshape(np.array(eval(my_matrix[f]))[-1],(np.array(eval(my_matrix[f]))[-1].size))))), delimiter=',')
            except: #python 3
                f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files'+'\\'+my_matrix[f] +'_'+estim_freq+'.txt' ,'ab')
                np.savetxt(f_handle, np.mat(np.hstack((datadate[-1], np.reshape(np.array(eval(my_matrix[f]))[-1],(np.array(eval(my_matrix[f]))[-1].size))))), delimiter=',')
            f_handle.close()

        my_matrix = np.array(['prmtr_new','Kp_new','thetap_new','Sigma_new','a_new','lmda_new','lmda2_new','Phi_new','Q_new','U0_new','U1_new','A0_new','A1_new','V0','X0'])
        for f in np.arange(my_matrix.size):
            if (my_matrix[f] == 'V0') | (my_matrix[f]== 'X0'):
                try: #python 2
                    f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files'+'\\'+my_matrix[f] +'_'+estim_freq+'.txt' , 'a')
                    np.savetxt(f_handle, np.mat(np.hstack((datadate[-1], np.reshape(np.array(eval('kalman2.'+my_matrix[f])),(np.array(eval('kalman2.'+my_matrix[f])).size))))), delimiter=',')
                except: #python 3
                    f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files'+'\\'+my_matrix[f] +'_'+estim_freq+'.txt' , 'ab')
                    np.savetxt(f_handle, np.mat(np.hstack((datadate[-1], np.reshape(np.array(eval('kalman2.'+my_matrix[f])),(np.array(eval('kalman2.'+my_matrix[f])).size))))), delimiter=',')
                f_handle.close()
            else:
                try: #python 2
                    f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files'+'\\'+my_matrix[f] +'_'+estim_freq+'.txt' , 'a')
                    np.savetxt(f_handle, np.mat(np.hstack((datadate[-1], np.reshape(np.array(eval(my_matrix[f])),(np.array(eval(my_matrix[f])).size))))), delimiter=',')
                except: #python 3
                    f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files'+'\\'+my_matrix[f] +'_'+estim_freq+'.txt' , 'ab')
                    np.savetxt(f_handle, np.mat(np.hstack((datadate[-1], np.reshape(np.array(eval(my_matrix[f])),(np.array(eval(my_matrix[f])).size))))), delimiter=',')
                f_handle.close()

        for nn in ['nominal','ilb']:
            for m in range(eval('US_'+nn+'maturities.size')):
                for ss in ['yields_forecast','forecast_e' ,'forecast_se','forecast_rmse' ]:
                    try: #python 2
                        f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files\US'+nn+'s_'+ss+'_'+str(eval('US_'+nn+'maturities[m]'))+'yrMat_'+estim_freq+'.txt','a')
                        np.savetxt(f_handle, np.mat(np.hstack((datadate[-1], np.reshape(np.array(eval('US'+nn+'s[m].'+ss))[-1],(np.array(eval('US'+nn+'s[m].'+ss))[-1].size))))), delimiter=',')
                    except: #python 3
                        f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files\US'+nn+'s_'+ss+'_'+str(eval('US_'+nn+'maturities[m]'))+'yrMat_'+estim_freq+'.txt','ab')
                        np.savetxt(f_handle, np.mat(np.hstack((datadate[-1], np.reshape(np.array(eval('US'+nn+'s[m].'+ss))[-1],(np.array(eval('US'+nn+'s[m].'+ss))[-1].size))))), delimiter=',')
                    f_handle.close()

