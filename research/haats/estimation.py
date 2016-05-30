from __future__ import division
from asset_class import *
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
import itertools
from IPython.core.debugger import Tracer; debug_here = Tracer() #this is the approach that works for ipython debugging
import pdb
from math import exp
from scipy.optimize import minimize, fmin_slsqp
from scipy.linalg import expm
from scipy import integrate
from scipy.stats import norm
from scipy.stats import multivariate_normal
from io import open
import re
from kalman import *
from extract_parameters import *
from estim_constraints import *
import pymc as pymc2
import pymc3 as pymc3
import corner as triangle_plot
import collections

class Estimation:

    def __init__(self):
       class_name = self.__class__.__name__
        # print(class_name, "constructed")

    def __del__(self):
        class_name = self.__class__.__name__
        # print(class_name, "destroyed")


class Rolling(Estimation):

    def __init__(self):
        Estimation.__init__(self)

    def run_setup(self, data, US_ilbmaturities, US_nominalmaturities, \
            estim_freq='daily', num_states=4, fix_Phi=1, setdiag_Kp=1, initV='unconditional', stationarity_assumption='yes'):
        '''Set up initial guess parameters and other ingredients needed for estimation'''
        US_num_maturities = len(US_ilbmaturities) + len(US_nominalmaturities)

        ########################################################
        #Get and store data
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
            USilbs[m].setZeroYieldsTS(data['US_ILB'][[m]]/100.)
            USilbs[m].setZeroYieldsDates(data['US_ILB'].index)

        for m in range(US_nominalmaturities.size):
            USnominals[m].setZeroYieldsTS(data['US_NB'][[m]]/100.)
            USnominals[m].setZeroYieldsDates(data['US_NB'].index)
        # Horizontal stacking of yields data:
        Y = pd.concat([data['US_NB'], data['US_ILB']], axis=1)/100.
        self.Y = Y
        ########################################################

        # debug_here()
        ########################################################
        #Initialize parameters
        prmtr_dict = {} #dictionary to save parameters
        prmtr_size_dict = {} #dictionary to save parameters sizes

        # See haats_documentation.pdf and haats_documentation.lyx
        # initializing parameters
        # using some of the numbers from Christensen, Diebold, Rudebusch (2010) to initialize some parameters
        a = [0.677]
        Kp = np.reshape(np.identity(num_states),(num_states**2))
        lmda = [0.5319]
        sigmas = [0.0047, 0.00756, 0.02926, 0.00413]
        thetap = [0.06317, -0.1991, -0.00969, 0.03455]

        if num_states == 6:
            statevar_names = ['LN','S1','S2','C1','C2','LR']
            lmda2 = [0.5319/2]
            sigmas = [0.0047, 0.00756, 0.00756, 0.02926, 0.02926, 0.00413]
            thetap = [0.06317, -0.1991, -0.1991, -0.00969, -0.00969, 0.03455]
        else:
            statevar_names = ['LN','S','C','LR']

        Phi_prmtr = np.log(np.diag(np.cov(Y.values.T)))   # estimate Phi
        if fix_Phi == 0:
            prmtr_dict['Phi'] =  Phi_prmtr.tolist()  #  here we have to estimate the Phi

        prmtr_dict['a'] = a
        prmtr_dict['lmda'] = np.log(lmda ).tolist()  #  to impose non-negativity

        if setdiag_Kp == 1:
            prmtr_dict['Kp'] = np.diag(np.reshape(Kp, (num_states, num_states))).tolist() #here Kp is diagonal
        else:
            prmtr_dict['Kp'] = Kp.tolist()

        if num_states == 6:
            prmtr_dict['lmda2'] = np.log(lmda2).tolist()  # to impose non-negativity

        prmtr_dict['sigmas'] = np.log(sigmas)  # to impose non-negativity
        prmtr_dict['thetap'] = thetap

        # debug_here()

        for k in prmtr_dict.keys():
            prmtr_size_dict[k] = len(prmtr_dict[k])

        prmtr_dict = collections.OrderedDict(sorted(prmtr_dict.items()))
        prmtr_size_dict = collections.OrderedDict(sorted(prmtr_size_dict.items()))
        prmtr_0 = np.array(  list(itertools.chain.from_iterable(prmtr_dict.values()))  )#notice that the the values are sorted according the alphabetic order of the keys

        # debug_here()

        # this is to display the results in the console
        self.Nfeval_inner = 1

        # These next four lines are constraints we could use in the optimization.
        self.cons = ineq_cons(num_states, prmtr_size_dict, Phi_prmtr)
        # self.ieconsls = ineq_cons_ls(num_states, prmtr_size_dict, Phi_prmtr)
        # self.bnds_lvl = prmtr_bounds_lvl(num_states, prmtr_size_dict, Phi_prmtr)
        # self.bnds_exp = prmtr_bounds_exp(num_states, prmtr_size_dict, Phi_prmtr)
        # debug_here()

        self.num_states, self.US_num_maturities, self.US_nominalmaturities, self.US_ilbmaturities, self.dt, self.prmtr_size_dict = \
            num_states, US_num_maturities, US_nominalmaturities, US_ilbmaturities, dt, prmtr_size_dict
        self.Phi_prmtr, self.stationarity_assumption, self.initV = Phi_prmtr, stationarity_assumption, initV
        self.prmtr_0 = prmtr_0
        self.USilbs, self.USnominals = USilbs, USnominals
        self.statevar_names = statevar_names
        self.estim_freq = estim_freq


    def fit(self, estimation_method='em_mle', tolerance=1e-6, maxiter=50 , toltype='max_abs', \
            solver_mle='Nelder-Mead',maxiter_mle=1000, maxfev_mle=1000, ftol_mle=1e-6, xtol_mle=1e-6, constraints_mle='off', \
            priors_bayesian=None, maxiter_bayesian=1000, burnin_bayesian=None  ):
        '''Running Estimation Fit'''

        print('new fit begins__________________________________________')
        tic = time.clock()

        tol, prmtr_new_raw, self.Nfeval = np.inf, self.prmtr_0, 0
        #We map parameters back to un-transformed parameters before saving results
        prmtr_new = param_mapping(self.num_states, build_prmtr_dict(prmtr_new_raw, self.prmtr_size_dict))

        #Let us record the path of the parameters and objective as we iterate:
        fit_path_cols = [[k+'['+str(ki) +']' for ki in range(self.prmtr_size_dict[k])] for k in self.prmtr_size_dict.keys()]
        fit_path_cols = np.array(  list(itertools.chain.from_iterable(fit_path_cols))  )
        fit_path_inner_cols = np.hstack((['sub_objective'], fit_path_cols))
        fit_path_cols = np.hstack((['objective','criteria'],fit_path_cols))
        self.fit_path_inner = pd.DataFrame(np.hstack((np.mat([np.nan]), np.mat(prmtr_new))),columns=fit_path_inner_cols,\
                                           index=[[0],[0]])
        self.fit_path_inner.index.rename(['iter','sub_iter'], inplace=1)
        self.fit_path = pd.DataFrame(np.hstack((np.mat([np.nan,np.nan]), np.mat(prmtr_new))),columns=fit_path_cols)
        self.fit_path.index.rename('iter', inplace=1)
        optim_output = None
        latest_obj = self.fit_path.objective.iloc[-1]

        while tol>tolerance and self.Nfeval<maxiter:
            A0_out, A1_out, U0_out, U1_out, Q_out, Phi_out = extract_mats(prmtr_new_raw, self.num_states, self.US_nominalmaturities, self.US_ilbmaturities, self.dt, self.prmtr_size_dict, self.Phi_prmtr)

            kalman1 = Kalman(self.Y, A0_out, A1_out, U0_out, U1_out, Q_out, Phi_out, self.initV, statevar_names=self.statevar_names)  # default uses X0, V0 = unconditional mean and error variance

            Ytt, Yttl, Xtt, Xttl, Vtt, Vttl, Gain_t, eta_t = kalman1.filter()

            XtT, VtT, Jt = kalman1.smoother(Xtt, Xttl, Vtt)

            #We to double number of iterations for the last call; e.g. maxiter=maxiter*2 if tol<=tolerance or self.Nfeval>=maxiter else maxiter
            if estimation_method=='em_mle' or estimation_method == 'em_mle_with_bayesian_final_iteration':
                prmtr_update_raw, optim_output = self.em_mle(XtT, self.Y, prmtr_new_raw, self.num_states, self.stationarity_assumption, self.US_ilbmaturities, \
                                                         self.US_nominalmaturities, self.dt, self.Phi_prmtr, self.prmtr_size_dict, kalman1.X0 \
                                                         , method=solver_mle, \
                                                         maxiter=maxiter_mle*2 if tol<=tolerance or self.Nfeval>=maxiter else maxiter_mle, \
                                                         maxfev=maxfev_mle*2 if tol<=tolerance or self.Nfeval>=maxiter else maxfev_mle, \
                                                         ftol=ftol_mle/2.0 if tol<=tolerance or self.Nfeval>=maxiter else ftol_mle, \
                                                             xtol=xtol_mle/2.0 if tol<=tolerance or self.Nfeval>=maxiter else xtol_mle\
                                                         , constraints= constraints_mle  )
                prmtr_update = param_mapping(self.num_states, build_prmtr_dict(prmtr_update_raw, self.prmtr_size_dict))

            elif estimation_method=='em_bayesian':
                if self.Nfeval>0:
                    priors, loc = {}, 0
                    for k in self.prmtr_size_dict.keys():
                        for i in range(self.prmtr_size_dict[k]):
                            if k in ['a', 'lmda', 'lmda2', 'Kp']:
                                priors[k + '_%i' % i] = pymc2.Uniform(k + '_%i' % i, lower=optim_output['mcmc_params'][loc].stats()['mean']*0.8, upper=optim_output['mcmc_params'][loc].stats()['mean']*1.2)
                            elif k in ['thetap']:
                                priors[k + '_%i' % i] = pymc2.Normal(k + '_%i' % i, mu=optim_output['mcmc_params'][loc].stats()['mean'],  tau=1 / optim_output['mcmc_params'][loc].stats()['standard deviation'])
                            elif k in ['Phi', 'sigmas']:
                                priors[k + '_%i' % i] = pymc2.Uniform(k + '_%i' % i, lower=optim_output['mcmc_params'][loc].stats()['mean']*0.8, upper=optim_output['mcmc_params'][loc].stats()['mean']*1.2)
                            loc+=1
                else:
                    priors = priors_bayesian
                prmtr_update, optim_output = self.em_bayesian(XtT, self.Y, self.num_states, self.US_ilbmaturities, self.US_nominalmaturities, \
                                                        self.dt, self.Phi_prmtr, self.prmtr_size_dict, kalman1.X0, \
                                                        priors=priors, \
                                                        maxiter = maxiter_bayesian*2 if tol<=tolerance or self.Nfeval>=maxiter else maxiter_bayesian, \
                                                        burnin = burnin_bayesian*2 if tol<=tolerance or self.Nfeval>=maxiter else burnin_bayesian )
                prmtr_update_raw = prmtr_update
                print('\n')

            if np.array(optim_output['fun'])>latest_obj:
                print('bad iteration')
                print('latest obj (%f) is larger than starting obj (%f)' %(np.array(optim_output['fun']),latest_obj))
            tol = np.max(np.abs(prmtr_new - prmtr_update)) if toltype == 'max_abs' else np.sum(np.array(prmtr_new - prmtr_update) ** 2) if toltype == 'l2_norm' else np.sum(np.abs(prmtr_new - prmtr_update))
            prmtr_new, prmtr_new_raw = prmtr_update, prmtr_update_raw
            self.fit_path.loc[self.fit_path.index.values[-1] + 1] = np.hstack(
                (np.array([optim_output['fun'], tol]).tolist(), prmtr_new.tolist()))
            self.Nfeval_inner += 1
            print(self.fit_path.tail(1).to_string())
            self.Nfeval +=1
            latest_obj = optim_output['fun']

        if estimation_method == 'em_mle_with_bayesian_final_iteration':
            priors, loc = {}, 0
            for k in self.prmtr_size_dict.keys():
                for i in range(self.prmtr_size_dict[k]):
                    if k in ['a', 'lmda', 'lmda2', 'Kp']:
                        priors[k + '_%i' % i] = pymc2.Uniform(k + '_%i' % i, lower=prmtr_new[loc]*0.8, upper=prmtr_new[loc]*1.2)
                    elif k in ['thetap']:
                        priors[k + '_%i' % i] = pymc2.Normal(k + '_%i' % i, mu=prmtr_new[loc],  tau=1 / (np.abs(prmtr_new[loc])*0.01))
                    elif k in ['Phi', 'sigmas']:
                        priors[k + '_%i' % i] = pymc2.Uniform(k + '_%i' % i, lower=prmtr_new[loc], upper=prmtr_new[loc]*1.2)
                    loc+=1
            prmtr_update, optim_output = self.em_bayesian(XtT, self.Y, self.num_states, self.US_ilbmaturities,
                                                        self.US_nominalmaturities, self.dt, self.Phi_prmtr,
                                                        self.prmtr_size_dict, kalman1.X0, priors=priors,
                                                        maxiter=maxiter_bayesian ,burnin=burnin_bayesian )
            tol = np.max(np.abs(prmtr_new - prmtr_update)) if toltype == 'max_abs' else np.sum(np.array(prmtr_new - prmtr_update) ** 2) if toltype == 'l2_norm' else np.sum(np.abs(prmtr_new - prmtr_update))
            prmtr_new = prmtr_update
            self.fit_path.loc[self.fit_path.index.values[-1] + 1] = np.hstack(
                (np.array([optim_output['fun'], tol]).tolist(), prmtr_new.tolist()))
            self.Nfeval_inner += 1
            print('\n')
            print(self.fit_path.tail(1).to_string())
            self.Nfeval += 1

        toc = time.clock()
        print('processing time for fit: ' + str(toc - tic))
        self.prmtr, self.optim_output = prmtr_new, optim_output
        return prmtr_new, optim_output


    def em_mle(self, X, Y, prmtr_initial, num_states, stationarity_assumption, US_ilbmaturities,
               US_nominalmaturities, dt, \
               Phi_prmtr, prmtr_size_dict, X0,method='Nelder-Mead',maxiter=1000,maxfev=1000,ftol=1e-6,xtol=1e-6, constraints='off'):
        '''E-M Algorithm with MLE '''
        print('\n\n\n')

        latest_obj = self.fit_path_inner.sub_objective.iloc[-1]

        def objective_function(prmtr_):
            '''Given vector of parameters it returns the negative cumulative log-likelihood'''
            #First, given parameter vector, build parameter matrices:
            if num_states == 4:
                a, Kp, lmda, Phi, sigma11, sigma22, sigma33, sigma44, Sigma, thetap = extract_vars(prmtr_, num_states, prmtr_size_dict, Phi_prmtr)
            elif num_states == 6:
                a, Kp, lmda, lmda2, Phi, sigma11, sigma22, sigma22_2, sigma33, sigma33_2, sigma44, Sigma, thetap = extract_vars( \
                    prmtr_, num_states, prmtr_size_dict, Phi_prmtr)
            # try:
            #     min(np.real(np.linalg.eig(np.array(Kp))[0])) > 0
            # except:
            #     print('bad')
            if (min(np.real(np.linalg.eig(np.array(Kp))[0])) < 0) & ( stationarity_assumption == 'yes'):  # imposing restriction on Kp; need positive eig vals
                cum_log_likelihood = np.mat([-np.inf])
            else:
                try:
                    A0, A1, U0, U1, Q, Phi = extract_mats(prmtr_, num_states, US_nominalmaturities, US_ilbmaturities, dt,\
                                                          prmtr_size_dict, Phi_prmtr)
                    yy = np.hstack((np.mat(Y.values), np.mat(X.values)))  # staking y_t and x_t
                    xx = np.hstack((np.mat(X.values),np.vstack((X0.T, np.mat(X.iloc[0:-1, :].values)))))  # staking x_t and x_{t-1}
                    # define mean of stacked variables
                    mean = (np.repeat(np.vstack((A0, U0)), Y.shape[0], axis=1) + \
                            np.vstack((np.hstack((A1, A1 * 0)), np.hstack((U1 * 0, U1)))) * xx.T).T
                    # define variance of stacked variables
                    variance = np.vstack((np.hstack((Phi, np.zeros((Phi.shape[0],Q.shape[1])) )), \
                                          np.hstack((np.zeros((Q.shape[0],Phi.shape[1])) * 0, Q))))
                    var = multivariate_normal(cov=variance)
                    # Then compute cumulative log likelihood
                    cum_log_likelihood = np.mat(np.sum(var.logpdf(yy-mean)))
                except:
                    # print("Unexpected error:", sys.exc_info()[0])
                    cum_log_likelihood =  np.mat([-np.inf])

            self.Nfeval_inner += 1
            self.fit_path_inner.ix[(self.fit_path.index.values[-1], self.fit_path_inner.index.values[-1][1]+1),:] = \
                np.hstack(( np.array((-1)*cum_log_likelihood)[0].tolist(),
                            param_mapping(self.num_states, build_prmtr_dict(prmtr_, self.prmtr_size_dict)).tolist() ))
            print(self.fit_path_inner.tail(1).to_string())

            return (-1) * np.reshape(np.array(cum_log_likelihood), 1, 0)  #important to reshape to scalar
        # debug_here()
        #See http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize for description of optimizer
        if method == 'COBYLA':
            tic = time.clock()
            if constraints=='off':
                optim_output = minimize(objective_function, prmtr_initial, method='COBYLA', tol=ftol,  options={'iprint': 1, 'disp': True, 'maxiter': maxiter, 'catol': ftol, 'rhobeg': 1.0})
            else:
                optim_output = minimize(objective_function, prmtr_initial, method='COBYLA', tol=ftol, constraints=self.cons, options={'iprint': 1, 'disp': True, 'maxiter': maxiter, 'catol': ftol, 'rhobeg': 1.0})
            toc = time.clock()
            print(method+': '+str(toc-tic))
        else:
            tic = time.clock()
            if constraints=='off':
                optim_output = minimize(objective_function, prmtr_initial, method=method, options={'disp': 1, 'maxiter':maxiter, 'maxfev':maxfev, 'xtol':xtol, 'ftol':ftol}) #Nelder-Mead works well
            else:
                optim_output = minimize(objective_function, prmtr_initial, method=method, constraints=self.cons, options={'disp': 1, 'maxiter':maxiter, 'maxfev':maxfev, 'xtol':xtol, 'ftol':ftol}) #Nelder-Mead works well
            toc = time.clock()
            print(method+': '+str(toc-tic))
        # debug_here()

        # tic = time.clock()
        # optim_output = minimize(objective_function, prmtr_initial, method='Nelder-Mead', options={'disp': 1, 'maxiter':maxiter, 'maxfev':maxfev})
        # toc = time.clock()
        # print('opt: '+str(toc-tic))
        # optim_output = minimize(objective_function, prmtr_initial, method='trust-ncg', options={'disp': 1, 'maxiter':maxiter, 'maxfev':maxfev}) #jacobian required
        # optim_output = minimize(objective_function, prmtr_initial, method='dogleg', options={'disp': 1, 'maxiter':maxiter, 'maxfev':maxfev}) # requires the gradient and Hessian
        # optim_output = minimize(objective_function, prmtr_initial, method='CG', options={'disp': 1, 'maxiter':maxiter, 'maxfev':maxfev}) #lead to nans in prmtr
        # optim_output = minimize(objective_function, prmtr_initial, method='Newton-CG', options={'disp': 1, 'maxiter':maxiter, 'maxfev':maxfev}) # Jacobian is required
        # optim_output = minimize(objective_function, prmtr_initial, method='TNC', options={'disp': 1, 'maxiter':maxiter, 'maxfev':maxfev}) #lead to nans in prmtr, too slow, steps too small
        # optim_output = minimize(objective_function, prmtr_initial, method='Powell', options={'disp': 1, 'maxiter':maxiter, 'maxfev':maxfev}) #too slow
        # optim_output = minimize(objective_function, prmtr_initial, method='COBYLA', options={'disp': 1, 'maxiter':maxiter, 'maxfev':maxfev}) #this could work. but can get stuck
        # optim_output = minimize(objective_function, prmtr_initial, method='BFGS', options={'disp': 1, 'maxiter':maxiter, 'maxfev':maxfev}) #this could work but can lead to error with nan in prmtr
        # optim_output = minimize(objective_function, prmtr_initial, method='L-BFGS-B', options={'disp': 1, 'maxiter':maxiter, 'maxfev':maxfev}) #this could work but can take too long
        # optim_output = minimize(objective_function, prmtr_initial, method='SLSQP', options={'disp': 1, 'maxiter':maxiter, 'maxfev':maxfev}) #this could work but can lead to error with nan in prmtr
        # optim_output = minimize(objective_function, prmtr_initial, method='SLSQP', constraints = self.cons, options={'disp': 1, 'maxiter':maxiter, 'maxfev':maxfev})# only COBYLA and SLSQP allow constraints;  #this could work but can lead to error with nan in prmtr
        # optim_output = minimize(objective_function, prmtr_initial, method='Nelder-Mead', options={'disp': 1, 'maxiter':maxiter, 'maxfev':maxfev}) #this could work


        #Let's make sure that we have (weakly) improved on the previous iteration otherwise, return the previous optimal results
        count_=0
        while np.array(optim_output['fun'])>latest_obj  and count_<=4:
            #repeat optimization at most 5 times with slight noise to guess parameters
            print('bad iteration')
            print('latest obj (%f) is larger than starting obj (%f)' %(np.array(optim_output['fun']),latest_obj))
            if method == 'COBYLA':
                if constraints=='off':
                    optim_output = minimize(objective_function, prmtr_initial*(1+np.random.normal(size=prmtr_initial.shape[0])*1e-5), method='Nelder-Mead', options={'disp': 1, 'maxiter':maxiter*2, 'maxfev':maxfev*2})
                    # optim_output = minimize(objective_function, prmtr_initial*(1+np.random.normal(size=prmtr_initial.shape[0])*1e-5), method=method, options={'iprint': 1, 'disp': True, 'maxiter':maxiter*2})
                else:
                    optim_output = minimize(objective_function, prmtr_initial*(1+np.random.normal(size=prmtr_initial.shape[0])*1e-5), method=method, constraints=self.cons, options={'iprint': 1, 'disp': True, 'maxiter':maxiter*2})
            else:
                if constraints=='off':
                    optim_output = minimize(objective_function, prmtr_initial*(1+np.random.normal(size=prmtr_initial.shape[0])*1e-5), method='Nelder-Mead', options={'disp': 1, 'maxiter':maxiter*2, 'maxfev':maxfev*2})
                else:
                    optim_output = minimize(objective_function, prmtr_initial*(1+np.random.normal(size=prmtr_initial.shape[0])*1e-5), method=method, constraints=self.cons, options={'disp': 1, 'maxiter':maxiter*2, 'maxfev':maxfev*2})
            count_+=1
        if np.array(optim_output['fun'])>latest_obj:
            #Still unable to improve optimization, so just return original optimum
            print('bad iteration')
            print('latest obj (%f) is larger than starting obj (%f)' %(np.array(optim_output['fun']),latest_obj))
            print('resetting iteration')
            optim_output['x']=np.array(prmtr_initial)
            optim_output['fun']=latest_obj

        return np.array(optim_output['x']), optim_output


    def em_bayesian(self, X, Y, num_states, US_ilbmaturities, US_nominalmaturities, dt,  Phi_prmtr, prmtr_size_dict, \
                    X0, maxiter=1000, priors=None, burnin=None, method=None):
        '''E-M Algorithm with Bayesian approach '''
        print('\n\n\n')

        latest_obj = self.fit_path.objective.iloc[-1]

        global Nfeval_inner, fit_path_inner
        Nfeval_inner = self.Nfeval_inner #let's track the number of inner iterations
        burnin = maxiter/3 if burnin is None else burnin

        # Define priors. If priors not provided, use flat priors with appropriate ranges and normal prior for thetap
        if priors is None:
            priors = {}
            for k in prmtr_size_dict.keys():
                for i in range(prmtr_size_dict[k]):
                    if k in ['a','lmda','lmda2','Kp']:
                        priors[k+'_%i' % i] = pymc2.Uniform(k+'_%i' % i, lower=0.3, upper=0.99)
                    elif k in ['thetap']:
                        priors[k+'_%i' % i] = pymc2.Normal(k+'_%i' % i, mu=0, tau=1/0.01)
                    elif k in ['Phi','sigmas']:
                        priors[k+'_%i' % i] = pymc2.Uniform(k+'_%i' % i, lower=0.005, upper=0.02)
        params = pymc2.Container([priors.copy()[k] for k in priors.keys()])
        yy = np.hstack((np.mat(Y.values), np.mat(X.values)))  # staking y_t and x_t
        xx = np.hstack((np.mat(X.values), np.vstack((X0.T, np.mat(X.iloc[0:-1, :].values)))))  # staking x_t and x_{t-1}

        @pymc2.deterministic    #define mean of stacked variables
        def mean(params=params):
            prmtr_ = np.array(params)
            A0, A1, U0, U1, Q, Phi = extract_mats(prmtr_, num_states, US_nominalmaturities,
                                                  US_ilbmaturities, dt, \
                                                  prmtr_size_dict, Phi_prmtr, skip_mapping=1)
            mean_ = (np.repeat(np.vstack((A0, U0)), Y.shape[0], axis=1) + \
                            np.vstack((np.hstack((A1, A1 * 0)), np.hstack((U1 * 0, U1)))) * xx.T).T
            return mean_

        @pymc2.deterministic  # define precision of stacked variables
        def precision(params=params):
            prmtr_ = np.array(params)
            A0, A1, U0, U1, Q, Phi = extract_mats(prmtr_, num_states, US_nominalmaturities,
                                                  US_ilbmaturities, dt, \
                                                  prmtr_size_dict, Phi_prmtr, skip_mapping=1)
            variance = np.vstack((np.hstack((Phi, np.zeros((Phi.shape[0], Q.shape[1])))), \
                       np.hstack((np.zeros((Q.shape[0], Phi.shape[1])) * 0, Q))))

            precision_ = np.linalg.inv(variance)
            return precision_

        likelihood = pymc2.Container([pymc2.MvNormal('likelihood_%i' % t, mean[t, :], precision, value=yy[t, :], observed=True) \
                       for t in range(yy.shape[0])])

        # Inference
        m = pymc2.Model([params, likelihood])
        mc = pymc2.MCMC(m)

        if method is not None:
            if str.lower(method) == 'metropolis':
                mc.use_step_method(pymc2.Metropolis,params)
            elif str.lower(method) == 'slicer':
                mc.use_step_method(pymc2.Slicer,params)
            elif str.lower(method) == 'adaptivemetropolis':
                mc.use_step_method(pymc2.AdaptiveMetropolis,params)
            elif str.lower(method) == 'discretemetropolis':
                mc.use_step_method(pymc2.DiscreteMetropolis,params)
            elif str.lower(method) == 'binarymetropolis':
                mc.use_step_method(pymc2.BinaryMetropolis,params)
            elif str.lower(method) == 'gibbs':
                mc.use_step_method(pymc2.Gibbs,params)

        mc.sample(iter=maxiter, burn=burnin)    # this produces a single chain. to get multiple chains use loop (serially or in parallel;
                                                # see http://stackoverflow.com/questions/27446738/how-to-sample-multiple-chains-in-pymc3)

        if False:
            # Check for convergence.
            # See http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3097064/
            # and https://pymc-devs.github.io/pymc/modelchecking.html
            # https://pymc-devs.github.io/pymc/database.html
            for v in params:
                v.summary()
            scores = pymc2.geweke(mc, first=33, last=33)
            pymc2.Matplot.geweke_plot(scores)
            pymc2.Matplot.plot(mc)
            pymc2.raftery_lewis(mc, q=0.05, r=0.05)
            pymc2.gelman_rubin(mc) #Gelman-Rubin diagnostic requires multiple chains of the same length

            samples = np.array([v.trace() for v in params]).T
            #samples = samples[0] #if more than 1 chain, we plot the first
            import corner as triangle_plot
            # tmp = triangle_plot.corner(samples[:, :], labels=priors.keys(),
            #                       truths=[v.stats()['mean'] for v in params])
            tmp = triangle_plot.corner(samples, labels=priors.keys(),
                                   truths=[v.stats()['mean'] for v in params])

        optim_output={} # dictionary to contain results
        optim_output['x']=[v.stats()['mean'] for v in params]
        optim_output['mcmc'] = mc
        optim_output['mcmc_params'] = params

        #It will be convenient to also return the likelihood evaluated at the mean of the parameters. Although this is not needed
        A0, A1, U0, U1, Q, Phi = extract_mats(np.array(optim_output['x']), num_states, US_nominalmaturities,
                                              US_ilbmaturities, dt, \
                                              prmtr_size_dict, Phi_prmtr, skip_mapping=1)
        mean = (np.repeat(np.vstack((A0, U0)), Y.shape[0], axis=1) + \
                        np.vstack((np.hstack((A1, A1 * 0)), np.hstack((U1 * 0, U1)))) * xx.T).T
        variance = np.vstack((np.hstack((Phi, np.zeros((Phi.shape[0], Q.shape[1])))), \
                   np.hstack((np.zeros((Q.shape[0], Phi.shape[1])) * 0, Q))))
        var = multivariate_normal(cov=variance)
        cum_log_likelihood = np.mat(np.sum(var.logpdf(yy - mean)))
        optim_output['fun'] = ((-1) * np.reshape(np.array(cum_log_likelihood), 1, 0))[0]

        #Let's make sure that we have (weakly) improved on the previous iteration otherwise, return the previous optimal results
        count_=0
        while np.array(optim_output['fun'])>latest_obj  and count_<=4:
            #repeat optimization at most 5 times with slight noise to guess parameters
            print('bad iteration')
            print('latest obj (%f) is larger than starting obj (%f)' %(np.array(optim_output['fun']),latest_obj))
            mc.sample(iter=maxiter*2, burn=burnin*2)
            count_+=1
        if np.array(optim_output['fun'])>latest_obj:
            #Still unable to improve optimization, so just return original optimum
            print('bad iteration')
            print('latest obj (%f) is larger than starting obj (%f)' %(np.array(optim_output['fun']),latest_obj))
            print('resetting iteration')
            optim_output['x']=np.array(prmtr_initial)
            optim_output['fun']=latest_obj

        return np.array(optim_output['x']), optim_output


    def collect_results(self):
        '''Save results are attributes'''
        # debug_here()
        prmtr, optim_output = self.prmtr, self.optim_output
        # print(prmtr)
        # Extracting filtered states:
        if self.num_states == 4:
            self.a_new, self.Kp_new, self.lmda_new, self.Phi_new, self.sigma11_new, self.sigma22_new, self.sigma33_new, \
                self.sigma44_new, self.Sigma_new, self.thetap_new = extract_vars( prmtr, self.num_states, self.prmtr_size_dict, skip_mapping=1)

        elif self.num_states == 6:
            self.a_new, self.Kp_new, self.lmda_new, self.lmda2_new, self.Phi_new, self.sigma11_new, self.sigma22_new, \
                self.sigma22_2_new, self.sigma33_new, self.sigma33_2_new, self.sigma44_new, self.Sigma_new, \
                self.thetap_new = extract_vars(prmtr, self.num_states, self.prmtr_size_dict)

        self.A0_new, self.A1_new, self.U0_new, self.U1_new, self.Q_new, self.Phi_new = extract_mats(prmtr, self.num_states,
                                                                      self.US_nominalmaturities, self.US_ilbmaturities,
                                                                      self.dt, self.prmtr_size_dict, self.Phi_prmtr, skip_mapping=1)

        kalman2 = Kalman(self.Y, self.A0_new, self.A1_new, self.U0_new, self.U1_new, self.Q_new, self.Phi_new, self.initV, statevar_names=self.statevar_names)
        self.Ytt_new, self.Yttl_new, self.Xtt_new, self.Xttl_new, self.Vtt_new, self.Vttl_new, self.Gain_t_new, self.eta_t_new = kalman2.filter()
        self.XtT_new, self.VtT_new, self.Jt_new = kalman2.smoother(self.Xtt_new, self.Xttl_new, self.Vtt_new)

        # Computing Forecasts, RMSE, etc. :
        self.forecast_horizon = 90 * (self.estim_freq == 'daily') + 12 * (self.estim_freq == 'weekly') + 3 * (self.estim_freq == 'monthly')
        # debug_here()
        self.yields_forecast, self.yields_forecast_std, self.yields_forecast_cov = kalman2.forecast(self.XtT_new, self.forecast_horizon)

        self.forecast_e, self.forecast_se, self.forecast_mse, self.forecast_rmse, self.forecast_mse_all, self.forecast_rmse_all = kalman2.forecast_rmse(self.yields_forecast)
        self.fit_e, self.fit_se, self.fit_mse_all, self.fit_rmse_all = kalman2.fit_rmse(self.Ytt_new)

        # Referencing individual dataframe columns in USnominals and USilbs objecs
        for m in range(self.US_nominalmaturities.size):
            self.USnominals[m].yields_forecast = self.yields_forecast.iloc[:, m]
            self.USnominals[m].yields_forecast_std = self.yields_forecast_std.iloc[:, m]
            self.USnominals[m].yields_forecast_cov = self.yields_forecast_cov.iloc[:, m]

            self.USnominals[m].forecast_e, self.USnominals[m].forecast_se, self.USnominals[m].forecast_mse, \
            self.USnominals[m].forecast_rmse , self.USnominals[m].forecast_mse_all, \
            self.USnominals[m].forecast_rmse_all =  self.forecast_e.iloc[:, m], self.forecast_se.iloc[:, m], \
                                                    self.forecast_mse.iloc[:, m], self.forecast_rmse.iloc[:, m], \
                                                    self.forecast_mse_all.iloc[m], self.forecast_rmse_all.iloc[m]

        for m in range(self.US_ilbmaturities.size):
            self.USilbs[m].yields_forecast = self.yields_forecast.iloc[:, self.US_nominalmaturities.size + m]
            self.USilbs[m].yields_forecast_std = self.yields_forecast_std.iloc[:, self.US_nominalmaturities.size + m]
            self.USilbs[m].yields_forecast_cov = self.yields_forecast_cov.iloc[:, self.US_nominalmaturities.size + m]

            self.USilbs[m].forecast_e, self.USilbs[m].forecast_se, self.USilbs[m].forecast_mse, self.USilbs[m].forecast_rmse \
                ,self.USilbs[m].forecast_mse_all, self.USilbs[m].forecast_rmse_all = self.forecast_e.iloc[:, self.US_nominalmaturities.size + m], \
                                                                                      self.forecast_se.iloc[:, self.US_nominalmaturities.size + m], \
                                                                                      self.forecast_mse.iloc[:, self.US_nominalmaturities.size + m], \
                                                                                      self.forecast_rmse.iloc[:,self.US_nominalmaturities.size + m], \
                                                                                      self.forecast_mse_all.iloc[self.US_nominalmaturities.size + m], \
                                                                                      self.forecast_rmse_all.iloc[self.US_nominalmaturities.size + m]

        ######################################################################
        if self.num_states == 4:
            self.rho_n = np.mat(np.array([1, 1, 0, 0])).T
            self.rho_r = np.mat(np.array([0, self.a_new, 0, 1])).T
        elif num_states == 6:
            self.rho_n = np.mat(np.array([1, 1, 1, 0, 0, 0])).T
            self.rho_r = np.mat(np.array([0, self.a_new, self.a_new, 0, 0, 1])).T

        if self.num_states == 6:
            self.lmda2_new, self.sigma22_2_new, self.sigma33_2_new =  lmda2_new, sigma22_2_new, sigma33_2_new


    def expected_inflation(self):
        '''Compute expected inflation'''
        # Do not use smoother here to avoid look-ahead bias
        Kp_new, rho_n, rho_r, Sigma_new, thetap_new, Xtt_new, rho_n, rho_r = self.Kp_new, self.rho_n, self.rho_r, \
                                                                             self.Sigma_new, self.thetap_new, self.Xtt_new, self.rho_n, self.rho_r

        # Computing Expected Inflation:
        horizons = np.array(np.arange(100).T)  # horizon in years
        mttau = np.empty((self.Y.shape[0], self.num_states + 1, horizons.size))
        vttau = np.empty((self.Y.shape[0], (self.num_states + 1) ** 2, horizons.size))

        # First we solve ODE for Covariance matrix of augmented states
        v_ode_out = integrate.ode(v0teqns).set_integrator('dopri5', verbosity=1)
        v_ode_out.set_initial_value(np.zeros(((self.num_states + 1) ** 2, 1))[:, 0], horizons[0]).set_f_params(Kp_new,
                                                                                                               rho_n,
                                                                                                               rho_r,
                                                                                                               Sigma_new,
                                                                                                               thetap_new)
        t_out, v_out = np.array([0]), np.zeros(((self.num_states + 1) ** 2, 1)).T
        while v_ode_out.successful() and v_ode_out.t < horizons[-1]:
            v_ode_out.integrate(v_ode_out.t + 1)
            t_out = np.vstack((t_out, v_ode_out.t))
            v_out = np.vstack((v_out, v_ode_out.y))

        # Then we solve ODE for Mean of augmented states
        for tt in np.arange(0, self.Y.shape[0]):
            m_ode_out = integrate.ode(m0teqns).set_integrator('dopri5', verbosity=1)
            m_ode_out.set_initial_value(np.array(np.hstack((Xtt_new.values, np.zeros((Xtt_new.shape[0], 1)))))[tt, :], \
                                        horizons[0]).set_f_params(Kp_new, rho_n, rho_r, Sigma_new, thetap_new)
            t_out2, m_out = np.array([0]), np.array(np.hstack((Xtt_new.values, np.zeros((Xtt_new.shape[0], 1)))))[tt, :]
            while m_ode_out.successful() and m_ode_out.t < horizons[-1]:
                m_ode_out.integrate(m_ode_out.t + 1)
                t_out2 = np.vstack((t_out2, m_ode_out.t))
                m_out = np.vstack((m_out, m_ode_out.y))
            mttau[tt, :, :] = m_out.T
            vttau[tt, :, :] = v_out.T

        # Lastly, we are only interested in the column corresponding to variable that is not among the original state variables; so we extract it out
        mttau_nn = np.mat(mttau[:, -1, :])
        vttau_nn = np.mat(vttau[:, -1, :])

        # Now we can compute the expected inflation
        exp_inf = -mttau_nn + 0.5 * vttau_nn
        exp_inf = np.array(np.tile(-1.0 / horizons.T, (Xtt_new.shape[0], 1))) * np.array(exp_inf)
        exp_inf[:, 0] = np.array(((rho_n - rho_r).T) * (np.mat(thetap_new)))[0, 0]
        exp_inf = pd.DataFrame(np.mat(exp_inf), index=self.Y.index, columns=['horizon_%iyr' %i for i in range(exp_inf.shape[1])] )

        # Now we can compute the break-evens
        bk_mats = np.unique(np.vstack((self.US_nominalmaturities, self.US_ilbmaturities)))
        bk_evens = pd.DataFrame(
            (np.array([self.Y.iloc[:, np.in1d(self.US_nominalmaturities, m)].values  # Nominal bond with maturity = m
                       - (self.Y.iloc[:, self.US_nominalmaturities.size:].values)[:, np.in1d(self.US_ilbmaturities, m)]  # ILB with maturity = m
                       for m in bk_mats])[:, :, 0]).T
            , index=self.Y.index, columns=['horizon_%iyr' %i for i in bk_mats] )

        # %now we can compute the IRPs
        irps = bk_evens - exp_inf.iloc[:, bk_mats - 1]

        # lastly we compute the deflation probabilities
        prob_def = -np.array(mttau_nn) / (np.array(vttau_nn) ** 0.5)
        prob_def = norm.cdf(prob_def)
        prob_def[:, 0] = 0
        prob_def = pd.DataFrame(np.mat(prob_def), index=self.Y.index, columns=['horizon_%iyr' %i for i in range(prob_def.shape[1])] )

        self.bk_mats, self.exp_inf, self.irps, self.mttau, self.mttau_nn, self.prob_def, self.vttau, self.vttau_nn = \
            bk_mats, exp_inf, irps, mttau, mttau_nn, prob_def, vttau, vttau_nn

        return bk_mats, exp_inf, irps, mttau, mttau_nn, prob_def, vttau, vttau_nn


    def save_output(self, bk_mats, prmtr_new, exp_inf, mttau, mttau_nn, prob_def, vttau, vttau_nn):
        return 1
    #     # recording output
    #     datadate = np.array([np.union1d(dates['US_NB'], dates['US_ILB']).astype(DT.datetime)[tt].year*10000+np.union1d(dates['US_NB'], dates['US_ILB']).astype(DT.datetime)[tt].month*100+np.union1d(dates['US_NB'], dates['US_ILB']).astype(DT.datetime)[tt].day for tt in np.arange(0,np.union1d(dates['US_NB'], dates['US_ILB']).shape[0])])
    #     datadate = np.hstack((0, datadate))
    #
    #     if self.num_states ==4:
    #         lmda2_new=0
    #     # TODO: NEED TO FIX THIS CODE BELOW FOR SAVING THE OUTPUT AS TXT FILES. IT IS CURRENTLY NOT STARTING NEW LINE AFTER THE HEADERS.
    #     # creating txt files to record latest observation of each time series of key results
    #     my_matrix = np.array(['prob_def','irps','bk_evens','exp_inf','Y','Ytt_new','Yttl_new','Xtt_new','Xttl_new','Vtt_new','Vttl_new','prmtr_new','Kp_new','thetap_new','Sigma_new','a_new','lmda_new','lmda2_new','Phi_new','Q_new','U0_new','U1_new','A0_new','A1_new','V0','X0'])
    #     if os.path.isfile(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files\prob_def_'+estim_freq+'.txt')==False:  # if files do not exist, creat new ones with appropriate headers
    #         header_prob_def, header_exp_inf = (tuple(['dates'])+tuple([',horizon: '+str(vv)+'yr' for vv in horizons])+tuple(['\n']) for vvv in range(2))
    #         header_Y, header_Ytt_new,  header_Yttl_new = (tuple(['dates'])+tuple([',Nominal: '+str(US_nominalmaturities[vv])+'yr' for vv in range(US_nominalmaturities.size)])\
    #                                              + tuple([',ILB: '+str(US_ilbmaturities[vv])+'yr' for vv in range(US_ilbmaturities.size)])+tuple(['\n']) for vvv in range(3))
    #         header_irps, header_bk_evens = (tuple(['dates'])+tuple([','+str(vv)+'yr' for vv in bk_mats])+tuple(['\n']) for vvv in range(2))
    #         if self.num_states==4:
    #             header_Xtt_new,  header_Xttl_new, header_thetap_new, header_X0  = (tuple(['dates,Ln,S,C,Lr'])+tuple(['\n']) for vvv in range(4))
    #         if self.num_states==6:
    #             header_Xtt_new,  header_Xttl_new, header_thetap_new, header_X0  = (tuple(['dates,Ln,S1,S2,C1,C2,Lr'])+tuple(['\n']) for vvv in range(4))
    #         header_Vtt_new,  header_Vttl_new, header_V0 = (((tuple(['dates'])+tuple(np.reshape([[',V('+str(vv)+';'+str(vv2)+')'  for vv in range(num_states)] for vv2 in range(num_states)],(num_states**2)))+tuple(['\n']) )) for vvv in range(3))
    #         header_prmtr_new = tuple(['dates'])+tuple([',prmtr('+str(vv)+')' for vv in range(prmtr_new.size)])+tuple(['\n'])
    #       exp_inf  header_Kp_new = tuple(['dates'])+tuple(np.reshape([[',Kp('+str(vv)+';'+str(vv2)+')'  for vv in range(num_states)] for vv2 in range(num_states)],(num_states**2)))+tuple(['\n'])
    #         header_Sigma_new = tuple(['dates'])+tuple(np.reshape([[',Sigma('+str(vv)+';'+str(vv2)+')'  for vv in range(num_states)] for vv2 in range(num_states)],(num_states**2)))+tuple(['\n'])
    #         header_a_new, header_lmda_new, header_lmda2_new  = tuple(['dates,a'])+tuple(['\n']), tuple(['dates,lmda'])+tuple(['\n']), tuple(['dates,lmda2'])+tuple(['\n'])
    #         header_Phi_new = tuple(['dates'])+tuple(np.reshape([[',Sigma('+str(vv)+';'+str(vv2)+')'  for vv in range(US_num_maturities)] for vv2 in range(US_num_maturities)],(US_num_maturities**2)))+tuple(['\n'])
    #         header_Q_new = tuple(['dates'])+tuple(np.reshape([[',Q('+str(vv)+';'+str(vv2)+')'  for vv in range(num_states)] for vv2 in range(num_states)],(num_states**2)))+tuple(['\n'])
    #         header_U0_new = tuple(['dates'])+tuple([',U0('+str(vv)+')' for vv in range(num_states)])+tuple(['\n'])
    #         header_U1_new = tuple(['dates'])+tuple(np.reshape([[',U1('+str(vv)+';'+str(vv2)+')'  for vv in range(num_states)] for vv2 in range(num_states)],(num_states**2)))+tuple(['\n'])
    #         header_A0_new = tuple(['dates'])+tuple([',A0('+str(vv)+')' for vv in range(US_num_maturities)])+tuple(['\n'])
    #         header_A1_new = tuple(['dates'])+tuple(np.reshape([[',A1('+str(vv2)+';'+str(vv)+')'  for vv in range(num_states )] for vv2 in range(US_num_maturities)],(num_states*US_num_maturities)))+tuple(['\n'])
    #         header_US_nominal_yields_forecast, header_US_ilb_yields_forecast = tuple(['dates'])+tuple([','+str(vv)+' period' for vv in range(yields_forecast.shape[2])])+tuple(['\n']), tuple(['dates'])+tuple([','+str(vv)+' period' for vv in range(yields_forecast.shape[2])])+tuple(['\n'])
    #         header_US_nominal_forecast_e, header_US_ilb_forecast_e = tuple(['dates'])+tuple([','+str(vv)+' period' for vv in range(yields_forecast.shape[2])])+tuple(['\n']), tuple(['dates'])+tuple([','+str(vv)+' period' for vv in range(yields_forecast.shape[2])])+tuple(['\n'])
    #         header_US_nominal_forecast_se, header_US_ilb_forecast_se = tuple(['dates'])+tuple([','+str(vv)+' period' for vv in range(yields_forecast.shape[2])])+tuple(['\n']), tuple(['dates'])+tuple([','+str(vv)+' period' for vv in range(yields_forecast.shape[2])])+tuple(['\n'])
    #         header_US_nominal_forecast_rmse, header_US_ilb_forecast_rmse = (tuple(['dates'])+tuple([',rmse'])+tuple(['\n']) for vvv in range(2))
    #         for f in np.arange(my_matrix.size):
    #             try: #python 2
    #                 f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files'+'\\'+my_matrix[f] +'_'+estim_freq+'.txt', 'w' )
    #                 f_handle.write(np.array(eval('header_'+my_matrix[f])))
    #             except: #python 3
    #                 f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files'+'\\'+my_matrix[f] +'_'+estim_freq+'.txt', 'wb' )
    #                 np.savetxt(f_handle, np.array(eval('header_'+my_matrix[f])), fmt='%s', delimiter=',')
    #             f_handle.close()
    #         for nn in ['nominal','ilb']:
    #             for m in range(eval('US_'+nn+'maturities.size')):
    #                 for ss in ['yields_forecast','forecast_e' ,'forecast_se','forecast_rmse' ]:
    #                     try: #python 2
    #                         f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files\US'+nn+'s_'+ss+'_'+str(eval('US_'+nn+'maturities[m]'))+'yrMat_'+estim_freq+'.txt','w')
    #                         f_handle.write(np.array(eval('header_US_'+nn+'_'+ss)))
    #                     except: #python 3
    #                         f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files\US'+nn+'s_'+ss+'_'+str(eval('US_'+nn+'maturities[m]'))+'yrMat_'+estim_freq+'.txt','wb')
    #                         np.savetxt(f_handle, np.array(eval('header_US_'+nn+'_'+ss)), fmt='%s', delimiter=',')
    #                     f_handle.close()
    #     # append latest observation of each time series of key results:
    #     my_matrix = np.array(['prob_def','irps','bk_evens','exp_inf','Y','Ytt_new','Yttl_new','Xtt_new','Xttl_new','Vtt_new','Vttl_new'])
    #     for f in np.arange(my_matrix.size):
    #         try: #python 2
    #             f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files'+'\\'+my_matrix[f] +'_'+estim_freq+'.txt' , 'a')
    #             np.savetxt(f_handle, np.mat(np.hstack((datadate[-1], np.reshape(np.array(eval(my_matrix[f]))[-1],(np.array(eval(my_matrix[f]))[-1].size))))), delimiter=',')
    #         except: #python 3
    #             f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files'+'\\'+my_matrix[f] +'_'+estim_freq+'.txt' ,'ab')
    #             np.savetxt(f_handle, np.mat(np.hstack((datadate[-1], np.reshape(np.array(eval(my_matrix[f]))[-1],(np.array(eval(my_matrix[f]))[-1].size))))), delimiter=',')
    #         f_handle.close()
    #
    #     my_matrix = np.array(['prmtr_new','Kp_new','thetap_new','Sigma_new','a_new','lmda_new','lmda2_new','Phi_new','Q_new','U0_new','U1_new','A0_new','A1_new','V0','X0'])
    #     for f in np.arange(my_matrix.size):
    #         if (my_matrix[f] == 'V0') | (my_matrix[f]== 'X0'):
    #             try: #python 2
    #                 f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files'+'\\'+my_matrix[f] +'_'+estim_freq+'.txt' , 'a')
    #                 np.savetxt(f_handle, np.mat(np.hstack((datadate[-1], np.reshape(np.array(eval('kalman2.'+my_matrix[f])),(np.array(eval('kalman2.'+my_matrix[f])).size))))), delimiter=',')
    #             except: #python 3
    #                 f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files'+'\\'+my_matrix[f] +'_'+estim_freq+'.txt' , 'ab')
    #                 np.savetxt(f_handle, np.mat(np.hstack((datadate[-1], np.reshape(np.array(eval('kalman2.'+my_matrix[f])),(np.array(eval('kalman2.'+my_matrix[f])).size))))), delimiter=',')
    #             f_handle.close()
    #         else:
    #             try: #python 2
    #                 f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files'+'\\'+my_matrix[f] +'_'+estim_freq+'.txt' , 'a')
    #                 np.savetxt(f_handle, np.mat(np.hstack((datadate[-1], np.reshape(np.array(eval(my_matrix[f])),(np.array(eval(my_matrix[f])).size))))), delimiter=',')
    #             except: #python 3
    #                 f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files'+'\\'+my_matrix[f] +'_'+estim_freq+'.txt' , 'ab')
    #                 np.savetxt(f_handle, np.mat(np.hstack((datadate[-1], np.reshape(np.array(eval(my_matrix[f])),(np.array(eval(my_matrix[f])).size))))), delimiter=',')
    #             f_handle.close()
    #
    #     for nn in ['nominal','ilb']:
    #         for m in range(eval('US_'+nn+'maturities.size')):
    #             for ss in ['yields_forecast','forecast_e' ,'forecast_se','forecast_rmse' ]:
    #                 try: #python 2
    #                     f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files\US'+nn+'s_'+ss+'_'+str(eval('US_'+nn+'maturities[m]'))+'yrMat_'+estim_freq+'.txt','a')
    #                     np.savetxt(f_handle, np.mat(np.hstack((datadate[-1], np.reshape(np.array(eval('US'+nn+'s[m].'+ss))[-1],(np.array(eval('US'+nn+'s[m].'+ss))[-1].size))))), delimiter=',')
    #                 except: #python 3
    #                     f_handle = open(r'Z:\GMO\Research\AffineTermStructure\code\python_haats\txt_files\US'+nn+'s_'+ss+'_'+str(eval('US_'+nn+'maturities[m]'))+'yrMat_'+estim_freq+'.txt','ab')
    #                     np.savetxt(f_handle, np.mat(np.hstack((datadate[-1], np.reshape(np.array(eval('US'+nn+'s[m].'+ss))[-1],(np.array(eval('US'+nn+'s[m].'+ss))[-1].size))))), delimiter=',')
    #                 f_handle.close()
    #
    #
    def plot_results(self,Ytt, Yttl, Xtt, Xttl, Vtt, Vttl, Gain_t, eta_t, figures,US_ilbmaturities, US_nominalmaturities):
        return 1
    #     for vv in ['Ytt', 'Yttl', 'Xtt', 'Xttl', 'Vtt', 'Vttl', 'Gain_t', 'eta_t']:
    #         plt.close()
    #         fig, ax = plt.subplots()
    #         ax.plot(eval(vv))
    #         ax.set_title(vv)
    #         plt.draw()
    #         figures[vv] = fig
    #         figures['ax_'+vv] = ax
    #         figures[vv+'_name'] = '\\vv'
    #         filename = r"S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python" + \
    #             str(figures[vv+'_name']) + ".eps"
    #         # plt.savefig(filename, format="eps")
    #     plt.close()
    #     fig, ax = plt.subplots()
    #     ax.plot(self.Y)
    #     ax.set_title('Y')
    #     figures['Y'] = fig
    #     figures['ax_Y'] = ax
    #     figures['Y_name'] = '\\Y'
    #     plt.draw()
    #
    #     plt.close()
    #     fig, ax = plt.subplots(2,sharex=True)
    #     ax[0].plot(self.Y[:,range(US_nominalmaturities.size)],\
    #                label=[r'mat: '+str(US_nominalmaturities[vvv]) for vvv in range(US_nominalmaturities.size)])
    #     plt.gca().set_color_cycle(None)     # reset color cycle
    #     ax[0].plot(Ytt[:,range(US_nominalmaturities.size)], '--',\
    #                label=[r'mat: '+str(US_nominalmaturities[vvv]) for vvv in range(US_nominalmaturities.size)])
    #     ax[0].set_title('Realized and Model Nominal Yields')
    #     # handles, labels = figures['ax_YvYtt'].get_legend_handles_labels()
    #     # figures['ax_YvYtt'].legend(handles, [r'mat: '+str(np.hstack((US_maturities,US_maturities))[vvv]) for vvv in range(Y.shape[1])])
    #     ax[1].plot(self.Y[:,US_nominalmaturities.size+np.arange(US_ilbmaturities.size)],\
    #                label=[r'mat: '+str(US_ilbmaturities[vvv]) for vvv in range(US_ilbmaturities.size)])
    #     plt.gca().set_color_cycle(None)     # reset color cycle
    #     ax[1].plot(Ytt[:,US_nominalmaturities.size+np.arange(US_ilbmaturities.size)], '--',\
    #                label=[r'mat: '+str(US_ilbmaturities[vvv]) for vvv in range(US_ilbmaturities.size)])
    #     ax[1].set_title('Realized and Model ILB Yields')
    #     figures['YvYtt'] = fig
    #     figures['ax_YvYtt'] = ax
    #     figures['YvYtt_name'] = '\\YvYtt'
    #     plt.draw()
    #
    #     plt.close()
    #     fig, ax = plt.subplots(2, sharex=True)
    #     ax[0].plot(self.Y[:,range(US_nominalmaturities.size)] - Ytt[:,range(US_nominalmaturities.size)],\
    #                label=[r'mat: '+str(US_nominalmaturities[vvv]) for vvv in range(US_nominalmaturities.size)])
    #     ax[0].set_title('Realized vs. Model Nominal Yields')
    #     ax[1].plot(self.Y[:,US_nominalmaturities.size+np.arange(US_ilbmaturities.size)] - \
    #                Ytt[:,US_nominalmaturities.size+np.arange(US_ilbmaturities.size)],\
    #                label=[r'mat: '+str(US_ilbmaturities[vvv]) for vvv in range(US_ilbmaturities.size)])
    #     ax[1].set_title('Realized vs. Model ILB Yields')
    #     figures['Y_min_Ytt'] = fig
    #     figures['ax_Y_min_Ytt'] = ax
    #     figures['Y_min_Ytt_name'] = '\\Y_min_Ytt'
    #     plt.draw()
    #
    #     thetap = np.mat(np.linalg.inv(np.identity(self.U0.size) - self.U1)*self.U0)
    #     thetap_vec = np.tile(thetap.T, (Xtt.shape[0], 1))
    #
    #     plt.close()
    #     fig, ax = plt.subplots(2, sharex=False)
    #     ax[0].plot(Xtt, label=[""])
    #     ax[0].set_color_cycle(None)     # reset color cycle
    #     ax[0].plot(thetap_vec, '--', label=[""])
    #     ax[0].set_title('State Variables')
    #     handles, labels = ax[0].get_legend_handles_labels()
    #     if Xtt.shape[1] == 4:
    #         ax[0].legend(handles, [r'L^N_t', r'S_t', r'C_t', r'L^R_t', r'\theta_1', r'\theta_2', r'\theta_3', r'\theta_4'])
    #     elif Xtt.shape[1] == 5:
    #         ax[0].legend(handles, [r'L^N_t', r'S_t', r'C_t', r'L^R_t', r'\xi_t', r'\theta_1', r'\theta_2', r'\theta_3', r'\theta_4', r'\theta_5'])
    #     else:
    #         ax[0].legend(handles, [r'L^N_t', r'S^{(1)}_t', r'S^{(2)}_t', r'C^{(1)}_t', r'C^{(2)}_t', r'L^R_t', r'\theta_1', r'\theta_2', r'\theta_3', r'\theta_4', r'\theta_5', r'\theta_6'])
    #     ax[1].plot(eta_t)
    #     ax[1].set_title('State Variables Stochastic Error (\eta_t)')
    #     figures['XttvThetap'] = fig
    #     figures['ax_XttvThetap'] = ax
    #     figures['XttvThetap_name'] = '\\XttvThetap'
    #     plt.draw()
