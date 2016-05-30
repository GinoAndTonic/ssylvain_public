from __future__ import division
from scipy.linalg import expm
from scipy import integrate
from math import exp
import sys
import numpy as np
import pandas as pd
# import numpy.ma as ma
from scipy.linalg import solve_discrete_lyapunov
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt
import multiprocessing as multiprocessing
from joblib import Parallel, delayed
from functools import partial
import time
from IPython.core.debugger import Tracer; debug_here = Tracer() #this is the approach that works for ipython debugging
# plt.rc('text', usetex=True)   #need to install miktex
plt.close("all")
plt.close()
plt.ion()
__author__ = 'ssylvain'


''' REQUIRED INPUTS:
 1) the measurement: Y, it is a (T x m) matrix where each row is an observation and
                 each column is a different variable
 2-5) the parameters: A0 , A1 , U0, U1, Q, Phi are such that
                 Y{t+1} = A0 + A1 X{t}' + u{t+1}  : Measurement Equation
                 X{t} = U0 + U1 X{t-1}' + e{t}    : State Equation
                 here Y{t+1} and X{t} are column vectors of size m and n
                 respectively.
                 Phi = cov(u{t+1})
                 Q = cov(e{t})
 OPTIONAL INPUTS:
 6) initV specifies how to initialize the error variance V0:
     initV='steady_state' or initV='unconditional'(default) or initV='identity' matrix
 7) The initial state X0, otherwise the unconditional mean is used,
    X0 =U0/(eye() - U1)
 8) The initial error variance V0. If we specify V0 directly, it will overwrite
 initV.

 Thus this filter is flexible enough to allow for multi-dimension states and
 measurements.

 REQUIRED OUTPUTS:
 1-2) The filtered and conditional measurements, Ytt and Yttl which are a T x M matrix
         Ytt{t+1} = A0 + A1 Xtt{t}'
         Yttl{t+1} = A0 + A1 Xttl{t}'
 3) The filtered states, Xtt which is a T x n matrix,
 4) The conditional filtered states Xttl which is a T x n matrix
 5) The filtered error variance, Vtt which is a T x n^2 matrix,
 6) The conditional error variance Vttl which is a T x n^2 matrix
 7) The Kalman Gains, Gain_t
 8) The cumulative log likelihood: cum_log_likelihood which is a scalar.

 ALL VECTORS ARE COLUMN VECTORS.
 ALL ARRAYS SHOULD BE NUMPY ARRAYS.
'''


class Kalman:  # define super-class
    kalmanCount = 0

    def __init__(self, Y, A0, A1, U0, U1, Q, Phi, initV='unconditional', X0 = None, V0 = None, statevar_names = None):
        self.Y, self.A0, self.A1, self.U0, self.U1, self.Q, self.Phi, self.initV, self.X0, self.V0, self.statevar_names = Y, A0, A1, U0, U1, \
                                                                                                     Q, Phi, initV, X0, V0, statevar_names
        n = self.U0.size
        if self.X0 is None: # initialize X0 at unconditional mean
            self.X0 = np.mat(np.linalg.inv(np.identity(n) - self.U1)*self.U0)
        if self.V0 is None:
            if initV == 'steady_state':  # solve discrete Ricatti equation
                self.V0 = np.mat(solve_discrete_are(np.array(self.U1.T), np.array(self.A1.T), np.array(self.Q), np.array(self.Phi.T)))
            elif initV == 'unconditional':  # solve discrete Lyapunov equation
                self.V0 = np.mat(solve_discrete_lyapunov(np.array(self.U1.T), np.array(self.Q)))
            elif initV == 'identity':
                self.V0 = np.mat(np.identity(n))
        if statevar_names is None:
            self.statevar_names = np.arange(len(self.X0))
        Kalman.kalmanCount += 1
        # print("Calling Kalman constructor")


    def __del__(self):
        class_name = self.__class__.__name__
        # print(class_name, "destroyed")


    def filter(self):
        T = self.Y.shape[0]
        m = self.Y.shape[1]
        n = self.U0.size
        cum_log_likelihood = 0
        Ytt, Yttl = (self.Y * np.nan for vv in range(2))
        Xtt, Xttl = (pd.DataFrame(np.empty((T,n))*np.nan, index=self.Y.index, columns=self.statevar_names) for vv in range(2))
        Vtt, Vttl = (pd.DataFrame(np.empty((T,n**2))*np.nan, index=self.Y.index,
                                  columns=[str(Xtt.columns[i])+'_'+str(Xtt.columns[j]) for i in np.arange(n) for j in np.arange(n)]) for vv in range(2))
        Gain_t = (pd.DataFrame(np.empty((T,n*m))*np.nan, index=self.Y.index,
                                  columns=[str(Xtt.columns[i])+'_'+str(self.Y.columns[j]) for i in np.arange(n) for j in np.arange(m)]) )
        eta_t = Xtt * np.nan

        for t in range(T):
            y = np.mat(self.Y.iloc[t, :].values).T

            # in case there are missing data, we will remove them:
            m_ind = np.array(y != np.nan)[:,0]
            m2 = min(m, sum(m_ind))
            y, A0, A1, U0, U1, Phi, Q, X0, V0, initV = y[m_ind], self.A0[m_ind], \
                                                 self.A1[m_ind, :], self.U0, self.U1,\
                                                 self.Phi[np.tile(m_ind,(1,m2)).reshape(m2,m2)].reshape(m2,m2),\
                                                 self.Q, self.X0, self.V0, self.initV

            # STEP1: predicted states and variance
            if t == 0:
                xttl = U0 + U1 * X0
                if initV == 'steady_state':
                    vttl = V0
                else:
                    vttl = U1 * V0 * U1.T + Q
            else:
                xttl = U0 + U1 * xtt
                if initV == 'steady_state':
                    vttl = V0
                else:
                    vttl = U1 * vtt * U1.T + Q

                # STEP2: Update
            varepsilon = y - A0 - A1 * xttl
            S = Phi + A1 * vttl * A1.T
            try:
               np.linalg.inv(S)
            except:
                err_invS = sys.exc_info()[0]
                print(err_invS)
                cum_log_likelihood = - np.inf
                # return np.mat(Ytt), np.mat(Yttl), np.mat(Xtt), np.mat(Xttl), np.mat(Vtt), np.mat(Vttl), np.mat(Gain_t), cum_log_likelihood
                break

            Gain = vttl * A1.T * np.linalg.inv(S)
            xtt = xttl + Gain * varepsilon

            if initV == 'steady_state':
                vtt = V0
            else:
                vtt = vttl - Gain * A1 * vttl

            Ytt.iloc[t, m_ind] = (A0 + A1 * xtt).T
            Yttl.iloc[t, m_ind] = (A0 + A1 * xttl).T

            Xtt.iloc[t, :] = xtt.T       # recording filterered states along columns and dates along rows.
            Vtt.iloc[t, :] = (vtt.T).reshape(1, vtt.size)  # recording filterered variance along columns and dates along rows.

            Xttl.iloc[t, :] = xttl.T  # recording filtered states along columns and dates along rows.
            Vttl.iloc[t, :] = (vttl.T).reshape(1, vttl.size) # recording filterered variance along columns and dates along rows.

            Gain0 = np.empty((n, m)) * np.nan
            Gain0[:, m_ind] = Gain
            Gain_t.iloc[t, :] = Gain0.reshape(1, n*m)

            if t==0:
                eta_t.iloc[t, :] = (U0 + U1 * X0 - X0).T
            else:
                eta_t.iloc[t, :] = (U0 + U1 * xtt - np.mat(Xtt.iloc[t-1, :].values).T).T

        return Ytt, Yttl, Xtt, Xttl, Vtt, Vttl, Gain_t, eta_t


    def forecast(self, X, horizon=90):
        '''Given state variable X, return predicted measurement Y. For each t we forecast out for h=0,...,H.
        h=0 is the current (known) Y value.'''
        tic = time.clock()
        #pre-allocating space for mean and std of forecast paths. we use 2 index columns for the dataframe: date and horizon. For each date
        #there is a corresponding horizon
        T, m = self.Y.shape[0], self.Y.shape[1]
        y_avgfcst = pd.DataFrame( np.empty((T*(horizon+1), m))*np.nan, columns=self.Y.columns )
        y_avgfcst['date'] = np.repeat(self.Y.index, horizon+1)
        y_avgfcst['horizon'] = y_avgfcst['date'] + \
                           pd.TimedeltaIndex(np.tile(np.arange(horizon + 1), T).tolist(),unit=self.Y.index.freq._prefix)
        y_avgfcst.set_index(['date','horizon'],inplace=True)
        y_stdfcst = y_avgfcst.copy()
        y_covfcst = pd.DataFrame(np.empty((T*(horizon+1), m**2))*np.nan, index=y_stdfcst.index,
                                  columns=[str(y_stdfcst.columns[i])+'_'+str(y_stdfcst.columns[j]) for i in np.arange(m) for j in np.arange(m)])

        # now we do double loop over dates and horizons to fill ouput data matrix.
        # note that we will not yet fill in values for horizon=0; hence the h+1
        for t in np.arange(T):     # loop over dates
            varsigma = self.Phi*0
            x = np.mat(X.iloc[t, :]).T  #initialize state for h = 0
            for h in np.arange(horizon):     # loop over horizons
                varsigma = varsigma + self.A1 * (self.U1**h) * self.Q * (self.U1.T**h) * self.A1.T
                x = self.U0 + self.U1 * x
                # note that we will not yet fill in values for horizon=0; hence the h+1 on the next line
                loc_th = ( self.Y.index[t], (self.Y.index[t]+pd.TimedeltaIndex([h+1], unit=self.Y.index.freq._prefix))[0] ) # tuple for dual-index location
                size_th = y_avgfcst.loc[loc_th, :].shape # size of output at iteration t, h
                y_avgfcst.loc[loc_th, :] = np.reshape(self.A0 + self.A1 * x , m)
                y_stdfcst.loc[loc_th, :] = np.reshape(np.diag(varsigma)**(0.5), m)
                y_covfcst.loc[loc_th, :] = (varsigma.T).reshape(1,m**2)
            # print('done with date t='+str(t)+' forecasts')
        # here is where we fill in values for h=0 for all t
        y_avgfcst.iloc[np.arange(0, y_avgfcst.shape[0], horizon+1)] = np.reshape(self.Y.values, \
                                                                         y_avgfcst.iloc[np.arange(0, y_avgfcst.shape[0], horizon + 1)].shape)
        y_stdfcst.iloc[np.arange(0, y_avgfcst.shape[0], horizon+1)] = 0
        y_covfcst.iloc[np.arange(0, y_avgfcst.shape[0], horizon+1)] = 0
        toc = time.clock()
        print('processing time for forecast: '+str(toc-tic))
        return y_avgfcst, y_stdfcst, y_covfcst


    def forecast_rmse(self, y_fcst):  #RMSE calculations
        '''Returns error, squared error, mse(rmse): time series of mean(root-mean) squared error over horizons,
        mse(rmse): scalars for mean(root-mean) squared error over all dates and horizons'''
        # compute forecast error:  difference between y(h) and forecast(h).
        forecast_e = pd.DataFrame(\
            (self.Y.reindex(y_fcst.index.get_level_values('horizon')) - y_fcst.reset_index('date', drop=True)).reset_index())
        forecast_e['date'] = y_fcst.index.get_level_values('date')
        forecast_e.set_index(['date', 'horizon'], inplace=True)
        # compute squared forecast error:
        forecast_se = forecast_e ** 2
        # compute rmse for each date t by taking mean over horizons h=0,...,H
        forecast_mse = forecast_se.reset_index().groupby('date').mean()
        forecast_rmse = forecast_mse ** 0.5
        # compute rmse across all dates and horizons
        forecast_mse_all = pd.DataFrame(forecast_se.mean(),columns=['MSE'])
        forecast_rmse_all = pd.DataFrame(forecast_mse_all ** 0.5,columns=['RMSE'])

        return forecast_e, forecast_se, forecast_mse, forecast_rmse, forecast_mse_all, forecast_rmse_all


    def fit_rmse(self, Ytt):  #RMSE calculations
        '''Returns error, squared error, mse(rmse): time series of mean(root-mean) squared,
        mse(rmse): scalars for mean(root-mean) squared error over all dates'''
        #compute error of fit
        debug_here()
        fit_e = self.Y-Ytt
        # compute squared fit error:
        fit_se = fit_e ** 2
        # compute rmse across all dates
        fit_mse_all = pd.DataFrame(fit_se.mean(),columns=['MSE'])
        fit_rmse_all = pd.DataFrame(fit_mse_all ** 0.5,columns=['RMSE'])

        return fit_e, fit_se, fit_mse_all, fit_rmse_all


    def smoother(self, Xtt, Xttl, Vtt):
        '''Kalman smoother'''
        n = self.U0.shape[0]
        XtT = Xtt * np.nan
        VtT = Vtt * np.nan
        Jt = Vtt * np.nan
        XtT.iloc[-1,:] = Xtt.iloc[-1,:];
        VtT.iloc[-1,:] = Vtt.iloc[-1,:];
        for t in np.arange(XtT.shape[0]-1,0,-1):
            vtt = np.reshape(np.mat(Vtt.iloc[t-1, :].values).T,(n,n))
            vttl = np.reshape(np.mat(Vtt.iloc[t, :].values).T,(n,n))
            vtT = np.reshape(np.mat(VtT.iloc[t, :].values).T,(n,n))
            j = vtt * self.U1 * vttl

            xtt = np.mat(Xtt.iloc[t-1, :].values).T
            xttl = np.mat(Xttl.iloc[t, :].values).T
            xtT = np.mat(XtT.iloc[t,:].values).T

            xtT = xtt + j * (xtT - xttl)
            vtT = vtt + j * (vtT - vttl) * j.T

            VtT.iloc[t-1, :] = (vtT.T).reshape(1,vtT.size)  # recording filterered variance along columns and dates along rows.
            Jt.iloc[t-1, :] = (j.T).reshape(1,j.size)
            XtT.iloc[t-1, :] = xtT.T

        return XtT, VtT, Jt