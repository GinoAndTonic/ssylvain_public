from __future__ import division
from scipy.linalg import expm
from scipy import integrate
from math import exp
import sys
import numpy as np
# import numpy.ma as ma
from scipy.linalg import solve_discrete_lyapunov
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)   #need to install miktex
plt.close("all")
plt.close()
plt.ion()
__author__ = 'ssylvain'

# REQUIRED INPUTS:
# 1) the measurement: Y, it is a (T x m) matrix where each row is an observation and
#                 each column is a different variable
# 2-5) the parameters: A0 , A1 , U0, U1, Q, Phi are such that
#                 Y{t+1} = A0 + A1 X{t}' + u{t+1}  : Measurement Equation
#                 X{t} = U0 + U1 X{t-1}' + e{t}    : State Equation
#                 here Y{t+1} and X{t} are column vectors of size m and n
#                 respectively.
#                 Phi = cov(u{t+1})
#                 Q = cov(e{t})
# OPTIONAL INPUTS:
# 6) initV specifies how to initialize the error variance V0:
#     initV='steady_state' or initV='unconditional'(default) or initV='identity' matrix
# 7) The initial state X0, otherwise the unconditional mean is used,
#    X0 =U0/(eye() - U1)
# 8) The initial error variance V0. If we specify V0 directly, it will overwrite
# initV.

# Thus this filter is flexible enough to allow for multi-dimension states and
# measurements.

# REQUIRED OUTPUTS:
# 1-2) The filtered and conditional measurements, Ytt and Yttl which are a T x M matrix
#         Ytt{t+1} = A0 + A1 Xtt{t}'
#         Yttl{t+1} = A0 + A1 Xttl{t}'
# 3) The filtered states, Xtt which is a T x n matrix,
# 4) The conditional filtered states Xttl which is a T x n matrix
# 5) The filtered error variance, Vtt which is a T x n^2 matrix,
# 6) The conditional error variance Vttl which is a T x n^2 matrix
# 7) The Kalman Gains, Gain_t
# 8) The cumulative log likelihood: cum_log_likelihood which is a scalar.

# ALL VECTORS ARE COLUMN VECTORS.
# ALL ARRAYS SHOULD BE NUMPY ARRAYS.
#


class Kalman:  # define super-class
    kalmanCount = 0

    def __init__(self, Y, A0, A1, U0, U1, Q, Phi, initV='unconditional', X0 = None, V0 = None):
        self.Y, self.A0, self.A1, self.U0, self.U1, self.Q, self.Phi, self.initV, self.X0, self.V0 = Y, A0, A1, U0, U1, \
                                                                                                     Q, Phi, initV, X0, V0
        n = self.U0.size
        if self.X0 is None:
            self.X0 = np.mat(np.linalg.inv(np.identity(n) - self.U1)*self.U0)
        if self.V0 is None:
            if initV == 'steady_state':  # solve discrete Ricatti equation
                self.V0 = np.mat(solve_discrete_are(np.array(self.U1.T), np.array(self.A1.T), np.array(self.Q), np.array(self.Phi.T)))
            elif initV == 'unconditional':  # solve discrete Lyapunov equation
                self.V0 = np.mat(solve_discrete_lyapunov(np.array(self.U1.T), np.array(self.Q)))
            elif initV == 'identity':
                self.V0 = np.mat(np.identity(n))

        Kalman.kalmanCount += 1
        # print("Calling Kalman constructor")

    def __del__(self):
        class_name = self.__class__.__name__
        # print(class_name, "destroyed")

    def filter(self):
        # masking missing observation so as to not affect calculations
        # self.Y = ma.masked_array(self.Y, mask = np.array(self.Y == np.nan))
        try:
            T = self.Y.shape[0]
            m = self.Y.shape[1]
        except:
            err_shape = sys.exc_info()[0]
            T = self.Y.size
            m = 1
        n = self.U0.size
        cum_log_likelihood = 0
        Ytt, Yttl = (np.empty((T, m)) * np.nan for vv in range(2))
        Xtt, Xttl = (np.empty((T, n)) * np.nan for vv in range(2))
        Vtt, Vttl = (np.empty((T, n**2)) * np.nan for vv in range(2))
        Gain_t = np.empty((T, n*m)) * np.nan
        eta_t = np.empty((T, n)) * np.nan

        for t in range(T):
            y = self.Y[t, :].T
            # in case there are missing data, we will remove them:
            # m_ind = (y != np.nan)
            # m2 = min(m, sum(m_ind))
            # y, A0, A1, U0, U1, Phi, Q, X0, V0, initV = y[m_ind].T, self.A0[m_ind].T, \
            #                                     self.A1[np.tile(m_ind,(1,n))].reshape(m2,n), self.U0, self.U1,\
            #                                     self.Phi[np.tile(m_ind,(1,m2))].reshape(m2,m2), self.Q, self.X0, self.V0, self.initV

            m_ind = np.array((y != np.nan)).T[0,:]
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

            # STEP3: Calculate likelihood and store results
            if np.linalg.det(S) < 0:     # this is to avoid complex numbers from log(det(S))
                cum_log_likelihood = -np.inf
                # return np.mat(Ytt), np.mat(Yttl), np.mat(Xtt), np.mat(Xttl), np.mat(Vtt), np.mat(Vttl), np.mat(Gain_t), cum_log_likelihood
                break
            else:
                increment = -(y - A0 - A1 * xttl).T * \
                            np.linalg.inv(S)*(y - A0 - A1 * xttl)/2  \
                            - (y.size/2) * np.log(2 * np.pi) - (1/2) * np.log(np.linalg.det(S))

            # meed to burn first 2 observations in case initialization is introducing too much bias:
            #if t > 1:
            cum_log_likelihood += increment

            Ytt[t, m_ind] = (A0 + A1 * xtt).T
            Yttl[t, m_ind] = (A0 + A1 * xttl).T

            Xtt[t, :] = xtt.T       # recording filterered states along columns and dates along rows.
            Vtt[t, :] = (vtt.T).reshape(1, vtt.size)  # recording filterered variance along columns and dates along rows.

            Xttl[t, :] = xttl.T  # recording filtered states along columns and dates along rows.
            Vttl[t, :] = (vttl.T).reshape(1, vttl.size) # recording filterered variance along columns and dates along rows.

            Gain0 = np.empty((n, m)) * np.nan
            Gain0[:, m_ind] = Gain
            Gain_t[t, :] = Gain0.reshape(1, n*m)

            if t==0:
                eta_t[t, :] = (U0 + U1 * X0 - X0).T
            else:
                eta_t[t, :] = (U0 + U1 * xtt - np.mat(Xtt[t-1, :]).T).T
        # [XttT, VttT, VttlT, Jt, V0T, X0T] = smoother(Xtt, Xttl, Vtt, Vttl, A1, Gain_t, U1, V0, X0, Q)
        # v0T = V0T.reshape(n, n)
        # cum_log_likelihood += (X0-X0T).T*np.linalg.inv(v0T)*(X0-X0T)/2 - (n/2)*np.log(2*np.pi) -(1/2)*np.log(np.linalg.det(v0T))   # adding date zero likelihood

        # remove burn in periods:
        # Ytt[0:1,:] = np.nan
        # Yttl[0:1,:] = np.nan

        return np.mat(Ytt), np.mat(Yttl), np.mat(Xtt), np.mat(Xttl), np.mat(Vtt), np.mat(Vttl), np.mat(Gain_t), \
               np.mat(eta_t), np.reshape(np.array(cum_log_likelihood), 1, 0)

    def forecast(self, X, horizon=90):
        y_out = np.empty((X.shape[0], self.A0.size, horizon+1)) * np.nan
        for t in np.arange(X.shape[0]):     # loop over dates
            for h in np.arange(horizon):     # loop over horizons
                    if h == 0:
                        x = self.U0 + self.U1 * X[t, :].T
                    else:
                        x = self.U0 + self.U1 * x
                    y_out[t, :, h+1] = np.array(self.A0 + self.A1 * x)[:,0]
        y_out[:, :, 0] = self.Y
        return np.array(y_out)

    def rmse(self, y_fcst, m):  #RMSE calculations
        horizon = y_fcst.shape[1]
        T = y_fcst.shape[0]
        forecast_se = np.zeros((T-horizon, horizon))
        forecast_e = np.zeros((T-horizon, horizon))
        for t in range(T-horizon):
            forecast_se[t, :] = np.array(np.mat(self.Y[t:t+horizon, m]).T - np.mat(y_fcst[t, :]))**2
            forecast_e[t, :] = (np.mat(self.Y[t:t+horizon, m]).T - np.mat(y_fcst[t, :]))
        forecast_rmse = np.mean(forecast_se, axis=0)**0.5
        return np.mat(forecast_e), np.mat(forecast_se), np.mat(forecast_rmse).T

    def smoother(self, Xtt, Xttl, Vtt, Vttl, A1, Gain_t, eta_t, U1, V0, X0, Q):
        #n = U0.size
        #T = Xtt.shape[0]

        #XtT = Xtt * np.nan
        #VtT = Vtt * np.nan
        #VtlT = Vtt * np.nan

        #XtT[-1,:] = Xtt[-1,:];
        #VtT[-1,:] = Vtt[-1,:];

        #vtt_lag_temp=(np.reshape(Vtt[-1,:].T)(,n,n))).T;
        #vtlT = (np.identity(n)-Gain_t[-1,:].T*A1)*U1*vtt_lag_temp;
        #VtlT(end,:) = reshape(vtlT',1,n^2);

        #v0=V0;
        #v10=reshape(Vttl_ts(1,:)',n,n )';
        #j0=v0*U1'*(v10^(-1));

        #Jt=[];
        #for tt=1:size(Xtt_ts,1)-1
            #vtt=reshape(Vtt_ts(tt,:)',n,n )';
            #vttl=reshape(Vttl_ts(tt+1,:)',n,n )';
            #jt=vtt*U1'*(vttl^(-1));
            #Jt= [Jt;reshape(jt',1,numel(jt))];

        #for tt=size(Xtt_ts,1)-1:-1:1
            #vtt=reshape(Vtt_ts(tt,:)',n,n )';
            #vttl=reshape(Vttl_ts(tt+1,:)',n,n )';
            #jt=reshape(Jt(tt,:)',n,n)';

            #xtT=Xtt_ts(tt,:)'+jt*(XtT(tt+1,:)'-Xttl_ts(tt+1,:)');
            #vtT=vtt'+jt*(reshape(VtT(tt+1,:)',n,n)'-vttl')*jt';
            #XtT(tt,:) = xtT';
            #VtT(tt,:) = reshape(vtT',1,n^2);

            #if tt>1
                #vtlT=vtt*(reshape(Jt(tt-1,:)',n,n)')'+jt*(reshape(VtlT(tt+1,:)',n,n )'-U1*vtt)*((reshape(Jt(tt-1,:)',n,n)')');
            #else
                #vtlT=vtt*(j0)'+jt*(reshape(VtlT(tt+1,:)',n,n )'-U1*vtt)*(j0');

            #VtlT(tt,:)= reshape(vtlT',1,n^2);


            #v0T=v0+j0*(reshape(VtT(1,:)',n,n)' - reshape(Vttl_ts(1,:)',n,n)' )*(j0');
            #V0T=reshape(v0T',1,n^2);
            #X0T=X0 +j0*(XtT(1,:)'-Xttl(1,:)');
        print('need to write smoother code')

    def plot(self,Ytt, Yttl, Xtt, Xttl, Vtt, Vttl, Gain_t, eta_t, figures,US_ilbmaturities, US_nominalmaturities):
        for vv in ['Ytt', 'Yttl', 'Xtt', 'Xttl', 'Vtt', 'Vttl', 'Gain_t', 'eta_t']:
            plt.close()
            fig, ax = plt.subplots()
            ax.plot(eval(vv))
            ax.set_title(vv)
            plt.draw()
            figures[vv] = fig
            figures['ax_'+vv] = ax
            figures[vv+'_name'] = '\\vv'
            filename = r"S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python" + \
                str(figures[vv+'_name']) + ".png"
            # plt.savefig(filename, format="png")
        plt.close()
        fig, ax = plt.subplots()
        ax.plot(self.Y)
        ax.set_title('Y')
        figures['Y'] = fig
        figures['ax_Y'] = ax
        figures['Y_name'] = '\\Y'
        plt.draw()

        plt.close()
        fig, ax = plt.subplots(2,sharex=True)
        ax[0].plot(self.Y[:,range(US_nominalmaturities.size)],\
                   label=[r'mat: '+str(US_nominalmaturities[vvv]) for vvv in range(US_nominalmaturities.size)])
        plt.gca().set_color_cycle(None)     # reset color cycle
        ax[0].plot(Ytt[:,range(US_nominalmaturities.size)], '--',\
                   label=[r'mat: '+str(US_nominalmaturities[vvv]) for vvv in range(US_nominalmaturities.size)])
        ax[0].set_title('Realized and Model Nominal Yields')
        # handles, labels = figures['ax_YvYtt'].get_legend_handles_labels()
        # figures['ax_YvYtt'].legend(handles, [r'mat: '+str(np.hstack((US_maturities,US_maturities))[vvv]) for vvv in range(Y.shape[1])])
        ax[1].plot(self.Y[:,US_nominalmaturities.size+np.arange(US_ilbmaturities.size)],\
                   label=[r'mat: '+str(US_ilbmaturities[vvv]) for vvv in range(US_ilbmaturities.size)])
        plt.gca().set_color_cycle(None)     # reset color cycle
        ax[1].plot(Ytt[:,US_nominalmaturities.size+np.arange(US_ilbmaturities.size)], '--',\
                   label=[r'mat: '+str(US_ilbmaturities[vvv]) for vvv in range(US_ilbmaturities.size)])
        ax[1].set_title('Realized and Model ILB Yields')
        figures['YvYtt'] = fig
        figures['ax_YvYtt'] = ax
        figures['YvYtt_name'] = '\\YvYtt'
        plt.draw()

        plt.close()
        fig, ax = plt.subplots(2, sharex=True)
        ax[0].plot(self.Y[:,range(US_nominalmaturities.size)] - Ytt[:,range(US_nominalmaturities.size)],\
                   label=[r'mat: '+str(US_nominalmaturities[vvv]) for vvv in range(US_nominalmaturities.size)])
        ax[0].set_title('Realized vs. Model Nominal Yields')
        ax[1].plot(self.Y[:,US_nominalmaturities.size+np.arange(US_ilbmaturities.size)] - \
                   Ytt[:,US_nominalmaturities.size+np.arange(US_ilbmaturities.size)],\
                   label=[r'mat: '+str(US_ilbmaturities[vvv]) for vvv in range(US_ilbmaturities.size)])
        ax[1].set_title('Realized vs. Model ILB Yields')
        figures['Y_min_Ytt'] = fig
        figures['ax_Y_min_Ytt'] = ax
        figures['Y_min_Ytt_name'] = '\\Y_min_Ytt'
        plt.draw()

        thetap = np.mat(np.linalg.inv(np.identity(self.U0.size) - self.U1)*self.U0)
        thetap_vec = np.tile(thetap.T, (Xtt.shape[0], 1))

        plt.close()
        fig, ax = plt.subplots(2, sharex=False)
        ax[0].plot(Xtt, label=[""])
        ax[0].set_color_cycle(None)     # reset color cycle
        ax[0].plot(thetap_vec, '--', label=[""])
        ax[0].set_title('State Variables')
        handles, labels = ax[0].get_legend_handles_labels()
        if Xtt.shape[1] == 4:
            ax[0].legend(handles, [r'L^N_t', r'S_t', r'C_t', r'L^R_t', r'\theta_1', r'\theta_2', r'\theta_3', r'\theta_4'])
        elif Xtt.shape[1] == 5:
            ax[0].legend(handles, [r'L^N_t', r'S_t', r'C_t', r'L^R_t', r'\xi_t', r'\theta_1', r'\theta_2', r'\theta_3', r'\theta_4', r'\theta_5'])
        else:
            ax[0].legend(handles, [r'L^N_t', r'S^{(1)}_t', r'S^{(2)}_t', r'C^{(1)}_t', r'C^{(2)}_t', r'L^R_t', r'\theta_1', r'\theta_2', r'\theta_3', r'\theta_4', r'\theta_5', r'\theta_6'])
        ax[1].plot(eta_t)
        ax[1].set_title('State Variables Stochastic Error (\eta_t)')
        figures['XttvThetap'] = fig
        figures['ax_XttvThetap'] = ax
        figures['XttvThetap_name'] = '\\XttvThetap'
        plt.draw()
