from __future__ import division
import sys
from scipy import integrate
from math import exp
from scipy.linalg import expm
import numpy as np
from numpy import exp
import itertools
# from IPython.core.debugger import Tracer; debug_here = Tracer()
# import pdb
# import ipdb
__author__ = 'ssylvain'


def build_prmtr_dict(prmtr, prmtr_size_dict):
    '''given prmtr and prmtr_size_dict, return the prmtr_dict'''
    prmtr_dict={}
    loc = 0
    for k in prmtr_size_dict.keys():
        prmtr_dict[k] = prmtr[loc:loc+prmtr_size_dict[k]]
        loc = loc+prmtr_size_dict[k]
    return prmtr_dict


def param_mapping(num_states, prmtr_dict, skip_mapping=0):
    '''here we use use exp() to map some variables and guarantee positivity'''
    if skip_mapping==0:
        prmtr_dict_new = prmtr_dict.copy()

        prmtr_dict_new['lmda'] = np.exp(prmtr_dict_new['lmda'])     # lmda has lower bound 0

        if num_states ==6:
            prmtr_dict_new['lmda2'] = np.exp(prmtr_dict_new['lmda2'])     # lmda2 has lower bound 0

        if 'Phi' in prmtr_dict_new.keys():
            prmtr_dict_new['Phi'] = np.exp(prmtr_dict_new['Phi']) # Phi has lower bound 0

        prmtr_dict_new['sigmas'] = np.exp(prmtr_dict_new['sigmas']) # Sigma has lower bound 1e-16

        # prmtr_dict_new['Kp'] = prmtr_dict_new['Kp']  #Kp is unconstrained
        # prmtr_dict_new['a'] = prmtr_dict_new['a']  # a is unconstrained
        # prmtr_dict_new['thetap'] =  prmtr_dict_new['theta'] # Theta is unconstrained

        #return list of parameters, need to upack the list
        return np.array(list(itertools.chain.from_iterable(prmtr_dict_new.values())))

    else: #here we skip doing the mapping
        return np.array(list(itertools.chain.from_iterable(prmtr_dict.values())))


def extract_vars(prmtr, num_states, prmtr_size_dict, Phi_prmtr=None, skip_mapping=0):
    '''assign parameter list to parameter names'''
    prmtr_dict_unmapped = build_prmtr_dict(prmtr, prmtr_size_dict)
    prmtr_mapped = param_mapping(num_states, prmtr_dict_unmapped, skip_mapping=skip_mapping)
    prmtr_dict = build_prmtr_dict(prmtr_mapped, prmtr_size_dict)

    a, Kp, lmda, sigmas, thetap = prmtr_dict['a'], prmtr_dict['Kp'], prmtr_dict['lmda'], prmtr_dict['sigmas'], prmtr_dict['thetap']

    if 'Phi' in prmtr_dict.keys():
        Phi = prmtr_dict['Phi']
        Phi = np.mat(np.diag(np.array(Phi)))
    else:
        try:
            Phi = np.mat(np.diag(np.exp(Phi_prmtr)))  #here just pad with pre-specified value
        except:
            Phi = None
            # print('Need to provided values for Phi since it is assumed to be fixed:', sys.exc_info()[0])
            # raise

    if 'lmda2' in prmtr_dict.keys():
        lmda2 = prmtr_dict['lmda2']
        lmda2 = np.array(lmda2)

    thetap = np.mat(np.array(thetap)).T

    if num_states == 4:
        sigma11, sigma22, sigma33, sigma44  = np.array(sigmas)
    else:
        sigma11, sigma22, sigma22_2, sigma33, sigma33_2, sigma44  = np.array(sigmas)
    Sigma = np.mat(np.diag(np.array(sigmas)))

    lmda = np.array(lmda)
    a = np.array(a)
    if len(Kp)==num_states:
        Kp = np.mat(np.diag(np.array(Kp)))
    else:
        Kp = np.mat(np.array(Kp)).reshape(num_states, num_states)

    if num_states == 4:
        return a, Kp, lmda, Phi, sigma11, sigma22, sigma33, sigma44, Sigma, thetap
    elif num_states == 6:
        return a, Kp, lmda, lmda2, Phi, sigma11, sigma22, sigma22_2, sigma33, sigma33_2, sigma44, Sigma, thetap


def a1(bondtype, maturity, prmtr, num_states, prmtr_size_dict,skip_mapping=0):
    '''building A_1 matrix of parameters'''
    tau = maturity
    if num_states == 4:
        a, Kp, lmda, Phi, sigma11, sigma22, sigma33, sigma44, Sigma, thetap = extract_vars(prmtr, num_states, prmtr_size_dict, skip_mapping=skip_mapping)
        if bondtype == 'NominalBonds':
            out = np.float32(-(1.0 / tau) * np.array([-tau,
                                                    -1/lmda + exp(-lmda*tau)/lmda,
                                                    (lmda*tau - exp(lmda*tau) + 1)*exp(-lmda*tau)/lmda,
                                                    0])
                         )

        elif bondtype == 'InfLinkBonds':
            out = np.float32(-(1.0 / tau) * np.array([0,
                                                    -a/lmda + a*exp(-lmda*tau)/lmda,
                                                    a*(lmda*tau - exp(lmda*tau) + 1)*exp(-lmda*tau)/lmda,
                                                    -tau])
                         )
    elif num_states == 6:
        a, Kp, lmda, lmda2, Phi, sigma11, sigma22, sigma22_2, sigma33, sigma33_2, sigma44, Sigma, thetap = extract_vars(prmtr, num_states, prmtr_size_dict, skip_mapping=skip_mapping)
        if bondtype == 'NominalBonds':
            out = np.float32(-(1.0 / tau) * np.array([-tau,
                                                    -1/lmda + exp(-lmda*tau)/lmda,
                                                    -1/lmda2 + exp(-lmda2*tau)/lmda2,
                                                    (lmda*tau - exp(lmda*tau) + 1)*exp(-lmda*tau)/lmda,
                                                    (lmda2*tau - exp(lmda2*tau) + 1)*exp(-lmda*tau)/lmda2,
                                                    0])
                         )

        elif bondtype == 'InfLinkBonds':
            out = np.float32(-(1.0 / tau) * np.array([0,
                                                    -a/lmda + a*exp(-lmda*tau)/lmda,
                                                    -a/lmda2 + a*exp(-lmda2*tau)/lmda2,
                                                    a*(lmda*tau - exp(lmda*tau) + 1)*exp(-lmda*tau)/lmda,
                                                    a*(lmda2*tau - exp(lmda2*tau) + 1)*exp(-lmda2*tau)/lmda2,
                                                    -tau])
                         )
    return out


def a0(bondtype, maturity, prmtr, num_states, prmtr_size_dict,skip_mapping=0):
    '''building A_0 matrix of parameters'''
    tau = maturity
    if num_states == 4:
        a, Kp, lmda, Phi, sigma11, sigma22, sigma33, sigma44 , Sigma, thetap = extract_vars(prmtr, num_states, prmtr_size_dict, skip_mapping=skip_mapping)
        if bondtype == 'NominalBonds':
            out = np.float32(-(1.0 / tau) * np.array([sigma11**2*tau**3/6
                                                      + sigma22**2*(tau/(2*lmda**2)
                                                                    + exp(-lmda*tau)/lmda**3
                                                                    - exp(-2*lmda*tau)/(4*lmda**3))
                                                      + sigma33**2*(-tau**2*exp(-2*lmda*tau)/(4*lmda)
                                                                    + tau/(2*lmda**2)
                                                                    + tau*exp(-lmda*tau)/lmda**2
                                                                    - 3*tau*exp(-2*lmda*tau)/(4*lmda**2)
                                                                    + 2*exp(-lmda*tau)/lmda**3
                                                                    - 5*exp(-2*lmda*tau)/(8*lmda**3))
                                                      ])
                             )

        elif bondtype == 'InfLinkBonds':
            out = np.float32(-(1.0 / tau) * np.array([sigma22**2*(a**2*tau/(2*lmda**2)
                                                                  + a**2*exp(-lmda*tau)/lmda**3
                                                                  - a**2*exp(-2*lmda*tau)/(4*lmda**3))
                                                      + sigma33**2*(-a**2*tau**2*exp(-2*lmda*tau)/(4*lmda)
                                                                    + a**2*tau/(2*lmda**2)
                                                                    + a**2*tau*exp(-lmda*tau)/lmda**2
                                                                    - 3*a**2*tau*exp(-2*lmda*tau)/(4*lmda**2)
                                                                    + 2*a**2*exp(-lmda*tau)/lmda**3
                                                                    - 5*a**2*exp(-2*lmda*tau)/(8*lmda**3))
                                                      + sigma44**2*tau**3/6])
                             )
    elif num_states == 6:
        a, Kp, lmda, lmda2, Phi, sigma11, sigma22, sigma22_2, sigma33, sigma33_2, sigma44, Sigma, thetap = extract_vars(prmtr, num_states, prmtr_size_dict, skip_mapping=skip_mapping)
        if bondtype == 'NominalBonds':
            out = np.float32(-(1.0 / tau) * np.array([sigma11**2*tau**3/6
                                                      + sigma22**2*(tau/(2*lmda**2)
                                                                    + exp(-lmda*tau)/lmda**3
                                                                    - exp(-2*lmda*tau)/(4*lmda**3))
                                                      + sigma22_2**2*(tau/(2*lmda2**2)
                                                                      + exp(-lmda2*tau)/lmda2**3
                                                                      - exp(-2*lmda2*tau)/(4*lmda2**3))
                                                      + sigma33**2*(-tau**2*exp(-2*lmda*tau)/(4*lmda) + tau/(2*lmda**2)
                                                                    + tau*exp(-lmda*tau)/lmda**2
                                                                    - 3*tau*exp(-2*lmda*tau)/(4*lmda**2)
                                                                    + 2*exp(-lmda*tau)/lmda**3
                                                                    - 5*exp(-2*lmda*tau)/(8*lmda**3))
                                                      + sigma33_2**2*(-tau**2*exp(-2*lmda2*tau)/(4*lmda2)
                                                                      + tau/(2*lmda2**2) + tau*exp(-lmda2*tau)/lmda2**2
                                                                    - 3*tau*exp(-2*lmda2*tau)/(4*lmda2**2)
                                                                      + 2*exp(-lmda2*tau)/lmda2**3
                                                                      - 5*exp(-2*lmda2*tau)/(8*lmda2**3))
                                                      ])
                             )

        elif bondtype == 'InfLinkBonds':
            out = np.float32(-(1.0 / tau) * np.array([sigma22**2*(a**2*tau/(2*lmda**2)
                                                                  + a**2*exp(-lmda*tau)/lmda**3
                                                                  - a**2*exp(-2*lmda*tau)/(4*lmda**3))
                                                      + sigma22_2**2*(a**2*tau/(2*lmda2**2)
                                                                      + a**2*exp(-lmda2*tau)/lmda2**3
                                                                      - a**2*exp(-2*lmda2*tau)/(4*lmda2**3))
                                                      + sigma33**2*(-a**2*tau**2*exp(-2*lmda*tau)/(4*lmda)
                                                                    + a**2*tau/(2*lmda**2)
                                                                    + a**2*tau*exp(-lmda*tau)/lmda**2
                                                                    - 3*a**2*tau*exp(-2*lmda*tau)/(4*lmda**2)
                                                                    + 2*a**2*exp(-lmda*tau)/lmda**3
                                                                    - 5*a**2*exp(-2*lmda*tau)/(8*lmda**3))
                                                      + sigma33_2**2*(-a**2*tau**2*exp(-2*lmda2*tau)/(4*lmda2)
                                                                      + a**2*tau/(2*lmda2**2)
                                                                    + a**2*tau*exp(-lmda2*tau)/lmda2**2
                                                                      - 3*a**2*tau*exp(-2*lmda2*tau)/(4*lmda2**2)
                                                                    + 2*a**2*exp(-lmda2*tau)/lmda2**3
                                                                      - 5*a**2*exp(-2*lmda2*tau)/(8*lmda2**3))
                                                      + sigma44**2*tau**3/6])
                             )
    return out


def q(dt, prmtr, num_states, prmtr_size_dict,skip_mapping=0):
    '''computing Q matrix'''
    if num_states == 4:
        a, Kp, lmda, Phi, sigma11, sigma22, sigma33, sigma44, Sigma, thetap = extract_vars(prmtr, num_states, prmtr_size_dict, skip_mapping=skip_mapping)
    elif num_states == 6:
        a, Kp, lmda, lmda2, Phi, sigma11, sigma22, sigma22_2, sigma33, sigma33_2, sigma44, Sigma, thetap = extract_vars(prmtr, num_states, prmtr_size_dict, skip_mapping=skip_mapping)
    qout = np.empty((num_states,num_states)) * np.nan
    tempfunc = lambda x, r, c: (expm(-Kp * x)*Sigma*(Sigma.transpose())*(expm(-Kp * x).transpose()))[r, c]
    for r in range(num_states):
        for c in range(num_states):
            qout[r, c] = integrate.quad(tempfunc, 0, dt, args=(r, c))[0]
    return qout


def extract_mats(prmtr, num_states, US_nominalmaturities, US_ilbmaturities, dt, prmtr_size_dict, Phi_prmtr,skip_mapping=0):
    '''building all needed matricees of parameters'''
    if num_states == 4:
        a, Kp, lmda, Phi, sigma11, sigma22, sigma33, sigma44, Sigma, thetap = extract_vars(prmtr, num_states, prmtr_size_dict, Phi_prmtr, skip_mapping=skip_mapping)
    elif num_states == 6:
        a, Kp, lmda, lmda2, Phi, sigma11, sigma22, sigma22_2, sigma33, sigma33_2, sigma44, Sigma, thetap = extract_vars(prmtr, num_states, prmtr_size_dict, Phi_prmtr, skip_mapping=skip_mapping)
    A1N = np.array([a1('NominalBonds', mm, prmtr, num_states, prmtr_size_dict, skip_mapping=skip_mapping) for mm in US_nominalmaturities]).reshape(US_nominalmaturities.size, num_states)
    A1R = np.array([a1('InfLinkBonds', mm, prmtr, num_states, prmtr_size_dict, skip_mapping=skip_mapping) for mm in US_ilbmaturities]).reshape(US_ilbmaturities.size, num_states)
    A0N = np.array([a0('NominalBonds', mm, prmtr, num_states, prmtr_size_dict, skip_mapping=skip_mapping) for mm in US_nominalmaturities]).reshape(US_nominalmaturities.size, 1)
    A0R = np.array([a0('InfLinkBonds', mm, prmtr, num_states, prmtr_size_dict, skip_mapping=skip_mapping) for mm in US_ilbmaturities]).reshape(US_ilbmaturities.size, 1)
    A0 = np.mat(np.vstack((A0N, A0R)))
    A1 = np.mat(np.vstack((A1N, A1R)))
    U1 = np.mat(expm(-Kp*dt))
    U0 = np.mat((np.identity(num_states) - U1)*thetap)
    Q = q(dt, prmtr, num_states, prmtr_size_dict, skip_mapping=skip_mapping)
    return A0, A1, U0, U1, Q, Phi


def v0teqns(t, v0t, Kp, rho_n, rho_r, Sigma, thetap):
    '''Variance ODE for expected inflation calculation'''
    K = np.mat(np.vstack((np.hstack((Kp, np.zeros((Kp.shape[0],1)))), np.zeros((1,Kp.shape[0]+1))))) * np.mat(np.vstack((thetap, 0)))
    theta = np.mat(np.vstack((np.hstack((-Kp, np.zeros((Kp.shape[0],1)))), np.hstack(((rho_n-rho_r).T, np.zeros((1,1))))     ))  )
    Sigma_bar = np.mat(np.vstack((np.hstack((Sigma, np.zeros((Sigma.shape[0],1)))), np.zeros((1, Sigma.shape[0]+1)) )))
    Inn = np.mat(np.identity(Sigma.shape[0]+1))

    # re-sizing correctly. this only matters if finding moments for X
    K = K[0:v0t.size**0.5]
    theta = theta[0:v0t.size**0.5, 0:v0t.size**0.5]
    Sigma_bar = Sigma_bar[0:v0t.size**0.5, 0:v0t.size**0.5]
    Inn = Inn[0:v0t.size**0.5, 0:v0t.size**0.5]

    v = np.reshape(v0t, (v0t.size**0.5, v0t.size**0.5)).T
    dv = theta*np.mat(v) +np.mat(v)*theta.T +Sigma_bar*Sigma_bar.T
    dv0tdt = np.reshape(dv.T,(dv.shape[0]**2,1))
    return np.array(dv0tdt)


def m0teqns(t, m0t, Kp, rho_n, rho_r, Sigma, thetap):
    '''Mean ODE for expected inflation calculation'''
    K = np.mat(np.vstack((np.hstack((Kp, np.zeros((Kp.shape[0],1)))), np.zeros((1,Kp.shape[0]+1))))) * np.mat(np.vstack((thetap, 0)))
    theta = np.mat(np.vstack((np.hstack((-Kp, np.zeros((Kp.shape[0],1)))), np.hstack(((rho_n-rho_r).T, np.zeros((1,1))))     ))  )
    Sigma_bar = np.mat(np.vstack((np.hstack((Sigma, np.zeros((Sigma.shape[0],1)))), np.zeros((1, Sigma.shape[0]+1)) )))
    Inn = np.mat(np.identity(Sigma.shape[0]+1))

    # re-sizing correctly. this only matters if finding moments for X
    K = K[0:m0t.size]
    theta = theta[0:m0t.size, 0:m0t.size]
    Sigma_bar = Sigma_bar[0:m0t.size, 0:m0t.size]
    Inn = Inn[0:m0t.size, 0:m0t.size]

    dm0tdt = K + theta*np.mat(m0t).T
    return np.array(dm0tdt)