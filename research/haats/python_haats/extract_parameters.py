from __future__ import division
import sys
from scipy import integrate
from math import exp
from scipy.linalg import expm
import numpy as np
__author__ = 'ssylvain'


def prmtr_ext0(prmtr, num_states, Phi_prmtr, fix_Phi=1, setdiag_Kp=0):   #function to insert parameters for Phi
    if fix_Phi == 0: #here will not estimate Phi
        if setdiag_Kp==1: #here Kp is diagonal
            prmtr_ext = prmtr
            prmtr_ext = np.append(prmtr_ext[0], np.reshape(np.diag(prmtr_ext[1:num_states+1]),(num_states**2)))
            prmtr_ext = np.append(prmtr_ext, prmtr[num_states+1:])
        else:
            prmtr_ext = prmtr
    else:
        prmtr_ext = prmtr
        prmtr_ext = np.append(prmtr_ext[:-num_states-num_states], Phi_prmtr)
        prmtr_ext = np.append(prmtr_ext, prmtr[-num_states-num_states:])
        if setdiag_Kp==1:
            prmtr_ext2 = prmtr_ext
            prmtr_ext2 = np.append(prmtr_ext2[0], np.reshape(np.diag(prmtr_ext2[1:num_states+1]),(num_states**2)))
            prmtr_ext2 = np.append(prmtr_ext2, prmtr_ext[num_states+1:])
            prmtr_ext = prmtr_ext2
    return prmtr_ext


def param_mapping0(prmtr, num_states, US_num_maturities):
    prmtr_new = np.empty((prmtr.size)) * np.nan

    prmtr_new[0] = prmtr[0]     # a is unconstrained

    for vv in (np.arange(num_states**2)+1).tolist():     # Kp is unconstrained
        prmtr_new[vv] = prmtr[vv]

    prmtr_new[num_states**2+1] = 1e-16 - 1 + (prmtr[num_states**2+1]**2+1)**0.5     # lmda has lower bound 1e-16
    if num_states ==6:
        N6 = 1
        prmtr_new[num_states**2+1+N6] = 1e-16 - 1 + (prmtr[num_states**2+1+N6]**2+1)**0.5     # lmda has lower bound 1e-16
    else:
        N6 = 0

    for vv in (np.arange(US_num_maturities)+num_states**2+1+N6+1).tolist():     # Phi has lower bound 1e-16
        prmtr_new[vv] = 1e-16 - 1 + (prmtr[vv]**2+1)**0.5

    for vv in (np.arange(num_states)+US_num_maturities+num_states**2+1+N6+1).tolist():    # Sigma has lower bound 1e-16
        prmtr_new[vv] = 1e-16 - 1 + (prmtr[vv]**2+1)**0.5

    for vv in (np.arange(num_states)+num_states+US_num_maturities+num_states**2+1+N6+1).tolist():     # Theta is unconstrained
        prmtr_new[vv] = prmtr[vv]
    return prmtr_new


def param_mapping(prmtr, num_states, US_num_maturities):    #here we use use exp() to guarantee positivity
    prmtr_new = np.empty((prmtr.size)) * np.nan

    prmtr_new[0] = prmtr[0]     # a is unconstrained

    for vv in (np.arange(num_states**2)+1).tolist():     # Kp is unconstrained
        prmtr_new[vv] = prmtr[vv]

    prmtr_new[num_states**2+1] = np.exp(prmtr[num_states**2+1])     # lmda has lower bound 1e-16
    if num_states ==6:
        N6 = 1
        prmtr_new[num_states**2+1+N6] = np.exp(prmtr[num_states**2+1+N6])     # lmda has lower bound 1e-16
    else:
        N6 = 0

    for vv in (np.arange(US_num_maturities)+num_states**2+1+N6+1).tolist():     # Phi has lower bound 1e-16
        prmtr_new[vv] = np.exp(prmtr[vv])

    for vv in (np.arange(num_states)+US_num_maturities+num_states**2+1+N6+1).tolist():    # Sigma has lower bound 1e-16
        prmtr_new[vv] = np.exp(prmtr[vv])

    for vv in (np.arange(num_states)+num_states+US_num_maturities+num_states**2+1+N6+1).tolist():     # Theta is unconstrained
        prmtr_new[vv] = prmtr[vv]
    return prmtr_new


def extract_vars(prmtr, num_states, US_num_maturities):
    prmtr_new = param_mapping(prmtr, num_states, US_num_maturities)
    lst = prmtr_new.tolist()
    try:
        thetap = np.array([lst.pop() for pp in range(num_states)])[::-1]
        thetap = np.mat(thetap).T
        if num_states == 4:
            sigma44, sigma33, sigma22, sigma11 = (np.array(lst.pop()) for pp in range(num_states))
            Sigma = np.mat(np.diag([sigma11, sigma22, sigma33, sigma44]))
        elif num_states == 6:
            sigma44, sigma33_2, sigma33, sigma22_2, sigma22, sigma11 = (np.array(lst.pop()) for pp in range(num_states))
            Sigma = np.mat(np.diag([sigma11, sigma22, sigma22_2, sigma33, sigma33_2, sigma44]))
        Phi = np.mat(np.diag(np.array([lst.pop() for pp in range(US_num_maturities)])[::-1]))
        if num_states == 6:
            lmda2 = np.array(lst.pop())
        lmda = np.array(lst.pop())
        Kp = np.mat((np.array([lst.pop() for pp in range(num_states**2)])[::-1])).reshape(num_states, num_states)
        a = np.array(lst.pop())
    except:
        err_param = sys.exc_info()[0]
        print(err_param)
        if num_states == 4:
            a, Kp, lmda, Phi, sigma11, sigma22,sigma33, sigma44, Sigma, thetap = np.array(prmtr[0]), -np.mat(np.identity(num_states)),\
                                        -1, -np.mat(np.identity(US_num_maturities)), -1, -1, -1, -1, \
                                        -np.mat(np.identity(num_states)), thetap
        elif num_states == 6:
            a, Kp, lmda, lmda2, Phi, sigma11, sigma22, sigma22_2, sigma33, sigma33_2, sigma44, Sigma, thetap = np.array(prmtr[0]), -np.mat(np.identity(num_states)),\
                                        -1, -1, -np.mat(np.identity(US_num_maturities)), -1, -1, -1, -1,  -1, -1,\
                                        -np.mat(np.identity(num_states)), thetap
    if num_states == 4:
        return a, Kp, lmda, Phi, sigma11, sigma22, sigma33, sigma44, Sigma, thetap
    elif num_states == 6:
        return a, Kp, lmda, lmda2, Phi, sigma11, sigma22, sigma22_2, sigma33, sigma33_2, sigma44, Sigma, thetap


def a1(bondtype, maturity, prmtr, num_states, US_num_maturities):
    tau = maturity
    if num_states == 4:
        a, Kp, lmda, Phi, sigma11, sigma22, sigma33, sigma44, Sigma, thetap = extract_vars(prmtr, num_states, US_num_maturities)
        if bondtype == 'NominalBonds':
            out = np.float32(-(1 / tau) * np.array([-tau,
                                                -((1 - np.exp(-(lmda * tau))) / lmda),
                                                -((1 - np.exp(-(lmda * tau))) / lmda) + tau / np.exp(lmda * tau),
                                                0])
                         )

        elif bondtype == 'InfLinkBonds':
            out = np.float32(-(a / tau) * np.array([0,
                                                -((1 - np.exp(-(lmda * tau))) / lmda),
                                                -((1 - np.exp(-(lmda * tau))) / lmda) + tau / np.exp(lmda * tau),
                                                -tau / a])
                         )
    elif num_states == 6:
        a, Kp, lmda, lmda2, Phi, sigma11, sigma22, sigma22_2, sigma33, sigma33_2, sigma44, Sigma, thetap = extract_vars(prmtr, num_states, US_num_maturities)
        if bondtype == 'NominalBonds':
            out = np.float32(-(1 / tau) * np.array([-tau,
                                                -((1 - np.exp(-(lmda * tau))) / lmda),
                                                -((1 - np.exp(-(lmda2 * tau))) / lmda2),
                                                -((1 - np.exp(-(lmda * tau))) / lmda) + tau / np.exp(lmda * tau),
                                                -((1 - np.exp(-(lmda2 * tau))) / lmda2) + tau / np.exp(lmda2 * tau),
                                                0])
                         )

        elif bondtype == 'InfLinkBonds':
            out = np.float32(-(a / tau) * np.array([0,
                                                -((1 - np.exp(-(lmda * tau))) / lmda),
                                                -((1 - np.exp(-(lmda2 * tau))) / lmda2),
                                                -((1 - np.exp(-(lmda * tau))) / lmda) + tau / np.exp(lmda * tau),
                                                -((1 - np.exp(-(lmda2 * tau))) / lmda2) + tau / np.exp(lmda2 * tau),
                                                -tau / a])
                         )
    return out


def a0(bondtype, maturity, prmtr, num_states, US_num_maturities):
    tau = maturity
    if num_states == 4:
        a, Kp, lmda, Phi, sigma11, sigma22, sigma33, sigma44 , Sigma, thetap = extract_vars(prmtr, num_states, US_num_maturities)
        if bondtype == 'NominalBonds':
            out = np.float32(np.array([(-(sigma11 ** 2 * tau ** 2) / 6
                                    + (sigma22 ** 2 * (3 + np.exp(-2 * lmda * tau) - 4 / np.exp(lmda * tau) - 2 * lmda * tau)) / (
                                    4 * lmda ** 3 * tau)
                                    - (sigma33 ** 2 * (
                                    -5 - 6 * lmda * tau - 2 * lmda ** 2 * tau ** 2 + 8 * np.exp(lmda * tau) * (2 + lmda * tau) + np.exp(
                                    2 * lmda * tau) * (-11 + 4 * lmda * tau))) / (8 * np.exp(2 * lmda * tau) * lmda ** 3 * tau) )])
                         )

        elif bondtype == 'InfLinkBonds':
            out = np.float32(np.array([(-(sigma44 ** 2 * tau ** 2) / 6
                                    + (a ** 2 * sigma22 ** 2 * (3 + np.exp(-2 * lmda * tau) - 4 / np.exp(lmda * tau) - 2 * lmda * tau)) / (
                                    4 * lmda ** 3 * tau)
                                    - (a ** 2 * sigma33 ** 2 * (
                                    -5 - 6 * lmda * tau - 2 * lmda ** 2 * tau ** 2 + 8 * np.exp(lmda * tau) * (2 + lmda * tau) + np.exp(
                                     2 * lmda * tau) * (-11 + 4 * lmda * tau))) / (8 * np.exp(2 * lmda * tau) * lmda ** 3 * tau))])
                         )
    elif num_states == 6:
        a, Kp, lmda, lmda2, Phi, sigma11, sigma22, sigma22_2, sigma33, sigma33_2, sigma44, Sigma, thetap = extract_vars(prmtr, num_states, US_num_maturities)
        if bondtype == 'NominalBonds':
            out = np.float32(np.array([(-(sigma11 ** 2 * tau ** 2) / 6
                                    + (sigma22 ** 2 * (3 + np.exp(-2 * lmda * tau) - 4 / np.exp(lmda * tau) - 2 * lmda * tau)) / (
                                    4 * lmda ** 3 * tau)
                                    + (sigma22_2 ** 2 * (3 + np.exp(-2 * lmda2 * tau) - 4 / np.exp(lmda2 * tau) - 2 * lmda2 * tau)) / (
                                    4 * lmda2 ** 3 * tau)
                                    - (sigma33 ** 2 * (
                                    -5 - 6 * lmda * tau - 2 * lmda ** 2 * tau ** 2 + 8 * np.exp(lmda * tau) * (2 + lmda * tau) + np.exp(
                                    2 * lmda * tau) * (-11 + 4 * lmda * tau))) / (8 * np.exp(2 * lmda * tau) * lmda ** 3 * tau)
                                    - (sigma33_2 ** 2 * (
                                    -5 - 6 * lmda2 * tau - 2 * lmda2 ** 2 * tau ** 2 + 8 * np.exp(lmda2 * tau) * (2 + lmda2 * tau) + np.exp(
                                    2 * lmda2 * tau) * (-11 + 4 * lmda2 * tau))) / (8 * np.exp(2 * lmda2 * tau) * lmda2 ** 3 * tau)
                                       )])
                         )

        elif bondtype == 'InfLinkBonds':
            out = np.float32(np.array([(-(sigma44 ** 2 * tau ** 2) / 6
                                    + (a ** 2 * sigma22 ** 2 * (3 + np.exp(-2 * lmda * tau) - 4 / np.exp(lmda * tau) - 2 * lmda * tau)) / (
                                    4 * lmda ** 3 * tau)
                                    + (a ** 2 * sigma22_2 ** 2 * (3 + np.exp(-2 * lmda2 * tau) - 4 / np.exp(lmda2 * tau) - 2 * lmda2 * tau)) / (
                                    4 * lmda2 ** 3 * tau)
                                    - (a ** 2 * sigma33 ** 2 * (
                                    -5 - 6 * lmda * tau - 2 * lmda ** 2 * tau ** 2 + 8 * np.exp(lmda * tau) * (2 + lmda * tau) + np.exp(
                                     2 * lmda * tau) * (-11 + 4 * lmda * tau))) / (8 * np.exp(2 * lmda * tau) * lmda ** 3 * tau)
                                    - (a ** 2 * sigma33_2 ** 2 * (
                                    -5 - 6 * lmda2 * tau - 2 * lmda2 ** 2 * tau ** 2 + 8 * np.exp(lmda2 * tau) * (2 + lmda2 * tau) + np.exp(
                                     2 * lmda2 * tau) * (-11 + 4 * lmda2 * tau))) / (8 * np.exp(2 * lmda2 * tau) * lmda2 ** 3 * tau)
                                       )])
                         )
    return out


def q(dt, prmtr, num_states, US_num_maturities):
    if num_states == 4:
        a, Kp, lmda, Phi, sigma11, sigma22, sigma33, sigma44, Sigma, thetap = extract_vars(prmtr, num_states, US_num_maturities)
    elif num_states == 6:
        a, Kp, lmda, lmda2, Phi, sigma11, sigma22, sigma22_2, sigma33, sigma33_2, sigma44, Sigma, thetap = extract_vars(prmtr, num_states, US_num_maturities)
    qout = np.empty((num_states,num_states)) * np.nan
    tempfunc = lambda x, r, c: (expm(-Kp * x)*Sigma*(Sigma.transpose())*(expm(-Kp * x).transpose()))[r, c]
    for r in range(num_states):
        for c in range(num_states):
            qout[r, c] = integrate.quad(tempfunc, 0, dt, args=(r, c))[0]
    return qout


def extract_mats(prmtr, num_states, US_num_maturities, US_nominalmaturities, US_ilbmaturities, dt):
    if num_states == 4:
        a, Kp, lmda, Phi, sigma11, sigma22, sigma33, sigma44, Sigma, thetap = extract_vars(prmtr, num_states, US_num_maturities)
    elif num_states == 6:
        a, Kp, lmda, lmda2, Phi, sigma11, sigma22, sigma22_2, sigma33, sigma33_2, sigma44, Sigma, thetap = extract_vars(prmtr, num_states, US_num_maturities)
    A1N = np.array([a1('NominalBonds', mm, prmtr, num_states, US_num_maturities) for mm in US_nominalmaturities]).reshape(US_nominalmaturities.size, num_states)
    A1R = np.array([a1('InfLinkBonds', mm, prmtr, num_states, US_num_maturities) for mm in US_ilbmaturities]).reshape(US_ilbmaturities.size, num_states)
    A0N = np.array([a0('NominalBonds', mm, prmtr, num_states, US_num_maturities) for mm in US_nominalmaturities]).reshape(US_nominalmaturities.size, 1)
    A0R = np.array([a0('InfLinkBonds', mm, prmtr, num_states, US_num_maturities) for mm in US_ilbmaturities]).reshape(US_ilbmaturities.size, 1)
    A0 = np.mat(np.vstack((A0N, A0R)))
    A1 = np.mat(np.vstack((A1N, A1R)))
    U1 = np.mat(expm(-Kp*dt))
    U0 = np.mat((np.identity(num_states) - U1)*thetap)
    Q = q(dt, prmtr, num_states, US_num_maturities)
    return A0, A1, U0, U1, Q


def v0teqns(t, v0t, Kp, rho_n, rho_r, Sigma, thetap):
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