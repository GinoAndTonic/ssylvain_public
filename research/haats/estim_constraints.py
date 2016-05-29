from __future__ import division
import datetime as DT
import matplotlib.dates as mdates
import time
import numpy as np
# import numpy.ma as ma
import matplotlib.pyplot as plt
import sys
print(sys.argv)
from asset_class import *
from kalman import *
from extract_parameters import *
from math import exp
from scipy.optimize import minimize
from scipy.linalg import expm
from scipy import integrate
# from  matplotlib.pyplot import ion
# ion()
plt.close("all")
__author__ = 'ssylvain'


# constraints:
def eigvalConstraint(prmtrC, ij, num_states, prmtr_size_dict, Phi_prmtr):    # need to make sure real part of eigvals of Kp are positive
    if num_states == 4:
        a, Kp, lmda, Phi, sigma11_raw, sigma22, sigma33, sigma44_raw, Sigma, thetap = extract_vars(prmtrC, num_states, prmtr_size_dict, Phi_prmtr)
    elif num_states == 6:
        a, Kp, lmda, lmda2, Phi, sigma11, sigma22, sigma22_2, sigma33, sigma33_2, sigma44, Sigma, thetap = extract_vars(prmtrC, num_states, prmtr_size_dict, Phi_prmtr)
    Kp_eigvals, Kp_eigvecs = np.linalg.eig(np.array(Kp))
    return np.real(Kp_eigvals)[ij]


def boundedVolandLmda(prmtrC, ij, num_states, prmtr_size_dict, Phi_prmtr):  # need to make sure volatilities and lmda are positive and < inf
    if num_states == 4:
        a, Kp, lmda, Phi, sigma11_raw, sigma22, sigma33, sigma44_raw, Sigma, thetap = extract_vars(prmtrC, num_states, prmtr_size_dict, Phi_prmtr)
    elif num_states == 6:
        a, Kp, lmda, lmda2, Phi, sigma11, sigma22, sigma22_2, sigma33, sigma33_2, sigma44, Sigma, thetap = extract_vars(prmtrC, num_states, prmtr_size_dict, Phi_prmtr)
    nonegVol = np.vstack((np.diag(np.array(Phi)), np.diag(np.array(Sigma)), lmda))-1e-16
    return nonegVol[ij]


def prmtr_bounds_lvl(num_states, US_num_maturities):    # here we have to  impose non-negativity
    bds = [(-1e16, 1e16)]     # a
    for vv in range(num_states**2):     # Kp
        bds.append((-1e16, 1e16))
    bds.append((1e-16, 1e16))     # lmda
    for vv in range(US_num_maturities):     # Phi
        bds.append((1e-16, 1e16))
    for vv in range(num_states):     # Sigma
        bds.append((1e-16, 1e16))
    for vv in range(num_states):     # Theta
        bds.append((-1e16, 1e16))
    return bds


def prmtr_bounds_exp(num_states, US_num_maturities):    # here we have already imposed non-negativity by using exp()
    bds = [(-1e16, 1e16)]     # a
    for vv in range(num_states**2):     # Kp
        bds.append((-1e16, 1e16))
    bds.append((-1e-16, np.log(1e16)))     # lmda
    for vv in range(US_num_maturities):     # Phi
        bds.append((-1e-16, np.log(1e16)))
    for vv in range(num_states):     # Sigma
        bds.append((-1e-16, np.log(1e16)))
    for vv in range(num_states):     # Theta
        bds.append((-1e16, 1e16))
    return bds

def ineq_cons(num_states, prmtr_size_dict, Phi_prmtr): #constraint for Kp
    const_temp = []
    const_temp.append({'type': 'ineq', 'fun': lambda prmtr:  np.array(eigvalConstraint(prmtr, 0, num_states, prmtr_size_dict, Phi_prmtr))})
    const_temp.append({'type': 'ineq', 'fun': lambda prmtr:  np.array(eigvalConstraint(prmtr, 1, num_states, prmtr_size_dict, Phi_prmtr))})
    const_temp.append({'type': 'ineq', 'fun': lambda prmtr:  np.array(eigvalConstraint(prmtr, 2, num_states, prmtr_size_dict, Phi_prmtr))})
    const_temp.append({'type': 'ineq', 'fun': lambda prmtr:  np.array(eigvalConstraint(prmtr, 3, num_states, prmtr_size_dict, Phi_prmtr))})
    if num_states >= 5:
        const_temp.append({'type': 'ineq', 'fun': lambda prmtr:  np.array(eigvalConstraint(prmtr, 4, num_states, prmtr_size_dict, Phi_prmtr))})
        if num_states == 6:
            const_temp.append({'type': 'ineq', 'fun': lambda prmtr:  np.array(eigvalConstraint(prmtr, 5, num_states, prmtr_size_dict, Phi_prmtr))})
    const_temp = tuple(const_temp)  # need to provide constraints as a tuple of list of dictionaries
    return const_temp


def ineq_cons_ls(num_states, prmtr_size_dict, Phi_prmtr): #constraint for Kp for ls (least-square) optimizer
    const_temp = []
    const_temp.append(lambda prmtr:  np.array(eigvalConstraint(prmtr, 0, num_states, prmtr_size_dict, Phi_prmtr)))
    const_temp.append(lambda prmtr:  np.array(eigvalConstraint(prmtr, 1, num_states, prmtr_size_dict, Phi_prmtr)))
    const_temp.append(lambda prmtr:  np.array(eigvalConstraint(prmtr, 2, num_states, prmtr_size_dict, Phi_prmtr)))
    const_temp.append(lambda prmtr:  np.array(eigvalConstraint(prmtr, 3, num_states, prmtr_size_dict, Phi_prmtr)))
    if num_states >= 5:
        const_temp.append(lambda prmtr:  np.array(eigvalConstraint(prmtr, 4, num_states, prmtr_size_dict, Phi_prmtr)))
        if num_states == 6:
            const_temp.append(lambda prmtr:  np.array(eigvalConstraint(prmtr, 5, num_states, prmtr_size_dict, Phi_prmtr)))
    return const_temp