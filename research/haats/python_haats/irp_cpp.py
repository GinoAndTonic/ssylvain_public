from __future__ import division
__author__ = 'ssylvain'
# this is the source file were we do the heavy computations
from irp_h import *
import scipy.optimize as op
import numpy as np
from math import exp


lmda = 1
a = 1
sigma11 = 1
sigma22 = 1
sigma33 = 1
sigma44 = 1


def A1(bondtype, maturity):
    tau = maturity
    if bondtype == 'NominalBonds':
        out = np.float32(-(1 / tau) * np.array([-tau,
                                                -((1 - exp(-(lmda * tau))) / lmda),
                                                -((1 - exp(-(lmda * tau))) / lmda) + tau / exp(lmda * tau),
                                                0])
                         )

    elif bondtype == 'InfLinkBonds':
        out = np.float32(-(a / tau) * np.array([0,
                                                -((1 - exp(-(lmda * tau))) / lmda),
                                                -((1 - exp(-(lmda * tau))) / lmda) + tau / exp(lmda * tau),
                                                -tau / a])
                         )

    return out


def A0(bondtype, maturity):
    tau = maturity
    if bondtype == 'NominalBonds':
        out = np.float32(np.array([(-(sigma11 ^ 2 * tau ^ 2) / 6
                                    + (sigma22 ^ 2 * (3 + exp(-2 * lmda * tau) - 4 / exp(lmda * tau) - 2 * lmda * tau)) / (
                                    4 * lmda ^ 3 * tau)
                                    - (sigma33 ^ 2 * (
                                    -5 - 6 * lmda * tau - 2 * lmda ^ 2 * tau ^ 2 + 8 * exp(lmda * tau) * (2 + lmda * tau) + exp(
                                    2 * lmda * tau) * (-11 + 4 * lmda * tau))) / (8 * exp(2 * lmda * tau) * lmda ^ 3 * tau) )])
                         )

    elif bondtype == 'InfLinkBonds':
        out = np.float32(np.array([(-(sigma44 ^ 2 * tau ^ 2) / 6
                                    + (a ^ 2 * sigma22 ^ 2 * (3 + exp(-2 * lmda * tau) - 4 / exp(lmda * tau) - 2 * lmda * tau)) / (
                                    4 * lmda ^ 3 * tau)
                                    - (a ^ 2 * sigma33 ^ 2 * (
                                    -5 - 6 * lmda * tau - 2 * lmda ^ 2 * tau ^ 2 + 8 * exp(lmda * tau) * (2 + lmda * tau) + exp(
                                     2 * lmda * tau) * (-11 + 4 * lmda * tau))) / (8 * exp(2 * lmda * tau) * lmda ^ 3 * tau))])
                         )

    return out


import thread
import threading

def raw_input_with_timeout(prompt, timeout=10.0):
    print prompt,
    timer = threading.Timer(timeout, thread.interrupt_main)
    astring = None
    try:
        timer.start()
        astring = raw_input(prompt)
    except KeyboardInterrupt:
        pass
    timer.cancel()
    return astring

#def lnlike(theta, x, y, yerr):
# m, b, lnf = theta
#model = m * x + b
#inv_sigma2 = 1.0 / (yerr ** 2 + model ** 2 * exp(2 * lnf))
#return -0.5 * (np.sum((y - model) ** 2 * inv_sigma2 - log(inv_sigma2)))


#nll = lambda *args: -lnlike(*args)
#result = op.minimize(nll, [m_true, b_true, log(f_true)], args=(x, y, yerr))
#m_ml, b_ml, lnf_ml = result["x"]



