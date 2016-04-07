from __future__ import division
__author__ = 'ssylvain'
import random
import unittest
import datetime as DT
import matplotlib.dates as mdates
import time
import numpy as np
# import numpy.ma as ma
import matplotlib.pyplot as plt
import sys
print sys.argv
from xlrd import open_workbook
from irp_h import *
from kalman import *
from extract_parameters import *
from math import exp
from scipy.optimize import minimize
from scipy.linalg import expm
from scipy import integrate
# from  matplotlib.pyplot import ion
# ion()

Y = np.mat(np.random.rand(100,14))
A0 = np.mat(np.random.rand(14,1))
A1 = np.mat(np.random.rand(14,4))
U0 = np.mat(np.random.rand(4,1))
U1 = np.mat(np.identity(4)*np.random.rand(1))
Q = np.mat(np.identity(4))
Phi = np.mat(np.identity(14))
initV = 'unconditional'


kalman1 = Kalman(Y, A0, A1, U0, U1, Q, Phi, initV)     # default uses X0, V0 = unconditional mean and error variance
Ytt, Yttl, Xtt, Xttl, Vtt, Vttl, Gain_t, cum_log_likelihood = kalman1.filter()

import os, signal
import psutil
import subprocess
doc = subprocess.Popen(["start", "/WAIT", \
                        "S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python\optim_iter.pdf",\
                        "optim_iter.pdf"],\
                        shell=True)
doc.poll()
psutil.Process(doc.pid).get_children()[0].kill()



adobe = r'C:\Program Files (x86)\Adobe\Reader 11.0\Reader\AcroRd32.exe'
doc = subprocess.Popen([adobe, \
                        "S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python\optim_iter.pdf"])

from FuncDesigner import *
from openopt import NLP
from numpy import arange

a, b, c = oovars('a b c')
c.ub = 1.06

n = 1000

f = sum((a - arange(1, n+1) / 1000.0)**2) \
    + (b-1.5)**2 + (c-15)**2

startPoint = {a:[0]*n, # vector with n zeros, however, you'd better use numpy arrays instead of Python lists
	              b: 2,
	              c: 40}
constraints = [(b-15)**2<4, c**4<0.0016, a[0] + a[1] > 2, b + sum(a) < 15]
p = NLP(f, startPoint, constraints=constraints)

r = p.solve('fmincon', matlab="matlab")
# notebook Intel Atom 1.6 GHz:
#Solver:   Time Elapsed = 199.73        (includes MATLAB session create time ~ 30 sec)
#objFunValue: 252.99325 (feasible, MaxResidual = 7.82818e-10)
a_opt, b_opt, c_opt = r(a,b,c)
print(b_opt,c_opt)# (13.0, 0.20000002446305162)
print(a_opt[:5]) # [ 0.9995      1.0005     -0.49849998 -0.49749998 -0.49649998]