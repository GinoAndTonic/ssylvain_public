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
import matplotlib as mplt
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

b1 = InfLinkBonds(4, 'USA')
b1.setZeroYieldsTS(12)
b1.setZeroYieldsDates(49)
print b1.getCountry()
print b1.getZeroYieldsTS()
print b1.getZeroYieldsDates()