from __future__ import division
import sys
print(sys.argv)
from import_data import *
from estimation import *
plt.close("all")
plt.close()
plt.ion()
# plt.rc('text', usetex=True)     #for TeX interpretation of title and legends
__author__ = 'ssylvain'


start_time = time.time()

# The interveal between each rolling window: the gap by which the estimationd window shifts
# (e.g. with tgap = 1, rolling window is updated daily)
tgap = 30

# Rolling window: 0 if using expanding window, 1 if using rolling window
rolling = 0

#Rolling window size: size of rolling window (in years).. Use inf for full sample estimation
windowsize = np.inf;

np.set_printoptions(precision=32, suppress=True) #increase precision on  numeric values

# global variables
global a, Kp, lmda,lmda2,  Phi, sigma11, sigma22, sigma22_2, sigma33, sigma33_2,\
    sigma44, Sigma, thetap, Nfeval, figures, cum_log_likelihood,\
    Nfeval_vec, cum_log_likelihood_vec, doc, Nfeval_inner


################################################

# PRIMITIVES:
figures = []
# use allow_missing_data= 1 to extract ILB and Nominal dates where both are non-missing
allow_missing_data = 0

# set frequency of the data: daily, monthly, quarterly, yearly
estim_freq = 'daily'

fix_Phi = 0     # "1" if you want to fix the volatility of observed yields using covar of historical data
                # "0" if you want to jointly estimate it with other model parameters
setdiag_Kp = 0  # "1" if you want to Kp to be diagonal so the state variables are assumed independent
                # "0" if you want to Kp to be unrestricted

# options for initializing the Kalman filter error variance:
#'steady_state' or 'unconditional' or 'identity' matrix
initV = 'unconditional'

# number of hidden state variables 4, or 6
num_states = 4

# Specify the maturities of data we want to use
US_ilbmaturities = np.array([2, 3,  5, 6, 8, 9, 10])
US_nominalmaturities = np.array([2, 3,  5, 6, 8, 9, 10])
US_maturities = np.hstack((US_nominalmaturities, US_ilbmaturities))

############################################################

# Set start and end dates for estimation
sdate, edate = '2004-01-01', '2010-01-01'
print("start date: %s" % sdate)
print("end date: %s" % edate)

# extract data for desired maturities and dates
tips_data, nominal_data = ImportData.importUS_Data(US_ilbmaturities, US_nominalmaturities,plots=0,save=1)
data = ImportData.extract_subset(tips_data, nominal_data, sdate, edate, allow_missing_data, estim_freq)

estimation =Rolling()
estimation.run(data, US_ilbmaturities, US_nominalmaturities, \
                estim_freq=estim_freq, num_states=num_states,\
                fix_Phi=fix_Phi, setdiag_Kp=setdiag_Kp, initV=initV)

end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))
