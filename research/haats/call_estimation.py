from __future__ import division
import sys
import glob
#print(sys.argv)
from pathos.parallel import ParallelPool as PPool
import multiprocessing as multiprocessing
import multiprocess
import copy
#easy_install -f . pathos
# pip install git+https://github.com/uqfoundation/pathos
from import_data import *
from estimation import *
np.random.seed(int(DT.datetime.fromtimestamp(time.time()).strftime('%Y%m%d'))) #np.random.seed(222)
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



################################################

# PRIMITIVES:
figures = []
# use allow_missing_data= 1 to extract ILB and Nominal dates where both are non-missing
allow_missing_data = 0

# set frequency of the data: daily, monthly, quarterly, yearly
estim_freq = 'weekly'

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
sdate, edate = '2010-01-01', '2015-11-23'#time.strftime("%Y-%m-%d") #'2010-01-01'
print("start date: %s" % sdate)
print("end date: %s" % edate)

# extract data for desired maturities and dates
tips_data, nominal_data = ImportData.importUS_Data(US_ilbmaturities, US_nominalmaturities,plots=0,save=1)
data = ImportData.extract_subset(tips_data, nominal_data, sdate, edate, allow_missing_data, estim_freq)

estimation1 =Rolling()
estimation1.run_setup(data, US_ilbmaturities, US_nominalmaturities, \
                estim_freq=estim_freq, num_states=num_states,\
                fix_Phi=fix_Phi, setdiag_Kp=setdiag_Kp, initV=initV)

estimation_method,tolerance, maxiter ,toltype,solver_mle,maxiter_mle, maxfev_mle, ftol_mle, xtol_mle, \
    constraints_mle, priors_bayesian, maxiter_bayesian, burnin_bayesian, multistart, ncpus = 'em_mle', 1e-4, 2 , 'max_abs', \
    'Nelder-Mead', 5, 5, 0.0001, 0.0001, 'off', None, 1000, None, 2, 2

# estimation_method,tolerance, maxiter ,toltype,solver_mle,maxiter_mle, maxfev_mle, ftol_mle, xtol_mle, \
#     constraints_mle, priors_bayesian, maxiter_bayesian, burnin_bayesian, multistart, ncpus = 'em_mle', 1e-6, 15 , \
#     'max_abs', 'Powell', 10000, 10000, 0.01, 0.01, 'off', None, 1000, None, 25, 8

estimation1.fit(estimation_method=estimation_method, tolerance=tolerance, maxiter=maxiter, toltype=toltype, \
                solver_mle=solver_mle, maxiter_mle=maxiter_mle, maxfev_mle=maxfev_mle, ftol_mle=ftol_mle,
                xtol_mle=xtol_mle, constraints_mle=constraints_mle, \
                priors_bayesian=priors_bayesian, maxiter_bayesian=maxiter_bayesian, burnin_bayesian=burnin_bayesian,
                multistart=multistart, ncpus=ncpus)

estimation1.collect_results()
estimation1.expected_inflation() #do not use smoother here to avoid look-ahead bias
estimation1.save_output()
estimation1.plot_results()

# Delete temporary files:
map(os.remove, glob.glob(r""+str.replace(os.getcwd(), '\\', '/')+"/output/parallel_worker_output"+"*.txt"))


end_time = time.time()


# Save workspace
filename='/output/data/shelve.out'
my_shelf = shelve.open(filename,'n') # 'n' for new
for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()


print("--- %s seconds ---" % (end_time - start_time))
