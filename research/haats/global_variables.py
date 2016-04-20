__author__ = 'ssylvain'

global estim_freq, initV, num_states, stationarity_assumption, US_ilbmaturities, US_nominalmaturities, US_num_maturities, dt
estim_freq, initV, num_states, stationarity_assumption, US_ilbmaturities, US_nominalmaturities, US_num_maturities, dt = ([] for i in range(8))


class GlobalVars:

    # @staticmethod   # making methods unbound from class object
    #def __init__(self):
    #     print('calling GlobalVars constructor')

    #  making methods unbound from class object
    #def global_variables_init():
    #    estim_freq, initV, num_states, stationarity_assumption, US_ilbmaturities, US_nominalmaturities, US_num_maturities, dt = (
    #        [] for i in range(8))

    @staticmethod   # making methods unbound from class object
    def global_variables_update( estim_freq_update, initV_update, num_states_update, stationarity_assumption_update, US_ilbmaturities_update,
                                US_nominalmaturities_update, US_num_maturities_update, dt_update):
        estim_freq, initV, num_states, stationarity_assumption, US_ilbmaturities, US_nominalmaturities, US_num_maturities, dt = estim_freq_update, initV_update, num_states_update,\
                                                                                                                                stationarity_assumption_update, US_ilbmaturities_update, \
                                                                                                                               US_nominalmaturities_update, US_num_maturities_update, dt_update

    @staticmethod   # making methods unbound from class object
    def global_variables_get():
        return estim_freq, initV, num_states, stationarity_assumption, US_ilbmaturities, US_nominalmaturities, US_num_maturities, dt


    # # making methods unbound from class object
    #def __del__(self):
    #    print("GlobalVarsdestroyed")