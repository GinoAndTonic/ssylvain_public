from __future__ import division
import shelve
import psutil
import subprocess
import os
import datetime as DT
import matplotlib.dates as mdates
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
import pylab as plab
import sys
print(sys.argv)
from xlrd import open_workbook
import xlwt
from irp_h import *
from kalman import *
from extract_parameters import *
from math import exp
from scipy.optimize import minimize, fmin_slsqp
from scipy.linalg import expm
from scipy import integrate
from estim_constraints import *
from scipy.stats import norm


class ImportData:

    def __init__(self):
       class_name = self.__class__.__name__
        # print(class_name, "constructed")

    def __del__(self):
        class_name = self.__class__.__name__
        # print(class_name, "destroyed")

    def importUS_Data(self, US_ilbmaturities, US_nominalmaturities):
        plt.close("all")
        #########################################################################################################################
        # importing US ILB data
        book_US_ILB = open_workbook(r"Z:\GMO\Research\AffineTermStructure\code\data\feds200805.xlsx")
        sheet_US_ILB = book_US_ILB.sheet_by_index(0)

        # read header values into the list. for tips, 1st row is row 19
        s_ilb = 18
        keys_US_ILB = [str(cell.value)
                       for cell in sheet_US_ILB.row(s_ilb)]
        keys_US_ILB[0] = 'datadate'
        print(keys_US_ILB)
        
        arr = []
        for rowind in range(sheet_US_ILB.nrows)[s_ilb+1:]:
            arr.append([cell.value for cell in sheet_US_ILB.row(rowind)])
        arr = np.array(arr)
        arr[arr==''] = 'missing'  #need to replace empty cells with 'missing'
        
        rawdata_US_ILB = np.rec.fromrecords(arr, names=keys_US_ILB)
        rawdata_US_ILB = np.sort(rawdata_US_ILB, order='datadate')    # need to make sure that the data is sorted by date
        col_labels = np.array(['TIPSY' + str(c).zfill(2) for c in US_ilbmaturities])    # need to padd with a zero before converting to string
        print(col_labels)
        
        # US_ILB_dates = rawdata_US_ILB['datadate']     #using this approach keeps the unicode 'u' label for dates. instead use approach below
        US_ILB_dates = []
        US_ILB_dates.append([str(cell.value) for cell in sheet_US_ILB.col(0)])
        del US_ILB_dates[0][0:s_ilb+1]
        US_ILB_dates = np.transpose(np.sort(np.array(US_ILB_dates,dtype=np.datetime64)))
        
        data_US_ILB = np.zeros((US_ILB_dates.size, col_labels.size))
        for c in range(col_labels.size):
            try: #this is for python 2
                data_US_ILB[:, c] = np.genfromtxt(np.array(rawdata_US_ILB[col_labels[c]], dtype='str'))  # convert missing/strings to 'nan'
            except: #this is for python 3
                data_US_ILB[:, c] = np.genfromtxt(np.array(rawdata_US_ILB[col_labels[c]], dtype='bytes'))  # convert missing/strings to 'nan'

        #########################################################################################################################       
        # importing US NB data
        book_US_NB = open_workbook(r"Z:\GMO\Research\AffineTermStructure\code\data\feds200628.xlsx")
        
        sheet_US_NB = book_US_NB.sheet_by_index(0)
        
        # read header values into the list. for tips, 1st row is row 19
        s_nb = 9
        keys_US_NB = [str(cell.value)
                       for cell in sheet_US_NB.row(s_nb)]
        keys_US_NB[0] = 'datadate'
        print(keys_US_NB)
        
        arr = []
        for rowind in range(sheet_US_NB.nrows)[s_nb+1:]:
            arr.append([cell.value for cell in sheet_US_NB.row(rowind)])
        arr =  np.array(arr)
        arr[arr==''] = 'missing'  #need to replace empty cells with 'missing'
        
        rawdata_US_NB = np.rec.fromrecords(arr, names=keys_US_NB)
        rawdata_US_NB = np.sort(rawdata_US_NB, order='datadate')    # need to make sure that the data is sorted by date
        col_labels = np.array(['SVENY' + str(c).zfill(2) for c in US_nominalmaturities])    # need to padd with a zero before converting to string
        print(col_labels)
        
        # US_NB_dates = rawdata_US_NB['datadate']     #using this approach keeps the unicode 'u' label for dates. instead use approach below
        US_NB_dates = []
        US_NB_dates.append([str(cell.value) for cell in sheet_US_NB.col(0)])
        del US_NB_dates[0][0:s_nb+1]
        US_NB_dates = np.transpose(np.sort(np.array(US_NB_dates,dtype=np.datetime64)))
        
        data_US_NB = np.zeros((US_NB_dates.size, col_labels.size))
        for c in range(col_labels.size):
            try: #this is for python 2
                data_US_NB[:, c] = np.genfromtxt(np.array(rawdata_US_NB[col_labels[c]], dtype='str'))  # convert missing/strings to 'nan'
            except: #this is for python 3
                data_US_NB[:, c] = np.genfromtxt(np.array(rawdata_US_NB[col_labels[c]], dtype='bytes'))  # convert missing/strings to 'nan'
        #########################################################################################################################
        #plotting the data
        plt.close()
        fig, ax = plt.subplots(1)
        figures = {'fig1': fig, 'ax_fig1': ax}
        figures['fig1_name'] = '\\f1'
        figures['ax_fig1'].plot(US_NB_dates.astype(DT.datetime), data_US_NB)   # matplotlib does not support NumPy datetime64 objects (at least not yet). Therefore, convert x to Python datetime.datetime
        # rotate and align the tick labels so they look better
        figures['fig1'].autofmt_xdate()
        # use a more precise date string for the x axis locations in the
        # toolbar
        #ax1.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')  # DateFormatter('%Y-%m-%d')
        figures['ax_fig1'].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # figures['fig1'].suptitle('US Nominal Bonds')
        figures['ax_fig1'].set_title('US Nominal Bonds')
        labels = figures['ax_fig1'].get_xticklabels()
        plt.setp(labels, rotation=30)
        plt.draw()
        
        plt.close()
        fig, ax = plt.subplots(1)
        figures['fig2'] = fig
        figures['ax_fig2'] = ax
        figures['fig2_name'] = '\\f2'
        figures['ax_fig2'].plot(US_ILB_dates.astype(DT.datetime), data_US_ILB)   #matplotlib does not support NumPy datetime64 objects (at least not yet). Therefore, convert x to Python datetime.datetime
        # rotate and align the tick labels so they look better
        figures['fig2'].autofmt_xdate()
        # use a more precise date string for the x axis locations in the
        # toolbar
        #ax1.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')  # DateFormatter('%Y-%m-%d')
        figures['ax_fig2'].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # plt.title('US InfLinkBonds Bonds')
        figures['ax_fig2'].set_title('US InfLinkBonds Bonds')
        labels = figures['ax_fig2'].get_xticklabels()
        plt.setp(labels, rotation=30)
        plt.draw()
    
        # saving figures
        filename = r"Z:\GMO\Research\AffineTermStructure\code\figures" + \
                    str(figures['fig1_name']) + ".png"
        # plt.savefig(filename, format="png")
        figures['fig1'].savefig(filename, format="png")
        # os.startfile(filename)
        
        filename = r"Z:\GMO\Research\AffineTermStructure\code\figures" + \
                    str(figures['fig2_name']) + ".png"
        # plt.savefig(filename, format="png")
        figures['fig2'].savefig(filename, format="png")
        # os.startfile(filename)
        plt.close()
        #########################################################################################################################
        #saving data
        output_filename=r'Z:\GMO\Research\AffineTermStructure\code\python_haats\outputfiles\raw_data_US.out'
        my_shelf = shelve.open(output_filename,'n') # 'n' for new
        for key in ['data_US_NB', 'US_NB_dates', 'data_US_ILB', 'US_ILB_dates']:
            try:
                my_shelf[key] = locals()[key]
            except:
                # __builtins__, my_shelf, and imported modules can not be shelved.
                print('ERROR shelving: {0}'.format(key))
        my_shelf.close()
        #########################################################################################################################
        return data_US_NB, US_NB_dates, data_US_ILB, US_ILB_dates

    def extract_subset(self, data_US_NB, US_NB_dates, data_US_ILB, US_ILB_dates, sdate, edate,allow_missing_data=0, estim_freq='daily'):
        # storing copies of the data in dictionaries:
        data = {'US_NB': data_US_NB, 'US_ILB': data_US_ILB}
        dates = {'US_NB': US_NB_dates, 'US_ILB': US_ILB_dates}

        if allow_missing_data == 1:
            for vv in ['US_NB', 'US_ILB']:
                loc = np.array((dates[vv] >= sdate)&(dates[vv] <= edate))[:, 0]   # valid US NB dates
                dates[vv] = dates[vv][loc]
                data[vv] = data[vv][loc, :]
            # taking the union of dates and padding data where appropriate
            tempdates = np.union1d(dates['US_NB'], dates['US_ILB'])
            for vv in ['US_NB', 'US_ILB' ]:
                tempdata = np.empty([tempdates.size, data[vv].shape[1]]) * np.nan     # matrix with nan
                temploc = np.in1d(tempdates, dates[vv])
                tempdata[temploc, :] = data[vv]
                dates[vv] = tempdates
                data[vv] = tempdata
        else:
            for vv in ['US_NB', 'US_ILB' ]:
                loc2 = np.array((dates[vv] >= sdate)&(dates[vv] <= edate))[:, 0]    # valid US NB dates
                dates[vv] = dates[vv][loc2]
                data[vv] = data[vv][loc2, :]
                loc3 = np.array(np.sum(data[vv] != np.nan, axis=1) == data[vv].shape[1])
                dates[vv] = dates[vv][loc3]
                data[vv] = data[vv][loc3, :]
            # keep only dates in common:
            loc_ilb2 = np.in1d(dates['US_ILB'], dates['US_NB'])  # intersecting dates
            dates['US_ILB'] = dates['US_ILB'][loc_ilb2]
            data['US_ILB'] = data['US_ILB'][loc_ilb2, :]

            loc_nb2 = np.in1d(dates['US_NB'], dates['US_ILB'])  # intersecting dates
            dates['US_NB'] = dates['US_NB'][loc_nb2]
            data['US_NB'] = data['US_NB'][loc_nb2, :]

        # extracting dates at the right interval
        if estim_freq == 'daily':
            timeloop = np.sort(np.arange(dates['US_NB'].shape[0]-1, 0, -1))  # daily increment
        elif estim_freq == 'weekly':
            timeloop = np.sort(np.arange(dates['US_NB'].shape[0]-1, 0, -5))  # weekly increment
        elif estim_freq == 'monthly':
            timeloop = np.sort(np.arange(dates['US_NB'].shape[0]-1, 0, -20))  # monthly increment
        elif estim_freq == 'quarterly':
            timeloop = np.sort(np.arange(dates['US_NB'].shape[0]-1, 0, -60))  # quarterly increment

        for vv in ['US_NB', 'US_ILB']:
            dates[vv] = dates[vv][timeloop]
            data[vv] = data[vv][timeloop, :]

        return data, dates