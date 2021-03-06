from __future__ import division
# from IPython.core.debugger import Tracer; debug_here = Tracer() #this is the approach that works for ipython debugging
import os
import numpy as np
# import seaborn as sns
# sns.set("talk", font_scale=0.8, rc={"lines.linewidth": 1}) #see https://stanford.edu/~mwaskom/software/seaborn/tutorial/aesthetics.html
# plt = sns.plt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
#print(sys.argv)
import pandas as pd
from xml.sax import ContentHandler, parse
import time
import datetime as DT
import time


class ImportData:

    def __init__(self):
       class_name = self.__class__.__name__
        # print(class_name, "constructed")

    def __del__(self):
        class_name = self.__class__.__name__
        # print(class_name, "destroyed")

    @staticmethod
    def update_us_data():
        '''Get Nelson-Siegel-Svensson fitted yield data for TIPS and Nominal bonds from Fed website'''

        #########################################################################################################################
        # importing US ILB data
        tic = time.clock()
        excelHandler = ExcelHandler()#instantiate object
        parse('https://www.federalreserve.gov/econresdata/researchdata/feds200805.xls', excelHandler)
        toc = time.clock()
        print('processing time for importing US ILB data: ' + str(toc - tic))

        #create dateframe
        tips_data = pd.DataFrame(
            pd.DataFrame(excelHandler.tables[0][18:]).values[1:,1:],#data
            index=pd.tseries.index.DatetimeIndex(    pd.DataFrame(excelHandler.tables[0][18:]).iloc[1:,0].str.strip().values    ),#row labels; dates
            columns=pd.DataFrame(excelHandler.tables[0][18:]).values[0,1:]#column labels
            )
        #clean up column names:
        tips_data.columns=tips_data.columns.str.strip() # get rid of spaces and carry over lines
        tips_data=tips_data.filter(regex='TIPSY') # keep only relevant columns
        #convert data to numeric
        for c in tips_data.columns:
            tips_data[c] = pd.to_numeric(tips_data[c], errors='coerce')

        tips_data = tips_data.sort_index()

        print(tips_data.head()) #check that data looks good
        print(tips_data.describe())

        data_US_ILB, US_ILB_dates = tips_data.values, tips_data.index

        print('tips_data column kept are:')
        print(tips_data.columns)

        #########################################################################################################################
        # importing US NB data
        tic = time.clock()
        excelHandler = ExcelHandler()#instantiate object
        parse('http://www.federalreserve.gov/econresdata/researchdata/feds200628.xls', excelHandler)
        toc = time.clock()
        print('processing time for importing US NB data: ' + str(toc - tic))

        #create dateframe
        nominal_data = pd.DataFrame(
            pd.DataFrame(excelHandler.tables[0][9:]).values[1:,1:],#data
            index=pd.tseries.index.DatetimeIndex(    pd.DataFrame(excelHandler.tables[0][9:]).iloc[1:,0].str.strip().values    ),#row labels; dates
            columns=pd.DataFrame(excelHandler.tables[0][9:]).values[0,1:]#column labels
            )
        #clean up column names:
        nominal_data.columns=nominal_data.columns.str.strip() # get rid of spaces and carry over lines
        nominal_data=nominal_data.filter(regex='SVENY')

        #convert data to numeric
        for c in nominal_data.columns:
            nominal_data[c] = pd.to_numeric(nominal_data[c],errors='coerce')

        nominal_data = nominal_data.sort_index()

        print(nominal_data.head()) #check that data looks good
        print(nominal_data.describe()) # keep only relevant columns

        data_US_NB, US_NB_dates = nominal_data.values, nominal_data.index

        print('nominal_data column kept are:')
        print(nominal_data.columns)
        #########################################################################################################################

        return tips_data, nominal_data

    @staticmethod
    def importUS_Data(US_ilbmaturities = None, US_nominalmaturities = None, plots=1, save=1):
        '''This is to import US nominal and tips yields from
        Gurkaynak and Wright (2006;2008)
        http://www.federalreserve.gov/econresdata/researchdata/feds200628.xls
        https://www.federalreserve.gov/econresdata/researchdata/feds200805.xls
        '''

        #today's date
        edate = np.array(time.strftime("%Y-%m-%d"), dtype=np.datetime64)  # in format : '2015-02-11'

        tips_f, nominal_f = r""+str.replace(os.getcwd(), '\\', '/')+"/output/data/tips_raw_data_US.csv" , r""+str.replace(os.getcwd(), '\\', '/')+"/output/data/nominal_raw_data_US.csv"

        #if files were already updated today, do not update again:
        if os.path.exists(tips_f) & os.path.exists(nominal_f) :
            if np.array(time.strftime("%Y-%m-%d",(time.gmtime(os.path.getmtime(tips_f)))), dtype=np.datetime64) == np.array(time.strftime("%Y-%m-%d",(time.gmtime(os.path.getmtime(nominal_f)))), dtype=np.datetime64) >= edate:
                tips_data, nominal_data = pd.read_csv(tips_f,index_col=0) , pd.read_csv(nominal_f,index_col=0)
                tips_data.index, nominal_data.index = pd.tseries.index.DatetimeIndex(tips_data.index), pd.tseries.index.DatetimeIndex(nominal_data.index)
            else:
                tips_data, nominal_data = ImportData.update_us_data()
        else:
            tips_data, nominal_data = ImportData.update_us_data()
        # debug_here()
        if US_ilbmaturities is None:
            US_ilbmaturities = np.array([i.replace('TIPSY','') for i in tips_data.columns.values], dtype=int)
        if US_nominalmaturities is None:
            US_nominalmaturities = np.array([i.replace('SVENY','') for i in nominal_data.columns.values], dtype=int)

        if plots == 1:
            #########################################################################################################################
            #plotting the data
            plt.close()
            fig, ax = plt.subplots(1)
            figures = {'fig1': fig, 'ax_fig1': ax}
            figures['fig1_name'] = '/nominal_raw_data_US'
            figures['ax_fig1'].plot(nominal_data.index.to_datetime().values, nominal_data.values)
            plt.legend(nominal_data.columns,loc='center left',fontsize=7,frameon=0, bbox_to_anchor=(1, 0.5))
            # rotate and align the tick labels so they look better
            figures['fig1'].autofmt_xdate()
            # use a more precise date string for the x axis locations in the toolbar
            figures['ax_fig1'].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            figures['ax_fig1'].set_title('US Nominal Bonds')
            plt.axes.labelcolor='black'
            plt.figsize=(8,8)
            plt.draw()

            plt.close()
            fig, ax = plt.subplots(1)
            figures['fig2'] = fig
            figures['ax_fig2'] = ax
            figures['fig2_name'] = '/tips_raw_data_US'
            figures['ax_fig2'].plot(tips_data.index.to_datetime().values, tips_data.values)
            plt.legend(tips_data.columns,loc='center left',fontsize=7,frameon=0, bbox_to_anchor=(1, 0.5))
            # rotate and align the tick labels so they look better
            figures['fig2'].autofmt_xdate()
            # use a more precise date string for the x axis locations in the toolbar
            figures['ax_fig2'].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            figures['ax_fig2'].set_title('US InfLinkBonds Bonds')
            plt.axes.labelcolor='black'
            plt.figsize=(8,8)
            plt.draw()

        if save==1:

            #########################################################################################################################

            # saving data
            if not os.path.exists(r""+str.replace(os.getcwd(), '\\', '/')+"/output/data"):
                os.makedirs(r""+str.replace(os.getcwd(), '\\', '/')+"/output/data") #make output directory if it doesn't exist
            nominal_data.to_csv(nominal_f)
            tips_data.to_csv(tips_f)

            # saving figures
            if plots == 1:
                if not os.path.exists(r""+str.replace(os.getcwd(), '\\', '/')+"/output/figures"):
                    os.makedirs(r""+str.replace(os.getcwd(), '\\', '/')+"/output/figures") #make output directory if it doesn't exist
                for format in ['eps','png']:
                    filename = r""+str.replace(os.getcwd(), '\\', '/')+"/output/figures" + \
                                str(figures['fig1_name']) + "." + format
                    figures['fig1'].savefig(filename, format=format)

                    filename = r""+str.replace(os.getcwd(), '\\', '/')+"/output/figures" + \
                                str(figures['fig2_name']) + "." + format
                    figures['fig2'].savefig(filename, format=format)
                plt.close()

            #########################################################################################################################\
        tips_data = tips_data[np.array(['TIPSY' + str(c).zfill(2) for c in US_ilbmaturities]) ]  # need to padd column names with a zero before converting to string
        nominal_data = nominal_data[np.array(['SVENY' + str(c).zfill(2) for c in US_nominalmaturities]) ] # need to padd column names with a zero before converting to string

        tips_data.rename(columns={i:i.replace('TIPSY','TIPS_y')  for i in tips_data.columns.values}, inplace=True)
        nominal_data.rename(columns={i:i.replace('SVENY','Nominals_y')  for i in nominal_data.columns.values}, inplace=True)

        return tips_data, nominal_data

    @staticmethod
    def extract_subset(tips_data, nominal_data, sdate, edate, allow_missing_data=0, estim_freq='daily'):
        '''Extract relevant dates and save pandas data frames for ILB and Nominal Bonds data in dictionary'''

        tips_data = tips_data.loc[(tips_data.index >= sdate)&(tips_data.index <= edate),:]
        nominal_data = nominal_data.loc[(nominal_data.index >= sdate)&(nominal_data.index <= edate),:]

        if allow_missing_data == 1:
            # taking the union of dates and padding with NaN data where appropriate
            tempdates = tips_data.index.union(nominal_data.index)
            tips_data = tips_data.reindex(tempdates).sort_index()
            nominal_data = nominal_data.reindex(tempdates).sort_index()

        else:
            # taking the intersection of dates:
            tempdates = tips_data.index.intersection(nominal_data.index)
            tips_data = tips_data.reindex(tempdates).sort_index()
            nominal_data = nominal_data.reindex(tempdates).sort_index()

        # extracting dates and data at the right frequency
        if estim_freq == 'daily':
            estim_freq = None  # daily increment
        elif estim_freq == 'weekly':
            estim_freq = '1w'  # weekly increment
        elif estim_freq == 'monthly':
            estim_freq = '1m'  # monthly increment
        elif estim_freq == 'quarterly':
            estim_freq = '1q'  # quarterly increment

        # storing copies of the data in dictionaries:
        data = {'US_NB': nominal_data, 'US_ILB': tips_data}

        if estim_freq is not None:
            for vv in ['US_NB', 'US_ILB']:
                data[vv] = data[vv].asfreq(estim_freq, method='pad')

        return data


class ExcelHandler(ContentHandler):
    '''This is to import Excel data saved as XML
    Reference https://goo.gl/KaOBG3 and
    http://stackoverflow.com/questions/33470130/read-excel-xml-xls-file-with-pandas
    '''

    def __init__(self):

        self.chars = [  ]
        self.cells = [  ]
        self.rows = [  ]
        self.tables = [  ]

    def characters(self, content):

        self.chars.append(content)

    def startElement(self, name, atts):

        if name=="Cell":
            self.chars = [  ]
        elif name=="Row":
            self.cells=[  ]
        elif name=="Table":
            self.rows = [  ]

    def endElement(self, name):

        if name=="Cell":
            self.cells.append(''.join(self.chars))
        elif name=="Row":
            self.rows.append(self.cells)
        elif name=="Table":
            self.tables.append(self.rows)