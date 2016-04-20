plt.close("all")
__author__ = 'ssylvain'
#########################################################################################################################

# importing US ILB data
book_US_ILB = open_workbook(r"S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\irp_py\data\feds200805.xlsx")

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
    data_US_ILB[:, c] = np.genfromtxt(np.array(rawdata_US_ILB[col_labels[c]], dtype='str'))  # convert missing/strings to 'nan'


#########################################################################################################################

# importing US NB data
book_US_NB = open_workbook(r"S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\irp_py\data\feds200628.xlsx")

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
    data_US_NB[:, c] = np.genfromtxt(np.array(rawdata_US_NB[col_labels[c]], dtype='str'))  # convert missing/strings to 'nan'

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
filename = r"S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python\figures" + \
            str(figures['fig1_name']) + ".png"
# plt.savefig(filename, format="png")
figures['fig1'].savefig(filename, format="png")
# os.startfile(filename)

filename = r"S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python\figures" + \
            str(figures['fig2_name']) + ".png"
# plt.savefig(filename, format="png")
figures['fig2'].savefig(filename, format="png")
# os.startfile(filename)
plt.close()