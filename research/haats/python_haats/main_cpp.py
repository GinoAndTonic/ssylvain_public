__author__ = 'ssylvain'
# this is the source file were we do the heavy computations
from irp_h import *
import sys
print sys.argv
from __future__ import division

asset1 = Asset('USA')
print(asset1.getTotal())

bond1 = InfLinkBonds(12, 'CAN')

bond2 = Bonds(4, 'MEX')

print(bond1.getName())

print(bond2.getBondMaturity())

print(bond2.getCountry())


# asset1 = Asset('false')
from xlrd import open_workbook

book = open_workbook(
    r"S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\irp_py\data\test.xls")
sheet = book.sheet_by_index(0)

# read header values into the list
keys = [sheet.cell(0, col_index).value for col_index in range(sheet.ncols)]

dict_list = []
for row_index in range(1, sheet.nrows):
    d = {keys[col_index]: sheet.cell(row_index, col_index).value
         for col_index in range(sheet.ncols)}
    dict_list.append(d)

print(dict_list)


book_US_ILB = open_workbook(r"S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\irp_py\data\feds200805.xlsx")
sheet_US_ILB = book_US_ILB.sheet_by_index(0)