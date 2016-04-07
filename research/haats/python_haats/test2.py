__author__ = 'ssylvain'
import sys, os
# sys.path.insert(0, r'S:\PMG\MAS\MAPS\Research\ssylvain\PythonPackages')
# fdes = __import__('FuncDesigner', globals(), locals(), [], -1)
# del sys.path[0]
# sys.path.insert(0, r'S:\PMG\MAS\MAPS\Research\ssylvain\PythonPackages\OpenOpt')
# openo = __import__('openopt', globals(), locals(), ['NLP'], -1)
# del sys.path[0]
os.chdir(r'S:\PMG\MAS\MAPS\Research\ssylvain\PythonPackages')
from  FuncDesigner import *
os.chdir(r'S:\PMG\MAS\MAPS\Research\ssylvain\PythonPackages')
from  OpenOpt import NLP
os.chdir(r'S:\PMG\MAS\MAPS\Research\ssylvain\MAS Projects\Inflation\InflationRiskPremia\python')
import numpy as np
from numpy import arange

# a, b, c = fdes.oovars('a', 'b', 'c')
a = oovar('a', size=2)
b = oovar('b', size=1)
c = oovar('c', size=1)
f = sum(a*[1, 2])**2+b**2+c**2
startPoint = {a:[100, 12], b:2, c:40} # however, you'd better use numpy arrays instead of Python lists
p = NLP(f, startPoint)
p.constraints = [(2*c+a-10)**2 < 1.5 + 0.1*b, (a-10)**2<1.5, a[0]>8.9, a+b > [ 7.97999836, 7.8552538 ], \
a < 9, (c-2)**2 < 1, b < -1.02, c > 1.01, (b + c * np.log10(a).sum() - 1) ** 2==0]
r = p.solve('ralg')
a_opt, b_opt, c_opt = r(a, b, c)
print(a_opt, b_opt, c_opt)
