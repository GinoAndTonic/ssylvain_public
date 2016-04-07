__author__ = 'ssylvain'
from FuncDesigner import *
from openopt import NLP

a, b, c = oovars('a', 'b', 'c')
f = sum(a*[1, 2])**2+b**2+c**2
startPoint = {a:[100, 12], b:2, c:40} # however, you'd better use numpy arrays instead of Python lists
p = NLP(f, startPoint)
p.constraints = [(2*c+a-10)**2 < 1.5 + 0.1*b, (a-10)**2<1.5, a[0]>8.9, a+b > [ 7.97999836, 7.8552538 ], \
a < 9, (c-2)**2 < 1, b < -1.02, c > 1.01, (b + c * log10(a).sum() - 1) ** 2==0]
r = p.solve('gsubg')
#r = p.solve('ipopt')
# r = p.solve('fmincon', matlab='MATLAB/R2013a/bin/matlab') # need Matlab instrument control toolbox to be able to call it from python via OpenOpt
a_opt, b_opt, c_opt = r(a, b, c)
print(a_opt, b_opt, c_opt)