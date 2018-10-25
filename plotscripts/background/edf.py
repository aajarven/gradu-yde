# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import math
import pylab
import random

def CDFifyXData(array):
	array = np.repeat(array, 2)
	array = np.delete(array, len(array)-1)
	return array


np.random.seed(100)

rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

N = 35
normalNumbers = np.random.normal(16.0, 2.0, N)
normalNumbersWide = np.random.normal(15.0, 4.0, N)
normalNumbersWide2 = np.random.normal(15.0, 4.0, N)
uniform = np.random.uniform(5.0, 25.0, N)

biggest = max(max(max(np.amax(normalNumbers), np.amax(normalNumbersWide)),
			  np.amax(uniform)), np.amax(normalNumbersWide2))

percentage = np.arange(0, 100, 100.0/N)
percentage = np.append(percentage, 100)
percentage = np.repeat(percentage, 2)
percentage = np.delete(percentage, 0)

normalNumbers.sort()
normalNumbersWide.sort()
normalNumbersWide2.sort()
uniform.sort()

normalNumbers = np.append(normalNumbers, biggest)
normalNumbersWide = np.append(normalNumbersWide, biggest)
normalNumbersWide2 = np.append(normalNumbersWide2, biggest)
uniform = np.append(uniform, biggest)

normalNumbers = CDFifyXData(normalNumbers)
normalNumbersWide = CDFifyXData(normalNumbersWide)
uniform = CDFifyXData(uniform)
normalNumbersWide2 = CDFifyXData(normalNumbersWide2)

plt.plot(normalNumbers, percentage, label='$\mu$=16.0, $\sigma$=2.0',
		 linewidth=2.0, color='C0')
plt.plot(normalNumbersWide, percentage, label='$\mu$=15.0, $\sigma$=4.0',
		 linewidth=2.0, color='C1')
plt.plot(normalNumbersWide2, percentage, label='$\mu$=15.0, $\sigma$=4.0',
		 linewidth=2.0, color='C4')
plt.plot(uniform, percentage, label='uniform',
		 linewidth=2.0, color='C2')

plt.xlabel('x')
plt.ylabel('EDF \enspace (\%)')
plt.legend(loc=0)
plt.xlim([2, 28])
plt.ylim([0, 100])

F = pylab.gcf()
F.set_size_inches(5.9, 3.2)

#plt.show()
plt.savefig('../../kuvat/edf.pdf', bbox_inches='tight')
