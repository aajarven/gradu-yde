# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import math
import pylab
import random


def normPDF(x, mu, sigma):
    return (1.0/(math.sqrt(2*np.pi)*sigma)) * np.exp(-0.5*((x-mu)/sigma)**2)

def uniformPDF(x, minX, maxX):
    return [(xx>minX and xx<maxX)/(maxX*1.0-minX*1.0) for xx in x]


np.random.seed(715517)

rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

N = 100000
normalNumbers = np.random.normal(16.0, 2.0, N)
normalNumbersWide = np.random.normal(15.0, 4.0, N)
uniform = np.random.uniform(5, 25, N)

biggest = max(max(np.amax(normalNumbers), np.amax(normalNumbersWide)),
	np.amax(uniform))

percentage = np.arange(0, 100, 100.0/N)
percentage = np.append(percentage, 100)

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.subplots_adjust(hspace=0)

normalNumbers.sort()
normalNumbersWide.sort()
uniform.sort()

normalNumbers = np.append(normalNumbers, biggest)
normalNumbersWide = np.append(normalNumbersWide, biggest)
uniform = np.append(uniform, biggest)

ax1.plot(normalNumbers, percentage, label='$\mu$=16.0, $\sigma$=2.0',
		 linewidth=2.0, color='C0')
ax1.plot(normalNumbersWide, percentage, label='$\mu$=15.0, $\sigma$=4.0',
		 linewidth=2.0, color='C1')
ax1.plot(uniform, percentage, label='uniform',
		 linewidth=2.0, color='C2')
ax1.set_ylabel('CDF \enspace (\%)')
ax1.set_ylim([0, 100])

pdf1 = normPDF(normalNumbers, 16, 2)
pdf2 = normPDF(normalNumbersWide, 15, 4)
uniform = np.insert(uniform, len(uniform)-1, 25)
uniform = np.insert(uniform, 0, 5)
pdf3 = uniformPDF(uniform, 5, 25)
ax2.plot(normalNumbers, pdf1, linewidth=2.0, label='$\mu$=16.0, $\sigma$=2.0',
		color='C0')
ax2.plot(normalNumbersWide, pdf2, linewidth=2.0, label='$\mu$=15.0, $\sigma$=4.0',
		 color='C1')
ax2.plot(uniform, pdf3, linewidth=2.0, label='uniform', color='C2')
ax2.set_ylabel('PDF')
ax2.set_ylim([0, .22])

# hide topmost ticklabel on lower plot
lastTickLabel = ax2.get_yticklabels()
lastTickLabel = lastTickLabel[len(lastTickLabel)-1]
plt.setp(lastTickLabel, visible=False)

plt.xlabel('x')
#ax1.ylabel('CDF')
ax1.legend(loc=0)
plt.xlim([2, 28])

F = pylab.gcf()
F.set_size_inches(5.9, 5)

#plt.show()
plt.savefig('../../kuvat/cdf.pdf', bbox_inches='tight')
