# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import math
import pylab
import random

random.seed(715517)

N = 150
normalNumbers = np.random.normal(10.0, 2.0, N)
normalNumbersLow = np.random.normal(5.0, 2.0, N)
normalNumbersWide = np.random.normal(10.0, 4.0, N)
normalNumbersWide2 = np.random.normal(10.0, 4.0, N)

biggest = max(max(max(np.amax(normalNumbers), np.amax(normalNumbersLow)),
        np.amax(normalNumbersWide)), np.amax(normalNumbersWide2))

percentage = np.arange(0, 100, 100.0/N)
percentage = np.append(percentage, 100)

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.subplots_adjust(hspace=0)

bins = np.arange(-2, biggest+1, 1.5)
ax2.hist(normalNumbers, bins, histtype='step', stacked=True, fill=False,
        linewidth=2.0)
ax2.hist(normalNumbersLow, bins, histtype='step', stacked=True, fill=False,
        linewidth=2.0)
ax2.hist(normalNumbersWide, bins, histtype='step', stacked=True, fill=False,
        linewidth=2.0)
ax2.hist(normalNumbersWide2, bins, histtype='step', stacked=True, fill=False,
        linewidth=2.0)

# hide topmost ticklabel on lower plot
lastTickLabel = ax2.get_yticklabels()
lastTickLabel = lastTickLabel[len(lastTickLabel)-1]
plt.setp(lastTickLabel, visible=False)

normalNumbers.sort()
normalNumbersLow.sort()
normalNumbersWide.sort()
normalNumbersWide2.sort()

normalNumbers = np.append(normalNumbers, biggest)
normalNumbersLow = np.append(normalNumbersLow, biggest)
normalNumbersWide = np.append(normalNumbersWide, biggest)
normalNumbersWide2 = np.append(normalNumbersWide2, biggest)

ax1.plot(normalNumbers, percentage, label='$\mu$=10.0, $\sigma$=2.0',
		 linewidth=2.0)
ax1.plot(normalNumbersLow, percentage, label='$\mu$=5.0, $\sigma$=2.0',
		 linewidth=2.0)
ax1.plot(normalNumbersWide, percentage, label='$\mu$=10.0, $\sigma$=4.0',
		 linewidth=2.0)
ax1.plot(normalNumbersWide2, percentage, label='$\mu$=10.0, $\sigma$=4.0',
		 linewidth=2.0)

plt.xlabel('values')
#ax1.ylabel('CDF')
ax1.legend(loc=0)
plt.xlim([-2, math.ceil(biggest)])

rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

F = pylab.gcf()
F.set_size_inches(5.90666, 8)

#plt.show()
plt.savefig('../kuvat/edf.png')
