# -*- coding:utf-8 -*-

from __future__ import print_function

import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import filereader
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import pylab

distances = filereader.readTextTable('../../misc/contaminationdistances.txt',
									 columnindex=0, datatype='float')
indexes = range(len(distances))
distances = np.sort(distances)

rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 13})

f, ax = plt.subplots(1, 1)

ax.plot(distances[::-1], indexes, color='k')
ax.set_xlabel("Range containing only type 1 particles")
ax.set_ylabel("Number of simulations")
ax.set_xlim(min(distances), max(distances))
ax.set_ylim(0, math.ceil(max(indexes)/10.0)*10)

F = pylab.gcf()
F.set_size_inches(5, 3)

plt.savefig('../../kuvat/uncontaminatedDistances.svg', bbox_inches='tight')
