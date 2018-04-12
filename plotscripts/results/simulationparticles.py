# -*- coding:utf-8 -*-

from __future__ import print_function

import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import filereader
import math
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import numpy as np

if __name__ == "__main__":
	infile = "../../misc/particlenumbers-noduplicates.txt"

	dm1 = np.asarray(filereader.readTextTable(infile, columnindex=0,
										   separator=','), dtype=int)
	dm2 = np.asarray(filereader.readTextTable(infile, columnindex=1,
										   separator=','), dtype=int)
	dm3 = np.asarray(filereader.readTextTable(infile, columnindex=2,
										   separator=','), dtype=int)
	
	nonzeroMask = dm1 != 0
	dm1 = dm1[nonzeroMask]
	dm2 = dm2[nonzeroMask]
	dm3 = dm3[nonzeroMask]

	fig = plt.figure()

	rc('font', **{'family':'serif','serif':['']})
	rc('text', usetex=True)
	matplotlib.rcParams['text.latex.unicode'] = True
	
	dm1 = dm1
	binlimits = np.arange(0, 3.1e7, 2e6)
	plt.hist(dm1, bins=binlimits, color='0.75', edgecolor='black')
	plt.xticks(binlimits)
	plt.xlim(min(binlimits), max(binlimits))
	plt.xlabel("Number of type 1 dark matter particles")	

	fig.set_size_inches(5, 3.5)
	plt.tight_layout()
	
	plt.savefig("../../kuvat/type1particles.svg")
