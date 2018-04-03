# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pylab
import random
from scipy.stats import linregress


def calculateY(x, slope, intercept):
	return intercept + slope * x

if __name__ == "__main__":
	np.random.seed(7155)
	saveloc_OLS = "../../kuvat/OLS.svg"
	saveloc_OLSproblem = "../../kuvat/OLSproblem.svg"

	N = 30
	x = np.random.uniform(-10.0, 10.0, N)
	y = [xx + np.random.normal(0.0, 3) for xx in x]

	slope1, intercept1, r_value1, p_value1, std_err1 = linregress(x,y)
	slope2, intercept2, r_value2, p_value2, std_err2 = linregress(y,x)
	plotxlimit = 11

	# minimized distances

	f = plt.figure()
	ax = f.add_subplot(111)

	for xx, yy in zip(x, y):
		ax.plot([xx, xx], [yy, calculateY(xx, slope1, intercept1)],
		  color='0.4', zorder=1)

	ax.scatter(x, y, color='k', s=20, zorder=3)
	ax.plot([-plotxlimit, plotxlimit], [intercept1 - slope1*plotxlimit,
									 intercept1 + slope1*plotxlimit],
		 color='k', linewidth=2.0, zorder=2)
	ax.set_xlim(-1*plotxlimit, plotxlimit)
	ax.set_ylim(min(y)-0.5, max(y)+0.5)

	plt.tight_layout()
	plt.subplots_adjust(wspace=0)
	plt.axis('off')
	f.set_size_inches(3, 3)
	plt.savefig(saveloc_OLS)

	plt.cla()
	plt.clf()

	# the two lines for fitting x to y and y to x
	f = plt.figure()
	ax = f.add_subplot(111)
	ax.scatter(x, y, color='k', s=20)
	ax.plot([-plotxlimit, plotxlimit], [intercept1 - slope1*plotxlimit,
									 intercept1 + slope1*plotxlimit],
		 color='k', linewidth=2.0, label="vertical distances")

	ax.plot([-plotxlimit, plotxlimit], [(plotxlimit + intercept2)*-1.0/slope2,
									 (plotxlimit - intercept2)/slope2],
		 color='k', linewidth=2.0, linestyle="--",
		 label="horizontal distances")

#	plt.legend(loc=2)
#	ax.legend(bbox_to_anchor=(1.1, 0.25), frameon=False)
	ax.legend(bbox_to_anchor=(0.78, 1.06), frameon=False)

	ax.set_xlim(-1*plotxlimit, plotxlimit)
	ax.set_ylim(min(y)-0.5, max(y)+0.5)
	ax.set_xlim(-1*plotxlimit, plotxlimit)
	ax.set_ylim(min(y)-0.5, max(y)+0.5)
	ax.set_aspect('equal')

	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)

	plt.margins(y=0.3)
	plt.tight_layout()
#	plt.subplots_adjust(wspace=0)
	plt.axis('off')
	f.set_size_inches(3.5, 3)
	plt.savefig(saveloc_OLSproblem)

