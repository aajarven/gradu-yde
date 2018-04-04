# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pylab
import random
from scipy.stats import linregress
from scipy import odr

def calculateY(x, slope, intercept):
	return intercept + slope * x

def line(B, x):
	return B[0]*x + B[1]

def orthogonalIntercept(x0, y0, k, b):
	"""
	Finds the point at which a line orthogonal to y = kx + b and passing
	through point (x0, y0) intercepts the original line
	"""
	x = x0 + (y0*k - b*k - x0*k**2)/(k**2 + 1)
	y = b + x0*k + (y0*k**2 - b*k**2 - x0*k**3)/(k**2 + 1)
	return (x, y)

if __name__ == "__main__":
	np.random.seed(7155)
	saveloc_OLS = "../../kuvat/OLS.svg"
	saveloc_OLSproblem = "../../kuvat/OLSproblem.svg"
	saveloc_TLS = "../../kuvat/TLS.svg"

	N = 30
	x = np.random.uniform(-10.0, 10.0, N)
	y = [xx - 0.2*xx + np.random.normal(0.0, 3) for xx in x]

	slope1, intercept1, r_value1, p_value1, std_err1 = linregress(x,y)
	slope2, intercept2, r_value2, p_value2, std_err2 = linregress(y,x)
	plotxlimit = 12

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
		 color='k', linewidth=2.0, label="fit using vertical\ndistances")

	ax.plot([-plotxlimit, plotxlimit], [(plotxlimit + intercept2)*-1.0/slope2,
									 (plotxlimit - intercept2)/slope2],
		 color='k', linewidth=2.0, linestyle="--",
		 label="fit using horizontal\ndistances")

	ax.legend(bbox_to_anchor=(1.02, 0.35), frameon=False)

	ax.set_xlim(-1*plotxlimit, plotxlimit)
	ax.set_ylim(min(y)-0.5, max(y)+0.5)
	ax.set_xlim(-1*plotxlimit, plotxlimit)
	ax.set_ylim(min(y)-0.5, max(y)+0.5)
	ax.set_aspect('equal')

	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)

	plt.margins(y=0.3)
	plt.tight_layout()
	plt.axis('off')
	f.set_size_inches(3.5, 3)
	plt.savefig(saveloc_OLSproblem)

	plt.cla()
	plt.clf()

	# TLS
	linear_model = odr.Model(line)
	data = odr.Data(x, y)
	linear_odr = odr.ODR(data, linear_model, beta0=[slope1, intercept1])
	slope3, intercept3 = linear_odr.run().beta
	
	f = plt.figure()
	ax = f.add_subplot(111)
	ax.scatter(x, y, color='k', s=20, zorder=3)
	ax.plot([-plotxlimit, plotxlimit], [intercept3 - slope3 * plotxlimit,
									 intercept3 + slope3 * plotxlimit],
		 color='k', linewidth=2.0, zorder=2)
	for x0, y0 in zip(x, y):
		xx, yy = orthogonalIntercept(x0, y0, slope3, intercept3)
		ax.plot([x0, xx], [y0, yy], color='k', zorder=1)

	ax.set_xlim(-1*plotxlimit, plotxlimit)
	ax.set_ylim(min(y)-0.5, max(y)+0.5)

	plt.tight_layout()
	plt.subplots_adjust(wspace=0)
	plt.axis('off')
	f.set_size_inches(3, 3)
	plt.savefig(saveloc_TLS)
