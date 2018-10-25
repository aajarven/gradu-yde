# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import math
import pylab
import random
from scipy import stats

def CDFifyXData(array):
	array = np.repeat(array, 2)
	array = np.delete(array, len(array)-1)
	return array


# find biggest difference between curves
def biggestDifference(data1X, data2X, y):
	bigDiff = 0 
	bigDiffX = 0
	bigDiffLowY = 0
	bigDiffHighY = 0
	index1 = 0
	index2 = 0
	
	while index1 < len(data1X)-1 and index2 < len(data2X)-1:
		if data1X[index1] < data2X[index2]:
			index1 = index1 + 1
			mod1 = -1
			mod2 = 0
		else:
			index2 = index2 + 1
			mod1 = 0
			mod2 = -1

		if abs(y[index1]-y[index2]) > bigDiff:
			bigDiff = abs(y[index1]-y[index2])
			bigDiffX = min(data1X[index1], data2X[index2])
			bigDiffLowY = min(y[index1], y[index2])
			bigDiffHighY = max(y[index1], y[index2])
	
	return (bigDiffX, bigDiffLowY, bigDiffHighY)

if __name__ == "__main__":
	np.random.seed(100)
	
	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)

	N = 35
	normalNumbers = np.random.normal(16.0, 2.0, N)
	normalNumbersWide = np.random.normal(15.0, 4.0, N)
	normalNumbersWide2 = np.random.normal(15.0, 4.0, N)

	(D1, pval1) = stats.ks_2samp(normalNumbers, normalNumbersWide)
	(D2, pval2) = stats.ks_2samp(normalNumbersWide, normalNumbersWide2)
	print("Normal vs Wide")
	print("D:\t"+str(D1))
	print("p:\t"+str(pval1))
	print("Wide vs wide2")
	print("D:\t"+str(D2))
	print("p:\t"+str(pval2))


	normalNumbers.sort()
	normalNumbersWide.sort()
	normalNumbersWide2.sort()

	biggest = max(max(np.amax(normalNumbers), np.amax(normalNumbersWide)),
				  np.amax(normalNumbersWide2))

	percentage = np.arange(0, 100, 100.0/N)

	(ksNormalWideX, ksNormalWideYLow, ksNormalWideYHigh) = biggestDifference(
		normalNumbers, normalNumbersWide, percentage)
	(ksWideWideX, ksWideWideYLow, ksWideWideYHigh) = biggestDifference(normalNumbersWide, normalNumbersWide2, percentage)

	percentage = np.append(percentage, 100)
	percentage = np.repeat(percentage, 2)
	percentage = np.delete(percentage, 0)

	normalNumbers = np.append(normalNumbers, biggest)
	normalNumbersWide = np.append(normalNumbersWide, biggest)
	normalNumbersWide2 = np.append(normalNumbersWide2, biggest)

	normalNumbers = CDFifyXData(normalNumbers)
	normalNumbersWide = CDFifyXData(normalNumbersWide)
	normalNumbersWide2 = CDFifyXData(normalNumbersWide2)

	plt.plot(normalNumbers, percentage, label='$\mu$=16.0, $\sigma$=2.0',
			 linewidth=2.0, color='g')
	plt.plot(normalNumbersWide, percentage, label='$\mu$=15.0, $\sigma$=4.0',
			 linewidth=2.0, color='b')
	plt.plot(normalNumbersWide2, percentage, label='$\mu$=15.0, $\sigma$=4.0',
			 linewidth=2.0, color='c')

	yLimits = [0, 100]
	axes = plt.gca()
	axes.set_ylim(yLimits)
	plt.axvline(x=ksNormalWideX, ymin=ksNormalWideYLow/(yLimits[1]-yLimits[0]),
				ymax=ksNormalWideYHigh/(yLimits[1]-yLimits[0]), color='m',
				linewidth=2)
	plt.axvline(x=ksWideWideX, ymin=ksWideWideYLow/(yLimits[1]-yLimits[0]),
				ymax=ksWideWideYHigh/(yLimits[1]-yLimits[0]), color='m',
				linewidth=2)

	plt.xlabel('x')
	plt.ylabel('EDF \enspace (\%)')
	plt.legend(loc=0)
	plt.xlim([2, 28])


	F = pylab.gcf()
	F.set_size_inches(5.9, 3.2)

	plt.savefig('../../kuvat/kstest.pdf', bbox_inches='tight')
