# -*- coding:utf-8 -*-

from __future__ import print_function

import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
import os

if __name__ == "__main__":
	simulationfiles = ("/home/aajarven/Z-drive/duuni/extragal/gradu-yde/"
					"plotscripts/input/upTo5Mpc-fullpath.txt")
	datafile = ("/home/aajarven/Z-drive/duuni/extragal/gradu-yde/plotscripts/"
			 "output/massdata.txt")
	outputdir = "/home/aajarven/Z-drive/duuni/extragal/gradu-yde/kuvat/PCA/"

	if not os.path.isfile(datafile):
		data = readAndSave(simulationfiles, datafile, mindist=1.0, maxdist=5.0,
			  eps=1.8, ms=10)
	else:
		data = np.loadtxt(datafile)

	result = np.hsplit(data, 12)
	(masses, timingArgumentMasses, H0s, zeropoints, inClusterZeros, outClusterZeros,
   allDispersions, unclusteredDispersions, clusterDispersions,
   radialVelocities, tangentialVelocities, LGdistances) = result

	# masking zeropoints
	allHaloesSanitymask = np.array([zeropoint < 5.0 and zeropoint > -5.0 for zeropoint
					   in zeropoints])
	inClusterSanitymask = np.array([zeropoint < 5.0 and zeropoint > -5.0 for zeropoint
					   in inClusterZeros])
	outClusterSanitymask = np.array([zeropoint < 5.0 and zeropoint > -5.0 for zeropoint
					   in outClusterZeros])
	sanitymask = np.logical_and(allHaloesSanitymask,
							 np.logical_and(inClusterSanitymask,
					   outClusterSanitymask))
	# apply mask and scale (zero mean and unit variance)
	masses = masses[sanitymask]
	timingArgumentMasses = timingArgumentMasses[sanitymask]
	H0s = H0s[sanitymask]
	zeropoints = zeropoints[sanitymask]
	inClusterZeros = inClusterZeros[sanitymask]
	outClusterZeros = outClusterZeros[sanitymask]
	allDispersions = allDispersions[sanitymask]
	clusterDispersions = clusterDispersions[sanitymask]
	unclusteredDispersions = unclusteredDispersions[sanitymask]
	radialVelocities = radialVelocities[sanitymask]
	tangentialVelocities = tangentialVelocities[sanitymask]
	LGdistances = LGdistances[sanitymask]


	plotdata = [(masses*1e-12, 'LG mass'), (timingArgumentMasses*1e-12, 'mass from TA'),
			 (H0s, 'HF slope'), (zeropoints, 'HF zero'),
			 (inClusterZeros, 'HF zero for clusters'),
			 (outClusterZeros, 'HF zero for non-clustered'),
			 (allDispersions, 'HF velocity dispersion'),
			 (clusterDispersions, 'HF velocity dispersion in clusters'),
			 (unclusteredDispersions, 'HF velocity dispersion outside clusters'),
			 (radialVelocities, 'radial velocity of M31'),
			 (tangentialVelocities, 'tangential velocity of M31'),
			 (LGdistances, 'distance to M31')]

	# tarvitaanko tätä?
	smallestMin = 0
	biggestMax = 0
	for i in range(len(plotdata)):
		plotdata[i] = (scale(plotdata[i][0]), plotdata[i][1])
		if smallestMin > min(plotdata[i][0]):
			smallestMin = min(plotdata[i][0])
		if biggestMax < max(plotdata[i][0]):
			biggestMax = max(plotdata[i][0])
#		print(str(min(plotdata[i][0])) + "\t" + str(max(plotdata[i][0])))
#	print(str(smallestMin) + "\t" + str(biggestMax))

	# making the scatter matrix
	fig, axes = plt.subplots(nrows=len(plotdata), ncols=len(plotdata),
						  figsize=(10,10))
	fig.subplots_adjust(hspace=0.1, wspace=0.1)
	plt.gcf().subplots_adjust(bottom=0.3)
	plt.gcf().subplots_adjust(left=0.3)
	plt.gcf().subplots_adjust(right=0.99)
	plt.gcf().subplots_adjust(top=0.99)

	for col in range(len(plotdata)):
		for row in range(len(plotdata)):
			ax = axes[row, col]
			ax.xaxis.set_visible(False)
			ax.yaxis.set_visible(False)
			ax.set_xlim(math.floor(min(plotdata[col][0])),
			   math.ceil(max(plotdata[col][0])))
			ax.set_ylim(math.floor(min(plotdata[row][0])),
			   math.ceil(max(plotdata[row][0])))
			ax.xaxis.set_ticklabels([])
			ax.yaxis.set_ticklabels([])
#			print(str(col) + ", " + str(row))

			if ax.is_first_col():
				ax.yaxis.set_visible(True)
				ax.yaxis.set_ticks_position('left')
				ax.set_ylabel(plotdata[row][1], rotation='horizontal',
				  size='small', horizontalalignment='right')
#				print(str(col) + ", " + str(row) + ": " + plotdata[row][1])
			if ax.is_last_row():
				ax.xaxis.set_visible(True)
				ax.xaxis.set_ticks_position('bottom')
				ax.set_xlabel(plotdata[col][1], rotation='vertical', size='small')
#				print(str(col) + ", " + str(row) + ": " + plotdata[col][1])
			
#			if col == 0:
#				print(plotdata[col][1] + "\t\t" + plotdata[row][1])
#				print(str(ax.get_xlim()) + "\t" + str(ax.get_ylim()))
#				print(plotdata[col][0])
			
			ax.scatter(plotdata[col][0], plotdata[row][0], marker='.', s=6,
			  edgecolors='none', facecolors='k')
	
	plt.savefig(outputdir + "scattermatrix.svg")
