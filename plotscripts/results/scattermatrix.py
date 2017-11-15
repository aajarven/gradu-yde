# -*- coding:utf-8 -*-

from __future__ import print_function

import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import numpy as np
import matplotlib.pyplot as plt
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

	data = scale(data, axis=0)

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
	print(np.mean(masses))
	print(np.mean(H0s))
	print(np.mean(tangentialVelocities))

	plotdata = [(masses, 'LG mass'), (timingArgumentMasses, 'mass from TA'),
			 (H0s, 'HF slope'), (zeropoints, 'HF zero'),
			 (inClusterZeros, 'HF zero for clusters'),
			 (outClusterZeros, 'HF zero for non-clustered'),
			 (allDispersions, 'HF velocity dispersion'),
			 (clusterDispersions, 'HF velocity dispersion in clusters'),
			 (unclusteredDispersions, 'HF velocity dispersion outside clusters'),
			 (radialVelocities, 'radial velocity of M31'),
			 (tangentialVelocities, 'tangential velocity of M31'),
			 (LGdistances, 'distance to M31')]

	# making the scatter matrix
	fig, axes = plt.subplots(nrows=len(plotdata), ncols=len(plotdata),
						  figsize=(10,10))
	fig.subplots_adjust(hspace=0.1, wspace=0.1)
	plt.gcf().subplots_adjust(bottom=0.3)
	plt.gcf().subplots_adjust(left=0.3)
	plt.gcf().subplots_adjust(right=0.99)
	plt.gcf().subplots_adjust(top=0.99)

	for i in range(len(plotdata)):
		for j in range(len(plotdata)):
			ax = axes[i, j]
			ax.xaxis.set_visible(False)
			ax.yaxis.set_visible(False)
			ax.xaxis.set_ticklabels([])
			ax.yaxis.set_ticklabels([])

			if ax.is_first_col():
				ax.yaxis.set_visible(True)
				ax.yaxis.set_ticks_position('left')
				ax.set_ylabel(plotdata[i][1], rotation='horizontal',
				  size='small', horizontalalignment='right')
			if ax.is_last_row():
				ax.xaxis.set_visible(True)
				ax.xaxis.set_ticks_position('bottom')
				ax.set_xlabel(plotdata[j][1], rotation='vertical', size='small')

			
			ax.scatter(plotdata[i][0], plotdata[j][0], marker='.', s=6,
			  edgecolors='none', facecolors='k')
	
	plt.savefig(outputdir + "scattermatrix.svg")
