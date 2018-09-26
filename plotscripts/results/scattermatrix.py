# -*- coding:utf-8 -*-

from __future__ import print_function

import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import math
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from sklearn.preprocessing import scale
from optparse import OptionParser
import os
from savePCdata import readAndSave

if __name__ == "__main__":

	parser = OptionParser(usage="usage: python scattermatrix.py [options]")
	parser.add_option("--outlierexclusion", action="store", default="tight",
				   help="exclude outliers, based on HF zero point distances. " +
				   "Criteria options none/loose/tight, default tight.",
				   dest="outlierExclusion")
	(opts, args) = parser.parse_args()

	simulationfiles = ("/home/aajarven/Z-drive/duuni/extragal/gradu-yde/"
					"plotscripts/input/upTo5Mpc-no229-fullpath.txt")
	datafile = ("/home/aajarven/Z-drive/duuni/extragal/gradu-yde/plotscripts/"
			 "output/massdata.txt")
	outputdir = "/home/aajarven/Z-drive/duuni/extragal/gradu-yde/kuvat/PCA/"

	if not os.path.isfile(datafile):
		data = readAndSave(simulationfiles, datafile, mindist=1.0, maxdist=5.0,
			  eps=0.16, ms=4, scale_eps=False)
	else:
		data = np.loadtxt(datafile)

	result = np.hsplit(data, 14)
	(masses, timingArgumentMasses, H0s, inClusterH0s, outClusterH0s,
  zeropoints, inClusterZeros, outClusterZeros, allDispersions,
  unclusteredDispersions, clusterDispersions, radialVelocities,
  tangentialVelocities, LGdistances) = result

	if opts.outlierExclusion != "none" and opts.outlierExclusion != "loose" and	opts.outlierExclusion != "tight":
		print("Unexpected value in outlierexclusion parameter. Allowed values"
		+ " are none (all data points are plotted), loose (-5 to 5 kpc range"
		+ " allowed) and tight (custom criteria creating visually satisfying"
		+ "result.")

	if opts.outlierExclusion != "none":
		if opts.outlierExclusion == "tight":
			allHaloesSanitymask = np.array([zeropoint < 3.0 and zeropoint > -3.0 for zeropoint
							   in zeropoints])
			inClusterSanitymask = np.array([zeropoint < 4.0 and zeropoint > -5.0 for zeropoint
							   in inClusterZeros])
			outClusterSanitymask = np.array([zeropoint < 4.0 and zeropoint > -1.0 for zeropoint
							   in outClusterZeros])
			sanitymask = np.logical_and(allHaloesSanitymask,
									 np.logical_and(inClusterSanitymask,
							   outClusterSanitymask))
		
		elif opts.outlierExclusion == "loose":
			allHaloesSanitymask = np.array([zeropoint < 5.0 and zeropoint > -5.0 for zeropoint
							   in zeropoints])
			inClusterSanitymask = np.array([zeropoint < 5.0 and zeropoint > -5.0 for zeropoint
							   in inClusterZeros])
			outClusterSanitymask = np.array([zeropoint < 5.0 and zeropoint > -5.0 for zeropoint
							   in outClusterZeros])
			sanitymask = np.logical_and(allHaloesSanitymask,
									 np.logical_and(inClusterSanitymask,
							   outClusterSanitymask))
		# apply mask
		masses = masses[sanitymask]
		timingArgumentMasses = timingArgumentMasses[sanitymask]
		H0s = H0s[sanitymask]
		inClusterH0s = inClusterH0s[sanitymask]
		outClusterH0s = outClusterH0s[sanitymask]
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
			 (H0s, r'$H_0$'),
			 (inClusterH0s, r'$H_0$ for clusters'),
			 (outClusterH0s, r'$H_0$ for non-clustered'),
			 (zeropoints, 'HF zero'),
			 (inClusterZeros, 'HF zero for\nclusters'),
			 (outClusterZeros, 'HF zero for\nnon-clustered'),
			 (allDispersions, 'HF velocity\ndispersion'),
			 (clusterDispersions, 'HF velocity dispersion\nin clusters'),
			 (unclusteredDispersions, 'HF velocity dispersion\noutside clusters'),
			 (radialVelocities, 'radial velocity\nof M31'),
			 (tangentialVelocities, 'tangential velocity\nof M31'),
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

	# making the scatter matrix
	fig, axes = plt.subplots(nrows=len(plotdata), ncols=len(plotdata),
						  figsize=(5.7,5.7))
	fig.subplots_adjust(hspace=0.08, wspace=0.08)
	plt.gcf().subplots_adjust(bottom=0.25)
	plt.gcf().subplots_adjust(left=0.25)
	plt.gcf().subplots_adjust(right=0.99)
	plt.gcf().subplots_adjust(top=0.99)

#	rc('font', **{'family':'serif','serif':['Palatino']})
#	rc('text', usetex=True)
#	params = {'text.latex.preamble' : [r'\usepackage{wasysym}']}
#	plt.rcParams.update(params)

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

			if ax.is_first_col():
				ax.yaxis.set_visible(True)
				ax.yaxis.set_ticks_position('left')
				ax.set_ylabel(plotdata[row][1], rotation='horizontal',
				  size='x-small', horizontalalignment='right',
				  verticalalignment="center")
			if ax.is_last_row():
				ax.xaxis.set_visible(True)
				ax.xaxis.set_ticks_position('bottom')
				ax.set_xlabel(plotdata[col][1], rotation='vertical',
				  size='x-small')
			
			ax.scatter(plotdata[col][0], plotdata[row][0], marker='.', s=4,
			  edgecolors='none', facecolors='k')
	if opts.outlierExclusion == "none":
		plt.savefig(outputdir + "scattermatrix-all.pdf")
	elif opts.outlierExclusion == "loose":
		plt.savefig(outputdir + "scattermatrix-looseOutlierCriteria.pdf")
	elif opts.outlierExclusion == "tight":
		plt.savefig(outputdir + "scattermatrix-tightOutlierCriteria.pdf")
