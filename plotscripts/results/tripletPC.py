# -*- coding:utf-8 -*-

from __future__ import print_function

import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import filereader
import physUtils
from savePCdata import readAndSave
from sibeliusConstants import *
import timingargument
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import os.path

from sklearn.preprocessing import scale
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error



if __name__ == "__main__":
	simulationfiles = ("/home/aajarven/Z-drive/duuni/extragal/gradu-yde/"
					"plotscripts/input/upTo5Mpc-fullpath.txt")
	datafile = ("/home/aajarven/Z-drive/duuni/extragal/gradu-yde/plotscripts/"
			 "output/massdata.txt")
	outputdir = "/home/aajarven/Z-drive/duuni/extragal/gradu-yde/kuvat/PCA/triplets/"
	random_state = 1
	n_folds = 10


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
	allHaloesSanitymask = np.array([zeropoint < 3.0 and zeropoint > -3.0 for zeropoint
					   in zeropoints])
	inClusterSanitymask = np.array([zeropoint < 4.0 and zeropoint > -5.0 for zeropoint
					   in inClusterZeros])
	outClusterSanitymask = np.array([zeropoint < 4.0 and zeropoint > -1.0 for zeropoint
					   in outClusterZeros])
	sanitymask = np.logical_and(allHaloesSanitymask,
							 np.logical_and(inClusterSanitymask,
					   outClusterSanitymask))

	masses = masses[sanitymask]
	zeropoints = zeropoints[sanitymask]
	inClusterZeros = inClusterZeros[sanitymask]
	outClusterZeros = outClusterZeros[sanitymask]
	allDispersions = allDispersions[sanitymask]
	clusterDispersions = clusterDispersions[sanitymask]
	unclusteredDispersions = unclusteredDispersions[sanitymask]

	y = masses

	dataTriplets = (
		(np.array([zeropoints, inClusterZeros, outClusterZeros]).T, "zeros",
   ["HF zero", "HF zero from clustered", "HF zero from non-clustered"]),
		(np.array([allDispersions, clusterDispersions,
			 unclusteredDispersions]).T, "dispersions", 
   ["HF velocity dispersion", "HF velocity dispersion from clustered", 
	"HF velocity dispersion from non-clustered"]))

	
	for (data, name, variables) in dataTriplets:
		pca = PCA()
		data_pca = pca.fit_transform(scale(data))
		components = pca.components_
	
		# print components
		print()
		print(variables)
		for i in range(len(components)):
			print(str(i+1), end='\t')
			for component in components[i]:
				print("{:.3f}".format(component), end='\t')
			print()
		print()

		rc('font', **{'family':'serif','serif':['Palatino']})
		rc('text', usetex=True)
		params = {'text.latex.preamble' : [r'\usepackage{wasysym}']}
		plt.rcParams.update(params)

		# Explained variance
		plt.plot(np.array(range(len(pca.explained_variance_ratio_)))+1,
			  pca.explained_variance_ratio_*100, '-o', linewidth=2.0, color='k')
		plt.ylim(0, 100)
		plt.xticks([1, 2, 3])
		plt.xlabel("Number of component")
		plt.ylabel("Percentage of variance explained by component")
		plt.title(name)
		plt.savefig(outputdir + name + "-PCA-variances.svg")
		plt.cla()
		plt.clf()


		plt.cla()
		plt.clf()
		n = len(data_pca)
		cv = cross_validation.KFold(n, n_folds=n_folds, shuffle=True, random_state=random_state)
		regr = LinearRegression()
		mse = []
		
		plt.cla()
		plt.clf()

		pca2 = PCA()
		indices = np.arange(0, len(y))
		train_indices, test_indices = cross_validation.train_test_split(indices,
																	 test_size=0.5,
																	 random_state=random_state)

		data_train = data[train_indices]
		data_test = data[test_indices]
		y_train = y[train_indices]
		y_test = y[test_indices]

		data_pca_train = pca2.fit_transform(scale(data_train))
		
		n = len(data_pca_train)
		cv = cross_validation.KFold(n, n_folds=n_folds, shuffle=True, random_state=random_state)

		mses = []
		for i in np.arange(1, 4):
			data_pca_test = pca2.transform(scale(data_test))[:,:i]
			regr = LinearRegression()
			regr.fit(data_pca_train[:,:i], y_train)
			pred = regr.predict(data_pca_test)
			mse = mean_squared_error(y_test, pred)
			mses.append(mse)


		plt.cla()
		plt.clf()
		plt.plot(np.arange(1, 4), np.array(mses), '-o', color='k', linewidth=2.0)
		plt.gca().set_ylim(bottom=0)
		plt.xlabel('Number of principal components')
		plt.ylabel(r'MSE ($M_{\astrosun}$)')
		plt.title('MSE of test set values from fit to training set')
		plt.savefig(outputdir + name + "-PCA-testresults.svg")
