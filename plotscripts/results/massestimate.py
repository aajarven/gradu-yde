# -*- coding:utf-8 -*-

from __future__ import print_function

import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')
sys.path.insert(0,
				'/home/ad/fshome1/u1/a/aajarven/Linux/.local/lib/python2.7/site-packages/sklearn')


import filereader
import physUtils
from savePCdata import readAndSave
from sibeliusConstants import *
import timingargument
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import colors
import numpy as np
import os.path

from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import train_test_split
import sklearn

print(sklearn.__version__)

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

	y = masses

	data = np.array([H0s, zeropoints, inClusterZeros, outClusterZeros,
				  allDispersions, radialVelocities, unclusteredDispersions,
				  radialVelocities, tangentialVelocities, LGdistances]).T

	
	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)
	params = {'text.latex.preamble' : [r'\usepackage{wasysym}']}
	plt.rcParams.update(params)


	##### Principal components #####
	# https://github.com/jcrouser/islr-python/blob/master/Lab%2011%20-%20PCR%20and%20PLS%20Regression%20in%20Python.ipynb
#	pca = PCA()
#	data_pca = pca.fit_transform(scale(data))
#	components = pca.components_
#
#	n_folds = 10
#	random_state = 1
#
#	print("component\tH0s\tzeropoints\tinClusterZeros\toutClusterZeros\t" + 
#	   "allDispersions\tclusterDispersions\tunclusteredDispersions\t" + 
#	   "radialVelocities\ttangentialVelocities\tLGdistances")
#	for i in range(len(components)):
#		print(str(i+1), end='\t')
#		for component in components[i]:
#			print("{:.3f}".format(component), end='\t')
#		print()
#	print()
#
#	# Explained variance
#	plt.plot(np.array(range(len(pca.explained_variance_ratio_)))+1,
#		  pca.explained_variance_ratio_*100, '-o', linewidth=2.0, color='k')
#	plt.xlabel("Number of component")
#	plt.ylabel("Percentage of variance explained by component")
#	plt.savefig(outputdir + "PCA-variances.svg")
#	plt.cla()
#	plt.clf()
#
#	# Cumulative variance
#	cumVariances = np.zeros(len(components))
#	print("\nExplained variance (cumulative):")
#	for i in range(len(cumVariances)):
#		cumVariances[i] = np.sum(pca.explained_variance_ratio_[:i+1])
#		print(str(i+1) + ":\t" + str(cumVariances[i]))
#	print()
#	plt.plot(np.array(range(len(cumVariances)))+1,
#		  cumVariances*100, '-o', linewidth=2.0, color='k')
#	plt.ylim(0, 100)
#	plt.xlabel("Number of component")
#	plt.ylabel("Cumulative variance explained by first components (\% of total)")
#	plt.savefig(outputdir + "PCA-cumvariances.svg")
#	plt.cla()
#	plt.clf()
#	
#
#	plt.cla()
#	plt.clf()
#	n = len(data_pca)
#	cv = cross_validation.KFold(n, n_folds=n_folds, shuffle=True, random_state=random_state)
#	regr = LinearRegression()
#	mse = []
#
#	# MSE for different number of PCs, in[14]
#	for i in np.arange(1, 10):
#		score = -1*cross_validation.cross_val_score(regr, data_pca[:,:i],
#											  y.ravel(), cv=cv,
#											  scoring='mean_squared_error').mean()
#		mse.append(score)
#	
#	plt.plot(np.array(range(len(mse)))+1, mse, '-o', color='k')
#	plt.xlabel('Number of principal components in regression')
#	plt.ylabel('MSE')
#	plt.title('Mean squared error of the points from the fit')
#	plt.savefig(outputdir + "PCA-MSE-cv" + str(n_folds) + ".svg")
#
#	
#	# train and test sets, in[16]
#	plt.cla()
#	plt.clf()
#
#	pca2 = PCA()
#	indices = np.arange(0, len(y))
#	train_indices, test_indices = cross_validation.train_test_split(indices,
#																 test_size=0.25,
#																 random_state=random_state)
#	data_train = data[train_indices]
#	data_test = data[test_indices]
#	y_train = y[train_indices]
#	y_test = y[test_indices]
#	timing_test = timingArgumentMasses[test_indices]
#
#	data_pca_train = pca2.fit_transform(scale(data_train))
#	
#	n = len(data_pca_train)
#	cv = cross_validation.KFold(n, n_folds=n_folds, shuffle=True, random_state=random_state)
#
#	# plot errors with different numbers of PCs
#	mse = []
#	for i in np.arange(1, 10):
#			score = -1*cross_validation.cross_val_score(regr,
#													 data_pca_train[:,:i],
#													 y_train.ravel(), cv=cv,
#													 scoring='mean_squared_error').mean()
#			mse.append(score)
#
#			plt.plot(np.array(range(len(mse)))+1, np.array(mse), '-o', color='k')
#			plt.xlabel('Number of principal components in regression')
#			plt.title('Mean squared error of the points in the training set' + 
#			 ' from the fit')
#			plt.ylabel('MSE')
#
#	plt.savefig(outputdir + "PCA-trainresults-cv" + str(n_folds) + ".svg")
#
#	mses = []
#	# testing, in[17]
#	# Prediction with test data
#	for i in np.arange(1, 10):
#		data_pca_test = pca2.transform(scale(data_test))[:,:i]
#		regr = LinearRegression()
#		regr.fit(data_pca_train[:,:i], y_train)
##		print(regr.coef_)
#		pred = regr.predict(data_pca_test)
#		mse = mean_squared_error(y_test, pred)
#		mses.append(mse)
#		print("Test data MSE with " + str(i) + " PCs:" + "\t" + str(mse))
#
#	# testing timing argument performance
#	timing_mse = mean_squared_error(y_test, timing_test)
#
#	plt.cla()
#	plt.clf()
#	plt.plot(np.arange(1, 10), np.array(mses), '-o', color='k', linewidth=2.0)
#	plt.plot(np.arange(1, 10), np.ones(9)*timing_mse, color='r', linewidth=2.0)
#	plt.gca().set_ylim(bottom=0)
#	plt.xlabel('Number of principal components')
#	plt.ylabel(r'MSE ($M_{\astrosun}$)')
#	plt.title('MSE of test set values from fit to training set')
#	plt.savefig(outputdir + "PCA-testresults.svg")
#
#	plt.cla()
#	plt.clf()
#
#	mse = []
#	cv = cross_validation.KFold(n, n_folds=n_folds, shuffle=True, random_state=2)
#
#	for i in np.arange(1, 10):
#		pls = PLSRegression(n_components=i, scale=False)
#		pls.fit(scale(data_pca),y)
#		score = cross_validation.cross_val_score(pls, data_pca, y, cv=cv, scoring='mean_squared_error').mean()
#		mse.append(-score)
#
#	plt.plot(np.arange(1, 10), np.array(mse), '-o', color='k', linewidth=2.0)
#	plt.xlabel('Number of principal components in PLS regression')
#	plt.ylabel('MSE')
#	plt.title('MSE of the points from the fit using PLSRegression')
#	plt.savefig(outputdir + "PCR-cv" + str(n_folds) + ".svg")
#	plt.cla()
#	plt.clf()

	# effect of tangential velocity on TA mass:
	p = plt.scatter(masses, timingArgumentMasses,
			 c=tangentialVelocities/radialVelocities, cmap='magma', s=30,
			 vmax=0.6,# vmin=0.01,
			 norm=colors.LogNorm())
	cbar = plt.colorbar(label=r'$v_{t}$ $v_{r}^{-1}$', extend='max')
	minorticks = p.norm([0.1, 0.11, 0.12, 0.13])#np.arange(0.05, 0.6, 0.1))
	cbar.ax.xaxis.set_ticks(minorticks, minor=True)
#	cbar.ax.minorticks_on()
	plt.xlabel(r'LG mass (M_{\astrosun})')
	plt.ylabel(r'LG mass from timing argument (M_{\astrosun})')
	plt.xlim(0.0, 8.0e12)
	plt.ylim(0.0, 8.0e12)
#	plt.show()
	plt.savefig(outputdir + "masscomparison.svg")
	plt.cla()
	plt.clf()


	####
	# PCA v 2.0
	####
	seed = 7
	X_train, X_test, Y_train, Y_test = train_test_split(data, y,
													 test_size=0.25,
													 random_state=seed)
	scaler = preprocessing.StandardScaler().fit(X_train)
	X_train_scaled = scaler.transform(X_train)
	Y_train_scaled = scaler.transform(Y_train)
	X_test_scaled = scaler.transform(X_test_scaled)
	Y_test_scaled = scaler.transform(Y_test_scaled)
	print(X_train_scaled)
	print(Y_train_scaled)
#	data_train_scaled = 

	pca = PCA()
	data_pca = pca.fit_transform(scale(data))
	components = pca.components_
