# -*- coding:utf-8 -*-

from __future__ import print_function

import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')
#sys.path.insert(0,
#				'/home/ad/fshome1/u1/a/aajarven/Linux/.local/lib/python2.7/site-packages/sklearn')


from savePCdata import readAndSave
from sibeliusConstants import *
import timingargument
from math import sqrt, ceil
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
from sklearn import model_selection  

from sklearn.preprocessing import scale
	

if __name__ == "__main__":
	simulationfiles = ("/home/aajarven/Z-drive/duuni/extragal/gradu-yde/"
					"plotscripts/input/upTo5Mpc-fullpath.txt")
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
  unclusteredDispersions, clusterDispersions,  radialVelocities,
  tangentialVelocities, LGdistances) = result
		
	# masking zeropoints
#	allHaloesSanitymask = np.array([zeropoint < 3.0 and zeropoint > -3.0 for zeropoint
#					   in zeropoints])
#	inClusterSanitymask = np.array([zeropoint < 4.0 and zeropoint > -5.0 for zeropoint
#					   in inClusterZeros])
#	outClusterSanitymask = np.array([zeropoint < 4.0 and zeropoint > -1.0 for zeropoint
#					   in outClusterZeros])
	allHaloesSanitymask = np.array([zeropoint < 5.0 and zeropoint > -5.0 for zeropoint
					   in zeropoints])
	inClusterSanitymask = np.array([zeropoint < 5.0 and zeropoint > -5.0 for zeropoint
					   in inClusterZeros])
	outClusterSanitymask = np.array([zeropoint < 5.0 and zeropoint > -5.0 for zeropoint
					   in outClusterZeros])
	sanitymask = np.logical_and(allHaloesSanitymask,
							 np.logical_and(inClusterSanitymask,
					   outClusterSanitymask))

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

	y = masses

	data = np.array([H0s, inClusterH0s, outClusterH0s, zeropoints,
				  inClusterZeros, outClusterZeros,allDispersions,
				  radialVelocities, unclusteredDispersions, radialVelocities,
				  tangentialVelocities, LGdistances]).T

	
#	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('font', family='serif')
	rc('text', usetex=True)
	params = {'text.latex.preamble' : [r'\usepackage{wasysym}']}
	plt.rcParams.update(params)


	####
	# PCA v 2.0
	####

	
	# calculate and print components from whole dataset
	pca = PCA()
	data_pca = pca.fit_transform(scale(data))
	components = pca.components_
	
	n = len(data_pca)
	variables = data.shape[1]

	print("component\tH0s\tinClusterH0s\toutClusterH0s\tzeropoints\tinClusterZeros" +
	   "\toutClusterZeros\tallDispersions\tclusterDispersions\t" + 
	   "unclusteredDispersions\tradialVelocities\ttangentialVelocities\tLGdistances")
	for i in range(len(components)):
		print(str(i+1), end='\t')
		for component in components[i]:
			print("{:.3f}".format(component), end='\t')
		print()
	print()


	# Scree plot 
	plt.plot(np.array(range(len(pca.explained_variance_ratio_)))+1,
		  pca.explained_variance_ratio_*100, '-o', linewidth=2.0, color='k')
	lims = plt.xlim()
	plt.xticks(np.arange(ceil(lims[0]), ceil(lims[1]), 1))
	plt.xlabel("Number of component")
	plt.ylabel("Percentage of variance explained by component")
	plt.savefig(outputdir + "scree.pdf")
	plt.cla()
	plt.clf()

	# Cumulative variance
	cumVariances = np.cumsum(pca.explained_variance_ratio_)*100
	print("Explained variance (cumulative):")
	for variance, index in zip(cumVariances, range(len(cumVariances))):
		print(str(index + 1) + ":\t" + str(variance))
	print()

	plt.plot(range(1, variables+1), cumVariances, '-o', color='k')
	plt.ylim(0, 100)
	plt.xlim(0.5, variables+0.5)
	lims = plt.xlim()
	plt.xticks(np.arange(ceil(lims[0]), ceil(lims[1]), 1))
	plt.xlabel("Number of component")
	plt.ylabel("Cumulative variance explained by first components (\% of total)")
	plt.savefig(outputdir + "cumulative_variances.pdf")
	plt.cla()
	plt.clf()


	# 10-fold CV to all data
	kfold_seed = 1
	n_folds = 10

	kfold = model_selection.KFold(n_splits = n_folds, shuffle=True,
									random_state=kfold_seed)
	regr = LinearRegression()
	mse = []

	print("Root mean squared error when fitting to all data")
	for i in range(1, variables + 1):
		score = -1 * model_selection.cross_val_score(regr, data_pca[:, :i], y,
											   cv=kfold,
											   scoring='neg_mean_squared_error').mean()
		print(score)
		mse.append(sqrt(score))
	print()

	plt.plot(range(1, len(mse)+1), mse, '-o', color='k')
	
	plt.xlabel("Principal components in regresion")
	plt.ylabel(r"Root mean squared error ($\mathrm{M}_{\astrosun}$)")
	plt.title("Effect of the number of used PCs on the mass fitting residual.\n"
		   + "The errors are from 10-fold CV of all data.")
	plt.xlim((0.5, variables+0.5))
	plt.xticks(range(1, variables+1))
	plt.gcf().set_size_inches(4.8, 3.5)
	plt.tight_layout()
	plt.savefig(outputdir + "rmse-alldata.pdf")
	plt.cla()
	plt.clf()

	
	### training and testing set ###
	pca2 = PCA()
	X_train, X_test, y_train, y_test, timing_train, timing_test = cross_validation.train_test_split(data,
																	  y,
																	  timingArgumentMasses,
																	  test_size=0.4,
																	  random_state=kfold_seed)
	X_train_reduced = pca2.fit_transform(scale(X_train))
	n = len(X_train_reduced)
	kfold2 = cross_validation.KFold(n, n_folds=n_folds, shuffle=True,
								 random_state=kfold_seed)

	# RMSE in training using k-fold cross validation
	RMSEs_train = []
	regr = LinearRegression()
	
	for i in range(variables):
		score = model_selection.cross_val_score(regr, X_train_reduced[:,:i+1],
							  y_train,
							  cv=kfold2,
							  scoring='neg_mean_squared_error').mean()
		RMSEs_train.append(sqrt(-score))
	
	# split to train and test
	n = len(data_pca)
	kfold_seed = 1
	n_folds = 10

	kfold = model_selection.KFold(n_splits = n_folds, shuffle=True,
									random_state=kfold_seed)
	regr = LinearRegression()
	mse = []
	
	shape = (-1, 1)
	print("RMSE of timing argument on test set: " +
	   str(sqrt(mean_squared_error(y_test, timing_test))))

	# test set performance using one PC
	X_test_reduced = pca2.transform(scale(X_test))[:,1]
	regr = LinearRegression()
	regr.fit(np.reshape(X_train_reduced[:,1], shape), np.reshape(y_train, shape))
	pred = regr.predict(np.reshape(X_test_reduced, shape))
	mse = mean_squared_error(y_test, pred)
	print("RMSE in test set with 1 PC: " + str(sqrt(mse)))

	
#	# TA comparison
	timing_mse = mean_squared_error(y_train, timing_train)


	plt.plot(range(1, variables+1), RMSEs_train, '-o', color='k')
	xlim = plt.xlim()
	plt.plot(xlim, [sqrt(timing_mse), sqrt(timing_mse)], color='r')
	plt.xlim(xlim)

	#plt.ylim(0, 1.3)
	plt.xlabel("Number of PCs in regression")
	lims = plt.xlim()
	plt.xticks(np.arange(ceil(lims[0]), ceil(lims[1]), 1))
	plt.ylabel(r"RMSE ($M_{\astrosun}$)")
	plt.title("Training error using " + str(n_folds) + " folds")
	plt.gca().set_ylim(bottom=0)
#	plt.gca().set_ylim(top=1.3)
	plt.savefig(outputdir + "training-RMSE.pdf")
	plt.cla()
	plt.clf()
	
