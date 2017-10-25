# -*- coding:utf-8 -*-

from __future__ import print_function

import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import clustering
import clusterAnalysis
import filereader
import LGfinder
from optparse import OptionParser
#import pandas as pd
import physUtils
from sibeliusConstants import *
import timingargument
from transitiondistance import findBestHubbleflow
import matplotlib.pyplot as plt
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
	outputdir = "/home/aajarven/Z-drive/duuni/extragal/gradu-yde/kuvat/PCA/"

	if not os.path.isfile(datafile):
		mindist = 1.0
		maxdist = 5.0
		eps = 1.8
		ms = 10

		masses = []
		H0s = []
		zeropoints = []
		inClusterZeros = []
		outClusterZeros = []
		allDispersions = []
		clusterDispersions = []
		unclusteredDispersions = []
		radialVelocities = []
		tangentialVelocities = []
		LGdistances = []
		timingArgumentMasses = []

		f = open(simulationfiles, 'r')
		for simdir in f.readlines():

			dirname = simdir.strip()
			staticVel = filereader.readAllFiles(dirname, "Subhalo/Velocity", 3)
			mass = filereader.readAllFiles(dirname, "Subhalo/Mass", 1)
			fullCOP = filereader.readAllFiles(dirname, "Subhalo/CentreOfPotential", 3)
			FoFcontamination = filereader.readAllFiles(dirname, "FOF/ContaminationCount",
													   1)
			groupNumbers = filereader.readAllFiles(dirname, "Subhalo/GroupNumber", 1)

			fullCOP = fullCOP/h0 # to Mpc

			# creation of mask with True if there is no contamination in Subhalo
			contaminationMask = np.asarray([FoFcontamination[int(group)-1]<1 for group in
											  groupNumbers])

			# save contaminated haloes for finding closest one
			contaminatedPositions = fullCOP[contaminationMask==False, :]

			# elimination of contaminated haloes
			staticVel = staticVel[contaminationMask,:]
			mass = mass[contaminationMask]
			cop = fullCOP[contaminationMask, :]
			groupNumbers = groupNumbers[contaminationMask]
			
			# to physical units
			mass = mass/h0*1e10 # to M_☉
			
			LGs = LGfinder.findLocalGroup(staticVel, mass, cop, quiet=True,
										  outputAll=True)
			unmaskedLGs = LGfinder.maskedToUnmasked(LGs, cop, fullCOP)
			bestLGindex = LGfinder.chooseClosestToCentre(unmaskedLGs, contaminationMask, fullCOP)
			LG = LGs[bestLGindex]

			if mass[LG[0]] > mass[LG[1]]:
				centreIndex = LG[1]
			else:
				centreIndex = LG[0]

			centre = cop[centreIndex]
			centreVel = staticVel[centreIndex]
			closestContDist = physUtils.findClosestDistance(centre,
															contaminatedPositions)
			
			distances = np.array([physUtils.distance(centre, c) for c in
													 cop])
			vel = physUtils.addExpansion(staticVel, cop, centre)
		
			LGrelVel = vel[LG[0]] - vel[LG[1]]
			LGrelVelComponents = physUtils.velocityComponents(LGrelVel, cop[LG[1]]-cop[LG[0]])
			LGdistance = physUtils.distance(cop[LG[0]], cop[LG[1]])

			closestContMask = distances < closestContDist
			distRangeMask = np.array([d < maxdist and d > mindist for d in
								distances])


			if min(distances[closestContMask==False]) < maxdist:
				print("Warning: closest contaminated halo closer than the edge of\
	 Hubble flow fitting range in simulation " + dirname + 
			 ".\nExcluding simulation from analysis.")
				continue
			
			# contamination and distance range cut
			mask = np.logical_and(closestContMask, distRangeMask)
			cop = cop[mask]
			vel = vel[mask]
			distances = distances[mask]

			# radial velocities
			radvel = np.array([physUtils.velocityComponents(vel[j] - centreVel,
															cop[j] - centre)[0]
							   for j in range(len(vel))])


			##### extracting interesting data starts #####

			(fit, flowstartdist) = findBestHubbleflow(distances, radvel)
			clusteringDB = clustering.runClustering(cop, centre, ms, eps)
			clusterMemberMask = clusteringDB.labels_ != -1 # True for in cluster
			(inClusterFit, inClusterFlowstart) = findBestHubbleflow(
				distances[clusterMemberMask], radvel[clusterMemberMask])
			(outClusterFit, outClusterFlowstart) = findBestHubbleflow(
				distances[clusterMemberMask == False], radvel[clusterMemberMask == False])

			# LG mass from MW and andromeda
			M_big2 = mass[LG[0]] + mass[LG[1]]
		
			masses.append(M_big2)
			H0s.append(fit[0])
			zeropoints.append(-fit[1]/fit[0])
			inClusterZeros.append(-inClusterFit[1]/inClusterFit[0])
			outClusterZeros.append(-outClusterFit[1]/outClusterFit[0])
			allDispersions.append(np.std(radvel))
			clusterDispersions.append(clusterAnalysis.dispersionOfClusters(clusteringDB,
																	 radvel))
			unclusteredDispersions.append(clusterAnalysis.dispersionOfUnclustered(
				clusteringDB, radvel))
			radialVelocities.append(LGrelVelComponents[0])
			tangentialVelocities.append(LGrelVelComponents[1])
			LGdistances.append(LGdistance)
			timingArgumentMass = timingargument.timingArgumentMass(-1 *
														  LGrelVelComponents[0],
														  LGdistance*1000.0,
														  13.815, G)
			timingArgumentMasses.append(timingArgumentMass)

		##### finalizing data #####

		masses = np.array(masses)
		H0s = np.array(H0s)
		zeropoints = np.array(zeropoints)
		inClusterZeros = np.array(inClusterZeros)
		outClusterZeros = np.array(outClusterZeros)
		allDispersions = np.array(allDispersions)
		clusterDispersions = np.array(clusterDispersions)
		unclusteredDispersions = np.array(unclusteredDispersions)
		radialVelocities = np.array(radialVelocities)
		tangentialVelocities = np.array(tangentialVelocities)
		LGdistances = np.array(LGdistances)
		timingArgumentMasses = np.array(timingArgumentMasses)
		print(len(masses))
		print(len(timingArgumentMasses))

		data = np.array([masses, timingArgumentMasses, H0s, zeropoints, inClusterZeros,
				   outClusterZeros, allDispersions,  unclusteredDispersions,
				   clusterDispersions, radialVelocities, tangentialVelocities,
				   LGdistances]).T

		np.savetxt(datafile, data)
	else:
		data = np.loadtxt(datafile)
		result = np.hsplit(data, 12)
		(masses, timingArgumentMasses, H0s, zeropoints, inClusterZeros, outClusterZeros,
   allDispersions, unclusteredDispersions, clusterDispersions,
   radialVelocities, tangentialVelocities, LGdistances) = result
		

	#### debug ####
	print(time_s)
	print(time_Gyr)

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


	##### Principal components #####
	# https://github.com/jcrouser/islr-python/blob/master/Lab%2011%20-%20PCR%20and%20PLS%20Regression%20in%20Python.ipynb
	pca = PCA()
	data_pca = pca.fit_transform(scale(data))
	components = pca.components_

	n_folds = 10
	random_state = 1

	print("component\tH0s\tzeropoints\tinClusterZeros\toutClusterZeros\t" + 
	   "allDispersions\tclusterDispersions\tunclusteredDispersions\t" + 
	   "radialVelocities\ttangentialVelocities\tLGdistances")
	for i in range(len(components)):
		print(str(i+1), end='\t')
		for component in components[i]:
			print("{:.3f}".format(component), end='\t')
		print()
	print()

	# Explained variance
	plt.plot(np.array(range(len(pca.explained_variance_ratio_)))+1,
		  pca.explained_variance_ratio_*100, linewidth=2.0, color='k')
	plt.xlabel("Number of component")
	plt.ylabel("Percentage of variance explained by component")
	plt.savefig(outputdir + "PCA-variances.svg")

	print("Number of PCs and cumulative variances:")
	for i in range(len(pca.explained_variance_ratio_)-1):
		print(str(i+1) + "\t" + str(np.sum(pca.explained_variance_ratio_[:i+1])))
	print()

	plt.cla()
	plt.clf()
	n = len(data_pca)
	cv = cross_validation.KFold(n, n_folds=n_folds, shuffle=True, random_state=random_state)
	regr = LinearRegression()
	mse = []

	# MSE for different number of PCs, in[14]
	for i in np.arange(1, 10):
		score = -1*cross_validation.cross_val_score(regr, data_pca[:,:i],
											  y.ravel(), cv=cv,
											  scoring='mean_squared_error').mean()
		mse.append(score)
	
	plt.plot(np.array(range(len(mse)))+1, mse, '-o', color='k')
	plt.xlabel('Number of principal components in regression')
	plt.ylabel('MSE')
	plt.title('Mean squared error of the points from the fit')
	plt.savefig(outputdir + "PCA-MSE-cv" + str(n_folds) + ".svg")

	
	# train and test sets, in[16]
	plt.cla()
	plt.clf()

	pca2 = PCA()
	indices = np.arange(0, len(y))
	train_indices, test_indices = cross_validation.train_test_split(indices,
																 test_size=0.5,
																 random_state=random_state)
#	data_train, data_test , y_train, y_test = cross_validation.train_test_split(data, y,
#																		  test_size=0.5,
#																		  random_state=random_state)
	data_train = data[train_indices]
	data_test = data[test_indices]
	y_train = y[train_indices]
	y_test = y[test_indices]
	timing_test = timingArgumentMasses[test_indices]

	data_pca_train = pca2.fit_transform(scale(data_train))
	
	n = len(data_pca_train)
	cv = cross_validation.KFold(n, n_folds=n_folds, shuffle=True, random_state=random_state)

	# plot errors with different numbers of PCs
	mse = []
	for i in np.arange(1, 10):
			score = -1*cross_validation.cross_val_score(regr,
													 data_pca_train[:,:i],
													 y_train.ravel(), cv=cv,
													 scoring='mean_squared_error').mean()
			mse.append(score)

			plt.plot(np.array(range(len(mse)))+1, np.array(mse), '-o', color='k')
			plt.xlabel('Number of principal components in regression')
			plt.title('Mean squared error of the points in the training set' + 
			 ' from the fit')
			plt.ylabel('MSE')

	plt.savefig(outputdir + "PCA-trainresults-cv" + str(n_folds) + ".svg")

	mses = []
	# testing, in[17]
	# Prediction with test data
	for i in np.arange(1, 10):
		data_pca_test = pca2.transform(scale(data_test))[:,:i]
		regr = LinearRegression()
		regr.fit(data_pca_train[:,:i], y_train)
#		print(regr.coef_)
		pred = regr.predict(data_pca_test)
		mse = mean_squared_error(y_test, pred)
		mses.append(mse)
		print("Test data MSE with " + str(i) + " PCs:" + "\t" + str(mse))

	# testing timing argument performance
	timing_mse = mean_squared_error(y_test, timing_test)

	plt.cla()
	plt.clf()
	plt.plot(np.arange(1, 10), np.array(mses), '-o', color='k', linewidth=2.0)
	plt.plot(np.arange(1, 10), np.ones(9)*timing_mse, color='r', linewidth=2.0)
	plt.xlabel('Number of principal components')
	plt.ylabel('MSE')
	plt.title('MSE of test set values from fit to training set')
	plt.savefig(outputdir + "PCA-testresults.svg")

	plt.cla()
	plt.clf()

	mse = []
	cv = cross_validation.KFold(n, n_folds=n_folds, shuffle=True, random_state=2)

	for i in np.arange(1, 10):
		pls = PLSRegression(n_components=i, scale=False)
		pls.fit(scale(data_pca),y)
		score = cross_validation.cross_val_score(pls, data_pca, y, cv=cv, scoring='mean_squared_error').mean()
		mse.append(-score)

	plt.plot(np.arange(1, 10), np.array(mse), '-o', color='k', linewidth=2.0)
	plt.xlabel('Number of principal components in PLS regression')
	plt.ylabel('MSE')
	plt.title('MSE of the points from the fit using PLSRegression')
	plt.savefig(outputdir + "PCR-cv" + str(n_folds) + ".svg")
