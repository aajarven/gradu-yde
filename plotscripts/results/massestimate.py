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
from transitiondistance import findBestHubbleflow
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import scale
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error


if __name__ == "__main__":
	inputfile =	"/home/aajarven/Z-drive/duuni/extragal/gradu-yde/plotscripts/input/upTo5Mpc-fullpath.txt"
	outputdir = "/home/aajarven/Z-drive/duuni/extragal/gradu-yde/kuvat/"

	mindist = 1.0
	maxdist = 5.0
	eps = 1.8
	ms = 10

	masses = []
	zeropoints = []
	inClusterZeros = []
	outClusterZeros = []
	allDispersions = []
	clusterDispersions = []
	unclusteredDispersions = []
	radialVelocities = []
	tangentialVelocities = []
	LGdistances = []

	f = open(inputfile, 'r')
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



	##### finalizing data #####

	masses = np.array(masses)
	zeropoints = np.array(zeropoints)
	inClusterZeros = np.array(inClusterZeros)
	outClusterZeros = np.array(outClusterZeros)
	allDispersions = np.array(allDispersions)
	clusterDispersions = np.array(clusterDispersions)
	unclusteredDispersions = np.array(unclusteredDispersions)
	radialVelocities = np.array(radialVelocities)
	tangentialVelocities = np.array(tangentialVelocities)
	LGdistances = np.array(LGdistances)


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

	data = np.array([zeropoints, inClusterZeros, outClusterZeros,
				  allDispersions, radialVelocities, unclusteredDispersions,
				  radialVelocities, tangentialVelocities, LGdistances]).T

#	data = np.column_stack((zeropoints, inClusterZeros,
#							 outClusterZeros, allDispersions,
#							 clusterDispersions, unclusteredDispersions,
#							 radialVelocities, tangentialVelocities,
#							 LGdistances))


	##### Principal components #####
	# https://github.com/jcrouser/islr-python/blob/master/Lab%2011%20-%20PCR%20and%20PLS%20Regression%20in%20Python.ipynb
	pca = PCA()
	data_pca = pca.fit_transform(scale(data))
	components = pca.components_

	n_folds = 2

	print("component\tzeropoints\tinClusterZeros\toutClusterZeros\t" + 
	   "allDispersions\tclusterDispersions\tunclusteredDispersions\t" + 
	   "radialVelocities\ttangentialVelocities\tLGdistances")
	for i in range(len(components)):
		print(str(i+1), end='\t')
		for component in components[i]:
			print("{:.6f}".format(component), end='\t')
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
	cv = cross_validation.KFold(n, n_folds=n_folds, shuffle=True, random_state=1)
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
	plt.savefig(outputdir + "PCA-MSE-cv" + str(n_folds) + ".svg")

	
	# train and test sets, in[16]
	plt.cla()
	plt.clf()

	pca2 = PCA()
	data_train, data_test , y_train, y_test = cross_validation.train_test_split(data, y,
																		  test_size=0.25,
																		  random_state=1)
	data_pca_train = pca2.fit_transform(scale(data_train))
	
	n = len(data_pca_train)
	cv = cross_validation.KFold(n, n_folds=n_folds-1, shuffle=True, random_state=1)

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
			plt.ylabel('MSE')

	plt.savefig(outputdir + "PCA-trainresults-cv" + str(n_folds - 1) + ".svg")

	# testing, in[17]
	data_pca_test = pca2.transform(scale(data_test))[:,:3]
	regr = LinearRegression()
	regr.fit(data_pca_train[:,:3], y_train)

	# Prediction with test data
	pred = regr.predict(data_pca_test)
	mse = mean_squared_error(y_test, pred)
	print("Test data MSE with 2 PCs:" + "\t" + str(mse))


	plt.cla()
	plt.clf()

	mse = []
	cv = cross_validation.KFold(n, n_folds=2, shuffle=True, random_state=2)

	for i in np.arange(1, 10):
		pls = PLSRegression(n_components=i, scale=False)
		pls.fit(scale(data_pca),y)
		score = cross_validation.cross_val_score(pls, data_pca, y, cv=cv, scoring='mean_squared_error').mean()
		mse.append(-score)

	plt.plot(np.arange(1, 10), np.array(mse), '-o', color='k', linewidth=2.0)
	plt.xlabel('Number of principal components in PLS regression')
	plt.ylabel('MSE')
	plt.savefig(outputdir + "PCR-cv2.svg")
