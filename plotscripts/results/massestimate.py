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
	inputfile =	"/home/aajarven/Z-drive/duuni/extragal/gradu-yde/plotscripts/input/allButDuplicates-fullpath.txt" 
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

	x = np.column_stack((zeropoints, inClusterZeros,
							 outClusterZeros, allDispersions,
							 clusterDispersions, unclusteredDispersions,
							 radialVelocities, tangentialVelocities,
							 LGdistances))


	##### PCR #####
	pca = PCA()
	X_reduced = pca.fit_transform(scale(x))

	plt.plot(np.array(range(len(pca.explained_variance_ratio_)))+1,
		  pca.explained_variance_ratio_, linewidth=2.0)
	plt.xlabel("Number of component")
	plt.ylabel("Percentage of variance explained by component")
	plt.savefig(outputdir + "PCA-variances.svg")

	plt.cla()
	plt.clf()

#	n = len(X_reduced)
#	print(n)
#	kf_3 = cross_validation.KFold(n, n_folds=3, shuffle=True, random_state=1)
#	regr = LinearRegression()
#	mse = []
#	score = -1*cross_validation.cross_val_score(regr, np.ones((n,1)),
#											 y.ravel(), cv=kf_3,
#											 scoring='mean_squared_error').mean()
#	mse.append(score)
#	for i in np.arange(1, 9):
#		score = -1*cross_validation.cross_val_score(regr, X_reduced[:,:i], y,
#											  cv=kf_3,
#											  scoring='mean_squared_error').mean()
#		mse.append(score)
#	plt.plot(mse, '-v')
#	plt.xlabel('Number of principal components in regression')
#	plt.ylabel('MSE')
#	plt.title('MW mass')
#	plt.xlim(xmin=-1);
#	plt.show()
#
#	plt.cla()
#	plt.clf()

	pca2 = PCA()
	regr = LinearRegression()

	# Split into training and test sets
	X_train, X_test , y_train, y_test = cross_validation.train_test_split(x, y,
																		  test_size=1.0/3,
																		  random_state=1)
	# Scale the data
	X_reduced_train = pca2.fit_transform(scale(X_train))
	n = len(X_reduced_train)
	# 3-fold CV, with shuffle
	kf_3 = cross_validation.KFold(n, n_folds=3, shuffle=True, random_state=1)
	mse = []
	# Calculate MSE with only the intercept (no principal components in regression)
	score = -1*cross_validation.cross_val_score(regr, np.ones((n,1)),
												y_train.ravel(), cv=kf_3,
												scoring='mean_squared_error').mean()
	mse.append(score)
	# Calculate MSE using CV for the 9 principle components, adding one component
	# at the time.
	for i in np.arange(1, 9):
		score = -1*cross_validation.cross_val_score(regr, X_reduced_train[:,:i],
												 y_train.ravel(), cv=kf_3,
												 scoring='mean_squared_error').mean()
		mse.append(score)
	print(mse)
	print(np.arange(1, len(mse)+1))
	plt.plot(np.arange(1, len(mse)+1), np.array(mse))
	plt.xlabel('Number of principal components in regression')
	plt.ylabel('MSE')
	plt.savefig(outputdir + "PCA-3foldMSE-LinearRegression.svg")

	plt.cla()
	plt.clf()


	n = len(X_train)
	kf_3 = cross_validation.KFold(n, n_folds=3, shuffle=True, random_state=1)
	mse = []
	for i in np.arange(1, 9):
		pls = PLSRegression(n_components=i)
		score = cross_validation.cross_val_score(pls, scale(X_train), y_train,
										   cv=kf_3,
										   scoring='mean_squared_error').mean()
		mse.append(-score)
	
	# Plot results
	plt.plot(np.arange(1, len(mse)+1), np.array(mse))
	plt.xlabel('Number of principal components in regression')
	plt.ylabel('MSE')
	plt.savefig(outputdir + "PCA-3foldMSE-PLSRegression.svg")
