# -*- coding:utf-8 -*-

from __future__ import print_function

import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import clustering
import clusterAnalysis
import filereader
import LGfinder
from clusteredHF import outClusterFit, inClusterFit
import physUtils
from sibeliusConstants import *
import timingargument
from transitiondistance import findBestHubbleflow, simpleFit
import matplotlib.pyplot as plt
import numpy as np


# also returns the data
def readAndSave(simulationfiles, datafile, mindist=1.5, maxdist=5.0, eps=1.6,
				ms=4, scale_eps=False):
	masses = []
	H0s = []
	inClusterH0s = []
	outClusterH0s = []
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

		massCentre = physUtils.massCentre(cop[LG[0]], cop[LG[1]], mass[LG[0]],
									mass[LG[1]])
		closestContDist = physUtils.findClosestDistance(massCentre,
												  contaminatedPositions)
		
		if closestContDist < maxdist:
			print("Warning: closest contaminated halo closer than the edge of\
	  Hubble flow fitting range in simulation " + dirname + 
	  ".\nExcluding simulation from analysis.")
			continue


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


		# contamination and distance range cut
		mask = np.logical_and(closestContMask, distRangeMask)
		cop = cop[mask]
		vel = vel[mask]
		distances = distances[mask]

		# radial velocities
		radvel = np.array([physUtils.velocityComponents(vel[j] - centreVel,
											   cop[j] - centre)[0]
				  for j in range(len(vel))])


		##### extracting clustering data #####

		clusteringDB = clustering.runClustering(cop, centre, ms, eps,
										  meansep=scale_eps)
		labels = clusteringDB.labels_
		uniqueLabels = set(labels)
		clusterMemberMask = labels != -1 # True for haloes in cluster

		## all haloes ##
		(H0, zero) = simpleFit(distances, radvel)
		radvelResiduals = np.empty(radvel.shape)
		for i in range(len(radvel)):
			radvelResiduals[i] = radvel[i] - (distances[i] - zero) * H0
		zeropoints.append(zero)
		H0s.append(H0)
		allDispersions.append(np.std(radvelResiduals, ddof=1))

		## outside clusters ##
		(H0, zero, dispersion) = outClusterFit(clusteringDB, radvel, distances,
										 minHaloes=10)
		outClusterH0s.append(H0)
		outClusterZeros.append(zero)
		unclusteredDispersions.append(dispersion)

		## inside clusters ##
		(H0, zero, dispersion) = inClusterFit(clusteringDB, radvel, distances,
										minSize=10)
		inClusterH0s.append(H0)
		inClusterZeros.append(zero)
		clusterDispersions.append(dispersion)

		# LG mass from MW and andromeda
		M_big2 = mass[LG[0]] + mass[LG[1]]

		masses.append(M_big2)
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
	inClusterH0s = np.array(inClusterH0s)
	outClusterH0s = np.array(outClusterH0s)
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

	data = np.array([masses, timingArgumentMasses, H0s, inClusterH0s,
				  outClusterH0s, zeropoints, inClusterZeros, outClusterZeros,
				  allDispersions,  unclusteredDispersions, clusterDispersions,
				  radialVelocities, tangentialVelocities, LGdistances]).T


	np.savetxt(datafile, data)

	return data
