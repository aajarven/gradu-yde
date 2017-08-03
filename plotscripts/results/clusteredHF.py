# -*- coding:utf-8 -*-

from __future__ import print_function
import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import clustering
import clusterAnalysis
import filereader
import LGfinder
from optparse import OptionParser
import physUtils
from sibeliusConstants import *
from transitiondistance import simpleFit
import matplotlib.pyplot as plt
import numpy as np
import sys


if __name__ == "__main__":
#	inputfile = "../input/lgfound-fullpath.txt" 
	inputfile = "../input/hundred.txt"
	outputdir = "../../kuvat/"

	mindist = 1.0
	maxdist = 5.0
	eps = 1.8
	ms = 10

	allZeros = []
	inClusterZeros = []
	outClusterZeros = []
	allH0s = []
	inClusterH0s = []
	outClusterH0s = []
	inClusterMaxMasses = []

	i = 0
	f = open(inputfile, 'r')
	for simdir in f.readlines():
		print(i)
		i += 1
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
		if closestContDist < maxdist:
			continue
		
		distances = np.array([physUtils.distance(centre, c) for c in
												 cop])
		vel = physUtils.addExpansion(staticVel, cop, centre)
	
		LGrelVel = vel[LG[0]] - vel[LG[1]]
		LGrelVelComponents = physUtils.velocityComponents(LGrelVel, cop[LG[1]]-cop[LG[0]])
		LGdistance = physUtils.distance(cop[LG[0]], cop[LG[1]])

		mask = np.array([d < maxdist and d > mindist for d in
							distances])
		cop = cop[mask]
		vel = vel[mask]
		distances = distances[mask]
		mass = mass[mask]

		# radial velocities
		radvel = np.array([physUtils.velocityComponents(vel[j] - centreVel,
														cop[j] - centre)[0]
						   for j in range(len(vel))])


		##### extracting interesting data starts #####

		(H0, zero) = simpleFit(distances, radvel)
		clusteringDB = clustering.runClustering(cop, centre, ms, eps)
		clusterMemberMask = clusteringDB.labels_ != -1 # True for in cluster
		(inClusterH0, inClusterZero) = simpleFit(distances[clusterMemberMask],
										   radvel[clusterMemberMask])
		(outClusterH0, outClusterZero) = simpleFit(
			distances[clusterMemberMask == False],
			radvel[clusterMemberMask == False])
		inClusterMaxMass = max(mass[clusterMemberMask])

		allZeros.append(zero)
		allH0s.append(H0)
		inClusterZeros.append(inClusterZero)
		inClusterH0s.append(inClusterH0)
		outClusterZeros.append(outClusterZero)
		outClusterH0s.append(outClusterH0)
		inClusterMaxMasses.append(inClusterMaxMass)


	##### plotting #####

	allZeros = np.array(allZeros)
	allH0s = np.array(allH0s)
	inClusterZeros = np.array(inClusterZeros)
	inClusterH0 = np.array(inClusterH0)
	outClusterZeros = np.array(outClusterZeros)
	outClusterH0 = np.array(outClusterH0)
	inClusterMaxMasses = np.array(inClusterMaxMasses)

	# masking zeropoints
#	sanitymask = allZeros < 5.0
#	inClusterSanitymask = inClusterZeros < 5.0
#	outClusterSanitymask = outClusterZeros > -5.0

	# x limits for zeropoints 
#	xmin = min(min(np.amin(allZeros[sanitymask]),
#				np.amin(inClusterZeros[inClusterSanitymask])),
#			np.amin(outClusterZeros[outClusterSanitymask]))
#	xmax = max(max(np.amax(allZeros[sanitymask]),
#				np.amax(inClusterZeros[inClusterSanitymask])),
#			np.amax(outClusterZeros[outClusterSanitymask]))
#	xmin = np.floor(xmin*2.0)/2.0
#	xmax = np.ceil(xmax*2.0)/2.0

	plt.scatter(inClusterMaxMasses, inClusterH0s)
#	plt.xlabel("Hubble flow zero point (Mpc from LG centre)")
#	plt.ylabel("Combined mass of Milky Way and Andromeda (Solar masses)")
#	plt.xlim(xmin, xmax)
#	plt.savefig(outputdir+"zeropoints.png")
	plt.show()

