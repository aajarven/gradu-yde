# -*- coding:utf-8 -*-

from __future__ import print_function
import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import clustering
import clusterAnalysis
import filereader
import LGfinder
import physUtils
from sibeliusConstants import *
from transitiondistance import simpleFit
import math
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import sys

def blackBoxplot(bp):
	for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
		plt.setp(bp[element], color='k')


if __name__ == "__main__":
	inputfile = "../input/upTo5Mpc-no229-fullpath.txt" 
#	inputfile = "../input/hundred.txt"
	outputdir = "../../kuvat/"

	mindist = 1.0
	maxdist = 5.0
	eps = 0.16
	ms = 4
	massThreshold = 8e11 #M_sun

	allZeros = []
	inClusterZeros = []
	outClusterZeros = []
	allH0s = []
	inClusterH0s = []
	outClusterH0s = []
	massCutH0s = []
	massCutZeros = []

	simIndex = 0
	f = open(inputfile, 'r')
	for simdir in f.readlines():
		print(simIndex)
		simIndex += 1
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
		#TLSfit(distances, radvel)
		(H0, zero) = simpleFit(distances, radvel)
		clusteringDB = clustering.runClustering(cop, centre, ms, eps,
										  meansep=False)
		labels = clusteringDB.labels_
		uniqueLabels = set(labels)

		clusterMemberMask = labels != -1 # True for in cluster
		print(sum([not membership for membership in clusterMemberMask]))
		(inClusterH0, inClusterZero) = simpleFit(distances[clusterMemberMask],
										   radvel[clusterMemberMask])
		(outClusterH0, outClusterZero) = simpleFit(
			distances[clusterMemberMask == False],
			radvel[clusterMemberMask == False])

		allZeros.append(zero)
		allH0s.append(H0)
		inClusterZeros.append(inClusterZero)
		inClusterH0s.append(inClusterH0)
		outClusterZeros.append(outClusterZero)
		outClusterH0s.append(outClusterH0)

		# mass exclusion
		allowedClusterNumbers = []
		for i in uniqueLabels:
			if max(mass[labels==i]) < massThreshold:
				allowedClusterNumbers.append(i)
		clusterNumberModifier = (-1 if -1 in uniqueLabels else 0)
		print(str(len(uniqueLabels) + clusterNumberModifier) + ", "
		+ str(len(allowedClusterNumbers) + clusterNumberModifier))

		maxMassMask = np.array([label in allowedClusterNumbers for label in
						  labels])
		(maskedH0, maskedZero) = simpleFit(distances[maxMassMask],
									 radvel[maxMassMask])
		massCutH0s.append(maskedH0)
		massCutZeros.append(maskedZero)

		

	##### plotting #####

	allZeros = np.array(allZeros)
	allH0s = np.array(allH0s)
	inClusterZeros = np.array(inClusterZeros)
	inClusterH0s = np.array(inClusterH0s)
	outClusterZeros = np.array(outClusterZeros)
	outClusterH0s = np.array(outClusterH0s)
	massCutH0s = np.array(massCutH0s)
	massCutZeros = np.array(massCutZeros)

	print(inClusterH0s[np.argsort(inClusterH0s)[:10]])
	print(inClusterH0s[np.argsort(inClusterH0s)[-10:]])
	print(inClusterZeros[np.argsort(inClusterZeros)[:10]])
	print(inClusterZeros[np.argsort(inClusterZeros)[-10:]])
	print("")
	minindex = np.argmin(inClusterH0s)
	print(minindex)
	print(inClusterH0s[minindex])
	print(inClusterZeros[minindex])
	print(allH0s[minindex])
	print(allZeros[minindex])

	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)
	params = {'text.latex.preamble' : [r'\usepackage{wasysym}']}
	plt.rcParams.update(params)

	fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
	bp1 = ax1.boxplot([massCutH0s, inClusterH0s, outClusterH0s, allH0s], vert=False)
	blackBoxplot(bp1)
	bp2 = ax2.boxplot([massCutZeros, inClusterZeros, outClusterZeros, allZeros], vert=False)
	blackBoxplot(bp2)

	ax1.set_yticklabels(["Haloes in clusters with\nall members less massive\n"
					  r"than $8^{11}~M_{\astrosun}$",
					  "Haloes in clusters", "Haloes outside clusters", "All haloes"
					 ], ha='right', multialignment='right')
	ylims = ax1.get_xlim()
	ax1.set_xticks(np.arange(math.ceil(ylims[0]/10)*10, ylims[1], 10), minor=True)
	ax1.set_xlabel(r"$H_0$ (km/s/Mpc)")
	ylims = ax2.get_xlim()
	ax2.set_xticks(np.arange(math.ceil(ylims[0]), ylims[1], 1.0), minor=True)
	ax2.set_xticks(np.arange(-5, ylims[1], 5.0), minor=False)
	ax2.set_xlabel("Distance to Hubble\nflow zero point (Mpc)")
#	ax1.set_xlim([-25, 145])
#	ax2.set_xlim([-4, 4])

	plt.tight_layout(rect=[0.065, 0.115, 1.0, 1.0])
#	plt.tight_layout()
	fig.set_size_inches(5.9, 2.6)

#	plt.xlabel("Hubble flow zero point (Mpc from LG centre)")
#	plt.ylabel("Combined mass of Milky Way and Andromeda (Solar masses)")
#	plt.xlim(xmin, xmax)
	plt.savefig(outputdir+"clusteredHFparameters.pdf")

