# -*- coding:utf-8 -*-

from __future__ import print_function

import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import filereader
import LGfinder
import physUtils
import scipy
from scipy.stats import linregress
from sibeliusConstants import *
import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib import rc


if __name__ == "__main__":
	inputfile = "../input/lgfound-fullpath.txt" 
	savelocH0 = "../../kuvat/shelledH0.svg"
	savelocZero = "../../kuvat/zeros.svg"

	lines =  sum(1 for line in open(inputfile))
	limitsH0 = [[d, d+2.0] for d in np.arange(0.0, 8.0, 0.1)]
	limitsZeros = [[d, d+4] for d in np.arange(0.0, 6.0, 1.0)]
	H0 = np.full((lines, len(limitsH0)), np.nan)
	H0min = np.full((1, len(limitsH0)), np.nan)
	H0max = np.full((1, len(limitsH0)), np.nan)
	zeros = np.full((lines, len(limitsZeros)), np.nan)

	f = open(inputfile, 'r')
	sim = -1
	for simdir in f.readlines():
		sim += 1
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

		# contamination cut
		mask = distances < closestContDist
		cop = cop[mask]
		vel = vel[mask]
		distances = distances[mask]

		# radial velocities
		radvel = np.array([physUtils.velocityComponents(vel[j] - centreVel,
														cop[j] - centre)[0]
						   for j in range(len(vel))])


		##### extracting H0 #####
		for limit, index in zip(limitsH0, range(len(limitsH0))):
			# jump to next if the haloes don't go all the way to the outer limit
			if max(distances) < limit[1]:
				continue

			mask = np.array([d > limit[0] and d < limit[1] for d in distances])
#			fit = np.polyfit(distances[mask], radvel[mask], 1)
			k, b, r_value, p_value, std_err = linregress(distances[mask],
													  radvel[mask])
			H0[sim, index] = k 

		for limit, index in zip(limitsZeros, range(len(limitsZeros))):
			if max(distances) < limit[1]:
				continue

			mask = np.array([d > limit[0] and d < limit[1] for d in distances])
			k, b, r, p, std_err = linregress(distances[mask], radvel[mask])
			zeros[sim, index] = -b/k


	
	# standard errors
	for i in range(len(limitsH0)):
		data = H0[:, i]
		data = data[~np.isnan(data)]
		sem = scipy.stats.sem(data)
		std = np.std(data)
		
		H0min[0, i] = np.nanmean(data) - std
		H0max[0, i] = np.nanmean(data) + std

	centers = [(l[0]+l[1])/2 for l in limitsH0]
	
	#### plotting #####
	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)
	
	fig = plt.figure()

	plt.plot(centers, H0max.flatten(), linewidth=2.0, color='0.75')
	plt.plot(centers, H0min.flatten(), linewidth=2.0, color='0.75')
	plt.plot(centers, np.nanmean(H0, axis=0), linewidth=2.0, color='k')
	
	plt.xlabel("Distance of bin centre from Milky Way (Mpc)")
	plt.ylabel("Mean $H_0$ in 2.0 Mpc bin (km/s/Mpc)")	
	

	plt.tight_layout()

	fig.set_size_inches(5.9, 5)
	
	plt.savefig(savelocH0)


	plt.cla()
	plt.clf()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	
	mask = ~np.isnan(zeros)
	filteredZeros = [d[m] for d,m in zip(zeros.T, mask.T)]
	centers = [(l[0]+l[1])/2.0 for l in limitsZeros]
	plt.boxplot(filteredZeros)
	ax.set_xticklabels(centers)
	plt.ylim([-2, 4])

	#TODO ligatures not working (e.g. fl)
	plt.xlabel("Distance of bin centre from Milky Way (Mpc)")
	plt.ylabel("Hubble flow zero point distance from MW (Mpc)")	
	
	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)
	plt.tight_layout()

	fig.set_size_inches(5.9, 5)

	plt.savefig(savelocZero)
