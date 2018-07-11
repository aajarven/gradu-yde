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

"""
	Returns the extent of the range containing given percent of values in data
	set. The parameter percentIncluded should be in range [0, 1].
"""
def rangeAroundMedian(data, percentIncluded):
	if percentIncluded < 0 or percentIncluded > 1.0:
		raise ValueError("Percentage outside the allowed range")
	lowerLimit = (1.0 - percentIncluded)/2
	upperLimit = 1.0 - lowerLimit

	lowerIndex = max(0, int(math.ceil(len(data)*lowerLimit)-1))
	upperIndex = min(len(data), int(math.floor(len(data)*upperLimit)-1))
	
	sortedData=sorted(data)
	rangeMin = sortedData[lowerIndex]
	rangeMax = sortedData[upperIndex]
	return (rangeMin, rangeMax)


if __name__ == "__main__":
	inputfile = "../input/lgfound-fullpath.txt" 
	savelocH0 = "../../kuvat/shelledH0.pdf"
	savelocZero = "../../kuvat/zeros.pdf"
	savelocCumulative = "../../kuvat/overdensity+H0.pdf"
	lines =  sum(1 for line in open(inputfile))

	limitsH0 = [[d, d+2.0] for d in np.arange(0.0, 8.0, 0.1)]
	H0 = np.full((lines, len(limitsH0)), np.nan)
	H0min50 = np.full(len(limitsH0), np.nan)
	H0max50 = np.full(len(limitsH0), np.nan)
	H0min75 = np.full(len(limitsH0), np.nan)
	H0max75 = np.full(len(limitsH0), np.nan)
	H0min90 = np.full(len(limitsH0), np.nan)
	H0max90 = np.full(len(limitsH0), np.nan)

	limitsZeros = [[d, d+2.0] for d in np.arange(0.0, 8.0, 0.1)]
	zeros = np.full((lines, len(limitsZeros)), np.nan)
	zerosMin50 = np.full(len(limitsZeros), np.nan)
	zerosMax50 = np.full(len(limitsZeros), np.nan)
	zerosMin75 = np.full(len(limitsZeros), np.nan)
	zerosMax75 = np.full(len(limitsZeros), np.nan)
	zerosMin90 = np.full(len(limitsZeros), np.nan)
	zerosMax90 = np.full(len(limitsZeros), np.nan)

	limitsCumulative = np.arange(0.5, 10.0, 0.1)
	cumulativeDensity = np.full((lines, len(limitsCumulative)), np.nan)
	cumulativeH0s = np.full((lines, len(limitsCumulative)), np.nan)

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
		mass = mass[mask]

		# radial velocities
		radvel = np.array([physUtils.velocityComponents(vel[j] - centreVel,
														cop[j] - centre)[0]
						   for j in range(len(vel))])


		##### extracting interesting numbers #####
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

		for limit, index in zip(limitsCumulative,
						  range(len(limitsCumulative))):
			if max(distances) < limit:
				continue
			mask = np.array([d < limit for d in distances])
			k, b, r_value, p_value, std_err = linregress(distances[mask],
													  radvel[mask])
			cumulativeH0s[sim, index] = k
			cumulativeDensity[sim, index] = np.sum(mass[mask]) / (4.0/3.0 *
														math.pi * (limit**3))


	
	# ranges for H0
	for i in range(len(limitsH0)):
		data = H0[:, i]
		data = data[~np.isnan(data)]
		
		(H0min50[i], H0max50[i]) = rangeAroundMedian(data, 0.5)
		(H0min75[i], H0max75[i]) = rangeAroundMedian(data, 0.75)
		(H0min90[i], H0max90[i]) = rangeAroundMedian(data, 0.9)

	# ranges for zero point
	for i in range(len(limitsZeros)):
		data = zeros[:, i]
		data = data[~np.isnan(data)]
		
		(zerosMin50[i], zerosMax50[i]) = rangeAroundMedian(data, 0.5)
		(zerosMin75[i], zerosMax75[i]) = rangeAroundMedian(data, 0.75)
		(zerosMin90[i], zerosMax90[i]) = rangeAroundMedian(data, 0.9)

	H0centers = [(l[0]+l[1])/2 for l in limitsH0]
	zeroCenters = [(l[0]+l[1])/2 for l in limitsZeros]
	
	#### plotting #####
	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)

	## H0 ##
	fig = plt.figure()
	ax = plt.axes()

	ax.fill_between(H0centers, H0min90, H0max90,
				 color='0.9', label="90 \% range")
	ax.fill_between(H0centers, H0min75, H0max75,
				 color='0.8', label="75 \% range")
	ax.fill_between(H0centers, H0min50, H0max50,
				 color='0.7', label="50 \% range")
	
	ax.plot(H0centers, np.nanmedian(H0, axis=0), linewidth=2.0, color='k',
		 label="median")
	plt.xlim([min(H0centers), max(H0centers)])

	ax.legend()

	plt.xlabel("Distance of bin centre from the Milky Way (Mpc)")
	plt.ylabel("$H_0$ in 2.0 Mpc bin (km/s/Mpc)")	
	
	fig.set_size_inches(4.55, 3.7)
	plt.tight_layout()
	plt.savefig(savelocH0)

	plt.cla()
	plt.clf()

	## Zero point ##
	fig = plt.figure()
	ax = plt.axes()

	ax.fill_between(zeroCenters, zerosMin90, zerosMax90,
				 color='0.9', label="90 \% range")
	ax.fill_between(zeroCenters, zerosMin75, zerosMax75,
				 color='0.8', label="75 \% range")
	ax.fill_between(zeroCenters, zerosMin50, zerosMax50,
				 color='0.7', label="50 \% range")

	ax.plot(zeroCenters, np.nanmedian(zeros, axis=0), linewidth=2.0, color='k',
		label="median")
	
	#print(np.sum(~np.isnan(zeros), axis=0))
	
	ax.legend(loc=3)

	plt.xlabel("Distance of bin centre from the Milky Way (Mpc)")
	plt.xlim([min(zeroCenters), max(zeroCenters)])	

	plt.tight_layout()

	plt.xlabel("Distance of bin centre from the Milky Way (Mpc)")
	plt.ylabel("Hubble flow zero point distance\nfrom the Milky Way (Mpc)")	
	
	fig.set_size_inches(4.55, 3.7)
	plt.tight_layout()
	plt.savefig(savelocZero)

	plt.cla()
	plt.clf()


	## overdensity vs HF inside ##
	fig, ax1 = plt.subplots()
	
	ax1.plot(limitsCumulative, np.nanmedian(cumulativeH0s, axis=0), 'b')
	ax1.tick_params('y', colors='b')
	ax1.set_ylabel("Median $H_0$ within radius (km/s/Mpc)")
#	ax1.set_xlabel("Distance from the Milky Way (Mpc)")
	
	ax2 = ax1.twinx()
	ax2.plot(limitsCumulative, np.nanmedian(cumulativeDensity,
	   								 axis=0)/critical_density, 'r')
	ax2.tick_params('y', colors='r')
	ax2.set_ylabel("Median overdensity within radius")
	ax2.set_xlabel("Distance from the Milky Way (Mpc)")
	ax2.set_xlim([min(limitsCumulative), max(limitsCumulative)])
	ax2.set_xticks(np.arange(math.ceil(min(limitsCumulative)), max(limitsCumulative)+0.01,
						 1), minor=False)


	fig.set_size_inches(4.5, 2.9)
	plt.tight_layout()
	plt.savefig(savelocCumulative)

