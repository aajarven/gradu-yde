# -*- coding:utf-8 -*-

from __future__ import print_function

import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import filereader
import LGfinder
import physUtils
from scipy.stats import linregress
from sibeliusConstants import *
import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib import rc


if __name__ == "__main__":
	inputfile = "../input/lgfound-fullpath.txt" 
	saveloc = "../../kuvat/shelledH0.svg"

	lines =  sum(1 for line in open(inputfile))
	limits = [[d, d+2.0] for d in np.arange(0.0, 8.0, 0.1)]
	H0 = np.full((lines, len(limits)), np.nan)
	H0min = np.full((lines, len(limits)), np.nan)
	H0max = np.full((lines, len(limits)), np.nan)
	zeros = np.full((lines, len(limits)), np.nan)

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


		##### extracting interesting data starts #####
		for limit, index in zip(limits, range(len(limits))):
			# jump to next if the haloes don't go all the way to the outer limit
			if max(distances) < limit[1]:
				continue

			mask = np.array([d > limit[0] and d < limit[1] for d in distances])
#			fit = np.polyfit(distances[mask], radvel[mask], 1)
			k, b, r_value, p_value, std_err = linregress(distances[mask],
													  radvel[mask])
			H0[sim, index] = k 
			zeros[sim, index] = -b/k

			# http://mathworld.wolfram.com/LeastSquaresFitting.html
			meanx = np.mean(distances[mask])
			ssxx = 0
			for x in distances[mask]:
				ssxx += (x - meanx)**2
			
			meany = np.mean(radvel[mask])
			ssyy = 0
			for y in radvel[mask]:
				ssyy += (y - meany)**2
			
			ssxy = 0
			for x, y in zip(distances[mask], radvel[mask]):
				ssxy += (x - meanx)*(y - meany)
			s = math.sqrt((ssyy - k*ssxy)/(np.sum(mask)-2))
			
			# standard errors
			SEk = s/math.sqrt(ssxx)
			SEa = s*math.sqrt(1/np.sum(mask)+meanx**2/ssxx)

			H0max[sim, index] = k + SEk
			H0min[sim, index] = k - SEk



	##### plotting #####
#	print(limits)
#	print(np.nanmean(H0, axis=0))

	print(np.nanmean(H0max, axis=0) - np.nanmean(H0min, axis=0))

	centers = [(l[0]+l[1])/2 for l in limits]
	plt.plot(centers, np.nanmean(H0, axis=0), linewidth=2.0, color='k')
	plt.plot(centers, np.nanmean(H0max, axis=0), linewidth=2.0, color='0.75')
	plt.plot(centers, np.nanmean(H0min, axis=0), linewidth=2.0, color='0.75')
	plt.xlabel("Distance of bin centre from LG centre (Mpc)")
	plt.ylabel("Mean H0 when fitted using haloes in bin")	
	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)
	plt.tight_layout()
	
	plt.savefig(saveloc)
