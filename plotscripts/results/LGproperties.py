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
from matplotlib import rcParams


if __name__ == "__main__":
	inputfile = "../input/lgfound-fullpath.txt" 
	saveloc = "../../kuvat/LGproperties.svg"

	lines =  sum(1 for line in open(inputfile))
	radvel = np.full(lines, np.nan)
	tanvel = np.full(lines, np.nan)
	distance = np.full(lines, np.nan)
	mass = np.full(lines, np.nan)
	massdifference = np.full(lines, np.nan)
	massratio = np.full(lines, np.nan)

	f = open(inputfile, 'r')
	sim = -1
	for simdir in f.readlines():
		sim += 1
		dirname = simdir.strip()
		staticVel = filereader.readAllFiles(dirname, "Subhalo/Velocity", 3)
		masses = filereader.readAllFiles(dirname, "Subhalo/Mass", 1)
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
		masses = masses[contaminationMask]
		cop = fullCOP[contaminationMask, :]
		groupNumbers = groupNumbers[contaminationMask]
		
		# to physical units
		masses = masses/h0*1e10 # to M_☉
		
		LGs = LGfinder.findLocalGroup(staticVel, masses, cop, quiet=True,
									  outputAll=True)
		unmaskedLGs = LGfinder.maskedToUnmasked(LGs, cop, fullCOP)
		bestLGindex = LGfinder.chooseClosestToCentre(unmaskedLGs, contaminationMask, fullCOP)
		LG = LGs[bestLGindex]

		if masses[LG[0]] > masses[LG[1]]:
			centreIndex = LG[1]
		else:
			centreIndex = LG[0]

		centre = cop[centreIndex]
		centreVel = staticVel[centreIndex]
		
		vel = physUtils.addExpansion(staticVel, cop, centre)
	
		LGrelVel = vel[LG[0]] - vel[LG[1]]
		(rad, tan) = physUtils.velocityComponents(LGrelVel, cop[LG[1]]-cop[LG[0]])

		radvel[sim] = rad
		tanvel[sim] = tan
		distance[sim] = physUtils.distance(cop[LG[0]], cop[LG[1]])
		mass[sim] = masses[LG[0]] + masses[LG[1]]
		massdifference[sim] = math.fabs(masses[LG[0]] - masses[LG[1]])
		massratio[sim] = masses[centreIndex] / (masses[LG[0]] + masses[LG[1]])
		
	
	#### plotting #####
	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)
	rcParams['text.latex.preamble'] = [r'\usepackage{wasysym}']

	fig, axarr = plt.subplots(3, 2)

	ax = axarr[0, 0]
	ax.hist(radvel, color='0.75')
	ax.set_xlabel("Radial velocity (km/s)")
	ax.set_ylabel("Number of simulations")
	
	ax = axarr[0, 1]
	ax.hist(tanvel, color='0.75')
	ax.set_xlabel("Tangential velocity (km/s)")
	ax.set_ylabel("Number of simulations")

	ax = axarr[1, 0]
	ax.hist(distance, color='0.75')
	ax.set_xlabel("Distance (Mpc)")
	ax.set_ylabel("Number of simulations")

	ax = axarr[1, 1]
	ax.hist(mass/1e12, color='0.75')
	ax.set_xlabel(r'Combined mass ($10^{12}\ M_{\astrosun}$)',
			   multialignment='center')
	ax.set_ylabel("Number of simulations")

	ax = axarr[2, 0]
	ax.hist(massdifference/1e12, color='0.75')
	ax.set_xlabel(r"Mass difference ($10^{12}\ M_{\astrosun}$)",
			   multialignment='center')
	ax.set_ylabel("Number of simulations")

	ax = axarr[2, 1]
	ax.hist(massratio, color='0.75')
	ax.set_xlabel(r"Percent of mass in the\\more massive primary",
			   multialignment='center', linespacing=10.0)
	ax.set_ylabel("Number of simulations")
	
	plt.tight_layout()

	fig.set_size_inches(5.9, 6.5)

	plt.savefig(saveloc)
