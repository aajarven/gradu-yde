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
from matplotlib import ticker


def percentFormat(x, position):
	s = str(int(100 * x))
	return s + r'$\%$'

if __name__ == "__main__":
	inputfile = "../input/lgfound-fullpath.txt" 
	saveloc = "../../kuvat/LGproperties.pdf"

	lines =  sum(1 for line in open(inputfile))
	radvel = np.full(lines, np.nan)
	tanvel = np.full(lines, np.nan)
	distance = np.full(lines, np.nan)
	mass = np.full(lines, np.nan)
	massdifference = np.full(lines, np.nan)
	massratio = np.full(lines, np.nan)
	overdensity2mpc = np.full(lines, np.nan)

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
		
		distances = np.array([physUtils.distance(centre, pos) for pos in cop])
		mass2mpc = sum(masses[np.where(distances<=2)])
		overdensity2mpc[sim] = (mass2mpc/(4.0/3.0*math.pi*(2**3)))/critical_density
		
	
	#### plotting #####
	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)
	rcParams['text.latex.preamble'] = [r'\usepackage{wasysym}']
	rcParams.update({'font.size': 13})
	formatter = ticker.FuncFormatter(percentFormat)

	fig, axarr = plt.subplots(4, 2, sharey=True)

	ax = axarr[0, 0]
	weights = np.ones_like(radvel)/float(len(radvel))
	ax.hist(radvel, weights=weights, color='0.75', edgecolor='black',
		 bins=np.arange(10, 176, 15))
	ax.set_xlabel("Radial velocity (km/s)")
	ax.set_xticks(np.arange(10, 176, 30))
	ax.yaxis.set_major_formatter(formatter)
	
	ax = axarr[0, 1]
	weights = np.ones_like(tanvel)/float(len(tanvel))
	ax.hist(tanvel, weights=weights, color='0.75', edgecolor='black',
		 bins=np.arange(0, 51, 5))
	ax.set_xlabel("Tangential velocity (km/s)")
	ax.set_xticks(np.arange(0, 51, 10))
	ax.yaxis.set_major_formatter(formatter)

	ax = axarr[1, 0]
	weights = np.ones_like(distance)/float(len(distance))
	ax.hist(distance, weights=weights, color='0.75', edgecolor='black',
		 bins=np.arange(0.6, 1.001, 0.05))
	majorLocator = ticker.MultipleLocator(0.1)
	minorLocator = ticker.MultipleLocator(0.05)
	ax.xaxis.set_major_locator(majorLocator)
	ax.xaxis.set_minor_locator(minorLocator)
	ax.set_xlabel("Distance (Mpc)")
	ax.yaxis.set_major_formatter(formatter)

	ax = axarr[1, 1]
	weights = np.ones_like(mass)/float(len(overdensity2mpc))
	ax.hist(overdensity2mpc, weights=weights, color='0.75', edgecolor='black',
		 bins=np.arange(0.2, 3.0, 0.2))
	ax.set_xlabel(r'$\frac{\rho_{r<2~\mathrm{Mpc}}}{\rho_{crit}}$',
			   multialignment='center')
	ax.set_xticks(np.arange(0.2, 3.1, 0.4))
	ax.yaxis.set_major_formatter(formatter)
	ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
	
	ax = axarr[2, 0]
	weights = np.ones_like(mass)/float(len(mass))
	ax.hist(mass/1e12, weights=weights, color='0.75', edgecolor='black',
		 bins=np.arange(0.5, 6.6, 0.5))
	ax.set_xlabel(r'Combined mass ($10^{12}\ M_{\astrosun}$)',
			   multialignment='center')
	ax.set_xticks(np.arange(1, 6.6, 1))
	ax.yaxis.set_major_formatter(formatter)
	ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

	ax = axarr[2, 1]
	weights = np.ones_like(massdifference)/float(len(massdifference))
	ax.hist(massdifference/1e12, weights=weights, color='0.75',
		 edgecolor='black', bins=np.arange(0, 4.1, 0.5))
	ax.set_xlabel(r"Mass difference ($10^{12}\ M_{\astrosun}$)",
			   multialignment='center')
	ax.yaxis.set_major_formatter(formatter)
	ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
	majorLocator = ticker.MultipleLocator(1.0)
	minorLocator = ticker.MultipleLocator(0.5)
	ax.yaxis.set_major_formatter(formatter)
	ax.xaxis.set_major_locator(majorLocator)
	ax.xaxis.set_minor_locator(minorLocator)

	ax = axarr[3, 0]
	weights = np.ones_like(massratio)/float(len(massratio))
	ax.hist(massratio*100, weights=weights, color='0.75', edgecolor='black',
		 bins=np.arange(10, 51, 5))
	ax.set_xlabel(r"Percent of mass in the\\less massive primary",
			   multialignment='center', linespacing=10.0)
	majorLocator = ticker.MultipleLocator(5)
#	minorLocator = ticker.MultipleLocator(0.05)
	ax.yaxis.set_major_formatter(formatter)
	ax.xaxis.set_major_locator(majorLocator)
#	ax.xaxis.set_minor_locator(minorLocator)

	plt.tight_layout()
	plt.subplots_adjust(hspace = 0.8)

	fig.set_size_inches(5.9, 8)# 6.5)

	plt.savefig(saveloc)
