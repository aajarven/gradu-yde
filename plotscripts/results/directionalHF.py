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
from matplotlib import rc
import numpy as np
import sys

import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.projections import PolarAxes

#https://stackoverflow.com/a/44190325
def setup_axes(fig, rect, maxR):
    tr = PolarAxes.PolarTransform() 

    theta0, theta1 = 0, np.pi
    r0, r1 = 0, maxR
    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(theta0, theta1, r0, r1))

    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax1)

    # adjust axis
    ax1.axis["left"].set_axis_direction("bottom")
    ax1.axis["right"].set_axis_direction("top")
    ax1.axis["bottom"].set_visible(False)
    ax1.axis["top"].set_axis_direction("bottom")
    ax1.axis["top"].toggle(ticklabels=True, label=True)
    ax1.axis["top"].major_ticklabels.set_axis_direction("top")
    ax1.axis["top"].label.set_axis_direction("top")
#    ax1.axis["left"].label.set_text(r"z")
#    ax1.axis["top"].label.set_text(r"RA")

    # create a parasite axes whose transData in RA, cz
    aux_ax = ax1.get_aux_axes(tr)

    aux_ax.patch = ax1.patch  
    ax1.patch.zorder = 0.9  

    return ax1, aux_ax


if __name__ == "__main__":
#	inputfile = "../input/allButDuplicates-fullpath.txt" 
	inputfile = "../input/hundred.txt"
	outputdir = "../../kuvat/"

	mindist = 1.0
	maxdist = 5.0
	eps = 1.8
	ms = 10

	binlimits = np.linspace(0.0, np.pi, 19)
	angles = []
	zeros = []
	h0s = []
	for i in range(len(binlimits)-1):
		angles.append((binlimits[i]+binlimits[i+1])/2.0)
		h0s.append([])
		zeros.append([])

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
			MWindex = LG[1]
			andromedaIndex = LG[0]
		else:
			MWindex = LG[0]
			andromedaIndex = LG[1]

		MWcop = cop[MWindex]
		closestContDist = physUtils.findClosestDistance(MWcop,
														contaminatedPositions)
		if closestContDist < maxdist:
			continue
		
		LGvector = cop[andromedaIndex] - cop[MWindex]
		
		centreVel = staticVel[MWindex]
		distances = np.array([physUtils.distance(MWcop, c) for c in
												 cop])
		vel = physUtils.addExpansion(staticVel, cop, MWcop)
	
		LGrelVel = vel[LG[0]] - vel[LG[1]]
		LGrelVelComponents = physUtils.velocityComponents(LGrelVel, cop[LG[1]]-cop[LG[0]])
		LGdistance = physUtils.distance(cop[LG[0]], cop[LG[1]])

		mask = np.array([d < maxdist and d > mindist for d in
							distances])
		cop = cop[mask]
		vel = vel[mask]
		distances = distances[mask]

		radvel = np.array([physUtils.velocityComponents(vel[j] - centreVel,
														cop[j] - MWcop)[0]
						   for j in range(len(vel))])

		directions = np.array([physUtils.angleBetween(LGvector, pos-MWcop) for pos in
						 cop])
	

		##### extracting interesting data starts #####
		for angleIndex in range(len(binlimits)-1):
			angleMask = np.array([direction < binlimits[angleIndex+1] and
						 direction >= binlimits[angleIndex] for direction in
						 directions])
#			print("[" + str(binlimits[angleIndex]) + ", " +
#		 str(binlimits[angleIndex+1]) + "]:\t" + str(np.sum(angleMask)) )
			
			if np.sum(angleMask) < 10:
				zeros[angleIndex].append(np.nan)
				h0s[angleIndex].append(np.nan)
			else:
				(H0, zero) = simpleFit(distances[angleMask], radvel[angleMask])
				zeros[angleIndex].append(zero)
				h0s[angleIndex].append(H0)




	##### plotting #####

	h0s = np.array(h0s)
	zeros = np.array(zeros)
	

	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)
	params = {'text.latex.preamble' : [r'\usepackage{wasysym}']}
	plt.rcParams.update(params)

#	fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))

	fig = plt.figure()
	ax1, aux_ax1 = setup_axes(fig, 211, 90)
	ax1.axis["left"].label.set_text(r"$H_0$ (km/s/Mpc)")
	ax1.axis["top"].label.set_text(r"$\phi$")

	aux_ax1.plot(angles, np.nanmean(h0s, axis=1))

	ax2, aux_ax2 = setup_axes(fig, 212, 2.5)
	aux_ax2.plot(angles, np.nanmean(zeros, axis=1))
	ax2.axis["left"].label.set_text(r"$H_0$ (km/s/Mpc)")
	ax2.axis["top"].label.set_text(r"$\phi$")

#	print(h0s)
#	print(np.nanmean(h0s, axis=1))
#	ax1.plot(angles, np.nanmean(h0s, axis=1))
#	ax2.plot(angles, np.nanmean(zeros, axis=1))
#	ax2.set_xlim([0, np.pi])

	plt.tight_layout()
#	fig.set_size_inches(5.9, 2.6)

#	plt.xlabel("Hubble flow zero point (Mpc from LG centre)")
#	plt.ylabel("Combined mass of Milky Way and Andromeda (Solar masses)")
#	plt.xlim(xmin, xmax)
	plt.savefig(outputdir+"directionalHF.svg")

