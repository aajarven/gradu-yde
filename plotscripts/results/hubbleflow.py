#-*- coding:utf-8 -*- 

import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import h5py
import hubblediagram
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib import rc
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D
import heapq
import math
import random
import LGfinder 
import numpy.ma as ma
import filereader
import physUtils
from sibeliusConstants import *
from optparse import OptionParser
import pylab


if __name__ == "__main__":

	inputs = [
		"/scratch/sawala/milkomedia_ii/milkomedia_193_DMO/groups_008_z000p000/",
		"/scratch/sawala/milkomedia_ii/milkomedia_168_DMO/groups_008_z000p000/"
	]
	saveloc = "../../kuvat/hubblediagrams.svg"

	simIndex = 0
	f, axes = plt.subplots(1, 2)

	f = plt.figure()
	bigaxis = f.add_subplot(111)    # The big subplot
	ax1 = f.add_subplot(121)
	ax2 = f.add_subplot(122)

	for (dirname, ax) in zip(inputs, [ax1, ax2]):
		simIndex = simIndex + 1

		vel = filereader.readAllFiles(dirname, "Subhalo/Velocity", 3)
		mass = filereader.readAllFiles(dirname, "Subhalo/Mass", 1)
		fullCOP = filereader.readAllFiles(dirname, "Subhalo/CentreOfPotential", 3)
		FoFcontamination = filereader.readAllFiles(dirname, "FOF/ContaminationCount",
											 1)
		groupNumbers = filereader.readAllFiles(dirname, "Subhalo/GroupNumber", 1)

		# to physical units
		mass = mass/h0*1e10 # to M_☉
		fullCOP = fullCOP/h0 # to Mpc

		# creation of mask with True if there is no contamination in Subhalo
		contaminationMask = np.asarray([FoFcontamination[int(group)-1]<1 for
								  group in groupNumbers])

		# save contaminated haloes for finding closest one
		contaminatedPositions = fullCOP[contaminationMask==False, :]

		# elimination of contaminated haloes
		vel = vel[contaminationMask,:]
		mass = mass[contaminationMask]
		cop = fullCOP[contaminationMask, :]
		groupNumbers = groupNumbers[contaminationMask]

		LGs = LGfinder.findLocalGroup(vel, mass, cop, quiet=True, outputAll=True)
		unmaskedLGs = LGfinder.maskedToUnmasked(LGs, cop, fullCOP)
		bestLGindex = LGfinder.chooseClosestToCentre(unmaskedLGs, contaminationMask, fullCOP)
		LG = LGs[bestLGindex]

		(main1ind, main2ind) = LGfinder.orderIndices(LG, mass)

		# center on MW 
		massCentre = cop[main1ind]
		massCentreVel = vel[main1ind]

		# expansion
		vel = physUtils.addExpansion(vel, cop, massCentre)

		dist = np.zeros(len(vel))
		speed = np.zeros(len(vel))

		# find most distant contaminated
		closestContDistance = physUtils.findClosestDistance(massCentre,
													  contaminatedPositions)

		# distances and velocities relative to centre
		for i in range(len(vel)):
			dist[i] = physUtils.distance(cop[i], massCentre)
			relVel = vel[i]-massCentreVel # velocity relative to movement of the mass CentreOfPotential
			diffInLocation = cop[i]-massCentre # vector defining difference in position 
			speed[i] = physUtils.velocityComponents(relVel, diffInLocation)[0] # radial velocity

		colours = np.zeros((len(vel), 4))
		colours[:, 3] = 1

		ax.scatter(dist[dist<closestContDistance],
			  speed[dist<closestContDistance],
			  color=colours[dist<closestContDistance], s=1)

		ax.set_xlim([0,closestContDistance])
		ax.set_ylim([-700, 900])
		
		ax.tick_params(which="major", length=6)
		ax.tick_params(which="minor", length=3.5)
		ax.xaxis.set_ticks([])
		ax.xaxis.set_minor_locator(FixedLocator(range(0, 6, 1)))
		ax.xaxis.set_minor_formatter(FormatStrFormatter("%3d"))
		ax.yaxis.set_ticks([0])
		minorticks = range(-800, 1000, 200)
		minorticks.remove(0)
		ax.yaxis.set_minor_locator(FixedLocator(minorticks))
		

		if simIndex == 1:
			ax.set_ylabel('Radial velocity (km/s)')
			ax.yaxis.set_minor_formatter(FormatStrFormatter("%3d"))
			
		else:
			ax.tick_params(labelleft=False)  
			ax.yaxis.set_minor_formatter(NullFormatter())

	bigaxis.spines['top'].set_color('none')
	bigaxis.spines['bottom'].set_color('none')
	bigaxis.spines['left'].set_color('none')
	bigaxis.spines['right'].set_color('none')
	bigaxis.tick_params(labelcolor='w', top='off', bottom='off', left='off',
				right='off')
	bigaxis.set_xlabel("Distance from MW centre (Mpc)")

	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)

	plt.tight_layout()
	plt.subplots_adjust(wspace=0)
	
	f.set_size_inches(5.9, 3)

	plt.savefig(saveloc)
