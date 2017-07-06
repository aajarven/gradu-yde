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

	inputs = "../input/hyvat-fullpath.txt"
	saveloc = "../../kuvat/all/hubbleflows/"

	f = open(inputs, 'r')
	simIndex = 0
	for dirname in f.readlines():
		dirname = dirname.strip()
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

		plt.scatter(dist[dist<closestContDistance],
			  speed[dist<closestContDistance],
			  color=colours[dist<closestContDistance], s=4)

		axes = plt.gca()
		axes.set_xlim([0,closestContDistance])
		
		plt.xlabel('distance (Mpc)')
		plt.ylabel('velocity (km/s)')

		rc('font', **{'family':'serif','serif':['Palatino']})
		rc('text', usetex=True)

		F = pylab.gcf()
		F.set_size_inches(5.9, 5)

		plt.savefig(saveloc+"hubbleflow-"+str(simIndex)+".svg")
