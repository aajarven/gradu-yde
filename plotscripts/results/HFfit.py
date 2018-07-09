#-*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import LGfinder
import filereader
import physUtils
from sibeliusConstants import *
import transitiondistance

if __name__ == "__main__":
	
	dirname = "/scratch/sawala/milkomedia_ii/milkomedia_129_DMO/groups_008_z000p000/"
	saveloc = "/home/aajarven/Z-drive/duuni/extragal/gradu-yde/kuvat/hubblefit.pdf"

	staticVel = filereader.readAllFiles(dirname, "Subhalo/Velocity", 3)
	mass = filereader.readAllFiles(dirname, "Subhalo/Mass", 1)
	fullCOP = filereader.readAllFiles(dirname, "Subhalo/CentreOfPotential", 3)
	FoFcontamination = filereader.readAllFiles(dirname, "FOF/ContaminationCount",
											   1)
	groupNumbers = filereader.readAllFiles(dirname, "Subhalo/GroupNumber", 1)

	# to physical units
	mass = mass/h0*1e10 # to M_☉
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

	LGs = LGfinder.findLocalGroup(staticVel, mass, cop, quiet=True,
								  outputAll=True)
	LG = LGs[LGfinder.chooseClosestToCentre(LGs, contaminationMask, fullCOP)]

	if mass[LG[0]] > mass[LG[1]]:
		centreIndex = LG[1]
	else:
		centreIndex = LG[0]
	centre = cop[centreIndex]
	centreVel = staticVel[centreIndex]
	
	closestContDist = physUtils.findClosestDistance(centre,
													contaminatedPositions)
	distances = np.array([physUtils.distance(centre, c) for c in cop])
	vel = physUtils.addExpansion(staticVel, cop, centre)

	# LG mass from MW and andromeda
	M_big2 = mass[LG[0]] + mass[LG[1]]
	
	originalDistances = list(distances)
	
	# contamination cut
	distmask = distances < closestContDist
	cop = cop[distmask]
	vel = vel[distmask]
	distances = distances[distmask]

	# radial velocities
	radvel = np.array([physUtils.velocityComponents(vel[j] - centreVel,
													cop[j] - centre)[0]
					   for j in range(len(vel))])
	
	(fit, flowstartdist) = transitiondistance.findBestHubbleflow(distances, radvel)
	print("HF slope: " + str(fit[0]))

	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)

	fig, ax = plt.subplots()
	
	ax.scatter(distances[distances < flowstartdist],
			   radvel[distances < flowstartdist], s=1, color=[0.6, 0.6,
															  0.6])
	ax.scatter(distances[distances >= flowstartdist],
			   radvel[distances >= flowstartdist], s=1, color=[0, 0, 0])
	
	# line
	ax.plot([0, max(distances)], [fit[1],
								  fit[1]+fit[0]*max(distances)],
			color=[0, 0, 0])

	ax.set_xlim(xmin=0, xmax=max(distances))
	ax.set_xlabel("Distance from Milky Way (Mpc)")
	ax.set_ylabel("Radial velocity (km/s)")


	plt.tight_layout(rect=[0.05, 0.12, 1, 1])
	
	fig.set_size_inches(4, 4)
	
	plt.savefig(saveloc)
