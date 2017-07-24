#-*- coding:utf-8 -*- 

import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import clustering
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import LGfinder 
import filereader
import physUtils
from sibeliusConstants import *
from sklearn.cluster import DBSCAN


if __name__ == "__main__":

	inputfile = "../input/lgfound-fullpath.txt" 
	saveloc = "../../kuvat/clusteringParameters.svg"

	simIndex = 0

	samples = 100
	minEPS = 0.5
	maxEPS = 3.0
	EPSvalues = np.arange(minEPS, maxEPS+(maxEPS-minEPS)/samples,
					   (maxEPS-minEPS)/samples)
	minMS = 2
	maxMS = 20
	MSvalues = np.arange(minMS, maxMS+1)
	clusters = np.zeros((len(EPSvalues), len(MSvalues)))

	f = open(inputfile, 'r')
	sim = -1
	for simdir in f.readlines():
		sim += 1
		print(sim)
		dirname = simdir.strip()
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
		centre = cop[main1ind]

		# distance range
		distances = np.array([physUtils.distance(centre, pos) for pos in cop])
		distanceMask = np.array([d > 1.5 and d < 5.0 for d in distances])
		cop = cop[distanceMask]

		directions = np.array([physUtils.sphericalCoordinates(pos - centre) for
						   pos in cop])
		fitdata = clustering.precomputeDistances(directions)

		meansep = np.mean([min(x[x>0]) for x in fitdata])

		for EPSindex in range(len(EPSvalues)):
			for MSindex in range(len(MSvalues)):
				db = DBSCAN(eps=EPSvalues[EPSindex]*meansep,
				min_samples=MSvalues[MSindex], metric='precomputed', ).fit(fitdata)
				labels = db.labels_
				clusters[EPSindex, MSindex] += len(set(labels)) - (1 if -1 in
													  labels else 0)


	meanClusters = clusters / (simIndex+1)

	np.savetxt("/home/aajarven/Z-drive/duuni/extragal/gradu-yde/plotscripts/tmp-out/meanClusters.txt",
			meanClusters)

	plt.imshow(meanClusters)

	

	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)

	plt.tight_layout()

	fig = plt.gcf()
	fig.set_size_inches(5.9, 3)

	plt.savefig(saveloc)
