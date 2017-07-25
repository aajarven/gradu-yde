#-*- coding:utf-8 -*- 

import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import clustering
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
import LGfinder 
import filereader
import physUtils
from sibeliusConstants import *
from sklearn.cluster import DBSCAN


if __name__ == "__main__":

	inputfile = "../input/hyvat-fullpath.txt"
	saveloc = "../../kuvat/clusteringParameters.svg"

	simIndex = 0

	samples = 50
	minEPS = 0.5 
	maxEPS = 13.0
	EPSvalues = np.arange(minEPS, maxEPS+(maxEPS-minEPS)/samples,
					   (maxEPS-minEPS)/samples)
	minMS = 1
	maxMS = 21
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
				eps = EPSvalues[EPSindex]*meansep
				ms = MSvalues[MSindex]
				db = DBSCAN(eps=eps, min_samples=ms, metric='precomputed', ).fit(fitdata)
				labels = db.labels_
				clusters[EPSindex, MSindex] += len(set(labels)) - (1 if -1 in
													  labels else 0)



	np.savetxt("/home/aajarven/Z-drive/duuni/extragal/gradu-yde/plotscripts/tmp-out/clusters.txt",
			clusters)
	meanClusters = clusters / (sim)

	np.savetxt("/home/aajarven/Z-drive/duuni/extragal/gradu-yde/plotscripts/tmp-out/meanClusters.txt",
			meanClusters)

	matplotlib.rcParams['axes.unicode_minus'] = False

	fig = plt.gcf()
	ax = plt.gca()


	pcm = ax.pcolormesh(MSvalues, EPSvalues, meanClusters,
					 norm=colors.LogNorm(vmin=1, vmax=30), cmap='viridis')
	cb = fig.colorbar(pcm, ax=ax, extend='both')
	cb.ax.minorticks_on()

	xticks = np.arange(np.ceil(min(MSvalues)), np.floor(max(MSvalues)), 2)
	yticks = np.arange(np.ceil(min(EPSvalues)), np.floor(max(EPSvalues)), 1)

	ax.axis([min(MSvalues), max(MSvalues), min(EPSvalues), max(EPSvalues)])
	ax.set_xticks(xticks + 0.5*(MSvalues[1]-MSvalues[0]), minor=False)
	ax.set_yticks(yticks + 0.5*(EPSvalues[1]-EPSvalues[0]), minor=False)
	ax.set_xticklabels(xticks.astype(int), minor=False)
	ax.set_yticklabels(yticks, minor=False)
	ax.set_xlim(min(MSvalues), max(MSvalues))
	ax.set_ylim(min(EPSvalues), max(EPSvalues))


	ax.set_adjustable("box-forced")

#	plt.imshow(meanClusters, interpolation='nearest', cmap='viridis')
#	plt.colorbar()
	

	rc('font', **{'family':'serif','serif':['Palatino']})
#	rc('text', usetex=True)

	plt.tight_layout()


	fig.set_size_inches(4.2, 4.0)

	plt.savefig(saveloc)
