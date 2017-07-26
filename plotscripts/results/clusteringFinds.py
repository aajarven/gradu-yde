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

	inputfile = "../input/lgfound-fullpath.txt"
	saveloc = "../../kuvat/clusteringParameters.svg"

	simIndex = 0

	samples = 32
	minEPS = 0.5 
	maxEPS = 8.5
	EPSvalues = np.arange(minEPS, maxEPS+(maxEPS-minEPS)/samples,
					   (maxEPS-minEPS)/samples)
	minMS = 1
	maxMS = 35
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

	# replace zeroes with small value for log plotting
	meanClusters[meanClusters == 0] = np.finfo(np.float).eps

	matplotlib.rcParams['axes.unicode_minus'] = False
	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)

	fig = plt.gcf()
	ax = plt.gca()

	pcm = ax.pcolormesh(MSvalues, EPSvalues, meanClusters, cmap='magma',
					 vmin=1, vmax=20, edgecolors='face')
#					 norm=colors.LogNorm(vmin=1, vmax=20))
	cb = fig.colorbar(pcm, ax=ax, extend='both', ticks=[1, 5, 10, 15, 20],
				  label='Mean number of clusters found')
	cb.ax.set_yticklabels([1, 5, 10, 15, 20])

	minorticks = pcm.norm(np.arange(1, 20, 1))
	cb.ax.yaxis.set_ticks(minorticks, minor=True)


	xticks = np.arange(2, max(MSvalues), 2)
	yticks = np.arange(np.ceil(min(EPSvalues)), max(EPSvalues), 1)

	ax.axis([min(MSvalues), max(MSvalues), min(EPSvalues), max(EPSvalues)])
	ax.set_xticks(xticks + 0.5*(MSvalues[1]-MSvalues[0]), minor=False)
	ax.set_yticks(yticks + 0.5*(EPSvalues[1]-EPSvalues[0]), minor=False)
	ax.set_xticklabels(xticks.astype(int), minor=False)
	ax.set_yticklabels(yticks, minor=False)
	ax.set_xlim(min(MSvalues), max(MSvalues))
	ax.set_ylim(min(EPSvalues), max(EPSvalues))

#	ax.set_adjustable("box-forced")

	ax.set_xlabel("Minsamples")
	ax.set_ylabel(r"$\varepsilon$ (mean distances to neighbour)",
			   multialignment='center')

#	plt.imshow(meanClusters, interpolation='nearest', cmap='viridis')
#	plt.colorbar()
	

	plt.tight_layout()


	fig.set_size_inches(5.9, 4.0)

	plt.savefig(saveloc)
