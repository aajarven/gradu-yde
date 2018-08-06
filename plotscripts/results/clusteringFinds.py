#-*- coding:utf-8 -*- 

import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import clustering
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
from matplotlib.ticker import FixedLocator
import LGfinder 
import filereader
import physUtils
from sibeliusConstants import *
from sklearn.cluster import DBSCAN



if __name__ == "__main__":

	inputfile = "../input/upTo5Mpc-fullpath.txt"
#	inputfile = "../input/one-fullpath.txt"
	saveloc = "../../kuvat/"

	simIndex = 0

	samples = 30
	minEPS = 0.01 
	maxEPS = 0.31
	EPSvalues = np.arange(minEPS, maxEPS+(maxEPS-minEPS)/samples,
					   (maxEPS-minEPS)/samples)
	minMS = 1
	maxMS = 31
	MSvalues = np.arange(minMS, maxMS+1)
	clusters = np.zeros((len(EPSvalues), len(MSvalues)))
	diameters = np.zeros((len(EPSvalues), len(MSvalues)))

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

#		meansep = np.mean([min(x[x>0]) for x in fitdata])

		for EPSindex in range(len(EPSvalues)):
			for MSindex in range(len(MSvalues)):
				eps = EPSvalues[EPSindex]#*meansep
				ms = MSvalues[MSindex]
				db = DBSCAN(eps=eps, min_samples=ms, metric='precomputed', ).fit(fitdata)
				labels = db.labels_
				labels_nozero = set(labels[labels != -1])
				
				numberOfClusters = len(set(labels)) - (1 if -1 in labels else 0)
				clusters[EPSindex, MSindex] += numberOfClusters
				
				for label in labels_nozero:
					mask = labels==label
					mask2D = np.ones_like(fitdata)
					mask2D[:, mask] = 0
					mask2D[np.logical_not(mask), :] = 1

					distancesInCluster = np.ma.masked_array(fitdata, mask2D)
					diameters[EPSindex, MSindex] += np.amax(distancesInCluster)


	meanClusters = clusters / (sim+1)
#	print(diameters)
#	print(clusters)
	meanDiameters = np.divide(diameters, clusters)
	meanDiameters = np.multiply(meanDiameters, 180.0/np.pi)
	meanDiametersNoInf = np.ma.masked_where( np.logical_or(
		np.isinf(meanDiameters), np.isnan(meanDiameters)), meanDiameters)

	# replace zeroes with small value for log plotting
	meanClusters[meanClusters == 0] = np.finfo(np.float).eps
	matplotlib.rcParams['axes.unicode_minus'] = False
	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)
	params = {'text.latex.preamble' : [r'\usepackage{gensymb}']}
	plt.rcParams.update(params)

	# number of clusters plotting
	fig = plt.gcf()
	ax = plt.gca()
	
	pcm = ax.pcolormesh(MSvalues, EPSvalues, meanClusters, cmap='magma',
					 vmin=1, vmax=25, edgecolors='face')
	cb = fig.colorbar(pcm, ax=ax, extend='both', ticks=[1, 5, 10, 15, 20, 25],
				  label='Mean number of clusters found')
	cb.ax.set_yticklabels([1, 5, 10, 15, 20, 25])

	minorticks = pcm.norm(np.arange(1, 25, 1))
	cb.ax.yaxis.set_ticks(minorticks, minor=True)

	xticks = np.array([1, 5, 10, 15, 20, 25, 30])
	xminorticks = np.arange(1, 31, 1)+0.5

	#yticks = np.arange(min(EPSvalues), max(EPSvalues), 0.05)
	yticks = np.arange(0.05, 0.31, 0.05)

	ax.set_xticks(xticks + 0.5*(MSvalues[1]-MSvalues[0]), minor=False)
	ax.set_yticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30] + 0.5*(EPSvalues[1] -
														   EPSvalues[0]), minor=False)
	ax.set_yticks(EPSvalues + 0.5*(EPSvalues[1] - EPSvalues[0]), minor=True)
#	ax.set_yticks()#yticks + 0.5*(EPSvalues[1]-EPSvalues[0]), minor=False)
	ax.set_xticklabels(xticks.astype(int), minor=False)
	ax.set_yticklabels(yticks, minor=False)
	ax.xaxis.set_minor_locator(FixedLocator(xminorticks))

	ax.set_ylim(min(EPSvalues), max(EPSvalues))
	ax.set_xlabel("MinPts (subhaloes)")
	ax.set_ylabel(r"$\varepsilon$ (radians)",
			   multialignment='center')

	plt.tight_layout(rect=[0.03, 0.03, 1, 1])#rect=[0.03, 0.04, 0.99, 0.999])
	fig.set_size_inches(4, 3.2)

	plt.savefig(saveloc + "clusteringParameters.pdf")
	
	
	# diameter plotting
	plt.cla()
	plt.clf()

	fig = plt.gcf()
	ax = plt.gca()
	
	pcm = ax.pcolormesh(MSvalues, EPSvalues, meanDiametersNoInf, cmap='magma',
					vmin=0, vmax=120, edgecolors='face')
	cb = fig.colorbar(pcm, ax=ax, extend='max',
				  label=r'Mean diameter of cluster ($^\circ$)')

	ax.axis([min(MSvalues), max(MSvalues), min(EPSvalues), max(EPSvalues)])
	ax.set_xticks(xticks + 0.5*(MSvalues[1]-MSvalues[0]), minor=False)
	ax.set_yticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30] + 0.5*(EPSvalues[1] -
														   EPSvalues[0]),
			   minor=False)
	ax.set_yticks(EPSvalues + 0.5*(EPSvalues[1] - EPSvalues[0]), minor=True)
	#ax.set_yticks(yticks + 0.5*(EPSvalues[1]-EPSvalues[0]), minor=False)
	ax.set_xticklabels(xticks.astype(int), minor=False)
	ax.set_yticklabels(yticks, minor=False)
	ax.xaxis.set_minor_locator(FixedLocator(xminorticks))
	ax.set_xlim(min(MSvalues), max(MSvalues))
	ax.set_ylim(min(EPSvalues), max(EPSvalues))
	ax.set_xlabel("MinPts (subhaloes)")
	ax.set_ylabel(r"$\varepsilon$ (radians)",
			   multialignment='center')

	plt.tight_layout()#rect=[-0.03, -0.03, 1.03, 1.05])
	plt.autoscale()
	fig.set_size_inches(4, 3.2)

	plt.savefig(saveloc + "clusterDiameter.pdf")
