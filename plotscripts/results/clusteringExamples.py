# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import clustering
import LGfinder
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import gridspec
import numpy as np
import physUtils
from sklearn.cluster import DBSCAN

# HUOM OBS NB: kaatuu jos colours on liian lyhyt
def loopClusterPlotting(fitdata, axes, clusteringParameters, colours):
	for ax, parameters in zip(axes, clusteringParameters):
		eps = parameters[1]
		#eps = clustering.meanSeparationEps(fitdata, parameters[1])
		db = DBSCAN(eps=eps, min_samples=parameters[0], metric='precomputed', ).fit(fitdata)
		labels = db.labels_
		uniqueLabels = set(labels)
		clusters = len(set(labels)) - (1 if -1 in labels else 0)
		print(clusters)
		for l in uniqueLabels:
			if l == -1:
				col = 'k'
				size = 2
			else:
				col = colours[l]
				size = 16
			
			mask = (labels == l)
			members = directions[mask]
			ax.scatter(members[:, 0], members[:, 1], facecolors=col,
			  edgecolors='k', linewidth=0.9, s=size)
			ax.grid(b=True)
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_title(r"$\varepsilon$={:.2f}".format(parameters[1]) + ", minsamples=" +
				str(parameters[0]))

if __name__ == "__main__":

	infile = "/scratch/sawala/milkomedia_ii/milkomedia_97_DMO/groups_008_z000p000/"
	saveloc = "../../kuvat/"

	LGdicts = LGfinder.readAndFind(infile, output='haloes', expansion=True)
	
	# find LG analogue that is closest to the centre of the high res area
	maxUncontaminatedDist = 0
	for LGdict in LGdicts:
		if LGdict['contaminationDistance'] > maxUncontaminatedDist:
			maxUncontaminatedDist = LGdict['contaminationDistance']
			d = LGdict
		
	centre = d['centre']
	if d['mass1'] < d['mass2']:
		centre = d['cop1']
	else:
		centre = d['cop2']

	cop = d['cops']
	distances = np.array([physUtils.distance(centre, pos) for pos in cop])

	distmask = np.array([distance < 5.0 and distance > 1.5 for distance in
					  distances])
	
	distances = distances[distmask]
	cop = cop[distmask]

	directions = np.array([physUtils.sphericalCoordinates(pos - centre) for
						   pos in cop])

	fitdata = clustering.precomputeDistances(directions)

	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)

	# general examples

	clusteringParameters = [(4, 0.10), (2, 0.16),
						 (4, 0.16), (6, 0.16),
						 (4, 0.24), (12, 0.16)]
	f, axarr = plt.subplots(3, 2, subplot_kw=dict(projection='mollweide'))
	colours = plt.cm.tab20(np.linspace(0, 1, 20))
	
	loopClusterPlotting(fitdata, axarr.flatten(), clusteringParameters,
					 colours)

	plt.tight_layout(rect=[-0.02, 0, 1.02, 1])
	plt.subplots_adjust(wspace=0)
	
	f.set_size_inches(5.9, 5.62)

	plt.savefig(saveloc + "clusteringExamples.svg")


	# examples of values near the chosen value
	plt.cla()
	plt.clf()

	gs = gridspec.GridSpec(3, 2, height_ratios=[3.55, 2, 2])
	bigax = plt.subplot(gs[0, :], projection="mollweide")
	smallax1 = plt.subplot(gs[1, 0], projection="mollweide")
	smallax2 = plt.subplot(gs[1, 1], projection="mollweide")
	smallax3 = plt.subplot(gs[2, 0], projection="mollweide")
	smallax4 = plt.subplot(gs[2, 1], projection="mollweide")

	colours = plt.cm.tab10(np.linspace(0, 1, 10))
	clusteringParameters = [(4, 0.16),
						 (4, 0.14), (7, 0.16),
						 (4, 0.18), (10, 0.16)]
	loopClusterPlotting(fitdata, [bigax, smallax1, smallax2, smallax3,
							   smallax4], clusteringParameters, colours)

	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)
	
	f.set_size_inches(5.9, 7.0)
	
	plt.savefig(saveloc + "smallClusteringVariations.svg")
