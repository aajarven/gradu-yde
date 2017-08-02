# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import clustering
import LGfinder
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import physUtils
from sklearn.cluster import DBSCAN

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

	
	clusteringParameters = [(6, 4.5), (15, 4.5),
						 (6, 2.0), (15, 2.0),
						 (6, 1.0), (15, 1.0)]
	f, axarr = plt.subplots(3, 2, subplot_kw=dict(projection='mollweide'))
	colours = plt.cm.Set1(np.linspace(0, 1, 14))
	
	for ax, parameters in zip(axarr.flatten(), clusteringParameters):
		eps = clustering.meanSeparationEps(fitdata, parameters[1])
		db = DBSCAN(eps=eps, min_samples=parameters[0], metric='precomputed', ).fit(fitdata)
		labels = db.labels_
		uniqueLabels = set(labels)
		clusters = len(set(labels)) - (1 if -1 in labels else 0)

		for l in uniqueLabels:
			if l == -1:
				col = 'k'
				size = 1
			else:
				col = colours[l] # oletetaan alle 12 klusteria, muulloin kaatuu
				size = 16
			
			mask = (labels == l)
			members = directions[mask]
			ax.scatter(members[:, 0], members[:, 1], facecolors=col,
			  edgecolors='k', s=size)
			ax.grid(b=True)
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_title(r"$\varepsilon$=" + str(parameters[1]) + ", minsamples=" +
				str(parameters[0]))
	
	
	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)

	plt.tight_layout(rect=[-0.05, 0, 1.05, 1])
	plt.subplots_adjust(wspace=0)
	
	f.set_size_inches(5.9, 5.3)

	plt.savefig(saveloc + "clusteringExamples.svg")
