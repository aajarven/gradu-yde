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
from sklearn.neighbors import NearestNeighbors

if __name__ == "__main__":
#	simlist = "../input/allButDuplicates-fullpath.txt"
	simdir = "/scratch/sawala/milkomedia_ii/milkomedia_0_DMO/groups_008_z000p000/"

	LGdicts = LGfinder.readAndFind(simdir, output='haloes', expansion=True)
	
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
	neighbours= NearestNeighbors(n_neighbors=5,
										metric='precomputed').fit(fitdata)
	four_distances, indices = neighbours.kneighbors(n_neighbors=4, return_distance=True)
	print(four_distances)
