# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import clustering
import LGfinder
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
import numpy as np
import physUtils
from sklearn.neighbors import NearestNeighbors

if __name__ == "__main__":
	#f = plt.figure()
	#ax = f.add_subplot(111)
	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)
	matplotlib.rcParams.update({'font.size': 13})

	fig = plt.figure()

	simlist_path = "../input/allButDuplicates-fullpath.txt"
	simlist_file = open(simlist_path, 'r')
	simlist = [line.strip() for line in simlist_file]

#	simlist =["/scratch/sawala/milkomedia_ii/milkomedia_0_DMO/groups_008_z000p000/"]
	
	for simdir in simlist:
		print(simdir)

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
		if max(distances) < 5.0:
			print("distance criteria failed")
			continue
		distmask = np.array([distance < 5.0 and distance > 1.5 for distance in
						  distances])

		distances = distances[distmask]
		cop = cop[distmask]

		directions = np.array([physUtils.sphericalCoordinates(pos - centre) for
							   pos in cop])

		fitdata = clustering.precomputeDistances(directions)
		meansep = np.mean([min(x[x>0]) for x in fitdata])
		
		neighbours= NearestNeighbors(n_neighbors=5,
											metric='precomputed').fit(fitdata)
#		neighbours= NearestNeighbors(n_neighbors=5,
#											metric='precomputed').fit(np.divide(fitdata,
#											  meansep))
		k_distances, indices = neighbours.kneighbors(n_neighbors=4, return_distance=True)
		four_distances = k_distances[:,3]

		plt.plot(range(0, len(four_distances)), sorted(four_distances,
												reverse=True), alpha=0.1, c='k')
		
	
	plt.xlabel(r"Subhalo index")
	plt.ylabel('Distance to the $4^{\mathrm{th}}$ nearest\nneighbour (radians)',
			multialignment='center')
	plt.plot([0, 500], [0.17, 0.17])
	
	plt.xlim(0, None)
	plt.ylim(0, None)
	fig.set_size_inches(4.2, 3.5)
	plt.tight_layout()
	#plt.subplots_adjust(wspace=0)
	plt.savefig("../../kuvat/4-distances.svg")
