# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

from filereader import readTextTable
import anisotropymap
import LGfinder
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg
import numpy as np
import os
import physUtils
import sys
from transitiondistance import findBestHubbleflow


if __name__ == "__main__":

	dirname = "/scratch/sawala/milkomedia_ii/milkomedia_97_DMO/groups_008_z000p000/"
	saveloc = ("/home/aajarven/Z-drive/duuni/extragal/gradu-yde/kuvat/"
			"mollweide-anisotropy.svg")

	LGdicts = LGfinder.readAndFind(dirname, output='haloes')
	
	# find LG analogue that is closest to the centre of the high res area
	maxUncontaminatedDist = 0
	for LGdict in LGdicts:
		if LGdict['contaminationDistance'] > maxUncontaminatedDist:
			maxUncontaminatedDist = LGdict['contaminationDistance']
			d = LGdict
		
	if d['mass1'] < d['mass2']:
		centre = d['cop1']
	else:
		centre = d['cop2']

	cop = d['cops']
	relvel = d['vels'] - d['centrevel']
	distances = np.array([physUtils.distance(centre, pos) for pos in cop])

	distmask = np.array([distance > 1.5 and distance < 5.0 for distance in distances])
	distances = distances[distmask]

	directions = np.array([physUtils.sphericalCoordinates(pos - centre) for
						   pos in cop[distmask]])

	mass = d['masses'][distmask]

	radvel = np.array([physUtils.velocityComponents(relvel[j], cop[j] -
													centre)[0] for j in
													range(len(cop[distmask]))])

	fit = findBestHubbleflow(distances, radvel)[0]
	for j in range(len(radvel)):
		radvel[j] = radvel[j] - (fit[0]*distances[j] + fit[1])

	plt.subplot(111, projection="mollweide")
	plt.grid(True)
	fig = plt.gcf()
	ax = plt.gca()
	ax.set_xticklabels([])
#	plt.title("Haloes around Milky Way analogue\nfrom 1.5 Mpc to 5.0 Mpc away",
#		   y=1.08)

	cmap = anisotropymap.shiftedColorMap(plt.cm.get_cmap('RdBu'),
						   midpoint=-min(radvel)/(max(radvel)-min(radvel)))

	fig.set_size_inches(5.9, 2.9)

	sc = plt.scatter(directions[:,0], directions[:,1], c=radvel, 
					 cmap=cmap, s=18)
	matplotlib.rcParams['axes.unicode_minus'] = False
	cb = plt.colorbar(sc, fraction=0.046, pad=0.04)

	cb.set_label("Deviation from Hubble flow fit (km/s)")
	
	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)

	plt.tight_layout()

	plt.savefig(saveloc)

