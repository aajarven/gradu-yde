# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

from filereader import readTextTable
import anisotropymap
import LGfinder
from matplotlib import rc
import matplotlib.colors as colors
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
			"mollweide-anisotropy.pdf")

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

	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)
	#matplotlib.rcParams['axes.unicode_minus'] = False
	
	plt.subplot(111, projection="mollweide")
	plt.grid(True)
	fig = plt.gcf()
	ax = plt.gca()
	ax.set_xticklabels([])
#	plt.title("Haloes around Milky Way analogue\nfrom 1.5 Mpc to 5.0 Mpc away",
#		   y=1.08)

	cmap_centered = anisotropymap.shiftedColorMap(plt.cm.get_cmap('RdBu_r'),
						   midpoint=-min(radvel)/(max(radvel)-min(radvel)))
	cmap_min = -600.0
	cmap_max = 400.0
	cmap_range = cmap_max - cmap_min
	cmap_bluerange = colors.LinearSegmentedColormap.from_list('velocitymap',
												 plt.cm.get_cmap('RdBu_r')(np.linspace(0,
												  cmap_range/(-2*cmap_min))))
	cmap_redrange = colors.LinearSegmentedColormap.from_list('velocitymap',
												 plt.cm.get_cmap('RdBu_r')(np.linspace(-0.25,
												  1.0)))


	fig.set_size_inches(5.9, 2.9)

	# colormap options: cmap_centered, cmap_bluerange and cmap_redrange
	sc = plt.scatter(directions[:,0], directions[:,1], c=radvel, 
					 cmap=cmap_centered, s=18, edgecolors='k')
	cb = plt.colorbar(sc, fraction=0.046, pad=0.04)

	cb.set_label("Deviation from Hubble flow fit (km/s)")
	

	plt.tight_layout()

	plt.savefig(saveloc)

