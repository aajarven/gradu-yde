import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import clustering
import LGfinder
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pylab
import physUtils
from matplotlib import rc
from sklearn.cluster import DBSCAN

if __name__ == "__main__":
	simpath = "/scratch/sawala/milkomedia_ii/milkomedia_26_DMO/groups_008_z000p000/"
	saveloc = "../../kuvat/mollweide+hubble.pdf"
	mindist = 1.5
	maxdist = 5.0
	eps = 0.16
	ms = 4

	LGdicts = LGfinder.readAndFind(simpath, output='haloes', expansion=True)

	maxUncontaminatedDist = 0
	for LGdict in LGdicts:
		if LGdict['contaminationDistance'] > maxUncontaminatedDist:
			maxUncontaminatedDist = LGdict['contaminationDistance']
			bestLG = LGdict
	
	if bestLG['mass1'] < bestLG['mass2']:
		MWindex = bestLG['ind1']
		centre = bestLG['cop1']
	else:
		MWindex = bestLG['ind2']
		centre = bestLG['cop2']

	cop = bestLG['cops']
	
	distances = np.array([physUtils.distance(centre, pos) for pos in cop])
	distmask = np.array([distance < maxdist and distance > mindist for distance
					  in distances])

	distances = distances[distmask]
	relvel = (bestLG['vels'] - bestLG['vels'][MWindex])[distmask]
	cop = cop[distmask]
	directions = np.array([physUtils.sphericalCoordinates(pos - centre) for pos
						in cop])
	radvel = np.array([physUtils.velocityComponents(relvel[j], cop[j] -
												 centre)[0] for j in
					range(len(cop))])

	fitdata = clustering.precomputeDistances(directions)
	db = DBSCAN(eps=eps, min_samples=ms, metric='precomputed', ).fit(fitdata)
	labels = db.labels_
	unique_labels = set(labels)
	clusters = len(unique_labels) - (1 if -1 in labels else 0)

	colours = plt.cm.tab20(np.linspace(0, 1, 20))


	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)

	F = pylab.gcf()
	F.set_size_inches( (5, 5) )
	
	# mollweide
	gs = gridspec.GridSpec(2, 1)
	plt.subplot(gs[0, 0], projection="mollweide")

	plt.grid(True)
	ax = plt.gca()
	ax.set_xticklabels([])

	for k, col in zip(unique_labels, colours):
		if k == -1:
			# Black small dots used for noise.
			col = 'k'
			size = 1
		else:
			size = 16

		class_member_mask = (labels == k)

		xy = directions[class_member_mask]
		plt.scatter(xy[:, 0], xy[:, 1], facecolors=col, edgecolors='k',
					s=size)

	
	# Hubble
	plt.subplot(gs[1, 0])
	
	for k, col in zip(unique_labels, colours):
		if k == -1:
			col = 'k'
			size = 1
		else:
			size = 16

		class_member_mask = (labels == k)

		plt.scatter(distances[class_member_mask],
					radvel[class_member_mask],
					facecolors=col, edgecolors='k',
					s=size)
#		plt.title("Distances and radial velocities of same haloes")
		axes = plt.gca()
		distRange = maxdist - mindist
		axes.set_xlim([mindist - distRange*0.01,
					   maxdist + distRange*0.01])
		plt.xlabel("Distance (Mpc)")
		plt.ylabel("Radial velocity (km/s)")

	plt.tight_layout()
	plt.savefig(saveloc)
