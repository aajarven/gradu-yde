# -*- coding: utf-8 -*-

from __future__ import print_function
import LGfinder
import math
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from optparse import OptionParser
import os
import pylab
import physUtils
import random
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import sys
from transitiondistance import findBestHubbleflow

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

	fitdata = precomputeDistances(directions)
   
	db = DBSCAN(eps=eps, min_samples=ms, metric='precomputed', ).fit(fitdata)
	labels = db.labels_
	clusters = len(set(labels)) - (1 if -1 in labels else 0) 
