# -*- coding:utf-8 -*-

from __future__ import print_function
import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')
sys.path.insert(0, '/home/aajarven/Z-drive/duuni/extragal/gradu-yde/plotscripts/background/')

import clustering
import clusterAnalysis
import filereader
from kstest import biggestDifference, CDFifyXData 
import LGfinder
import physUtils
from sibeliusConstants import *
from transitiondistance import simpleFit
import math
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy import stats
import sys

def blackBoxplot(bp):
	for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
		plt.setp(bp[element], color='k')

def ksplot(ax, x1, x2, y, label1, label2, verticalX, verticalMinY, verticalMaxY):
	ax.plot(x1, y, label=label1)
	ax.plot(x2, y, label=label2)
	ax.set_ylim(min(y), max(y))
	yLimits = ax.get_ylim()
	ax.axvline(x=verticalX, ymin=verticalMinY/(yLimits[1] - yLimits[0]),
			 ymax=verticalMaxY/(yLimits[1] - yLimits[0]))


def inClusterFit(db, velocities, distances, minSize=10, massMask = []):
	"""
	Calculates the mean H_0, zero point and velocity dispersion of the Hubble flow when
	it is fitted separately to each cluster with at least minSize members.
	Returns a tuple (H_0, zeropoint, dispersion).

	db : DBSCAN output
		The clustering database used to determine the cluster memberships of
		different subhaloes
	velocities : numpy array
		The radial velocities of the subhaloes
	distances : numpy array
		The distances to the subhaloes
	minSize : integer
		Minimum number of members in each cluster that is used in HF fitting.
		Default value is a minimum of 10 subhaloes in a cluster.
	massMask : boolean array
        Array with the same shape as velocities, containing true for subhaloes
        that have masses in allowed range and false for others. Only if all
        subhaloes in a cluster have accepted masses is that cluster used in HF
		fitting. Default is that all masses are accepted, represented by an empty array [].
	"""
	uniqueLabels = set(db.labels_)
	if -1 in uniqueLabels:
		uniqueLabels.remove(-1)
	uniqueLabels = np.array(list(uniqueLabels))

	H0s = np.full(len(uniqueLabels), np.nan)
	zeros = np.full(len(uniqueLabels), np.nan)
	deviations = np.full(len(uniqueLabels), np.nan)
	
	for i in range(len(uniqueLabels)):
		clusterMask = (db.labels_ == uniqueLabels[i])
		if np.sum(clusterMask) >= minSize:
			if len(massMask) == 0 or np.all(massMask[clusterMask]):
				(H0, zero) = simpleFit(distances[clusterMask],
						   velocities[clusterMask])
				H0s[i] = H0
				zeros[i] = zero

				velExcess = [radvel - (distance - zero) * H0 for radvel,
				 distance in zip(velocities[clusterMask], distances[clusterMask])]
				deviations[i] = np.std(velExcess, ddof=1)

#	if len(massMask) == 0:
#		print(np.count_nonzero(~np.isnan(zeros)))

	return (np.nanmean(H0s), np.nanmean(zeros), np.nanmean(deviations))


def outClusterFit(db, velocities, distances, minHaloes=10):
	"""
	Calculates the H_0, zero point and velocity dispersion of the Hubble flow
	when it is fitted using haloes that do not belong to any cluster. If there
	are not enough data points, tuple of NaNs is returned.

	db : DBSCAN output
		The clustering database used to determine the cluster memberships of
		different subhaloes
	velocities : numpy array
		The radial velocities of the subhaloes
	distances : numpy array
		The distances to the subhaloes
	minHaloes : integer
		Minimum number of subhaloes outside clusters in order for the HF to be
		fitted.	Default value is a minimum of 10 subhaloes in a cluster.
	"""
	outClusterMask = (db.labels_ == -1)
	if np.sum(outClusterMask) >= minHaloes:
		(H0, zero) = simpleFit(distances[outClusterMask],
							velocities[outClusterMask])
		velExcess = [radvel - (distance - zero) * H0 for radvel,
		 distance in zip(velocities[outClusterMask], distances[outClusterMask])]
		deviation = np.std(velExcess, ddof=1)

		return (H0, zero, deviation)


if __name__ == "__main__":
	inputfile = "../input/upTo5Mpc-no229-fullpath.txt" 
#	inputfile = "../input/upTo5Mpc-fullpath.txt" 
#	inputfile = "../input/hundred.txt"
#	inputfile = "../input/ten-fullpath.txt"
	outputdir = "../../kuvat/"

	mindist = 1.0
	maxdist = 5.0
	eps = 0.16
	ms = 4
	massThreshold = 8e11 #M_sun

	allZeros = []
	inClusterZeros = []
	outClusterZeros = []
	allH0s = []
	inClusterH0s = []
	outClusterH0s = []
	massCutH0s = []
	massCutZeros = []
	allDispersions = []
	inClusterDispersions = []
	outClusterDispersions = []
	massCutDispersions = []

	simIndex = 0
	f = open(inputfile, 'r')
	for simdir in f.readlines():
		print(simIndex)
		simIndex += 1
		dirname = simdir.strip()
		staticVel = filereader.readAllFiles(dirname, "Subhalo/Velocity", 3)
		mass = filereader.readAllFiles(dirname, "Subhalo/Mass", 1)
		fullCOP = filereader.readAllFiles(dirname, "Subhalo/CentreOfPotential", 3)
		FoFcontamination = filereader.readAllFiles(dirname, "FOF/ContaminationCount",
												   1)
		groupNumbers = filereader.readAllFiles(dirname, "Subhalo/GroupNumber", 1)

		fullCOP = fullCOP/h0 # to Mpc

		# creation of mask with True if there is no contamination in Subhalo
		contaminationMask = np.asarray([FoFcontamination[int(group)-1]<1 for group in
										  groupNumbers])

		# save contaminated haloes for finding closest one
		contaminatedPositions = fullCOP[contaminationMask==False, :]

		# elimination of contaminated haloes
		staticVel = staticVel[contaminationMask,:]
		mass = mass[contaminationMask]
		cop = fullCOP[contaminationMask, :]
		groupNumbers = groupNumbers[contaminationMask]
		
		# to physical units
		mass = mass/h0*1e10 # to M_☉
		
		LGs = LGfinder.findLocalGroup(staticVel, mass, cop, quiet=True,
									  outputAll=True)
		unmaskedLGs = LGfinder.maskedToUnmasked(LGs, cop, fullCOP)
		bestLGindex = LGfinder.chooseClosestToCentre(unmaskedLGs, contaminationMask, fullCOP)
		LG = LGs[bestLGindex]

		if mass[LG[0]] > mass[LG[1]]:
			centreIndex = LG[1]
		else:
			centreIndex = LG[0]

		centre = cop[centreIndex]
		centreVel = staticVel[centreIndex]
		massCentre = physUtils.massCentre(cop[LG[0]], cop[LG[1]], mass[LG[0]],
									mass[LG[1]])
		closestContDist = physUtils.findClosestDistance(massCentre,
														contaminatedPositions)
		if closestContDist < maxdist:
			continue
		
		distances = np.array([physUtils.distance(centre, c) for c in
												 cop])
		vel = physUtils.addExpansion(staticVel, cop, centre)
	
		LGrelVel = vel[LG[0]] - vel[LG[1]]
		LGrelVelComponents = physUtils.velocityComponents(LGrelVel, cop[LG[1]]-cop[LG[0]])
		LGdistance = physUtils.distance(cop[LG[0]], cop[LG[1]])

		mask = np.array([d < maxdist and d > mindist for d in
							distances])
		cop = cop[mask]
		vel = vel[mask]
		distances = distances[mask]
		mass = mass[mask]

		# radial velocities
		radvel = np.array([physUtils.velocityComponents(vel[j] - centreVel,
														cop[j] - centre)[0]
						   for j in range(len(vel))])


		##### extracting interesting data starts #####
		clusteringDB = clustering.runClustering(cop, centre, ms, eps,
										  meansep=False)
		labels = clusteringDB.labels_
		uniqueLabels = set(labels)

		clusterMemberMask = labels != -1 # True for in cluster
#		print(sum([not membership for membership in clusterMemberMask]))
#		(inClusterH0, inClusterZero) = simpleFit(distances[clusterMemberMask],
#										   radvel[clusterMemberMask])
#		(outClusterH0, outClusterZero) = simpleFit(
#			distances[clusterMemberMask == False],
#			radvel[clusterMemberMask == False])

		# all haloes
		(H0, zero) = simpleFit(distances, radvel)
		radvelResiduals = np.empty(radvel.shape)
		for i in range(len(radvel)):
			radvelResiduals[i] = radvel[i] - (distances[i] - zero) * H0
		allZeros.append(zero)
		allH0s.append(H0)
		allDispersions.append(np.std(radvelResiduals, ddof=1))

		# outside clusters
		(H0, zero, dispersion) = outClusterFit(clusteringDB, radvel, distances,
										 minHaloes=10)
		outClusterH0s.append(H0)
		outClusterZeros.append(zero)
		outClusterDispersions.append(dispersion)

		# inside clusters
		(H0, zero, dispersion) = inClusterFit(clusteringDB, radvel, distances,
										minSize=10)
		inClusterH0s.append(H0)
		inClusterZeros.append(zero)
		inClusterDispersions.append(dispersion)

		# mass cut clusters
		massMask = (mass <= massThreshold)
		(H0, zero, dispersion) = inClusterFit(clusteringDB, radvel, distances,
										minSize=10, massMask=massMask)
		if math.isnan(H0) or math.isnan(zero) or math.isnan(dispersion):
			print("No clusters with allowed minimum mass")
		else:
			massCutH0s.append(H0)
			massCutZeros.append(zero)
			massCutDispersions.append(dispersion)

		if len(np.where(clusteringDB.labels_ == -1)[0]) < 30:
			print("few unclustered: "+ str(len(np.where(clusteringDB.labels_ == -1)[0])))
		if len(np.where(clusteringDB.labels_ != -1)[0]) < 30:
			print("few clustered: "+ str(len(np.where(clusteringDB.labels_ != -1)[0])))
			
		

	##### plotting #####
	allZeros = np.array(allZeros)
	allH0s = np.array(allH0s)
	inClusterZeros = np.array(inClusterZeros)
	inClusterH0s = np.array(inClusterH0s)
	outClusterZeros = np.array(outClusterZeros)
	outClusterH0s = np.array(outClusterH0s)
	massCutH0s = np.array(massCutH0s)
	massCutZeros = np.array(massCutZeros)
	allDispersions = np.array(allDispersions)
	inClusterDispersions = np.array(inClusterDispersions)
	outClusterDispersions = np.array(outClusterDispersions)

#	print(inClusterH0s[np.argsort(inClusterH0s)[:10]])
#	print(inClusterH0s[np.argsort(inClusterH0s)[-10:]])
#	print(inClusterZeros[np.argsort(inClusterZeros)[:10]])
#	print(inClusterZeros[np.argsort(inClusterZeros)[-10:]])
#	print("")
#	minindex = np.argmin(inClusterH0s)
#	print(minindex)
#	print(inClusterH0s[minindex])
#	print(inClusterZeros[minindex])
#	print(allH0s[minindex])
#	print(allZeros[minindex])

	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)
	params = {'text.latex.preamble' : [r'\usepackage{wasysym}']}
	plt.rcParams.update(params)

	fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
	bp1 = ax1.boxplot([massCutH0s, inClusterH0s, outClusterH0s, allH0s], vert=False)
	blackBoxplot(bp1)
	bp2 = ax2.boxplot([massCutZeros, inClusterZeros, outClusterZeros, allZeros], vert=False)
	blackBoxplot(bp2)

	ax1.set_yticklabels(["Haloes in clusters with\nall members less massive\n"
					  r"than $8 \times 10^{11}~M_{\astrosun}$",
					  "Haloes in clusters", "Haloes outside clusters", "All haloes"
					 ], ha='right', multialignment='right')
	xlims = ax1.get_xlim()
	ax1.set_xticks(np.arange(math.ceil(xlims[0]/10)*10, xlims[1], 10), minor=True)
	ax1.set_xlabel(r"$H_0$ (km/s/Mpc)")
	xlims = ax2.get_xlim()
	ax2.set_xticks(np.arange(math.ceil(xlims[0]), xlims[1], 0.5), minor=True)
	ax2.set_xlim([-2.5, 3])
#	print([min(allZeros), max(allZeros)])
#	ax2.set_xticks(np.arange(-5, ylims[1], 5.0), minor=False)
	ax2.set_xlabel("Distance to Hubble\nflow zero point (Mpc)")
#	ax1.set_xlim([-25, 145])
#	ax2.set_xlim([-4, 4])

	print("")
	print("Outliers:")
	print("allH0s:", [H0 for H0 in allH0s if H0 < -10 or H0 > 140])
	print("inClusterH0s:", [H0 for H0 in inClusterH0s if H0 < -10 or H0 > 140])
	print("outClusterH0s:", [H0 for H0 in outClusterH0s if H0 < -10 or H0 > 140])
	print("massCutH0s:", [H0 for H0 in massCutH0s if H0 < -10 or H0 > 140])
	print("allZeros: " + str([zero for zero in allZeros if zero <
								 -2.5 or zero > 3.0]))
	print("inClusterZeros: " + str([zero for zero in inClusterZeros if zero <
								 -2.5 or zero > 3.0]))
	print("outClusterZeros: " + str([zero for zero in outClusterZeros if zero <
								 -2.5 or zero > 3.0]))
	print("massCutZeros: " + str([zero for zero in massCutZeros if zero <
								 -2.5 or zero > 3.0]))

	plt.tight_layout(rect=[0.065, 0.115, 1.0, 1.0])
#	plt.tight_layout()
	fig.set_size_inches(5.9, 2.6)

#	plt.xlabel("Hubble flow zero point (Mpc from LG centre)")
#	plt.ylabel("Combined mass of Milky Way and Andromeda (Solar masses)")
#	plt.xlim(xmin, xmax)
	plt.savefig(outputdir+"clusteredHFparameters.pdf")


	##### velocity dispersion #####
	plt.cla()
	plt.clf()
	
	fig, ax = plt.subplots()
	bp = ax.boxplot([massCutDispersions, inClusterDispersions, outClusterDispersions,
				   allDispersions], vert=False)
	blackBoxplot(bp)
	ax.set_xlabel("Velocity dispersion around\nthe Hubble flow (km/s)")
	ax.set_yticklabels(["Haloes in clusters with all\nmembers less massive\n"
					r"than $8 \times 10^{11}~M_{\astrosun}$", "Haloes in clusters",
					"Haloes outside clusters", "All haloes"], ha='right',
					multialignment='right')
	ax.set_xticks(range(0, 201, 10), minor=True)
	plt.tight_layout(rect=[0.15, 0.1, 1.0, 1.0])
	fig.set_size_inches(4.0, 2.6)
	plt.savefig(outputdir + "clusteredHFdispersions.pdf")

	### ks-test ###
	plt.cla()
	plt.clf()

	print("")
	print("all vs out H0")
	print(stats.ks_2samp(allH0s, outClusterH0s))
	print("all vs in H0")
	print(stats.ks_2samp(allH0s, inClusterH0s))
	print("all vs mass cut H0")
	print(stats.ks_2samp(allH0s, massCutH0s))
	print("in vs mass cut H0")
	print(stats.ks_2samp(inClusterH0s, massCutH0s))
	print("")
	print("all vs out zeros")
	print(stats.ks_2samp(allZeros, outClusterZeros))
	print("all vs in zeros")
	print(stats.ks_2samp(allZeros, inClusterZeros))
	print("all vs mass cut zeros")
	print(stats.ks_2samp(allZeros, massCutZeros))
	print("")
	print("all vs out dispersion")
	print(stats.ks_2samp(allDispersions, outClusterDispersions))
	print("out vs in dispersion")
	print(stats.ks_2samp(outClusterDispersions, inClusterDispersions))
	print("in vs mass cut dispersion")
	print(stats.ks_2samp(inClusterDispersions, massCutDispersions))


#	(DH0, pvalH0) = stats.ks_2samp(allH0s, inClusterH0s)
#	(Dzero, pvalZero) = stats.ks_2samp(allZeros, inClusterZeros)
#	(Ddispersion, pvalDispersion) = stats.ks_2samp(allDispersions, inClusterDispersions)
#	print("H0")
#	print("D:\t"+str(DH0))
#	print("p:\t"+str(pvalH0))
#	print("Zero")
#	print("D:\t"+str(Dzero))
#	print("p:\t"+str(pvalZero))
#	print("Dispersion")
#	print("D:\t"+str(Ddispersion))
#	print("p:\t"+str(pvalDispersion))
#	print("medians")
#	print(np.median(allDispersions))
#	print(np.median(inClusterH0s))
#	print("means")
#	print(np.mean(allDispersions))
#	print(np.mean(inClusterDispersions))
#	print("t-test")
#	print(stats.ttest_ind(allDispersions, inClusterDispersions, equal_var=False))
#	print(stats.ttest_ind(allDispersions, inClusterDispersions,
#					   equal_var=True))
	
#	allH0s.sort()
#	inClusterH0s.sort()
#	allZeros.sort()
#	inClusterZeros.sort()
#	allDispersions.sort()
#	inClusterDispersions.sort()
#
##	print(len(allH0s), len(inClusterZeros), len(allZeros), len(inClusterZeros),
##	   len(allDispersions), len(inClusterDispersions))
#	n_datapoints = len(allH0s)
#
#	percentage = np.arange(0, 100, 100.0/len(allH0s))
#
#	(H0x, H0lowY, H0highY) = biggestDifference(allH0s, inClusterH0s, percentage)
#	(zeroX, zeroLowY, zeroHighY) = biggestDifference(allZeros, inClusterZeros,
#												  percentage)
#	(dispersionX, dispersionLowY, dispersionHighY) = biggestDifference(
#		allDispersions, inClusterDispersions, percentage)
#
#	percentage = np.append(percentage, 100)
#	percentage = np.repeat(percentage, 2)
#	percentage = np.delete(percentage, 0)
#	
#	maxH0 = max(allH0s[n_datapoints - 1], inClusterH0s[n_datapoints -1])
#	maxZero = max(allZeros[n_datapoints - 1], inClusterZeros[n_datapoints - 1])
#	maxDispersion = max(allDispersions[n_datapoints - 1],
#					 inClusterDispersions[n_datapoints - 1])
#	allH0s = np.append(allH0s, maxH0)
#	inClusterH0s = np.append(inClusterH0s, maxH0)
#	allZeros = np.append(allZeros, maxZero)
#	inClusterZeros = np.append(inClusterZeros, maxZero)
#	allDispersions = np.append(allDispersions, maxDispersion)
#	inClusterDispersions = np.append(inClusterDispersions, maxDispersion)
#
#	allH0s = CDFifyXData(allH0s)
#	inClusterH0s = CDFifyXData(inClusterH0s)
#	allZeros = CDFifyXData(allZeros)
#	inClusterZeros = CDFifyXData(inClusterZeros)
#	allDispersions = CDFifyXData(allDispersions)
#	inClusterDispersions = CDFifyXData(inClusterDispersions)
#
#
#	fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
#	ksplot(ax1, allH0s, inClusterH0s, percentage, "All haloes", "Haloes in "
#		"clusters", H0x, H0lowY, H0highY)
#	ax1.set_xlabel(r"$H_0$" +  " (km/s/Mpc)")
#	ax1.set_ylabel("\%")
#	ksplot(ax2, allZeros, inClusterZeros, percentage, "All haloes",
#		"Haloes in clusters", zeroX, zeroLowY, zeroHighY)
#	ax2.set_xlabel("Zero point distance (Mpc)")
#	ax2.set_ylabel("\%")
#	ksplot(ax3, allDispersions, inClusterDispersions, percentage, "All haloes",
#		"Haloes in clusters", dispersionX, dispersionLowY, dispersionHighY)
#	ax3.set_xlabel("Velocity dispersion (km/s)")
#	ax1.set_ylabel("\%")
#	
#	plt.tight_layout()
#	fig.set_size_inches(4.0, 6.0)
#
#	plt.savefig(outputdir + "ksresults.pdf")
