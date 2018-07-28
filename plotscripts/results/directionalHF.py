# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import clustering
import clusterAnalysis
import filereader
import LGfinder
import physUtils
from sklearn.utils import resample
from sibeliusConstants import *
import transitiondistance
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import sys

from matplotlib.transforms import Affine2D
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import angle_helper
from mpl_toolkits.axisartist.grid_finder import MaxNLocator, DictFormatter
from mpl_toolkits.axisartist.floating_axes import GridHelperCurveLinear, FloatingSubplot

# https://stackoverflow.com/a/32235971
def fractional_polar_axes(f, thlim=(0, 180), rlim=(0, 1), step=(30, 0.25),
						  thlabel='theta', rlabel='r', ticklabels=True, rlabels = None, subplot=111):
	'''Return polar axes that adhere to desired theta (in deg) and r limits. steps for theta
	and r are really just hints for the locators.'''
	th0, th1 = thlim # deg
	r0, r1 = rlim
	thstep, rstep = step

	# scale degrees to radians:
	tr_scale = Affine2D().scale(np.pi/180., 1.)
	pa = PolarAxes
	tr = tr_scale + pa.PolarTransform()
	theta_grid_locator = angle_helper.LocatorDMS((th1-th0)//thstep)
	r_grid_locator = MaxNLocator((r1-r0)//rstep)
	theta_tick_formatter = angle_helper.FormatterDMS()
	if rlabels:
		rlabels = DictFormatter(rlabels)

	grid_helper = GridHelperCurveLinear(tr,
									 extremes=(th0, th1, r0, r1),
									 grid_locator1=theta_grid_locator,
									 grid_locator2=r_grid_locator,
									 tick_formatter1=theta_tick_formatter,
									 tick_formatter2=rlabels)

	a = FloatingSubplot(f, subplot, grid_helper=grid_helper)
	f.add_subplot(a)
	
	# adjust x axis (theta):
	a.axis["bottom"].set_visible(False)
	a.axis["top"].set_axis_direction("bottom") # tick direction
	a.axis["top"].toggle(ticklabels=ticklabels, label=bool(thlabel))
	a.axis["top"].major_ticklabels.set_axis_direction("top")
	a.axis["top"].label.set_axis_direction("top")
	a.axis["top"].major_ticklabels.set_pad(10)

	# adjust y axis (r):
	a.axis["left"].set_axis_direction("bottom") # tick direction
	a.axis["right"].set_axis_direction("top") # tick direction
	a.axis["left"].toggle(ticklabels=True, label=bool(rlabel))
	
	# add labels:
	a.axis["top"].label.set_text(thlabel)
	a.axis["left"].label.set_text(rlabel)
	
	# create a parasite axes whose transData is theta, r:
	auxa = a.get_aux_axes(tr)
	
	# make aux_ax to have a clip path as in a?:
	auxa.patch = a.patch 
	# this has a side effect that the patch is drawn twice, and possibly over some other
	# artists. So, we decrease the zorder a bit to prevent this:
	a.patch.zorder = -2

	# add sector lines for both dimensions:
	thticks = grid_helper.grid_info['lon_info'][0]
	rticks = grid_helper.grid_info['lat_info'][0]
	for th in thticks[1:-1]: # all but the first and last
		auxa.plot([th, th], [r0, r1], ':', c='grey', zorder=-1, lw=0.5)
		for ri, r in enumerate(rticks):
			# plot first r line as axes border in solid black only if it  isn't at r=0
			if ri == 0 and r != 0:
				ls, lw, color = 'solid', 1, 'black'
			else:
				ls, lw, color = 'dashed', 0.5, 'grey'
				# From http://stackoverflow.com/a/19828753/2020363
				auxa.add_artist(plt.Circle([0, 0], radius=r, ls=ls, lw=lw, color=color, fill=False,
							   transform=auxa.transData._b, zorder=-1))

	return auxa


if __name__ == "__main__":
	inputfile = "../input/lgfound-fullpath.txt" 
#	inputfile = "../input/hundred.txt"
	outputdir = "../../kuvat/"

	mindist = 0.0
	maxdist = 5.0
	eps = 1.8
	ms = 10

	onAxisH0 = []
	toM31H0 = []
	awayFromM31H0 = []
	offAxisH0 = []
	plotBinLimits = np.linspace(0.0, np.pi, 10)
	angles = []
	zeros = []
	h0s = []
	for i in range(len(plotBinLimits)-1):
		angles.append((plotBinLimits[i]+plotBinLimits[i+1])/2.0)
		h0s.append([])
		zeros.append([])

	excludedBins = 0
	excludedSims = 0

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
			MWindex = LG[1]
			andromedaIndex = LG[0]
		else:
			MWindex = LG[0]
			andromedaIndex = LG[1]

		MWcop = cop[MWindex]
#		massCentre = physUtils.massCentre(cop[MWindex], cop[andromedaIndex],
#									mass[MWindex], mass[andromedaIndex])
		closestContDist = physUtils.findClosestDistance(MWcop,
												  contaminatedPositions)
		if closestContDist < maxdist:
			excludedSims += 1
			print("excluding " + simdir)
			continue

		LGvector = cop[andromedaIndex] - cop[MWindex]

		centreVel = staticVel[MWindex]
		distances = np.array([physUtils.distance(MWcop, c) for c in
						cop])
		vel = physUtils.addExpansion(staticVel, cop, MWcop)

		LGrelVel = vel[LG[0]] - vel[LG[1]]
		LGrelVelComponents = physUtils.velocityComponents(LGrelVel, cop[LG[1]]-cop[LG[0]])
		LGdistance = physUtils.distance(cop[LG[0]], cop[LG[1]])

		mask = np.array([d < maxdist and d >= mindist for d in
				   distances])
		cop = cop[mask]
		vel = vel[mask]
		distances = distances[mask]

		radvel = np.array([physUtils.velocityComponents(vel[j] - centreVel,
												  cop[j] - MWcop)[0]
					 for j in range(len(vel))])

		directions = np.array([physUtils.angleBetween(LGvector, pos-MWcop) for pos in
						 cop])


		##### extracting interesting data starts #####
		for angleIndex in range(len(plotBinLimits)-1):
			angleMask = np.array([direction < plotBinLimits[angleIndex+1] and
						 direction >= plotBinLimits[angleIndex] for direction in
						 directions])
			#			print("[" + str(plotBinLimits[angleIndex]) + ", " +
			#		 str(plotBinLimits[angleIndex+1]) + "]:\t" + str(np.sum(angleMask)) )

			if np.sum(angleMask) < 15:
				zeros[angleIndex].append(np.nan)
				h0s[angleIndex].append(np.nan)
				excludedBins += 1
			else:
				(fit, flowstartdist) = transitiondistance.findBestHubbleflow(distances[angleMask],
										  radvel[angleMask])
				H0 = fit[0]
				zero = -fit[1]/fit[0] 
				zeros[angleIndex].append(zero)
				h0s[angleIndex].append(H0)

		# on and off-axis measurements
#		onAxisMask = np.array([angle < np.pi/4.0 or angle > np.pi*3.0/4.0 for
#						 angle in directions])
		toM31Mask = np.array([angle < np.pi/4.0 for angle in directions])
		awayFromM31Mask = np.array([angle > np.pi*3.0/4.0 for angle in
							 directions])
		onAxisMask = np.logical_or(toM31Mask, awayFromM31Mask)
		offAxisMask = np.logical_not(onAxisMask)

		offAxisH0.append(transitiondistance.findBestHubbleflow(distances[offAxisMask],
							 radvel[offAxisMask])[0][0])
		toM31H0.append(transitiondistance.findBestHubbleflow(distances[toM31Mask],
													   radvel[toM31Mask])[0][0])
		awayFromM31H0.append(transitiondistance.findBestHubbleflow(distances[awayFromM31Mask],
													   radvel[awayFromM31Mask])[0][0])
		onAxisH0.append(transitiondistance.findBestHubbleflow(distances[onAxisMask],
														radvel[onAxisMask])[0][0])

	print("Total sims excluded: "+str(excludedSims))
	print("Total bins excluded: "+str(excludedBins))


	##### plotting #####

	angles = np.array(angles)*180.0/np.pi
	h0s = np.array(h0s)
	zeros = np.array(zeros)
	onAxisH0 = np.array(onAxisH0)
	offAxisH0 = np.array(offAxisH0)

	meanH0s = np.nanmean(h0s, axis=1)
	meanZeros = np.nanmean(zeros, axis=1)
	H0std = np.nanstd(h0s, axis=1, ddof=1)
	zerostd = np.nanstd(zeros, axis=1)


	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)
	params = {'text.latex.preamble' : [r'\usepackage{wasysym}']}
	plt.rcParams.update(params)


	### semicircle plots ###
	fig = plt.figure()

	H0locations = [0, 20, 40, 60, 80, 100, 120]
	H0labels = ['0', '20', '40', '60', '80', '100', '120']
	H0ticks = {loc : label for loc, label in zip(H0locations, H0labels)}
	ax1 = fractional_polar_axes(fig, thlim=(0., 180.), rlim=(0, 125),
							 step=(10.0, 20), rlabels=H0ticks, subplot=211,
							 thlabel=r'$\phi$',	rlabel=r'$H_{0}$ (km/s/Mpc)')
	ax1.errorbar(angles, meanH0s, yerr=H0std, fmt='o', ecolor='k', capsize=0,
			  color='k')


	zeroLocations = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
	zeroLabels = ['-6', '-4', '-2', '0','2', '4', '6']
	zeroTicks = {loc : label for loc, label in zip(zeroLocations, zeroLabels)}
	ax2 = fractional_polar_axes(fig, thlim=(0., 180.), rlim=(0.0, 10.0),
							 step=(10.0, 2.0), rlabels=zeroTicks, subplot=212,
							 thlabel=r'$\phi$',
							 rlabel=r'Hubble flow zero point distance (Mpc)')
	ax2.errorbar(angles, meanZeros+6.0, xerr=0, yerr=zerostd, fmt='o',
			  ecolor='k', color='k', capsize=0)

	plt.tight_layout()
	fig.set_size_inches(5.3, 6.5)

	plt.savefig(outputdir+"directionalHF.pdf")

	### off vs on axis ratio ###
	plt.cla()
	plt.clf()

	fig = plt.figure()
	ax = fig.add_subplot(111)

	ratio = onAxisH0/offAxisH0
	print("median: " + str(np.median(ratio)))
#	print(np.count_nonzero(ratio>1.0))
#	print(np.count_nonzero(ratio<1.0))

	weights = np.ones_like(ratio)/float(len(ratio))*100
	ax.hist(ratio, bins=np.arange(0.5, 3.5, 0.25), color='0.75', edgecolor='k')
	ax.set_xlabel((r"$\frac{\displaystyle H_{0,\ \mathrm{on\ axis}}}"
			   "{\displaystyle H_{0,\ \mathrm{off\ axis}}}$"))
	ax.set_xticks(np.arange(0.5, 3.5, 0.5), minor=False)
	ax.set_ylabel("Simulations (\%)")

	fig.set_size_inches(3, 2.8)
	plt.tight_layout()
	plt.savefig(outputdir+"directionalHF-ratios.pdf")


	## bootstrapping ##
	bootstrapRatios = np.zeros(5000)
	for i in range(len(bootstrapRatios)):
		bootstrapRatios[i] = np.median(resample(ratio))
	print("5th percentile of bootstrapped median: " +
		str(np.percentile(bootstrapRatios, 5.0)))
	print("95th percentile of bootstrapped median: " +
	   str(np.percentile(bootstrapRatios, 95.0)))


	### three direction histogram ###
	plt.cla()
	plt.clf()

	fig = plt.figure()

	alpha = 0.7
	bins = np.arange(10, 130, 10)
	plt.hist(toM31H0, histtype="step", bins=bins, label="Towards M31", alpha=alpha)
	plt.hist(offAxisH0, histtype="step", bins=bins, label="Off axis", alpha=alpha)
	plt.hist(awayFromM31H0, histtype="step", bins=bins, label="Away from M31", alpha=alpha)

	plt.legend(loc=2)
	plt.xlabel("$H_0$")
	plt.ylabel("Simulations")
	fig.set_size_inches(3.4, 2.6)
	plt.tight_layout(rect=[0, -0.05, 1, 1])
	plt.savefig(outputdir + "threeDirectionH0.pdf")
