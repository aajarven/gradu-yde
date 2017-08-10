# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
import sys
sys.path.insert(0, '/scratch/aajarven/plotscripts/')

import clustering
import clusterAnalysis
import filereader
import LGfinder
from optparse import OptionParser
import physUtils
from sibeliusConstants import *
from transitiondistance import simpleFit
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
	#pa = axes(polar="true") # Create a polar axis
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
	print(a)
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
	print(auxa)
	# make aux_ax to have a clip path as in a?:
	auxa.patch = a.patch 
	# this has a side effect that the patch is drawn twice, and possibly over some other
	# artists. So, we decrease the zorder a bit to prevent this:
	a.patch.zorder = -2

	# add sector lines for both dimensions:
	thticks = grid_helper.grid_info['lon_info'][0]
	rticks = grid_helper.grid_info['lat_info'][0]
	print(grid_helper.grid_info['lat_info'])
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



#import mpl_toolkits.axisartist.floating_axes as floating_axes
#from matplotlib.projections import PolarAxes
#from mpl_toolkits.axisartist import angle_helper
#from mpl_toolkits.axisartist.grid_finder import MaxNLocator, DictFormatter
#
## https://stackoverflow.com/a/44190325 + https://stackoverflow.com/a/32235971
#def setup_axes(fig, rect, maxR):
	#	tr = PolarAxes.PolarTransform() 
	#
	#	theta0, theta1 = 0, np.pi
	#	r0, r1 = 0, maxR
	#
	#	theta_grid_locator = angle_helper.LocatorDMS(np.pi/4)
	#	r_grid_locator = MaxNLocator(5)
	#
	#	grid_helper = floating_axes.GridHelperCurveLinear(
	#		tr, extremes=(theta0, theta1, r0, r1),
	#		grid_locator1=theta_grid_locator, grid_locator2=r_grid_locator)
	#
	#	ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
	#	fig.add_subplot(ax1)
	#
	#	# adjust axis
	#	ax1.axis["left"].set_axis_direction("bottom")
	#	ax1.axis["right"].set_axis_direction("top")
	#	ax1.axis["bottom"].set_visible(False)
	#	ax1.axis["top"].set_axis_direction("bottom")
	#	ax1.axis["top"].toggle(ticklabels=True, label=True)
	#	ax1.axis["top"].major_ticklabels.set_axis_direction("top")
	#	ax1.axis["top"].label.set_axis_direction("top")
	#	#    ax1.axis["left"].label.set_text(r"z")
	#	#    ax1.axis["top"].label.set_text(r"RA")
	#
	#	# create a parasite axes whose transData in RA, cz
	#	aux_ax = ax1.get_aux_axes(tr)
	#
	#	aux_ax.patch = ax1.patch  
	#	ax1.patch.zorder = 0.9  
	#
	#	return ax1, aux_ax


if __name__ == "__main__":
	#	inputfile = "../input/allButDuplicates-fullpath.txt" 
	inputfile = "../input/hundred.txt"
	outputdir = "../../kuvat/"

	mindist = 1.0
	maxdist = 5.0
	eps = 1.8
	ms = 10

	binlimits = np.linspace(0.0, np.pi, 19)
	angles = []
	zeros = []
	h0s = []
	for i in range(len(binlimits)-1):
		angles.append((binlimits[i]+binlimits[i+1])/2.0)
		h0s.append([])
		zeros.append([])

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
		closestContDist = physUtils.findClosestDistance(MWcop,
												  contaminatedPositions)
		if closestContDist < maxdist:
			continue

		LGvector = cop[andromedaIndex] - cop[MWindex]

		centreVel = staticVel[MWindex]
		distances = np.array([physUtils.distance(MWcop, c) for c in
						cop])
		vel = physUtils.addExpansion(staticVel, cop, MWcop)

		LGrelVel = vel[LG[0]] - vel[LG[1]]
		LGrelVelComponents = physUtils.velocityComponents(LGrelVel, cop[LG[1]]-cop[LG[0]])
		LGdistance = physUtils.distance(cop[LG[0]], cop[LG[1]])

		mask = np.array([d < maxdist and d > mindist for d in
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
		for angleIndex in range(len(binlimits)-1):
			angleMask = np.array([direction < binlimits[angleIndex+1] and
						 direction >= binlimits[angleIndex] for direction in
						 directions])
			#			print("[" + str(binlimits[angleIndex]) + ", " +
			#		 str(binlimits[angleIndex+1]) + "]:\t" + str(np.sum(angleMask)) )

			if np.sum(angleMask) < 10:
				zeros[angleIndex].append(np.nan)
				h0s[angleIndex].append(np.nan)
			else:
				(H0, zero) = simpleFit(distances[angleMask], radvel[angleMask])
				zeros[angleIndex].append(zero)
				h0s[angleIndex].append(H0)

	angles = np.array(angles)


	##### plotting #####

	h0s = np.array(h0s)
	zeros = np.array(zeros)


	rc('font', **{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)
	params = {'text.latex.preamble' : [r'\usepackage{wasysym}']}
	plt.rcParams.update(params)


	fig = plt.figure()

#	r_locs = [0, .25, .5, .75, 1]
#	r_labels = ['5', '10', '15', '20', '25']
#	r_ticks = {loc : label for loc, label in zip(r_locs, r_labels)}
#	a1 = fractional_polar_axes(fig, thlim=(-90, 90),step=(10, 0.2),
#							theta_offset=90, rlabels = r_ticks)

	H0locations = [0, 20, 40, 60, 80, 100]
	H0labels = ['0', '20', '40', '60', '80', '100']
	H0ticks = {loc : label for loc, label in zip(H0locations, H0labels)}
	ax1 = fractional_polar_axes(fig, thlim=(0., 180.), rlim=(0, 100), step=(10., 20),
							rlabels=H0ticks, subplot=211, thlabel=r'$\phi$',
							rlabel=r'$H_{0}$ (km/s/Mpc)')
	ax1.plot(angles*180.0/np.pi, np.nanmean(h0s, axis=1))

	print(np.nanmean(zeros, axis=1))

	zeroLocations = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
	zeroLabels = ['-2', '-1', '0', '1', '2', '3']
	zeroTicks = {loc : label for loc, label in zip(zeroLocations, zeroLabels)}
	ax2 = fractional_polar_axes(fig, thlim=(0., 180.), rlim=(0.0, 5.0),
							 step=(10, 1.0), rlabels=zeroTicks, subplot=212,
							 thlabel=r'$\phi$',
							 rlabel=r'Hubble flow zero point distance (Mpc)')
	ax2.plot(angles*180.0/np.pi, np.nanmean(zeros, axis=1)+2.0)

#	fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))

#	fig = plt.figure()
#	ax1, aux_ax1 = setup_axes(fig, 211, 90)
#	ax1.axis["left"].label.set_text(r"$H_0$ (km/s/Mpc)")
#	ax1.axis["top"].label.set_text(r"$\phi$")
#
#	aux_ax1.plot(angles, np.nanmean(h0s, axis=1))
#
#	ax2, aux_ax2 = setup_axes(fig, 212, 2.5)
#	aux_ax2.plot(angles, np.nanmean(zeros, axis=1))
#	ax2.axis["left"].label.set_text(r"$H_0$ (km/s/Mpc)")
#	ax2.axis["top"].label.set_text(r"$\phi$")

#	print(h0s)
#	print(np.nanmean(h0s, axis=1))
#	ax1.plot(angles, np.nanmean(h0s, axis=1))
#	ax2.plot(angles, np.nanmean(zeros, axis=1))
#	ax2.set_xlim([0, np.pi])

	plt.tight_layout()
	fig.set_size_inches(5.3, 6.5)

#	plt.xlabel("Hubble flow zero point (Mpc from LG centre)")
#	plt.ylabel("Combined mass of Milky Way and Andromeda (Solar masses)")
#	plt.xlim(xmin, xmax)
plt.savefig(outputdir+"directionalHF.svg")

