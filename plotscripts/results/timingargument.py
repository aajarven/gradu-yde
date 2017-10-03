# -*- coding:utf-8 -*-

from __future__ import print_function

import logging
from math import sin, cos, pi
from scipy.optimize import fsolve#, minimize
import sys

def phiFunction(phi, radvel, distance, T):
#	return ((sin(phi)**2 - phi * sin(phi)) / ((1 + cos(phi))**2) -
#		 radvel*1.0/distance*T)
	return (distance*1.0/T*sin(phi)*(phi-sin(phi))/((1-cos(phi))**2)-radvel)

def timingArgumentMass(radvel, distance, T):
	G =	4.498768e-6 # kpc^3/Gyr^2/Msun
	
#	phi = minimize(phiFunction, 4.0, args=(radvel, distance, T)).x
	phi = fsolve(phiFunction, 4.11, args=(radvel, distance, T))
	if len(phi) != 1:
		logging.exception("Unexpected length (" + str(len(phi)) + ") for " + 
					"solution vector of phi")
		sys.exit(1)
	phi = (phi % (2*pi))[0]

	a = distance / (1-cos(phi))
#	print("a="+str(a))

	M = a**3/G*(phi-sin(phi))**2/(T**2)
#	print("{:.2e}".format(M))
	return M

if __name__ == "__main__":
	print("{:.2e}".format(timingArgumentMass(-119.0, 730.0, 20.0)))
