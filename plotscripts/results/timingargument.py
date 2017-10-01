# -*- coding:utf-8 -*-

from __future__ import print_function

import logging
from math import sin, cos, pi
from scipy.optimize import fsolve
import sys

def phiFunction(phi, radvel, distance, T):
	return ((sin(phi)**2 - phi * sin(phi)) / ((1 + cos(phi))**2) -
		 radvel*1.0/distance*T)

def timingArgumentMass(radvel, distance, T):
	G =	4.498768e-6 # kpc^3/Gyr^2/Msun

	phi2 = fsolve(phiFunction, 0, args=(radvel, distance, T))
	if len(phi2) != 1:
		logging.exception("Unexpected length (" + str(len(phi)) + ") for " + 
					"solution vector of phi2")
		sys.exit(1)
	phi2 = (phi2 % (2*pi))[0]
	E2 = (phi2 + pi) % (2*pi)	# eccentric anomaly
	print(E2)

	Omega = (E2 - sin(E2) - pi) / T
	print(Omega)

	a = distance / (1-cos(E2))
	print(a)

	M = Omega**2*a**3/G
	print("{:.2e}".format(M))

if __name__ == "__main__":
	timingArgumentMass(-125.0, 650.0, 14.0)
