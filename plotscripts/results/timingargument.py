# -*- coding:utf-8 -*-

from __future__ import print_function

import logging
from math import sin, cos, pi
from scipy.optimize import fsolve#, minimize
import sys

"""
Estimates the mass of a galaxy pair using timing argument.

Parameters
----------
radvel : float
	relative radial velocity of the galaxy pair in (e.g. in kpc/Gyr)
distance : float
	distance between the galaxies (e.g. in kpc)
T : float
	age of the universe (e.g. in Gyr)
G : float
	gravitational constant (e.g. in kpc³/Gyr²/M☉)

Returns
-------
float
estimate for the combined mass of the galaxy pair in solar masses

Raises
------
PositiveTimingVelocityException
	when a positive velocity is given as radvel
UnexpectedSolutionNumberException
	when root finding algorithm finds more than one root for eccentric anomaly

"""
def timingArgumentMass(radvel, distance, T, G):
	if radvel > 0:
		print(('Radial velocity must be negative to estimate galaxy masses '
		 'using timing argument'))
		raise PositiveTimingVelocityException(str)

#	G =	4.498768e-6 # kpc^3/Gyr^2/Msun

	phi = fsolve(phiFunction, 4.11, args=(radvel, distance, T))
	if len(phi) != 1:
		print("Unexpected length (" + str(len(phi)) + ") for solution " +
		"vector of phi")
		raise UnexpectedSolutionNumberException()
	phi = (phi % (2*pi))[0]

	a = distance / (1-cos(phi))

	M = a**3/G*(phi-sin(phi))**2/(T**2)
	return M


def phiFunction(phi, radvel, distance, T):
	return (distance*1.0/T*sin(phi)*(phi-sin(phi))/((1-cos(phi))**2)-radvel)


class PositiveTimingVelocityException(Exception):
	def __init__(self, value):
		self.value = value
		def __str__(self):
			return repr(self.value)

class UnexpectedSolutionNumberException(Exception):
	def __init__(self, value):
		self.value = value
		def __str__(self):
			return repr(self.value)


if __name__ == "__main__":
	print("{:.2e}".format(timingArgumentMass(-119.0, 730.0, 20.0, 4.498768e-6)))
