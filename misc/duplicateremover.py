# -*- coding: utf-8 -*-

import sys
from optparse import OptionParser

"""
Removes duplicate rows
"""

if __name__ == "__main__":
	parser = OptionParser(usage="usage: python duplicateremover.py input output")
	(opts, args) = parser.parse_args()

	if len(args) < 2:
		print("Give input and output files as arguments")
		sys.exit()

	f = 
