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

	infile = args[0]
	outfile = args[1]

	try:
		f = open(infile, 'r')
	except IOError:
		print("File {} cannot be opened".format(infile))
	
	lines = []
	line = f.readline()
	while len(line) > 0:
		if line not in lines:
			lines.append(line)
		line = f.readline()
	
	f.close()
	
	f = open(outfile, 'w')
	for line in lines:
		f.write(line)
	f.close()

