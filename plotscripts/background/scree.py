# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import math
import pylab
import random

np.random.seed(715517)

A = np.random.randn(9, 14)
A = np.asmatrix(A) * np.asmatrix(A.T)
U, S, V = np.linalg.svd(A)
eigvals = S**2 / np.cumsum(S)[-1]
eigvals2 = S**2 / np.sum(S)
assert (eigvals == eigvals2).all()

rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

fig = plt.figure()
sing_vals = np.arange(len(eigvals)) + 1
plt.plot(sing_vals, eigvals, 'ko-', linewidth=2)
plt.xlabel('Principal Component')
plt.ylabel('Variance')

plt.xticks(np.arange(1,10,1))

F = pylab.gcf()
F.set_size_inches(3.5, 2.5)

plt.tight_layout()
plt.savefig('../../kuvat/scree.svg')
