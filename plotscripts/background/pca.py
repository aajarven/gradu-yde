# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import math
import pylab
import random
from sklearn.decomposition import PCA


def aesthetics(ax, xmin, xmax, ymin, ymax, xlabel, ylabel, title,
			   xLabelOffset=0.0, yLabelOffset=0.0):
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel, rotation=0)
	ax.set_title(title)
	ax.title.set_position([.5, 1.05])
	ax.spines['left'].set_position('zero')
	ax.spines['right'].set_color('none')
	ax.spines['bottom'].set_position('zero')
	ax.spines['top'].set_color('none')
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')
	xticks = np.arange(xmin, xmax+0.00001, 0.5)
	yticks = np.arange(ymin, ymax+0.00001, 0.5)
	ax.set_xticks(xticks[xticks!=0])
	ax.set_yticks(yticks[yticks!=0])
	ax.set_xlim([xmin, xmax])
	ax.set_ylim([ymin, ymax])
	ticklab = ax.xaxis.get_ticklabels()[0]
	trans = ticklab.get_transform()
	ax.xaxis.set_label_coords(xmax, 0.4+xLabelOffset, transform=trans)
	ticklab = ax.yaxis.get_ticklabels()[0]
	trans = ticklab.get_transform()
	ax.yaxis.set_label_coords(0.25, ymax+yLabelOffset, transform=trans)


def rotate(refX, refY, x, y):
	angle = -math.atan2(refY, refX)
	
	xPrime = np.zeros_like(x)
	yPrime = np.zeros_like(y)

	for i in range(len(x)):
		xPrime[i] = x[i]*math.cos(angle) - y[i]*math.sin(angle)
		yPrime[i] = x[i]*math.sin(angle) + y[i]*math.cos(angle)

	return (xPrime, yPrime)


np.random.seed(717717)

N=30

x = np.arange(0, 2, 2.0/(N))
y = np.arange(0, 1.5, 1.5/(N))+0.25
xDeviations = np.random.normal(0.0, 0.3, N)
yDeviations = np.random.normal(0.0, 0.3, N)

x = x + xDeviations
y = y + yDeviations

f, (ax1, ax2, ax3) = plt.subplots(3, 1)

ax1.scatter(x, y, marker='.', edgecolor='k', facecolor='k')

xCenter = np.mean(x)
yCenter = np.mean(y)

ax1.scatter(xCenter, yCenter, marker='x', color='r', s=60)
aesthetics(ax1, -1.0, 2.5, -0.5, 2.5, '$x$', '$y$', 'Original data',
		   yLabelOffset=-0.1)
x = x - xCenter
y = y - yCenter

ax2.scatter(x, y, marker='.', edgecolor='k', facecolor='k')
aesthetics(ax2, -2.0, 1.5, -1.5, 1.5, '$x$', '$y$', 'Centered data',
		   yLabelOffset=-0.1)

xy=np.array([x,y]).T
pca=PCA(n_components=1)
xy_pca=pca.fit_transform(xy)
xy_n=pca.inverse_transform(xy_pca)

components = pca.components_

lineStart = components[0] * -2.0 / components[0, 0]
lineEnd = components[0] * 1.5 / components[0, 0]
ax2.plot([lineStart[0], lineEnd[0]], [lineStart[1], lineEnd[1]], color='k')

(rotatedX, rotatedY) = rotate(xy_n[0, 1], xy_n[0, 1], x, y)
(transformedX, transformedY) = rotate(components[0, 0], components[0, 1], x,
									  y)
ax3.scatter(transformedX, transformedY, marker='.', edgecolor='k', facecolor='k')
aesthetics(ax3, -2.0, 1.5, -1.5, 1.5, r'$PC_{1}$', '~$PC_{2}$',
		   'Principal components', xLabelOffset=-0.15, yLabelOffset=-0.1)

pylab.gcf().set_size_inches(4.0, 8.0)
plt.tight_layout()
rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.savefig('../../kuvat/pca-illustrated.svg')

