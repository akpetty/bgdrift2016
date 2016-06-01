############################################################## 
# Date: 01/01/16
# Name: plot_season_BG_conc_BTNT.py
# Author: Alek Petty
# Description: Script to plot ice conc in the BG region from NASATEAM/BOOTSTRAP
# Input requirements: Monthly ice conc data from both renalyases
#                     Also needs the functions in BG_functions
# Output: Lineplot of seasonal BG ice conc

import matplotlib
matplotlib.use("AGG")
import sys
sys.path.append('/Users/aapetty/GPYTHON/')
import alek_objects as apy
# basemap import
from mpl_toolkits.basemap import Basemap, shiftgrid
# Numpy import
import numpy as np
from pylab import *
from scipy.io import netcdf
import numpy.ma as ma
import string
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid.inset_locator import mark_inset
from mpl_toolkits.axes_grid.anchored_artists import AnchoredSizeBar
from scipy import stats
from matplotlib import rc
from netCDF4 import Dataset
from glob import glob

mpl.rc("ytick",labelsize=10)
mpl.rc("xtick",labelsize=10)
rcParams['font.size']=10
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

m = Basemap(projection='npstere',boundinglat=66,lon_0=0, resolution='l'  )

figpath='./Figures/'
datapath='./Data_output/'

team1='NASA_TEAM'
team2='BOOTSTRAP'
start_year=1980
end_year=2013
num_years = end_year-start_year+1
years_str=str(start_year)+'-'+str(end_year)

ice_concNT = load(datapath+'/ice_conc_months-'+years_str+team1+'.txt')
ice_concBT = load(datapath+'/ice_conc_months-'+years_str+team2+'.txt')

lats = load(datapath+'/ice_conc_lats'+team1+'.txt')
lons = load(datapath+'/ice_conc_lons'+team1+'.txt')
xpts, ypts = m(lons, lats)


num_months = ice_concNT.shape[1]
#beau_lonlat = [-170., -130., 82., 72.]
beau_lonlat = [-175., -125., 85., 70.]

ice_concNT_BG=ma.masked_all((ice_concNT.shape))
ice_concBT_BG=ma.masked_all((ice_concNT.shape))
ice_concNT_BG_season_means = ma.masked_all((ice_concNT.shape[0], 4))
ice_concBT_BG_season_means = ma.masked_all((ice_concBT.shape[0], 4))

for i in xrange(num_years):
	for j in xrange(num_months):
		ice_concNT_BG[i, j] = ma.masked_where((lons<beau_lonlat[0]) | (lons>beau_lonlat[1]) | (lats>beau_lonlat[2])| (lats<beau_lonlat[3]), ice_concNT[i, j])
		ice_concBT_BG[i, j] = ma.masked_where((lons<beau_lonlat[0]) | (lons>beau_lonlat[1]) | (lats>beau_lonlat[2])| (lats<beau_lonlat[3]), ice_concBT[i, j])
	for x in xrange(4):
		ice_concNT_BG_season_means[i, x] = ma.mean(ice_concNT_BG[i, x*3:(x*3)+3])
		ice_concBT_BG_season_means[i, x] = ma.mean(ice_concBT_BG[i, x*3:(x*3)+3])

#ice_concNT_BG_season_means.dump('/Users/aapetty/DATA_OUTPUT/CONC_OUT/ice_concNT_BG_season_means.txt')

ice_concNT_BG_ann_mean=ma.mean(ice_concNT_BG_season_means, axis=1)
ice_concBT_BG_ann_mean=ma.mean(ice_concBT_BG_season_means, axis=1)

colors= ['g', 'b', 'r', 'm']
month_str = ['JFM', 'AMJ', 'JAS', 'OND']
years = np.arange(start_year, end_year+1, 1)

fig = figure(figsize=(6,3))
for plotnum in xrange(4):
	vars()['ax'+str(plotnum+1)] = subplot(2, 2, plotnum+1)
	ax_temp = gca()
	ax_temp.fill_between(years, ice_concNT_BG_ann_mean[:], ice_concBT_BG_ann_mean[:], alpha=0.2, edgecolor='k', facecolor='k')
	ax_temp.fill_between(years, ice_concNT_BG_season_means[:, plotnum], ice_concBT_BG_season_means[:, plotnum], alpha=0.2, edgecolor=colors[plotnum], facecolor=colors[plotnum])

	ax_temp.plot(years, ice_concNT_BG_ann_mean[:], linestyle='-',linewidth=1, color='k')
	ax_temp.plot(years, ice_concNT_BG_season_means[:, plotnum], linestyle='-',linewidth=1, color=colors[plotnum])
	ax_temp.plot(years, ice_concBT_BG_ann_mean[:], linestyle='--',linewidth=1, color='k')
	ax_temp.plot(years, ice_concBT_BG_season_means[:, plotnum], linestyle='--',linewidth=1, color=colors[plotnum])

	ax_temp.set_xlim(years[0], years[-1])
	ax_temp.set_yticks(np.arange(0.2, 1.2, 0.2))
	ax_temp.set_xticks(np.arange(1983, 2014, 6))
	ax_temp.yaxis.grid(True)
	ax_temp.xaxis.grid(True, which='major')
	ax_temp.set_ylim(0.2, 1.0)
	ax_temp.annotate(month_str[plotnum] , xy=(0.05, 0.05), color=colors[plotnum], xycoords='axes fraction', horizontalalignment='left', verticalalignment='bottom')

ax2.set_yticklabels([])
ax4.set_yticklabels([])
ax1.set_xticklabels([])
ax2.set_xticklabels([])

ax3.set_xlabel( 'Years')
ax4.set_xlabel( 'Years')
ax1.set_ylabel( 'Ice concentration')
ax3.set_ylabel( 'Ice concentration')



subplots_adjust( right = 0.97, left = 0.08, top=0.97, bottom=0.135, wspace=0.06, hspace=0.09)

savefig(figpath+'/seasonal_BG_conc_NTBT_ann4.pdf', dpi=300)
close(fig)


