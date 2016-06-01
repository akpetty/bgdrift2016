############################################################## 
# Date: 01/01/16
# Name: plot_summer_extent_NTBT.py
# Author: Alek Petty
# Description: Script to plot sumemr ice extent in the Arctic and BG region from NASATEAM/BOOTSTRAP
# Input requirements: Monthly ice conc data from both renalyases
#                     Also needs the functions in BG_functions
# Output: Lineplot of Arctic BG ice extent in summer


import matplotlib
matplotlib.use("AGG")
from mpl_toolkits.basemap import Basemap, shiftgrid
import numpy as np
from pylab import *
from scipy.io import netcdf
import numpy.ma as ma
from matplotlib import rc

mpl.rc("ytick",labelsize=10)
mpl.rc("xtick",labelsize=10)
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

m = Basemap(projection='npstere',boundinglat=66,lon_0=0, resolution='l'  )

beau_lonlat = [-175., -125., 85., 70.]

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

ice_concNT = ice_concNT.filled(0)
ice_extNT = where((ice_concNT >=0.15), 1, 0)
ice_extBGNT=np.copy(ice_extNT)

ice_concBT = ice_concBT.filled(0)
ice_extBT = where((ice_concBT >=0.15), 1, 0)
ice_extBGBT=np.copy(ice_extBT)

for x in xrange(ice_extNT.shape[0]):
	for y in xrange(ice_extNT.shape[1]):
		ice_extNT[x, y] = where((lats >=85), 1, ice_extNT[x, y])
		ice_extBGNT[x, y] = where((lons<beau_lonlat[0]) | (lons>beau_lonlat[1]) | (lats>beau_lonlat[2])| (lats<beau_lonlat[3]), 0, ice_extBGNT[x, y])
		
		ice_extBT[x, y] = where((lats >=85), 1, ice_extBT[x, y])
		ice_extBGBT[x, y] = where((lons<beau_lonlat[0]) | (lons>beau_lonlat[1]) | (lats>beau_lonlat[2])| (lats<beau_lonlat[3]), 0, ice_extBGBT[x, y])

ice_extNTS = np.sum(np.sum(ice_extNT[:, 8],axis=2), axis=1)*25*25/1e6
ice_extBGNT= np.sum(np.sum(ice_extBGNT[:, 8],axis=2), axis=1)*25*25/1e6

ice_extBTS = np.sum(np.sum(ice_extBT[:, 8],axis=2), axis=1)*25*25/1e6
ice_extBGBT= np.sum(np.sum(ice_extBGBT[:, 8],axis=2), axis=1)*25*25/1e6

majorLocator   = MultipleLocator(3)
majorFormatter = FormatStrFormatter('%d')
minorLocator   = MultipleLocator(1)

years = np.arange(1980, 2014, 1)
fig = figure(figsize=(5,2.3))
ax1 = gca()
ax1.set_xlim(years[0], years[-1])

ax1.xaxis.set_major_locator(majorLocator)
ax1.xaxis.set_major_formatter(majorFormatter)
ax1.xaxis.set_minor_locator(minorLocator)


p1 = fill_between(years, ice_extNTS, ice_extBTS, alpha=0.2, edgecolor='k', facecolor='k')
pl1 = plot(years, ice_extNTS, linestyle='-',linewidth=1, color='k')
pl11 = plot(years, ice_extBTS, linestyle='--',linewidth=1, color='k')

ax2 = ax1.twinx()

p2 = fill_between(years, ice_extBGNT, ice_extBGBT, alpha=0.2, edgecolor='b', facecolor='b')
pl2 = ax2.plot(years, ice_extBGNT, linestyle='-',linewidth=1, color='b')
pl22 = ax2.plot(years, ice_extBGBT, linestyle='--',linewidth=1, color='b')

ax1.yaxis.grid(True)
ax1.xaxis.grid(True, which='major')
ax1.set_ylim(2, 8)
ax1.set_xlabel( 'Years',fontsize=10)

ax1.set_ylabel( 'Ice extent'+r' (10$^{6}$ km$^2$)',fontsize=10)
ax2.set_ylim(0.5, 2)
ax2.set_yticks(np.linspace(0.5, 2, 7))

ax2.set_ylabel( 'Ice extent'+r' (10$^{6}$ km$^2$)',color='b', fontsize=10, labelpad = 14, rotation=270)
ax2.spines['right'].set_color('b')
#ax1.yaxis.label.set_color('red')
ax2.tick_params(axis='y', colors='b')

plts_net = pl1+pl2
#labs = [l.get_label() for l in plts]

Regions = ['Arctic Ocean', 'Beaufort Gyre']

leg = ax1.legend(plts_net, Regions, loc=3, ncol=2,columnspacing=0.1, handletextpad=0.0001, borderaxespad=0.)
#leg = ax1.legend(plts, labs, loc=2)
llines = leg.get_lines()
setp(llines, linewidth=2.0)
ltext  = leg.get_texts()
setp(ltext, fontsize=10)
leg.get_frame().set_alpha(0.5)


subplots_adjust( right = 0.88, left = 0.08, top=0.965, bottom=0.17)

savefig(figpath+'/summer_Arctic_BG_extent'+years_str+'NTBT.pdf', dpi=300)
close(fig)


