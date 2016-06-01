############################################################## 
# Date: 01/01/16
# Name: plot_map_BG_gates_SMDT.py
# Author: Alek Petty
# Description: Script to plot map of MDT and BG regions
# Input requirements: MDT and BATHY
# Output: Map of Arctic MDT and BG highlighted

import BG_functions as BGF
import matplotlib
matplotlib.use("AGG")
from mpl_toolkits.basemap import Basemap, shiftgrid
import numpy as np
from pylab import *
import numpy.ma as ma
import string
from matplotlib.patches import Polygon
from scipy import stats
from scipy.interpolate import griddata
from matplotlib import rc
from netCDF4 import Dataset
from glob import glob
import os

datapath='./Data/'
figpath = './Figures/'

mpl.rc("ytick",labelsize=10)
mpl.rc("xtick",labelsize=10)
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
#mpl.rc('text', usetex=True)
m = Basemap(projection='npstere',boundinglat=69,lon_0=0, resolution='l'  )

lon1 = -125.
lon2 = -155.
lat1 = 72.
lat2 = 76.
lat3 = 80.

beau_lonlat_wind = [-175., -125., 85., 70.]

beau_lonlat_ice = [-170., -130., 82., 72.]


lons_beau_wind, lats_beau_wind = BGF.calc_beau_lonlat(beau_lonlat_wind)

lons_beau_ice, lats_beau_ice = BGF.calc_beau_lonlat(beau_lonlat_ice)

gate1_lons = np.linspace(lon1, lon2, 10)
gate1_lats = np.linspace(lat3, lat3, 10)
gate2_lons = np.linspace(lon2, lon2, 10)
gate2_lats = np.linspace(lat1, lat2, 10)
gate3_lons = np.linspace(lon2, lon2, 10)
gate3_lats = np.linspace(lat2, lat3, 10)

xpts_g1, ypts_g1 = m(gate1_lons, gate1_lats)
xpts_g2, ypts_g2 = m(gate2_lons, gate2_lats)
xpts_g3, ypts_g3 = m(gate3_lons, gate3_lats)

xpts_w, ypts_w = m(lons_beau_wind, lats_beau_wind)
xpts_i, ypts_i = m(lons_beau_ice, lats_beau_ice)

xa,ya = m(-64.2,68) # we define the corner 1
x2a,y2a = m(152,66) # then corner 2
xa,ya = m(-58,70) # we define the corner 1
x2a,y2a = m(145,65) # then corner 2


iceS_mdt = loadtxt(datapath+'/MDT/ICEn_MSS-GOCO2S.nn.510f.xyz')
    # data is a table-like structure (a numpy recarray) in which you can access columns and rows easily
lon_m = iceS_mdt[:, 0].astype(float)
lat_m = iceS_mdt[:, 1].astype(float)
mdt = iceS_mdt[:, 2].astype(float)
xpts_m, ypts_m = m(lon_m, lat_m)

xxS = np.arange(m.xmin,m.xmax, 25000)
yyS = np.arange(m.ymin,m.ymax, 25000)
xx2d, yy2d = meshgrid(xxS, yyS)

#mdt2d = m.transform_scalar(mdt, lon_m,lat_m,100,100,returnxy=False)

mdt2d = griddata((xpts_m, ypts_m), mdt, (xx2d, yy2d), method='linear')
mdt2d = ma.masked_where(isnan(mdt2d), mdt2d)
minval = -0.4
maxval = 0.4

bathy_file = Dataset(datapath+'/BATHY/IBCAO_V3_30arcsec_RR.grd','r')
#bathy_file.variables()
dlat=5
dlon=10
bathy = bathy_file.variables['z'][::dlat, ::dlon]
lon_m = bathy_file.variables['x'][::dlon]
lat_m = bathy_file.variables['y'][::dlat]
xpts_m, ypts_m = m(*np.meshgrid(lon_m, lat_m))

bathy_levels=np.arange(-250, -5000, -1000)

fig = figure(figsize=(4,4))
ax=gca()
subplots_adjust(bottom=0.01, left=0.01, top = 0.99, right=0.99)

im1 = m.pcolormesh(xx2d, yy2d, mdt2d, shading='flat', cmap=cm.RdYlBu_r, vmin=minval, vmax=maxval, rasterized=True, zorder=1)
im2 = m.contour(xpts_m, ypts_m, bathy, levels=bathy_levels, colors='0.8', linewidths = 0.25, zorder=2)

m.drawmeridians(np.arange(-180.,181.,30.), latmax=90, yoffset=-(m.ymax-m.ymin)*0.04, linewidth = 0.25, zorder=2)
m.drawparallels(np.arange(-80., 86.,5), xoffset=-(m.xmax-m.xmin)*0.08, linewidth = 0.25, zorder=2)

m.plot(xpts_g1, ypts_g1, '-', linewidth = 2.5, color='b', zorder=3)
m.plot(xpts_g2, ypts_g2, '-', linewidth = 2.5,color='g', zorder=3)
m.plot(xpts_g3, ypts_g3, '-', linewidth = 2.5,color='m', zorder=3)
#m.plot(xpts_g4, ypts_g4, '-', linewidth = 2,color='m')

m.plot(xpts_w, ypts_w, '-', linewidth = 2, color='k', zorder=3)
m.plot(xpts_i, ypts_i, '--', linewidth = 2, color='k', zorder=3)




m.fillcontinents(color='0.7',lake_color='grey', zorder=5)
ax.set_xlim(xa,x2a)
ax.set_ylim(ya,y2a)

x,y = m(165,70-0.2)
ax.text(x,y, r'70$^{\circ}$N',size=7, horizontalalignment='center',verticalalignment='center',color='k',zorder = 9)
x,y = m(165,75-0.2)
ax.text(x,y, r'75$^{\circ}$N',size=7, horizontalalignment='center',verticalalignment='center',color='k',zorder = 9)
x,y = m(165,80-0.2)
ax.text(x,y, r'80$^{\circ}$N',size=7, horizontalalignment='center',verticalalignment='center',color='k',zorder = 9)
x,y = m(165,85-0.2)
ax.text(x,y, r'85$^{\circ}$N',size=7, horizontalalignment='center',verticalalignment='center',color='k',zorder = 9)

x,y = m(1,80.1)
ax.text(x,y, r'0$^{\circ}$E',size=7, horizontalalignment='center',verticalalignment='center',color='k',zorder = 9)
x,y = m(91,77)
ax.text(x,y, r'90$^{\circ}$E',size=7, horizontalalignment='center',verticalalignment='center',color='k',zorder = 9)
x,y = m(181.5,70.4)
ax.text(x,y, r'180$^{\circ}$E',size=7, horizontalalignment='center',verticalalignment='center',color='k',zorder = 9)
x,y = m(271.5,75)
ax.text(x,y, r'270$^{\circ}$E',size=7, horizontalalignment='center',verticalalignment='center',color='k',zorder = 9)

x,y = m(-150,75)
ax.text(x,y, '*A',size=9, horizontalalignment='center',verticalalignment='center',color='k',zorder = 9)
x,y = m(-150,78)
ax.text(x,y, '*B',size=9, horizontalalignment='center',verticalalignment='center',color='k',zorder = 9)
x,y = m(-140,77)
ax.text(x,y, '*C',size=9, horizontalalignment='center',verticalalignment='center',color='k',zorder = 9)
x,y = m(-140,74)
ax.text(x,y, '*D',size=9, horizontalalignment='center',verticalalignment='center',color='k',zorder = 9)

cax = fig.add_axes([0.04, 0.955, 0.18, 0.027])
cbar = colorbar(im1,cax=cax, orientation='horizontal', extend='both', use_gridspec=True)
cbar.set_ticks(np.linspace(minval, maxval, 2))
cbar.set_clim(minval, maxval)
cbar.set_label('MDT (m)', labelpad=-0.7)
cbar.solids.set_rasterized(True)
#m.drawcoastlines(linewidth=0.5)
#tight_layout()
savefig(figpath+'/area_flux_gates_quads_wind_map_SMDT.png', dpi=300)
#savefig('/Users/apetty/NOAA/FIGURES/DRIFT_PAPER_FIGS/area_flux_gates_quads_wind_map_SMDT.pdf', dpi=300)
close(fig)



