############################################################## 
# Date: 01/01/16
# Name: plot_calc_trend_seasons_ann.py
# Author: Alek Petty
# Description: Script to plot annual and seasonal trends in ice concentration (maps)			   
# Input requirements: Seasonal/annual ice concentration trend data, produced using calc_conc_trend_ann.py and calc_conc_trend_seasons.py                   
# Output: Map of concentration trends over the Arctic

import matplotlib
matplotlib.use("AGG")
from mpl_toolkits.basemap import Basemap, shiftgrid
import numpy as np
from pylab import *
import numpy.ma as ma
from matplotlib import rc
from netCDF4 import Dataset
from glob import glob
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def BG_box(beau_region):
	lons_beau = np.zeros((40))
	lons_beau[0:10] = np.linspace(beau_region[0], beau_region[1], 10)
	lons_beau[10:20] = np.linspace(beau_region[1], beau_region[1], 10)
	lons_beau[20:30] = np.linspace(beau_region[1], beau_region[0], 10)
	lons_beau[30:40] = np.linspace(beau_region[0], beau_region[0], 10)
	lats_beau = np.zeros((40))
	lats_beau[0:10] = np.linspace(beau_region[2], beau_region[2], 10)
	lats_beau[10:20] = np.linspace(beau_region[2], beau_region[3], 10)
	lats_beau[20:30] = np.linspace(beau_region[3], beau_region[3], 10)
	lats_beau[30:40] = np.linspace(beau_region[3], beau_region[2], 10)

	return lons_beau, lats_beau

mpl.rc("ytick",labelsize=10)
mpl.rc("xtick",labelsize=10)
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

m = Basemap(projection='npstere',boundinglat=66,lon_0=0, resolution='l'  )
#m = apy.polar_stere(-170., -127, 70, 84, resolution='l')

datapath='./Data_output/'
figpath = './Figures/'

alg = 'NASA_TEAM'
num_years_req=10

lats = load(datapath+'/ice_conc_lats'+alg+'.txt')
lons = load(datapath+'/ice_conc_lons'+alg+'.txt')
xpts, ypts = m(lons, lats)

start_year=1980
end_year=2013

conc_seasons_trend=load(datapath+'/conc_seasons_trend'+alg+'nyrs'+str(num_years_req)+str(start_year)+'-'+str(end_year)+'.txt')
conc_seasons_sig=load(datapath+'/conc_seasons_sig'+alg+'nyrs'+str(num_years_req)+str(start_year)+'-'+str(end_year)+'.txt')
conc_seasons_r=load(datapath+'/conc_seasons_r'+alg+'nyrs'+str(num_years_req)+str(start_year)+'-'+str(end_year)+'.txt')

conc_ann_trend=load(datapath+'/conc_ann_trend'+alg+'nyrs'+str(num_years_req)+str(start_year)+'-'+str(end_year)+'.txt')
conc_ann_sig=load(datapath+'/conc_ann_sig'+alg+'nyrs'+str(num_years_req)+str(start_year)+'-'+str(end_year)+'.txt')
conc_ann_r=load(datapath+'/conc_ann_r'+alg+'nyrs'+str(num_years_req)+str(start_year)+'-'+str(end_year)+'.txt')

#MAKE SURE POLAR HOLE IS MASKED OUT
for x in xrange(4):
	conc_seasons_trend[x]=ma.masked_where(lats>85,conc_seasons_trend[x])
	conc_seasons_sig[x]=ma.masked_where(lats>85,conc_seasons_sig[x])
conc_ann_trend=ma.masked_where(lats>85,conc_ann_trend)
conc_ann_sig=ma.masked_where(lats>85,conc_ann_sig)

#beau_lonlat = [-170., -130., 82., 72.]
beau_lonlat = [-175., -125., 85., 70.]
lons_beau, lats_beau = BG_box(beau_lonlat)
xb, yb = m(lons_beau, lats_beau) # forgot this line

minval = -2.5
maxval = 2.5
sig_level=95

axes=[]
axes.append([0.01, 0.25, 0.33, 0.485])
axes.append([0.34, 0.51, 0.33, 0.485])
axes.append([0.67, 0.51, 0.33, 0.485])
axes.append([0.34, 0.01, 0.33, 0.485])
axes.append([0.67, 0.01, 0.33, 0.485])
axesname = ['ax1', 'ax2', 'ax3', 'ax4', 'ax5', 'ax6', 'ax7', 'ax8', 'ax9', 'ax10', 'ax11', 'ax12', 'ax13', 'ax14', 'ax15', 'ax16']
month_strs=['(b) JFM', '(c) AMJ', '(d) JAS', '(e) OND']

norm = MidpointNormalize(midpoint=0)
fig = figure(figsize=(6,4))
vars()[axesname[0]] = fig.add_axes(axes[0])

im =m.pcolormesh(xpts, ypts, conc_ann_trend*100., cmap=cm.RdBu_r, norm=norm,vmin=minval-0.01, vmax=maxval, zorder=1, rasterized=True)
im2=m.contour(xpts, ypts, conc_ann_sig, levels=[sig_level], colors='y', linewidths=0.5,zorder=2)
m.fillcontinents(color='0.7',lake_color='grey', zorder=4)
m.drawparallels(np.arange(90,-90,-10), linewidth = 0.25, zorder=5)
m.drawmeridians(np.arange(-180.,180.,30.), linewidth = 0.25, zorder=5)
m.plot(xb, yb, '-', linewidth=1.5, color='k', zorder=3)
vars()[axesname[0]].annotate('(a) Annual', xy=(0.01, 0.91), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='bottom', zorder=10)
	
for p in xrange(4):
	vars()[axesname[p]] = fig.add_axes(axes[p+1])

	im =m.pcolormesh(xpts, ypts, conc_seasons_trend[p]*100., norm=norm,cmap=cm.RdBu_r, vmin=minval-0.01, vmax=maxval, zorder=1, rasterized=True)
	im2=m.contour(xpts, ypts, conc_seasons_sig[p], levels=[sig_level], colors='y',linewidths=0.5, zorder=2)
	m.fillcontinents(color='0.7',lake_color='grey', zorder=4)
	m.drawparallels(np.arange(90,-90,-10), linewidth = 0.25, zorder=5)
	m.drawmeridians(np.arange(-180.,180.,30.), linewidth = 0.25, zorder=5)
	
	m.plot(xb, yb, '-', linewidth=1.5, color='k', zorder=3)
	vars()[axesname[p]].annotate(month_strs[p], xy=(0.01, 0.91), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='bottom', zorder=10)
	
cax = fig.add_axes([0.02, 0.2, 0.31, 0.04])
cbar = colorbar(im, cax=cax, orientation='horizontal', norm=norm, extend='both',use_gridspec=True)
cbar.set_label('Concentration trend (%/yr)', fontsize=10)
xticks = np.linspace(minval, maxval, 5)
cbar.set_ticks(xticks)
cbar.solids.set_rasterized(True)
subplots_adjust( right = 0.99, left = 0.01, top=0.99, bottom=0.01, wspace=0.03, hspace=0.03)

#savefig(figpath+'/Arctic_conc_trend_season_ann'+alg+str(num_years_req)+str(start_year)+'-'+str(end_year)+'.pdf', dpi=200)
savefig(figpath+'/Arctic_conc_trend_season_ann'+alg+str(num_years_req)+str(start_year)+'-'+str(end_year)+'.png', dpi=300)
close(fig)



