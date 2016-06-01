############################################################## 
# Date: 01/01/16
# Name: plot_drift_curl_trend_seasons_top.py
# Author: Alek Petty
# Description: Script to plot seasonal ice drift curl trends over the Arctic/BG
# Input requirements: Ice drift curl trend data, produced by calc_curl_trend.py (for all differnt FOWLER/CERSAT products)     
# Output: Maps and lineplots of ice drift curl trends from 3 reanalyses


import BG_functions as BGF
import matplotlib
matplotlib.use("AGG")
from mpl_toolkits.basemap import Basemap, shiftgrid
import numpy as np
from pylab import *
import numpy.ma as ma
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid.inset_locator import mark_inset
from mpl_toolkits.axes_grid.anchored_artists import AnchoredSizeBar
from matplotlib import rc


def BG_box():
	lons_beau = np.zeros((40))
	lons_beau[0:10] = np.linspace(beau_region[3], beau_region[2], 10)
	lons_beau[10:20] = np.linspace(beau_region[2], beau_region[2], 10)
	lons_beau[20:30] = np.linspace(beau_region[2], beau_region[3], 10)
	lons_beau[30:40] = np.linspace(beau_region[3], beau_region[3], 10)
	lats_beau = np.zeros((40))
	lats_beau[0:10] = np.linspace(beau_region[1], beau_region[1], 10)
	lats_beau[10:20] = np.linspace(beau_region[1], beau_region[0], 10)
	lats_beau[20:30] = np.linspace(beau_region[0], beau_region[0], 10)
	lats_beau[30:40] = np.linspace(beau_region[0], beau_region[1], 10)

	return lons_beau, lats_beau

rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize']=9
rcParams['ytick.labelsize']=9
rcParams['font.size']=9
#rcParams['text.usetex'] = True
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
#majorLocator   = MultipleLocator(3)
majorFormatter = FormatStrFormatter('%d')
minorLocator   = MultipleLocator(1)


m = Basemap(projection='npstere',boundinglat=66,lon_0=0, resolution='l')

datapath='./Data_output/'
figpath = './Figures/'

beau_region = [72., 82., -130, -170]

start_year = 1980
end_year = 2013
num_years = end_year -start_year + 1
minval=-2
maxval=2
month_str = ['JFM', 'AMJ', 'JAS', 'OND']
year_str = '1980-2013'
units_trend = r'm s$^{-2}$yr$^{-1}$'

extra='100km'

product = 'FOWLER'

drift_curl_seasons_trend = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_trend'+extra+'.txt')
drift_curl_seasons_sig = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_sig'+extra+'.txt')
drift_curl_seasons_r = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_r'+extra+'.txt')
drift_curl_seasons_BG_mean = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_mean'+extra+'.txt')
drift_curl_seasons_BG_tline = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_tline'+extra+'.txt')

product = 'FOWLER_MA'

drift_curl_seasons_trend_ma = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_trend'+extra+'.txt')
drift_curl_seasons_sig_ma = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_sig'+extra+'.txt')
drift_curl_seasons_r_ma = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_r'+extra+'.txt')
drift_curl_seasons_BG_mean_ma = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_mean'+extra+'.txt')
drift_curl_seasons_BG_tline_ma = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_tline'+extra+'.txt')


start_year = 2008
end_year = 2013
product = 'ASCAT_6DAY'
drift_curl_seasons_BG_mean_A6 = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_mean'+extra+'.txt')
drift_curl_seasons_BG_tline_A6 = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_tline'+extra+'.txt')
product = 'ASCAT_3DAYMFILL'
drift_curl_seasons_BG_mean_A3 = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_mean'+extra+'.txt')
drift_curl_seasons_BG_tline_A3 = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_tline'+extra+'.txt')
start_year = 1992
end_year = 2008
product = 'QSCAT_3DAYMFILL'
#extra=''
drift_curl_seasons_BG_mean_Q = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_mean'+extra+'.txt')
drift_curl_seasons_BG_tline_Q = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_tline'+extra+'.txt')


xpts = loadtxt(datapath+'/xpts_100km.txt')
ypts = loadtxt(datapath+'/ypts_100km.txt')
lons_t = loadtxt(datapath+'/lons_100km.txt')
lats_t = loadtxt(datapath+'/lats_100km.txt')



plotnum_sr = ([1, 3, 5, 7], [2, 4, 6, 8])
xa,ya = m(-80,73) # we define the corner 1
x2a,y2a = m(162,70) # then corner 2

axes=[]
axes.append([0.005, 0.668, 0.24, 0.33])
axes.append([0.255, 0.668, 0.24, 0.33])
axes.append([0.505, 0.668, 0.24, 0.33])
axes.append([0.755, 0.668, 0.24, 0.33])

axes.append([0.07, 0.35, 0.45, 0.24])
axes.append([0.54, 0.35, 0.45, 0.24])
axes.append([0.07, 0.08, 0.45, 0.24])
axes.append([0.54, 0.08, 0.45, 0.24])

ymins = [-4.5, -3, -3, -8.]
ymaxs = [0.5, 0.5, 0.5, 0.5]

fig = figure(figsize=(6.67, 5.0))

for x in xrange(4):
	vars()['ax'+str(x+1)] = fig.add_axes(axes[x])
	ax_t=gca()
	#REPLACE MASKED WITH 0 AS THIS WILL SHOW UP AS WHITE
	drift_curl_seasons_trend_0 = drift_curl_seasons_trend.filled(0)
	im1 = m.pcolormesh(xpts , ypts, drift_curl_seasons_trend_0[x]/1e-9, vmin=minval, vmax=maxval, cmap=plt.cm.RdBu_r,shading='gouraud', zorder=4, rasterized=True)
	im2 = m.contour(xpts, ypts, drift_curl_seasons_sig[x],levels=[95], colors='y', zorder=5)
	# LOWER THE SCALE THE LARGER THE ARROW#res = 3
	#Q = m.quiver(xpts[::res, ::res], ypts[::res, ::res], drift_xy_seasons_trend[x, 0, ::res, ::res], drift_xy_seasons_trend[x, 1,  ::res, ::res], units='inches',scale=0.025, zorder=5)

	m.drawparallels(np.arange(90,-90,-10), linewidth = 0.25, zorder=10)
	m.drawmeridians(np.arange(-180.,180.,30.), linewidth = 0.25, zorder=10)
	m.fillcontinents(color='0.7',lake_color='grey', zorder=6)

	lons_beau, lats_beau = BG_box()
	xb, yb = m(lons_beau, lats_beau) # forgot this line
	m.plot(xb, yb, '--', linewidth=2, color='k', zorder=5)

	xS, yS = m(275, 74)
	ax_t.text(xS, yS, 'NSIDC/PP'+'\n'+month_str[x], zorder = 11, backgroundcolor = 'w')

	ax_t.set_xlim(xa,x2a)
	ax_t.set_ylim(ya,y2a)

years_F = np.arange(1980, 2014, 1)
years_A = np.arange(2008, 2014, 1)
years_Q = np.arange(1992, 2009, 1)
years_ALL= np.arange(1980, 2014, 1)

for x in xrange(4):
	vars()['ax'+str(x+5)] = fig.add_axes(axes[x+4])
	ax_t=gca()
	
	pl_F = plot(years_F, drift_curl_seasons_BG_mean[x]/1e-8, linestyle='-',linewidth=1, color='m')
	pl_F_ma = plot(years_F, drift_curl_seasons_BG_mean_ma[x]/1e-8, linestyle='-',linewidth=1, color='y')
	if (x==0):
		pl_A3 = plot(years_A, drift_curl_seasons_BG_mean_A3[0]/1e-8, linestyle='-',linewidth=1, color='b')
		pl_A6 = plot(years_A, drift_curl_seasons_BG_mean_A6[0]/1e-8, linestyle='-',linewidth=1, color='r')
		pl_Q = plot(years_Q, drift_curl_seasons_BG_mean_Q[0]/1e-8, linestyle='-',linewidth=1, color='g')
	if (x==3):
		pl_A3 = plot(years_A, drift_curl_seasons_BG_mean_A3[1]/1e-8, linestyle='-',linewidth=1, color='b')
		pl_A6 = plot(years_A, drift_curl_seasons_BG_mean_A6[1]/1e-8, linestyle='-',linewidth=1, color='r')
		pl_Q = plot(years_Q, drift_curl_seasons_BG_mean_Q[1]/1e-8, linestyle='-',linewidth=1, color='g')
	ax_t.annotate(month_str[x], xy=(0.05, 0.87), xycoords='axes fraction', horizontalalignment='left', verticalalignment='bottom')


	#sig_BG_str = '%.0f' % (drift_curl_seasons_BG_sig[x])
	#trend_BG_str = '%.1e' % (drift_curl_seasons_BG_trend[x])
	#trend_BG_sig_str = trend_BG_str+units_trend+' ('+sig_BG_str+'%)'
	#ax_t.annotate(trend_BG_sig_str, xy=(0.03, 0.1), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='bottom')

	ax_t.set_xlim(years_ALL[0], years_A[-1])
	#ax_t.set_xticks(years_ALL[::3])
	ax_t.set_xticklabels( [])
	ax_t.xaxis.set_major_locator(MultipleLocator(6))
	ax_t.xaxis.set_minor_locator(minorLocator)
	ax_t.yaxis.grid(True)
	ax_t.xaxis.grid(True, which='major')
	ax_t.set_ylim(-8, 2)
	
plts_net = pl_Q+pl_A3+pl_A6+pl_F+pl_F_ma
leg = ax5.legend(plts_net, ['CSAT/QS', 'CSAT/AS-3', 'CSAT/AS-6', 'NSIDC/PP', 'NSIDC/PP-MASK'], loc=3, ncol=2,fontsize=9, columnspacing=0.1, handletextpad=0.0001, frameon=False)
leg.set_zorder(20)
ax6.set_yticklabels( [])
ax8.set_yticklabels( [])

ax7.xaxis.set_major_formatter(majorFormatter)
ax8.xaxis.set_major_formatter(majorFormatter)
#year_strs=['`80','`83','`86','`89', '`92','`95','`98','`01','`04', '`07', '`10', '`13']
#ax_t.set_xticklabels(year_strs)
ax7.set_xlabel( 'Years')
ax8.set_xlabel( 'Years')
#ax5.set_ylabel( 'Mean BG Ice Drift Curl '+r'(10$^{-8}$m s$^{-2}$)',fontsize=10)




#leg = ax1.legend(plts, labs, loc=2)
llines = leg.get_lines()
setp(llines, linewidth=2.0)
#leg.get_frame().set_alpha(0.5)

cax = fig.add_axes([0.35, 0.66, 0.3, 0.015])
cbar = colorbar(im1,cax=cax, orientation='horizontal', extend='both', use_gridspec=True)
cbar.set_label(r'Drift curl trend (10$^{-9}$m s$^{-2}$yr$^{-1}$)', fontsize=9, labelpad=-2)
xticks = np.linspace(minval, maxval, 5)
cbar.set_ticks(xticks)
cbar.formatter.set_powerlimits((-3, 4))
cbar.formatter.set_scientific(True)
cbar.update_ticks() 
ax = fig.add_axes( [0., 0., 1, 1] )
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

cbar.solids.set_rasterized(True)

ax.text(.02, 0.34, 'Drift curl '+r'(10$^{-8}$m s$^{-2}$)', rotation=90,horizontalalignment='center', verticalalignment='center')


#savefig(out_path+'curl_trend_BG_top.png', dpi=300)
savefig(figpath+'curl_trend_BG_top'+extra+'.pdf', dpi=300)
close(fig)
