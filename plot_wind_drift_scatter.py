############################################################## 
# Date: 01/01/16
# Name: plot_wind_drift_scatter.py
# Author: Alek Petty
# Description: Script to plot wind curl, drift curl scatter
# Input requirements: Drift and wind curl data
#                     PIOMAS BG thickness estimates
# Output: Scatter plot of wind curl, and wind curl expected from linear relationship

import BG_functions as BGF
import numpy as np
from pylab import *
import scipy.io
from scipy.io import netcdf

rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize']=8
rcParams['ytick.labelsize']=8
rcParams['font.size']=10
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['figure.figsize'] = 3.6, 3
majorLocator   = MultipleLocator(3)
majorFormatter = FormatStrFormatter('%d')
minorLocator   = MultipleLocator(1)

figpath='./Figures/'
datapath='./Data_output/'

years = '1980-2013'
grid_str='100km'
reanals= ['NCEP2', 'JRA', 'ERA']
reanal=0
month_strs = ['J-M', 'A-J', 'J-S', 'O-D']
box_str='wbox'
#box_str='ibox'
start_year = 1980
end_year=2013

time_F= np.arange(start_year,2014, 1)

product = 'FOWLER'
drift_curl_seasons_BG_mean_F = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_mean'+grid_str+'.txt')

drift_curl_BG_mean_F_ann = np.mean(drift_curl_seasons_BG_mean_F, axis=0)
drift_curl_BG_trend_ann, drift_curl_BG_sig_ann, drift_curl_BG_r_ann,drift_curl_BG_int_ann  = BGF.correlate(time_F, drift_curl_BG_mean_F_ann)
drift_curl_BG_tline_ann = (time_F*drift_curl_BG_trend_ann)+drift_curl_BG_int_ann


wind_curl_seasons_BG=ma.masked_all((3, 4, 34))
for r in range(3):
	for x in xrange(4):
		wind_path = datapath+'/WINDS/'+reanals[r]+'/WINDCURL/'+grid_str+'/'
		wind_curl_seasons_BG[r, x] = loadtxt(wind_path+years+month_strs[x]+'/ave_wind_curl_years_b'+box_str+'.txt')

wind_curl_BG_ann= ma.mean(wind_curl_seasons_BG[:, :, start_year-1980::], axis=1)


line = ma.masked_all((wind_curl_BG_ann.shape))
trend_wi=[]
sig_wi=[]
r_wi=[]
inter_wi=[]
for r in xrange(3):
	trend, sig, r_val, inter = BGF.correlate(wind_curl_BG_ann[r], drift_curl_BG_mean_F_ann)
	trend_wi.append(trend)
	sig_wi.append(sig)
	r_wi.append(r_val)
	inter_wi.append(inter)
	line[r] = trend_wi[r]*(wind_curl_BG_ann[r]) + inter_wi[r]
		
ratio = (drift_curl_BG_mean_F_ann-line)

r=0

fig = figure()
subplots_adjust( right = 0.98, left = 0.15, bottom=0.14, hspace=0.25, wspace=0.05, top=0.95)

ax=gca()
im1 = scatter(wind_curl_BG_ann[r]/1e-5, drift_curl_BG_mean_F_ann/1e-8, c=time_F, cmap=cm.RdYlBu, rasterized=True)
plot(wind_curl_BG_ann[r]/1e-5, line[r]/1e-8, 'k')
year_line = 28
plot([wind_curl_BG_ann[r, year_line]/1e-5, wind_curl_BG_ann[r, year_line]/1e-5], [drift_curl_BG_mean_F_ann[year_line]/1e-8, line[r, year_line]/1e-8], '-g')
ax.annotate('calc - pred' , xy=(0.47, 0.33), xycoords='axes fraction', fontsize=10, color='g', horizontalalignment='left', verticalalignment='bottom')

r_str = '%.2f' % r_wi[r]
ax.annotate('r:'+r_str, xy=(0.08, 0.7), xycoords='axes fraction', color='k', fontsize=10, horizontalalignment='left', verticalalignment='bottom')

gca().invert_xaxis()
cax = fig.add_axes([0.83, 0.47, 0.03, 0.45])
cbar = colorbar(im1,cax=cax, orientation='vertical', extend='neither', use_gridspec=True)
cbar.set_ticks(np.arange(1980, 2014, 3))
cbar.solids.set_rasterized(True)
ax.xaxis.grid(True)
ax.yaxis.grid(True)

ax.set_xlabel('Wind curl '+r'(10$^{-5}$m s$^{-2}$)')
ax.set_ylabel('Drift curl '+r'(10$^{-8}$m s$^{-2}$)')

savefig(figpath+'/wind_drift_correlation'+box_str+str(start_year)+'.pdf', dpi=300)
close(fig)








