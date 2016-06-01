############################################################## 
# Date: 01/01/16
# Name: plot_wind_drift_curl_ann_dcorr.py
# Author: Alek Petty
# Description: Script to plot wind curl, drift curl lineplots (annual)
# Input requirements: Drift and wind curl data
# Output: Line plots of wind curl, and wind curl expected from linear relationship

import BG_functions as BGF
import numpy as np
from pylab import *
import scipy.io
from scipy.io import netcdf

def calc_trends(var1, var2, time=0):

	trend = []
	sig = []
	r = []
	inter = []
	for x in xrange(4):
		if (time==0):
			tre_s, sig_s, r_s, int_s = BGF.correlate(var1[x], var2[x])
		if (time==1):
			tre_s, sig_s, r_s, int_s = BGF.correlate(var1, var2[x])
		
		trend.append(tre_s)
		sig.append(sig_s)
		r.append(r_s)
		inter.append(int_s)

	return trend, sig, r, inter

rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize']=8
rcParams['ytick.labelsize']=8
rcParams['font.size']=10
#rcParams['text.usetex'] = True
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['figure.figsize'] = 3.6, 3
majorLocator   = MultipleLocator(3)
majorFormatter = FormatStrFormatter('%d')
minorLocator   = MultipleLocator(1)

years = '1980-2013'
grid_str='100km'
reanals= ['NCEP2', 'ERA', 'JRA']
reanal=1
month_strs = ['J-M', 'A-J', 'J-S', 'O-D']
#box_str=''
box_str='wbox'
start_year = 1980
end_year=2013

figpath='./Figures/'
datapath='./Data_output/'

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
#ice_curl_seasons_BG=ice_curl_seasons_BG[:, start_year-1980::]


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
		
#line_ice = ma.masked_all((ice_curl_seasons_BG.shape))
#trend_ice, sig_ice, r_ice, inter_ice = calc_trends(time_F, drift_curl_seasons_BG_meanF, time=1)
#for x in xrange(4):
#	line_ice[x] = trend_ice[x]*time_F + inter_ice[x]

line_wind = ma.masked_all((wind_curl_BG_ann.shape))
trend_wind=[]
sig_wind=[]
r_wind=[]
inter_wind=[]


for r in xrange(3):
	trend, sig, r_val, inter = BGF.correlate(time_F, wind_curl_BG_ann[r])
	trend_wind.append(trend)
	sig_wind.append(sig)
	r_wind.append(r_val)
	inter_wind.append(inter)
	line_wind[r] = (trend_wind[r]*time_F) + inter_wind[r]

ratio = (drift_curl_BG_mean_F_ann-line)

#DECORRELATED TRENDS
wind_curl_BG_ann_dcorr=wind_curl_BG_ann-line_wind
#wind_curl_BG_ann_dcorr = signal.detrend(wind_curl_BG_ann, axis=1)
#wind_curl_BG_ann_dcorr1 = mlab.detrend_linear(wind_curl_BG_ann[0])

drift_curl_BG_mean_F_ann_dcorr=drift_curl_BG_mean_F_ann-drift_curl_BG_tline_ann

r_wi_dcorr=[]
for r in xrange(3):
	trend, sig, r_val, inter = BGF.correlate(wind_curl_BG_ann_dcorr[r], drift_curl_BG_mean_F_ann_dcorr)
	r_wi_dcorr.append(r_val)

beau_region = [72., 82., -130, -170]


fig = figure()
subplots_adjust( right = 0.98, left = 0.19, bottom=0.1, hspace=0.24, wspace=0.05, top=0.915)

ax1 = subplot(3, 1, 1)
p13 = ax1.plot(time_F, wind_curl_BG_ann[2]/1e-5,linestyle='-', color='b')
p11 = ax1.plot(time_F, wind_curl_BG_ann[0]/1e-5,linestyle='-', color='m')
p12 = ax1.plot(time_F, wind_curl_BG_ann[1]/1e-5,linestyle='-', color='g')

p131 = ax1.plot(time_F, line_wind[2]/1e-5,linestyle='--', color='b')
p111 = ax1.plot(time_F, line_wind[0]/1e-5,linestyle='--', color='m')
p121 = ax1.plot(time_F, line_wind[1]/1e-5,linestyle='--', color='g')


trend_wind_str = '%.1e' % trend_wind[0]
sig_wind_str = '%2d' % sig_wind[0]
ax1.annotate(trend_wind_str+' ('+sig_wind_str+'%)' , xy=(0.03, -0.2), xycoords='axes fraction', color='m', fontsize=9, horizontalalignment='left', verticalalignment='bottom')
trend_wind_str = '%.1e' % trend_wind[1]
sig_wind_str = '%2d' % sig_wind[1]
ax1.annotate(''+trend_wind_str+' ('+sig_wind_str+'%)' , xy=(0.35, -0.2), xycoords='axes fraction', color='g', fontsize=9, horizontalalignment='left', verticalalignment='bottom')
trend_wind_str = '%.1e' % trend_wind[2]
sig_wind_str = '%2d' % sig_wind[2]
ax1.annotate(trend_wind_str+' ('+sig_wind_str+'%)' , xy=(0.68, -0.2), xycoords='axes fraction', color='b', fontsize=9, horizontalalignment='left', verticalalignment='bottom')
ax1.set_ylim(-8, 3)

ax2 = subplot(3, 1, 2)

p2 = ax2.plot(time_F, drift_curl_BG_mean_F_ann/1e-8,linestyle='-', color='r')
p21 = ax2.plot(time_F, drift_curl_BG_tline_ann/1e-8,linestyle='--', color='r')

trend_ice_str = '%.1e' % drift_curl_BG_trend_ann
sig_ice_str = '%2d' % drift_curl_BG_sig_ann
ax2.annotate(trend_ice_str+' ('+sig_ice_str+'%)' , xy=(0.03, 0.1), xycoords='axes fraction', color='r',fontsize=9, horizontalalignment='left', verticalalignment='bottom')
r_str = '%.2f' % r_wi[0]
r_dcorr_str = '%.2f' % r_wi_dcorr[0]
ax2.annotate('r:'+r_str+' (r*:'+r_dcorr_str+')', xy=(0.05, -0.2), xycoords='axes fraction', color='m', fontsize=9, horizontalalignment='left', verticalalignment='bottom')
r_str = '%.2f' % r_wi[1]
r_dcorr_str = '%.2f' % r_wi_dcorr[1]
ax2.annotate('r:'+r_str+' (r*:'+r_dcorr_str+')', xy=(0.37, -0.2), xycoords='axes fraction', color='g', fontsize=9, horizontalalignment='left', verticalalignment='bottom')
r_str = '%.2f' % r_wi[2]
r_dcorr_str = '%.2f' % r_wi_dcorr[2]
ax2.annotate('r:'+r_str+' (r*:'+r_dcorr_str+')', xy=(0.68, -0.2), xycoords='axes fraction', color='b', fontsize=9, horizontalalignment='left', verticalalignment='bottom')
ax2.set_ylim(-3.5, 0.5)


ax3 = subplot(3, 1, 3)
p7 = ax3.plot(time_F, ratio[2]/1e-8,linestyle='-', color='b')

p5 = ax3.plot(time_F, ratio[0]/1e-8,linestyle='-', color='m')
#p4 = ax_temp.plot(time_F-2.5, PD.rolling_mean(ratio[0, season]/1e-8, 5),linestyle='--', linewidth=1, color='m')
p6 = ax3.plot(time_F, ratio[1]/1e-8,linestyle='-', color='g')

ax3.set_ylim(-2.1, 1.4)
		
for x in xrange(3):
	vars()['ax'+str(x+1)] = subplot(3, 1, x+1)
	ax_temp=gca()
	#ax_temp.annotate(season_strs[season] , xy=(0.03, 1.), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='bottom')
	ax_temp.axhline(0, linewidth=1,linestyle='--', color='k')
	ax_temp.set_xlim(time_F[0], time_F[-1])
	ax_temp.xaxis.grid(True, which='major')
	ax_temp.yaxis.grid(True)
	ax_temp.yaxis.set_major_locator(MaxNLocator(5))
	ax_temp.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	ax_temp.xaxis.set_major_locator(MultipleLocator(6))
	ax_temp.xaxis.set_minor_locator(MultipleLocator(1))
	ax_temp.set_xticklabels( [])



ax1.set_ylabel('Wind curl \n '+r'(10$^{-5}$m s$^{-2}$)')
ax2.set_ylabel('Drift curl \n  '+r'(10$^{-8}$m s$^{-2}$)')
ax3.set_ylabel('(Calc - Pred drift) \n '+r'(10$^{-8}$m s$^{-2}$)')

ax3.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax3.set_xlabel('Years',labelpad=-0.4)


plts_net = p11+p12+p13
reanal_str= ['NCEP-R2', 'ERA-I', 'JRA-55']
leg = ax1.legend(plts_net, reanal_str, loc=1, ncol=3,columnspacing=0.1, handletextpad=0.0001, bbox_to_anchor=(1.0, 1.5), frameon=False)
llines = leg.get_lines()
setp(llines, linewidth=2.0)
ltext  = leg.get_texts()
setp(ltext, fontsize=9)

savefig(figpath+'/wind_curl_ratios_3'+box_str+str(start_year)+'_annual_dcorr.pdf', dpi=300)
close(fig)








