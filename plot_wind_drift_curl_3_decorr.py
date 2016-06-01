############################################################## 
# Date: 01/01/16
# Name: plot_wind_drift_curl_3_dcorr.py
# Author: Alek Petty
# Description: Script to plot wind curl, drift curl lineplots (seasonal)
# Input requirements: Drift and wind curl data
# Output: Line plots of wind curl, and wind curl expected from linear relationship (seasonal)

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
rcParams['figure.figsize'] = 6.8, 6
majorLocator   = MultipleLocator(3)
majorFormatter = FormatStrFormatter('%d')
minorLocator   = MultipleLocator(1)

years = '1980-2013'
grid_str='100km'
reanals= ['NCEP2', 'ERA', 'JRA']
reanal=1
month_strs = ['J-M', 'A-J', 'J-S', 'O-D']
box_str='wbox'
#box_str='ibox'
start_year = 1980
end_year=2013

figpath='./Figures/'
datapath='./Data_output/'

product = 'FOWLER'
drift_curl_seasons_trend_F = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_trend'+grid_str+'.txt')
drift_curl_seasons_sig_F = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_sig'+grid_str+'.txt')
drift_curl_seasons_r_F = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_r'+grid_str+'.txt')
drift_curl_seasons_BG_mean_F = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_mean'+grid_str+'.txt')
drift_curl_seasons_BG_tline_F = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_tline'+grid_str+'.txt')
drift_curl_seasons_BG_trend_F= load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_trend'+grid_str+'.txt')
drift_curl_seasons_BG_sig_F = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_sig'+grid_str+'.txt')
drift_curl_seasons_BG_r_F = load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_r'+grid_str+'.txt')
drift_curl_seasons_BG_int_F= load(datapath+product+'/'+str(start_year)+'-'+str(end_year)+'drift_curl_seasons_BG_int'+grid_str+'.txt')


wind_curl_seasons_BG=ma.masked_all((3, 4, 34))
for r in range(3):
	for x in xrange(4):
		wind_path = datapath+'/WINDS/'+reanals[r]+'/WINDCURL/'+grid_str+'/'
		wind_curl_seasons_BG[r, x] = loadtxt(wind_path+years+month_strs[x]+'/ave_wind_curl_years_b'+box_str+'.txt')


#ice_curl_seasons_BG=load(drift_path+years+'drift_curl_seasons_BG_mean.txt')


wind_curl_seasons_BG=wind_curl_seasons_BG[:, :, start_year-1980::]
#ice_curl_seasons_BG=ice_curl_seasons_BG[:, start_year-1980::]
time_F= np.arange(start_year,2014, 1)

line = ma.masked_all((wind_curl_seasons_BG.shape))
trend_wi=[]
sig_wi=[]
r_wi=[]
inter_wi=[]
for r in xrange(3):
	trend, sig, r_val, inter = calc_trends(wind_curl_seasons_BG[r], drift_curl_seasons_BG_mean_F, time=0)
	trend_wi.append(trend)
	sig_wi.append(sig)
	r_wi.append(r_val)
	inter_wi.append(inter)
	for x in xrange(4):
		line[r, x] = trend_wi[r][x]*(wind_curl_seasons_BG[r, x]) + inter_wi[r][x]
		
#line_ice = ma.masked_all((ice_curl_seasons_BG.shape))
#trend_ice, sig_ice, r_ice, inter_ice = calc_trends(time_F, drift_curl_seasons_BG_meanF, time=1)
#for x in xrange(4):
#	line_ice[x] = trend_ice[x]*time_F + inter_ice[x]

line_wind = ma.masked_all((wind_curl_seasons_BG.shape))
trend_wind=[]
sig_wind=[]
r_wind=[]
inter_wind=[]


for r in xrange(3):
	trend, sig, r_val, inter = calc_trends(time_F, wind_curl_seasons_BG[r], time=1)
	trend_wind.append(trend)
	sig_wind.append(sig)
	r_wind.append(r_val)
	inter_wind.append(inter)
	for x in xrange(4):
		line_wind[r, x] = (trend_wind[r][x]*time_F) + inter_wind[r][x]

ratio = (drift_curl_seasons_BG_mean_F-line)

#DECORRELATED TRENDS
wind_curl_seasons_BG_dcorr=wind_curl_seasons_BG-line_wind

drift_curl_seasons_BG_mean_F_dcorr=drift_curl_seasons_BG_mean_F-drift_curl_seasons_BG_tline_F

r_wi_dcorr=[]
for r in xrange(3):
	trend, sig, r_val, inter = calc_trends(wind_curl_seasons_BG_dcorr[r], drift_curl_seasons_BG_mean_F_dcorr, time=0)
	r_wi_dcorr.append(r_val)


beau_region = [72., 82., -130, -170]


season_strs=['JFM', 'AMJ', 'JAS', 'OND']
region_strs=['125W-140W/80N north-south', '140W-155W/80N north-south', '155W/72-76N east-west', '155W/76-80N east-west']
plotnum_sr = ([1, 3, 5], [2, 4, 6], [7, 9, 11], [8, 10, 12])

fig = figure()
subplots_adjust( right = 0.98, left =0.1, bottom=0.06, hspace=0.25, wspace=0.05, top=0.95)
for season in xrange(4):

		plotnum = plotnum_sr[season][0]
		vars()['ax'+str(plotnum)] = subplot(6, 2, plotnum)
		ax_temp = gca()
		p13 = ax_temp.plot(time_F, wind_curl_seasons_BG[2, season]/1e-5,linestyle='-', color='b')
		p11 = ax_temp.plot(time_F, wind_curl_seasons_BG[0, season]/1e-5,linestyle='-', color='m')
		p12 = ax_temp.plot(time_F, wind_curl_seasons_BG[1, season]/1e-5,linestyle='-', color='g')

		p131 = ax_temp.plot(time_F, line_wind[2, season]/1e-5,linestyle='--', color='b')
		p111 = ax_temp.plot(time_F, line_wind[0, season]/1e-5,linestyle='--', color='m')
		p121 = ax_temp.plot(time_F, line_wind[1, season]/1e-5,linestyle='--', color='g')
		
		
		trend_wind_str = '%.1e' % trend_wind[0][season]
		sig_wind_str = '%.0f' % sig_wind[0][season]
		ax_temp.annotate(trend_wind_str+' ('+sig_wind_str+'%)' , xy=(0.12, -0.2), xycoords='axes fraction', color='m', fontsize=9, horizontalalignment='left', verticalalignment='bottom')
		trend_wind_str = '%.1e' % trend_wind[1][season]
		sig_wind_str = '%.0f' % sig_wind[1][season]
		ax_temp.annotate(''+trend_wind_str+' ('+sig_wind_str+'%)' , xy=(0.42, -0.2), xycoords='axes fraction', color='g', fontsize=9, horizontalalignment='left', verticalalignment='bottom')
		trend_wind_str = '%.1e' % trend_wind[2][season]
		sig_wind_str = '%.0f' % sig_wind[2][season]
		ax_temp.annotate(trend_wind_str+' ('+sig_wind_str+'%)' , xy=(0.715, -0.2), xycoords='axes fraction', color='b', fontsize=9, horizontalalignment='left', verticalalignment='bottom')
		ax_temp.set_ylim(-8, 3)

		plotnum = plotnum_sr[season][1]
		vars()['ax'+str(plotnum)] = subplot(6, 2, plotnum)
		ax_temp = gca()
		p2 = ax_temp.plot(time_F, drift_curl_seasons_BG_mean_F[season]/1e-8,linestyle='-', color='r')
		p21 = ax_temp.plot(time_F, drift_curl_seasons_BG_tline_F[season]/1e-8,linestyle='--', color='r')
		#p3 = ax_temp.plot(time_F, drift_curl_seasons_BG_mean_Q[season]/1e-8,linestyle='-', color='r')
		#p31 = ax_temp.plot(time_F, line_ice[season]/1e-8,linestyle='--', color='r')
		#p4 = ax_temp.plot(time_F, drift_curl_seasons_BG_mean_AS3[season]/1e-8,linestyle='-', color='r')
		#p41 = ax_temp.plot(time_F, line_ice[season]/1e-8,linestyle='--', color='r')
		#p5 = ax_temp.plot(time_F, drift_curl_seasons_BG_mean_F[season]/1e-8,linestyle='-', color='r')
		#p51 = ax_temp.plot(time_F, line_ice[season]/1e-8,linestyle='--', color='r')

		trend_ice_str = '%.1e' % drift_curl_seasons_BG_trend_F[season]
		sig_ice_str = '%2d' % drift_curl_seasons_BG_sig_F[season]
		ax_temp.annotate(trend_ice_str+' ('+sig_ice_str+'%)' , xy=(0.01, 0.1), xycoords='axes fraction', color='r',fontsize=9, horizontalalignment='left', verticalalignment='bottom')
		r_str = '%.2f' % r_wi[0][season]
		r_dcorr_str = '%.2f' % r_wi_dcorr[0][season]
		ax_temp.annotate('r:'+r_str+' (r*:'+r_dcorr_str+')', xy=(0.15, -0.2), xycoords='axes fraction', color='m', fontsize=9, horizontalalignment='left', verticalalignment='bottom')
		r_str = '%.2f' % r_wi[1][season]
		r_dcorr_str = '%.2f' % r_wi_dcorr[1][season]
		ax_temp.annotate('r:'+r_str+' (r*:'+r_dcorr_str+')', xy=(0.45, -0.2), xycoords='axes fraction', color='g', fontsize=9, horizontalalignment='left', verticalalignment='bottom')
		r_str = '%.2f' % r_wi[2][season]
		r_dcorr_str = '%.2f' % r_wi_dcorr[2][season]
		ax_temp.annotate('r:'+r_str+' (r*:'+r_dcorr_str+')', xy=(0.75, -0.2), xycoords='axes fraction', color='b', fontsize=9, horizontalalignment='left', verticalalignment='bottom')
		ax_temp.set_ylim(-3.5, 0.5)
		plotnum = plotnum_sr[season][2]

		vars()['ax'+str(plotnum)] = subplot(6, 2, plotnum)
		ax_temp = gca()
		p7 = ax_temp.plot(time_F, ratio[2, season]/1e-8,linestyle='-', color='b')
		
		p5 = ax_temp.plot(time_F, ratio[0, season]/1e-8,linestyle='-', color='m')
		#p4 = ax_temp.plot(time_F-2.5, PD.rolling_mean(ratio[0, season]/1e-8, 5),linestyle='--', linewidth=1, color='m')
		p6 = ax_temp.plot(time_F, ratio[1, season]/1e-8,linestyle='-', color='g')
		#p6 = ax_temp.plot(time_F-2.5, PD.rolling_mean(ratio[1, season]/1e-8, 5),linestyle='--', linewidth=1, color='b')
		
		#p8 = ax_temp.plot(time_F-2.5, PD.rolling_mean(ratio[2, season]/1e-8, 5),linestyle='--', linewidth=1, color='g')
		ax_temp.set_ylim(-2.1, 1.4)


		

for season in xrange(4):
	for x in xrange(3):
		plotnum = plotnum_sr[season][x]
		vars()['ax'+str(plotnum)] = subplot(6, 2, plotnum)
		ax_temp=gca()
		ax_temp.annotate(season_strs[season] , xy=(0.0, 1.), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='bottom')
		ax_temp.axhline(0, linewidth=1,linestyle='--', color='k')
		ax_temp.set_xlim(time_F[0], time_F[-1])
		ax_temp.xaxis.grid(True, which='major')
		ax_temp.yaxis.grid(True)
		ax_temp.yaxis.set_major_locator(MaxNLocator(5))
		ax_temp.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
		#if (x==2):
	#		ax_temp.yaxis.set_major_locator(MultipleLocator(7))
		ax_temp.xaxis.set_major_locator(MultipleLocator(6))
		ax_temp.xaxis.set_minor_locator(MultipleLocator(1))
		ax_temp.set_xticklabels( [])
		#if (season==1) | (season==3):
		#	ax_temp.set_yticklabels([])

		if ((season==1) or (season==3)):
			ax_temp.set_yticklabels( [])


ax1.set_ylabel('Wind curl \n '+r'(10$^{-5}$m s$^{-2}$)')
ax3.set_ylabel('Drift curl \n  '+r'(10$^{-8}$m s$^{-2}$)')
ax5.set_ylabel('(Calc - Pred drift) \n '+r'(10$^{-8}$m s$^{-2}$)')

ax7.set_ylabel('Wind curl  \n '+r'(10$^{-5}$m s$^{-2}$)')
ax9.set_ylabel('Drift curl  \n '+r'(10$^{-8}$m s$^{-2}$)')
ax11.set_ylabel('(Calc - Pred drift) \n '+r'(10$^{-8}$m s$^{-2}$)')

ax11.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax12.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax11.set_xlabel('Years')
ax12.set_xlabel('Years')

plts_net = p11+p12+p13
#labs = [l.get_label() for l in plts]
reanal_str= ['NCEP-R2', 'ERA-I', 'JRA-55']
leg = ax2.legend(plts_net, reanal_str, loc=1, ncol=3,columnspacing=0.1, handletextpad=0.0001, bbox_to_anchor=(1.02, 1.45), frameon=False)
#leg = ax1.legend(plts, labs, loc=2)
llines = leg.get_lines()
setp(llines, linewidth=2.0)
ltext  = leg.get_texts()
setp(ltext, fontsize=9)
#leg.get_frame().set_alpha(2.0)
#ax = fig.add_axes( [0., 0., 1, 1] )
#ax.set_axis_off()
#x.set_xlim(0, 1)
#ax.set_ylim(0, 1)
#plts_net = p1+p2+p3+p4+p41
#regions = ['3 day AS (CSAT)', '6 day AS (CSAT)', 'QS (CSAT)', 'NSIDC', 'NSIDC (no conc)']
#leg = ax1.legend(plts_net, regions, loc=2, ncol=5,columnspacing=0.3, handletextpad=0.0001, prop={'size':10}, bbox_to_anchor=(0.0, 1.8))

#ax.text(.02, 0.5, r'Area flux (10$^3$ km$^2$)', rotation=90,horizontalalignment='center', fontsize=11, verticalalignment='center')
savefig(figpath+'/wind_curl_ratios_3'+box_str+str(start_year)+'_dcorr.pdf', dpi=300)
close(fig)








