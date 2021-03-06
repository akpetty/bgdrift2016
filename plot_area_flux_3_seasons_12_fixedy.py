############################################################## 
# Date: 01/01/16
# Name: plot_area_flux_3_seasons_12_fixedy.py
# Author: Alek Petty
# Description: Script to plot area fluxes on fixed y axes
# Input requirements: area flux estimates from FOWLER and CERSAT                 
# Output: plot of area transports through 3 flux gates in the BG

import numpy as np
from pylab import *
import scipy.io
from scipy.io import netcdf

mpl.rc("ytick",labelsize=10)
mpl.rc("xtick",labelsize=10)
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})


figpath = './Figures/'
datapath='./Data_output/FLUX_GATES/'

#LOAD AAS MASKED IN CASE NO DATA EXISTS FOR THAT TYPE
ann_area_flux_AS3 = ma.masked_all((4, 3, 6))
ann_area_flux_AS6 = ma.masked_all((4, 3, 6))
ann_area_flux_QS = ma.masked_all((4, 3, 17))
ann_area_flux_F= ma.masked_all((4, 3, 34))
ann_area_flux_F_noconc= ma.masked_all((4, 3, 34))



conc_team_s = 'nt'

for season in xrange(1,5,1):
	if season==1:
		months = [0, 1, 2]
		month_str = 'JFM'
		ann_area_flux_F[0] = load(datapath+'area_flux_F_155-125W_72-80N3_'+month_str+conc_team_s+'.txt')
		ann_area_flux_F_noconc[0] = load(datapath+'area_flux_F_155-125W_72-80N3_'+month_str+conc_team_s+'noconc.txt')
		ann_area_flux_AS3[0] = load(datapath+'/area_flux_AS3_155-125W_72-80N3_'+month_str+conc_team_s+'.txt')
		ann_area_flux_AS6[0] = load(datapath+'/area_flux_AS6_155-125W_72-80N3_'+month_str+conc_team_s+'.txt')
		ann_area_flux_QS[0] = load(datapath+'/area_flux_QS_155-125W_72-80N3_'+month_str+conc_team_s+'.txt')
	
	if season==2:
		months = [3, 4, 5]
		month_str = 'AMJ'
		ann_area_flux_F[1] = load(datapath+'/area_flux_F_155-125W_72-80N3_'+month_str+conc_team_s+'.txt')
		ann_area_flux_F_noconc[1] = load(datapath+'/area_flux_F_155-125W_72-80N3_'+month_str+conc_team_s+'noconc.txt')

	if season==3:
		months = [6, 7, 8]
		month_str = 'JAS'
		ann_area_flux_F[2] = load(datapath+'/area_flux_F_155-125W_72-80N3_'+month_str+conc_team_s+'.txt')
		ann_area_flux_F_noconc[2] = load(datapath+'/area_flux_F_155-125W_72-80N3_'+month_str+conc_team_s+'noconc.txt')

	if season==4:
		months = [9, 10, 11]
		month_str = 'OND'
		ann_area_flux_F[3] = load(datapath+'/area_flux_F_155-125W_72-80N3_'+month_str+conc_team_s+'.txt')
		ann_area_flux_F_noconc[3] = load(datapath+'/area_flux_F_155-125W_72-80N3_'+month_str+conc_team_s+'noconc.txt')
		ann_area_flux_AS3[3] = load(datapath+'/area_flux_AS3_155-125W_72-80N3_'+month_str+conc_team_s+'.txt')
		ann_area_flux_AS6[3] = load(datapath+'/area_flux_AS6_155-125W_72-80N3_'+month_str+conc_team_s+'.txt')
		ann_area_flux_QS[3] = load(datapath+'/area_flux_QS_155-125W_72-80N3_'+month_str+conc_team_s+'.txt')

time_AS= np.linspace(2008,2013, 6)
time_QS= np.linspace(1992,2008, 17)
time_F= np.linspace(1980,2013, 34)

sd = np.std(ann_area_flux_F, axis=2)
mean = np.mean(ann_area_flux_F, axis=2)

season_strs=['JFM', 'AMJ', 'JAS', 'OND']
region_strs=['North-south transport', 'East-west transport (south)', 'East-west transport (north)']
axesname = ['ax1', 'ax2', 'ax3', 'ax4', 'ax5', 'ax6', 'ax7', 'ax8', 'ax']

fig = figure(figsize=(8,7))
subplots_adjust( right = 0.992, left = 0.05, bottom=0.055, hspace=0.15, wspace=0.08)

#plotnum_sr = ([1, 3, 5, 7], [2, 4, 6, 8], [9, 11, 13, 15], [10, 12, 14, 16])
plotnum=0
for season in xrange(4):
	for region in xrange(3):
	
		plotnum +=1
		#print plotnum
		vars()['ax'+str(plotnum)] = subplot(4, 3, plotnum)

		p1 = vars()['ax'+str(plotnum)].plot(time_AS, ann_area_flux_AS3[season, region]/100,linestyle='-',marker='x', color='b', markersize=4, zorder = 3)
		p2 = vars()['ax'+str(plotnum)].plot(time_AS, ann_area_flux_AS6[season, region]/100,linestyle='-',marker='x', color='r', markersize=4, zorder = 3)
		p3 = vars()['ax'+str(plotnum)].plot(time_QS, ann_area_flux_QS[season, region]/100,linestyle='-',marker='x', color='g', markersize=4, zorder = 3)
		p4 = vars()['ax'+str(plotnum)].plot(time_F, ann_area_flux_F[season, region]/100,linestyle='-',marker='x', color='m', markersize=4, zorder = 3)
		p41 = vars()['ax'+str(plotnum)].plot(time_F, ann_area_flux_F_noconc[season, region]/100,linestyle='--',marker='x', color='m', markersize=4, zorder = 3)
		ax_temp = gca()

		xmin = time_F[0]
		xmax = time_F[-1]

		ax_temp.axhline(mean[season, region]/100, linewidth=1, color='k')
		ax_temp.axhspan(mean[season, region]/100-(2*sd[season, region]/100),mean[season, region]/100+(2*sd[season, region]/100), facecolor='0.8', linewidth=0.1, zorder = 1)
		ax_temp.axhspan(mean[season, region]/100-sd[season, region]/100,mean[season, region]/100+sd[season, region]/100, facecolor='0.6', linewidth=0.1, zorder = 2)

		ax_temp.set_xlim(xmin, xmax)
		ax_temp.set_ylim(-2, 4)
		#ax_temp.yaxis.grid(True)
		ax_temp.xaxis.grid(True, which='major')
		
		ax_temp.yaxis.set_major_locator(MaxNLocator(7))
		ax_temp.xaxis.set_major_locator(MultipleLocator(6))
		ax_temp.xaxis.set_minor_locator(MultipleLocator(1))
		ax_temp.set_xticklabels( [])
		colors = ['r', 'g', 'm']
		#if (region==0):
		ax_temp.annotate(season_strs[season] , xy=(0.03, 0.85), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='bottom')
		if (season==0):
			ax_temp.annotate(region_strs[region] , xy=(0.2, 1.05), xycoords='axes fraction', fontsize=11, horizontalalignment='left', verticalalignment='bottom')
		if (region>0):
			ax_temp.set_yticklabels( [])

ax1.annotate('Import' , xy=(0.04, 0.48), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='bottom')
ax1.annotate('Export' , xy=(0.04, 0.15), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='bottom')
ax2.annotate('Import' , xy=(0.04, 0.13), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='bottom')
ax2.annotate('Export' , xy=(0.04, 0.71), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='bottom')
ax3.annotate('Import' , xy=(0.04, 0.1), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='bottom')
ax3.annotate('Export' , xy=(0.04, 0.44), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='bottom')

ax10.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax11.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax12.xaxis.set_major_formatter(FormatStrFormatter('%d'))
#ax10.set_xlabel('Years', fontsize=10)
ax11.set_xlabel('Years', fontsize=10)


ax = fig.add_axes( [0., 0., 1, 1] )
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plts_net = p4+p41+p3+p1+p2
regions = [  'NSIDC/PP', 'NSIDC/PP-NOCONC','CSAT/QS','CSAT/AS-3', 'CSAT/AS-6']
leg = ax1.legend(plts_net, regions, loc=2, ncol=5,columnspacing=1., handletextpad=0.001, prop={'size':11}, bbox_to_anchor=(0.06, 1.425), frameon=False)

ax.text(.015, 0.5, r'Area transport (10$^5$ km$^2$)', rotation=90,horizontalalignment='center', fontsize=11, verticalalignment='center')

savefig(figpath+'/area_flux_season_3_all_seasons_fixy'+conc_team_s+'.pdf', dpi=300)
close(fig)




