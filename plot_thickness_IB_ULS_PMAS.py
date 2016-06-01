############################################################## 
# Date: 01/01/16
# Name: plot_thickness_IB_ULS_PMAS.py
# Author: Alek Petty
# Description: Script to plot ULS/IB/PIOMAS BG Thickness estimates
# Input requirements: Moorings seasonal data
#                     PIOMAS BG thickness estimates
# Output: plot of BG ice thickness estimates

import numpy as np
from pylab import *
import scipy.io
from scipy.io import netcdf

mpl.rc("ytick",labelsize=9)
mpl.rc("xtick",labelsize=9)
rcParams['font.size']=10
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

figpath='./Figures/'
datapath='./Data_output/'

mean_seasons_all = load(datapath+'/mooringsBG_seasons_all.txt')
thickness_PM = load(datapath+'/ThicknessBG_PMAS.txt')
thickness_PM=thickness_PM.T

#IB results from RM&F2003, updated to 2013
ice_bridge = array([2.57, 1.52, 1.88, 1.60, 2.04])
years_ib = np.arange(2010, 2015, 1)
yearsM = np.arange(2003, 2003+10 + 1, 1)

#ice equivalent snow from Warren 1999/Rothrock 2008
snow=[0.1, 0.12, 0.02, 0.07]

years = np.arange(1980, 2013+1, 1)
colors= ['g', 'b', 'r', 'm']
month_str = ['JFM', 'AMJ', 'JAS', 'OND']

fig = figure(figsize=(6,3))
for plotnum in xrange(4):
	vars()['ax'+str(plotnum+1)] = subplot(2, 2, plotnum+1)
	ax_temp = gca()

	ax_temp.plot(years, thickness_PM[plotnum], linestyle='-',linewidth=1, color='k')

	p11 = ax_temp.plot(yearsM, mean_seasons_all[0, plotnum]*1.107 - snow[plotnum], linestyle='',marker='s', markersize='4', color=colors[plotnum])
	p12 = ax_temp.plot(yearsM, mean_seasons_all[1, plotnum]*1.107 - snow[plotnum], linestyle='',marker='o', markersize='4', color=colors[plotnum])
	p13 = ax_temp.plot(yearsM, mean_seasons_all[2, plotnum]*1.107 - snow[plotnum], linestyle='',marker='v', markersize='4', color=colors[plotnum])
	p14 = ax_temp.plot(yearsM, mean_seasons_all[3, plotnum]*1.107 - snow[plotnum], linestyle='',marker='^', markersize='4', color=colors[plotnum])

	ax_temp.fill_between(yearsM, np.amin(mean_seasons_all[:, plotnum]*1.107 - snow[plotnum], axis=0), np.amax(mean_seasons_all[:, plotnum]*1.107 - snow[plotnum], axis=0), alpha=0.2, edgecolor=colors[plotnum], facecolor=colors[plotnum])


	if (plotnum==1):
		pib = plot(years_ib, ice_bridge, linestyle='--',linewidth=2, marker='*',markersize='8', color='0.4')

	ax_temp.set_xlim(years[0], years[-1])
	ax_temp.set_xticks(np.arange(1983, 2014, 6))
	ax_temp.yaxis.grid(True)
	ax_temp.xaxis.grid(True, which='major')
	ax_temp.set_ylim(0, 4)
	ax_temp.set_yticks(np.linspace(0, 4, 5))
	ax_temp.annotate(month_str[plotnum] , xy=(0.05, 0.85), color=colors[plotnum], xycoords='axes fraction', horizontalalignment='left', verticalalignment='bottom')


ax2.set_yticklabels([])
ax4.set_yticklabels([])
ax1.set_xticklabels([])
ax2.set_xticklabels([])

ax3.set_xlabel( 'Years')
ax4.set_xlabel( 'Years')
ax1.set_ylabel( 'Thickness (m)')
ax3.set_ylabel( 'Thickness (m)')

pa = plot(0, 0, marker='s',linestyle='',color='k')
pb = plot(0, 0, marker='o',linestyle='',color='k')
pc = plot(0, 0, marker='v',linestyle='',color='k')
pd = plot(0, 0, marker='^',linestyle='',color='k')
pe = plot(0, 0, marker='*',linestyle='',color='0.4')

plts_m = pa+pb+pc+pd+pe
mooring_str = ['a', 'b', 'c', 'd', 'IB']
leg2 = ax2.legend(plts_m, mooring_str, loc=3, ncol=5,columnspacing=0.5, handlelength=1 , handletextpad=0.01, borderaxespad=0., bbox_to_anchor=(0., 0.), frameon=False, numpoints=1)
ltextm  = leg2.get_texts()
setp(ltextm, fontsize=9)
leg2.get_frame().set_alpha(0.5)

subplots_adjust( right = 0.97, left = 0.06, top=0.95, bottom=0.135, wspace=0.06, hspace=0.1)

savefig(figpath+'seasonal_BG_thickness_NTBT_ann4P_snow.pdf', dpi=300)
close(fig)


