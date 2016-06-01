############################################################## 
# Date: 01/01/16
# Name: plot_pressure_concthickFILL.py
# Author: Alek Petty
# Description: Script to plot BG ice strength using NASA Team/Bootstrap data
# Output: lineplot of seasonal BG ice strength (variable conc/thickness and both)


import numpy as np
from pylab import *
import scipy.io
from scipy.io import netcdf
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize']=9
rcParams['ytick.labelsize']=9
rcParams['legend.fontsize']=9
rcParams['font.size']=9
rc('font',**{'family':'sans-serif','sans-serif':['Arial']})


C=20.
alg = 'BOOTSTRAP'
#alg = 'NASA_TEAM'

figpath='./Figures/'
datapath='./Data_output/'

Thickness_BG = load(datapath+'/ThicknessBG_PMAS.txt')
Ice_conc_BGNT = load(datapath+'/IceconcBG_NASA_TEAM.txt')
Ice_conc_BGBT = load(datapath+'/IceconcBG_BOOTSTRAP.txt')
#Pressure_BG = load('/Users/aapetty/DATA_OUTPUT/PIOMAS/Pressure_BG_fromseasons.txt')

Pressure_BGNT=(Thickness_BG*exp(-C*(1-Ice_conc_BGNT)))
Pressure_BGBT=(Thickness_BG*exp(-C*(1-Ice_conc_BGBT)))

Ice_conc_BG_FIXNT=ma.masked_all((Ice_conc_BGNT.shape))
Ice_conc_BG_FIXBT=ma.masked_all((Ice_conc_BGBT.shape))
Thickness_BG_FIX=ma.masked_all((Thickness_BG.shape))


for x in xrange(34):
	Ice_conc_BG_FIXNT[x, :] = np.mean(Ice_conc_BGNT[0:10], axis=0)
	Ice_conc_BG_FIXBT[x, :] = np.mean(Ice_conc_BGBT[0:10], axis=0)
	Thickness_BG_FIX[x, :] = np.mean(Thickness_BG[0:10], axis=0)

Pressure_FBG_FIXCONCNT=Thickness_BG*exp(-C*(1-Ice_conc_BG_FIXNT))
Pressure_FBG_FIXTHICKNT=Thickness_BG_FIX*exp(-C*(1-Ice_conc_BGNT))
Pressure_FBG_FIXCONCBT=Thickness_BG*exp(-C*(1-Ice_conc_BG_FIXBT))
Pressure_FBG_FIXTHICKBT=Thickness_BG_FIX*exp(-C*(1-Ice_conc_BGBT))


maxval=2
var_label='pressure'

years = np.arange(1980, 2013+1, 1)
colors= ['g', 'b', 'r', 'm']
month_str = ['JFM', 'AMJ', 'JAS', 'OND']

fig = figure(figsize=(6,3))
for plotnum in xrange(4):
	vars()['ax'+str(plotnum+1)] = subplot(2, 2, plotnum+1)
	#print vars()['ax'+str(plotnum+1)]
	ax_temp = gca()
	
	ax_temp.fill_between(years, Pressure_FBG_FIXCONCNT.T[plotnum], Pressure_FBG_FIXCONCBT.T[plotnum], alpha=0.3, edgecolor='none', facecolor='y', zorder=1)
	ax_temp.fill_between(years, Pressure_FBG_FIXTHICKNT.T[plotnum], Pressure_FBG_FIXTHICKBT.T[plotnum], alpha=0.3, edgecolor='none', facecolor='g', zorder=1)
	
	p2 = ax_temp.plot(years, Pressure_FBG_FIXCONCNT.T[plotnum], linestyle='-',linewidth=1, color='y', zorder=1)
	p3 = ax_temp.plot(years, Pressure_FBG_FIXTHICKNT.T[plotnum], linestyle='-',linewidth=1, color='g', zorder=1)

	p21 = ax_temp.plot(years, Pressure_FBG_FIXCONCBT.T[plotnum], linestyle='--',linewidth=1, color='y', zorder=1)
	p31 = ax_temp.plot(years, Pressure_FBG_FIXTHICKBT.T[plotnum], linestyle='--',linewidth=1, color='g', zorder=1)

	p1 = ax_temp.plot(years, Pressure_BGNT.T[plotnum], linestyle='-',linewidth=1.2, color='r', zorder=2)
	p11 = ax_temp.plot(years, Pressure_BGBT.T[plotnum], linestyle='--',linewidth=1.2, color='r', zorder=2)
	ax_temp.fill_between(years, Pressure_BGNT.T[plotnum], Pressure_BGBT.T[plotnum], alpha=0.9, edgecolor='r', facecolor='r', zorder=2)
	
	if (plotnum==2):
		plts_net = p1+p2+p3
	ax_temp.set_xlim(years[0], years[-1])
	ax_temp.set_ylim(0, maxval)
	ax_temp.set_xticks(np.arange(1983, 2014, 6))
	ax_temp.yaxis.grid(True)
	ax_temp.xaxis.grid(True, which='major')
	ax_temp.annotate(month_str[plotnum] , xy=(0.84, 0.84), color='k', xycoords='axes fraction', horizontalalignment='left', verticalalignment='bottom')

ax2.set_yticklabels([])
ax4.set_yticklabels([])
ax1.set_xticklabels([])
ax2.set_xticklabels([])

ax3.set_xlabel( 'Years')
ax4.set_xlabel( 'Years')
ax1.set_ylabel( r'$P/P^{\star}$')
ax3.set_ylabel(  r'$P/P^{\star}$')


labels=['variable concentration\nand thickness', 'variable thickness\nconstant concentration', 'variable concentration\nconstant thickness', ]
#labs = [l.get_label() for l in plts]

leg = ax3.legend(plts_net, labels, loc=2, ncol=1,columnspacing=0.1, handletextpad=0.0001, borderaxespad=0.0, frameon=False)
#leg = ax1.legend(plts, labs, loc=2)
llines = leg.get_lines()
setp(llines, linewidth=2.0)
#ltext  = leg.get_texts()
#setp(ltext, fontsize='small')

subplots_adjust( right = 0.96, left = 0.08, top=0.97, bottom=0.135, wspace=0.06, hspace=0.13)

savefig(figpath+'/seasonal_BG_'+var_label+'fixconcthick2ALG.pdf', dpi=300)
savefig(figpath+'/seasonal_BG_'+var_label+'fixconcthick2ALG.jpg', dpi=200)
close(fig)
