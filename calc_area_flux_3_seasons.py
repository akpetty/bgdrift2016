############################################################## 
# Date: 01/01/16
# Name: calc_area_flux_3_seasons.py
# Author: Alek Petty
# Description: Script to calculate transport through three flux gates
# Input requirements: CERSAT and FOWLER daily drift data (vectors in x/y direction) and projection lat/lon
#                     Also needs some functions in BG_functions
# Output: Seasonal ice area transport estimates
# Notes: Need to make sure CALCF etc are set to 1 to calculate for all the different drift products

import BG_functions as BGF
from mpl_toolkits.basemap import Basemap, shiftgrid
import numpy as np
from pylab import *
import numpy.ma as ma
from scipy.interpolate import griddata
from netCDF4 import Dataset
from glob import glob
import os


def get_dist(lon1,lon2,lat1,lat2): 
	# function to compute great circle distance between point lat1 and lon1 and arrays of points given by lons, lats 
     # great circle distance. in KM (kilometers)
	R = 6371.
	lon1r = radians(lon1)
	lon2r = radians(lon2)
	lat1r = radians(lat1)
	lat2r = radians(lat2)
	dlonr = radians(abs(lon2 - lon1))
	dlatr = radians(abs(lat2 - lat1))

	a = (sin(dlatr/2)**2) + cos(lat1)*cos(lat2)*sin(dlonr/2)**2
	c = 2*arctan2(sqrt(a), sqrt(1-a))
	d = abs(R*c)
	return d

def find_flux(drift_uv, xpts, ypts, startlon, endlon, startlat, endlat, numpts, uv=0, conc_int=1):

	num_years = drift_uv.shape[0]
	if uv==0:
		#constant lon - zonal vectors
		#numpts = numpts
		dlon = abs(endlon-startlon)/float(numpts)
		dlat = abs(endlat-startlat)/float(numpts)
		lons2 = np.linspace(startlon, startlon, float(numpts))
		lats2 = np.linspace(startlat, endlat, float(numpts))
		distances = get_dist(lons2,lons2,lats2 - dlat/2,lats2 + dlat/2)
	if uv==1:
		#constant lat - meridional vectors
		dlon = abs(endlon-startlon)/float(numpts)
		dlat = abs(endlat-startlat)/float(numpts)
		lons2 = np.linspace(startlon, endlon, float(numpts))
		lats2 = np.linspace(endlat, endlat, float(numpts))
		distances = get_dist(lons2-(dlon/2.),lons2+(dlon/2.),lats2,lats2)
	
	xpts2, ypts2 = m(lons2, lats2)	
	ann_area_flux = []
	area_flux_months = []
	i=0
	for x in xrange(num_years):
		
		for y in xrange(num_months):

			data = ma.getdata(drift_uv[x, y, uv])
			mask = ma.getmask(drift_uv[x, y, uv])
			index = np.where(mask==False)
			if (size(index)>0):
				data_masked = data[index]
				xpoints = xpts[index]
				ypoints = ypts[index]
				points = (xpoints, ypoints)
				zi = griddata(points,data_masked,(xpts2,ypts2),method='linear')
				zi_ma = ma.masked_array(zi,np.isnan(zi))

				if (conc_int==1):
					conc_data = ma.getdata(ice_conc_season[x, y])
					conc_mask = ma.getmask(ice_conc_season[x, y])
					conc_index = np.where(conc_mask==False)
					conc_ma = conc_data[conc_index]
					points_ic = (xpts_ic[conc_index], ypts_ic[conc_index])
					conc_i = griddata(points_ic,conc_ma,(xpts2,ypts2),method='linear')
					conc_i_ma = ma.masked_array(conc_i,np.isnan(conc_i))
					#1000 converts zi in m to km then flux in km^2 to 10^3km^2
				
					flux = (np.sum(zi_ma*conc_i_ma*distances)/(1000.**2))*(60.*60.*24.*30.)
				else:
					flux = (np.sum(zi_ma*distances)/(1000.**2))*(60.*60.*24.*30.)
				area_flux_months.append(flux)
			else:
				area_flux_months.append(ma.masked_all((1)))

			#plot_drift(drift_uv[x, y, uv], zi, x, y, str(uv), xpts2, ypts2)
		#print area_flux	
		area_flux_months_ma = ma.masked_array(area_flux_months,np.isnan(area_flux_months))
		ann_area_flux.append(np.sum(area_flux_months_ma[(x*num_months):(x*num_months)+num_months]))

		#print ann_area_flux

	return ann_area_flux, np.sum(distances)

def plot_seasonal_flux_F():
	fig = figure(figsize=(6,6))
	subplots_adjust( right = 0.99, left = 0.1, top=0.98, bottom=0.04, hspace=0.08)


	time_AS= np.linspace(2008,2012, 6)
	time_QS= np.linspace(1992,2008, 17)
	time_F= np.linspace(1980,2012, 33)

	#time_year_AS = np.arange(4,len(area_flux[0]), 7)

	ax1 = subplot(4,1,1)
	p4 = ax1.plot(time_F, ann_area_flux_F[0],linestyle='-',marker='x', color='m', markersize=4)

	#ax1.plot(time_year, ann_area_flux[0],linestyle='-',marker='x', color='r', markersize=5)
	xmin = time_F[0]
	xmax = time_F[-1]
	ax1.set_xlim(xmin, xmax)
	ax1.set_xticklabels( [])
	ax1.xaxis.set_major_locator(FixedLocator(np.arange(time_F[0], time_F[-1]+1, 2)))

	#ax1.text(xmin+0.5, 110, '(a) 125W-140W (80N) North-South')
	ax1.annotate('(a) 125W-140W (80N) North-South', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12, horizontalalignment='left', verticalalignment='bottom')
	
	#ax1.text(xmin+0.5, 110, '(a) 150W-120W (80N) North-South ('+month_str+' Net)')

	#ax1.set_ylim(-50, 200)

	#ax1.set_ylabel(r'Area flux (10$^3$ km$^2$)')
	ax1.yaxis.grid(True)
	ax1.xaxis.grid(True)

	ax2 = subplot(4,1,2)

	#p5 = ax2.plot(time_AS, ann_area_flux_AS[1],linestyle='-',marker='x', color='b', markersize=4)
	#p6 = ax2.plot(time_AS, ann_area_flux_AS[3],linestyle='-',marker='x', color='r', markersize=4)
	#p7 = ax2.plot(time_QS, ann_area_flux_QS[1],linestyle='-',marker='x', color='g', markersize=4)
	p8 = ax2.plot(time_F, ann_area_flux_F[1],linestyle='-',marker='x', color='m', markersize=4)
	#ax2.plot(time_year, ann_area_flux[1],linestyle='-',marker='x', color='r', markersize=5)

	ax2.set_xlim(xmin, xmax)
	ax2.set_xticklabels( [])
	ax2.xaxis.set_major_locator(FixedLocator(np.arange(time_F[0], time_AS[-1]+1, 2)))

	ax2.annotate('(b) 125W-140W (80N) North-South', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12, horizontalalignment='left', verticalalignment='bottom')
	#ax2.text(xmin+0.5, 1000, '(b) 125W-140W (80N) North-South')

	#ax2.set_ylim(-100, 1200)
	#ax2.set_ylabel(r'Area flux (10$^3$ km$^2$)')

	ax2.yaxis.grid(True)
	ax2.xaxis.grid(True)

	ax3 = subplot(4,1,3)

	#p9 = ax3.plot(time_AS, (ann_area_flux_AS[1]-ann_area_flux_AS[0])*1000./area_div,linestyle='-',marker='x', color='b', markersize=4)
	#p10 = ax3.plot(time_AS, (ann_area_flux_AS[3]-ann_area_flux_AS[2])*1000./area_div,linestyle='-',marker='x', color='r', markersize=4)
	#p11 = ax3.plot(time_QS, (ann_area_flux_QS[1]-ann_area_flux_QS[0])*1000./area_div,linestyle='-',marker='x', color='g', markersize=4)
	p12 = ax3.plot(time_F, ann_area_flux_F[2],linestyle='-',marker='x', color='m', markersize=4)

	ax3.set_xlim(xmin, xmax)
	ax3.set_xticklabels([] )
	ax3.xaxis.set_major_locator(FixedLocator(np.arange(time_F[0], time_AS[-1]+1, 2)))

	#ax3.text(xmin+0.5, 1000, '(c) 72-76N (155W) East-West' )
	ax3.annotate('(c) 72-76N (155W) East-West' , xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12, horizontalalignment='left', verticalalignment='bottom')
	#ax3.set_ylim(-1, 1)
	#ax3.set_ylabel(r'Divergence (conc/seasonal period)')
	ax3.yaxis.grid(True)
	ax3.xaxis.grid(True)

	ax4 = subplot(4,1,4)

	#p9 = ax3.plot(time_AS, (ann_area_flux_AS[1]-ann_area_flux_AS[0])*1000./area_div,linestyle='-',marker='x', color='b', markersize=4)
	#10 = ax3.plot(time_AS, (ann_area_flux_AS[3]-ann_area_flux_AS[2])*1000./area_div,linestyle='-',marker='x', color='r', markersize=4)
	#p11 = ax3.plot(time_QS, (ann_area_flux_QS[1]-ann_area_flux_QS[0])*1000./area_div,linestyle='-',marker='x', color='g', markersize=4)
	p12 = ax4.plot(time_F,ann_area_flux_F[3],linestyle='-',marker='x', color='m', markersize=4)

	ax4.set_xlim(xmin, xmax)
	ax4.set_xticklabels( ['80','82','84','86','88','90','92','94', '96','98', '00','02', '04','06','08','10','12'])
	ax4.xaxis.set_major_locator(FixedLocator(np.arange(time_F[0], time_F[-1]+1, 2)))

	#ax4.text(xmin+0.5, 1000, '(d) 76-80N (155W) East-West' )
	ax4.annotate('(d) 76-80N (155W) East-West' , xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12, horizontalalignment='left', verticalalignment='bottom')
	#ax3.set_ylim(-1, 1)
	#ax3.set_ylabel(r'Divergence (conc/seasonal period)')
	ax4.yaxis.grid(True)
	ax4.xaxis.grid(True)

	ax = fig.add_axes( [0., 0., 1, 1] )
	ax.set_axis_off()
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	#plts_net = p1+p2+p3+p4
	#regions = ['3 day AS (CSAT)', '6 day AS (CSAT)', 'QS (CSAT)', 'NSIDC']
	#leg = ax1.legend(plts_net, regions, loc=2, ncol=4,columnspacing=0.3, handletextpad=0.0001)
	ax.text(.02, 0.5, r'Area flux (10$^3$ km$^2$)', rotation=90,horizontalalignment='center', size='medium', verticalalignment='center')
	savefig(outpath+'/area_flux_season_quads'+month_str+'.pdf', dpi=300)
	close(fig)

def plot_seasonal_flux_all():
	fig = figure(figsize=(6,6))
	subplots_adjust( right = 0.99, left = 0.1, top=0.98, bottom=0.04, hspace=0.08)


	time_AS= np.linspace(2008,2013, 7)
	time_QS= np.linspace(1992,2008, 17)
	time_F= np.linspace(1980,2012, 33)

	#time_year_AS = np.arange(4,len(area_flux[0]), 7)

	ax1 = subplot(4,1,1)
	p1 = ax1.plot(time_AS, ann_area_flux_AS3[0],linestyle='-',marker='x', color='b', markersize=4)
	p2 = ax1.plot(time_AS, ann_area_flux_AS6[0],linestyle='-',marker='x', color='r', markersize=4)
	p3 = ax1.plot(time_QS, ann_area_flux_QS[0],linestyle='-',marker='x', color='g', markersize=4)
	p4 = ax1.plot(time_F, ann_area_flux_F[0],linestyle='-',marker='x', color='m', markersize=4)

	#ax1.plot(time_year, ann_area_flux[0],linestyle='-',marker='x', color='r', markersize=5)
	xmin = time_F[0]
	xmax = time_AS[-1]
	ax1.set_xlim(xmin, xmax)
	ax1.set_xticklabels( [])
	ax1.xaxis.set_major_locator(FixedLocator(np.arange(time_F[0], time_AS[-1]+1, 2)))


	ax1.annotate('(a) 125W-140W (80N) North-South', xy=(0.03, 0.63), xycoords='axes fraction', fontsize=12, horizontalalignment='left', verticalalignment='bottom')
	
	#ax1.text(xmin+0.5, 110, '(a) 150W-120W (80N) North-South ('+month_str+' Net)')

	#ax1.set_ylim(-50, 200)

	#ax1.set_ylabel(r'Area flux (10$^3$ km$^2$)')
	ax1.yaxis.grid(True)
	ax1.xaxis.grid(True)

	ax2 = subplot(4,1,2)

	p5 = ax2.plot(time_AS, ann_area_flux_AS3[1],linestyle='-',marker='x', color='b', markersize=4)
	p6 = ax2.plot(time_AS, ann_area_flux_AS6[1],linestyle='-',marker='x', color='r', markersize=4)
	p7 = ax2.plot(time_QS, ann_area_flux_QS[1],linestyle='-',marker='x', color='g', markersize=4)
	p8 = ax2.plot(time_F, ann_area_flux_F[1],linestyle='-',marker='x', color='m', markersize=4)
	#ax2.plot(time_year, ann_area_flux[1],linestyle='-',marker='x', color='r', markersize=5)

	ax2.set_xlim(xmin, xmax)
	ax2.set_xticklabels( [])
	ax2.xaxis.set_major_locator(FixedLocator(np.arange(time_F[0], time_AS[-1]+1, 2)))

	ax2.annotate('(b) 125W-140W (80N) North-South', xy=(0.03, 0.85), xycoords='axes fraction', fontsize=12, horizontalalignment='left', verticalalignment='bottom')
	#ax2.text(xmin+0.5, 1000, '(b) 125W-140W (80N) North-South')

	#ax2.set_ylim(-100, 1200)
	#ax2.set_ylabel(r'Area flux (10$^3$ km$^2$)')

	ax2.yaxis.grid(True)
	ax2.xaxis.grid(True)

	ax3 = subplot(4,1,3)

	p9 = ax3.plot(time_AS, ann_area_flux_AS3[2],linestyle='-',marker='x', color='b', markersize=4)
	p10 = ax3.plot(time_AS, ann_area_flux_AS6[2],linestyle='-',marker='x', color='r', markersize=4)
	p11 = ax3.plot(time_QS, ann_area_flux_QS[2],linestyle='-',marker='x', color='g', markersize=4)
	p12 = ax3.plot(time_F, ann_area_flux_F[2],linestyle='-',marker='x', color='m', markersize=4)

	ax3.set_xlim(xmin, xmax)
	ax3.set_xticklabels([] )
	ax3.xaxis.set_major_locator(FixedLocator(np.arange(time_F[0], time_AS[-1]+1, 2)))

	#ax3.text(xmin+0.5, 1000, '(c) 72-76N (155W) East-West' )
	ax3.annotate('(c) 72-76N (155W) East-West' , xy=(0.03, 0.85), xycoords='axes fraction', fontsize=12, horizontalalignment='left', verticalalignment='bottom')
	#ax3.set_ylim(-1, 1)
	#ax3.set_ylabel(r'Divergence (conc/seasonal period)')
	ax3.yaxis.grid(True)
	ax3.xaxis.grid(True)

	ax4 = subplot(4,1,4)

	p13 = ax4.plot(time_AS, ann_area_flux_AS3[3],linestyle='-',marker='x', color='b', markersize=4)
	p14 = ax4.plot(time_AS, ann_area_flux_AS6[3],linestyle='-',marker='x', color='r', markersize=4)
	p15 = ax4.plot(time_QS, ann_area_flux_QS[3],linestyle='-',marker='x', color='g', markersize=4)
	p16 = ax4.plot(time_F,ann_area_flux_F[3],linestyle='-',marker='x', color='m', markersize=4)

	ax4.set_xlim(xmin, xmax)
	ax4.set_xticklabels( ['80','82','84','86','88','90','92','94', '96','98', '00','02', '04','06','08','10','12'])
	ax4.xaxis.set_major_locator(FixedLocator(np.arange(time_F[0], time_AS[-1]+1, 2)))

	#ax4.text(xmin+0.5, 1000, '(d) 76-80N (155W) East-West' )
	ax4.annotate('(d) 76-80N (155W) East-West' , xy=(0.03, 0.85), xycoords='axes fraction', fontsize=12, horizontalalignment='left', verticalalignment='bottom')
	#ax3.set_ylim(-1, 1)
	#ax3.set_ylabel(r'Divergence (conc/seasonal period)')
	ax4.yaxis.grid(True)
	ax4.xaxis.grid(True)

	ax = fig.add_axes( [0., 0., 1, 1] )
	ax.set_axis_off()
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	plts_net = p1+p2+p3+p4
	regions = ['3 day AS (CSAT)', '6 day AS (CSAT)', 'QS (CSAT)', 'NSIDC']
	leg = ax1.legend(plts_net, regions, loc=2, ncol=4,columnspacing=0.3, handletextpad=0.0001, prop={'size':10})

	ax.text(.02, 0.5, r'Area flux (10$^3$ km$^2$)', rotation=90,horizontalalignment='center', size='medium', verticalalignment='center')
	savefig(outpath+'/area_flux_season_quads_all'+month_str+'.pdf', dpi=300)
	close(fig)


m = Basemap(projection='npstere',boundinglat=58,lon_0=0, resolution='l'  )

figpath='./Figures/'
datapath='./Data_output/'
gatepath = datapath+'/FLUX_GATES'
rawdatapath='../../DATA/'

if not os.path.exists(gatepath):
	os.makedirs(gatepath)

lon1 = -125.
lon2 = -140.
lon3 = -155.
lat1 = 72.
lat2 = 76.
lat3 = 80.

region = str(int(-lon3))+'-'+str(int(-lon1))+'W'+'_'+str(int(lat1))+'-'+str(int(lat3))+'N'
# CHOOSE HOW MANY GRID POINTS TO INTERPOALTE ONTO THE GATES
numpts = 20

###### GET ICE CONC FOR YEARS AND MONTHS OF INTEREST ############
conc_team = 'NASA_TEAM'
conc_team_s = 'nt'
start_year=1980
end_year=2013
num_years = end_year-start_year+1
years_str=str(start_year)+'-'+str(end_year)

ice_conc_ann = load(datapath+'/ice_conc_months-'+years_str+conc_team+'.txt')
lats_ic = load(datapath+'/ice_conc_lats'+conc_team+'.txt')
lons_ic = load(datapath+'/ice_conc_lons'+conc_team+'.txt')
xpts_ic, ypts_ic = m(lons_ic, lats_ic)
############################################################

xptsC, yptsC = BGF.return_xpts_ypts(rawdatapath+'/ICE_DRIFT/CERSAT/DRIFT_ASCAT/3DAY/', m)
xptsF, yptsF = BGF.return_xpts_ypts_fowler(rawdatapath+'/ICE_DRIFT/FOWLER/DATA/', m)

#BGF.plot_gates_map(lon1, lon2,lon3, lat1, lat2, lat3)


CALCF=0
CALCFNOCONC=0
CALCFNOCONC2014=0
CALCF2014=0
CALCAS6=0
CALCAS3=0
CALCQS=1

for season in xrange(4):

	if season==0:
		months = [0, 1, 2]
		month_str = 'JFM'
	if season==1:
		months = [3, 4, 5]
		month_str = 'AMJ'
	if season==2:
		months = [6, 7, 8]
		month_str = 'JAS'
	if season==3:
		months = [9, 10, 11]
		month_str = 'OND'
	num_months = len(months)
	

	ann_area_flux_F = ma.masked_all((3, 34))
	ann_area_flux_F2014 = ma.masked_all((3, 35))
	ann_area_flux_F2014NOCONC = ma.masked_all((3, 35))
	ann_area_flux_AS3 = ma.masked_all((3, 6))
	ann_area_flux_AS6 = ma.masked_all((3, 6))
	ann_area_flux_QS = ma.masked_all((3, 17))


	if (CALCF==1):
		############################################################
		###### GET FOWLER DATA FOR YEARS AND MONTHS OF INTEREST ############

		#ALEK CALCULATED MONTHLY MEANS
		drift_F= load(datapath+'/FOWLER/1980-2013-drift_data_months_uv.txt')
		#drift_F= load('/Users/aapetty/NOAA/FOWLER/1980-2012-S-A-drift_data_months_means_uv.txt')
		#CHANGE E +VE TO W +VE 
		drift_F = -drift_F
		#FOWLER CALCULATED MONTHLY MEANS
		ice_conc_season = ice_conc_ann[:, months[0]:months[-1]+1]


		ann_area_flux_F[0], distances = find_flux(drift_F[:, months[0]:months[-1]+1], xptsF, yptsF, lon3, lon1, lat3, lat3, numpts, uv=1, conc_int=1)
		ann_area_flux_F[1], distances = find_flux(drift_F[:, months[0]:months[-1]+1], xptsF, yptsF, lon3, lon3, lat1, lat2, numpts, uv=0, conc_int=1)
		ann_area_flux_F[2], distances = find_flux(drift_F[:, months[0]:months[-1]+1], xptsF, yptsF, lon3, lon3, lat2, lat3, numpts, uv=0, conc_int=1)

		ann_area_flux_F.dump(gatepath+'/area_flux_F_'+region+'3_'+month_str+conc_team_s+'.txt')
		############################################################
		############################################################

	if (CALCFNOCONC==1):
		############################################################
		###### GET FOWLER DATA FOR YEARS AND MONTHS OF INTEREST - NO CONCENTRATION INTERPOLATION############

		#ALEK CALCULATED MONTHLY MEANS
		drift_F= load(datapath+'/FOWLER/1980-2013-drift_data_months_uv.txt')
		#drift_F= load('/Users/aapetty/NOAA/FOWLER/1980-2012-S-A-drift_data_months_means_uv.txt')
		#CHANGE E +VE TO W +VE 
		drift_F = -drift_F
		#FOWLER CALCULATED MONTHLY MEANS
		ice_conc_season = ice_conc_ann[:, months[0]:months[-1]+1]

		ann_area_flux_F[0], gate_length1F = find_flux(drift_F[:, months[0]:months[-1]+1], xptsF, yptsF, lon3, lon1, lat3, lat3, numpts, uv=1, conc_int=0)
		ann_area_flux_F[1], gate_length2F = find_flux(drift_F[:, months[0]:months[-1]+1], xptsF, yptsF, lon3, lon3, lat1, lat2, numpts, uv=0, conc_int=0)
		ann_area_flux_F[2], gate_length3F = find_flux(drift_F[:, months[0]:months[-1]+1], xptsF, yptsF, lon3, lon3, lat2, lat3, numpts, uv=0, conc_int=0)

		ann_area_flux_F.dump(gatepath+'/area_flux_F_'+region+'3_'+month_str+conc_team_s+'noconc.txt')
		############################################################
		############################################################
		gate_lengths = [gate_length1F, gate_length2F, gate_length3F]
		savetxt(gatepath+'/gats_lengthsF.txt', gate_lengths)

	if (CALCF2014==1):
		############################################################
		###### GET FOWLER DATA FOR YEARS AND MONTHS OF INTEREST - NO CONCENTRATION INTERPOLATION############

		#ALEK CALCULATED MONTHLY MEANS
		drift_F= load(datapath+'/FOWLER/1980-2014-drift_data_months_uv.txt')
		#drift_F= load('/Users/aapetty/NOAA/FOWLER/1980-2012-S-A-drift_data_months_means_uv.txt')
		#CHANGE E +VE TO W +VE 
		drift_F = -drift_F
		#FOWLER CALCULATED MONTHLY MEANS

		ice_conc_season = ice_conc_ann[:, months[0]:months[-1]+1]

		ann_area_flux_F2014[0], gate_length1F = find_flux(drift_F[:, months[0]:months[-1]+1], xptsF, yptsF, lon3, lon1, lat3, lat3, numpts, uv=1, conc_int=1)
		ann_area_flux_F2014[1], gate_length2F = find_flux(drift_F[:, months[0]:months[-1]+1], xptsF, yptsF, lon3, lon3, lat1, lat2, numpts, uv=0, conc_int=1)
		ann_area_flux_F2014[2], gate_length3F = find_flux(drift_F[:, months[0]:months[-1]+1], xptsF, yptsF, lon3, lon3, lat2, lat3, numpts, uv=0, conc_int=1)

		ann_area_flux_F2014.dump(gatepath+'/area_flux_F_'+region+'3_'+month_str+conc_team_s+'2014.txt')
		############################################################
		############################################################
		#gate_lengths = [gate_length1F, gate_length2F, gate_length3F]
		#savetxt('/Users/aapetty/NOAA/ICE_DRIFT/FLUX_GATES/gats_lengthsF.txt', gate_lengths)

	if (CALCFNOCONC2014==1):
		############################################################
		###### GET FOWLER DATA FOR YEARS AND MONTHS OF INTEREST - NO CONCENTRATION INTERPOLATION############

		#ALEK CALCULATED MONTHLY MEANS
		drift_F= load(datapath+'/FOWLER/1980-2014-drift_data_months_uv.txt')
		#drift_F= load('/Users/aapetty/NOAA/FOWLER/1980-2012-S-A-drift_data_months_means_uv.txt')
		#CHANGE E +VE TO W +VE 
		drift_F = -drift_F
		#FOWLER CALCULATED MONTHLY MEANS

		#ice_conc_season = ice_conc_ann[:, months[0]:months[-1]+1]

		ann_area_flux_F2014NOCONC[0], gate_length1F = find_flux(drift_F[:, months[0]:months[-1]+1], xptsF, yptsF, lon3, lon1, lat3, lat3, numpts, uv=1, conc_int=0)
		ann_area_flux_F2014NOCONC[1], gate_length2F = find_flux(drift_F[:, months[0]:months[-1]+1], xptsF, yptsF, lon3, lon3, lat1, lat2, numpts, uv=0, conc_int=0)
		ann_area_flux_F2014NOCONC[2], gate_length3F = find_flux(drift_F[:, months[0]:months[-1]+1], xptsF, yptsF, lon3, lon3, lat2, lat3, numpts, uv=0, conc_int=0)

		ann_area_flux_F2014NOCONC.dump(gatepath+'/area_flux_F_'+region+'3_'+month_str+conc_team_s+'noconc2014.txt')
		############################################################
		############################################################
		#gate_lengths = [gate_length1F, gate_length2F, gate_length3F]
		#savetxt('/Users/aapetty/NOAA/ICE_DRIFT/FLUX_GATES/gats_lengthsF.txt', gate_lengths)

	print season
	if (season==0) or (season==3):

		print season
		############################################################
		if CALCAS6==1:
			ice_conc_season = ice_conc_ann[28:34, months[0]:months[-1]+1]

			###### GET CERSAT DATA FOR YEARS AND MONTHS OF INTEREST ############
			drift_uv6day = ma.masked_all((6, 12, 2, 66, 76))
			drift_uv6day_temp= load(datapath+'/ASCAT_6DAY/2008-2013-JFM-OND-drift_data_months_uv.txt')
			#CHANGE E +VE TO W +VE
			drift_uv6day_temp = -drift_uv6day_temp
			drift_uv6day[:, 0:3]=drift_uv6day_temp[:, 0:3]
			drift_uv6day[:, 9:12]=drift_uv6day_temp[:, 3:7]

			
			ann_area_flux_AS6[0], gate_length1C = find_flux(drift_uv6day[:, months[0]:months[-1]+1], xptsC, yptsC, lon3, lon1, lat3, lat3, numpts, uv=1, conc_int=1)
			ann_area_flux_AS6[1], gate_length2C = find_flux(drift_uv6day[:, months[0]:months[-1]+1], xptsC, yptsC, lon3, lon3, lat1, lat2, numpts, uv=0, conc_int=1)
			ann_area_flux_AS6[2], gate_length3C = find_flux(drift_uv6day[:, months[0]:months[-1]+1], xptsC, yptsC, lon3, lon3, lat2, lat3, numpts, uv=0, conc_int=1)
			############################################################
			############################################################
			ann_area_flux_AS6.dump(gatepath+'/area_flux_AS6_'+region+'3_'+month_str+conc_team_s+'.txt')


			gate_lengths = [gate_length1C, gate_length2C, gate_length3C]
			savetxt(gatepath+'/gats_lengthsC.txt', gate_lengths)

		if CALCAS3==1:
			############################################################
			###### GET CERSAT DATA FOR YEARS AND MONTHS OF INTEREST ############
			ice_conc_season = ice_conc_ann[28:34, months[0]:months[-1]+1]

			drift_uv3day = ma.masked_all((6, 12, 2, 66, 76))
			drift_uv3day_temp= load(datapath+'/ASCAT_3DAYMFILL/2008-2013-JFM-OND-drift_data_months_uv.txt')
			#CHANGE E +VE TO W +VE
			drift_uv3day_temp= -drift_uv3day_temp
			drift_uv3day[:, 0:3]=drift_uv3day_temp[:, 0:3]
			drift_uv3day[:, 9:12]=drift_uv3day_temp[:, 3:7]

			
			ann_area_flux_AS3[0], distances = find_flux(drift_uv3day[:, months[0]:months[-1]+1], xptsC, yptsC, lon3, lon1, lat3, lat3, numpts, uv=1, conc_int=1)
			ann_area_flux_AS3[1], distances = find_flux(drift_uv3day[:, months[0]:months[-1]+1], xptsC, yptsC, lon3, lon3, lat1, lat2, numpts, uv=0, conc_int=1)
			ann_area_flux_AS3[2], distances = find_flux(drift_uv3day[:, months[0]:months[-1]+1], xptsC, yptsC, lon3, lon3, lat2, lat3, numpts, uv=0, conc_int=1)
			############################################################
			############################################################
			ann_area_flux_AS3.dump(gatepath+'/area_flux_AS3_'+region+'3_'+month_str+conc_team_s+'.txt')
		

		if CALCQS==1:
			mfill=0
			if (mfill==1):
				mfillstr='MFILL'
			else:
				mfillstr=''
			############################################################
			###### GET CERSAT DATA FOR YEARS AND MONTHS OF INTEREST ############
			ice_conc_season = ice_conc_ann[12:29, months[0]:months[-1]+1]

			drift_QS = ma.masked_all((17, 12, 2, 66, 76))
			drift_QS_temp= load(datapath+'/QSCAT_3DAY'+mfillstr+'/1992-2008-JFM-OND-drift_data_months_uv.txt')
			#CHANGE E +VE TO W +VE and N +VE to S +VE
			#REMOVE LAST YEAR AS NO DECEMBER DATA
			drift_QS_temp = -drift_QS_temp
			drift_QS[:, 0:3]=drift_QS_temp[:, 0:3]
			drift_QS[:, 9:12]=drift_QS_temp[:, 3:7]

			ann_area_flux_QS[0], distances = find_flux(drift_QS[:, months[0]:months[-1]+1], xptsC, yptsC, lon3, lon1, lat3, lat3, numpts, uv=1, conc_int=1)
			ann_area_flux_QS[1], distances = find_flux(drift_QS[:, months[0]:months[-1]+1], xptsC, yptsC, lon3, lon3, lat1, lat2, numpts, uv=0, conc_int=1)
			ann_area_flux_QS[2], distances = find_flux(drift_QS[:, months[0]:months[-1]+1], xptsC, yptsC, lon3, lon3, lat2, lat3, numpts, uv=0, conc_int=1)
			############################################################
			############################################################
			ann_area_flux_QS.dump(gatepath+'/area_flux_QS_'+region+'3_'+month_str+conc_team_s+mfillstr+'.txt')


#plot_seasonal_flux_all()



















