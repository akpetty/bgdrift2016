############################################################## 
# Date: 01/01/16
# Name: calc_wind_curl_lineplot_ERA_NCEP_JRA.py
# Author: Alek Petty
# Description: Script to read in the 3 reanalysis wind fields and calculate the curl
# Input requirements: ERA/JRA/NCEP daily wind data 
#                     Also needs the functions in BG_functions
# Output: Seasonal wind curls (squared)

import BG_functions as BGF
from mpl_toolkits.basemap import Basemap, shiftgrid
import numpy as np
from pylab import *
import os


#+++CONFIGURATION++++++
m = Basemap(projection='npstere',boundinglat=66,lon_0=0, resolution='l'  )
dx_res = 100000.
arr_res = int(ceil(100000/dx_res))
nx = int((m.xmax-m.xmin)/dx_res)+1; ny = int((m.ymax-m.ymin)/dx_res)+1
grid_str=str(int(dx_res/1000))+'km'
lons_100km, lats_100km = m.makegrid(nx, ny)

start_year = 1980
end_year = 2013
num_years = end_year - start_year + 1
years_str= str(start_year)+'-'+str(end_year)

REANALS = ['NCEP2', 'ERA', 'JRA']
month_strings = ['J', 'F', 'M', 'A','M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

box=0
if (box==0):
	box_str='wbox'
	beau_lonlat = [-175., -125., 85., 70.]
if (box==1):
	box_str='ibox'
	beau_lonlat = [-170., -130., 82., 72.]

cent_lonlat = [-150., 10., 90., 81.]
years = [0, 0, 0]
season_months = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]

#DO FOR EACH REANALYSIS
for r in xrange(0, 3):
	print r
	#DO FOR EACH SEASON
	for s in xrange(4):
		print s
		REANAL=REANALS[r]
		months = season_months[s]

		months_str= month_strings[months[0]]+'-'+month_strings[months[-1]]
		date_str = years_str+months_str

		figpath = './Figures/WIND_FIGS/'
		rawdatapath = '../../DATA/'
		datapath = datapath+'/WINDS/'+REANAL+'/DATA'
		outpath = './Data_output/WINDS/'+REANAL+'/WINDCURL/'+grid_str+'/'

		if not os.path.exists(figpath):
			os.makedirs(figpath)
		if not os.path.exists(datapath):
			os.makedirs(datapath)
		if not os.path.exists(outpath+date_str+'/'):
			os.makedirs(outpath+date_str+'/')

		calc_curl=1
		# SET TO 0 IF ALREADY CALCULATED CURL AND JUST WANT TO CALCULATE TRENDS ETC AGAIN
		if calc_curl==1:
			if REANAL=='NCEP2':
				BGF.calc_ncep2_wind_curl_NEW(m, start_year, end_year, years, months, months_str, date_str, grid_str, box_str,dx_res, arr_res, nx, ny, figpath, datapath,outpath, plot=1)
			if REANAL=='ERA':
				BGF.calc_era_wind_curl_NEW(m, start_year, end_year, years, months, months_str, date_str, grid_str, box_str,dx_res, arr_res, nx, ny, figpath, datapath, outpath, plot=1)
			if REANAL=='JRA':
				BGF.calc_jra_wind_curl_NEW(m, start_year, end_year, years, months, months_str, date_str, grid_str, box_str, dx_res, arr_res, nx, ny, figpath, datapath,outpath, plot=1)
			if REANAL=='NCEP':
				BGF.calc_ncep_wind_curl_NEW(m, start_year, end_year, years, months, months_str, date_str, grid_str, box_str,dx_res, arr_res, nx, ny, figpath, datapath,outpath, plot=1)
		#apy.plot_wind_curl_years(m, start_year, years, datapath, date_str, fig_path, grid_str, arr_res, months_str)
		#CALC WIND CURL IN THE BG AND SAVE DATA THEN PLOT OUT LINEPLOT
		BGF.calc_wind_curl_trend_bc(REANAL, grid_str, date_str, figpath, outpath, years_str, months_str, box_str, arr_res, start_year, end_year, nx, ny,dx_res, m, beau_lonlat, cent_lonlat)



