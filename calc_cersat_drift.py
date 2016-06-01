############################################################## 
# Date: 01/01/16
# Name: Alek Petty
# Description: Script to read in the daily CERSAT drift vectors and create monthly drift file to be used in the monthly ice area flux estimates.
# Input requirements: Various CERSAT daily drift data and projection lat/lon
#					  Also needs the functions in BG_functions
# Output: Monthly drift vectors on the same grid, masked however chosen. Also rotates vectors (uv file) so the components are in lat/lon.
# Notes: Need to make sure qscat etc are set to 1 to calculate for all the different drift products

import BG_functions as BGF
from mpl_toolkits.basemap import Basemap, shiftgrid
from pylab import *
import os

m = Basemap(projection='npstere',boundinglat=58,lon_0=0, resolution='l'  )

qscat=0
qscat_mfill=0
qscat_mfill_xy=0
ascat3=0
ascat3_mfill=0
ascat6=0
ascat6_xy=1

rawdatapath='../../Data/'
datapath='./Data_output/'


if qscat==1:
	start_year = 1992
	end_year = 2008
	years = [0, 0, 0, 0, 0, 0]
	months = [0, 1, 2, 9, 10, 11]
	initpath = rawdatapath+'ICE_DRIFT/CERSAT/DRIFT_QUICKSCAT/3DAY/'
	savepath = datapath+'/QSCAT_3DAY/'
	if not os.path.exists(initpath):
		os.makedirs(initpath)
	if not os.path.exists(savepath):
		os.makedirs(savepath)
	month_str = 'JFM-OND'
	daylag = 3
	BGF.calc_cersat_drift_UV(m, initpath, savepath, start_year, end_year,  years, months, month_str, daylag)

if qscat_mfill==1:
	start_year = 1992
	end_year = 2008
	years = [0, 0, 0, 0, 0, 0]
	months = [0, 1, 2, 9, 10, 11]
	initpath = rawdatapath+'ICE_DRIFT/CERSAT/DRIFT_QUICKSCAT/3DAY_MFILL/'
	savepath = datapath+'/QSCAT_3DAYMFILL/'
	if not os.path.exists(initpath):
		os.makedirs(initpath)
	if not os.path.exists(savepath):
		os.makedirs(savepath)
	month_str = 'JFM-OND'
	daylag = 3
	BGF.calc_cersat_drift_UV(m, initpath, savepath, start_year, end_year,  years, months, month_str, daylag)

if qscat_mfill_xy==1:
	start_year = 2007
	end_year = 2008
	years = [0, 0, 0, 0, 0, 0]
	months = [0, 1, 2, 9, 10, 11]
	initpath = rawdatapath+'ICE_DRIFT/CERSAT/DRIFT_QUICKSCAT/3DAY_MFILL/'
	savepath = datapath+'/QSCAT_3DAYMFILL/'
	if not os.path.exists(initpath):
		os.makedirs(initpath)
	if not os.path.exists(savepath):
		os.makedirs(savepath)
	month_str = 'JFM-OND'
	daylag = 3
	BGF.calc_cersat_drift_XY(m, initpath, savepath, start_year, end_year,  years, months, month_str, daylag)

if ascat3==1:
	start_year = 2008
	end_year = 2013
	years = [0, 0, 0, 0, 0, 0]
	months = [0, 1, 2, 9, 10, 11]
	daylag = 3
	initpath = rawdatapath+'ICE_DRIFT/CERSAT/DRIFT_ASCAT/'+str(daylag)+'DAY/'
	savepath = datapath+'/ASCAT_3DAY/'
	if not os.path.exists(initpath):
		os.makedirs(initpath)
	if not os.path.exists(savepath):
		os.makedirs(savepath)
	month_str = 'JFM-OND'
	BGF.calc_cersat_drift_UV(m, initpath, savepath, start_year, end_year,  years, months, month_str, daylag)

if ascat3_mfill==1:
	start_year = 2008
	end_year = 2013
	years = [0, 0, 0, 0, 0, 0]
	months = [0, 1, 2, 9, 10, 11]
	daylag = 3
	initpath = rawdatapath+'ICE_DRIFT/CERSAT/DRIFT_ASCAT/'+str(daylag)+'DAY_MFILL/'
	savepath = datapath+'/ASCAT_3DAYMFILL/'
	if not os.path.exists(initpath):
		os.makedirs(initpath)
	if not os.path.exists(savepath):
		os.makedirs(savepath)
	month_str = 'JFM-OND'
	BGF.calc_cersat_drift_UV(m, initpath, savepath, start_year, end_year,  years, months, month_str, daylag)

if ascat6==1:
	start_year = 2008
	end_year = 2013
	years = [0, 0, 0, 0, 0, 0]
	months = [0, 1, 2, 9, 10, 11]
	daylag = 6
	initpath = rawdatapath+'ICE_DRIFT/CERSAT/DRIFT_ASCAT/'+str(daylag)+'DAY/'
	savepath = datapath+'/ASCAT_6DAY/'
	if not os.path.exists(initpath):
		os.makedirs(initpath)
	if not os.path.exists(savepath):
		os.makedirs(savepath)
	month_str = 'JFM-OND'
	BGF.calc_cersat_drift_UV(m, initpath, savepath, start_year, end_year,  years, months, month_str, daylag)

if ascat6_xy==1:
	start_year = 2008
	end_year = 2013
	years = [0, 0, 0, 0, 0, 0]
	months = [0, 1, 2, 9, 10, 11]
	daylag = 6
	initpath = rawdatapath+'ICE_DRIFT/CERSAT/DRIFT_ASCAT/'+str(daylag)+'DAY/'
	savepath = datapath+'/ASCAT_6DAY/'
	if not os.path.exists(initpath):
		os.makedirs(initpath)
	if not os.path.exists(savepath):
		os.makedirs(savepath)
	month_str = 'JFM-OND'
	BGF.calc_cersat_drift_XY(m, initpath, savepath, start_year, end_year,  years, months, month_str, daylag)




