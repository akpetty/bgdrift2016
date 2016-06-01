############################################################## 
# Date: 01/01/16
# Name: calc_ice_conc_monthly_NT_BT.py
# Author: Alek Petty
# Description: Script to calculate ice concentration (NASA_TEAM or BOOTSTRAP) as monthly averages from the daily data
# Input requirements: Ice concentration data
# Output: Monthly ice concentration on the same EASE? grid
# Info: Need to run for both algorithms

import numpy as np
from pylab import *
import numpy.ma as ma
from mpl_toolkits.basemap import Basemap, shiftgrid
from glob import glob

datapath = '../../DATA/'
outpath='./Data_output/'

m = Basemap(projection='npstere',boundinglat=66,lon_0=0, resolution='l'  )

alg=0
if (alg==0):
	team = 'NASA_TEAM'
	team_s = 'nt'
	header = 300
	datatype='uint8'
	scale_factor=250.
if (alg==1):
	team = 'BOOTSTRAP'
	team_s = 'bt'
	header = 0
	datatype='<i2'
	scale_factor=1000.

start_year=1980
end_year=2013
num_years = end_year-start_year+1
years_str=str(start_year)+'-'+str(end_year)

ice_conc = ma.masked_all((num_years, 12, 448, 304))

for year in xrange(num_years):
	for month in xrange(12):
		month_str = '%02d' % (month+1)
		year_str=str(1980+year)
		files = glob(datapath+'/ICE_CONC/'+team+'/ARCTIC/monthly/'+team_s+'_'+year_str+month_str+'*.bin')
		if (size(files)>1):
			print year_str+month_str
		fd = open(files[-1], 'r')
		data = fromfile(file=fd, dtype=datatype)
		data = data[header:]
		#FIRST 300 FILES ARE HEADER INFO
		ice_conc[year, month] = reshape(data, [448, 304])
#divide by 250 to express in concentration
ice_conc = ice_conc/scale_factor
#GREATER THAN 250 is mask/land etc
ice_conc = ma.masked_where(ice_conc>1., ice_conc)
#ice_conc = ma.masked_where(ice_conc<0.15, ice_conc)

flat = open(datapath+'/ICE_CONC/psn25lats_v3.dat', 'rb')
flon = open(datapath+'/ICE_CONC/psn25lons_v3.dat', 'rb')
lats = reshape(fromfile(file=flat, dtype='<i4')/100000., [448, 304])
lons = reshape(fromfile(file=flon, dtype='<i4')/100000., [448, 304])

ice_conc.dump(outpath+'/ice_conc_months-'+years_str+team+'.txt')
lats.dump(outpath+'/ice_conc_lats'+team+'.txt')
lons.dump(outpath+'/ice_conc_lons'+team+'.txt')

#CAN PROJECT ONTO BASEMAP GRID AND OUTPUT THESE COORDS IF YOU WANT
#xpts, ypts = m(lons, lats)





